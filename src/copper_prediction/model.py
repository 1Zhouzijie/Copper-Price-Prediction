"""PyTorch implementation of the CNN + three-GAE copper interval model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import CNN_FEATURE_DIM, CNN_WINDOW, NODE_FEATURE_DIM, default_graph_specs


@dataclass(frozen=True)
class CopperIntervalPredictorConfig:
    """Model hyperparameters for the copper interval predictor."""

    cnn_in_channels: int = CNN_FEATURE_DIM
    cnn_window: int = CNN_WINDOW
    cnn_channels: tuple[int, int] = (32, 64)
    cnn_kernel_size: int = 5
    cnn_output_dim: int = 64
    node_feature_dim: int = NODE_FEATURE_DIM
    gae_hidden_dim: int = 32
    gae_embedding_dim: int = 16
    graph_encoder: str = "gcn"
    gat_heads: int = 4
    gat_attention_dropout: float = 0.1
    gat_negative_slope: float = 0.2
    gat_use_edge_weights: bool = True
    fusion_mode: str = "concat"
    branch_attention_dim: int = 32
    mlp_hidden_dims: tuple[int, int] = (128, 64)
    dropout: float = 0.1


def normalize_adjacency(adjacency: Tensor, eps: float = 1e-8) -> Tensor:
    """Symmetrically normalize a batched adjacency matrix."""

    degree = adjacency.sum(dim=-1).clamp_min(eps)
    degree_inv_sqrt = degree.pow(-0.5)
    return degree_inv_sqrt.unsqueeze(-1) * adjacency * degree_inv_sqrt.unsqueeze(-2)


class CopperCNN1D(nn.Module):
    """One-dimensional CNN over the time axis of the copper feature matrix."""

    def __init__(self, config: CopperIntervalPredictorConfig) -> None:
        super().__init__()
        padding = config.cnn_kernel_size // 2
        c1, c2 = config.cnn_channels
        self.encoder = nn.Sequential(
            nn.Conv1d(config.cnn_in_channels, c1, kernel_size=config.cnn_kernel_size, padding=padding),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(c1, c2, kernel_size=config.cnn_kernel_size, padding=padding),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2, config.cnn_output_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode copper matrix.

        Args:
            x: Tensor with shape (batch, 10, 20).
        """

        if x.ndim != 3:
            raise ValueError(f"cnn input must be 3D (batch, features, window), got {tuple(x.shape)}")
        return self.projection(self.encoder(x))


class GraphConvolution(nn.Module):
    """Simple dense GCN layer for small fixed graphs."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        adjacency_norm = normalize_adjacency(adjacency)
        return self.linear(torch.bmm(adjacency_norm, x))


class GraphAttentionLayer(nn.Module):
    """Dense multi-head GAT layer for small weighted graphs.GAT图注意力层"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 4,
        concat: bool = True,
        attention_dropout: float = 0.1,
        negative_slope: float = 0.2,
        use_edge_weights: bool = True,
    ) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError("GAT heads must be at least 1")
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.use_edge_weights = use_edge_weights

        self.linear = nn.Linear(in_features, heads * out_features, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, out_features))
        self.attn_dst = nn.Parameter(torch.empty(heads, out_features))
        bias_dim = heads * out_features if concat else out_features
        self.bias = nn.Parameter(torch.zeros(bias_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, adjacency: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 3:
            raise ValueError(f"graph features must be 3D (batch, nodes, features), got {tuple(x.shape)}")
        if adjacency.ndim != 3:
            raise ValueError(f"adjacency must be 3D (batch, nodes, nodes), got {tuple(adjacency.shape)}")
        if x.shape[1] != adjacency.shape[1] or adjacency.shape[1] != adjacency.shape[2]:
            raise ValueError(
                "graph features and adjacency must agree on node count, "
                f"got x={tuple(x.shape)} adjacency={tuple(adjacency.shape)}"
            )

        batch_size, num_nodes, _ = x.shape
        h = self.linear(x).view(batch_size, num_nodes, self.heads, self.out_features)

        src_scores = torch.sum(h * self.attn_src.view(1, 1, self.heads, self.out_features), dim=-1)
        dst_scores = torch.sum(h * self.attn_dst.view(1, 1, self.heads, self.out_features), dim=-1)
        scores = src_scores.permute(0, 2, 1).unsqueeze(-1) + dst_scores.permute(0, 2, 1).unsqueeze(-2)
        scores = self.leaky_relu(scores)

        edge_mask = adjacency > 0
        if self.use_edge_weights:
            edge_bias = torch.log(adjacency.clamp_min(1e-8)).unsqueeze(1)
            scores = scores + edge_bias
        scores = scores.masked_fill(~edge_mask.unsqueeze(1), torch.finfo(scores.dtype).min)

        attention = torch.softmax(scores, dim=-1)
        attention_for_output = self.attention_dropout(attention)

        h_by_head = h.permute(0, 2, 1, 3)
        out = torch.matmul(attention_for_output, h_by_head)
        if self.concat:
            out = out.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, self.heads * self.out_features)
        else:
            out = out.mean(dim=1)
        return out + self.bias, attention


class GAEBranch(nn.Module):
    """Graph auto-encoder branch for one graph."""

    def __init__(
        self,
        num_nodes: int,
        copper_index: int,
        config: CopperIntervalPredictorConfig,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.copper_index = copper_index
        self.embedding_dim = config.gae_embedding_dim
        self.conv1 = GraphConvolution(config.node_feature_dim, config.gae_hidden_dim)
        self.conv2 = GraphConvolution(config.gae_hidden_dim, config.gae_embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

    def encode(self, x: Tensor, adjacency: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"graph features must be 3D (batch, nodes, features), got {tuple(x.shape)}")
        if adjacency.ndim != 3:
            raise ValueError(f"adjacency must be 3D (batch, nodes, nodes), got {tuple(adjacency.shape)}")
        if x.shape[1] != self.num_nodes:
            raise ValueError(f"expected {self.num_nodes} nodes, got {x.shape[1]}")
        h = F.relu(self.conv1(x, adjacency))
        h = self.dropout(h)
        return self.conv2(h, adjacency)

    @staticmethod
    def decode(z: Tensor) -> Tensor:
        return torch.sigmoid(torch.bmm(z, z.transpose(1, 2)))

    def graph_vector(self, z: Tensor) -> Tensor:
        copper_embedding = z[:, self.copper_index, :]
        graph_mean = z.mean(dim=1)
        return torch.cat([copper_embedding, graph_mean], dim=-1)

    def forward(self, x: Tensor, adjacency: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z = self.encode(x, adjacency)
        reconstruction = self.decode(z)
        vector = self.graph_vector(z)
        return vector, z, reconstruction


class GATGAEBranch(nn.Module):
    """Graph auto-encoder branch with a GAT encoder."""

    def __init__(
        self,
        num_nodes: int,
        copper_index: int,
        config: CopperIntervalPredictorConfig,
    ) -> None:
        super().__init__()
        if config.gae_hidden_dim % config.gat_heads != 0:
            raise ValueError("gae_hidden_dim must be divisible by gat_heads for the first GAT layer")
        self.num_nodes = num_nodes
        self.copper_index = copper_index
        self.embedding_dim = config.gae_embedding_dim
        hidden_per_head = config.gae_hidden_dim // config.gat_heads
        self.gat1 = GraphAttentionLayer(
            in_features=config.node_feature_dim,
            out_features=hidden_per_head,
            heads=config.gat_heads,
            concat=True,
            attention_dropout=config.gat_attention_dropout,
            negative_slope=config.gat_negative_slope,
            use_edge_weights=config.gat_use_edge_weights,
        )
        self.gat2 = GraphAttentionLayer(
            in_features=config.gae_hidden_dim,
            out_features=config.gae_embedding_dim,
            heads=config.gat_heads,
            concat=False,
            attention_dropout=config.gat_attention_dropout,
            negative_slope=config.gat_negative_slope,
            use_edge_weights=config.gat_use_edge_weights,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.last_attention_1: Tensor | None = None
        self.last_attention_2: Tensor | None = None

    def encode(self, x: Tensor, adjacency: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"graph features must be 3D (batch, nodes, features), got {tuple(x.shape)}")
        if adjacency.ndim != 3:
            raise ValueError(f"adjacency must be 3D (batch, nodes, nodes), got {tuple(adjacency.shape)}")
        if x.shape[1] != self.num_nodes:
            raise ValueError(f"expected {self.num_nodes} nodes, got {x.shape[1]}")

        h, attention_1 = self.gat1(x, adjacency)
        h = self.dropout(F.elu(h))
        z, attention_2 = self.gat2(h, adjacency)
        self.last_attention_1 = attention_1
        self.last_attention_2 = attention_2
        return z

    @staticmethod
    def decode(z: Tensor) -> Tensor:
        return torch.sigmoid(torch.bmm(z, z.transpose(1, 2)))

    def graph_vector(self, z: Tensor) -> Tensor:
        copper_embedding = z[:, self.copper_index, :]
        graph_mean = z.mean(dim=1)
        return torch.cat([copper_embedding, graph_mean], dim=-1)

    def forward(self, x: Tensor, adjacency: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z = self.encode(x, adjacency)
        reconstruction = self.decode(z)
        vector = self.graph_vector(z)
        return vector, z, reconstruction

    def attention_maps(self) -> dict[str, Tensor]:
        if self.last_attention_1 is None or self.last_attention_2 is None:
            return {}
        return {
            "attention_1": self.last_attention_1,
            "attention_2": self.last_attention_2,
        }


class BranchAttentionFusion(nn.Module):
    """Fuse market, demand, and supply graph vectors with CNN-guided attention."""

    def __init__(
        self,
        cnn_dim: int,
        graph_dim: int,
        attention_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query = nn.Linear(cnn_dim, attention_dim)
        self.key = nn.Linear(graph_dim, attention_dim)
        self.value = nn.Linear(graph_dim, graph_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(attention_dim)

    def forward(
        self,
        v1: Tensor,
        v_market: Tensor,
        v_demand: Tensor,
        v_supply: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        graph_vectors = torch.stack([v_market, v_demand, v_supply], dim=1)

        query = self.query(v1).unsqueeze(1)
        keys = self.key(graph_vectors)
        values = self.dropout(self.value(graph_vectors))

        scores = torch.sum(query * keys, dim=-1) / self.scale
        attention = torch.softmax(scores, dim=1)

        v_graph = torch.sum(attention.unsqueeze(-1) * values, dim=1)
        fused = torch.cat([v1, v_graph], dim=-1)
        return fused, attention, v_graph


class CopperIntervalPredictor(nn.Module):
    """CNN + market/demand/supply GAE fusion model."""

    def __init__(self, config: CopperIntervalPredictorConfig | None = None) -> None:
        super().__init__()
        self.config = config or CopperIntervalPredictorConfig()
        if self.config.graph_encoder not in {"gcn", "gat"}:
            raise ValueError("graph_encoder must be 'gcn' or 'gat'")
        if self.config.fusion_mode not in {"concat", "branch_attention"}:
            raise ValueError("fusion_mode must be 'concat' or 'branch_attention'")
        graph_specs = default_graph_specs()

        self.cnn = CopperCNN1D(self.config)
        graph_branch = GAEBranch if self.config.graph_encoder == "gcn" else GATGAEBranch
        self.market_gae = graph_branch(
            graph_specs["market"].num_nodes,
            graph_specs["market"].copper_index,
            self.config,
        )
        self.demand_gae = graph_branch(
            graph_specs["demand"].num_nodes,
            graph_specs["demand"].copper_index,
            self.config,
        )
        self.supply_gae = graph_branch(
            graph_specs["supply"].num_nodes,
            graph_specs["supply"].copper_index,
            self.config,
        )

        gae_vector_dim = self.config.gae_embedding_dim * 2
        self.branch_attention: BranchAttentionFusion | None = None
        if self.config.fusion_mode == "branch_attention":
            self.branch_attention = BranchAttentionFusion(
                cnn_dim=self.config.cnn_output_dim,
                graph_dim=gae_vector_dim,
                attention_dim=self.config.branch_attention_dim,
                dropout=self.config.dropout,
            )
            fusion_dim = self.config.cnn_output_dim + gae_vector_dim
        else:
            fusion_dim = self.config.cnn_output_dim + gae_vector_dim * 3

        mlp_layers: list[nn.Module] = []
        in_dim = fusion_dim
        for hidden_dim in self.config.mlp_hidden_dims:
            mlp_layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.config.dropout),
                ]
            )
            in_dim = hidden_dim
        mlp_layers.append(nn.Linear(in_dim, 2))
        self.head = nn.Sequential(*mlp_layers)

    def forward(
        self,
        cnn_x: Tensor,
        market_x: Tensor,
        market_adj: Tensor,
        demand_x: Tensor,
        demand_adj: Tensor,
        supply_x: Tensor,
        supply_adj: Tensor,
        return_aux: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        v1 = self.cnn(cnn_x)
        v2, z_market, a_market_hat = self.market_gae(market_x, market_adj)
        v3, z_demand, a_demand_hat = self.demand_gae(demand_x, demand_adj)
        v4, z_supply, a_supply_hat = self.supply_gae(supply_x, supply_adj)

        aux_extra: dict[str, Tensor] = {}
        if self.config.fusion_mode == "branch_attention":
            if self.branch_attention is None:
                raise RuntimeError("branch attention module is not initialized")
            fused, branch_attention, v_graph = self.branch_attention(v1, v2, v3, v4)
            aux_extra["branch_attention"] = branch_attention
            aux_extra["v_graph_attention"] = v_graph
        else:
            fused = torch.cat([v1, v2, v3, v4], dim=-1)

        prediction = self.head(fused)

        if not return_aux:
            return prediction
        aux = {
            "v1_cnn": v1,
            "v2_market": v2,
            "v3_demand": v3,
            "v4_supply": v4,
            "z_market": z_market,
            "z_demand": z_demand,
            "z_supply": z_supply,
            "a_market_hat": a_market_hat,
            "a_demand_hat": a_demand_hat,
            "a_supply_hat": a_supply_hat,
        }
        if self.config.graph_encoder == "gat":
            for prefix, branch in (
                ("market", self.market_gae),
                ("demand", self.demand_gae),
                ("supply", self.supply_gae),
            ):
                if isinstance(branch, GATGAEBranch):
                    for name, attention in branch.attention_maps().items():
                        aux[f"{prefix}_gat_{name}"] = attention.detach()
        aux.update(aux_extra)
        return prediction, aux


def interval_prediction_loss_components(
    prediction: Tensor,
    target: Tensor,
    bound_penalty_weight: float = 1.0,
) -> dict[str, Tensor]:
    """Return the supervised interval-loss components separately."""

    mse = F.mse_loss(prediction, target)
    low = prediction[:, 0]
    high = prediction[:, 1]
    bound_penalty = F.relu(low - high).pow(2).mean()
    supervised = mse + bound_penalty_weight * bound_penalty
    return {
        "mse_loss": mse,
        "bound_penalty_loss": bound_penalty,
        "supervised_loss": supervised,
    }


def interval_prediction_loss(
    prediction: Tensor,
    target: Tensor,
    bound_weight: float = 1.0,
) -> Tensor:
    """MSE loss plus a penalty when predicted low is above predicted high."""

    return interval_prediction_loss_components(
        prediction,
        target,
        bound_penalty_weight=bound_weight,
    )["supervised_loss"]


def gae_reconstruction_loss(reconstruction: Tensor, adjacency: Tensor) -> Tensor:
    """Reconstruction loss for one GAE branch."""

    return F.mse_loss(reconstruction, adjacency.clamp(0.0, 1.0))


def training_loss_components(
    prediction: Tensor,
    target: Tensor,
    aux: dict[str, Tensor],
    market_adj: Tensor,
    demand_adj: Tensor,
    supply_adj: Tensor,
    supervised_weight: float = 1.0,
    bound_penalty_weight: float = 1.0,
    reconstruction_weight: float = 5e-4,
) -> dict[str, Tensor]:
    """Return weighted total loss and every raw component used to build it."""

    interval = interval_prediction_loss_components(
        prediction,
        target,
        bound_penalty_weight=bound_penalty_weight,
    )
    market_reconstruction = gae_reconstruction_loss(aux["a_market_hat"], market_adj)
    demand_reconstruction = gae_reconstruction_loss(aux["a_demand_hat"], demand_adj)
    supply_reconstruction = gae_reconstruction_loss(aux["a_supply_hat"], supply_adj)
    reconstruction = (
        market_reconstruction
        + demand_reconstruction
        + supply_reconstruction
    ) / 3.0
    weighted_supervised = supervised_weight * interval["supervised_loss"]
    weighted_reconstruction = reconstruction_weight * reconstruction
    total = weighted_supervised + weighted_reconstruction
    return {
        "total_loss": total,
        **interval,
        "market_reconstruction_loss": market_reconstruction,
        "demand_reconstruction_loss": demand_reconstruction,
        "supply_reconstruction_loss": supply_reconstruction,
        "reconstruction_loss": reconstruction,
        "weighted_supervised_loss": weighted_supervised,
        "weighted_reconstruction_loss": weighted_reconstruction,
    }


def total_training_loss(
    prediction: Tensor,
    target: Tensor,
    aux: dict[str, Tensor],
    market_adj: Tensor,
    demand_adj: Tensor,
    supply_adj: Tensor,
    interval_weight: float | None = None,
    reconstruction_weight: float = 5e-4,
    supervised_weight: float = 1.0,
    bound_penalty_weight: float = 1.0,
) -> Tensor:
    """Return the combined supervised and graph-reconstruction loss.

    ``interval_weight`` is retained as a backward-compatible alias for the old
    bound-penalty weight. New code should use ``bound_penalty_weight`` so the
    role of each weight is explicit.
    """

    if interval_weight is not None:
        bound_penalty_weight = interval_weight
    return training_loss_components(
        prediction=prediction,
        target=target,
        aux=aux,
        market_adj=market_adj,
        demand_adj=demand_adj,
        supply_adj=supply_adj,
        supervised_weight=supervised_weight,
        bound_penalty_weight=bound_penalty_weight,
        reconstruction_weight=reconstruction_weight,
    )["total_loss"]
