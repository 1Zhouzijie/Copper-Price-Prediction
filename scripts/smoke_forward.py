"""Smoke test the PyTorch model forward pass with random tensors."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        print("torch is not installed. Install requirements.txt before running the model forward smoke test.")
        return

    from copper_prediction.model import CopperIntervalPredictor, CopperIntervalPredictorConfig

    batch_size = 4
    batch = {
        "cnn_x": torch.randn(batch_size, 10, 20),
        "market_x": torch.randn(batch_size, 20, 8),
        "market_adj": torch.eye(20).repeat(batch_size, 1, 1),
        "demand_x": torch.randn(batch_size, 18, 8),
        "demand_adj": torch.eye(18).repeat(batch_size, 1, 1),
        "supply_x": torch.randn(batch_size, 20, 8),
        "supply_adj": torch.eye(20).repeat(batch_size, 1, 1),
    }

    configs = [
        ("v1_gcn_concat", CopperIntervalPredictorConfig(graph_encoder="gcn", fusion_mode="concat")),
        ("v2_gcn_branch_attention", CopperIntervalPredictorConfig(graph_encoder="gcn", fusion_mode="branch_attention")),
        ("v3_gat_branch_attention", CopperIntervalPredictorConfig(graph_encoder="gat", fusion_mode="branch_attention")),
    ]

    for name, config in configs:
        model = CopperIntervalPredictor(config)
        prediction, aux = model(**batch, return_aux=True)
        print(f"model={name}")
        print("prediction", tuple(prediction.shape))
        print("v1_cnn", tuple(aux["v1_cnn"].shape))
        print("v2_market", tuple(aux["v2_market"].shape))
        print("v3_demand", tuple(aux["v3_demand"].shape))
        print("v4_supply", tuple(aux["v4_supply"].shape))
        if "branch_attention" in aux:
            print("branch_attention", tuple(aux["branch_attention"].shape))
        if "market_gat_attention_1" in aux:
            print("market_gat_attention_1", tuple(aux["market_gat_attention_1"].shape))
            print("market_gat_attention_2", tuple(aux["market_gat_attention_2"].shape))


if __name__ == "__main__":
    main()
