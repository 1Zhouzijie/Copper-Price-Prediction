"""Feature construction for the copper interval prediction model."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .config import (
    CNN_FEATURE_DIM,
    CNN_WINDOW,
    CORR_WINDOW,
    NODE_FEATURE_DIM,
    TOP_K_EDGES,
    GraphSpec,
    NodeSpec,
)


def _column(frame: pd.DataFrame, *candidates: str) -> pd.Series | None:
    lower_to_original = {str(col).lower(): col for col in frame.columns}
    for name in candidates:
        original = lower_to_original.get(name.lower())
        if original is not None:
            return pd.to_numeric(frame[original], errors="coerce")
    return None


def _required_column(frame: pd.DataFrame, *candidates: str) -> pd.Series:
    series = _column(frame, *candidates)
    if series is None:
        raise KeyError(f"missing required column, expected one of {candidates}")
    return series


def _zero_like(index: pd.Index) -> pd.Series:
    return pd.Series(0.0, index=index)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan)


def _safe_log_ratio(current: pd.Series | None, previous: pd.Series | None, index: pd.Index) -> pd.Series:
    if current is None or previous is None:
        return _zero_like(index)
    current = current.replace(0, np.nan)
    previous = previous.replace(0, np.nan)
    return np.log(_safe_divide(current, previous)).replace([np.inf, -np.inf], np.nan)


def _last_window(values: pd.DataFrame, end_date: pd.Timestamp, window: int) -> pd.DataFrame:
    values = values.sort_index().loc[:end_date]
    if len(values) < window:
        raise ValueError(f"need at least {window} rows up to {end_date}, got {len(values)}")
    return values.tail(window)


def _value_series(frame: pd.DataFrame, node_type: str) -> pd.Series:
    if node_type == "inventory":
        return _required_column(frame, "inventory", "stock", "value", "close")
    return _required_column(frame, "close", "settlement", "value")


def financial_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the 8-dimensional node features for financial nodes."""

    frame = frame.sort_index()
    close = _required_column(frame, "close", "settlement", "value")
    high = _column(frame, "high")
    low = _column(frame, "low")
    volume = _column(frame, "volume", "vol")

    returns = close.pct_change()
    ma20 = close.rolling(20).mean()

    features = pd.DataFrame(index=frame.index)
    features["r_1d"] = returns
    features["r_5d"] = close.pct_change(5)
    features["r_20d"] = close.pct_change(20)
    features["vol_5d"] = returns.rolling(5).std()
    features["vol_20d"] = returns.rolling(20).std()
    if high is not None and low is not None:
        features["range"] = _safe_divide(high - low, close.shift(1))
    else:
        features["range"] = 0.0
    features["volume_change"] = _safe_log_ratio(volume, volume.shift(1) if volume is not None else None, frame.index)
    features["ma20_gap"] = _safe_divide(close - ma20, ma20)
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def inventory_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the 8-dimensional node features for inventory nodes."""

    frame = frame.sort_index()
    inv = _required_column(frame, "inventory", "stock", "value", "close")
    change = inv.pct_change()
    ma20 = inv.rolling(20).mean()

    def rank_last(values: pd.Series) -> float:
        current = values.iloc[-1]
        return float((values <= current).mean())

    features = pd.DataFrame(index=frame.index)
    features["chg_1d"] = change
    features["chg_5d"] = inv.pct_change(5)
    features["chg_20d"] = inv.pct_change(20)
    features["vol_5d"] = change.rolling(5).std()
    features["vol_20d"] = change.rolling(20).std()
    features["rank_252d"] = inv.rolling(252, min_periods=20).apply(rank_last, raw=False)
    features["ma20_gap"] = _safe_divide(inv - ma20, ma20)
    features["destock_signal"] = np.sign(inv.pct_change(5))
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def copper_cnn_features(copper_frame: pd.DataFrame) -> pd.DataFrame:
    """Build the 10 copper features used by the one-dimensional CNN branch."""

    frame = copper_frame.sort_index()
    open_ = _required_column(frame, "open")
    high = _required_column(frame, "high")
    low = _required_column(frame, "low")
    close = _required_column(frame, "close", "settlement")
    volume = _column(frame, "volume", "vol")
    open_interest = _column(frame, "open_interest", "openinterest", "oi")
    prev_close = close.shift(1)

    features = pd.DataFrame(index=frame.index)
    features["open_ret"] = _safe_divide(open_ - prev_close, prev_close)
    features["high_ret"] = _safe_divide(high - prev_close, prev_close)
    features["low_ret"] = _safe_divide(low - prev_close, prev_close)
    features["close_ret"] = _safe_divide(close - prev_close, prev_close)
    features["intraday_ret"] = _safe_divide(close - open_, open_)
    features["range"] = _safe_divide(high - low, prev_close)
    features["upper_shadow"] = _safe_divide(high - np.maximum(open_, close), prev_close)
    features["lower_shadow"] = _safe_divide(np.minimum(open_, close) - low, prev_close)
    features["volume_change"] = _safe_log_ratio(volume, volume.shift(1) if volume is not None else None, frame.index)
    features["oi_change"] = _safe_log_ratio(
        open_interest,
        open_interest.shift(1) if open_interest is not None else None,
        frame.index,
    )
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_cnn_matrix(copper_frame: pd.DataFrame, end_date: pd.Timestamp, window: int = CNN_WINDOW) -> np.ndarray:
    """Return one CNN matrix with shape (10, window)."""

    features = copper_cnn_features(copper_frame)
    matrix = _last_window(features, end_date, window).to_numpy(dtype=np.float32).T
    if matrix.shape != (CNN_FEATURE_DIM, window):
        raise ValueError(f"expected CNN matrix {(CNN_FEATURE_DIM, window)}, got {matrix.shape}")
    return matrix


def _node_feature_frame(frame: pd.DataFrame, node: NodeSpec) -> pd.DataFrame:
    if node.node_type == "inventory":
        return inventory_features(frame)
    return financial_features(frame)


def _node_return_series(frame: pd.DataFrame, node: NodeSpec) -> pd.Series:
    values = _value_series(frame.sort_index(), node.node_type)
    return values.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def topk_spearman_adjacency(returns: pd.DataFrame, top_k: int = TOP_K_EDGES) -> np.ndarray:
    """Build a symmetric top-k absolute Spearman adjacency matrix."""

    corr = returns.corr(method="spearman").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    np.fill_diagonal(corr.values, 0.0)
    weights = corr.abs().to_numpy(dtype=np.float32)
    n_nodes = weights.shape[0]
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    for i in range(n_nodes):
        if n_nodes <= 1:
            continue
        k = min(top_k, n_nodes - 1)
        top_indices = np.argpartition(-weights[i], kth=k - 1)[:k]
        adjacency[i, top_indices] = weights[i, top_indices]

    adjacency = (adjacency + adjacency.T) / 2.0
    adjacency += np.eye(n_nodes, dtype=np.float32)
    return adjacency


def build_graph_inputs(
    asset_frames: Mapping[str, pd.DataFrame],
    graph_spec: GraphSpec,
    end_date: pd.Timestamp,
    corr_window: int = CORR_WINDOW,
    top_k: int = TOP_K_EDGES,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (adjacency, node_features) for a graph at one prediction date.

    `asset_frames` is keyed by the NodeSpec.name values in the graph spec.
    Each frame should be indexed by date. Financial frames need at least a
    close-like column; inventory frames need an inventory/stock/value column.
    """

    node_features: list[np.ndarray] = []
    returns_by_node: dict[str, pd.Series] = {}

    for node in graph_spec.nodes:
        if node.name not in asset_frames:
            raise KeyError(f"missing data for node {node.name!r} in {graph_spec.name} graph")
        frame = asset_frames[node.name].sort_index()
        feature_frame = _node_feature_frame(frame, node)
        if end_date not in feature_frame.index:
            feature_frame = feature_frame.reindex(feature_frame.index.union([end_date])).sort_index().ffill()
        row = feature_frame.loc[:end_date].tail(1)
        if row.empty:
            raise ValueError(f"no features available for node {node.name!r} up to {end_date}")
        node_features.append(row.to_numpy(dtype=np.float32).reshape(-1))
        returns_by_node[node.name] = _node_return_series(frame, node)

    x = np.vstack(node_features).astype(np.float32)
    if x.shape != (graph_spec.num_nodes, NODE_FEATURE_DIM):
        raise ValueError(f"expected node features {(graph_spec.num_nodes, NODE_FEATURE_DIM)}, got {x.shape}")

    returns = pd.DataFrame(returns_by_node).sort_index().loc[:end_date].tail(corr_window)
    if len(returns) < corr_window:
        raise ValueError(f"need at least {corr_window} return rows up to {end_date}, got {len(returns)}")
    returns = returns.ffill().fillna(0.0)
    adjacency = topk_spearman_adjacency(returns, top_k=top_k)
    return adjacency.astype(np.float32), x


def build_target(copper_frame: pd.DataFrame, end_date: pd.Timestamp) -> np.ndarray:
    """Build [next-day low, next-day high] targets relative to close at end_date."""

    frame = copper_frame.sort_index()
    close = _required_column(frame, "close", "settlement")
    low = _required_column(frame, "low")
    high = _required_column(frame, "high")

    positions = pd.Series(np.arange(len(frame)), index=frame.index)
    if end_date not in positions.index:
        raise KeyError(f"{end_date} is not in copper_frame index")
    pos = int(positions.loc[end_date])
    if pos + 1 >= len(frame):
        raise ValueError(f"cannot build next-day target for last row {end_date}")

    base = float(close.iloc[pos])
    target = np.array(
        [
            (float(low.iloc[pos + 1]) - base) / base,
            (float(high.iloc[pos + 1]) - base) / base,
        ],
        dtype=np.float32,
    )
    return target
