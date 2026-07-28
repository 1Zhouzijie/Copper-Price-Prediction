"""Smoke test feature construction with synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from copper_prediction.config import default_graph_specs  # noqa: E402
from copper_prediction.dataset import CopperGraphSampleBuilder  # noqa: E402


def financial_frame(index: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, size=len(index))))
    open_ = close * (1.0 + rng.normal(0.0, 0.003, size=len(index)))
    high = np.maximum(open_, close) * (1.0 + rng.random(len(index)) * 0.01)
    low = np.minimum(open_, close) * (1.0 - rng.random(len(index)) * 0.01)
    volume = rng.lognormal(mean=12.0, sigma=0.25, size=len(index))
    open_interest = rng.lognormal(mean=11.0, sigma=0.20, size=len(index))
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "open_interest": open_interest,
        },
        index=index,
    )


def inventory_frame(index: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    inventory = 200_000.0 + np.cumsum(rng.normal(0.0, 1_500.0, size=len(index)))
    inventory = np.clip(inventory, 20_000.0, None)
    return pd.DataFrame({"inventory": inventory}, index=index)


def main() -> None:
    dates = pd.bdate_range("2020-01-01", periods=320)
    specs = default_graph_specs()
    unique_nodes = {node.name: node for spec in specs.values() for node in spec.nodes}

    frames = {}
    for idx, node in enumerate(unique_nodes.values(), start=1):
        if node.node_type == "inventory":
            frames[node.name] = inventory_frame(dates, seed=idx)
        else:
            frames[node.name] = financial_frame(dates, seed=idx)

    copper_frame = frames["LME Copper"]
    builder = CopperGraphSampleBuilder(copper_frame, frames, specs)
    sample = builder.build(dates[260])

    print("date", sample.date.date())
    print("cnn_x", sample.cnn_x.shape)
    print("market", sample.market_adj.shape, sample.market_x.shape)
    print("demand", sample.demand_adj.shape, sample.demand_x.shape)
    print("supply", sample.supply_adj.shape, sample.supply_x.shape)
    print("target", sample.target.shape)


if __name__ == "__main__":
    main()
