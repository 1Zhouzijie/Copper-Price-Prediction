"""Sample builders for the CNN + three-GAE model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import GraphSpec, default_graph_specs
from .features import (
    build_cnn_matrix_from_features,
    build_graph_inputs_from_precomputed,
    build_target,
    copper_cnn_features,
    node_feature_frame,
    node_return_series,
)


@dataclass(frozen=True)
class CopperModelInputs:
    """Model inputs at prediction date t."""

    date: pd.Timestamp
    cnn_x: np.ndarray
    market_x: np.ndarray
    market_adj: np.ndarray
    demand_x: np.ndarray
    demand_adj: np.ndarray
    supply_x: np.ndarray
    supply_adj: np.ndarray


@dataclass(frozen=True)
class CopperSample(CopperModelInputs):
    """One supervised model sample at prediction date t."""

    target: np.ndarray


class CopperGraphSampleBuilder:
    """Build one sample following the documented data construction scheme."""

    def __init__(
        self,
        copper_frame: pd.DataFrame,
        asset_frames: Mapping[str, pd.DataFrame],
        graph_specs: Mapping[str, GraphSpec] | None = None,
    ) -> None:
        self.copper_frame = copper_frame.sort_index()
        self.asset_frames = asset_frames
        self.graph_specs = graph_specs or default_graph_specs()
        self.cnn_feature_frame = copper_cnn_features(self.copper_frame)
        self.node_feature_frames: dict[str, pd.DataFrame] = {}
        self.node_return_series: dict[str, pd.Series] = {}
        self.graph_return_frames: dict[str, pd.DataFrame] = {}
        self._input_cache: dict[pd.Timestamp, CopperModelInputs] = {}
        self._sample_cache: dict[pd.Timestamp, CopperSample] = {}

        seen: set[str] = set()
        for graph_spec in self.graph_specs.values():
            for node in graph_spec.nodes:
                if node.name in seen:
                    continue
                if node.name not in self.asset_frames:
                    raise KeyError(f"missing data for node {node.name!r}")
                frame = self.asset_frames[node.name].sort_index()
                self.node_feature_frames[node.name] = node_feature_frame(frame, node)
                self.node_return_series[node.name] = node_return_series(frame, node)
                seen.add(node.name)

        for graph_name, graph_spec in self.graph_specs.items():
            self.graph_return_frames[graph_name] = pd.DataFrame(
                {
                    node.name: self.node_return_series[node.name]
                    for node in graph_spec.nodes
                }
            ).sort_index()

    def build_inputs(self, date: pd.Timestamp) -> CopperModelInputs:
        date = pd.Timestamp(date)
        cached = self._input_cache.get(date)
        if cached is not None:
            return cached

        market_adj, market_x = build_graph_inputs_from_precomputed(
            self.node_feature_frames,
            self.graph_return_frames["market"],
            self.graph_specs["market"],
            date,
        )
        demand_adj, demand_x = build_graph_inputs_from_precomputed(
            self.node_feature_frames,
            self.graph_return_frames["demand"],
            self.graph_specs["demand"],
            date,
        )
        supply_adj, supply_x = build_graph_inputs_from_precomputed(
            self.node_feature_frames,
            self.graph_return_frames["supply"],
            self.graph_specs["supply"],
            date,
        )
        inputs = CopperModelInputs(
            date=date,
            cnn_x=build_cnn_matrix_from_features(self.cnn_feature_frame, date),
            market_x=market_x,
            market_adj=market_adj,
            demand_x=demand_x,
            demand_adj=demand_adj,
            supply_x=supply_x,
            supply_adj=supply_adj,
        )
        self._input_cache[date] = inputs
        return inputs

    def build(self, date: pd.Timestamp) -> CopperSample:
        date = pd.Timestamp(date)
        cached = self._sample_cache.get(date)
        if cached is not None:
            return cached

        inputs = self.build_inputs(date)
        sample = CopperSample(
            date=inputs.date,
            cnn_x=inputs.cnn_x,
            market_x=inputs.market_x,
            market_adj=inputs.market_adj,
            demand_x=inputs.demand_x,
            demand_adj=inputs.demand_adj,
            supply_x=inputs.supply_x,
            supply_adj=inputs.supply_adj,
            target=build_target(self.copper_frame, inputs.date),
        )
        self._sample_cache[date] = sample
        return sample


class TorchCopperDataset:
    """Minimal torch-compatible dataset.

    The class avoids importing torch until samples are requested, so the feature
    building code can still be inspected in environments without torch.
    """

    def __init__(self, builder: CopperGraphSampleBuilder, dates: Sequence[pd.Timestamp]) -> None:
        self.builder = builder
        self.dates = [pd.Timestamp(date) for date in dates]

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, index: int) -> dict[str, object]:
        import torch

        sample = self.builder.build(self.dates[index])
        return {
            "date": sample.date,
            "cnn_x": torch.from_numpy(sample.cnn_x).float(),
            "market_x": torch.from_numpy(sample.market_x).float(),
            "market_adj": torch.from_numpy(sample.market_adj).float(),
            "demand_x": torch.from_numpy(sample.demand_x).float(),
            "demand_adj": torch.from_numpy(sample.demand_adj).float(),
            "supply_x": torch.from_numpy(sample.supply_x).float(),
            "supply_adj": torch.from_numpy(sample.supply_adj).float(),
            "target": torch.from_numpy(sample.target).float(),
        }


def collate_copper_batch(samples: Sequence[dict[str, object]]) -> dict[str, object]:
    """Collate samples while keeping dates as a Python list."""

    import torch

    dates = [sample["date"] for sample in samples]
    tensor_keys = [key for key in samples[0].keys() if key != "date"]
    batch: dict[str, object] = {"date": dates}
    for key in tensor_keys:
        batch[key] = torch.stack([sample[key] for sample in samples])  # type: ignore[arg-type]
    return batch
