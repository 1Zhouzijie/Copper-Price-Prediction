"""CSV loading helpers for raw node data."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re

import pandas as pd

from .config import GraphSpec, NodeSpec, default_graph_specs


def safe_node_filename(node_name: str) -> str:
    """Return a filesystem-friendly CSV stem for a node name."""

    name = node_name.replace(" / ", "_")
    name = name.replace("/", "_")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def unique_nodes(graph_specs: Mapping[str, GraphSpec] | None = None) -> list[NodeSpec]:
    """Return graph nodes without duplicates, preserving first appearance."""

    specs = graph_specs or default_graph_specs()
    seen: set[str] = set()
    nodes: list[NodeSpec] = []
    for spec in specs.values():
        for node in spec.nodes:
            if node.name not in seen:
                nodes.append(node)
                seen.add(node.name)
    return nodes


def read_node_csv(path: Path) -> pd.DataFrame:
    """Read one node CSV and return a date-indexed frame."""

    frame = pd.read_csv(path)
    columns_lower = {str(col).lower(): col for col in frame.columns}
    date_col = columns_lower.get("date")
    if date_col is None:
        date_col = frame.columns[0]
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.set_index(date_col).sort_index()
    frame.index.name = "date"
    return frame


def read_manifest(path: Path) -> dict[str, Path]:
    """Read a node-to-path manifest.

    The manifest must contain columns:
    - node: exact NodeSpec.name
    - path: CSV path, absolute or relative to the manifest parent directory
    """

    manifest = pd.read_csv(path)
    required = {"node", "path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest {path} is missing columns: {sorted(missing)}")

    mapping: dict[str, Path] = {}
    for _, row in manifest.iterrows():
        csv_path = Path(str(row["path"]))
        if not csv_path.is_absolute():
            csv_path = path.parent / csv_path
        mapping[str(row["node"])] = csv_path
    return mapping


def candidate_paths(raw_dir: Path, node_name: str) -> list[Path]:
    """Return possible CSV paths for a node."""

    safe_name = safe_node_filename(node_name)
    candidates = [
        raw_dir / f"{safe_name}.csv",
        raw_dir / f"{node_name}.csv",
        raw_dir / f"{node_name.replace(' / ', '_')}.csv",
        raw_dir / f"{node_name.replace('/', '_')}.csv",
    ]
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def load_asset_frames(
    raw_dir: Path,
    graph_specs: Mapping[str, GraphSpec] | None = None,
    manifest_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load all required node frames from CSV files."""

    specs = graph_specs or default_graph_specs()
    manifest = read_manifest(manifest_path) if manifest_path else {}
    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for node in unique_nodes(specs):
        path = manifest.get(node.name)
        if path is None:
            for candidate in candidate_paths(raw_dir, node.name):
                if candidate.exists():
                    path = candidate
                    break
        if path is None or not path.exists():
            expected = ", ".join(str(p) for p in candidate_paths(raw_dir, node.name)[:2])
            missing.append(f"{node.name} (expected one of: {expected})")
            continue
        frames[node.name] = read_node_csv(path)

    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(f"missing CSV data for nodes:\n  - {joined}")
    return frames
