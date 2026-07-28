"""Normalize LME copper CSV columns and filename for the model.

The script accepts a raw LME copper file with either Chinese or English column
names and writes ``data/raw/LME Copper.csv`` with:

date,open,high,low,close,volume,open_interest
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


COLUMN_MAP = {
    "时间": "date",
    "日期": "date",
    "Date": "date",
    "date": "date",
    "开盘价": "open",
    "开盘": "open",
    "Open": "open",
    "open": "open",
    "最高价": "high",
    "最高": "high",
    "High": "high",
    "high": "high",
    "最低价": "low",
    "最低": "low",
    "Low": "low",
    "low": "low",
    "收盘价": "close",
    "收盘": "close",
    "Close": "close",
    "close": "close",
    "成交量": "volume",
    "Volume": "volume",
    "volume": "volume",
    "持仓量": "open_interest",
    "持仓": "open_interest",
    "OpenInterest": "open_interest",
    "Open Interest": "open_interest",
    "open_interest": "open_interest",
    "oi": "open_interest",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        help="Raw LME copper CSV. If omitted, auto-detects a file matching *lme*copper*.csv under --raw-dir.",
    )
    parser.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    return parser.parse_args()


def detect_input(raw_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in raw_dir.glob("*.csv")
        if "lme" in path.name.lower() and "copper" in path.name.lower()
    )
    if not candidates:
        raise FileNotFoundError(f"no LME copper CSV found under {raw_dir}")
    return candidates[0]


def normalize_lme(input_path: Path, output_path: Path) -> int:
    frame = pd.read_csv(input_path)
    frame = frame.rename(columns={col: COLUMN_MAP.get(str(col), col) for col in frame.columns})
    required = ["date", "open", "high", "low", "close", "volume", "open_interest"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"missing columns after rename: {missing}; got {list(frame.columns)}")

    frame = frame[required].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
    frame = frame.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", dir=output_path.parent, delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        frame.to_csv(temp_path, index=False)
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return len(frame)


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    input_path = Path(args.input) if args.input else detect_input(raw_dir)
    output_path = raw_dir / "LME Copper.csv"
    rows = normalize_lme(input_path, output_path)
    print(f"normalized {input_path} -> {output_path} ({rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
