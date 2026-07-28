"""Collect missing GAE financial nodes with yfinance.

This script is intended as a fallback for nodes that are not available through
the current AKShare interfaces. It writes model-ready CSV files under
``data/raw`` with:

date,open,high,low,close,volume
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
import os

import pandas as pd


proxy = "http://127.0.0.1:7890"
os.environ["HTTP_PROXY"] = proxy
os.environ["HTTPS_PROXY"] = proxy

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from copper_prediction.io import safe_node_filename  # noqa: E402


@dataclass(frozen=True)
class YFinanceSpec:
    node: str
    ticker: str
    note: str = ""


DEFAULT_SPECS: tuple[YFinanceSpec, ...] = (
    YFinanceSpec("ABB", "ABBN.SW", "SIX Swiss Exchange primary listing"),
    YFinanceSpec("Schneider Electric", "SU.PA", "Euronext Paris primary listing"),
    YFinanceSpec("Antofagasta", "ANTO.L", "London Stock Exchange primary listing"),
    YFinanceSpec("First Quantum Minerals", "FM.TO", "Toronto Stock Exchange primary listing"),
    YFinanceSpec("Anglo American", "AAL.L", "London Stock Exchange primary listing"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="20150101", help="YYYYMMDD.")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y%m%d"), help="YYYYMMDD, inclusive.")
    parser.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--symbols-file", help="Optional JSON list overriding or adding specs.")
    parser.add_argument("--only-nodes", help="Comma-separated node names to collect.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-proxy", action="store_true", help="Clear proxy environment variables before requests.")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def clear_proxy_environment() -> None:
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ]:
        os.environ.pop(key, None)


def load_specs(path: str | None) -> list[YFinanceSpec]:
    specs = list(DEFAULT_SPECS)
    if not path:
        return specs
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    overrides = payload.get("specs", payload)
    by_node = {spec.node: spec for spec in specs}
    for item in overrides:
        spec = YFinanceSpec(
            node=item["node"],
            ticker=item["ticker"],
            note=item.get("note", ""),
        )
        by_node[spec.node] = spec
    return list(by_node.values())


def filter_specs(specs: list[YFinanceSpec], only_nodes: str | None) -> list[YFinanceSpec]:
    if not only_nodes:
        return specs
    wanted = {name.strip() for name in only_nodes.split(",") if name.strip()}
    filtered = [spec for spec in specs if spec.node in wanted]
    found = {spec.node for spec in filtered}
    missing = sorted(wanted - found)
    if missing:
        raise SystemExit(f"--only-nodes contains unknown nodes: {missing}")
    return filtered


def ymd_to_date(value: str) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def normalize_yfinance_frame(frame: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.reset_index()
    rename = {
        "Date": "date",
        "Datetime": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    frame = frame.rename(columns=rename)
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing yfinance columns {missing}; got {list(frame.columns)}")

    out = frame[required].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    for column in required[1:]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    out = out.loc[~(out[["open", "high", "low", "close"]] <= 0).any(axis=1)].copy()
    prices = out[["open", "high", "low", "close"]]
    out["high"] = prices.max(axis=1)
    out["low"] = prices.min(axis=1)
    out["volume"] = out["volume"].fillna(0.0)
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    out = out[(out["date"] >= start) & (out["date"] <= end)]
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)


def fetch_one(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    import yfinance as yf

    start = ymd_to_date(start_date)
    # yfinance end is exclusive; add one day so CLI end-date behaves inclusive.
    end = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    frame = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        actions=False,
        threads=False,
    )
    return normalize_yfinance_frame(frame, start_date, end_date)


def save_node_csv(frame: pd.DataFrame, node: str, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{safe_node_filename(node)}.csv"
    frame.to_csv(path, index=False)
    return path


def main() -> int:
    args = parse_args()
    if args.no_proxy:
        clear_proxy_environment()
    specs = filter_specs(load_specs(args.symbols_file), args.only_nodes)
    raw_dir = Path(args.raw_dir)

    print(f"Collecting {len(specs)} yfinance nodes into {raw_dir}")
    for spec in specs:
        note = f" - {spec.note}" if spec.note else ""
        print(f"  {spec.node}: {spec.ticker}{note}")
    if args.dry_run:
        return 0

    failures: list[str] = []
    successes = 0
    for idx, spec in enumerate(specs, start=1):
        output_path = raw_dir / f"{safe_node_filename(spec.node)}.csv"
        if args.skip_existing and output_path.exists():
            print(f"[{idx:02d}/{len(specs):02d}] SKIP {spec.node}: exists -> {output_path}")
            successes += 1
            continue
        try:
            for attempt in range(args.retries + 1):
                try:
                    frame = fetch_one(spec.ticker, args.start_date, args.end_date)
                    if frame.empty:
                        raise ValueError(f"no rows returned for {spec.ticker}")
                    path = save_node_csv(frame, spec.node, raw_dir)
                    print(f"[{idx:02d}/{len(specs):02d}] OK   {spec.node}: {len(frame)} rows -> {path}")
                    successes += 1
                    break
                except Exception as exc:  # noqa: BLE001 - retry and report final exception.
                    if attempt >= args.retries:
                        raise
                    print(f"[{idx:02d}/{len(specs):02d}] RETRY {spec.node}: {type(exc).__name__}: {exc}")
                    time.sleep(args.retry_sleep)
        except Exception as exc:  # noqa: BLE001 - continue collecting remaining nodes.
            message = f"[{idx:02d}/{len(specs):02d}] FAIL {spec.node}: {type(exc).__name__}: {exc}"
            print(message)
            failures.append(message)
        time.sleep(args.sleep)

    print(f"\nDone. success={successes}, failed={len(failures)}")
    if failures:
        print("Failed nodes:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
