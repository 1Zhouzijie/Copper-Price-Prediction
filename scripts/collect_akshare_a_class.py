"""Collect AKShare-available GAE raw data.

This script writes CSV files under ``data/raw`` using the node names expected by
the copper CNN + GAE model. It covers the nodes that can reasonably be fetched
from the currently installed AKShare package:

- SHFE copper and China financial assets
- US/HK stocks and ADR/ETF proxies that AKShare can fetch directly
- China stock indexes and selected global index/FX series
- selected global futures from Sina
- US 10Y yield
- Baltic Dry Index and selected warehouse/inventory series

LME official copper, COMEX copper warehouse data, and non-AKShare-friendly
official/vendor datasets should still be collected by separate pipelines.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable
import warnings

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from copper_prediction.io import safe_node_filename  # noqa: E402


DATE_COLUMN_CANDIDATES = ("date", "日期", "时间")


@dataclass(frozen=True)
class CollectSpec:
    node: str
    kind: str
    symbol: str
    note: str = ""


DEFAULT_SPECS: tuple[CollectSpec, ...] = (
    # Market graph A-class nodes
    CollectSpec("COMEX Copper", "foreign_future", "HG", "Sina foreign futures copper proxy"),
    CollectSpec("SHFE Copper", "shfe_future_sina", "CU0", "SHFE copper continuous contract"),
    CollectSpec("Gold", "foreign_future", "GC", "COMEX gold futures"),
    CollectSpec("Silver", "foreign_future", "SI", "COMEX silver futures"),
    CollectSpec("Crude Oil", "foreign_future", "CL", "NYMEX WTI crude oil futures"),
    CollectSpec("Natural Gas", "foreign_future", "NG", "NYMEX natural gas futures"),
    CollectSpec("Aluminum", "foreign_future", "AHD", "LME aluminum 3M proxy"),
    CollectSpec("Zinc", "foreign_future", "ZSD", "LME zinc 3M proxy"),
    CollectSpec("Nickel", "foreign_future", "NID", "LME nickel 3M proxy"),
    CollectSpec("Lead", "foreign_future", "PBD", "LME lead 3M proxy"),
    CollectSpec("Iron Ore", "foreign_future", "FEF", "SGX iron ore proxy"),
    CollectSpec("DXY", "global_index_em", "美元指数"),
    CollectSpec("US 10Y Yield", "us10y_yield", "美国国债收益率10年", "US 10Y yield from bond_zh_us_rate"),
    CollectSpec("VIX", "us_stock", "VIXY", "temporary VIX futures ETF proxy"),
    CollectSpec("S&P 500", "us_stock", "SPY", "temporary S&P 500 ETF proxy"),
    CollectSpec("Nasdaq 100", "us_stock", "QQQ", "temporary Nasdaq 100 ETF proxy"),
    CollectSpec("MSCI Emerging Markets", "us_stock", "EEM", "temporary MSCI EM ETF proxy"),
    CollectSpec("CSI 300", "zh_index_em", "sh000300", "CSI 300"),
    CollectSpec("CNY/USD", "forex_inverse", "USDCNYC", "inverse of USD/CNY"),
    # Demand graph: US stocks / ADRs
    CollectSpec("NVIDIA", "us_stock", "NVDA"),
    CollectSpec("TSMC", "us_stock", "TSM", "TSMC ADR"),
    CollectSpec("AMD", "us_stock", "AMD"),
    CollectSpec("Broadcom", "us_stock", "AVGO"),
    CollectSpec("Eaton", "us_stock", "ETN"),
    CollectSpec("Schneider Electric", "us_stock", "SBGSY", "Schneider Electric ADR proxy"),
    CollectSpec("ABB", "us_stock", "ABB", "ABB ADR"),
    CollectSpec("Tesla", "us_stock", "TSLA"),
    # Demand graph: HK / China stocks
    CollectSpec("BYD", "hk_stock", "01211", "BYD H share; change to 002594 with kind zh_a_stock if preferred"),
    CollectSpec("CATL / Ningde Times", "zh_a_stock", "300750"),
    CollectSpec("NARI Technology", "zh_a_stock", "600406"),
    CollectSpec("China XD Electric", "zh_a_stock", "601179"),
    CollectSpec("Shanghai Electric", "zh_a_stock", "601727"),
    # Demand graph: China indexes. Verify these index proxies before final research use.
    CollectSpec("CSI New Energy Index", "zh_index_hist", "399808", "CSI New Energy; override if your provider uses another code"),
    CollectSpec("CSI Infrastructure Index", "zh_index_hist", "399995", "CSI infrastructure proxy"),
    CollectSpec("CSI Real Estate Index", "zh_index_hist", "000952", "CSI real-estate proxy"),
    # Supply graph: US-listed miners / ADRs
    CollectSpec("Freeport-McMoRan", "us_stock", "FCX"),
    CollectSpec("Southern Copper", "us_stock", "SCCO"),
    CollectSpec("BHP", "us_stock", "BHP", "BHP ADR"),
    CollectSpec("Rio Tinto", "us_stock", "RIO", "Rio Tinto ADR"),
    CollectSpec("Antofagasta", "us_stock", "ANFGY", "Antofagasta ADR/OTC proxy"),
    CollectSpec("First Quantum Minerals", "us_stock", "FQVLF", "First Quantum ADR/OTC proxy"),
    CollectSpec("Glencore", "us_stock", "GLNCY", "Glencore ADR proxy"),
    CollectSpec("Anglo American", "us_stock", "NGLOY", "Anglo American ADR proxy"),
    # Supply graph: China miners / smelters
    CollectSpec("Zijin Mining", "zh_a_stock", "601899"),
    CollectSpec("Jiangxi Copper", "zh_a_stock", "600362"),
    CollectSpec("Tongling Nonferrous", "zh_a_stock", "000630"),
    CollectSpec("Yunnan Copper", "zh_a_stock", "000878"),
    CollectSpec("China Molybdenum", "zh_a_stock", "603993"),
    # Supply graph: freight
    CollectSpec("Baltic Dry Index", "bdi", "BDI"),
    # Supply graph: inventory nodes available through AKShare.
    CollectSpec("LME Copper Inventory", "lme_stock", "铜-库存"),
    CollectSpec("SHFE Copper Inventory", "shfe_inventory", "沪铜"),
    CollectSpec("LME Cancelled Warrants", "lme_stock", "铜-注销仓单"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="20150101", help="YYYYMMDD, used by date-range-aware interfaces.")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y%m%d"), help="YYYYMMDD.")
    parser.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"), help="Output directory.")
    parser.add_argument("--symbols-file", help="Optional JSON file overriding or adding specs.")
    parser.add_argument("--only-nodes", help="Comma-separated node names to collect, useful for rerunning failures.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip nodes whose output CSV already exists.")
    parser.add_argument("--no-proxy", action="store_true", help="Clear proxy environment variables before AKShare requests.")
    parser.add_argument("--sleep", type=float, default=0.4, help="Seconds to sleep between AKShare requests.")
    parser.add_argument("--retries", type=int, default=2, help="Retries per node after the first attempt.")
    parser.add_argument("--retry-sleep", type=float, default=3.0, help="Seconds to sleep between retries.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first failed node.")
    parser.add_argument("--dry-run", action="store_true", help="Print collection plan without requesting data.")
    return parser.parse_args()


def load_specs(path: str | None) -> list[CollectSpec]:
    specs = list(DEFAULT_SPECS)
    if not path:
        return specs

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    overrides = payload.get("specs", payload)
    by_node = {spec.node: spec for spec in specs}
    for item in overrides:
        spec = CollectSpec(
            node=item["node"],
            kind=item["kind"],
            symbol=item["symbol"],
            note=item.get("note", ""),
        )
        by_node[spec.node] = spec
    return list(by_node.values())


def filter_specs(specs: list[CollectSpec], only_nodes: str | None) -> list[CollectSpec]:
    if not only_nodes:
        return specs
    wanted = {name.strip() for name in only_nodes.split(",") if name.strip()}
    filtered = [spec for spec in specs if spec.node in wanted]
    found = {spec.node for spec in filtered}
    missing = sorted(wanted - found)
    if missing:
        raise SystemExit(f"--only-nodes contains unknown nodes: {missing}")
    return filtered


def get_akshare() -> Any:
    try:
        import akshare as ak
    except ImportError as exc:
        raise SystemExit(
            "AKShare is not installed. Install it with:\n"
            "  python3 -m pip install akshare -U\n"
            "or after installing project requirements."
        ) from exc
    return ak


def clear_proxy_environment() -> None:
    proxy_keys = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ]
    for key in proxy_keys:
        os.environ.pop(key, None)


def first_existing_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower_map = {str(col).lower(): str(col) for col in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def numeric_column(frame: pd.DataFrame, *candidates: str) -> pd.Series | None:
    column = first_existing_column(frame, tuple(candidates))
    if column is None:
        return None
    return pd.to_numeric(frame[column], errors="coerce")


def date_series(frame: pd.DataFrame) -> pd.Series:
    column = first_existing_column(frame, DATE_COLUMN_CANDIDATES)
    if column is not None:
        values = frame[column]
        numeric_values = pd.to_numeric(values, errors="coerce")
        if numeric_values.notna().any():
            yyyymmdd = numeric_values.dropna().astype("int64")
            if ((yyyymmdd >= 19000101) & (yyyymmdd <= 21001231)).all():
                return pd.to_datetime(values.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
        return pd.to_datetime(values, errors="coerce")
    if not isinstance(frame.index, pd.RangeIndex):
        return pd.to_datetime(frame.index, errors="coerce").to_series(index=frame.index)
    return pd.to_datetime(frame.iloc[:, 0], errors="coerce")


def clean_dates(frame: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    frame = frame[(frame["date"] >= start) & (frame["date"] <= end)]
    frame = frame.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return frame.reset_index(drop=True)


def clean_financial_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame = frame.loc[~(frame[["open", "high", "low", "close"]] <= 0).any(axis=1)].copy()
    prices = frame[["open", "high", "low", "close"]]
    frame["high"] = prices.max(axis=1)
    frame["low"] = prices.min(axis=1)
    frame["volume"] = frame["volume"].fillna(0.0)
    return frame.reset_index(drop=True)


def normalize_ohlcv(frame: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["date"] = date_series(frame)
    out["open"] = numeric_column(frame, "open", "开盘", "开盘价", "今开")
    out["high"] = numeric_column(frame, "high", "最高", "最高价")
    out["low"] = numeric_column(frame, "low", "最低", "最低价")
    out["close"] = numeric_column(frame, "close", "收盘", "收盘价", "最新价", "latest", "指数", "value")
    volume = numeric_column(frame, "volume", "vol", "成交量", "总量", "amount", "成交额")
    out["volume"] = 0.0 if volume is None else volume
    if "open_interest" in frame.columns or "持仓量" in frame.columns or "hold" in frame.columns or "持仓" in frame.columns:
        oi = numeric_column(frame, "open_interest", "持仓量", "hold", "持仓")
        if oi is not None:
            out["open_interest"] = oi
    required = ["open", "high", "low", "close"]
    for column in required:
        if column not in out or out[column].isna().all():
            if column == "open" and "close" in out:
                out[column] = out["close"]
            elif column == "high" and "close" in out:
                out[column] = out["close"]
            elif column == "low" and "close" in out:
                out[column] = out["close"]
            else:
                raise ValueError(f"cannot normalize OHLCV data; missing {column!r}. columns={list(frame.columns)}")
    return clean_financial_frame(clean_dates(out, start_date, end_date))


def normalize_inventory(frame: pd.DataFrame, value: pd.Series, start_date: str, end_date: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["date"] = date_series(frame)
    out["inventory"] = pd.to_numeric(value, errors="coerce")
    out = clean_dates(out, start_date, end_date)
    out = out.dropna(subset=["inventory"])
    return out[["date", "inventory"]].reset_index(drop=True)


def normalize_close_only(frame: pd.DataFrame, value: pd.Series, start_date: str, end_date: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["date"] = date_series(frame)
    out["close"] = pd.to_numeric(value, errors="coerce")
    out["open"] = out["close"]
    out["high"] = out["close"]
    out["low"] = out["close"]
    out["volume"] = 0.0
    return clean_financial_frame(clean_dates(out[["date", "open", "high", "low", "close", "volume"]], start_date, end_date))


def prefixed_cn_symbol(symbol: str) -> str:
    if symbol.startswith(("6", "9")):
        return f"sh{symbol}"
    return f"sz{symbol}"


def prefixed_index_symbol(symbol: str) -> str:
    if symbol.startswith("399"):
        return f"sz{symbol}"
    return f"sh{symbol}"


def fetch_zh_a_stock(ak: Any, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    errors: list[str] = []
    attempts = [
        (
            "eastmoney stock_zh_a_hist",
            lambda: ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="",
            ),
        ),
        (
            "tencent stock_zh_a_hist_tx",
            lambda: ak.stock_zh_a_hist_tx(
                symbol=prefixed_cn_symbol(symbol),
                start_date=start_date,
                end_date=end_date,
                adjust="",
            ),
        ),
        (
            "sina stock_zh_a_daily",
            lambda: ak.stock_zh_a_daily(
                symbol=prefixed_cn_symbol(symbol),
                start_date=start_date,
                end_date=end_date,
                adjust="",
            ),
        ),
    ]
    for label, fetch in attempts:
        try:
            return normalize_ohlcv(fetch(), start_date, end_date)
        except Exception as exc:  # noqa: BLE001 - preserve fallback context.
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


def fetch_hk_stock(ak: Any, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    errors: list[str] = []
    attempts = [
        (
            "eastmoney stock_hk_hist",
            lambda: ak.stock_hk_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="",
            ),
        ),
        ("sina stock_hk_daily", lambda: ak.stock_hk_daily(symbol=symbol, adjust="")),
    ]
    for label, fetch in attempts:
        try:
            return normalize_ohlcv(fetch(), start_date, end_date)
        except Exception as exc:  # noqa: BLE001 - preserve fallback context.
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


def fetch_us_stock(ak: Any, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    frame = ak.stock_us_daily(symbol=symbol, adjust="")
    return normalize_ohlcv(frame, start_date, end_date)


def fetch_foreign_future(ak: Any, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    frame = ak.futures_foreign_hist(symbol=symbol)
    return normalize_ohlcv(frame, start_date, end_date)


def fetch_shfe_future_sina(ak: Any, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    frame = ak.futures_zh_daily_sina(symbol=symbol)
    return normalize_ohlcv(frame, start_date, end_date)


def fetch_zh_index_em(ak: Any, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    frame = ak.stock_zh_index_daily_em(symbol=symbol, start_date=start_date, end_date=end_date)
    return normalize_ohlcv(frame, start_date, end_date)


def fetch_zh_index_hist(ak: Any, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    errors: list[str] = []
    attempts = [
        (
            "eastmoney index_zh_a_hist",
            lambda: ak.index_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
            ),
        ),
        (
            "tencent stock_zh_a_hist_tx",
            lambda: ak.stock_zh_a_hist_tx(
                symbol=prefixed_index_symbol(symbol),
                start_date=start_date,
                end_date=end_date,
                adjust="",
            ),
        ),
    ]
    for label, fetch in attempts:
        try:
            return normalize_ohlcv(fetch(), start_date, end_date)
        except Exception as exc:  # noqa: BLE001 - preserve fallback context.
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


def fetch_global_index_em(ak: Any, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    frame = ak.index_global_hist_em(symbol=symbol)
    return normalize_ohlcv(frame, start_date, end_date)


def fetch_forex_inverse(ak: Any, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    frame = ak.forex_hist_em(symbol=symbol)
    close = numeric_column(frame, "close", "收盘", "收盘价", "最新价")
    if close is None:
        raise KeyError(f"missing forex close column. columns={list(frame.columns)}")
    close = close.replace(0, pd.NA)
    return normalize_close_only(frame, 1 / close, start_date, end_date)


def fetch_us10y_yield(ak: Any, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    frame = ak.bond_zh_us_rate(start_date=start_date)
    if symbol not in frame.columns:
        raise KeyError(f"missing {symbol!r} in bond_zh_us_rate columns={list(frame.columns)}")
    return normalize_close_only(frame, frame[symbol], start_date, end_date)


def fetch_bdi(ak: Any, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    del symbol
    frame = ak.macro_shipping_bdi()
    value = numeric_column(frame, "指数", "close", "最新价", "最新值")
    if value is None:
        raise KeyError(f"missing BDI value column. columns={list(frame.columns)}")
    return normalize_close_only(frame, value, start_date, end_date)


def fetch_lme_stock(ak: Any, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    frame = ak.macro_euro_lme_stock()
    if symbol not in frame.columns:
        raise KeyError(f"missing {symbol!r} in LME stock columns={list(frame.columns)}")
    return normalize_inventory(frame, frame[symbol], start_date, end_date)


def fetch_shfe_cu_receipt(ak: Any, start_date: str, end_date: str) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with contextlib.redirect_stdout(io.StringIO()):
            return ak.get_receipt(start_date=start_date, end_date=end_date, vars_list=["CU"])


def month_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    current = pd.Timestamp(start.year, start.month, 1)
    ranges: list[tuple[str, str]] = []
    while current <= end:
        month_start = max(current, start)
        month_end = min(current + pd.offsets.MonthEnd(0), end)
        ranges.append((month_start.strftime("%Y%m%d"), month_end.strftime("%Y%m%d")))
        current = current + pd.offsets.MonthBegin(1)
    return ranges


def fetch_shfe_cu_daily_receipt(ak: Any, start_date: str, end_date: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for date in pd.bdate_range(pd.to_datetime(start_date), pd.to_datetime(end_date)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with contextlib.redirect_stdout(io.StringIO()):
                    payload = ak.futures_shfe_warehouse_receipt(date=date.strftime("%Y%m%d"))
            copper_frame = payload.get("铜") if isinstance(payload, dict) else None
            if copper_frame is None or copper_frame.empty:
                continue

            status = copper_frame["ROWSTATUS"].astype(str)
            total = copper_frame.loc[
                (status == "2") & (copper_frame["WHABBRNAME"].astype(str) == "总计"),
                "WRTWGHTS",
            ]
            if total.empty:
                inventory = pd.to_numeric(copper_frame.loc[status == "0", "WRTWGHTS"], errors="coerce").sum()
            else:
                inventory = pd.to_numeric(total, errors="coerce").iloc[-1]
            records.append({"date": date.strftime("%Y%m%d"), "receipt": inventory})
        except Exception:  # noqa: BLE001 - some SHFE dates return empty non-JSON responses.
            continue
    return pd.DataFrame(records)


def fetch_shfe_cu_receipt_history(ak: Any, start_date: str, end_date: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    for month_start, month_end in month_ranges(start_date, end_date):
        try:
            frame = fetch_shfe_cu_receipt(ak, month_start, month_end)
        except Exception as exc:  # noqa: BLE001 - try the newer daily endpoint for format breaks.
            failures.append(f"{month_start}-{month_end}: get_receipt {type(exc).__name__}: {exc}")
            frame = fetch_shfe_cu_daily_receipt(ak, month_start, month_end)
        if not frame.empty:
            frames.append(frame)

    if frames:
        return pd.concat(frames, ignore_index=True)
    raise RuntimeError("no SHFE CU receipt rows returned; " + "; ".join(failures[-5:]))


def fetch_shfe_inventory(ak: Any, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    errors: list[str] = []
    attempts = [
        (
            "akshare SHFE CU registered receipt history",
            lambda: fetch_shfe_cu_receipt_history(ak, start_date, end_date),
        ),
        ("eastmoney futures_inventory_em", lambda: ak.futures_inventory_em(symbol=symbol)),
        ("99qh futures_inventory_99", lambda: ak.futures_inventory_99(symbol=symbol)),
    ]
    for label, fetch in attempts:
        try:
            frame = fetch()
            value = numeric_column(frame, "inventory", "库存", "仓单", "receipt", "value")
            if value is None:
                raise KeyError(f"missing inventory column. columns={list(frame.columns)}")
            return normalize_inventory(frame, value, start_date, end_date)
        except Exception as exc:  # noqa: BLE001 - preserve fallback context.
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


FETCHERS: dict[str, Callable[[Any, str, str, str], pd.DataFrame]] = {
    "zh_a_stock": fetch_zh_a_stock,
    "hk_stock": fetch_hk_stock,
    "us_stock": fetch_us_stock,
    "foreign_future": fetch_foreign_future,
    "shfe_future_sina": fetch_shfe_future_sina,
    "zh_index_em": fetch_zh_index_em,
    "zh_index_hist": fetch_zh_index_hist,
    "global_index_em": fetch_global_index_em,
    "forex_inverse": fetch_forex_inverse,
    "us10y_yield": fetch_us10y_yield,
    "bdi": fetch_bdi,
    "lme_stock": fetch_lme_stock,
    "shfe_inventory": fetch_shfe_inventory,
}


def save_node_csv(frame: pd.DataFrame, node: str, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{safe_node_filename(node)}.csv"
    frame.to_csv(path, index=False)
    return path


def collect_one(ak: Any, spec: CollectSpec, start_date: str, end_date: str, raw_dir: Path) -> tuple[Path, int]:
    fetcher = FETCHERS.get(spec.kind)
    if fetcher is None:
        raise KeyError(f"unsupported kind {spec.kind!r} for node {spec.node!r}")
    frame = fetcher(ak, spec.symbol, start_date, end_date)
    if frame.empty:
        raise ValueError(f"no rows returned for {spec.node!r} ({spec.kind}:{spec.symbol})")
    path = save_node_csv(frame, spec.node, raw_dir)
    return path, len(frame)


def main() -> int:
    args = parse_args()
    if args.no_proxy:
        clear_proxy_environment()
    specs = filter_specs(load_specs(args.symbols_file), args.only_nodes)
    raw_dir = Path(args.raw_dir)

    print(f"Collecting {len(specs)} AKShare-available nodes into {raw_dir}")
    for spec in specs:
        note = f" - {spec.note}" if spec.note else ""
        print(f"  {spec.node}: {spec.kind}:{spec.symbol}{note}")
    if args.dry_run:
        return 0

    ak = get_akshare()
    successes: list[str] = []
    failures: list[str] = []

    for idx, spec in enumerate(specs, start=1):
        output_path = raw_dir / f"{safe_node_filename(spec.node)}.csv"
        if args.skip_existing and output_path.exists():
            successes.append(spec.node)
            print(f"[{idx:02d}/{len(specs):02d}] SKIP {spec.node}: exists -> {output_path}")
            continue

        try:
            last_exc: Exception | None = None
            for attempt in range(args.retries + 1):
                try:
                    path, rows = collect_one(ak, spec, args.start_date, args.end_date, raw_dir)
                    break
                except Exception as exc:  # noqa: BLE001 - retry and report the final exception.
                    last_exc = exc
                    if attempt >= args.retries:
                        raise
                    print(
                        f"[{idx:02d}/{len(specs):02d}] RETRY {spec.node}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    time.sleep(args.retry_sleep)
            else:
                raise RuntimeError(f"unreachable retry state for {spec.node}") from last_exc
            successes.append(spec.node)
            print(f"[{idx:02d}/{len(specs):02d}] OK   {spec.node}: {rows} rows -> {path}")
        except Exception as exc:  # noqa: BLE001 - collection should continue by default.
            message = f"[{idx:02d}/{len(specs):02d}] FAIL {spec.node}: {type(exc).__name__}: {exc}"
            failures.append(message)
            print(message)
            if args.fail_fast:
                raise
        time.sleep(args.sleep)

    print(f"\nDone. success={len(successes)}, failed={len(failures)}")
    if failures:
        print("Failed nodes:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
