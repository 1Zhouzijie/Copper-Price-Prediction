"""Collect COMEX copper inventory from COCHILCO monthly bulletins.

COCHILCO publishes table 4.2 in its monthly electronic bulletin:
"Daily Copper Inventories Across Regions, Shown By Exchange". The COMEX TOTAL
column is used here as the model's COMEX Copper Inventory series.

Output format:
date,inventory
"""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path
import time

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
URL_TEMPLATE = "https://boletin.cochilco.cl/productos/boletin.asp?anio={year}&mes={month:02d}&tabla=tabla4_2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="20150101", help="YYYYMMDD.")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y%m%d"), help="YYYYMMDD.")
    parser.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    return parser.parse_args()


def month_range(start_date: str, end_date: str) -> list[pd.Timestamp]:
    start = pd.to_datetime(start_date).to_period("M").to_timestamp()
    end = pd.to_datetime(end_date).to_period("M").to_timestamp()
    return list(pd.date_range(start, end, freq="MS"))


def parse_chilean_number(value: object) -> float | None:
    text = str(value).strip()
    if not text or text.lower() == "nan" or text == "-":
        return None
    text = text.replace(".", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def find_inventory_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    for table in tables:
        text = " ".join(str(value) for value in table.head(4).to_numpy().ravel())
        if "COMEX" in text and ("Cierre Diario" in text or "Daily Closings" in text):
            return table
    raise ValueError("could not find COCHILCO table 4.2 inventory table")


def fetch_month(year: int, month: int) -> pd.DataFrame:
    url = URL_TEMPLATE.format(year=year, month=month)
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    table = find_inventory_table(tables)

    rows: list[dict[str, object]] = []
    for _, row in table.iloc[3:].iterrows():
        day = pd.to_numeric(row.iloc[0], errors="coerce")
        if pd.isna(day):
            continue
        day_int = int(day)
        try:
            date = pd.Timestamp(year=year, month=month, day=day_int)
        except ValueError:
            continue
        inventory = parse_chilean_number(row.iloc[6])
        if inventory is None:
            continue
        rows.append({"date": date, "inventory": inventory})
    return pd.DataFrame(rows)


def fetch_month_with_retries(year: int, month: int, retries: int, retry_sleep: float) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fetch_month(year, month)
        except Exception as exc:  # noqa: BLE001 - report the final exception with context.
            last_error = exc
            if attempt >= retries:
                raise
            print(f"RETRY {year}-{month:02d}: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(retry_sleep)
    raise RuntimeError(f"unreachable retry state for {year}-{month:02d}") from last_error


def main() -> int:
    args = parse_args()
    start = pd.to_datetime(args.start_date)
    end = pd.to_datetime(args.end_date)
    frames: list[pd.DataFrame] = []
    failures: list[str] = []

    months = month_range(args.start_date, args.end_date)
    print(f"Collecting COCHILCO COMEX copper inventory for {len(months)} months", flush=True)
    for idx, month in enumerate(months, start=1):
        try:
            frame = fetch_month_with_retries(month.year, month.month, args.retries, args.retry_sleep)
            if not frame.empty:
                frames.append(frame)
            print(f"[{idx:03d}/{len(months):03d}] OK   {month:%Y-%m}: {len(frame)} rows", flush=True)
        except Exception as exc:  # noqa: BLE001 - continue remaining months.
            message = f"[{idx:03d}/{len(months):03d}] FAIL {month:%Y-%m}: {type(exc).__name__}: {exc}"
            failures.append(message)
            print(message, flush=True)
        time.sleep(args.sleep)

    if not frames:
        raise SystemExit("no inventory rows collected")

    output = pd.concat(frames, ignore_index=True)
    output = output[(output["date"] >= start) & (output["date"] <= end)].copy()
    output = output.sort_values("date").drop_duplicates("date", keep="last")
    output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%d")
    output["inventory"] = pd.to_numeric(output["inventory"], errors="coerce")
    output = output.dropna(subset=["inventory"])

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "COMEX Copper Inventory.csv"
    output.to_csv(path, index=False)

    print(f"\nSaved {len(output)} rows -> {path}", flush=True)
    if failures:
        print(f"Failures: {len(failures)}", flush=True)
        for failure in failures:
            print(f"  {failure}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
