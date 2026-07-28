"""Evaluate a trained copper interval model against time-series baselines."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from copper_prediction.config import CORR_WINDOW, default_graph_specs  # noqa: E402
from copper_prediction.dataset import (  # noqa: E402
    CopperGraphSampleBuilder,
    TorchCopperDataset,
    collate_copper_batch,
)
from copper_prediction.io import load_asset_frames  # noqa: E402
from copper_prediction.model import CopperIntervalPredictor, CopperIntervalPredictorConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--baseline-window", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--skip-date-validation", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def candidate_dates(copper_frame: pd.DataFrame) -> list[pd.Timestamp]:
    dates = list(pd.to_datetime(copper_frame.index).sort_values())
    return dates[CORR_WINDOW:-1]


def filter_date_range(
    dates: list[pd.Timestamp],
    start_date: str | None,
    end_date: str | None,
) -> list[pd.Timestamp]:
    start = pd.to_datetime(start_date) if start_date else None
    end = pd.to_datetime(end_date) if end_date else None
    return [
        date
        for date in dates
        if (start is None or date >= start) and (end is None or date <= end)
    ]


def valid_dates(
    builder: CopperGraphSampleBuilder,
    dates: list[pd.Timestamp],
    skip_validation: bool,
) -> list[pd.Timestamp]:
    if skip_validation:
        return dates

    valid: list[pd.Timestamp] = []
    failures: list[str] = []
    for index, date in enumerate(dates, start=1):
        try:
            builder.build(date)
        except Exception as exc:  # noqa: BLE001 - report and skip unusable dates
            if len(failures) < 10:
                failures.append(f"{date.date()}: {exc}")
            continue
        valid.append(date)
        if index == 1 or index % 50 == 0 or index == len(dates):
            print(f"Validated dates: {index}/{len(dates)}")

    if failures:
        print("Skipped some dates that could not build samples:")
        for failure in failures:
            print(f"  - {failure}")
    return valid


def split_dates(
    dates: list[pd.Timestamp],
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[pd.Timestamp]]:
    if not 0 < train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("--train-ratio + --val-ratio must be less than 1")

    count = len(dates)
    train_end = max(1, int(count * train_ratio))
    val_end = min(max(train_end + 1, int(count * (train_ratio + val_ratio))), count)
    return {
        "train": dates[:train_end],
        "val": dates[train_end:val_end],
        "test": dates[val_end:],
    }


def model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[CopperIntervalPredictor, dict[str, Any]]:
    checkpoint: Any = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        checkpoint = {"model_state_dict": checkpoint}

    config_data = dict(checkpoint.get("config", {}))
    for tuple_key in ("cnn_channels", "mlp_hidden_dims"):
        if tuple_key in config_data:
            config_data[tuple_key] = tuple(config_data[tuple_key])
    config = CopperIntervalPredictorConfig(**config_data)

    model = CopperIntervalPredictor(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def model_predictions(
    model: CopperIntervalPredictor,
    builder: CopperGraphSampleBuilder,
    dates: list[pd.Timestamp],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        TorchCopperDataset(builder, dates),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_copper_batch,
    )
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader, start=1):
            model_inputs = {
                key: value.to(device)
                for key, value in raw_batch.items()
                if key not in {"date", "target"}
            }
            prediction = model(**model_inputs)
            predictions.append(prediction.detach().cpu().numpy())
            targets.append(raw_batch["target"].numpy())
            print(f"Evaluated batches: {batch_index}/{len(loader)}")

    return np.vstack(predictions), np.vstack(targets)


def required_series(frame: pd.DataFrame, *names: str) -> pd.Series:
    columns = {str(column).lower(): column for column in frame.columns}
    for name in names:
        column = columns.get(name.lower())
        if column is not None:
            return pd.to_numeric(frame[column], errors="coerce")
    raise KeyError(f"missing required column, expected one of {names}")


def realized_interval_returns(copper_frame: pd.DataFrame) -> pd.DataFrame:
    close = required_series(copper_frame, "close", "settlement")
    low = required_series(copper_frame, "low")
    high = required_series(copper_frame, "high")
    return pd.DataFrame(
        {
            "low_return": (low.shift(-1) - close) / close,
            "high_return": (high.shift(-1) - close) / close,
        },
        index=copper_frame.index,
    ).dropna()


def baseline_predictions(
    target_history: pd.DataFrame,
    dates: list[pd.Timestamp],
    window: int,
) -> dict[str, np.ndarray]:
    if window < 1:
        raise ValueError("--baseline-window must be at least 1")

    previous: list[np.ndarray] = []
    rolling_mean: list[np.ndarray] = []
    for date in dates:
        known_history = target_history.loc[target_history.index < date]
        if known_history.empty:
            raise ValueError(f"no baseline history available before {date.date()}")
        previous.append(known_history.iloc[-1].to_numpy(dtype=np.float32))
        rolling_mean.append(known_history.tail(window).mean().to_numpy(dtype=np.float32))
    return {
        "previous_interval": np.vstack(previous),
        f"rolling_mean_{window}": np.vstack(rolling_mean),
    }


def metric_row(
    method: str,
    prediction: np.ndarray,
    target: np.ndarray,
    close: np.ndarray,
) -> dict[str, float | int | str]:
    errors = prediction - target
    absolute_errors = np.abs(errors)
    price_errors = absolute_errors * close[:, None]
    predicted_width = prediction[:, 1] - prediction[:, 0]
    actual_width = target[:, 1] - target[:, 0]

    return {
        "method": method,
        "samples": int(len(target)),
        "return_mae": float(absolute_errors.mean()),
        "return_mae_pct": float(absolute_errors.mean() * 100.0),
        "low_return_mae_pct": float(absolute_errors[:, 0].mean() * 100.0),
        "high_return_mae_pct": float(absolute_errors[:, 1].mean() * 100.0),
        "return_rmse_pct": float(np.sqrt(np.square(errors).mean()) * 100.0),
        "price_mae": float(price_errors.mean()),
        "low_price_mae": float(price_errors[:, 0].mean()),
        "high_price_mae": float(price_errors[:, 1].mean()),
        "low_bias_pct": float(errors[:, 0].mean() * 100.0),
        "high_bias_pct": float(errors[:, 1].mean() * 100.0),
        "order_violation_rate": float(np.mean(prediction[:, 0] > prediction[:, 1])),
        "full_interval_coverage_rate": float(
            np.mean((prediction[:, 0] <= target[:, 0]) & (prediction[:, 1] >= target[:, 1]))
        ),
        "mean_predicted_width_pct": float(predicted_width.mean() * 100.0),
        "mean_actual_width_pct": float(actual_width.mean() * 100.0),
    }


def add_price_columns(
    frame: pd.DataFrame,
    method: str,
    prediction: np.ndarray,
) -> None:
    frame[f"{method}_low_return"] = prediction[:, 0]
    frame[f"{method}_high_return"] = prediction[:, 1]
    frame[f"{method}_low_price"] = frame["close"] * (1.0 + prediction[:, 0])
    frame[f"{method}_high_price"] = frame["close"] * (1.0 + prediction[:, 1])


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output_dir = args.output_dir or args.checkpoint.parent / "evaluation"

    graph_specs = default_graph_specs()
    print(f"Loading data from {args.raw_dir}")
    asset_frames = load_asset_frames(args.raw_dir, graph_specs, args.manifest)
    copper_frame = asset_frames["LME Copper"]
    builder = CopperGraphSampleBuilder(copper_frame, asset_frames, graph_specs)

    dates = candidate_dates(copper_frame)
    dates = filter_date_range(dates, args.start_date, args.end_date)
    dates = valid_dates(builder, dates, args.skip_date_validation)
    if len(dates) < 10:
        raise ValueError(f"not enough valid samples to evaluate: {len(dates)}")

    date_splits = split_dates(dates, args.train_ratio, args.val_ratio)
    evaluation_dates = date_splits[args.split]
    if args.limit_samples is not None:
        evaluation_dates = evaluation_dates[: args.limit_samples]
    if not evaluation_dates:
        raise ValueError(f"{args.split} split is empty")

    print(
        f"Evaluation split: {args.split} ({len(evaluation_dates)} samples, "
        f"{evaluation_dates[0].date()} -> {evaluation_dates[-1].date()})"
    )
    print(f"Device: {device}")

    model, checkpoint = model_from_checkpoint(args.checkpoint, device)
    prediction_by_method: dict[str, np.ndarray] = {}
    model_output, target = model_predictions(
        model,
        builder,
        evaluation_dates,
        args.batch_size,
        device,
    )
    prediction_by_method["model"] = model_output
    prediction_by_method.update(
        baseline_predictions(
            realized_interval_returns(copper_frame),
            evaluation_dates,
            args.baseline_window,
        )
    )

    close = required_series(copper_frame, "close", "settlement").loc[evaluation_dates].to_numpy(dtype=float)
    predictions = pd.DataFrame(
        {
            "date": evaluation_dates,
            "close": close,
            "actual_low_return": target[:, 0],
            "actual_high_return": target[:, 1],
            "actual_low_price": close * (1.0 + target[:, 0]),
            "actual_high_price": close * (1.0 + target[:, 1]),
        }
    )

    summary_rows: list[dict[str, float | int | str]] = []
    for method, method_prediction in prediction_by_method.items():
        summary_rows.append(metric_row(method, method_prediction, target, close))
        add_price_columns(predictions, method, method_prediction)
    summary = pd.DataFrame(summary_rows)

    baseline_name = f"rolling_mean_{args.baseline_window}"
    baseline_mae = float(summary.loc[summary["method"] == baseline_name, "return_mae"].iloc[0])
    summary["mae_improvement_vs_rolling_pct"] = (
        (baseline_mae - summary["return_mae"]) / baseline_mae * 100.0
    )

    display_columns = [
        "method",
        "samples",
        "return_mae_pct",
        "return_rmse_pct",
        "price_mae",
        "order_violation_rate",
        "full_interval_coverage_rate",
        "mae_improvement_vs_rolling_pct",
    ]
    display = summary[display_columns].copy()
    display["order_violation_rate"] *= 100.0
    display["full_interval_coverage_rate"] *= 100.0

    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print("\nEvaluation summary:")
    print(
        display.to_string(
            index=False,
            formatters={
                "return_mae_pct": "{:.4f}".format,
                "return_rmse_pct": "{:.4f}".format,
                "price_mae": "{:.2f}".format,
                "order_violation_rate": "{:.2f}".format,
                "full_interval_coverage_rate": "{:.2f}".format,
                "mae_improvement_vs_rolling_pct": "{:.2f}".format,
            },
        )
    )
    print("Rates and return errors are displayed as percentages; price MAE uses the CSV price unit.")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.split}_summary.csv"
    predictions_path = output_dir / f"{args.split}_predictions.csv"
    summary.to_csv(summary_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    print(f"Saved summary to {summary_path}")
    print(f"Saved predictions to {predictions_path}")


if __name__ == "__main__":
    main()
