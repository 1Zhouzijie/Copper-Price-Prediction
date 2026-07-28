"""Train the CNN + three-GAE copper interval predictor."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
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
from copper_prediction.model import (  # noqa: E402
    CopperIntervalPredictor,
    CopperIntervalPredictorConfig,
    training_loss_components,
)

LOSS_METRIC_KEYS = (
    "total_loss",
    "supervised_loss",
    "mse_loss",
    "bound_penalty_loss",
    "reconstruction_loss",
    "market_reconstruction_loss",
    "demand_reconstruction_loss",
    "supply_reconstruction_loss",
    "weighted_supervised_loss",
    "weighted_reconstruction_loss",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--start-date", default=None, help="Optional first sample date, e.g. 20230302.")
    parser.add_argument("--end-date", default=None, help="Optional last sample date, e.g. 20251117.")
    parser.add_argument("--supervised-weight", type=float, default=1.0)
    parser.add_argument("--bound-penalty-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=5e-4)
    parser.add_argument(
        "--interval-weight",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--selection-metric",
        choices=["mae", "supervised_loss", "total_loss"],
        default="mae",
        help="Validation metric used to select best_model.pt.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=12,
        help="Stop after this many epochs without validation improvement; 0 disables.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-6,
        help="Minimum validation-metric decrease counted as an improvement.",
    )
    parser.add_argument("--graph-encoder", choices=["gcn", "gat"], default="gcn")
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--gat-attention-dropout", type=float, default=0.1)
    parser.add_argument("--fusion-mode", choices=["concat", "branch_attention"], default="concat")
    parser.add_argument("--branch-attention-dim", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--skip-date-validation", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    # Need 60 days for correlations and one following day for the label.
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
    candidates: list[pd.Timestamp],
    skip_validation: bool,
) -> list[pd.Timestamp]:
    if skip_validation:
        return candidates

    valid: list[pd.Timestamp] = []
    failures: list[str] = []
    for index, date in enumerate(candidates, start=1):
        try:
            builder.build(date)
        except Exception as exc:  # noqa: BLE001 - report and skip unusable dates
            if len(failures) < 10:
                failures.append(f"{date.date()}: {exc}")
            continue
        valid.append(date)
        if index == 1 or index % 50 == 0 or index == len(candidates):
            print(f"Validated dates: {index}/{len(candidates)}")

    if failures:
        print("Skipped some dates that could not build samples:")
        for failure in failures:
            print(f"  - {failure}")
    return valid


def split_dates(
    dates: list[pd.Timestamp],
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[pd.Timestamp], list[pd.Timestamp], list[pd.Timestamp]]:
    if not 0 < train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("--train-ratio + --val-ratio must be less than 1")

    n_dates = len(dates)
    train_end = max(1, int(n_dates * train_ratio))
    val_end = max(train_end + 1, int(n_dates * (train_ratio + val_ratio)))
    val_end = min(val_end, n_dates)
    return dates[:train_end], dates[train_end:val_end], dates[val_end:]


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {"date": batch["date"]}
    for key, value in batch.items():
        if key == "date":
            continue
        moved[key] = value.to(device)
    return moved


def validate_training_args(args: argparse.Namespace) -> None:
    nonnegative_weights = {
        "--supervised-weight": args.supervised_weight,
        "--bound-penalty-weight": args.bound_penalty_weight,
        "--reconstruction-weight": args.reconstruction_weight,
    }
    for name, value in nonnegative_weights.items():
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be nonnegative")
    if args.early_stopping_min_delta < 0:
        raise ValueError("--early-stopping-min-delta must be nonnegative")


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in vars(args).items():
        values[key] = str(value) if isinstance(value, Path) else value
    return values


def date_range_metadata(dates: list[pd.Timestamp]) -> dict[str, Any]:
    if not dates:
        return {"samples": 0, "start": None, "end": None}
    return {
        "samples": len(dates),
        "start": str(dates[0].date()),
        "end": str(dates[-1].date()),
    }


def save_run_config(path: Path, run_config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def forward_loss(
    model: CopperIntervalPredictor,
    batch: dict[str, Any],
    supervised_weight: float,
    bound_penalty_weight: float,
    reconstruction_weight: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    prediction, aux = model(
        cnn_x=batch["cnn_x"],
        market_x=batch["market_x"],
        market_adj=batch["market_adj"],
        demand_x=batch["demand_x"],
        demand_adj=batch["demand_adj"],
        supply_x=batch["supply_x"],
        supply_adj=batch["supply_adj"],
        return_aux=True,
    )
    components = training_loss_components(
        prediction=prediction,
        target=batch["target"],
        aux=aux,
        market_adj=batch["market_adj"],
        demand_adj=batch["demand_adj"],
        supply_adj=batch["supply_adj"],
        supervised_weight=supervised_weight,
        bound_penalty_weight=bound_penalty_weight,
        reconstruction_weight=reconstruction_weight,
    )
    return components, prediction


def run_epoch(
    model: CopperIntervalPredictor,
    loader: DataLoader,
    device: torch.device,
    supervised_weight: float,
    bound_penalty_weight: float,
    reconstruction_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    metric_totals = {key: 0.0 for key in LOSS_METRIC_KEYS}
    metric_totals.update(
        {
            "mae": 0.0,
            "low_mae": 0.0,
            "high_mae": 0.0,
            "order_violation_rate": 0.0,
        }
    )
    total_count = 0

    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        with torch.set_grad_enabled(training):
            components, prediction = forward_loss(
                model,
                batch,
                supervised_weight,
                bound_penalty_weight,
                reconstruction_weight,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                components["total_loss"].backward()
                optimizer.step()

        batch_size = int(batch["target"].shape[0])
        for key in LOSS_METRIC_KEYS:
            metric_totals[key] += float(components[key].detach().cpu()) * batch_size

        absolute_error = torch.abs(prediction.detach() - batch["target"])
        metric_totals["mae"] += float(absolute_error.mean().cpu()) * batch_size
        metric_totals["low_mae"] += float(absolute_error[:, 0].mean().cpu()) * batch_size
        metric_totals["high_mae"] += float(absolute_error[:, 1].mean().cpu()) * batch_size
        metric_totals["order_violation_rate"] += (
            float((prediction.detach()[:, 0] > prediction.detach()[:, 1]).float().mean().cpu())
            * batch_size
        )
        total_count += batch_size

    if total_count == 0:
        return {key: float("nan") for key in metric_totals}
    return {key: value / total_count for key, value in metric_totals.items()}


def save_checkpoint(
    path: Path,
    model: CopperIntervalPredictor,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    selection_metric: str,
    selection_value: float,
    run_config: dict[str, Any],
    test_metrics: dict[str, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(model.config),
            "epoch": epoch,
            "train_loss": train_metrics["total_loss"],
            "val_loss": val_metrics["total_loss"],
            "test_loss": None if test_metrics is None else test_metrics["total_loss"],
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "selection_metric": selection_metric,
            "selection_value": selection_value,
            "run_config": run_config,
        },
        path,
    )


def history_row(
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    selection_metric: str,
    selection_value: float,
    improved: bool,
) -> dict[str, float | int | bool | str]:
    row: dict[str, float | int | bool | str] = {
        "epoch": epoch,
        "selection_metric": selection_metric,
        "selection_value": selection_value,
        "is_best": improved,
    }
    for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
    return row


def format_metrics(metrics: dict[str, float]) -> str:
    return (
        f"total={metrics['total_loss']:.6f} "
        f"supervised={metrics['supervised_loss']:.6f} "
        f"mse={metrics['mse_loss']:.6f} "
        f"bound={metrics['bound_penalty_loss']:.6f} "
        f"recon={metrics['reconstruction_loss']:.6f} "
        f"weighted_recon={metrics['weighted_reconstruction_loss']:.6f} "
        f"recon_market={metrics['market_reconstruction_loss']:.6f} "
        f"recon_demand={metrics['demand_reconstruction_loss']:.6f} "
        f"recon_supply={metrics['supply_reconstruction_loss']:.6f} "
        f"mae={metrics['mae']:.6f} "
        f"low_mae={metrics['low_mae']:.6f} "
        f"high_mae={metrics['high_mae']:.6f} "
        f"order_violation={metrics['order_violation_rate']:.4%}"
    )


def main() -> None:
    args = parse_args()
    if args.interval_weight is not None:
        print(
            "Warning: --interval-weight is deprecated; treating it as "
            "--bound-penalty-weight."
        )
        args.bound_penalty_weight = args.interval_weight
    validate_training_args(args)
    set_seed(args.seed)
    device = choose_device(args.device)

    graph_specs = default_graph_specs()
    print(f"Loading data from {args.raw_dir}")
    asset_frames = load_asset_frames(args.raw_dir, graph_specs, args.manifest)
    copper_frame = asset_frames["LME Copper"]
    builder = CopperGraphSampleBuilder(copper_frame, asset_frames, graph_specs)

    dates = candidate_dates(copper_frame)
    dates = filter_date_range(dates, args.start_date, args.end_date)
    if args.limit_samples is not None:
        dates = dates[: args.limit_samples]
    if dates:
        print(f"Candidate sample dates: {len(dates)} ({dates[0].date()} -> {dates[-1].date()})")
    dates = valid_dates(builder, dates, args.skip_date_validation)
    if len(dates) < 10:
        raise ValueError(f"not enough valid samples to train: {len(dates)}")

    train_dates, val_dates, test_dates = split_dates(dates, args.train_ratio, args.val_ratio)
    print(f"Valid samples: {len(dates)}")
    print(f"Train/val/test: {len(train_dates)}/{len(val_dates)}/{len(test_dates)}")
    print(f"Device: {device}")

    train_loader = DataLoader(
        TorchCopperDataset(builder, train_dates),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_copper_batch,
    )
    val_loader = DataLoader(
        TorchCopperDataset(builder, val_dates),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_copper_batch,
    )
    test_loader = DataLoader(
        TorchCopperDataset(builder, test_dates),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_copper_batch,
    )

    model_config = CopperIntervalPredictorConfig(
        graph_encoder=args.graph_encoder,
        gat_heads=args.gat_heads,
        gat_attention_dropout=args.gat_attention_dropout,
        fusion_mode=args.fusion_mode,
        branch_attention_dim=args.branch_attention_dim,
    )
    model = CopperIntervalPredictor(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    selection_metric_name = f"val_{args.selection_metric}"
    run_config = {
        "arguments": serializable_args(args),
        "model_config": asdict(model.config),
        "device": str(device),
        "data_split": {
            "all": date_range_metadata(dates),
            "train": date_range_metadata(train_dates),
            "val": date_range_metadata(val_dates),
            "test": date_range_metadata(test_dates),
        },
        "selection_metric": selection_metric_name,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(args.output_dir / "training_config.json", run_config)
    history: list[dict[str, float | int | bool | str]] = []
    history_path = args.output_dir / "training_history.csv"
    best_selection_value = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    print(
        "Loss weights: "
        f"supervised={args.supervised_weight:g} "
        f"bound_penalty={args.bound_penalty_weight:g} "
        f"reconstruction={args.reconstruction_weight:g}"
    )
    print(f"Best-checkpoint metric: {selection_metric_name}")
    if args.early_stopping_patience > 0:
        print(
            "Early stopping: "
            f"patience={args.early_stopping_patience} "
            f"min_delta={args.early_stopping_min_delta:g}"
        )
    else:
        print("Early stopping: disabled")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            args.supervised_weight,
            args.bound_penalty_weight,
            args.reconstruction_weight,
            optimizer,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                args.supervised_weight,
                args.bound_penalty_weight,
                args.reconstruction_weight,
            )

        selection_value = val_metrics[args.selection_metric]
        if not np.isfinite(selection_value):
            raise RuntimeError(
                f"{selection_metric_name} is not finite at epoch {epoch}: "
                f"{selection_value}"
            )
        improved = selection_value < (
            best_selection_value - args.early_stopping_min_delta
        )
        history.append(
            history_row(
                epoch,
                train_metrics,
                val_metrics,
                selection_metric_name,
                selection_value,
                improved,
            )
        )
        pd.DataFrame(history).to_csv(history_path, index=False)

        save_checkpoint(
            args.output_dir / "last_model.pt",
            model,
            epoch,
            train_metrics,
            val_metrics,
            selection_metric_name,
            selection_value,
            run_config,
        )
        if improved:
            best_selection_value = selection_value
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                args.output_dir / "best_model.pt",
                model,
                epoch,
                train_metrics,
                val_metrics,
                selection_metric_name,
                selection_value,
                run_config,
            )
        else:
            epochs_without_improvement += 1

        print(f"epoch={epoch:03d} best={improved}")
        print(f"  train {format_metrics(train_metrics)}")
        print(f"  val   {format_metrics(val_metrics)}")
        print(
            f"  selection {selection_metric_name}={selection_value:.6f} "
            f"best_epoch={best_epoch:03d} best_value={best_selection_value:.6f}"
        )

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch}: "
                f"{selection_metric_name} did not improve by at least "
                f"{args.early_stopping_min_delta:g} for "
                f"{args.early_stopping_patience} epochs."
            )
            break

    if test_dates:
        checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        with torch.no_grad():
            test_metrics = run_epoch(
                model,
                test_loader,
                device,
                args.supervised_weight,
                args.bound_penalty_weight,
                args.reconstruction_weight,
            )
        print(f"best_epoch={int(checkpoint['epoch']):03d}")
        print(f"  test  {format_metrics(test_metrics)}")
        save_checkpoint(
            args.output_dir / "best_model.pt",
            model,
            int(checkpoint["epoch"]),
            checkpoint["train_metrics"],
            checkpoint["val_metrics"],
            str(checkpoint["selection_metric"]),
            float(checkpoint["selection_value"]),
            run_config,
            test_metrics=test_metrics,
        )

    print(f"Saved run config to {args.output_dir / 'training_config.json'}")
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
