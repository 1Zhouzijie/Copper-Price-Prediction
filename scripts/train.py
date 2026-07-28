"""Train the CNN + three-GAE copper interval predictor."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
    total_training_loss,
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
    parser.add_argument("--reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--interval-weight", type=float, default=1.0)
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


def forward_loss(
    model: CopperIntervalPredictor,
    batch: dict[str, Any],
    interval_weight: float,
    reconstruction_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    loss = total_training_loss(
        prediction=prediction,
        target=batch["target"],
        aux=aux,
        market_adj=batch["market_adj"],
        demand_adj=batch["demand_adj"],
        supply_adj=batch["supply_adj"],
        interval_weight=interval_weight,
        reconstruction_weight=reconstruction_weight,
    )
    return loss, prediction


def run_epoch(
    model: CopperIntervalPredictor,
    loader: DataLoader,
    device: torch.device,
    interval_weight: float,
    reconstruction_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_mae = 0.0
    total_count = 0

    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        with torch.set_grad_enabled(training):
            loss, prediction = forward_loss(model, batch, interval_weight, reconstruction_weight)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = int(batch["target"].shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_mae += float(torch.mean(torch.abs(prediction.detach() - batch["target"])).cpu()) * batch_size
        total_count += batch_size

    if total_count == 0:
        return float("nan"), float("nan")
    return total_loss / total_count, total_mae / total_count


def save_checkpoint(
    path: Path,
    model: CopperIntervalPredictor,
    epoch: int,
    train_loss: float,
    val_loss: float,
    test_loss: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(model.config),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
        },
        path,
    )


def main() -> None:
    args = parse_args()
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
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae = run_epoch(
            model,
            train_loader,
            device,
            args.interval_weight,
            args.reconstruction_weight,
            optimizer,
        )
        with torch.no_grad():
            val_loss, val_mae = run_epoch(
                model,
                val_loader,
                device,
                args.interval_weight,
                args.reconstruction_weight,
            )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} train_mae={train_mae:.6f} "
            f"val_loss={val_loss:.6f} val_mae={val_mae:.6f}"
        )

        save_checkpoint(args.output_dir / "last_model.pt", model, epoch, train_loss, val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(args.output_dir / "best_model.pt", model, epoch, train_loss, val_loss)

    if test_dates:
        checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        with torch.no_grad():
            test_loss, test_mae = run_epoch(
                model,
                test_loader,
                device,
                args.interval_weight,
                args.reconstruction_weight,
            )
        print(f"test_loss={test_loss:.6f} test_mae={test_mae:.6f}")
        save_checkpoint(args.output_dir / "best_model.pt", model, int(checkpoint["epoch"]), checkpoint["train_loss"], checkpoint["val_loss"], test_loss)


if __name__ == "__main__":
    main()
