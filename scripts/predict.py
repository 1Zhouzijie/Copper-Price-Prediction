"""Predict the next-day LME copper interval with a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from copper_prediction.config import default_graph_specs  # noqa: E402
from copper_prediction.dataset import CopperGraphSampleBuilder, CopperModelInputs  # noqa: E402
from copper_prediction.io import load_asset_frames  # noqa: E402
from copper_prediction.model import CopperIntervalPredictor, CopperIntervalPredictorConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "models" / "best_model.pt")
    parser.add_argument("--date", default="latest", help="Prediction date t, e.g. 2024-12-31, or 'latest'.")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV path for the prediction result.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def close_at(copper_frame: pd.DataFrame, date: pd.Timestamp) -> float:
    lower_to_original = {str(col).lower(): col for col in copper_frame.columns}
    close_col = lower_to_original.get("close") or lower_to_original.get("settlement")
    if close_col is None:
        raise KeyError("LME Copper data must contain a close or settlement column")
    return float(copper_frame.loc[date, close_col])


def latest_valid_input_date(builder: CopperGraphSampleBuilder, copper_frame: pd.DataFrame) -> pd.Timestamp:
    for date in reversed(list(pd.to_datetime(copper_frame.index).sort_values())):
        try:
            builder.build_inputs(date)
        except Exception:
            continue
        return pd.Timestamp(date)
    raise ValueError("could not find any date that can build model inputs")


def tensorize(inputs: CopperModelInputs, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "cnn_x": torch.from_numpy(inputs.cnn_x).float().unsqueeze(0).to(device),
        "market_x": torch.from_numpy(inputs.market_x).float().unsqueeze(0).to(device),
        "market_adj": torch.from_numpy(inputs.market_adj).float().unsqueeze(0).to(device),
        "demand_x": torch.from_numpy(inputs.demand_x).float().unsqueeze(0).to(device),
        "demand_adj": torch.from_numpy(inputs.demand_adj).float().unsqueeze(0).to(device),
        "supply_x": torch.from_numpy(inputs.supply_x).float().unsqueeze(0).to(device),
        "supply_adj": torch.from_numpy(inputs.supply_adj).float().unsqueeze(0).to(device),
    }


def model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> CopperIntervalPredictor:
    checkpoint: Any = torch.load(checkpoint_path, map_location=device)
    config_data = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    if config_data:
        for tuple_key in ("cnn_channels", "mlp_hidden_dims"):
            if tuple_key in config_data:
                config_data[tuple_key] = tuple(config_data[tuple_key])
        config = CopperIntervalPredictorConfig(**config_data)
    else:
        config = CopperIntervalPredictorConfig()

    model = CopperIntervalPredictor(config).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    graph_specs = default_graph_specs()

    asset_frames = load_asset_frames(args.raw_dir, graph_specs, args.manifest)
    copper_frame = asset_frames["LME Copper"]
    builder = CopperGraphSampleBuilder(copper_frame, asset_frames, graph_specs)

    if args.date == "latest":
        date = latest_valid_input_date(builder, copper_frame)
    else:
        date = pd.Timestamp(args.date)

    inputs = builder.build_inputs(date)
    close_t = close_at(copper_frame, date)
    model = model_from_checkpoint(args.checkpoint, device)

    with torch.no_grad():
        prediction, aux = model(**tensorize(inputs, device), return_aux=True)

    low_return = float(prediction[0, 0].cpu())
    high_return = float(prediction[0, 1].cpu())
    pred_low = close_t * (1.0 + low_return)
    pred_high = close_t * (1.0 + high_return)

    if pred_low > pred_high:
        print("Warning: predicted low is above predicted high. The raw model outputs are shown unchanged.")

    result_row = {
        "date": date,
        "close": close_t,
        "pred_low_return": low_return,
        "pred_high_return": high_return,
        "pred_low_price": pred_low,
        "pred_high_price": pred_high,
    }
    if "branch_attention" in aux:
        attention = aux["branch_attention"][0].detach().cpu()
        result_row.update(
            {
                "market_attention": float(attention[0]),
                "demand_attention": float(attention[1]),
                "supply_attention": float(attention[2]),
            }
        )

    result = pd.DataFrame([result_row])

    print(result.to_string(index=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"Saved prediction to {args.output}")


if __name__ == "__main__":
    main()
