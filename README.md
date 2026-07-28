# Copper Price Prediction

This project currently contains a first implementation of the proposed copper
interval price prediction architecture:

```text
1D CNN branch for LME copper own features
    -> V1

Market GAE branch
    -> V2

Demand GAE branch
    -> V3

Supply GAE branch
    -> V4

[V1, V2, V3, V4]
    -> MLP
    -> [next-day low return, next-day high return]
```

## Files

| Path | Purpose |
|---|---|
| `src/copper_prediction/config.py` | Fixed graph node lists and node types. |
| `src/copper_prediction/features.py` | CNN matrix, GAE node features, and adjacency construction. |
| `src/copper_prediction/dataset.py` | Sample builder and torch-compatible dataset wrapper. |
| `src/copper_prediction/model.py` | 1D CNN, three GAE branches, MLP head, and losses. |
| `scripts/train.py` | Trains the model from CSV files under `data/raw/`. |
| `scripts/predict.py` | Loads a checkpoint and predicts the next-day copper interval. |
| `scripts/smoke_features.py` | Builds one synthetic sample without requiring torch. |
| `scripts/smoke_forward.py` | Runs a random model forward pass after torch is installed. |

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Smoke Tests

Feature construction does not require torch:

```bash
python3 scripts/smoke_features.py
```

Model forward pass requires torch:

```bash
python3 scripts/smoke_forward.py
```

## Expected Input Shapes

| Input | Shape |
|---|---|
| `cnn_x` | `[batch, 10, 20]` |
| `market_x` | `[batch, 20, 8]` |
| `market_adj` | `[batch, 20, 20]` |
| `demand_x` | `[batch, 18, 8]` |
| `demand_adj` | `[batch, 18, 18]` |
| `supply_x` | `[batch, 20, 8]` |
| `supply_adj` | `[batch, 20, 20]` |
| output | `[batch, 2]` |

## Raw Data Layout

Put raw CSV files under:

```text
data/raw/
```

Financial node CSV files should contain:

```text
date,open,high,low,close,volume
```

`volume` is optional for indexes, rates, FX, and volatility variables. LME Copper
should also include `open_interest` when available.

Inventory node CSV files should contain:

```text
date,inventory
```

By default, file names should match the node names in `config.py`, with `/`
replaced by `_` when needed. Examples:

```text
data/raw/LME Copper.csv
data/raw/Gold.csv
data/raw/NVIDIA.csv
data/raw/CNY_USD.csv
data/raw/CATL_Ningde Times.csv
data/raw/LME Copper Inventory.csv
```

If your files use different names, create a manifest CSV:

```csv
node,path
LME Copper,data/raw/my_lme_copper.csv
Gold,data/raw/xau.csv
```

Then pass it with `--manifest`.

## Train

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py \
  --raw-dir data/raw \
  --output-dir outputs/models \
  --epochs 50 \
  --batch-size 32 \
  --reconstruction-weight 0.0005 \
  --selection-metric mae \
  --early-stopping-patience 12
```

Outputs:

```text
outputs/models/best_model.pt
outputs/models/last_model.pt
outputs/models/training_config.json
outputs/models/training_history.csv
```

`best_model.pt` is selected by validation prediction MAE by default. The CSV
history records supervised MSE, bound penalty, each graph reconstruction loss,
weighted losses, interval MAE, lower/upper-bound MAE, and order violations for
both the training and validation splits.

The loss weights have distinct meanings:

```text
total_loss =
    supervised_weight * (mse_loss + bound_penalty_weight * bound_penalty_loss)
    + reconstruction_weight * reconstruction_loss
```

Defaults are `supervised_weight=1`, `bound_penalty_weight=1`, and
`reconstruction_weight=0.0005`. The legacy `--interval-weight` option remains
accepted as an alias for `--bound-penalty-weight`.

Useful debugging option:

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py --raw-dir data/raw --limit-samples 120 --epochs 2
```

## Predict

Use the latest date that can construct model inputs:

```bash
PYTHONPATH=src .venv/bin/python scripts/predict.py \
  --raw-dir data/raw \
  --checkpoint outputs/models/best_model.pt
```

Use a specific prediction date:

```bash
PYTHONPATH=src .venv/bin/python scripts/predict.py \
  --raw-dir data/raw \
  --checkpoint outputs/models/best_model.pt \
  --date 2024-12-31 \
  --output outputs/predictions/prediction.csv
```

The prediction output includes relative returns and restored prices:

```text
pred_low_return
pred_high_return
pred_low_price
pred_high_price
```

## Evaluate

Evaluate a checkpoint on the chronological test split and compare it with the
previous realized interval and a 20-day rolling-mean baseline:

```bash
python scripts/evaluate.py \
  --raw-dir data/raw \
  --checkpoint outputs/models/gcn_concat_20230302_20251117/best_model.pt \
  --start-date 20230302 \
  --end-date 20251117 \
  --split test \
  --device auto
```

The evaluator reports return MAE/RMSE, price MAE, invalid interval rate,
full-interval coverage, and improvement over the rolling baseline. Detailed
outputs are saved beside the checkpoint under `evaluation/`:

```text
test_summary.csv
test_predictions.csv
```
