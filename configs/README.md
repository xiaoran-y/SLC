# `configs/` — Experiment Configurations

Composable JSON configs for reproducible training and evaluation, loaded by `train.py`.

## Directory Structure

- `roots/` — workspace paths (`data_root`, `ckpt_root`, `result_root`)
- `datasets/` — dataset selection (`dataset`, `train_set`)
- `models/` — backbone architecture and regularization (AKT, DKT, etc.)
- `runtimes/` — acceleration switches (AMP, torch.compile, etc.)
- `training/` — training hyperparameters (max_iter, batch_size, lr, early_stop, etc.)
- `experiments/` — experiment entries that compose the above via `include`

## Format

Each file is a flat JSON dict. Reserved fields:
- `include`: list of other configs to merge (relative to `configs/`)
- Keys starting with `_` are treated as comments and ignored

All other keys map to `main.py` CLI arguments (`--<key> <value>`).

## Usage

```bash
python -u train.py --exp akt_as17_base --seeds 225,226,227
```
