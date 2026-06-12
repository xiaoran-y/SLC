# SLC: State-space Logit Correction

Official code repository for the paper:
**"Recovering Stranded Discrimination in Knowledge Tracing: Per-Item Bias Correction via Empirical-Bayes Shrinkage"**

Xiaoran Yan, Cheng Tang, Atsushi Shimada (Kyushu University)

Accepted at **ECML PKDD 2026** (Research Track).

📄 **Supplementary material**: [supplementary.pdf](./supplementary.pdf) — per-backbone results, classic calibrator comparison, prior-variance sensitivity, ridge equivalence, link ablation, calibration-fraction sweep, and cross-domain experiments.

## Overview

SLC is a lightweight post-hoc correction method that recovers per-item AUC headroom stranded by global calibrators. It models per-item logit bias as a Gaussian random effect and applies empirical-Bayes shrinkage via a Kalman smoother.

## Repository Structure

```
├── main.py / cli.py / experiment.py   # Training pipeline
├── run.py / load_data.py              # Train/test loops and data loading
├── train.py                           # Config-driven multi-seed training
├── ckpt.py / utils.py                 # Checkpoint and model utilities
├── model/                             # KT backbones (AKT, DKT, SAKT, DKVMN, LPKT, MF, NCF)
├── posthoc/                           # SLC core: per-item EB shrinkage (item_drift.py)
├── configs/                           # JSON experiment configs (composable)
├── tools/
│   ├── eval_temporal_calibration.py   # Main evaluation: all baselines + SLC + paper packs
│   ├── eval_pack_from_logits_npz.py   # Non-KT evaluation (flight-delay)
│   ├── exp_synthetic.py               # Synthetic regime-map experiments
│   ├── summarize_paper_packs.py       # Aggregate results across packs
│   └── train_flight_delay_backbone.py # Flight-delay backbone training
├── script/                            # Shell scripts for paper experiments
├── preprocess/                        # Data preprocessing (flight-delay)
└── dataset/                           # Data directory (see dataset/README.md)
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- scikit-learn
- numpy
- pandas (flight-delay preprocessing)
- scipy (drift analysis tools)

## Quick Start

### 1. Train a backbone

```bash
python -u train.py --exp akt_as17_base --seeds 225,226,227
```

Checkpoints are saved to `_ckpts/<model>/<dataset>_<save_tag>/best.pt`.

### 2. Run SLC evaluation

```bash
CKPT=_ckpts/akt_pid/<dataset>_<save_tag>/best.pt \
DATASET=assist2017_pid_uid_time_pos \
TRAIN_SET=1 \
bash script/run_eval_pack.sh
```

Results are written to `_paper_packs/<timestamp>_<pack_name>/`.

### 3. Full paper reproduction

```bash
# Train all backbones (5 models × 4 datasets × 3 seeds = 60 runs)
# Each run creates _runs/train_<exp>_<timestamp>/ with save_tags.json
# and checkpoints under _ckpts/<model>/<dataset>_<save_tag>/best.pt
for exp in configs/experiments/*_base.json; do
    name=$(basename "$exp" .json)
    python -u train.py --exp "$name" --seeds 225,226,227
done

# Generate evaluation packs (point CKPT to each trained checkpoint)
# See script/run_eval_pack.sh for single-pack usage
# See script/run_*.sh for batch scripts (set SAVE_TAG to match your training run)
```

## Datasets

See `dataset/README.md` for download links and preprocessing instructions.

## Key Files

- **SLC algorithm**: `posthoc/item_drift.py` (269 lines)
- **All baselines**: `tools/eval_temporal_calibration.py` (Base, Platt, Temp, Iso, Hist, ResCal, ResCal+Iso, Ridge, SLC, SLC-T, SLC+Iso)
- **Synthetic experiments**: `tools/exp_synthetic.py`
