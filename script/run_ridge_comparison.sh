#!/usr/bin/env bash
# =============================================================================
# Ridge vs SLC comparison experiment for Appendix B.
# Grid: AS17 + AS09 × AKT + DKT × 3 seeds (225, 226, 227)
# Total: 12 packs
#
# Usage (on the server with checkpoints):
#   cd final
#   bash script/run_ridge_comparison.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TS="${RIDGE_TS:-20260312_ridge}"
SAVE_TAG="${SAVE_TAG:-20260225_full_v1}"  # Override if checkpoints use a different tag
METHODS="global_sigmoid,item_bias_mean_monotone,item_bias_ridge,item_bias_shrinkage_static"
PACK_LEVEL="minimal"

DATASETS=(
  "assist2017_pid_uid_time_pos"
  "assist2009_pid_uid_time"
)
DS_SHORTS=(
  "as17"
  "as09"
)

MODELS=("akt_pid" "dkt")
MODEL_CKPT_DIRS=("akt_pid" "dkt")
MODEL_SHORTS=("akt" "dkt")
# Config tag fragments for checkpoint path resolution
MODEL_CFG_SHORTS=("akt" "dkt")

SEEDS=(225 226 227)

count=0
total=12

for di in "${!DATASETS[@]}"; do
  ds="${DATASETS[$di]}"
  ds_short="${DS_SHORTS[$di]}"

  for mi in "${!MODELS[@]}"; do
    model="${MODELS[$mi]}"
    ckpt_dir="${MODEL_CKPT_DIRS[$mi]}"
    model_short="${MODEL_SHORTS[$mi]}"
    cfg_short="${MODEL_CFG_SHORTS[$mi]}"

    for seed in "${SEEDS[@]}"; do
      count=$((count + 1))
      pack_name="ridge_${model_short}_${ds_short}_s${seed}"
      ckpt="_ckpts/${ckpt_dir}/${ds}_final_${cfg_short}_${ds_short}_base_${SAVE_TAG}_s${seed}/best.pt"

      if [[ ! -f "$ckpt" ]]; then
        echo "[${count}/${total}] SKIP (ckpt not found): $ckpt"
        continue
      fi

      # Skip if pack already exists
      pack_dir="_paper_packs/${TS}_${pack_name}"
      if [[ -d "$pack_dir" ]] && [[ -f "$pack_dir/meta.json" ]]; then
        echo "[${count}/${total}] SKIP (already exists): $pack_dir"
        continue
      fi

      echo "[${count}/${total}] Running: ${model_short} / ${ds_short} / s${seed}"
      python -u tools/eval_temporal_calibration.py \
        --dataset "$ds" \
        --train_set 1 \
        --data_root dataset \
        --ckpt "$ckpt" \
        --ts "$TS" \
        --pack_name "$pack_name" \
        --pack_level "$PACK_LEVEL" \
        --only_methods "$METHODS"
    done
  done
done

echo ""
echo "=== Ridge comparison done (${count}/${total} attempted) ==="
echo "Results in: _paper_packs/${TS}_ridge_*/"
echo ""
echo "To summarize:"
echo "  python -u tools/summarize_paper_packs.py --prefix ${TS}"
