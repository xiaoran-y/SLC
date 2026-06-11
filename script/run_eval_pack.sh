#!/usr/bin/env bash
# =============================================================================
# Paper-ready eval pack from an existing backbone checkpoint.
#
# Usage:
#   mamba activate akt
#   CKPT=_ckpts/akt_pid/<dataset>_<save_tag>/best.pt \
#   DATASET=assist2017_pid_uid_time_pos TRAIN_SET=1 \
#   bash script/run_eval_pack.sh
#
# Optional:
#   DRIFT_ENV=path/to/drift_env.sh
#   ONLY_METHODS="raw_score,global_sigmoid,item_bias_shrinkage_static"
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

fix_thread_env() {
  local k v head
  for k in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS; do
    v="${!k:-}"
    if [[ -z "$v" ]]; then
      continue
    fi
    head="${v%%,*}"
    if [[ ! "$head" =~ ^[0-9]+$ ]] || [[ "$head" -le 0 ]]; then
      export "$k"="1"
    else
      export "$k"="$head"
    fi
  done
}
fix_thread_env

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
DATASET="${DATASET:-assist2017_pid_uid_time_pos}"
TRAIN_SET="${TRAIN_SET:-1}"
DATA_ROOT="${DATA_ROOT:-dataset}"
CKPT="${CKPT:-}"
PACK_NAME="${PACK_NAME:-${DATASET}_eval_pack}"
OUTLINE="${OUTLINE:-paper/0_outline/ecml_pkdd_outline.md}"
ONLY_METHODS="${ONLY_METHODS:-}"
PACK_LEVEL="${PACK_LEVEL:-minimal}"

CALIB_FRAC="${CALIB_FRAC:-1.0}"
ECE_BINS="${ECE_BINS:-15}"
TIME_SLICES="${TIME_SLICES:-10}"
LATE_Q="${LATE_Q:-0.8}"

RESCAL_MIN_COUNT="${RESCAL_MIN_COUNT:-50}"
RESCAL_SHRINK_TAU="${RESCAL_SHRINK_TAU:-50}"

DRIFT_ENV="${DRIFT_ENV:-}"
DRIFT_TIME_BINS="${DRIFT_TIME_BINS:-10}"
DRIFT_PROCESS_VAR="${DRIFT_PROCESS_VAR:-0.01}"
ITEM_BIAS_PRIOR_VAR="${ITEM_BIAS_PRIOR_VAR:-1.0}"
ITEM_BIAS_SHRINK_TAU="${ITEM_BIAS_SHRINK_TAU:-0.0}"
DRIFT_OBS_MIN_WEIGHT="${DRIFT_OBS_MIN_WEIGHT:-1.0}"

if [[ -z "$CKPT" ]]; then
  echo "[error] CKPT is required." >&2
  exit 2
fi
if [[ "$CKPT" != /* ]]; then
  CKPT="$REPO_ROOT/$CKPT"
fi
if [[ ! -f "$CKPT" ]]; then
  echo "[error] CKPT not found: $CKPT" >&2
  exit 2
fi

echo "=============================================="
echo "Eval pack"
echo "dataset:   $DATASET (train_set=$TRAIN_SET)"
echo "ckpt:      $CKPT"
echo "pack_name: $PACK_NAME"
echo "pack:      level=$PACK_LEVEL"
echo "calib:     frac=$CALIB_FRAC"
echo "rescal:    min_count=$RESCAL_MIN_COUNT shrink_tau=$RESCAL_SHRINK_TAU"
echo "drift:     env=${DRIFT_ENV:-<none>} bins=$DRIFT_TIME_BINS q=$DRIFT_PROCESS_VAR prior=$ITEM_BIAS_PRIOR_VAR shrink_tau=$ITEM_BIAS_SHRINK_TAU obs_min_w=$DRIFT_OBS_MIN_WEIGHT"
echo "methods:   ${ONLY_METHODS:-<all>}"
echo "=============================================="

extra_env=()
if [[ -n "$DRIFT_ENV" ]]; then
  extra_env+=(--drift_env "$DRIFT_ENV")
fi

python -u tools/eval_temporal_calibration.py \
  --dataset "$DATASET" \
  --train_set "$TRAIN_SET" \
  --data_root "$DATA_ROOT" \
  --ckpt "$CKPT" \
  --pack_name "$PACK_NAME" \
  --ts "$TS" \
  --outline "$OUTLINE" \
  --pack_level "$PACK_LEVEL" \
  --only_methods "$ONLY_METHODS" \
  --ece_bins "$ECE_BINS" \
  --time_slices "$TIME_SLICES" \
  --late_q "$LATE_Q" \
  --calib_frac "$CALIB_FRAC" \
  --rescal_min_count "$RESCAL_MIN_COUNT" \
  --rescal_shrink_tau "$RESCAL_SHRINK_TAU" \
  "${extra_env[@]}" \
  --drift_time_bins "$DRIFT_TIME_BINS" \
  --drift_process_var "$DRIFT_PROCESS_VAR" \
  --item_bias_prior_var "$ITEM_BIAS_PRIOR_VAR" \
  --item_bias_shrink_tau "$ITEM_BIAS_SHRINK_TAU" \
  --drift_obs_min_weight "$DRIFT_OBS_MIN_WEIGHT"
