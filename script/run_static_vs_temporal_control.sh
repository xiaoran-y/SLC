#!/usr/bin/env bash
# =============================================================================
# Static-vs-dynamic item-bias control on existing checkpoints.
#
# Example:
#   SUITE_TS=20260225_full_v1 \
#   SUITE_DIR=_runs/paper_suite_20260225_full_v1 \
#   PACK_TS=20260308_static_vs_temporal \
#   DATASETS="assist2017_pid_uid_time_pos assist2009_pid_uid_time" \
#   MODELS="akt dkt sakt dkvmn lpkt" \
#   SEEDS="225 226 227" \
#   bash script/run_static_vs_temporal_control.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SUITE_TS="${SUITE_TS:-}"
if [[ -z "$SUITE_TS" ]]; then
  echo "[error] SUITE_TS is required." >&2
  exit 2
fi

PACK_TS="${PACK_TS:-${SUITE_TS}_static_vs_temporal}"
SUITE_DIR="${SUITE_DIR:-_runs/paper_suite_${SUITE_TS}}"
OUT_DIR="${OUT_DIR:-_runs/static_vs_temporal_${PACK_TS}}"
mkdir -p "$OUT_DIR/_logs"

TRAIN_SET="${TRAIN_SET:-1}"
DATA_ROOT="${DATA_ROOT:-dataset}"
CKPT_ROOT="${CKPT_ROOT:-_ckpts}"
DATASETS="${DATASETS:-assist2017_pid_uid_time_pos assist2009_pid_uid_time algebra_merged_pid_uid_time eedi_task12_pid_uid_time}"
MODELS="${MODELS:-akt dkt sakt dkvmn lpkt}"
SEEDS="${SEEDS:-225 226 227}"

RESCAL_MIN_COUNT="${RESCAL_MIN_COUNT:-50}"
RESCAL_SHRINK_TAU="${RESCAL_SHRINK_TAU:-50}"
DRIFT_ENV="${DRIFT_ENV:-}"
DRIFT_TIME_BINS="${DRIFT_TIME_BINS:-10}"
DRIFT_PROCESS_VAR="${DRIFT_PROCESS_VAR:-0.01}"
ITEM_BIAS_PRIOR_VAR="${ITEM_BIAS_PRIOR_VAR:-1.0}"
ITEM_BIAS_SHRINK_TAU="${ITEM_BIAS_SHRINK_TAU:-0.0}"
DRIFT_OBS_MIN_WEIGHT="${DRIFT_OBS_MIN_WEIGHT:-1.0}"
SKIP_EXISTING_PACKS="${SKIP_EXISTING_PACKS:-on}"
ONLY_METHODS="${ONLY_METHODS:-raw_score,global_sigmoid,item_bias_shrinkage_static,item_bias_shrinkage_dynamic}"

ds_alias() {
  case "$1" in
    assist2017_pid_uid_time_pos) echo "as17" ;;
    assist2009_pid_uid_time) echo "as09" ;;
    algebra_merged_pid_uid_time) echo "algebra" ;;
    eedi_task12_pid_uid_time) echo "eedi" ;;
    *) echo "$1" ;;
  esac
}

model_id() {
  case "$1" in
    akt) echo "akt_pid" ;;
    dkt|sakt|dkvmn|lpkt) echo "$1" ;;
    *) echo "$1" ;;
  esac
}

read_save_tag() {
  local train_dir="$1"
  local seed="$2"
  local json_path="$train_dir/save_tags.json"
  if [[ ! -f "$json_path" ]]; then
    echo "[error] save_tags.json not found: $json_path" >&2
    exit 2
  fi
  python - "$json_path" "$seed" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
seed = str(int(sys.argv[2]))
d = json.loads(p.read_text(encoding="utf-8"))
if seed not in d:
    raise SystemExit(f"seed not found in save_tags.json: {seed}")
print(d[seed])
PY
}

echo "=============================================="
echo "[static_vs_temporal_control]"
echo "suite_ts: $SUITE_TS"
echo "suite_dir:$SUITE_DIR"
echo "pack_ts:  $PACK_TS"
echo "datasets: $DATASETS"
echo "models:   $MODELS"
echo "seeds:    $SEEDS"
echo "methods:  $ONLY_METHODS"
echo "=============================================="

for ds in $DATASETS; do
  a="$(ds_alias "$ds")"
  for mp in $MODELS; do
    mid="$(model_id "$mp")"
    exp="${mp}_${a}_base"
    train_dir="$SUITE_DIR/train_${exp}_${SUITE_TS}"
    for seed in $SEEDS; do
      save_tag="$(read_save_tag "$train_dir" "$seed")"
      ckpt="$CKPT_ROOT/$mid/${ds}_${save_tag}/best.pt"
      if [[ "$ckpt" != /* ]]; then
        ckpt="$REPO_ROOT/$ckpt"
      fi
      if [[ ! -f "$ckpt" ]]; then
        echo "[error] ckpt not found: $ckpt" >&2
        exit 2
      fi

      pack_name="static_dynamic_${mid}_${a}_s${seed}"
      pack_dir="$REPO_ROOT/_paper_packs/${PACK_TS}_${pack_name}"
      log="$OUT_DIR/_logs/${ds}.${mid}.s${seed}.log"

      echo ""
      echo "[eval_pack] ds=$ds model=$mid seed=$seed out=$pack_dir"
      if [[ "$SKIP_EXISTING_PACKS" == "on" ]] && [[ -f "$pack_dir/tables/metrics_overall.csv" ]] && [[ -f "$pack_dir/tables/metrics_time_slices.csv" ]]; then
        echo "[skip] eval_pack already complete: $pack_dir"
        continue
      fi

      TS="$PACK_TS" \
      PACK_NAME="$pack_name" \
      CKPT="$ckpt" \
      DATASET="$ds" \
      TRAIN_SET="$TRAIN_SET" \
      DATA_ROOT="$DATA_ROOT" \
      ONLY_METHODS="$ONLY_METHODS" \
      RESCAL_MIN_COUNT="$RESCAL_MIN_COUNT" \
      RESCAL_SHRINK_TAU="$RESCAL_SHRINK_TAU" \
      DRIFT_ENV="$DRIFT_ENV" \
      DRIFT_TIME_BINS="$DRIFT_TIME_BINS" \
      DRIFT_PROCESS_VAR="$DRIFT_PROCESS_VAR" \
      ITEM_BIAS_PRIOR_VAR="$ITEM_BIAS_PRIOR_VAR" \
      ITEM_BIAS_SHRINK_TAU="$ITEM_BIAS_SHRINK_TAU" \
      DRIFT_OBS_MIN_WEIGHT="$DRIFT_OBS_MIN_WEIGHT" \
      bash script/run_eval_pack.sh 2>&1 | tee "$log"
    done
  done
done

python -u tools/summarize_paper_packs.py --prefix "$PACK_TS"

echo ""
echo "Packs:   _paper_packs/${PACK_TS}_*"
echo "Summary: _paper_packs/${PACK_TS}_summary"
