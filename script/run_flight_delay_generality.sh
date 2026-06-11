#!/usr/bin/env bash
# =============================================================================
# Flight-delay generality validation (non-KT).
#
# Goal:
#   - compare a weaker backbone (`bbA`, no route features) and a stronger one (`bbB`)
#   - run the same score-only vs item-bias post-hoc pipeline used in KT
#
# Pack defaults:
#   raw_score, global_sigmoid, global_sigmoid_time,
#   item_bias_mean(_monotone), item_bias_shrinkage_static(_monotone),
#   item_bias_shrinkage_dynamic
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PACK_TS="${PACK_TS:-$(date +%Y%m%d_%H%M%S)}"
SEEDS="${SEEDS:-225,226,227}"

ZIP_PATH="${ZIP_PATH:-_raw_data/flight-delay-dataset-20182022.zip}"
RAW_DIR="${RAW_DIR:-dataset/flight_delay_raw}"
DATA_DIR="${DATA_DIR:-dataset/flight_delay}"
LOGITS_DIR="${LOGITS_DIR:-dataset/flight_delay/logits}"

CHUNKSIZE="${CHUNKSIZE:-200000}"
MAX_ROWS="${MAX_ROWS:-0}"
EPOCHS="${EPOCHS:-1}"
MIN_ROUTE_COUNT_PER_YEAR="${MIN_ROUTE_COUNT_PER_YEAR:-50}"

RESCAL_MIN_COUNT="${RESCAL_MIN_COUNT:-50}"
RESCAL_SHRINK_TAU="${RESCAL_SHRINK_TAU:-50}"
DRIFT_TIME_BINS="${DRIFT_TIME_BINS:-24}"
DRIFT_PROCESS_VAR="${DRIFT_PROCESS_VAR:-0.01}"
ITEM_BIAS_PRIOR_VAR="${ITEM_BIAS_PRIOR_VAR:-1.0}"
ITEM_BIAS_SHRINK_TAU="${ITEM_BIAS_SHRINK_TAU:-0.0}"
DRIFT_OBS_MIN_WEIGHT="${DRIFT_OBS_MIN_WEIGHT:-1.0}"
PACK_LEVEL="${PACK_LEVEL:-minimal}"

echo "=============================================="
echo "Flight-delay generality suite"
echo "pack_ts:   $PACK_TS"
echo "seeds:     $SEEDS"
echo "zip:       $ZIP_PATH"
echo "raw_dir:   $RAW_DIR"
echo "data_dir:  $DATA_DIR"
echo "logits:    $LOGITS_DIR"
echo "chunksize: $CHUNKSIZE max_rows=$MAX_ROWS epochs=$EPOCHS"
echo "min_route_count_per_year: $MIN_ROUTE_COUNT_PER_YEAR"
echo "pack_level: $PACK_LEVEL"
echo "=============================================="

python -u preprocess/preprocess_flight_delay.py \
  --zip_path "$ZIP_PATH" \
  --raw_dir "$RAW_DIR" \
  --out_dir "$DATA_DIR" \
  --years "2018,2019" \
  --min_route_count_per_year "$MIN_ROUTE_COUNT_PER_YEAR" \
  --chunksize "$CHUNKSIZE" \
  --max_rows "$MAX_ROWS" \
  --keep_extracted on

IFS=',' read -r -a seed_arr <<< "$SEEDS"

for seed in "${seed_arr[@]}"; do
  echo ""
  echo "[train+export] seed=$seed bbA (no-route)"
  python -u tools/train_flight_delay_backbone.py \
    --zip_path "$ZIP_PATH" \
    --raw_dir "$RAW_DIR" \
    --data_dir "$DATA_DIR" \
    --out_dir "$LOGITS_DIR" \
    --bb bbA \
    --seed "$seed" \
    --epochs "$EPOCHS" \
    --chunksize "$CHUNKSIZE" \
    --max_rows "$MAX_ROWS"

  echo ""
  echo "[train+export] seed=$seed bbB (with-route)"
  python -u tools/train_flight_delay_backbone.py \
    --zip_path "$ZIP_PATH" \
    --raw_dir "$RAW_DIR" \
    --data_dir "$DATA_DIR" \
    --out_dir "$LOGITS_DIR" \
    --bb bbB \
    --seed "$seed" \
    --epochs "$EPOCHS" \
    --chunksize "$CHUNKSIZE" \
    --max_rows "$MAX_ROWS"

  echo ""
  echo "[eval-pack] seed=$seed bbA"
  python -u tools/eval_pack_from_logits_npz.py \
    --calib_npz "$LOGITS_DIR/flight_bba_s${seed}_calib.npz" \
    --test_npz "$LOGITS_DIR/flight_bba_s${seed}_test.npz" \
    --dataset "flight_delay_route_month" \
    --model "flight_bbA" \
    --pack_name "flight_bbA_s${seed}" \
    --ts "$PACK_TS" \
    --pack_level "$PACK_LEVEL" \
    --rescal_min_count "$RESCAL_MIN_COUNT" \
    --rescal_shrink_tau "$RESCAL_SHRINK_TAU" \
    --drift_time_bins "$DRIFT_TIME_BINS" \
    --drift_process_var "$DRIFT_PROCESS_VAR" \
    --item_bias_prior_var "$ITEM_BIAS_PRIOR_VAR" \
    --item_bias_shrink_tau "$ITEM_BIAS_SHRINK_TAU" \
    --drift_obs_min_weight "$DRIFT_OBS_MIN_WEIGHT"

  echo ""
  echo "[eval-pack] seed=$seed bbB"
  python -u tools/eval_pack_from_logits_npz.py \
    --calib_npz "$LOGITS_DIR/flight_bbb_s${seed}_calib.npz" \
    --test_npz "$LOGITS_DIR/flight_bbb_s${seed}_test.npz" \
    --dataset "flight_delay_route_month" \
    --model "flight_bbB" \
    --pack_name "flight_bbB_s${seed}" \
    --ts "$PACK_TS" \
    --pack_level "$PACK_LEVEL" \
    --rescal_min_count "$RESCAL_MIN_COUNT" \
    --rescal_shrink_tau "$RESCAL_SHRINK_TAU" \
    --drift_time_bins "$DRIFT_TIME_BINS" \
    --drift_process_var "$DRIFT_PROCESS_VAR" \
    --item_bias_prior_var "$ITEM_BIAS_PRIOR_VAR" \
    --item_bias_shrink_tau "$ITEM_BIAS_SHRINK_TAU" \
    --drift_obs_min_weight "$DRIFT_OBS_MIN_WEIGHT"
done

echo ""
echo "[summary] prefix=$PACK_TS"
python -u tools/summarize_paper_packs.py --prefix "$PACK_TS"

echo ""
echo "=============================================="
echo "Done."
echo "Packs:   _paper_packs/${PACK_TS}_*"
echo "Summary: _paper_packs/${PACK_TS}_summary"
echo "=============================================="
