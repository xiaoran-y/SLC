#!/usr/bin/env bash
# Drift + headroom audit (paper pre-check).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Some platforms set invalid thread env vars (e.g., OMP_NUM_THREADS=auto),
# which causes noisy libgomp warnings. Sanitize to a valid integer.
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
OUT_ROOT="${OUT_ROOT:-_runs}"
DATA_ROOT="${DATA_ROOT:-dataset}"

OUT_PATH="${OUT_ROOT}/drift_headroom_${TS}.txt"
mkdir -p "$OUT_ROOT"

{
  echo "# Drift + Headroom audit (${TS})"
  echo "# data_root: ${DATA_ROOT}"

  echo ""
  echo "===================="
  echo "DATASET: assist2017_pid_uid_time_pos (group=pid, min_count=50)"
  echo "===================="
  python -u tools/analyze_pid_drift.py --dataset assist2017_pid_uid_time_pos --data_root "$DATA_ROOT" --group pid --min_count 50 --split_a train --split_b test
  echo ""
  python -u tools/analyze_pid_drift_headroom.py --dataset assist2017_pid_uid_time_pos --data_root "$DATA_ROOT" --group pid --min_count 50 --split_a train --split_b test --quantile_bins 5

  echo ""
  echo "===================="
  echo "DATASET: eedi_task12_pid_uid_time (group=pid, min_count=50)"
  echo "===================="
  python -u tools/analyze_pid_drift.py --dataset eedi_task12_pid_uid_time --data_root "$DATA_ROOT" --group pid --min_count 50 --split_a train --split_b test
  echo ""
  python -u tools/analyze_pid_drift_headroom.py --dataset eedi_task12_pid_uid_time --data_root "$DATA_ROOT" --group pid --min_count 50 --split_a train --split_b test --quantile_bins 5

  echo ""
  echo "===================="
  echo "DATASET: algebra_merged_pid_uid_time (group=pid, min_count=20)"
  echo "===================="
  python -u tools/analyze_pid_drift.py --dataset algebra_merged_pid_uid_time --data_root "$DATA_ROOT" --group pid --min_count 20 --split_a train --split_b test
  echo ""
  python -u tools/analyze_pid_drift_headroom.py --dataset algebra_merged_pid_uid_time --data_root "$DATA_ROOT" --group pid --min_count 20 --split_a train --split_b test --quantile_bins 5

  echo ""
  echo "===================="
  echo "DATASET: assist2009_pid_uid_time (group=q, min_count=50)"
  echo "===================="
  echo "# Note: pid-level is too sparse for AS09 under time split; use q-level as a control/proxy."
  python -u tools/analyze_pid_drift.py --dataset assist2009_pid_uid_time --data_root "$DATA_ROOT" --group q --min_count 50 --split_a train --split_b test
  echo ""
  python -u tools/analyze_pid_drift_headroom.py --dataset assist2009_pid_uid_time --data_root "$DATA_ROOT" --group q --min_count 50 --split_a train --split_b test --quantile_bins 5
} | tee "$OUT_PATH"

echo ""
echo "[ok] wrote: $OUT_PATH"
