#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_meta_n_units(data_dir: Path, *, group: str) -> int:
    meta_path = data_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found under {data_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if group == "pid":
        key = "n_pid"
    elif group == "q":
        key = "n_question"
    else:
        raise ValueError(f"Unknown group: {group!r}")
    n = int(meta.get(key, 0))
    if n <= 0:
        raise ValueError(f"Invalid {key} in {meta_path}: {n}")
    return n


def _load_split_counts(path: Path, *, n_units: int, group: str) -> tuple[np.ndarray, np.ndarray]:
    n = np.zeros(n_units + 1, dtype=np.int64)
    x = np.zeros(n_units + 1, dtype=np.float64)
    with path.open("r", encoding="utf-8") as f:
        while True:
            header = f.readline()
            if not header:
                break
            pid_line = f.readline()
            q_line = f.readline()
            a_line = f.readline()
            if not (pid_line and a_line):
                raise ValueError(f"Unexpected EOF in {path} after header: {header.strip()!r}")
            unit_line = pid_line if group == "pid" else q_line
            unit_arr = np.fromstring(unit_line.strip(), dtype=np.int64, sep=",")
            a_arr = np.fromstring(a_line.strip(), dtype=np.int64, sep=",")
            if unit_arr.size == 0:
                continue
            if a_arr.size != unit_arr.size:
                raise ValueError(f"{group}/a length mismatch in {path}: {group}={unit_arr.size} a={a_arr.size}")
            n += np.bincount(unit_arr, minlength=n_units + 1).astype(np.int64, copy=False)
            x += np.bincount(
                unit_arr, weights=a_arr.astype(np.float64), minlength=n_units + 1
            ).astype(np.float64, copy=False)
    return n, x


def _sigmoid_clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p, 1e-6, 1.0 - 1e-6)


def _bce(pred: np.ndarray, target: np.ndarray) -> float:
    p = _sigmoid_clip(pred.astype(np.float64, copy=False))
    y = target.astype(np.float64, copy=False)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def _auc_fast(pred: np.ndarray, target: np.ndarray) -> float:
    # AUC via rank statistics (no sklearn dependency). Handles ties.
    y = target.astype(np.int64, copy=False)
    p = pred.astype(np.float64, copy=False)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    y_sorted = y[order]

    # Rank with ties: average rank for ties in p.
    n = y_sorted.size
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    r = 1.0
    while i < n:
        j = i + 1
        while j < n and p_sorted[j] == p_sorted[i]:
            j += 1
        avg_rank = 0.5 * (r + (r + (j - i) - 1.0))
        ranks[i:j] = avg_rank
        r += float(j - i)
        i = j

    sum_pos_ranks = float(ranks[y_sorted == 1].sum())
    auc = (sum_pos_ranks - (pos * (pos + 1) / 2.0)) / float(pos * neg)
    return float(auc)


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantify baseline headroom vs per-unit P(y) drift (time split).")
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument(
        "--data_root",
        type=str,
        default="dataset",
        help="Dataset root dir (default: dataset).",
    )
    ap.add_argument(
        "--group",
        type=str,
        default="pid",
        choices=["pid", "q"],
        help="Drift unit: pid (problem id) or q (concept/skill id).",
    )
    ap.add_argument("--train_set", type=int, default=1)
    ap.add_argument("--min_count", type=int, default=20, help="Min unit count in BOTH splits for drift + eval.")
    ap.add_argument("--split_a", type=str, default="train", choices=["train", "valid", "test"])
    ap.add_argument("--split_b", type=str, default="test", choices=["train", "valid", "test"])
    ap.add_argument("--quantile_bins", type=int, default=5, help="Number of |Δp| quantile bins over units.")
    ap.add_argument("--max_rows", type=int, default=0, help="Optional: cap number of eval interactions (0=all).")
    args = ap.parse_args()

    dataset = str(args.dataset)
    train_set = int(args.train_set)
    group = str(args.group)
    data_root = Path(str(args.data_root))
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    data_dir = data_root / dataset
    n_units = _read_meta_n_units(data_dir, group=group)

    split_a = str(args.split_a)
    split_b = str(args.split_b)
    path_a = data_dir / f"{dataset}_{split_a}{train_set}.csv"
    path_b = data_dir / f"{dataset}_{split_b}{train_set}.csv"
    if not path_a.is_file():
        raise FileNotFoundError(f"Split file not found: {path_a}")
    if not path_b.is_file():
        raise FileNotFoundError(f"Split file not found: {path_b}")

    n_a, x_a = _load_split_counts(path_a, n_units=n_units, group=group)
    n_b, x_b = _load_split_counts(path_b, n_units=n_units, group=group)

    min_count = int(args.min_count)
    valid_unit = (n_a >= min_count) & (n_b >= min_count)
    valid_unit[0] = False  # ids start at 1
    unit_ids = np.flatnonzero(valid_unit).astype(np.int64)
    if unit_ids.size == 0:
        raise SystemExit("[drift_headroom] No units satisfy min_count; lower --min_count.")

    pa = (x_a[unit_ids] / n_a[unit_ids].astype(np.float64)).astype(np.float64)
    pb = (x_b[unit_ids] / n_b[unit_ids].astype(np.float64)).astype(np.float64)
    abs_delta = np.abs(pb - pa)

    q = int(args.quantile_bins)
    if q < 2:
        raise ValueError("--quantile_bins must be >=2")
    edges = np.quantile(abs_delta, q=np.linspace(0.0, 1.0, q + 1), method="linear")
    # Ensure monotonicity and include max.
    edges[0] = 0.0
    edges[-1] = float(max(edges[-1], abs_delta.max()))

    # Map unit -> bin id.
    unit_to_bin = np.full(n_units + 1, -1, dtype=np.int64)
    bin_id = np.digitize(abs_delta, edges, right=True) - 1
    bin_id = np.clip(bin_id, 0, q - 1)
    unit_to_bin[unit_ids] = bin_id

    # Iterate eval split_b and collect predictions.
    max_rows = int(args.max_rows)
    y_all: list[int] = []
    p_train_all: list[float] = []
    p_oracle_all: list[float] = []
    b_all: list[int] = []

    # Precompute lookup arrays for pa/pb on unit index.
    p_train = np.zeros(n_units + 1, dtype=np.float64)
    p_test = np.zeros(n_units + 1, dtype=np.float64)
    p_train[unit_ids] = pa
    p_test[unit_ids] = pb

    seen = 0
    with path_b.open("r", encoding="utf-8") as f:
        while True:
            header = f.readline()
            if not header:
                break
            pid_line = f.readline()
            q_line = f.readline()
            a_line = f.readline()
            if not (pid_line and a_line):
                raise ValueError(f"Unexpected EOF in {path_b} after header: {header.strip()!r}")
            unit_line = pid_line if group == "pid" else q_line
            unit_arr = np.fromstring(unit_line.strip(), dtype=np.int64, sep=",")
            a_arr = np.fromstring(a_line.strip(), dtype=np.int64, sep=",")
            if unit_arr.size == 0:
                continue
            if a_arr.size != unit_arr.size:
                raise ValueError(f"{group}/a length mismatch in {path_b}: {group}={unit_arr.size} a={a_arr.size}")

            bins = unit_to_bin[unit_arr]
            m = bins >= 0
            if not np.any(m):
                continue
            unit_m = unit_arr[m]
            a_m = a_arr[m]
            bins_m = bins[m]

            y_all.extend(a_m.astype(int).tolist())
            p_train_all.extend(p_train[unit_m].astype(float).tolist())
            p_oracle_all.extend(p_test[unit_m].astype(float).tolist())
            b_all.extend(bins_m.astype(int).tolist())

            seen += int(m.sum())
            if max_rows > 0 and seen >= max_rows:
                break

    y = np.asarray(y_all, dtype=np.int64)
    p_tr = np.asarray(p_train_all, dtype=np.float64)
    p_or = np.asarray(p_oracle_all, dtype=np.float64)
    b = np.asarray(b_all, dtype=np.int64)

    print(
        f"[drift_headroom] dataset={dataset} split_b={split_b} evaluated_interactions={y.size} "
        f"group={group} units={unit_ids.size} min_count={min_count} bins={q}"
    )
    print("[drift_headroom] |Δp| quantile edges:", ", ".join(f"{e:.4f}" for e in edges.tolist()))

    def _row(mask: np.ndarray) -> tuple[float, float, float, float, int]:
        yy = y[mask]
        pt = p_tr[mask]
        po = p_or[mask]
        return _auc_fast(pt, yy), _auc_fast(po, yy), _bce(pt, yy), _bce(po, yy), int(yy.size)

    # Overall.
    auc_tr, auc_or, bce_tr, bce_or, n_all = _row(np.ones_like(y, dtype=bool))
    print(
        f"[overall] n={n_all} auc(trainP)={auc_tr:.4f} auc(oracleP)={auc_or:.4f} "
        f"Δauc={auc_or - auc_tr:+.4f} bce(trainP)={bce_tr:.4f} bce(oracleP)={bce_or:.4f} Δbce={bce_or - bce_tr:+.4f}"
    )

    print("[by_drift_bin] (higher bin => larger |Δp|)")
    for bi in range(q):
        m = b == bi
        if not np.any(m):
            continue
        auc_tr, auc_or, bce_tr, bce_or, nn = _row(m)
        lo = edges[bi]
        hi = edges[bi + 1]
        print(
            f"  - bin{bi} |Δp|∈[{lo:.4f},{hi:.4f}] n={nn} "
            f"auc(trainP)={auc_tr:.4f} auc(oracleP)={auc_or:.4f} Δauc={auc_or - auc_tr:+.4f} "
            f"bce(trainP)={bce_tr:.4f} bce(oracleP)={bce_or:.4f} Δbce={bce_or - bce_tr:+.4f}"
        )


if __name__ == "__main__":
    main()
