#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_meta(data_dir: Path) -> dict:
    meta_path = data_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found under {data_dir}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _load_train_pid_a_count(path: Path, *, n_pid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flat arrays (pid, a, count) for the train split."""
    pid_all: list[np.ndarray] = []
    a_all: list[np.ndarray] = []
    c_all: list[np.ndarray] = []

    with path.open("r", encoding="utf-8") as f:
        while True:
            header = f.readline()
            if not header:
                break
            tokens = header.strip().split(",")
            if len(tokens) < 2:
                raise ValueError(f"Bad header line in {path}: {header!r}")
            try:
                base_offset = int(tokens[2]) if len(tokens) > 2 else 0
            except (ValueError, TypeError):
                base_offset = 0

            pid_line = f.readline()
            q_line = f.readline()
            a_line = f.readline()
            if not (pid_line and q_line and a_line):
                raise ValueError(f"Unexpected EOF in {path} after header: {header.strip()!r}")

            pid = np.fromstring(pid_line.strip(), dtype=np.int64, sep=",")
            q = np.fromstring(q_line.strip(), dtype=np.int64, sep=",")
            a = np.fromstring(a_line.strip(), dtype=np.int64, sep=",")
            if pid.size == 0:
                continue
            if not (pid.size == q.size == a.size):
                raise ValueError(f"Length mismatch in {path}: pid={pid.size} q={q.size} a={a.size}")

            # count is a (monotone) global index inside each user after time split.
            # We treat it as a proxy for "time within train".
            count = (base_offset + np.arange(1, pid.size + 1, dtype=np.int64)).astype(np.int64)

            m = (q > 0) & (pid > 0)
            if not np.any(m):
                continue
            pid_m = pid[m]
            a_m = a[m]
            c_m = count[m]
            if pid_m.max(initial=0) > n_pid:
                raise ValueError(f"pid out of range in {path}: max={int(pid_m.max())} expected <= {n_pid}")
            pid_all.append(pid_m)
            a_all.append(a_m)
            c_all.append(c_m)

    if not pid_all:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return (
        np.concatenate(pid_all).astype(np.int64, copy=False),
        np.concatenate(a_all).astype(np.int64, copy=False),
        np.concatenate(c_all).astype(np.int64, copy=False),
    )


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    try:
        from scipy.stats import spearmanr  # type: ignore[import-not-found]

        rho, p = spearmanr(x, y)
        return float(rho), float(p)
    except Exception:
        c = np.corrcoef(x.astype(np.float64), y.astype(np.float64))[0, 1]
        return float(c), float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train-only predictability of per-pid drift residual (assist2017 focus).")
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--train_set", type=int, default=1)
    ap.add_argument("--data_root", type=str, default="dataset")
    ap.add_argument("--runs_root", type=str, default="_runs", help="Default root for derived analysis artifacts.")
    ap.add_argument("--train_cut", type=float, default=0.5, help="Quantile split point over train count proxy.")
    ap.add_argument("--min_count_half", type=int, default=20)
    ap.add_argument(
        "--target_csv",
        type=str,
        default="",
        help="CSV with per-pid drift decomposition, containing delta_p_residual. "
        "If empty, defaults to logs/pid_drift/<dataset_source>_drift_causes_skill_min20.csv when available.",
    )
    ap.add_argument("--target_col", type=str, default="delta_p_residual")
    ap.add_argument("--abs_target_col", type=str, default="abs_delta_p", help="Optional: used only for reporting bins.")
    ap.add_argument("--n_bins", type=int, default=5)
    ap.add_argument("--cut_abs_target", type=float, default=0.05, help="Threshold for sign_acc on |target|.")
    ap.add_argument("--out_csv", type=str, default="", help="Write merged per-pid table to CSV.")
    args = ap.parse_args()

    dataset = str(args.dataset)
    train_set = int(args.train_set)
    data_root = Path(str(args.data_root))
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    runs_root = Path(str(args.runs_root))
    if not runs_root.is_absolute():
        runs_root = REPO_ROOT / runs_root
    train_cut = float(args.train_cut)
    min_count_half = int(args.min_count_half)
    n_bins = int(args.n_bins)
    cut_abs_target = float(args.cut_abs_target)

    data_dir = data_root / dataset
    meta = _read_meta(data_dir)
    n_pid = int(meta.get("n_pid", 0))
    if n_pid <= 0:
        raise ValueError(f"Invalid n_pid in {data_dir/'meta.json'}: {n_pid}")

    train_path = data_dir / f"{dataset}_train{train_set}.csv"
    if not train_path.is_file():
        raise FileNotFoundError(f"Missing split file: {train_path}")

    pid_flat, a_flat, c_flat = _load_train_pid_a_count(train_path, n_pid=n_pid)
    if pid_flat.size == 0:
        raise SystemExit("[drift_pred] No train interactions found.")

    # Split train by count proxy (time within train).
    cut = np.quantile(c_flat.astype(np.float64), q=train_cut, method="linear")
    early = c_flat <= cut
    late = c_flat > cut

    # Aggregate per pid.
    n_tr0 = np.bincount(pid_flat[early], minlength=n_pid + 1).astype(np.int64, copy=False)
    x_tr0 = np.bincount(pid_flat[early], weights=a_flat[early].astype(np.float64), minlength=n_pid + 1).astype(
        np.float64, copy=False
    )
    n_tr1 = np.bincount(pid_flat[late], minlength=n_pid + 1).astype(np.int64, copy=False)
    x_tr1 = np.bincount(pid_flat[late], weights=a_flat[late].astype(np.float64), minlength=n_pid + 1).astype(
        np.float64, copy=False
    )

    valid = (n_tr0 >= min_count_half) & (n_tr1 >= min_count_half)
    valid[0] = False
    pid_ids = np.flatnonzero(valid).astype(np.int64)
    if pid_ids.size == 0:
        raise SystemExit("[drift_pred] No pids satisfy min_count_half in BOTH halves; lower --min_count_half.")

    p_tr0 = (x_tr0[pid_ids] / n_tr0[pid_ids].astype(np.float64)).astype(np.float64)
    p_tr1 = (x_tr1[pid_ids] / n_tr1[pid_ids].astype(np.float64)).astype(np.float64)
    delta_train = (p_tr1 - p_tr0).astype(np.float64)

    # Load target (drift residual) table.
    target_csv = str(args.target_csv).strip()
    if not target_csv:
        # Default: <runs_root>/pid_drift/<source>_drift_causes_skill_min20.csv (assist2017 -> assist2017_...)
        source = str(meta.get("source", dataset))
        target_csv = str(runs_root / "pid_drift" / f"{source}_drift_causes_skill_min20.csv")
    target_path = Path(target_csv)
    if not target_path.is_absolute():
        target_path = REPO_ROOT / target_path
    if not target_path.is_file():
        raise FileNotFoundError(f"target_csv not found: {target_path}")

    try:
        import pandas as pd
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pandas is required for analyze_pid_drift_predictability.py") from e

    df = pd.read_csv(target_path)
    if "pid" not in df.columns:
        raise ValueError(f"target_csv must have pid column: {target_path}")
    if args.target_col not in df.columns:
        raise ValueError(f"target_csv missing target_col={args.target_col!r}: columns={list(df.columns)}")
    df = df.set_index("pid", drop=False)

    # Align with valid pid_ids.
    pid_in = [int(pid) for pid in pid_ids.tolist() if int(pid) in df.index]
    if not pid_in:
        raise SystemExit("[drift_pred] No overlapping pids between train-only set and target_csv.")

    target = df.loc[pid_in, args.target_col].to_numpy(dtype=np.float64, copy=False)
    abs_target = np.abs(target)

    # Re-align delta_train arrays to pid_in order.
    pid_to_pos = {int(pid): i for i, pid in enumerate(pid_ids.tolist())}
    idx = np.asarray([pid_to_pos[int(pid)] for pid in pid_in], dtype=np.int64)
    delta_train_in = delta_train[idx]

    rho1, p1 = _spearman(delta_train_in, target)
    rho2, p2 = _spearman(np.abs(delta_train_in), abs_target)

    # Sign accuracy on large targets.
    mask_large = abs_target >= cut_abs_target
    sign_acc = float("nan")
    if int(mask_large.sum()) > 0:
        sign_acc = float((np.sign(delta_train_in[mask_large]) == np.sign(target[mask_large])).mean())

    # Drift bins by |target| (for reporting; this is a diagnostic / leak-allowed analysis).
    if n_bins < 2:
        raise ValueError("--n_bins must be >= 2")
    edges = np.quantile(abs_target, q=np.linspace(0.0, 1.0, n_bins + 1), method="linear").astype(np.float64)
    edges[0] = 0.0
    edges[-1] = float(max(edges[-1], float(abs_target.max(initial=0.0))))
    drift_bin = np.digitize(abs_target, edges, right=True) - 1
    drift_bin = np.clip(drift_bin, 0, n_bins - 1)

    print(
        f"[drift_pred] dataset={dataset} train_cut={train_cut:.3f} min_count_half={min_count_half} "
        f"pids_used={len(pid_in)} target={args.target_col}"
    )
    print(f"[drift_pred] spearman(delta_train, target): rho={rho1:.4f} p={p1:.3g}")
    print(f"[drift_pred] spearman(|delta_train|, |target|): rho={rho2:.4f} p={p2:.3g}")
    if np.isfinite(sign_acc):
        print(f"[drift_pred] sign_acc(|target|>={cut_abs_target:.3f})={sign_acc:.4f} (n={int(mask_large.sum())})")

    print("[drift_pred] by drift_bin:")
    for b in range(n_bins):
        m = drift_bin == b
        if not np.any(m):
            continue
        rho_b, _p_b = _spearman(delta_train_in[m], target[m])
        mm = abs_target[m] >= cut_abs_target
        sa_b = float("nan")
        if int(mm.sum()) > 0:
            sa_b = float((np.sign(delta_train_in[m][mm]) == np.sign(target[m][mm])).mean())
        print(
            f"  - bin{b}: n={int(m.sum())} mean|target|={float(abs_target[m].mean()):.4f} rho={rho_b:.4f} sign_acc={sa_b:.4f}"
        )

    out_csv = str(args.out_csv).strip()
    if out_csv:
        out_path = Path(out_csv)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df = pd.DataFrame(
            {
                "pid": np.asarray(pid_in, dtype=np.int64),
                "n_tr0": n_tr0[pid_in].astype(np.int64),
                "p_tr0": (x_tr0[pid_in] / n_tr0[pid_in].astype(np.float64)).astype(np.float64),
                "n_tr1": n_tr1[pid_in].astype(np.int64),
                "p_tr1": (x_tr1[pid_in] / n_tr1[pid_in].astype(np.float64)).astype(np.float64),
                "delta_train": delta_train_in.astype(np.float64),
                "target": target.astype(np.float64),
                "abs_target": abs_target.astype(np.float64),
                "drift_bin": drift_bin.astype(np.int64),
            }
        )
        out_df.to_csv(out_path, index=False)
        print(f"[drift_pred] wrote {out_path}")


if __name__ == "__main__":
    main()
