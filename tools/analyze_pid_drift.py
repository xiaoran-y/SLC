#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

TestMode = Literal["ztest", "fisher", "auto"]


@dataclass(frozen=True)
class SplitCounts:
    n: np.ndarray  # int64, shape [n_units+1]
    x: np.ndarray  # float64 (#correct), shape [n_units+1]


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


def _bincount_accumulate(n_units: int, unit_line: str, a_line: str, *, group: str) -> tuple[np.ndarray, np.ndarray]:
    unit_arr = np.fromstring(unit_line.strip(), dtype=np.int64, sep=",")
    a_arr = np.fromstring(a_line.strip(), dtype=np.int64, sep=",")
    if unit_arr.size == 0:
        return np.zeros(n_units + 1, dtype=np.int64), np.zeros(n_units + 1, dtype=np.float64)
    if a_arr.size != unit_arr.size:
        raise ValueError(f"{group}/a length mismatch: {group}={unit_arr.size} a={a_arr.size}")
    if unit_arr.min(initial=1) < 0 or unit_arr.max(initial=0) > n_units:
        raise ValueError(
            f"{group} out of range: min={unit_arr.min()} max={unit_arr.max()} expected [0,{n_units}]"
        )
    n = np.bincount(unit_arr, minlength=n_units + 1).astype(np.int64, copy=False)
    x = np.bincount(unit_arr, weights=a_arr.astype(np.float64), minlength=n_units + 1).astype(np.float64, copy=False)
    return n, x


def load_split_counts(path: Path, *, n_units: int, group: str) -> SplitCounts:
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
            dn, dx = _bincount_accumulate(n_units, unit_line, a_line, group=group)
            n += dn
            x += dx
    return SplitCounts(n=n, x=x)


def _fdr_bh(pvals: np.ndarray, *, q: float) -> np.ndarray:
    if pvals.ndim != 1:
        raise ValueError("pvals must be 1D")
    if not (0.0 < q < 1.0):
        raise ValueError(f"q must be in (0,1), got {q}")

    ok = np.isfinite(pvals)
    out = np.zeros_like(pvals, dtype=bool)
    if not np.any(ok):
        return out

    p = pvals[ok]
    m = p.size
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    thresh = (np.arange(1, m + 1, dtype=np.float64) / float(m)) * float(q)
    passed = ranked <= thresh
    if not np.any(passed):
        return out
    k = int(np.max(np.where(passed)[0]))
    cutoff = float(ranked[k])
    out[ok] = pvals[ok] <= cutoff
    return out


def _ztest_pvals(x_a: np.ndarray, n_a: np.ndarray, x_b: np.ndarray, n_b: np.ndarray) -> np.ndarray:
    try:
        from scipy.stats import norm  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scipy is required for ztest p-values") from e

    n1 = n_a.astype(np.float64)
    n2 = n_b.astype(np.float64)
    p1 = x_a / n1
    p2 = x_b / n2
    p_pool = (x_a + x_b) / (n1 + n2)
    se = np.sqrt(np.clip(p_pool * (1.0 - p_pool), 0.0, 1.0) * (1.0 / n1 + 1.0 / n2))
    z = np.zeros_like(se, dtype=np.float64)
    np.divide((p2 - p1), se, out=z, where=se > 0.0)
    pvals = 2.0 * norm.sf(np.abs(z))
    pvals = np.where(np.isfinite(pvals), pvals, 1.0)
    pvals = np.where(se > 0.0, pvals, 1.0)
    return pvals.astype(np.float64, copy=False)


def _ztest_z(x_a: np.ndarray, n_a: np.ndarray, x_b: np.ndarray, n_b: np.ndarray) -> np.ndarray:
    n1 = n_a.astype(np.float64)
    n2 = n_b.astype(np.float64)
    p1 = x_a / n1
    p2 = x_b / n2
    p_pool = (x_a + x_b) / (n1 + n2)
    se = np.sqrt(np.clip(p_pool * (1.0 - p_pool), 0.0, 1.0) * (1.0 / n1 + 1.0 / n2))
    z = np.zeros_like(se, dtype=np.float64)
    np.divide((p2 - p1), se, out=z, where=se > 0.0)
    z = np.where(np.isfinite(z), z, 0.0)
    return z.astype(np.float64, copy=False)


def _fisher_pvals(x_a: np.ndarray, n_a: np.ndarray, x_b: np.ndarray, n_b: np.ndarray) -> np.ndarray:
    try:
        from scipy.stats import fisher_exact  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scipy is required for fisher p-values") from e

    pvals = np.ones_like(x_a, dtype=np.float64)
    for i in range(x_a.size):
        a1 = int(x_a[i])
        b1 = int(n_a[i] - x_a[i])
        a2 = int(x_b[i])
        b2 = int(n_b[i] - x_b[i])
        _odds, p = fisher_exact([[a1, b1], [a2, b2]], alternative="two-sided")
        pvals[i] = float(p)
    return pvals


def _summarize_percentiles(name: str, values: np.ndarray) -> str:
    if values.size == 0:
        return f"{name}: (empty)"
    qs = np.percentile(values, [10, 25, 50, 75, 90, 95, 99])
    return (
        f"{name}: "
        f"p10={qs[0]:.4f} p25={qs[1]:.4f} p50={qs[2]:.4f} "
        f"p75={qs[3]:.4f} p90={qs[4]:.4f} p95={qs[5]:.4f} p99={qs[6]:.4f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze P(y|unit) drift between time-split partitions.")
    ap.add_argument("--dataset", type=str, required=True, help="Dataset name under data/<dataset>/")
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
    ap.add_argument("--split_a", type=str, default="train", choices=["train", "valid", "test"])
    ap.add_argument("--split_b", type=str, default="test", choices=["train", "valid", "test"])
    ap.add_argument("--min_count", type=int, default=20, help="Minimum per-split unit count to include in stats/tests.")
    ap.add_argument("--test", type=str, default="ztest", choices=["ztest", "fisher", "auto"])
    ap.add_argument("--fdr_q", type=float, default=0.05)
    ap.add_argument(
        "--count_bins",
        type=str,
        default="1,5,20,100,500,2000,1000000000",
        help="Comma-separated bin edges for total_count; last edge should be very large.",
    )
    ap.add_argument("--out_csv", type=str, default="", help="Optional: write per-unit stats to CSV.")
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

    tag = "pid_drift" if group == "pid" else "q_drift"
    print(
        f"[{tag}] dataset={dataset} n_{group}={n_units} split_a={split_a} split_b={split_b} min_count={int(args.min_count)}"
    )
    ca = load_split_counts(path_a, n_units=n_units, group=group)
    cb = load_split_counts(path_b, n_units=n_units, group=group)

    n_a = ca.n
    n_b = cb.n
    x_a = ca.x
    x_b = cb.x

    min_count = int(args.min_count)
    valid = (n_a >= min_count) & (n_b >= min_count)
    unit_ids = np.arange(n_units + 1, dtype=np.int64)[valid]
    unit_ids = unit_ids[unit_ids > 0]
    if unit_ids.size == 0:
        raise SystemExit(f"[{tag}] No units satisfy min_count; lower --min_count.")

    idx = unit_ids
    na = n_a[idx].astype(np.int64, copy=False)
    nb = n_b[idx].astype(np.int64, copy=False)
    xa = x_a[idx].astype(np.float64, copy=False)
    xb = x_b[idx].astype(np.float64, copy=False)

    pa = xa / na.astype(np.float64)
    pb = xb / nb.astype(np.float64)
    delta = pb - pa
    abs_delta = np.abs(delta)
    z = _ztest_z(xa, na, xb, nb)

    test_mode: TestMode = str(args.test)  # type: ignore[assignment]
    if test_mode == "ztest":
        pvals = _ztest_pvals(xa, na, xb, nb)
    elif test_mode == "fisher":
        pvals = _fisher_pvals(xa, na, xb, nb)
    elif test_mode == "auto":
        # Use z-test as default; fall back to fisher only for very small totals.
        pvals = _ztest_pvals(xa, na, xb, nb)
        small = (na + nb) < 200
        if np.any(small):
            pvals[small] = _fisher_pvals(xa[small], na[small], xb[small], nb[small])
    else:
        raise ValueError(f"Unknown test mode: {test_mode}")

    sig_fdr = _fdr_bh(pvals, q=float(args.fdr_q))
    sig_005 = pvals < 0.05

    total = na + nb
    log_total = np.log1p(total.astype(np.float64))
    try:
        from scipy.stats import spearmanr  # type: ignore[import-not-found]

        rho, rho_p = spearmanr(log_total, abs_delta)
        corr_summary = f"spearman(log1p(total), |Δp|) rho={float(rho):.4f} p={float(rho_p):.3g}"
    except Exception:
        corr = np.corrcoef(log_total, abs_delta)[0, 1]
        corr_summary = f"pearson(log1p(total), |Δp|) r={float(corr):.4f}"

    print(
        f"[{tag}] evaluated_units={unit_ids.size}  sig(p<0.05)={int(sig_005.sum())}  sig(FDR@{float(args.fdr_q):.2g})={int(sig_fdr.sum())}"
    )
    print(f"[{tag}] {corr_summary}")
    print(f"[{tag}] {_summarize_percentiles('|Δp|', abs_delta)}")
    print(f"[{tag}] {_summarize_percentiles('Δp', delta)}")

    # Bin summary.
    edges = [int(x) for x in str(args.count_bins).split(",") if str(x).strip()]
    if len(edges) < 2 or sorted(edges) != edges:
        raise ValueError("--count_bins must be sorted and have >=2 edges")
    bins = np.digitize(total, edges, right=True) - 1
    # bins in [0, len(edges)-2]
    print(f"[{tag}] by total_count bin:")
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        m = bins == b
        if not np.any(m):
            continue
        s = (
            f"  - [{lo},{hi}): n_unit={int(m.sum())} "
            f"mean|Δp|={float(abs_delta[m].mean()):.4f} "
            f"sig_fdr={float(sig_fdr[m].mean()):.3f}"
        )
        print(s)

    # Top-k.
    k = min(20, unit_ids.size)
    top = np.argsort(-abs_delta)[:k]
    print(f"[{tag}] top{k} by |Δp|:")
    for i in top.tolist():
        print(
            f"  {group}={int(unit_ids[i])} "
            f"na={int(na[i])} pa={float(pa[i]):.3f} "
            f"nb={int(nb[i])} pb={float(pb[i]):.3f} "
            f"Δp={float(delta[i]):+.3f} p={float(pvals[i]):.3g} fdr={bool(sig_fdr[i])}"
        )

    out_csv = str(args.out_csv).strip()
    if out_csv:
        try:
            import pandas as pd
        except Exception as e:  # pragma: no cover
            raise RuntimeError("pandas is required for --out_csv") from e

        df = pd.DataFrame(
            {
                group: unit_ids.astype(np.int64),
                f"n_{split_a}": na.astype(np.int64),
                f"x_{split_a}": xa.astype(np.float64),
                f"p_{split_a}": pa.astype(np.float64),
                f"n_{split_b}": nb.astype(np.int64),
                f"x_{split_b}": xb.astype(np.float64),
                f"p_{split_b}": pb.astype(np.float64),
                "delta_p": delta.astype(np.float64),
                "abs_delta_p": abs_delta.astype(np.float64),
                "z": z.astype(np.float64),
                "pvalue": pvals.astype(np.float64),
                f"sig_fdr_q{float(args.fdr_q):g}": sig_fdr.astype(bool),
            }
        )
        out_path = Path(out_csv)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[{tag}] wrote {out_path}")


if __name__ == "__main__":
    main()
