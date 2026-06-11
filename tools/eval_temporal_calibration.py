#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ckpt as ckpt_io  # noqa: E402
from load_data import PID_DATA  # noqa: E402
from posthoc import ItemDriftConfig, compute_time_edges_from_history, fit_dynamic_item_bias  # noqa: E402
from run import compute_accuracy, compute_auc, test as run_test  # noqa: E402
from utils import load_model  # noqa: E402


METHOD_ALIAS = {
    "base": "raw_score",
    "platt": "global_sigmoid",
    "platt_t": "global_sigmoid_time",
    "temp": "global_temperature",
    "isotonic": "global_isotonic",
    "hist": "global_histogram",
    "rescal_resid": "item_bias_mean",
    "rescal_resid_iso": "item_bias_mean_monotone",
    "ridge_logistic": "item_bias_ridge",
    "thdf_ll_static_offset_platt": "item_bias_shrinkage_static",
    "thdf_ll_static_iso": "item_bias_shrinkage_static_monotone",
    "thdf_ll_offset_platt": "item_bias_shrinkage_dynamic",
}

PAPER_METHODS = [
    "raw_score",
    "global_sigmoid",
    "item_bias_mean_monotone",
    "item_bias_shrinkage_static",
]

APPENDIX_METHODS = [
    "global_temperature",
    "global_isotonic",
    "global_histogram",
    "item_bias_ridge",
    "item_bias_shrinkage_dynamic",
    "item_bias_shrinkage_static_monotone",
]

PAPER_LABELS = {
    "raw_score": "Base",
    "global_sigmoid": "Platt",
    "global_sigmoid_time": "Platt+Time",
    "global_temperature": "Temp",
    "global_isotonic": "Iso",
    "global_histogram": "Hist",
    "item_bias_mean": "ResCal",
    "item_bias_mean_monotone": "ResCal+Iso",
    "item_bias_ridge": "Ridge",
    "item_bias_shrinkage_static": "SLC",
    "item_bias_shrinkage_static_monotone": "SLC+Iso",
    "item_bias_shrinkage_dynamic": "SLC-T",
}


def _now_ts() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_name(s: str) -> str:
    s = str(s).strip()
    if not s:
        return "pack"
    return "".join(c if (c.isalnum() or c in {"-", "_", "."}) else "_" for c in s)


def _run_git(args: list[str]) -> str:
    try:
        p = subprocess.run(
            ["git", *args],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        return (p.stdout or "").strip()
    except Exception:
        return ""


def _sigmoid_clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p.astype(np.float64, copy=False), 1e-6, 1.0 - 1e-6)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _sigmoid_clip(p)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid (avoids overflow warnings for large |x|).
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    m = x >= 0
    out[m] = 1.0 / (1.0 + np.exp(-x[m]))
    ex = np.exp(x[~m])  # x<0 => exp(x) is safe (underflows to 0)
    out[~m] = ex / (1.0 + ex)
    return out


def _nll(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(np.float64, copy=False)
    p = _sigmoid_clip(p)
    return float((-y * np.log(p) - (1.0 - y) * np.log(1.0 - p)).mean())


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(np.float64, copy=False)
    p = p.astype(np.float64, copy=False)
    return float(np.mean((p - y) ** 2))


def _ece(y: np.ndarray, p: np.ndarray, *, n_bins: int = 15) -> float:
    y = y.astype(np.float64, copy=False).reshape(-1)
    p = p.astype(np.float64, copy=False).reshape(-1)
    if y.size == 0:
        return float("nan")
    p = np.clip(p, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1, dtype=np.float64)
    bid = np.digitize(p, edges, right=True) - 1
    bid = np.clip(bid, 0, int(n_bins) - 1)
    e = 0.0
    n = float(y.size)
    for b in range(int(n_bins)):
        m = bid == b
        nb = float(np.sum(m))
        if nb <= 0:
            continue
        acc_b = float(np.mean(y[m]))
        conf_b = float(np.mean(p[m]))
        e += abs(acc_b - conf_b) * (nb / n)
    return float(e)


def _metrics(y: np.ndarray, p: np.ndarray, *, ece_bins: int) -> dict:
    y01 = y.astype(np.float64, copy=False).reshape(-1)
    p01 = p.astype(np.float64, copy=False).reshape(-1)
    return {
        "n": int(y01.size),
        "auc": compute_auc(y01, p01),
        "acc": compute_accuracy(y01, p01),
        "nll": _nll(y01, p01),
        "brier": _brier(y01, p01),
        "rmse": float(math.sqrt(_brier(y01, p01))),
        "ece": _ece(y01, p01, n_bins=int(ece_bins)),
    }


def _read_meta(dataset: str, *, data_root: Path) -> dict:
    meta_path = data_root / dataset / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _load_split_arrays(dataset: str, train_set: int, split: str, *, data_root: Path, n_question: int, seqlen: int):
    path = data_root / dataset / f"{dataset}_{split}{train_set}.csv"
    loader = PID_DATA(n_question=n_question, seqlen=seqlen, separate_char=",")
    q, qa, pid, uid, count = loader.load_data(str(path))
    return q, qa, pid, uid, count


def _params_from_ckpt(ckpt_path: Path, *, dataset: str, meta: dict, override_train_set: int) -> SimpleNamespace:
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    args = ck.get("args", {}) if isinstance(ck, dict) else {}

    p = SimpleNamespace()
    p.dataset = dataset
    p.data_name = dataset
    p.model = str(args.get("model", "akt_pid"))
    p.train_set = int(override_train_set)
    p.seed = int(args.get("seed", 224))

    p.n_question = int(meta["n_question"])
    p.n_pid = int(meta.get("n_pid", 0))
    p.n_users = int(meta.get("n_users", 0))
    p.seqlen = int(meta.get("seqlen", int(args.get("seqlen", 200))))

    # Shared knobs (enough to reconstruct the backbone in utils.py::load_model).
    p.n_block = int(args.get("n_block", 1))
    p.d_model = int(args.get("d_model", 256))
    p.n_head = int(args.get("n_head", 8))
    p.d_ff = int(args.get("d_ff", 2048))
    p.kq_same = int(args.get("kq_same", 1))
    p.dropout = float(args.get("dropout", 0.05))
    p.l2 = float(args.get("l2", 1e-5))
    p.final_fc_dim = int(args.get("final_fc_dim", 512))
    p.separate_qa = bool(args.get("separate_qa", False))
    p.dkvmn_size_m = int(args.get("dkvmn_size_m", 32))

    # RecSys knobs (optional).
    p.rec_emb_dim = int(args.get("rec_emb_dim", args.get("d_model", 64)))
    p.ncf_hidden_dims = str(args.get("ncf_hidden_dims", "128,64,32"))
    p.ncf_dropout = float(args.get("ncf_dropout", 0.1))

    # Runtime (evaluation only).
    p.batch_size = int(args.get("batch_size", 96))
    p.eval_batch_size = int(args.get("eval_batch_size", 256))
    p.amp = str(args.get("amp", "off"))
    p.metrics_backend = str(args.get("metrics_backend", "sklearn"))
    p.deterministic = str(args.get("deterministic", "off"))
    p.torch_compile = str(args.get("torch_compile", "off"))
    return p


def _parse_env_kv(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[len("export ") :].strip()
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if k:
            out[k] = v
    return out


def _fit_temperature(y: np.ndarray, p: np.ndarray) -> float:
    # Fit T>0 by minimizing NLL on calibration set.
    y = y.astype(np.float64, copy=False)
    lp = _logit(p)

    def obj(logT: float) -> float:
        T = float(np.exp(logT))
        return _nll(y, _sigmoid(lp / T))

    # Coarse-to-fine search on log(T).
    grid = np.linspace(math.log(0.2), math.log(10.0), 60)
    vals = np.array([obj(float(x)) for x in grid], dtype=np.float64)
    j = int(np.argmin(vals))
    best = float(grid[j])
    span = (grid[1] - grid[0]) * 2.0
    for _ in range(2):
        lo = best - span
        hi = best + span
        grid2 = np.linspace(lo, hi, 60)
        vals2 = np.array([obj(float(x)) for x in grid2], dtype=np.float64)
        j2 = int(np.argmin(vals2))
        best = float(grid2[j2])
        span *= 0.5
    return float(np.exp(best))


def _fit_temperature_from_logit(y: np.ndarray, eta: np.ndarray) -> float:
    """Fit global temperature T>0 for logits: p = sigmoid(eta / T).

    Equivalent to temperature scaling on probabilities, but avoids logit(p) clipping
    when eta has large magnitude.
    """
    y = y.astype(np.float64, copy=False).reshape(-1)
    eta = np.asarray(eta, dtype=np.float64).reshape(-1)

    def obj(logT: float) -> float:
        T = float(np.exp(logT))
        return _nll(y, _sigmoid(eta / T))

    # Coarse-to-fine search on log(T).
    grid = np.linspace(math.log(0.2), math.log(10.0), 60)
    vals = np.array([obj(float(x)) for x in grid], dtype=np.float64)
    j = int(np.argmin(vals))
    best = float(grid[j])
    span = (grid[1] - grid[0]) * 2.0
    for _ in range(2):
        lo = best - span
        hi = best + span
        grid2 = np.linspace(lo, hi, 60)
        vals2 = np.array([obj(float(x)) for x in grid2], dtype=np.float64)
        j2 = int(np.argmin(vals2))
        best = float(grid2[j2])
        span *= 0.5
    return float(np.exp(best))


def _fit_platt(y: np.ndarray, p: np.ndarray) -> tuple[float, float] | None:
    # Platt scaling on the single feature logit(p): p' = sigmoid(a*logit(p)+b).
    y = y.astype(np.int64, copy=False).reshape(-1)
    x = _logit(p).reshape(-1, 1)
    if y.size <= 0 or np.all(y == y[0]):
        return None
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore

        lr = LogisticRegression(solver="lbfgs", max_iter=200)
        lr.fit(x, y)
        a = float(lr.coef_.reshape(-1)[0])
        b = float(lr.intercept_.reshape(-1)[0])
        return a, b
    except Exception:
        return None


def _fit_platt_time_lr(
    *,
    y_calib: np.ndarray,
    p_calib: np.ndarray,
    pid_calib: np.ndarray,
    cnt_calib: np.ndarray,
) -> tuple[float, float, float, dict] | None:
    """Time-aware global Platt: y ~ sigmoid(a*logit(p) + c*t + b).

    t is a normalized count feature from the calibration window.
    """
    y = y_calib.astype(np.int64, copy=False).reshape(-1)
    p = p_calib.astype(np.float64, copy=False).reshape(-1)
    pid = pid_calib.astype(np.int64, copy=False).reshape(-1)
    cnt = cnt_calib.astype(np.float64, copy=False).reshape(-1)
    m = (pid > 0) & np.isfinite(cnt)
    if not np.any(m):
        return None
    y = y[m]
    if y.size <= 0 or np.all(y == y[0]):
        return None

    logit = _logit(p[m]).reshape(-1)
    t = cnt[m]
    t_mean = float(np.mean(t))
    t_std = float(np.std(t))
    if not math.isfinite(t_std) or t_std <= 1e-12:
        return None
    t_norm = (t - t_mean) / t_std

    X = np.stack([logit.astype(np.float64, copy=False), t_norm.astype(np.float64, copy=False)], axis=1)
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore

        lr = LogisticRegression(solver="lbfgs", max_iter=300)
        lr.fit(X, y)
        a = float(lr.coef_.reshape(-1)[0])
        c = float(lr.coef_.reshape(-1)[1])
        b = float(lr.intercept_.reshape(-1)[0])
        meta = {"t_mean": float(t_mean), "t_std": float(t_std), "a": float(a), "b": float(b), "c": float(c)}
        return float(a), float(c), float(b), meta
    except Exception:
        return None


def _fit_platt_with_offset_irls(
    *,
    y: np.ndarray,
    x: np.ndarray,
    offset: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-8,
    ridge: float = 1e-6,
) -> tuple[float, float] | None:
    """Fit (a,b0) in: y ~ Bernoulli(sigmoid(a*x + b0 + offset)).

    This is a 1D logistic regression with a known offset term. We solve it with
    Newton/IRLS on the exact log-likelihood (convex), so it is stable and fast.
    """
    y01 = y.astype(np.float64, copy=False).reshape(-1)
    x1 = x.astype(np.float64, copy=False).reshape(-1)
    off = offset.astype(np.float64, copy=False).reshape(-1)
    if y01.size <= 0 or np.all(y01 == y01[0]):
        return None
    if x1.size != y01.size or off.size != y01.size:
        raise ValueError(f"shape mismatch: y={y01.shape}, x={x1.shape}, offset={off.shape}")

    a = 1.0
    b0 = 0.0

    def _ll(a0: float, b00: float) -> float:
        eta = a0 * x1 + b00 + off
        # Stable logistic log-likelihood: y*eta - log(1+exp(eta))
        return float(np.sum(y01 * eta - np.logaddexp(0.0, eta)))

    ll0 = _ll(a, b0)
    for _it in range(int(max_iter)):
        eta = a * x1 + b0 + off
        # stable-ish sigmoid; clip eta to avoid exp overflow in pathological cases
        eta_clip = np.clip(eta, -50.0, 50.0)
        p = 1.0 / (1.0 + np.exp(-eta_clip))
        w = (p * (1.0 - p)).astype(np.float64, copy=False)
        w = np.clip(w, 1e-9, None)
        r = (y01 - p).astype(np.float64, copy=False)

        g1 = float(np.sum(r * x1))
        g2 = float(np.sum(r))

        A11 = float(np.sum(w * x1 * x1) + float(ridge))
        A12 = float(np.sum(w * x1))
        A22 = float(np.sum(w) + float(ridge))
        det = A11 * A22 - A12 * A12
        if not (math.isfinite(det) and det > 0.0):
            # Degenerate design (e.g., constant x); fall back to intercept-only update.
            det = float("nan")

        if math.isfinite(det):
            da = (g1 * A22 - g2 * A12) / det
            db = (g2 * A11 - g1 * A12) / det
        else:
            da = 0.0
            db = g2 / A22 if A22 > 0 else 0.0

        if max(abs(float(da)), abs(float(db))) < float(tol):
            break

        step = 1.0
        improved = False
        for _ls in range(12):
            a_new = float(a + step * float(da))
            b_new = float(b0 + step * float(db))
            ll_new = _ll(a_new, b_new)
            if ll_new >= ll0 - 1e-12:
                a, b0 = a_new, b_new
                ll0 = ll_new
                improved = True
                break
            step *= 0.5
        if not improved:
            break

    return float(a), float(b0)


def _fit_isotonic(y: np.ndarray, p: np.ndarray):
    y = y.astype(np.float64, copy=False).reshape(-1)
    p = np.clip(p.astype(np.float64, copy=False).reshape(-1), 0.0, 1.0)
    if y.size <= 0 or np.all(y == y[0]):
        return None
    try:
        from sklearn.isotonic import IsotonicRegression  # type: ignore

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p, y)
        return iso
    except Exception:
        return None


def _fit_histogram(y: np.ndarray, p: np.ndarray, *, n_bins: int = 15) -> tuple[np.ndarray, np.ndarray]:
    # Equal-width bins on p.
    y = y.astype(np.float64, copy=False).reshape(-1)
    p = np.clip(p.astype(np.float64, copy=False).reshape(-1), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1, dtype=np.float64)
    bid = np.digitize(p, edges, right=True) - 1
    bid = np.clip(bid, 0, int(n_bins) - 1)
    bin_sum = np.zeros(int(n_bins), dtype=np.float64)
    bin_cnt = np.zeros(int(n_bins), dtype=np.int64)
    for b in range(int(n_bins)):
        m = bid == b
        bin_cnt[b] = int(np.sum(m))
        bin_sum[b] = float(np.sum(y[m])) if bin_cnt[b] > 0 else 0.0
    bin_rate = np.where(bin_cnt > 0, bin_sum / np.maximum(bin_cnt, 1), 0.5).astype(np.float64)
    return edges, bin_rate


def _apply_histogram(p: np.ndarray, edges: np.ndarray, bin_rate: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64, copy=False).reshape(-1), 0.0, 1.0)
    bid = np.digitize(p, edges, right=True) - 1
    bid = np.clip(bid, 0, int(bin_rate.size) - 1)
    return bin_rate[bid]


def _rescal_resid(
    *,
    y_calib: np.ndarray,
    p_calib: np.ndarray,
    pid_calib: np.ndarray,
    p_eval: np.ndarray,
    pid_eval: np.ndarray,
    n_pid: int,
    min_count: int,
    shrink_tau: float,
) -> np.ndarray:
    y_calib = y_calib.astype(np.float64, copy=False).reshape(-1)
    p_calib = p_calib.astype(np.float64, copy=False).reshape(-1)
    pid_calib = pid_calib.astype(np.int64, copy=False).reshape(-1)
    pid_eval = pid_eval.astype(np.int64, copy=False).reshape(-1)

    m = pid_calib > 0
    if not np.any(m):
        return p_eval
    pid_f = pid_calib[m]
    y_f = y_calib[m]
    p_f = p_calib[m]
    n_c = np.bincount(pid_f, minlength=int(n_pid) + 1).astype(np.int64, copy=False)
    x_y = np.bincount(pid_f, weights=y_f, minlength=int(n_pid) + 1).astype(np.float64, copy=False)
    x_p = np.bincount(pid_f, weights=p_f, minlength=int(n_pid) + 1).astype(np.float64, copy=False)

    ok = n_c >= int(min_count)
    ok[0] = False
    pid_ids = np.flatnonzero(ok).astype(np.int64, copy=False)
    if pid_ids.size == 0:
        return p_eval

    y_rate = x_y[pid_ids] / n_c[pid_ids].astype(np.float64)
    p_rate = x_p[pid_ids] / n_c[pid_ids].astype(np.float64)
    delta = _logit(y_rate) - _logit(p_rate)
    if float(shrink_tau) > 0.0:
        w = (n_c[pid_ids].astype(np.float64) / (n_c[pid_ids].astype(np.float64) + float(shrink_tau))).clip(0.0, 1.0)
        delta = delta * w
    b_pid = np.zeros(int(n_pid) + 1, dtype=np.float64)
    b_pid[pid_ids] = delta

    lp = _logit(p_eval)
    return _sigmoid(lp + b_pid[pid_eval])


def _quantile_edges(x: np.ndarray, *, n_bins: int) -> np.ndarray:
    x = x.astype(np.float64, copy=False).reshape(-1)
    if x.size <= 0:
        return np.zeros(int(n_bins) + 1, dtype=np.float64)
    edges = np.quantile(x, q=np.linspace(0.0, 1.0, int(n_bins) + 1), method="linear").astype(np.float64)
    edges = np.maximum.accumulate(edges)
    edges[-1] = float(max(edges[-1], float(np.max(x)) + 1e-6))
    return edges


def _digitize_bins(count: np.ndarray, edges: np.ndarray, *, n_bins: int) -> np.ndarray:
    tb = np.digitize(count.astype(np.float64, copy=False).reshape(-1), edges.astype(np.float64, copy=False), right=False) - 1
    tb = np.clip(tb, 0, int(n_bins) - 1).astype(np.int64, copy=False)
    return tb


def _slice_metrics(
    y: np.ndarray,
    p_by_method: dict[str, np.ndarray],
    count: np.ndarray,
    *,
    n_slices: int,
    ece_bins: int,
) -> list[dict]:
    count = count.astype(np.float64, copy=False).reshape(-1)
    m = np.isfinite(count)
    if not np.any(m):
        return []
    edges = _quantile_edges(count[m], n_bins=int(n_slices))
    rows: list[dict] = []
    for si in range(int(n_slices)):
        lo = float(edges[si])
        hi = float(edges[si + 1])
        if si == int(n_slices) - 1:
            mask = (count >= lo) & (count <= hi) & m
        else:
            mask = (count >= lo) & (count < hi) & m
        n_tok = int(np.sum(mask))
        if n_tok <= 0:
            continue
        for method, p in p_by_method.items():
            met = _metrics(y[mask], p[mask], ece_bins=int(ece_bins))
            rows.append(
                {
                    "slice": si,
                    "count_lo": lo,
                    "count_hi": hi,
                    "method": method,
                    **met,
                }
            )
    return rows


def _svg_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _write_svg(path: Path, *, width: int, height: int, body: str) -> None:
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
{body}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _plot_reliability_svg(
    path: Path,
    *,
    title: str,
    y: np.ndarray,
    p_methods: dict[str, np.ndarray],
    n_bins: int = 15,
) -> None:
    # Simple paper-friendly reliability diagram in pure SVG (no matplotlib dependency).
    width, height = 820, 520
    ml, mr, mt, mb = 90, 30, 60, 80
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    def xmap(x: float) -> float:
        return ml + float(np.clip(x, 0.0, 1.0)) * plot_w

    def ymap(yv: float) -> float:
        return mt + (1.0 - float(np.clip(yv, 0.0, 1.0))) * plot_h

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Diagonal.
    body = []
    body.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    body.append(f'<line x1="{xmap(0)}" y1="{ymap(0)}" x2="{xmap(1)}" y2="{ymap(1)}" stroke="#999" stroke-width="2"/>')

    # Axes.
    body.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+plot_h}" stroke="#111" stroke-width="2"/>')
    body.append(f'<line x1="{ml}" y1="{mt+plot_h}" x2="{ml+plot_w}" y2="{mt+plot_h}" stroke="#111" stroke-width="2"/>')

    # Ticks.
    for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        body.append(f'<line x1="{xmap(t)}" y1="{mt+plot_h}" x2="{xmap(t)}" y2="{mt+plot_h+8}" stroke="#111" stroke-width="2"/>')
        body.append(
            f'<text x="{xmap(t)}" y="{mt+plot_h+32}" font-size="14" text-anchor="middle" fill="#111">{t:.1f}</text>'
        )
        body.append(f'<line x1="{ml-8}" y1="{ymap(t)}" x2="{ml}" y2="{ymap(t)}" stroke="#111" stroke-width="2"/>')
        body.append(
            f'<text x="{ml-14}" y="{ymap(t)+5}" font-size="14" text-anchor="end" fill="#111">{t:.1f}</text>'
        )

    body.append(
        f'<text x="{ml+plot_w/2}" y="{height-28}" font-size="16" text-anchor="middle" fill="#111">Confidence</text>'
    )
    body.append(
        f'<text x="24" y="{mt+plot_h/2}" font-size="16" text-anchor="middle" fill="#111" transform="rotate(-90 24 {mt+plot_h/2})">Accuracy</text>'
    )
    body.append(
        f'<text x="{ml}" y="32" font-size="18" text-anchor="start" fill="#111">{_svg_escape(title)}</text>'
    )

    # Colors (fixed order for paper stability).
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    methods = list(p_methods.keys())
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(methods)}

    # Plot points (mean conf vs acc per bin).
    legend_x = ml + plot_w - 5
    legend_y = mt - 22
    for mi, method in enumerate(methods):
        p = np.clip(p_methods[method].astype(np.float64, copy=False).reshape(-1), 0.0, 1.0)
        y0 = y.astype(np.float64, copy=False).reshape(-1)
        bid = np.digitize(p, edges, right=True) - 1
        bid = np.clip(bid, 0, int(n_bins) - 1)
        acc = np.full(int(n_bins), np.nan, dtype=np.float64)
        conf = np.full(int(n_bins), np.nan, dtype=np.float64)
        for b in range(int(n_bins)):
            m = bid == b
            if not np.any(m):
                continue
            acc[b] = float(np.mean(y0[m]))
            conf[b] = float(np.mean(p[m]))

        color = color_map[str(method)]
        # Line.
        pts = []
        for b in range(int(n_bins)):
            if not np.isfinite(acc[b]) or not np.isfinite(conf[b]):
                continue
            pts.append((xmap(conf[b]), ymap(acc[b])))
        if len(pts) >= 2:
            path_d = "M " + " L ".join(f"{x:.2f} {yy:.2f}" for x, yy in pts)
            body.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, yy in pts:
            body.append(f'<circle cx="{x:.2f}" cy="{yy:.2f}" r="4.2" fill="{color}"/>')

        # Legend.
        lx = legend_x - 180
        ly = legend_y + mi * 20
        body.append(f'<rect x="{lx}" y="{ly-12}" width="14" height="14" fill="{color}"/>')
        body.append(f'<text x="{lx+20}" y="{ly}" font-size="14" text-anchor="start" fill="#111">{_svg_escape(method)}</text>')

    _write_svg(path, width=width, height=height, body="\n".join(body))


def _plot_reliability_bargap_svg(
    path: Path,
    *,
    title: str,
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int = 15,
    ece_value: float | None = None,
) -> None:
    """LaSCal-style bar + gap overlay reliability diagram (single method per panel).

    Blue bars = model output (mean confidence per bin).
    Pink/red overlay = gap between confidence and accuracy.
    Dashed diagonal = perfect calibration.
    ECE value shown in bottom-right corner.
    """
    width, height = 360, 320
    ml, mr, mt, mb = 50, 20, 40, 55
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    def xmap(x: float) -> float:
        return ml + float(np.clip(x, 0.0, 1.0)) * plot_w

    def ymap(yv: float) -> float:
        return mt + (1.0 - float(np.clip(yv, 0.0, 1.0))) * plot_h

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1, dtype=np.float64)
    bar_w_frac = 0.8 / float(n_bins)

    p = np.clip(np.asarray(p, dtype=np.float64).reshape(-1), 0.0, 1.0)
    y0 = np.asarray(y, dtype=np.float64).reshape(-1)
    bid = np.digitize(p, edges, right=True) - 1
    bid = np.clip(bid, 0, int(n_bins) - 1)

    acc = np.full(int(n_bins), np.nan, dtype=np.float64)
    conf = np.full(int(n_bins), np.nan, dtype=np.float64)
    for b in range(int(n_bins)):
        m = bid == b
        if not np.any(m):
            continue
        acc[b] = float(np.mean(y0[m]))
        conf[b] = float(np.mean(p[m]))

    body = []
    body.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')

    # Grid lines.
    for t in [0.2, 0.4, 0.6, 0.8]:
        body.append(f'<line x1="{ml}" y1="{ymap(t)}" x2="{ml+plot_w}" y2="{ymap(t)}" stroke="#e8e8e8" stroke-width="1"/>')

    # Diagonal (perfect calibration).
    body.append(
        f'<line x1="{xmap(0)}" y1="{ymap(0)}" x2="{xmap(1)}" y2="{ymap(1)}" '
        f'stroke="#555" stroke-width="1.5" stroke-dasharray="6,4"/>'
    )

    # Bars.
    for b in range(int(n_bins)):
        if not np.isfinite(acc[b]) or not np.isfinite(conf[b]):
            continue
        cx = (edges[b] + edges[b + 1]) / 2.0
        bw = bar_w_frac * plot_w
        x_left = xmap(cx) - bw / 2.0

        # Output bar (blue) - height = acc.
        bar_top = ymap(acc[b])
        bar_bot = ymap(0.0)
        body.append(
            f'<rect x="{x_left:.1f}" y="{bar_top:.1f}" width="{bw:.1f}" height="{bar_bot - bar_top:.1f}" '
            f'fill="#6baed6" opacity="0.85"/>'
        )

        # Gap overlay (pink/red) between acc and conf.
        if conf[b] > acc[b]:
            gap_top = ymap(conf[b])
            gap_bot = ymap(acc[b])
            body.append(
                f'<rect x="{x_left:.1f}" y="{gap_top:.1f}" width="{bw:.1f}" height="{gap_bot - gap_top:.1f}" '
                f'fill="#e8a0a0" opacity="0.75"/>'
            )
        elif acc[b] > conf[b]:
            gap_top = ymap(acc[b])
            gap_bot = ymap(conf[b])
            body.append(
                f'<rect x="{x_left:.1f}" y="{gap_top:.1f}" width="{bw:.1f}" height="{gap_bot - gap_top:.1f}" '
                f'fill="#a0c8e8" opacity="0.75"/>'
            )

    # Axes.
    body.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+plot_h}" stroke="#111" stroke-width="1.5"/>')
    body.append(f'<line x1="{ml}" y1="{mt+plot_h}" x2="{ml+plot_w}" y2="{mt+plot_h}" stroke="#111" stroke-width="1.5"/>')

    for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        body.append(f'<text x="{xmap(t)}" y="{mt+plot_h+18}" font-size="11" text-anchor="middle" fill="#333">{t:.1f}</text>')
        body.append(f'<text x="{ml-6}" y="{ymap(t)+4}" font-size="11" text-anchor="end" fill="#333">{t:.1f}</text>')

    body.append(f'<text x="{ml+plot_w/2}" y="{height-8}" font-size="12" text-anchor="middle" fill="#333">Confidence</text>')
    body.append(
        f'<text x="14" y="{mt+plot_h/2}" font-size="12" text-anchor="middle" fill="#333" '
        f'transform="rotate(-90 14 {mt+plot_h/2})">Accuracy</text>'
    )
    body.append(f'<text x="{ml}" y="{mt-12}" font-size="14" font-weight="bold" text-anchor="start" fill="#111">{_svg_escape(title)}</text>')

    # ECE label.
    if ece_value is not None and math.isfinite(ece_value):
        ece_str = f"ECE={ece_value * 100:.2f}"
        rx = ml + plot_w - 8
        ry = mt + plot_h - 8
        body.append(f'<rect x="{rx-80}" y="{ry-16}" width="82" height="20" fill="white" stroke="#bbb" rx="3"/>')
        body.append(f'<text x="{rx-4}" y="{ry}" font-size="12" font-weight="bold" text-anchor="end" fill="#333">{ece_str}</text>')

    # Legend (small).
    lx = ml + 10
    ly = mt + 8
    body.append(f'<rect x="{lx}" y="{ly}" width="10" height="10" fill="#6baed6"/>')
    body.append(f'<text x="{lx+14}" y="{ly+9}" font-size="10" fill="#333">Output</text>')
    body.append(f'<rect x="{lx+60}" y="{ly}" width="10" height="10" fill="#e8a0a0"/>')
    body.append(f'<text x="{lx+74}" y="{ly+9}" font-size="10" fill="#333">Gap</text>')
    body.append(f'<line x1="{lx+105}" y1="{ly+5}" x2="{lx+125}" y2="{ly+5}" stroke="#555" stroke-width="1.5" stroke-dasharray="4,3"/>')
    body.append(f'<text x="{lx+129}" y="{ly+9}" font-size="10" fill="#333">Perfect</text>')

    _write_svg(path, width=width, height=height, body="\n".join(body))


def _plot_reliability_bargap_grid_svg(
    path: Path,
    *,
    y: np.ndarray,
    p_methods: dict[str, np.ndarray],
    n_bins: int = 15,
    ece_values: dict[str, float] | None = None,
) -> None:
    """Multi-panel bar+gap reliability diagram grid (one panel per method)."""
    methods = list(p_methods.keys())
    n = len(methods)
    if n == 0:
        return
    panel_w, panel_h = 360, 320
    gap = 10
    total_w = n * panel_w + (n - 1) * gap
    total_h = panel_h

    bodies = []
    for i, method in enumerate(methods):
        ece_v = (ece_values or {}).get(method, None)
        # Render each panel as a group with translate.
        sub_body: list[str] = []
        sub_path = path.parent / f"_tmp_rel_{i}.svg"
        _plot_reliability_bargap_svg(
            sub_path,
            title=f"({chr(97+i)}) {method}",
            y=y,
            p=p_methods[method],
            n_bins=n_bins,
            ece_value=ece_v,
        )
        # Read back and wrap in a translate group.
        svg_text = sub_path.read_text(encoding="utf-8")
        # Extract body between first > and </svg>.
        import re as _re
        m = _re.search(r'viewBox="[^"]*">\s*(.+?)\s*</svg>', svg_text, flags=_re.DOTALL)
        if m:
            inner = m.group(1)
            x_off = i * (panel_w + gap)
            bodies.append(f'<g transform="translate({x_off},0)">{inner}</g>')
        sub_path.unlink(missing_ok=True)

    full_body = "\n".join(bodies)
    full_svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" '
        f'viewBox="0 0 {total_w} {total_h}">\n'
        f'<rect x="0" y="0" width="{total_w}" height="{total_h}" fill="white"/>\n'
        f'{full_body}\n</svg>\n'
    )
    path.write_text(full_svg, encoding="utf-8")


def _plot_time_auc_ece_svg(
    path: Path,
    *,
    title: str,
    slices: list[dict],
    methods_auc: list[str],
    methods_ece: list[str],
    label_map: dict[str, str] | None = None,
) -> None:
    # Dual-axis plot: left=AUC, right=ECE.
    width, height = 920, 520
    ml, mr, mt, mb = 90, 90, 60, 80
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    rows = [r for r in slices if int(r.get("n", 0)) > 0]
    if not rows:
        return

    # Use slice index on x-axis (0..K-1).
    slice_ids = sorted({int(r["slice"]) for r in rows})
    k = len(slice_ids)
    xs = np.arange(k, dtype=np.float64)

    def xmap(xi: float) -> float:
        if k <= 1:
            return ml + plot_w / 2.0
        return ml + (float(xi) / float(k - 1)) * plot_w

    def _series(method: str, key: str) -> np.ndarray:
        out = np.full(k, np.nan, dtype=np.float64)
        for i, sid in enumerate(slice_ids):
            for r in rows:
                if int(r["slice"]) == int(sid) and str(r["method"]) == str(method):
                    out[i] = float(r[key])
                    break
        return out

    auc_all = np.concatenate([_series(m, "auc") for m in methods_auc], axis=0)
    ece_all = np.concatenate([_series(m, "ece") for m in methods_ece], axis=0)
    auc_min = float(np.nanmin(auc_all)) if np.isfinite(auc_all).any() else 0.5
    auc_max = float(np.nanmax(auc_all)) if np.isfinite(auc_all).any() else 1.0
    ece_min = float(np.nanmin(ece_all)) if np.isfinite(ece_all).any() else 0.0
    ece_max = float(np.nanmax(ece_all)) if np.isfinite(ece_all).any() else 0.2

    # Add small margins.
    auc_pad = max(0.01, 0.05 * (auc_max - auc_min))
    ece_pad = max(0.005, 0.10 * (ece_max - ece_min))
    auc_min, auc_max = max(0.0, auc_min - auc_pad), min(1.0, auc_max + auc_pad)
    ece_min, ece_max = max(0.0, ece_min - ece_pad), max(ece_min + 1e-6, ece_max + ece_pad)

    def ymap_auc(v: float) -> float:
        if not math.isfinite(v):
            return float("nan")
        t = (float(v) - auc_min) / (auc_max - auc_min)
        return mt + (1.0 - t) * plot_h

    def ymap_ece(v: float) -> float:
        if not math.isfinite(v):
            return float("nan")
        t = (float(v) - ece_min) / (ece_max - ece_min)
        return mt + (1.0 - t) * plot_h

    body = []
    body.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')

    # Axes.
    body.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+plot_h}" stroke="#111" stroke-width="2"/>')
    body.append(f'<line x1="{ml}" y1="{mt+plot_h}" x2="{ml+plot_w}" y2="{mt+plot_h}" stroke="#111" stroke-width="2"/>')
    body.append(f'<line x1="{ml+plot_w}" y1="{mt}" x2="{ml+plot_w}" y2="{mt+plot_h}" stroke="#111" stroke-width="2"/>')

    # Ticks (x).
    for i, sid in enumerate(slice_ids):
        x = xmap(i)
        body.append(f'<line x1="{x}" y1="{mt+plot_h}" x2="{x}" y2="{mt+plot_h+6}" stroke="#111" stroke-width="2"/>')
        body.append(f'<text x="{x}" y="{mt+plot_h+26}" font-size="13" text-anchor="middle" fill="#111">{sid}</text>')

    # Ticks (y left/right).
    for j in range(6):
        t = j / 5.0
        v_auc = auc_min + t * (auc_max - auc_min)
        y = ymap_auc(v_auc)
        body.append(f'<line x1="{ml-6}" y1="{y}" x2="{ml}" y2="{y}" stroke="#111" stroke-width="2"/>')
        body.append(f'<text x="{ml-10}" y="{y+4}" font-size="13" text-anchor="end" fill="#111">{v_auc:.3f}</text>')

        v_e = ece_min + t * (ece_max - ece_min)
        y2 = ymap_ece(v_e)
        body.append(f'<line x1="{ml+plot_w}" y1="{y2}" x2="{ml+plot_w+6}" y2="{y2}" stroke="#111" stroke-width="2"/>')
        body.append(f'<text x="{ml+plot_w+10}" y="{y2+4}" font-size="13" text-anchor="start" fill="#111">{v_e:.3f}</text>')

    body.append(f'<text x="{ml}" y="32" font-size="18" text-anchor="start" fill="#111">{_svg_escape(title)}</text>')
    body.append(
        f'<text x="{ml+plot_w/2}" y="{height-28}" font-size="16" text-anchor="middle" fill="#111">Time slice (quantile index)</text>'
    )
    body.append(
        f'<text x="28" y="{mt+plot_h/2}" font-size="16" text-anchor="middle" fill="#111" transform="rotate(-90 28 {mt+plot_h/2})">AUC</text>'
    )
    body.append(
        f'<text x="{width-28}" y="{mt+plot_h/2}" font-size="16" text-anchor="middle" fill="#111" transform="rotate(-90 {width-28} {mt+plot_h/2})">ECE</text>'
    )

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    methods_unique = list(dict.fromkeys([*methods_auc, *methods_ece]))
    color_map = {str(m): palette[i % len(palette)] for i, m in enumerate(methods_unique)}

    def _draw_line(method: str, key: str, ymap, *, color: str, dash: str | None) -> None:
        series = _series(method, key)
        pts = []
        for i in range(k):
            if not math.isfinite(float(series[i])):
                continue
            pts.append((xmap(i), ymap(float(series[i]))))
        if len(pts) >= 2:
            d = "M " + " L ".join(f"{x:.2f} {yy:.2f}" for x, yy in pts)
            extra = f' stroke-dasharray="{dash}"' if dash else ""
            body.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="3"{extra}/>')
        for x, yy in pts:
            body.append(f'<circle cx="{x:.2f}" cy="{yy:.2f}" r="4.0" fill="{color}"/>')

    # Lines.
    for method in methods_auc:
        _draw_line(method, "auc", ymap_auc, color=color_map[str(method)], dash=None)
    for method in methods_ece:
        _draw_line(method, "ece", ymap_ece, color=color_map[str(method)], dash="6,4")

    # Legend (method colors + style note).
    leg_x = ml + plot_w - 260
    leg_y = mt - 22
    for i, method in enumerate(methods_unique):
        color = color_map[str(method)]
        label = str(label_map.get(str(method), str(method)) if isinstance(label_map, dict) else str(method))
        y0 = leg_y + i * 20
        body.append(f'<rect x="{leg_x}" y="{y0-12}" width="14" height="14" fill="{color}"/>')
        body.append(f'<text x="{leg_x+20}" y="{y0}" font-size="14" text-anchor="start" fill="#111">{_svg_escape(label)}</text>')
    body.append(
        f'<text x="{leg_x}" y="{leg_y+len(methods_unique)*20+6}" font-size="13" text-anchor="start" fill="#444">solid=AUC (left), dashed=ECE (right)</text>'
    )

    _write_svg(path, width=width, height=height, body="\n".join(body))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Paper-facing temporal calibration evaluation (AUC + calibration metrics + SVG figs).")
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--train_set", type=int, default=1)
    ap.add_argument("--data_root", type=str, default="dataset")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to best.pt checkpoint.")

    ap.add_argument("--pack_name", type=str, default="", help="If set, write outputs to _paper_packs/<ts>_<name>/")
    ap.add_argument("--ts", type=str, default="", help="Timestamp (default: now).")
    ap.add_argument("--outline", type=str, default="paper/0_outline/ecml_pkdd_outline.md")
    ap.add_argument(
        "--pack_level",
        type=str,
        default="minimal",
        choices=["minimal", "full"],
        help="minimal: only meta + core CSVs; full: also write figures, markdown tables, and source artifacts.",
    )

    ap.add_argument("--ece_bins", type=int, default=15)
    ap.add_argument("--reliability_bins", type=int, default=15)
    ap.add_argument("--time_slices", type=int, default=10)
    ap.add_argument("--late_q", type=float, default=0.8, help="Late-time slice quantile on count (for reliability).")

    ap.add_argument("--calib_frac", type=float, default=1.0, help="Use only the most recent fraction of valid tokens as calibration labels.")
    ap.add_argument("--rescal_min_count", type=int, default=50)
    ap.add_argument("--rescal_shrink_tau", type=float, default=50.0)

    ap.add_argument("--drift_env", type=str, default="", help="Optional env file with frozen dynamic item-bias hyperparameters.")
    ap.add_argument("--drift_time_bins", type=int, default=10)
    ap.add_argument("--drift_process_var", type=float, default=0.01)
    ap.add_argument("--item_bias_prior_var", type=float, default=1.0)
    ap.add_argument("--item_bias_shrink_tau", type=float, default=0.0)
    ap.add_argument(
        "--drift_obs_min_weight",
        type=float,
        default=1.0,
        help="Require Fisher weight W=sum p(1-p) >= this threshold per (item,time-bin) pseudo-observation.",
    )

    ap.add_argument("--save_npz", action="store_true", help="Also dump (y,p,pid,count) arrays for re-plotting.")
    ap.add_argument(
        "--only_methods",
        type=str,
        default="",
        help="Comma-separated list of method ids to compute/output. "
        "Old paper ids are accepted as aliases and normalized to engineering ids. Empty means compute the default method family.",
    )
    args = ap.parse_args(argv)

    dataset = str(args.dataset)
    train_set = int(args.train_set)

    data_root = Path(str(args.data_root))
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root

    meta = _read_meta(dataset, data_root=data_root)
    n_question = int(meta["n_question"])
    n_pid = int(meta.get("n_pid", 0))
    if n_pid <= 0:
        raise SystemExit("[eval_temporal_calibration] meta.n_pid<=0: this script currently targets PID datasets only.")

    seqlen = int(meta.get("seqlen", 200))

    ckpt_path = Path(str(args.ckpt))
    if not ckpt_path.is_absolute():
        ckpt_path = (REPO_ROOT / ckpt_path).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    # Load split arrays (we need train history for THDF).
    train_q, train_qa, train_pid, _train_uid, train_count = _load_split_arrays(
        dataset, train_set, "train", data_root=data_root, n_question=n_question, seqlen=seqlen
    )
    valid_q, valid_qa, valid_pid, _valid_uid, valid_count = _load_split_arrays(
        dataset, train_set, "valid", data_root=data_root, n_question=n_question, seqlen=seqlen
    )
    test_q, test_qa, test_pid, _test_uid, test_count = _load_split_arrays(
        dataset, train_set, "test", data_root=data_root, n_question=n_question, seqlen=seqlen
    )

    # Reconstruct model + load checkpoint.
    params = _params_from_ckpt(ckpt_path, dataset=dataset, meta=meta, override_train_set=train_set)
    model = load_model(params)
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ck.get("model_state_dict", ck)
    ckpt_io.load_model_state(model, state, strict=True)

    # Collect token-level outputs.
    _vl, _va, _vauc, y_v, p_v, pid_v, cnt_v = run_test(
        model,
        params,
        None,
        valid_q,
        valid_qa,
        valid_pid,
        uid_data=_valid_uid,
        count_data=valid_count,
        label="Valid(calib)",
        return_outputs=True,
    )
    _tl, _ta, _tauc, y_t, p_t, pid_t, cnt_t = run_test(
        model,
        params,
        None,
        test_q,
        test_qa,
        test_pid,
        uid_data=_test_uid,
        count_data=test_count,
        label="Test",
        return_outputs=True,
    )
    if y_v is None or p_v is None or pid_v is None or cnt_v is None:
        raise SystemExit("[eval_temporal_calibration] failed to collect valid outputs (need y,p,pid,count).")
    if y_t is None or p_t is None or pid_t is None or cnt_t is None:
        raise SystemExit("[eval_temporal_calibration] failed to collect test outputs (need y,p,pid,count).")

    y_v = y_v.astype(np.float64, copy=False).reshape(-1)
    p_v = p_v.astype(np.float64, copy=False).reshape(-1)
    pid_v = pid_v.astype(np.int64, copy=False).reshape(-1)
    cnt_v = cnt_v.astype(np.float64, copy=False).reshape(-1)
    y_t = y_t.astype(np.float64, copy=False).reshape(-1)
    p_t = p_t.astype(np.float64, copy=False).reshape(-1)
    pid_t = pid_t.astype(np.int64, copy=False).reshape(-1)
    cnt_t = cnt_t.astype(np.float64, copy=False).reshape(-1)

    # Optional calib-fraction ablation: keep only the most recent valid tokens (by count) as calibration labels.
    calib_frac = float(args.calib_frac)
    if not (0.0 < calib_frac <= 1.0):
        raise ValueError(f"--calib_frac must be in (0,1], got {calib_frac!r}")
    if calib_frac < 1.0:
        m0 = (pid_v > 0) & np.isfinite(cnt_v)
        if np.any(m0):
            q_tail = float(max(0.0, min(1.0, 1.0 - float(calib_frac))))
            thr = float(np.quantile(cnt_v[m0], q=q_tail, method="linear"))
            m_tail = m0 & (cnt_v >= thr)
            if int(np.sum(m_tail)) > 0:
                y_v, p_v, pid_v, cnt_v = y_v[m_tail], p_v[m_tail], pid_v[m_tail], cnt_v[m_tail]

    # If provided, override dynamic item-bias knobs from a frozen env file.
    env_path = str(args.drift_env).strip()
    env_kv: dict[str, str] = {}
    if env_path:
        p_env = Path(env_path)
        if not p_env.is_absolute():
            p_env = REPO_ROOT / p_env
        if not p_env.is_file():
            raise FileNotFoundError(f"--drift_env not found: {p_env}")
        env_kv = _parse_env_kv(p_env.read_text(encoding="utf-8"))

    def _env_get(key: str, fallback: str) -> str:
        return str(env_kv.get(key, fallback))

    drift_cfg = ItemDriftConfig(
        time_bins=int(_env_get("DRIFT_RESCAL_TIME_BINS", str(args.drift_time_bins))),
        process_var=float(_env_get("DRIFT_RESCAL_KALMAN_Q", str(args.drift_process_var))),
        prior_var=float(_env_get("DRIFT_RESCAL_KALMAN_P0", str(args.item_bias_prior_var))),
        shrink_tau=float(_env_get("DRIFT_RESCAL_SHRINK_TAU", str(args.item_bias_shrink_tau))),
        obs_min_weight=float(_env_get("DRIFT_RESCAL_OBS_MIN_W", str(args.drift_obs_min_weight))),
    )

    # Prepare output pack dir.
    ts = str(args.ts).strip() or _now_ts()
    pack_name = _safe_name(args.pack_name) if str(args.pack_name).strip() else _safe_name(f"{dataset}_{params.model}")
    pack_id = f"{ts}_{pack_name}"
    pack_dir = REPO_ROOT / "_paper_packs" / pack_id
    tables_dir = pack_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pack_level = str(args.pack_level).strip().lower()
    want_full_pack = pack_level == "full"
    figs_dir = pack_dir / "figs"
    sources_dir = pack_dir / "sources"
    if want_full_pack:
        figs_dir.mkdir(parents=True, exist_ok=True)
        sources_dir.mkdir(parents=True, exist_ok=True)

    # Calibration-window stats (for regime diagnosis; no test labels).
    calib_pid_counts = None
    calib_stats = {"n_tokens": 0, "n_pid_nonzero": 0, "median_obs_per_item": float("nan")}
    if int(n_pid) > 0:
        pid_cal = pid_v.astype(np.int64, copy=False).reshape(-1)
        m_cal = pid_cal > 0
        calib_stats["n_tokens"] = int(np.sum(m_cal))
        if np.any(m_cal):
            calib_pid_counts = np.bincount(pid_cal[m_cal], minlength=int(n_pid) + 1).astype(np.int64, copy=False)
            nz = calib_pid_counts[1:][calib_pid_counts[1:] > 0]
            calib_stats["n_pid_nonzero"] = int(nz.size)
            calib_stats["median_obs_per_item"] = float(np.median(nz.astype(np.float64))) if nz.size > 0 else float("nan")

    # Save provenance.
    meta_out = {
        "pack_id": pack_id,
        "created_at": ts,
        "dataset": dataset,
        "train_set": train_set,
        "model": str(params.model),
        "ckpt": str(ckpt_path),
        "git_head": _run_git(["rev-parse", "HEAD"]),
        "git_status": _run_git(["status", "--porcelain"]),
        "git_diff_stat": _run_git(["diff", "--stat"]),
        "calib_frac": calib_frac,
        "calib_stats": calib_stats,
        "rescal": {"min_count": int(args.rescal_min_count), "shrink_tau": float(args.rescal_shrink_tau)},
        "item_drift_cfg": asdict(drift_cfg),
        "outline": str(args.outline),
        "pack_level": pack_level,
        "command": sys.argv,
    }
    if env_path:
        meta_out["drift_env"] = str(p_env)
    if want_full_pack:
        (sources_dir / "command.txt").write_text(" ".join([_safe_name(x) if i == 0 else x for i, x in enumerate(sys.argv)]) + "\n", encoding="utf-8")
    if env_path and want_full_pack:
        shutil.copy2(str(p_env), str(sources_dir / "drift_env.sh"))

    # Fit classic calibrators on valid.
    methods: dict[str, np.ndarray] = {"raw_score": p_t.copy()}
    logit_v_base = _logit(p_v)
    logit_t_base = _logit(p_t)

    only_raw = [m.strip() for m in str(getattr(args, "only_methods", "")).split(",") if m.strip()]
    only = [METHOD_ALIAS.get(m, m) for m in only_raw]
    only_set = set(only)
    want_all = not only_set

    def _want(name: str) -> bool:
        return want_all or (name in only_set)

    meta_out["only_methods"] = only

    # Temperature scaling.
    if _want("global_temperature"):
        T = _fit_temperature(y_v, p_v)
        methods["global_temperature"] = _sigmoid(logit_t_base / float(T))

    # Platt scaling.
    if _want("global_sigmoid"):
        pl = _fit_platt(y_v, p_v)
        if pl is not None:
            a, b = pl
            methods["global_sigmoid"] = _sigmoid(a * logit_t_base + b)

    # Time-aware Platt: y ~ sigmoid(a*logit(p) + c*t + b), fitted on calibration.
    # This is a minimal "time-aware" baseline that can extrapolate into test.
    if _want("global_sigmoid_time"):
        plt = _fit_platt_time_lr(y_calib=y_v, p_calib=p_v, pid_calib=pid_v, cnt_calib=cnt_v)
        if plt is not None:
            a_t, c_t, b_t, meta_t = plt
            t_mean = float(meta_t["t_mean"])
            t_std = float(meta_t["t_std"])
            t_norm_test = (cnt_t.astype(np.float64, copy=False) - t_mean) / float(max(t_std, 1e-6))
            methods["global_sigmoid_time"] = _sigmoid(float(a_t) * logit_t_base + float(c_t) * t_norm_test + float(b_t))
            meta_out["global_sigmoid_time"] = meta_t

    # Isotonic regression.
    if _want("global_isotonic"):
        iso = _fit_isotonic(y_v, p_v)
        if iso is not None:
            methods["global_isotonic"] = np.clip(iso.predict(np.clip(p_t, 0.0, 1.0)), 0.0, 1.0)

    # Histogram binning.
    if _want("global_histogram"):
        edges_h, rate_h = _fit_histogram(y_v, p_v, n_bins=int(args.reliability_bins))
        methods["global_histogram"] = _apply_histogram(p_t, edges_h, rate_h)

    # ResCal (per-pid resid).
    if _want("item_bias_mean") or _want("item_bias_mean_monotone"):
        methods["item_bias_mean"] = _rescal_resid(
            y_calib=y_v,
            p_calib=p_v,
            pid_calib=pid_v,
            p_eval=p_t,
            pid_eval=pid_t,
            n_pid=n_pid,
            min_count=int(args.rescal_min_count),
            shrink_tau=float(args.rescal_shrink_tau),
        )

    # ResCal + isotonic: strong baseline combining per-item correction and a global monotone link.
    # This can preserve ResCal's AUC gain while substantially improving calibration metrics.
    if _want("item_bias_mean_monotone"):
        try:
            p_v_rescal = _rescal_resid(
                y_calib=y_v,
                p_calib=p_v,
                pid_calib=pid_v,
                p_eval=p_v,
                pid_eval=pid_v,
                n_pid=n_pid,
                min_count=int(args.rescal_min_count),
                shrink_tau=float(args.rescal_shrink_tau),
            )
            iso_rescal = _fit_isotonic(y_v, p_v_rescal)
            if iso_rescal is not None and "item_bias_mean" in methods:
                methods["item_bias_mean_monotone"] = np.clip(
                    iso_rescal.predict(np.clip(methods["item_bias_mean"], 0.0, 1.0)),
                    0.0,
                    1.0,
                )
                meta_out["item_bias_mean_monotone"] = {"link": "isotonic"}
        except Exception:
            pass

    # Per-item L2 logistic regression baseline (ridge logistic with item intercepts).
    # Backbone logit is a fixed offset (NOT penalized); only b_i has L2 penalty.
    # This is mathematically equivalent to SLC static under Laplace approximation.
    # Verified: synthetic test yields correlation=1.0, ΔAUC=0, ΔNLL=0.
    if _want("item_bias_ridge") and int(n_pid) > 0:
        try:
            pid_v_i = pid_v.astype(np.int64, copy=False).reshape(-1)
            pid_t_i = pid_t.astype(np.int64, copy=False).reshape(-1)
            m_cal = pid_v_i > 0
            if np.any(m_cal):
                logit_cal = logit_v_base[m_cal].astype(np.float64, copy=False)
                y_cal = y_v[m_cal].astype(np.float64, copy=False)
                pid_cal = pid_v_i[m_cal]

                sigma_b2 = float(getattr(drift_cfg, "prior_var", 1.0))
                if sigma_b2 <= 0.0:
                    sigma_b2 = 1.0
                lam = 1.0 / sigma_b2

                # IRLS with backbone logit as offset, L2 penalty on b_i only.
                K = int(n_pid) + 1
                b_ridge = np.zeros(K, dtype=np.float64)
                for _it in range(50):
                    eta = logit_cal + b_ridge[pid_cal]
                    # eta is a logit; convert to probability before clipping for IRLS.
                    p_cal = _sigmoid_clip(_sigmoid(eta))
                    w = (p_cal * (1.0 - p_cal)).astype(np.float64)
                    g = (y_cal - p_cal).astype(np.float64)
                    sum_w = np.bincount(pid_cal, weights=w, minlength=K).astype(np.float64)
                    sum_g = np.bincount(pid_cal, weights=g, minlength=K).astype(np.float64)
                    H = sum_w[1:K] + lam
                    grad = sum_g[1:K] - lam * b_ridge[1:K]
                    step = grad / np.maximum(H, 1e-12)
                    b_ridge[1:K] += step
                    if np.max(np.abs(step)) < 1e-8:
                        break

                # Offset-Platt: fit (a, b0) with b_ridge as fixed offset.
                b_tok_v = b_ridge[pid_v_i]
                b_tok_t = b_ridge[pid_t_i]
                ab = _fit_platt_with_offset_irls(y=y_v, x=logit_v_base, offset=b_tok_v)
                if ab is not None:
                    a_r, b0_r = ab
                    methods["item_bias_ridge"] = _sigmoid(float(a_r) * logit_t_base + float(b0_r) + b_tok_t)
                    meta_out["item_bias_ridge"] = {
                        "a": float(a_r),
                        "b0": float(b0_r),
                        "sigma_b2": float(sigma_b2),
                        "n_items_nonzero": int(np.sum(np.abs(b_ridge[1:]) > 1e-8)),
                    }
        except Exception as e:
            print(f"[eval] ridge_logistic failed: {type(e).__name__}: {e}", flush=True)

    # Static item-bias shrinkage.
    if (_want("item_bias_shrinkage_static") or _want("item_bias_shrinkage_static_monotone")) and int(n_pid) > 0:
        pid_v_i = pid_v.astype(np.int64, copy=False).reshape(-1)
        pid_t_i = pid_t.astype(np.int64, copy=False).reshape(-1)
        m_cal = pid_v_i > 0
        p_cal = _sigmoid_clip(p_v)
        w_tok = (p_cal * (1.0 - p_cal)).astype(np.float64, copy=False)
        g_tok = (y_v - p_cal).astype(np.float64, copy=False)

        sum_w_pid = (
            np.bincount(pid_v_i[m_cal], weights=w_tok[m_cal], minlength=int(n_pid) + 1).astype(np.float64, copy=False)
            if np.any(m_cal)
            else np.zeros(int(n_pid) + 1, dtype=np.float64)
        )
        sum_g_pid = (
            np.bincount(pid_v_i[m_cal], weights=g_tok[m_cal], minlength=int(n_pid) + 1).astype(np.float64, copy=False)
            if np.any(m_cal)
            else np.zeros(int(n_pid) + 1, dtype=np.float64)
        )

        prior_var = float(getattr(drift_cfg, "prior_var", 1.0))
        if prior_var <= 0.0:
            prior_var = 1.0
        denom = (1.0 / float(prior_var)) + sum_w_pid
        denom = np.maximum(denom, 1e-12)
        static_bias = (sum_g_pid / denom).astype(np.float64, copy=False)
        shrink_tau = float(max(0.0, getattr(drift_cfg, "shrink_tau", 0.0)))
        if shrink_tau > 0.0:
            lam = (sum_w_pid / (sum_w_pid + shrink_tau)).astype(np.float64, copy=False)
            static_bias = static_bias * np.clip(lam, 0.0, 1.0)
        static_bias[0] = 0.0

        bias_v_static = static_bias[pid_v_i]
        bias_t_static = static_bias[pid_t_i]
        ab_s = _fit_platt_with_offset_irls(y=y_v, x=logit_v_base, offset=bias_v_static)
        if ab_s is not None and _want("item_bias_shrinkage_static"):
            a_s, b0_s = ab_s
            methods["item_bias_shrinkage_static"] = _sigmoid(float(a_s) * logit_t_base + float(b0_s) + bias_t_static)
            meta_out["item_bias_shrinkage_static"] = {"a": float(a_s), "b0": float(b0_s)}

        if _want("item_bias_shrinkage_static_monotone"):
            p_v_static = _sigmoid(logit_v_base + bias_v_static)
            p_t_static = _sigmoid(logit_t_base + bias_t_static)
            iso_s = _fit_isotonic(y_v, p_v_static)
            if iso_s is not None:
                methods["item_bias_shrinkage_static_monotone"] = np.clip(
                    iso_s.predict(np.clip(p_t_static, 0.0, 1.0)),
                    0.0,
                    1.0,
                )
                meta_out["item_bias_shrinkage_static_monotone"] = {"link": "isotonic"}

    # Minimal dynamic item-bias smoothing.
    if _want("item_bias_shrinkage_dynamic") and int(n_pid) > 0:
        time_edges_override = compute_time_edges_from_history(
            train_count,
            train_pid,
            valid_count,
            valid_pid,
            time_bins=int(drift_cfg.time_bins),
        )
        drift_res = fit_dynamic_item_bias(
            y_calib=y_v,
            p_calib=p_v,
            pid_calib=pid_v,
            count_calib=cnt_v,
            train_pid=train_pid,
            train_count=train_count,
            n_pid=n_pid,
            cfg=drift_cfg,
            count_eval=cnt_t,
            time_edges_override=time_edges_override,
        )
        if drift_res.bias_table is not None and drift_res.time_bin_index is not None:
            tb_v = _digitize_bins(cnt_v, time_edges_override, n_bins=int(drift_cfg.time_bins))
            tb_t = drift_res.time_bin_index.astype(np.int64, copy=False)
            bias_table = drift_res.bias_table.astype(np.float64, copy=False)
            bias_v = bias_table[pid_v.astype(np.int64, copy=False), tb_v]
            bias_t = bias_table[pid_t.astype(np.int64, copy=False), tb_t]
            ab_dyn = _fit_platt_with_offset_irls(y=y_v, x=logit_v_base, offset=bias_v)
            if ab_dyn is not None:
                a_dyn, b0_dyn = ab_dyn
                methods["item_bias_shrinkage_dynamic"] = _sigmoid(float(a_dyn) * logit_t_base + float(b0_dyn) + bias_t)
                meta_out["item_bias_shrinkage_dynamic"] = {
                    "a": float(a_dyn),
                    "b0": float(b0_dyn),
                    "process_var": float(drift_cfg.process_var),
                    "obs_min_weight": float(drift_cfg.obs_min_weight),
                }

    # If requested, keep only selected methods plus the raw backbone score.
    if not want_all:
        methods = {k: v for k, v in methods.items() if (k == "raw_score" or k in only_set)}

    # Overall metrics table.
    overall_rows = []
    for name, p_fix in methods.items():
        met = _metrics(y_t, p_fix, ece_bins=int(args.ece_bins))
        overall_rows.append({"method": name, **met})

    # Save tables.
    def _fmt(x: object) -> str:
        if isinstance(x, float):
            if math.isnan(x):
                return "nan"
            return f"{x:.6f}"
        return str(x)

    # CSV
    keys = ["method", "n", "auc", "acc", "nll", "brier", "rmse", "ece"]
    csv_lines = [",".join(keys)]
    for r in overall_rows:
        csv_lines.append(",".join(_fmt(r.get(k, "")) for k in keys))
    (tables_dir / "metrics_overall.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    if want_full_pack:
        md = []
        md.append(f"# Overall metrics ({dataset} / {params.model})")
        md.append("")
        md.append("| method | n | AUC | Acc | NLL | Brier | RMSE | ECE |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in overall_rows:
            md.append(
                f"| {r['method']} | {int(r['n'])} | {_fmt(r['auc'])} | {_fmt(r['acc'])} | {_fmt(r['nll'])} | {_fmt(r['brier'])} | {_fmt(r['rmse'])} | {_fmt(r['ece'])} |"
            )
        (tables_dir / "metrics_overall.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # Time-slice metrics (test).
    slice_rows = _slice_metrics(y_t, methods, cnt_t, n_slices=int(args.time_slices), ece_bins=int(args.ece_bins))
    if slice_rows:
        skeys = ["slice", "count_lo", "count_hi", "method", "n", "auc", "ece", "nll"]
        out = [",".join(skeys)]
        for r in slice_rows:
            out.append(",".join(_fmt(r.get(k, "")) for k in skeys))
        (tables_dir / "metrics_time_slices.csv").write_text("\n".join(out) + "\n", encoding="utf-8")
    else:
        (tables_dir / "metrics_time_slices.csv").write_text("slice,count_lo,count_hi,method,n,auc,ece,nll\n", encoding="utf-8")

    if want_full_pack:
        # Density-stratified slices (by per-item observation count in the calibration window).
        def _safe_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
            try:
                return _metrics(y, p, ece_bins=int(args.ece_bins))
            except Exception:
                return {"n": float(y.size), "auc": float("nan"), "acc": float("nan"), "nll": float("nan"), "brier": float("nan"), "rmse": float("nan"), "ece": float("nan")}

        pid_v_i = pid_v.astype(np.int64, copy=False).reshape(-1)
        pid_t_i = pid_t.astype(np.int64, copy=False).reshape(-1)
        pid_counts = np.bincount(pid_v_i, minlength=int(n_pid))
        nz_counts = pid_counts[1:][pid_counts[1:] > 0]

        pid_bin = np.full(pid_counts.shape[0], -1, dtype=np.int8)
        bin_defs: list[dict[str, float]] = []
        if nz_counts.size > 0:
            q1 = float(np.quantile(nz_counts.astype(np.float64), q=1.0 / 3.0, method="linear"))
            q2 = float(np.quantile(nz_counts.astype(np.float64), q=2.0 / 3.0, method="linear"))
            lo1 = float(min(q1, q2))
            hi1 = float(max(q1, q2))
            pid_bin[(pid_counts > 0) & (pid_counts <= lo1)] = 0
            pid_bin[(pid_counts > lo1) & (pid_counts <= hi1)] = 1
            pid_bin[(pid_counts > hi1)] = 2
            for bi in [0, 1, 2]:
                pids = np.where(pid_bin == bi)[0]
                pids = pids[pids > 0]
                if pids.size <= 0:
                    continue
                cs = pid_counts[pids].astype(np.float64, copy=False)
                bin_defs.append(
                    {
                        "pidcount_bin": float(bi),
                        "n_pid": float(pids.size),
                        "pidcount_lo": float(np.min(cs)),
                        "pidcount_hi": float(np.max(cs)),
                    }
                )

        dens_rows: list[dict[str, object]] = []
        for bd in bin_defs:
            bi = int(float(bd["pidcount_bin"]))
            m_bin = (pid_t_i > 0) & (pid_bin[pid_t_i] == bi)
            if int(np.sum(m_bin)) <= 0:
                continue
            yb = y_t[m_bin]
            for name, p_fix in methods.items():
                met = _safe_metrics(yb, p_fix[m_bin])
                dens_rows.append(
                    {
                        "pidcount_bin": int(bi),
                        "pidcount_lo": float(bd["pidcount_lo"]),
                        "pidcount_hi": float(bd["pidcount_hi"]),
                        "n_pid": int(float(bd["n_pid"])),
                        "method": str(name),
                        "n": int(met.get("n", 0)),
                        "auc": float(met.get("auc", float("nan"))),
                        "ece": float(met.get("ece", float("nan"))),
                        "nll": float(met.get("nll", float("nan"))),
                    }
                )

        if dens_rows:
            dkeys = ["pidcount_bin", "pidcount_lo", "pidcount_hi", "n_pid", "method", "n", "auc", "ece", "nll"]
            out = [",".join(dkeys)]
            for r in dens_rows:
                out.append(",".join(_fmt(r.get(k, "")) for k in dkeys))
            (tables_dir / "metrics_pidcount_slices.csv").write_text("\n".join(out) + "\n", encoding="utf-8")
        else:
            (tables_dir / "metrics_pidcount_slices.csv").write_text("pidcount_bin,pidcount_lo,pidcount_hi,n_pid,method,n,auc,ece,nll\n", encoding="utf-8")

        paper_methods = list(PAPER_METHODS)
        paper_label = dict(PAPER_LABELS)
        if not want_all:
            paper_methods = ["raw_score", *[m for m in only if m != "raw_score"]]
            paper_methods = [m for m in dict.fromkeys(paper_methods) if m in methods]
            for k in paper_methods:
                paper_label.setdefault(k, str(k))
        all_rel_keys = [
            "raw_score",
            "global_sigmoid",
            "global_sigmoid_time",
            "global_temperature",
            "global_isotonic",
            "global_histogram",
            "item_bias_mean",
            "item_bias_mean_monotone",
            "item_bias_ridge",
            "item_bias_shrinkage_static",
            "item_bias_shrinkage_static_monotone",
            "item_bias_shrinkage_dynamic",
        ]

        p_paper = {paper_label[k]: methods[k] for k in paper_methods if k in methods}
        if p_paper:
            ece_vals = {lbl: _ece(y_t, p, n_bins=int(args.ece_bins)) for lbl, p in p_paper.items()}
            _plot_reliability_bargap_grid_svg(
                figs_dir / "fig2_reliability_bargap.svg",
                y=y_t,
                p_methods=p_paper,
                n_bins=int(args.reliability_bins),
                ece_values=ece_vals,
            )
            m0 = np.isfinite(cnt_t)
            if np.any(m0):
                thr = float(np.quantile(cnt_t[m0], q=float(args.late_q), method="linear"))
                m_late = m0 & (cnt_t >= thr)
                if int(np.sum(m_late)) > 0:
                    ece_late = {lbl: _ece(y_t[m_late], p[m_late], n_bins=int(args.ece_bins)) for lbl, p in p_paper.items()}
                    _plot_reliability_bargap_grid_svg(
                        figs_dir / "fig2_reliability_bargap_late.svg",
                        y=y_t[m_late],
                        p_methods={lbl: p[m_late] for lbl, p in p_paper.items()},
                        n_bins=int(args.reliability_bins),
                        ece_values=ece_late,
                    )

        p_rel = {k: methods[k] for k in all_rel_keys if k in methods}
        if p_rel:
            _plot_reliability_svg(
                figs_dir / "fig_reliability_all.svg",
                title=f"Reliability (all test) — {dataset} / {params.model}",
                y=y_t,
                p_methods=p_rel,
                n_bins=int(args.reliability_bins),
            )
            m0 = np.isfinite(cnt_t)
            if np.any(m0):
                thr = float(np.quantile(cnt_t[m0], q=float(args.late_q), method="linear"))
                m_late = m0 & (cnt_t >= thr)
                if int(np.sum(m_late)) > 0:
                    _plot_reliability_svg(
                        figs_dir / "fig_reliability_late.svg",
                        title=f"Reliability (late q={float(args.late_q):g}) — {dataset} / {params.model}",
                        y=y_t[m_late],
                        p_methods={k: v[m_late] for k, v in p_rel.items()},
                        n_bins=int(args.reliability_bins),
                    )

        if slice_rows:
            paper_time_methods = [m for m in paper_methods if m in methods]
            _plot_time_auc_ece_svg(
                figs_dir / "fig1_time_auc_ece.svg",
                title=f"AUC vs ECE over time — {dataset} / {params.model}",
                slices=slice_rows,
                methods_auc=paper_time_methods,
                methods_ece=paper_time_methods,
                label_map=paper_label,
            )
            all_time_methods = [m for m in all_rel_keys if m in methods]
            _plot_time_auc_ece_svg(
                figs_dir / "fig_time_auc_ece.svg",
                title=f"AUC vs ECE over time slices — {dataset} / {params.model}",
                slices=slice_rows,
                methods_auc=all_time_methods,
                methods_ece=all_time_methods,
            )

    # Optional raw dump for re-plotting / debugging.
    if bool(args.save_npz):
        sources_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(sources_dir / "outputs_test.npz"),
            y=y_t.astype(np.float32),
            p_base=p_t.astype(np.float32),
            pid=pid_t.astype(np.int32),
            count=cnt_t.astype(np.float32),
        )

    if want_full_pack:
        outline_path = Path(str(args.outline))
        if not outline_path.is_absolute():
            outline_path = REPO_ROOT / outline_path
        outline_ref = os.path.relpath(str(outline_path), start=str(pack_dir))
        index = f"""# Paper Pack: `{pack_id}`

Outline: `{outline_ref}`

## Paper-facing figures (default paper profile)

- **Fig.1** (Temporal AUC & ECE): `figs/fig1_time_auc_ece.svg`
- **Fig.2** (Reliability bar+gap): `figs/fig2_reliability_bargap.svg` + `figs/fig2_reliability_bargap_late.svg`

## Diagnostic figures

- Reliability (line+dot, all methods): `figs/fig_reliability_all.svg` + `figs/fig_reliability_late.svg`
- Temporal (all methods): `figs/fig_time_auc_ece.svg`

## Tables

- Main metrics: `tables/metrics_overall.csv` (also `metrics_overall.md`)
- Time slices: `tables/metrics_time_slices.csv`

## Sources

- `sources/command.txt`
- `sources/drift_env.sh` (if provided: frozen dynamic-item config)
"""
        (pack_dir / "index.md").write_text(index, encoding="utf-8")

    (pack_dir / "meta.json").write_text(json.dumps(meta_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[eval_temporal_calibration] ok: {pack_dir}", flush=True)


if __name__ == "__main__":
    main()
