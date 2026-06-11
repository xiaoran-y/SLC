#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def _sigmoid_clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _sigmoid_clip(p)
    return np.log(p / (1.0 - p))


def _nll(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = _sigmoid_clip(p).reshape(-1)
    return float((-y * np.log(p) - (1.0 - y) * np.log(1.0 - p)).mean())


def _ece(y: np.ndarray, p: np.ndarray, *, n_bins: int = 15) -> float:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if y.size <= 0:
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


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if y.size <= 0 or np.all(y == y[0]):
        return float("nan")
    try:
        from sklearn import metrics  # type: ignore

        return float(metrics.roc_auc_score(y, p))
    except Exception:
        return float("nan")


def _fit_platt(y: np.ndarray, logit_x: np.ndarray) -> tuple[float, float] | None:
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    x = np.asarray(logit_x, dtype=np.float64).reshape(-1, 1)
    if y.size <= 0 or np.all(y == y[0]):
        return None
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore

        lr = LogisticRegression(solver="lbfgs", max_iter=300)
        lr.fit(x, y)
        a = float(lr.coef_.reshape(-1)[0])
        b = float(lr.intercept_.reshape(-1)[0])
        return float(a), float(b)
    except Exception:
        return None


def _fit_platt_time_lr(y: np.ndarray, logit_x: np.ndarray, t: np.ndarray) -> tuple[float, float, float, dict] | None:
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    x = np.asarray(logit_x, dtype=np.float64).reshape(-1)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    m = np.isfinite(t)
    if not np.any(m):
        return None
    y = y[m]
    if y.size <= 0 or np.all(y == y[0]):
        return None
    x = x[m]
    t = t[m]
    t_mean = float(np.mean(t))
    t_std = float(np.std(t))
    if not math.isfinite(t_std) or t_std <= 1e-12:
        return None
    t_norm = (t - t_mean) / t_std
    X = np.stack([x, t_norm], axis=1)
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
    """Fit (a,b) in: y ~ sigmoid(a*x + b + offset)."""
    y01 = np.asarray(y, dtype=np.float64).reshape(-1)
    x1 = np.asarray(x, dtype=np.float64).reshape(-1)
    off = np.asarray(offset, dtype=np.float64).reshape(-1)
    if y01.size <= 0 or np.all(y01 == y01[0]):
        return None
    if x1.size != y01.size or off.size != y01.size:
        raise ValueError("shape mismatch")

    a = 1.0
    b = 0.0

    def _ll(a0: float, b0: float) -> float:
        eta = a0 * x1 + b0 + off
        return float(np.sum(y01 * eta - np.logaddexp(0.0, eta)))

    ll0 = _ll(a, b)
    for _it in range(int(max_iter)):
        eta = a * x1 + b + off
        eta_clip = np.clip(eta, -50.0, 50.0)
        p = 1.0 / (1.0 + np.exp(-eta_clip))
        w = np.clip(p * (1.0 - p), 1e-9, None)
        r = y01 - p

        g1 = float(np.sum(r * x1))
        g2 = float(np.sum(r))

        A11 = float(np.sum(w * x1 * x1) + float(ridge))
        A12 = float(np.sum(w * x1))
        A22 = float(np.sum(w) + float(ridge))
        det = A11 * A22 - A12 * A12

        if not (math.isfinite(det) and det > 0.0):
            da = 0.0
            db = g2 / A22 if A22 > 0 else 0.0
        else:
            da = (g1 * A22 - g2 * A12) / det
            db = (g2 * A11 - g1 * A12) / det

        if max(abs(float(da)), abs(float(db))) < float(tol):
            break

        step = 1.0
        improved = False
        for _ls in range(12):
            a_new = float(a + step * float(da))
            b_new = float(b + step * float(db))
            ll_new = _ll(a_new, b_new)
            if ll_new >= ll0 - 1e-12:
                a, b = a_new, b_new
                ll0 = ll_new
                improved = True
                break
            step *= 0.5
        if not improved:
            break

    return float(a), float(b)


def _kalman_smooth_1d(z: np.ndarray, r: np.ndarray, has: np.ndarray, *, qv: float, p0: float) -> np.ndarray:
    T = int(z.shape[0])
    x_f = np.zeros(T, dtype=np.float64)
    P_f = np.zeros(T, dtype=np.float64)
    x_pred = np.zeros(T, dtype=np.float64)
    P_pred = np.zeros(T, dtype=np.float64)
    x = 0.0
    P = float(p0)
    qv = float(max(0.0, qv))
    for t in range(T):
        xp = x
        Pp = P + qv
        x_pred[t] = xp
        P_pred[t] = Pp
        if bool(has[t]):
            rt = float(r[t])
            denom = Pp + rt
            K = 0.0 if denom <= 0.0 else (Pp / denom)
            x = xp + K * (float(z[t]) - xp)
            P = (1.0 - K) * Pp
        else:
            x = xp
            P = Pp
        x_f[t] = x
        P_f[t] = P
    # RTS smoother
    x_s = x_f.copy()
    P_s = P_f.copy()
    for t in range(T - 2, -1, -1):
        denom = float(P_pred[t + 1])
        if denom <= 0.0:
            continue
        J = float(P_f[t]) / denom
        x_s[t] = x_f[t] + J * (x_s[t + 1] - x_pred[t + 1])
        P_s[t] = P_f[t] + (J * J) * (P_s[t + 1] - P_pred[t + 1])
    _ = P_s
    return x_s


def _kalman_smooth_rows(z: np.ndarray, r: np.ndarray, has: np.ndarray, *, qv: float, p0: float) -> np.ndarray:
    z2 = np.asarray(z, dtype=np.float64)
    r2 = np.asarray(r, dtype=np.float64)
    has2 = np.asarray(has, dtype=bool)
    if z2.shape != r2.shape or z2.shape != has2.shape or z2.ndim != 2:
        raise ValueError("shape mismatch for row smoother")
    n, T = int(z2.shape[0]), int(z2.shape[1])
    out = np.zeros((n, T), dtype=np.float64)
    for i in range(n):
        out[i, :] = _kalman_smooth_1d(z2[i, :], r2[i, :], has2[i, :], qv=qv, p0=p0)
    return out


def _fit_flat_kalman_bias_table(
    *,
    y: np.ndarray,
    p: np.ndarray,
    pid: np.ndarray,
    tbin: np.ndarray,
    n_pid: int,
    time_bins: int,
    qv: float,
    p0: float = 1.0,
    w_min: float = 0.0,
) -> np.ndarray:
    """Flat per-item SSM on logit residuals (no global/concept layers)."""
    y1 = np.asarray(y, dtype=np.float64).reshape(-1)
    p1 = _sigmoid_clip(np.asarray(p, dtype=np.float64).reshape(-1))
    pid1 = np.asarray(pid, dtype=np.int64).reshape(-1)
    tb1 = np.asarray(tbin, dtype=np.int64).reshape(-1)
    if not (y1.size == p1.size == pid1.size == tb1.size):
        raise ValueError("shape mismatch")

    w_tok = (p1 * (1.0 - p1)).astype(np.float64, copy=False)
    g_tok = (y1 - p1).astype(np.float64, copy=False)
    eps_w = max(1e-9, float(w_min))

    idx = (pid1 * int(time_bins) + tb1).astype(np.int64, copy=False)
    size = (int(n_pid) + 1) * int(time_bins)
    sum_g = np.bincount(idx, weights=g_tok, minlength=size).astype(np.float64, copy=False).reshape(int(n_pid) + 1, int(time_bins))
    sum_w = np.bincount(idx, weights=w_tok, minlength=size).astype(np.float64, copy=False).reshape(int(n_pid) + 1, int(time_bins))

    has = sum_w >= eps_w
    z = np.zeros_like(sum_g, dtype=np.float64)
    r = np.zeros_like(sum_w, dtype=np.float64)
    z[has] = sum_g[has] / sum_w[has]
    r[has] = 1.0 / sum_w[has]

    b_table = _kalman_smooth_rows(z, r, has, qv=float(qv), p0=float(p0))
    b_table[0, :] = 0.0
    return b_table.astype(np.float64, copy=False)


@dataclass(frozen=True)
class SynthCfg:
    n_users: int
    n_items: int
    n_concepts: int
    time_bins: int
    obs_per_item: int
    drift_q: float
    alpha: float
    b0: float
    noise: float
    gen: str  # 1pl|2pl
    ece_bins: int
    solver_iters: int
    w_min: float


def _simulate(cfg: SynthCfg, *, seed: int) -> list[dict]:
    rng = np.random.RandomState(int(seed))

    N = int(cfg.n_users)
    K = int(cfg.n_items)
    T = int(cfg.time_bins)
    obs = int(cfg.obs_per_item)
    Q = float(cfg.drift_q)

    # Time split by bins.
    t1 = int(round(0.6 * T))
    t2 = int(round(0.8 * T))
    t1 = max(1, min(t1, T - 2))
    t2 = max(t1 + 1, min(t2, T - 1))
    calib_bins = np.arange(t1, t2, dtype=np.int64)
    test_bins = np.arange(t2, T, dtype=np.int64)

    theta = rng.randn(N).astype(np.float64)  # ability
    b0_item = rng.randn(K).astype(np.float64)  # initial difficulty
    # Drifted difficulty: random walk with increment variance Q.
    inc = rng.randn(K, T).astype(np.float64) * math.sqrt(max(Q, 0.0))
    inc[:, 0] = 0.0
    b_item_t = b0_item.reshape(-1, 1) + np.cumsum(inc, axis=1)

    a_item = np.ones(K, dtype=np.float64)
    if str(cfg.gen).strip().lower() == "2pl":
        a_item = np.clip(1.0 + 0.3 * rng.randn(K).astype(np.float64), 0.2, 3.0)

    # True drift correction (to map from b(0) to b(t)): beta(i,t) = b(0) - b(t).
    beta_true = (b0_item.reshape(-1, 1) - b_item_t).astype(np.float64, copy=False)

    y_cal: list[np.ndarray] = []
    logit0_cal: list[np.ndarray] = []
    pid_cal: list[np.ndarray] = []
    t_cal: list[np.ndarray] = []
    eta_true_test: list[np.ndarray] = []
    y_test: list[np.ndarray] = []
    logit0_test: list[np.ndarray] = []
    pid_test: list[np.ndarray] = []
    t_test: list[np.ndarray] = []

    for t in range(T):
        # interactions for this time bin
        pid0 = np.repeat(np.arange(K, dtype=np.int64), obs)
        uid0 = rng.randint(0, N, size=pid0.size).astype(np.int64, copy=False)
        eta_true = (a_item[pid0] * theta[uid0] - b_item_t[pid0, int(t)]).astype(np.float64, copy=False)
        p_true = _sigmoid(eta_true)
        y = (rng.rand(pid0.size) < p_true).astype(np.float64, copy=False)

        # Frozen backbone: uses difficulty at t=0 with affine distortion + optional noise.
        eta0 = (cfg.alpha * (theta[uid0] - b0_item[pid0]) + cfg.b0).astype(np.float64, copy=False)
        if float(cfg.noise) > 0.0:
            eta0 = eta0 + float(cfg.noise) * rng.randn(pid0.size).astype(np.float64)

        if int(t) in calib_bins:
            y_cal.append(y)
            logit0_cal.append(eta0)
            pid_cal.append(pid0 + 1)  # 1-indexed like KT
            t_cal.append(np.full(pid0.size, int(t), dtype=np.int64))
        if int(t) in test_bins:
            eta_true_test.append(eta_true)
            y_test.append(y)
            logit0_test.append(eta0)
            pid_test.append(pid0 + 1)
            t_test.append(np.full(pid0.size, int(t), dtype=np.int64))

    y_cal_np = np.concatenate(y_cal, axis=0).astype(np.float64, copy=False)
    logit0_cal_np = np.concatenate(logit0_cal, axis=0).astype(np.float64, copy=False)
    pid_cal_np = np.concatenate(pid_cal, axis=0).astype(np.int64, copy=False)
    t_cal_np = np.concatenate(t_cal, axis=0).astype(np.int64, copy=False)

    y_test_np = np.concatenate(y_test, axis=0).astype(np.float64, copy=False)
    logit0_test_np = np.concatenate(logit0_test, axis=0).astype(np.float64, copy=False)
    pid_test_np = np.concatenate(pid_test, axis=0).astype(np.int64, copy=False)
    t_test_np = np.concatenate(t_test, axis=0).astype(np.int64, copy=False)
    eta_true_test_np = np.concatenate(eta_true_test, axis=0).astype(np.float64, copy=False)

    p_base_test = _sigmoid(logit0_test_np)
    p_oracle_test = _sigmoid(eta_true_test_np)

    rows: list[dict] = []

    def _row(method: str, *, p_test: np.ndarray, mse_cal: float | None = None, mse_test: float | None = None) -> dict:
        out = {
            "seed": int(seed),
            "gen": str(cfg.gen),
            "Q": float(cfg.drift_q),
            "obs_per_item": int(cfg.obs_per_item),
            "method": str(method),
            "auc": float(_auc(y_test_np, p_test)),
            "ece": float(_ece(y_test_np, p_test, n_bins=int(cfg.ece_bins))),
            "nll": float(_nll(y_test_np, p_test)),
            "regret_auc": float(_auc(y_test_np, p_oracle_test) - _auc(y_test_np, p_test)),
            "mse_beta_calib": float(mse_cal) if mse_cal is not None else float("nan"),
            "mse_beta_test": float(mse_test) if mse_test is not None else float("nan"),
        }
        return out

    rows.append(_row("oracle", p_test=p_oracle_test))
    rows.append(_row("raw_score", p_test=p_base_test))

    # Platt.
    ab = _fit_platt(y_cal_np, logit0_cal_np)
    if ab is None:
        return rows
    a_pl, b_pl = ab
    p_platt_test = _sigmoid(a_pl * logit0_test_np + b_pl)
    rows.append(_row("global_sigmoid", p_test=p_platt_test))

    # Time-aware Platt.
    abt = _fit_platt_time_lr(y_cal_np, logit0_cal_np, t_cal_np.astype(np.float64))
    if abt is not None:
        a_t, c_t, b_t, meta_t = abt
        t_norm_test = (t_test_np.astype(np.float64) - float(meta_t["t_mean"])) / float(max(float(meta_t["t_std"]), 1e-6))
        rows.append(_row("global_sigmoid_time", p_test=_sigmoid(a_t * logit0_test_np + c_t * t_norm_test + b_t)))

    # Static / dynamic item-bias shrinkage.
    def _run_item_bias(*, qv: float, name: str) -> None:
        a0, b0 = float(a_pl), float(b_pl)
        b_table0 = np.zeros((K + 1, T), dtype=np.float64)
        for _it in range(int(max(1, cfg.solver_iters))):
            off_cal0 = b_table0[pid_cal_np, t_cal_np]
            p_adj0 = _sigmoid(a0 * logit0_cal_np + b0 + off_cal0)
            b_table0 = _fit_flat_kalman_bias_table(
                y=y_cal_np,
                p=p_adj0,
                pid=pid_cal_np,
                tbin=t_cal_np,
                n_pid=int(K),
                time_bins=int(T),
                qv=float(qv),
                p0=1.0,
                w_min=float(cfg.w_min),
            )
            off_cal0 = b_table0[pid_cal_np, t_cal_np]
            ab0 = _fit_platt_with_offset_irls(y=y_cal_np, x=logit0_cal_np, offset=off_cal0)
            if ab0 is None:
                break
            a0, b0 = float(ab0[0]), float(ab0[1])

        off_test0 = b_table0[pid_test_np, t_test_np]
        eta0 = a0 * logit0_test_np + b0 + off_test0
        mse_cal0 = float(np.mean((b_table0[1:, calib_bins] - beta_true[:, calib_bins]) ** 2))
        mse_te0 = float(np.mean((b_table0[1:, test_bins] - beta_true[:, test_bins]) ** 2))
        rows.append(_row(name, p_test=_sigmoid(eta0), mse_cal=mse_cal0, mse_test=mse_te0))

    _run_item_bias(qv=0.0, name="item_bias_shrinkage_static")
    _run_item_bias(qv=float(Q), name="item_bias_shrinkage_dynamic")
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def _svg_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = [int(max(0, min(255, int(x)))) for x in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


def _lerp(a: float, b: float, t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return float(a + (b - a) * t)


def _lerp_rgb(c0: tuple[int, int, int], c1: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return (
        int(round(_lerp(c0[0], c1[0], t))),
        int(round(_lerp(c0[1], c1[1], t))),
        int(round(_lerp(c0[2], c1[2], t))),
    )


def _heatmap_diverging_color(v: float, *, vmax: float) -> str:
    if not math.isfinite(v):
        return "#ddd"
    vmax = float(max(1e-12, vmax))
    white = (255, 255, 255)
    red = (220, 38, 38)
    blue = (37, 99, 235)
    if v >= 0.0:
        t = float(np.clip(v / vmax, 0.0, 1.0))
        return _rgb_to_hex(_lerp_rgb(white, red, t))
    t = float(np.clip((-v) / vmax, 0.0, 1.0))
    return _rgb_to_hex(_lerp_rgb(white, blue, t))


def _heatmap_seq_color(v: float, *, vmin: float, vmax: float) -> str:
    if not math.isfinite(v):
        return "#ddd"
    lo = float(vmin)
    hi = float(max(vmin + 1e-12, vmax))
    t = float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))
    white = (255, 255, 255)
    red = (220, 38, 38)
    return _rgb_to_hex(_lerp_rgb(white, red, t))


def _write_heatmap_svg(
    path: Path,
    *,
    title: str,
    x_labels: list[str],
    y_labels: list[str],
    values: np.ndarray,
    color_fn,
    legend: str,
) -> None:
    values = np.asarray(values, dtype=np.float64)
    n_y, n_x = int(values.shape[0]), int(values.shape[1])
    cell = 42
    lm, tm = 140, 70
    w = lm + n_x * cell + 40
    h = tm + n_y * cell + 60

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="#fff"/>')
    parts.append(f'<text x="{lm}" y="32" font-size="18" font-family="monospace" fill="#111">{_svg_escape(title)}</text>')
    parts.append(f'<text x="{lm}" y="54" font-size="13" font-family="monospace" fill="#444">{_svg_escape(legend)}</text>')

    # grid
    for yi in range(n_y):
        y0 = tm + yi * cell
        lbl = y_labels[yi] if yi < len(y_labels) else str(yi)
        parts.append(f'<text x="{lm-8}" y="{y0+26}" font-size="13" text-anchor="end" font-family="monospace" fill="#111">{_svg_escape(lbl)}</text>')
        for xi in range(n_x):
            x0 = lm + xi * cell
            v = float(values[yi, xi])
            col = str(color_fn(v))
            parts.append(f'<rect x="{x0}" y="{y0}" width="{cell}" height="{cell}" fill="{col}" stroke="#eee"/>')
            if math.isfinite(v):
                parts.append(
                    f'<text x="{x0+cell/2}" y="{y0+26}" font-size="12" text-anchor="middle" font-family="monospace" fill="#111">{_svg_escape(f"{v:.3f}")}</text>'
                )

    for xi in range(n_x):
        x0 = lm + xi * cell
        lbl = x_labels[xi] if xi < len(x_labels) else str(xi)
        parts.append(
            f'<text x="{x0+cell/2}" y="{tm+n_y*cell+22}" font-size="13" text-anchor="middle" font-family="monospace" fill="#111">{_svg_escape(lbl)}</text>'
        )
    parts.append(f'<text x="{lm}" y="{tm+n_y*cell+44}" font-size="13" font-family="monospace" fill="#444">x=obs/item</text>')
    parts.append(f'<text x="18" y="{tm+16}" font-size="13" font-family="monospace" fill="#444">y=Q</text>')

    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthetic regime-map experiment for drift-aware post-hoc calibration.")
    ap.add_argument("--out_csv", type=str, default="_paper_packs/synthetic_regime_map.csv")
    ap.add_argument("--out_dir", type=str, default="_paper_packs/synthetic_figs")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--gen", type=str, default="1pl", choices=["1pl", "2pl"])
    ap.add_argument("--n_users", type=int, default=1000)
    ap.add_argument("--n_items", type=int, default=200)
    ap.add_argument("--n_concepts", type=int, default=20)
    ap.add_argument("--time_bins", type=int, default=20)
    ap.add_argument("--Q_grid", type=str, default="0.0,0.005,0.02,0.05,0.1,0.2")
    ap.add_argument("--obs_grid", type=str, default="5,10,30,100,300")
    ap.add_argument("--alpha", type=float, default=0.8, help="Affine distortion on backbone logit.")
    ap.add_argument("--b0", type=float, default=0.1, help="Backbone logit intercept.")
    ap.add_argument("--noise", type=float, default=0.0, help="Optional Gaussian noise added to backbone logit.")
    ap.add_argument("--ece_bins", type=int, default=15)
    ap.add_argument("--solver_iters", type=int, default=3)
    ap.add_argument("--w_min", type=float, default=0.0, help="Minimum Fisher weight per (item,time) cell to accept a pseudo-observation.")
    args = ap.parse_args()

    seeds = [int(x) for x in str(args.seeds).replace(" ", "").split(",") if x.strip()]
    Q_grid = [float(x) for x in str(args.Q_grid).replace(" ", "").split(",") if x.strip()]
    obs_grid = [int(x) for x in str(args.obs_grid).replace(" ", "").split(",") if x.strip()]

    all_rows: list[dict] = []
    for sd in seeds:
        for Q in Q_grid:
            for obs in obs_grid:
                cfg = SynthCfg(
                    n_users=int(args.n_users),
                    n_items=int(args.n_items),
                    n_concepts=int(args.n_concepts),
                    time_bins=int(args.time_bins),
                    obs_per_item=int(obs),
                    drift_q=float(Q),
                    alpha=float(args.alpha),
                    b0=float(args.b0),
                    noise=float(args.noise),
                    gen=str(args.gen),
                    ece_bins=int(args.ece_bins),
                    solver_iters=int(args.solver_iters),
                    w_min=float(args.w_min),
                )
                all_rows.extend(_simulate(cfg, seed=int(sd)))

    out_csv = Path(str(args.out_csv))
    if not out_csv.is_absolute():
        out_csv = (REPO_ROOT / out_csv).resolve()
    _write_csv(out_csv, all_rows)

    # Aggregate over seeds for regime-map figures.
    out_dir = Path(str(args.out_dir))
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()

    cfg_keys = sorted({(str(r.get("gen", "")), float(r["Q"]), int(r["obs_per_item"])) for r in all_rows})
    Qs = sorted({q for _g, q, _o in cfg_keys})
    Os = sorted({o for _g, _q, o in cfg_keys})
    if Qs and Os:
        # mean delta AUC (static item-bias - global sigmoid)
        mat = np.full((len(Qs), len(Os)), np.nan, dtype=np.float64)
        # mean delta AUC (dynamic item-bias - static item-bias)
        mat_td = np.full_like(mat, np.nan)
        # MSE(β̂,β*) on calibration window (dynamic item-bias)
        mse_mat = np.full_like(mat, np.nan)
        for yi, Q in enumerate(Qs):
            for xi, O in enumerate(Os):
                auc_s = [
                    float(r["auc"])
                    for r in all_rows
                    if float(r.get("Q", float("nan"))) == Q
                    and int(r.get("obs_per_item", -1)) == O
                    and str(r.get("method")) == "item_bias_shrinkage_static"
                ]
                auc_t = [
                    float(r["auc"])
                    for r in all_rows
                    if float(r.get("Q", float("nan"))) == Q
                    and int(r.get("obs_per_item", -1)) == O
                    and str(r.get("method")) == "item_bias_shrinkage_dynamic"
                ]
                auc_p = [
                    float(r["auc"])
                    for r in all_rows
                    if float(r.get("Q", float("nan"))) == Q
                    and int(r.get("obs_per_item", -1)) == O
                    and str(r.get("method")) == "global_sigmoid"
                ]
                mse_t = [
                    float(r.get("mse_beta_calib", float("nan")))
                    for r in all_rows
                    if float(r.get("Q", float("nan"))) == Q
                    and int(r.get("obs_per_item", -1)) == O
                    and str(r.get("method")) == "item_bias_shrinkage_dynamic"
                ]
                if auc_s and auc_p:
                    mat[yi, xi] = float(np.nanmean(auc_s) - np.nanmean(auc_p))
                if auc_t and auc_s:
                    mat_td[yi, xi] = float(np.nanmean(auc_t) - np.nanmean(auc_s))
                if mse_t:
                    mse_mat[yi, xi] = float(np.nanmean(mse_t))

        vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
        _write_heatmap_svg(
            out_dir / "regime_delta_auc_static_minus_global_sigmoid.svg",
            title="Synthetic regime map: ΔAUC (static item-bias − global sigmoid)",
            x_labels=[str(o) for o in Os],
            y_labels=[str(q) for q in Qs],
            values=mat,
            color_fn=lambda v: _heatmap_diverging_color(float(v), vmax=vmax),
            legend="mean over seeds",
        )
        vmax_td = float(np.nanmax(np.abs(mat_td))) if np.isfinite(mat_td).any() else 1.0
        _write_heatmap_svg(
            out_dir / "regime_delta_auc_dynamic_minus_static.svg",
            title="Synthetic regime map: ΔAUC (dynamic item-bias − static item-bias)",
            x_labels=[str(o) for o in Os],
            y_labels=[str(q) for q in Qs],
            values=mat_td,
            color_fn=lambda v: _heatmap_diverging_color(float(v), vmax=vmax_td),
            legend="mean over seeds",
        )
        vmin = float(np.nanmin(mse_mat)) if np.isfinite(mse_mat).any() else 0.0
        vmax2 = float(np.nanmax(mse_mat)) if np.isfinite(mse_mat).any() else 1.0
        _write_heatmap_svg(
            out_dir / "regime_mse_beta_calib_dynamic.svg",
            title="Synthetic regime map: MSE(β̂,β*) on calibration window (dynamic item-bias)",
            x_labels=[str(o) for o in Os],
            y_labels=[str(q) for q in Qs],
            values=mse_mat,
            color_fn=lambda v: _heatmap_seq_color(float(v), vmin=vmin, vmax=vmax2),
            legend="mean over seeds",
        )

    print(f"[ok] wrote {out_csv} (rows={len(all_rows)})", flush=True)
    print(f"[ok] figs: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
