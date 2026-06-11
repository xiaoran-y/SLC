from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ItemDriftConfig:
    time_bins: int = 10
    process_var: float = 0.01
    prior_var: float = 1.0
    shrink_tau: float = 0.0
    obs_min_weight: float = 0.0


@dataclass
class ItemDriftResult:
    bias_table: np.ndarray
    time_edges: np.ndarray
    time_bin_index: np.ndarray | None
    calibration_end: float
    dynamic_bias_var: np.ndarray | None = None
    dynamic_bias: np.ndarray | None = None
    static_bias: np.ndarray | None = None
    static_bias_var: np.ndarray | None = None


def fit_dynamic_item_bias(
    *,
    y_calib: np.ndarray,
    p_calib: np.ndarray,
    pid_calib: np.ndarray,
    count_calib: np.ndarray,
    train_pid,
    train_count,
    n_pid: int,
    cfg: ItemDriftConfig,
    count_eval: np.ndarray | None = None,
    time_edges_override: np.ndarray | None = None,
) -> ItemDriftResult:
    time_bins = int(cfg.time_bins)
    if time_bins < 2:
        time_bins = 10
    process_var = float(max(0.0, cfg.process_var))
    prior_var = float(cfg.prior_var)
    if prior_var <= 0.0:
        prior_var = 1.0

    y = _to_np(y_calib).astype(np.float64, copy=False).reshape(-1)
    p = _sigmoid_clip(_to_np(p_calib).astype(np.float64, copy=False).reshape(-1))
    pid = _to_np(pid_calib).astype(np.int64, copy=False).reshape(-1)
    cnt = _to_np(count_calib).astype(np.float64, copy=False).reshape(-1)
    calibration_end = float(np.max(cnt)) if cnt.size > 0 else 0.0

    if time_edges_override is not None:
        time_edges = np.asarray(time_edges_override, dtype=np.float64).reshape(-1)
        if time_edges.size != int(time_bins) + 1:
            raise ValueError(f"time_edges_override must have shape [{int(time_bins) + 1}], got {time_edges.shape}")
        time_edges = np.maximum.accumulate(time_edges)
        if float(time_edges[-1]) <= float(time_edges[0]):
            time_edges[-1] = float(time_edges[0] + 1e-6)
    else:
        time_edges = _time_edges_from_history(train_count, train_pid, cnt, pid, n_bins=time_bins)
        if time_edges is None:
            raise RuntimeError("failed to build time edges from history (no pid>0 tokens)")

    tb_calib = digitize_time_bins(cnt, time_edges, time_bins=time_bins)
    tb_eval = None
    if count_eval is not None:
        ce = _to_np(count_eval).astype(np.float64, copy=False).reshape(-1)
        tb_eval = digitize_time_bins(ce, time_edges, time_bins=time_bins)

    w_tok = (p * (1.0 - p)).astype(np.float64, copy=False)
    g_tok = (y - p).astype(np.float64, copy=False)

    idx = (pid * int(time_bins) + tb_calib).astype(np.int64, copy=False)
    size = (int(n_pid) + 1) * int(time_bins)
    sum_g = np.bincount(idx, weights=g_tok, minlength=size).astype(np.float64, copy=False).reshape(int(n_pid) + 1, time_bins)
    sum_w = np.bincount(idx, weights=w_tok, minlength=size).astype(np.float64, copy=False).reshape(int(n_pid) + 1, time_bins)

    obs_min_weight = float(max(0.0, getattr(cfg, "obs_min_weight", 0.0)))
    w_th = float(max(1e-9, obs_min_weight))
    has = sum_w >= w_th
    has[0, :] = False

    z = np.zeros_like(sum_g, dtype=np.float64)
    r = np.zeros_like(sum_w, dtype=np.float64)
    z[has] = sum_g[has] / sum_w[has]
    r[has] = 1.0 / sum_w[has]

    n_eff = sum_w.sum(axis=1).astype(np.float64, copy=False)
    sum_g_pid = sum_g.sum(axis=1).astype(np.float64, copy=False)
    if n_eff.shape[0] > 0:
        n_eff[0] = 0.0
        sum_g_pid[0] = 0.0
    denom = (1.0 / float(prior_var)) + n_eff
    denom = np.maximum(denom, 1e-12)
    static_bias = (sum_g_pid / denom).astype(np.float64, copy=False)
    static_bias_var = (1.0 / denom).astype(np.float64, copy=False)
    static_bias[0] = 0.0

    z_dyn = z - static_bias.reshape(-1, 1)
    if process_var <= 0.0:
        dynamic_bias = np.zeros_like(z_dyn, dtype=np.float64)
        dynamic_bias_var = np.zeros_like(z_dyn, dtype=np.float64)
    else:
        dynamic_bias, dynamic_bias_var = _kalman_smooth_rows_with_var(z_dyn, r, has, qv=process_var, p0=prior_var)

    bias_table = static_bias.reshape(-1, 1) + dynamic_bias
    bias_table[0, :] = 0.0

    shrink_tau = float(max(0.0, getattr(cfg, "shrink_tau", 0.0)))
    if shrink_tau > 0.0:
        lam = (n_eff / (n_eff + shrink_tau)).astype(np.float64, copy=False)
        lam = np.clip(lam, 0.0, 1.0)
        if lam.shape[0] > 0:
            lam[0] = 0.0
        static_bias = static_bias * lam
        static_bias_var = static_bias_var * (lam**2)
        dynamic_bias = dynamic_bias * lam.reshape(-1, 1)
        dynamic_bias_var = dynamic_bias_var * (lam.reshape(-1, 1) ** 2)
        bias_table = bias_table * lam.reshape(-1, 1)
        bias_table[0, :] = 0.0

    return ItemDriftResult(
        bias_table=bias_table.astype(np.float64, copy=False),
        time_edges=time_edges.astype(np.float64, copy=False),
        time_bin_index=tb_eval,
        calibration_end=calibration_end,
        dynamic_bias_var=dynamic_bias_var.astype(np.float64, copy=False),
        dynamic_bias=dynamic_bias.astype(np.float64, copy=False),
        static_bias=static_bias.astype(np.float64, copy=False),
        static_bias_var=static_bias_var.astype(np.float64, copy=False),
    )


def _to_np(a) -> np.ndarray:
    return a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)


def _sigmoid_clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p.astype(np.float64, copy=False), 1e-6, 1.0 - 1e-6)


def _time_edges_from_history(
    train_count,
    train_pid,
    calib_count: np.ndarray,
    calib_pid: np.ndarray,
    *,
    n_bins: int,
) -> np.ndarray | None:
    tc_np = _to_np(train_count).astype(np.float64, copy=False).reshape(-1)
    tp_np = _to_np(train_pid).astype(np.int64, copy=False).reshape(-1)
    cc_np = _to_np(calib_count).astype(np.float64, copy=False).reshape(-1)
    cp_np = _to_np(calib_pid).astype(np.int64, copy=False).reshape(-1)

    time_parts = []
    m_tr = tp_np > 0
    if np.any(m_tr):
        time_parts.append(tc_np[m_tr].astype(np.float64, copy=False))
    m_cal = cp_np > 0
    if np.any(m_cal):
        time_parts.append(cc_np[m_cal].astype(np.float64, copy=False))
    if not time_parts:
        return None

    t = np.concatenate(time_parts, axis=0).astype(np.float64, copy=False)
    if t.size <= 0:
        return None
    edges = np.quantile(t, q=np.linspace(0.0, 1.0, int(n_bins) + 1), method="linear").astype(np.float64)
    edges = np.maximum.accumulate(edges)
    edges[0] = float(np.min(t))
    edges[-1] = float(np.max(t) + 1e-6)
    return edges


def compute_time_edges_from_history(
    train_count,
    train_pid,
    calib_count,
    calib_pid,
    *,
    time_bins: int,
) -> np.ndarray:
    edges = _time_edges_from_history(
        train_count,
        train_pid,
        _to_np(calib_count).astype(np.float64, copy=False),
        _to_np(calib_pid).astype(np.int64, copy=False),
        n_bins=int(time_bins),
    )
    if edges is None:
        raise RuntimeError("failed to build time edges from history (no pid>0 tokens)")
    return edges


def digitize_time_bins(count: np.ndarray, time_edges: np.ndarray, *, time_bins: int) -> np.ndarray:
    tb = np.digitize(count.astype(np.float64, copy=False), time_edges, right=False) - 1
    tb = np.clip(tb, 0, int(time_bins) - 1).astype(np.int64, copy=False)
    return tb


def _kalman_smooth_1d_with_var(
    z: np.ndarray,
    r: np.ndarray,
    has: np.ndarray,
    *,
    qv: float,
    p0: float,
) -> tuple[np.ndarray, np.ndarray]:
    steps = int(z.shape[0])
    x_f = np.zeros(steps, dtype=np.float64)
    p_f = np.zeros(steps, dtype=np.float64)
    x_pred = np.zeros(steps, dtype=np.float64)
    p_pred = np.zeros(steps, dtype=np.float64)
    x = 0.0
    p = float(p0)
    qv = float(max(0.0, qv))

    for t in range(steps):
        x_prior = x
        p_prior = p + qv
        x_pred[t] = x_prior
        p_pred[t] = p_prior
        if bool(has[t]):
            rt = float(r[t])
            denom = p_prior + rt
            gain = 0.0 if denom <= 0.0 else (p_prior / denom)
            x = x_prior + gain * (float(z[t]) - x_prior)
            p = (1.0 - gain) * p_prior
        else:
            x = x_prior
            p = p_prior
        x_f[t] = x
        p_f[t] = p

    x_s = x_f.copy()
    p_s = p_f.copy()
    for t in range(steps - 2, -1, -1):
        denom = float(p_pred[t + 1])
        if denom <= 0.0:
            continue
        j = float(p_f[t]) / denom
        x_s[t] = x_f[t] + j * (x_s[t + 1] - x_pred[t + 1])
        p_s[t] = p_f[t] + (j * j) * (p_s[t + 1] - p_pred[t + 1])
    return x_s, p_s


def _kalman_smooth_rows_with_var(
    z2: np.ndarray,
    r2: np.ndarray,
    has2: np.ndarray,
    *,
    qv: float,
    p0: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_out = np.zeros_like(z2, dtype=np.float64)
    p_out = np.zeros_like(z2, dtype=np.float64)
    rows = np.flatnonzero(np.any(has2, axis=1)).astype(np.int64, copy=False)
    for i in rows:
        ii = int(i)
        x_i, p_i = _kalman_smooth_1d_with_var(z2[ii, :], r2[ii, :], has2[ii, :], qv=qv, p0=p0)
        x_out[ii, :] = x_i
        p_out[ii, :] = p_i
    return x_out, p_out
