#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from posthoc import ItemDriftConfig, fit_dynamic_item_bias  # noqa: E402
import eval_temporal_calibration as etc  # noqa: E402


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(str(path)) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _fixed_time_edges(cnt_calib: np.ndarray, cnt_test: np.ndarray, *, time_bins: int) -> tuple[np.ndarray, float, float]:
    all_cnt = np.concatenate([cnt_calib.reshape(-1), cnt_test.reshape(-1)], axis=0).astype(np.float64, copy=False)
    c_min = float(np.nanmin(all_cnt)) if all_cnt.size > 0 else 0.0
    c_max = float(np.nanmax(all_cnt)) if all_cnt.size > 0 else 1.0
    lo = float(math.floor(c_min))
    hi = float(math.ceil(c_max) + 1.0)
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, int(time_bins) + 1, dtype=np.float64)
    edges = np.maximum.accumulate(edges)
    edges[-1] = float(max(edges[-1], hi + 1e-6))
    return edges, lo, hi


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Paper-ready eval pack from logits npz (non-KT datasets).")
    ap.add_argument("--calib_npz", type=str, required=True)
    ap.add_argument("--test_npz", type=str, required=True)
    ap.add_argument("--dataset", type=str, default="flight_delay_route_month")
    ap.add_argument("--model", type=str, default="flight_bbA")
    ap.add_argument("--pack_name", type=str, default="")
    ap.add_argument("--ts", type=str, default="")
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
    ap.add_argument("--late_q", type=float, default=0.8)
    ap.add_argument("--calib_frac", type=float, default=1.0)
    ap.add_argument("--rescal_min_count", type=int, default=50)
    ap.add_argument("--rescal_shrink_tau", type=float, default=50.0)
    ap.add_argument("--drift_time_bins", type=int, default=24)
    ap.add_argument("--drift_process_var", type=float, default=0.01)
    ap.add_argument("--item_bias_prior_var", type=float, default=1.0)
    ap.add_argument("--item_bias_shrink_tau", type=float, default=0.0)
    ap.add_argument("--drift_obs_min_weight", type=float, default=1.0)
    args = ap.parse_args(argv)

    calib_path = Path(str(args.calib_npz))
    if not calib_path.is_absolute():
        calib_path = (REPO_ROOT / calib_path).resolve()
    test_path = Path(str(args.test_npz))
    if not test_path.is_absolute():
        test_path = (REPO_ROOT / test_path).resolve()
    if not calib_path.is_file():
        raise FileNotFoundError(f"calib_npz not found: {calib_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"test_npz not found: {test_path}")

    c = _load_npz(calib_path)
    t = _load_npz(test_path)

    def _need(d: dict[str, np.ndarray], key: str) -> np.ndarray:
        if key not in d:
            raise KeyError(f"missing key={key!r} in npz")
        return np.asarray(d[key])

    y_v = _need(c, "y").astype(np.float64, copy=False).reshape(-1)
    logit_v_base = _need(c, "logit").astype(np.float64, copy=False).reshape(-1)
    pid_v = _need(c, "pid").astype(np.int64, copy=False).reshape(-1)
    cnt_v = _need(c, "count").astype(np.float64, copy=False).reshape(-1)
    y_t = _need(t, "y").astype(np.float64, copy=False).reshape(-1)
    logit_t_base = _need(t, "logit").astype(np.float64, copy=False).reshape(-1)
    pid_t = _need(t, "pid").astype(np.int64, copy=False).reshape(-1)
    cnt_t = _need(t, "count").astype(np.float64, copy=False).reshape(-1)

    if y_v.size <= 0:
        raise ValueError(f"empty calib split in: {calib_path}")
    if y_t.size <= 0:
        raise ValueError(f"empty test split in: {test_path}")

    n_pid = int(0)
    if "n_pid" in c:
        try:
            n_pid = int(np.asarray(c["n_pid"]).reshape(-1)[0])
        except Exception:
            n_pid = 0
    if n_pid <= 0:
        n_pid = int(max(int(pid_v.max(initial=0)), int(pid_t.max(initial=0))))

    calib_frac = float(args.calib_frac)
    if not (0.0 < calib_frac <= 1.0):
        raise ValueError(f"--calib_frac must be in (0,1], got {calib_frac!r}")
    if calib_frac < 1.0:
        m0 = (pid_v > 0) & np.isfinite(cnt_v)
        if np.any(m0):
            q_tail = float(max(0.0, min(1.0, 1.0 - calib_frac)))
            thr = float(np.quantile(cnt_v[m0], q=q_tail, method="linear"))
            m_tail = m0 & (cnt_v >= thr)
            if int(np.sum(m_tail)) > 0:
                y_v, logit_v_base, pid_v, cnt_v = y_v[m_tail], logit_v_base[m_tail], pid_v[m_tail], cnt_v[m_tail]

    p_v = etc._sigmoid(logit_v_base)
    p_t = etc._sigmoid(logit_t_base)

    ts = str(args.ts).strip() or etc._now_ts()
    pack_name = etc._safe_name(args.pack_name) if str(args.pack_name).strip() else etc._safe_name(f"{args.dataset}_{args.model}")
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

    calib_stats = {"n_tokens": 0, "n_pid_nonzero": 0, "median_obs_per_item": float("nan")}
    pid_cal = pid_v.astype(np.int64, copy=False).reshape(-1)
    m_cal = pid_cal > 0
    calib_stats["n_tokens"] = int(np.sum(m_cal))
    pid_counts = np.zeros(int(n_pid) + 1, dtype=np.int64)
    if np.any(m_cal) and int(n_pid) > 0:
        pid_counts = np.bincount(pid_cal[m_cal], minlength=int(n_pid) + 1).astype(np.int64, copy=False)
        nz = pid_counts[1:][pid_counts[1:] > 0]
        calib_stats["n_pid_nonzero"] = int(nz.size)
        calib_stats["median_obs_per_item"] = float(np.median(nz.astype(np.float64))) if nz.size > 0 else float("nan")

    drift_cfg = ItemDriftConfig(
        time_bins=int(args.drift_time_bins),
        process_var=float(args.drift_process_var),
        prior_var=float(args.item_bias_prior_var),
        shrink_tau=float(args.item_bias_shrink_tau),
        obs_min_weight=float(args.drift_obs_min_weight),
    )
    time_edges_override, lo, hi = _fixed_time_edges(cnt_v, cnt_t, time_bins=int(drift_cfg.time_bins))

    meta_out = {
        "pack_id": pack_id,
        "created_at": ts,
        "dataset": str(args.dataset),
        "train_set": 1,
        "model": str(args.model),
        "sources": {"calib_npz": str(calib_path), "test_npz": str(test_path)},
        "git_head": etc._run_git(["rev-parse", "HEAD"]),
        "git_status": etc._run_git(["status", "--porcelain"]),
        "git_diff_stat": etc._run_git(["diff", "--stat"]),
        "calib_frac": float(calib_frac),
        "calib_stats": calib_stats,
        "rescal": {"min_count": int(args.rescal_min_count), "shrink_tau": float(args.rescal_shrink_tau)},
        "item_drift_cfg": asdict(drift_cfg),
        "time_edges_override": {
            "time_bins": int(drift_cfg.time_bins),
            "lo": float(lo),
            "hi": float(hi),
        },
        "outline": str(args.outline),
        "pack_level": pack_level,
        "command": sys.argv,
    }

    methods: dict[str, np.ndarray] = {"raw_score": p_t.copy()}

    T = etc._fit_temperature(y_v, p_v)
    methods["global_temperature"] = etc._sigmoid(logit_t_base / float(T))

    pl = etc._fit_platt(y_v, p_v)
    if pl is not None:
        a, b = pl
        methods["global_sigmoid"] = etc._sigmoid(float(a) * logit_t_base + float(b))

    plt = etc._fit_platt_time_lr(y_calib=y_v, p_calib=p_v, pid_calib=pid_v, cnt_calib=cnt_v)
    if plt is not None:
        a_t, c_t, b_t, meta_t = plt
        t_mean = float(meta_t["t_mean"])
        t_std = float(meta_t["t_std"])
        t_norm_test = (cnt_t.astype(np.float64, copy=False) - t_mean) / float(max(t_std, 1e-6))
        methods["global_sigmoid_time"] = etc._sigmoid(float(a_t) * logit_t_base + float(c_t) * t_norm_test + float(b_t))
        meta_out["global_sigmoid_time"] = meta_t

    iso = etc._fit_isotonic(y_v, p_v)
    if iso is not None:
        methods["global_isotonic"] = np.clip(iso.predict(np.clip(p_t, 0.0, 1.0)), 0.0, 1.0)

    edges_h, rate_h = etc._fit_histogram(y_v, p_v, n_bins=int(args.reliability_bins))
    methods["global_histogram"] = etc._apply_histogram(p_t, edges_h, rate_h)

    methods["item_bias_mean"] = etc._rescal_resid(
        y_calib=y_v,
        p_calib=p_v,
        pid_calib=pid_v,
        p_eval=p_t,
        pid_eval=pid_t,
        n_pid=n_pid,
        min_count=int(args.rescal_min_count),
        shrink_tau=float(args.rescal_shrink_tau),
    )

    try:
        p_v_rescal = etc._rescal_resid(
            y_calib=y_v,
            p_calib=p_v,
            pid_calib=pid_v,
            p_eval=p_v,
            pid_eval=pid_v,
            n_pid=n_pid,
            min_count=int(args.rescal_min_count),
            shrink_tau=float(args.rescal_shrink_tau),
        )
        iso_rescal = etc._fit_isotonic(y_v, p_v_rescal)
        if iso_rescal is not None:
            methods["item_bias_mean_monotone"] = np.clip(
                iso_rescal.predict(np.clip(methods["item_bias_mean"], 0.0, 1.0)),
                0.0,
                1.0,
            )
            meta_out["item_bias_mean_monotone"] = {"link": "isotonic"}
    except Exception:
        pass

    if int(n_pid) > 0:
        p_cal = etc._sigmoid(logit_v_base)
        w_tok = (p_cal * (1.0 - p_cal)).astype(np.float64, copy=False)
        g_tok = (y_v - p_cal).astype(np.float64, copy=False)
        m_item = pid_v > 0
        sum_w_pid = np.bincount(pid_v[m_item], weights=w_tok[m_item], minlength=int(n_pid) + 1).astype(np.float64, copy=False)
        sum_g_pid = np.bincount(pid_v[m_item], weights=g_tok[m_item], minlength=int(n_pid) + 1).astype(np.float64, copy=False)
        prior_var = float(max(1e-12, drift_cfg.prior_var))
        denom = np.maximum((1.0 / prior_var) + sum_w_pid, 1e-12)
        static_bias = (sum_g_pid / denom).astype(np.float64, copy=False)
        shrink_tau = float(max(0.0, drift_cfg.shrink_tau))
        if shrink_tau > 0.0:
            lam = (sum_w_pid / (sum_w_pid + shrink_tau)).astype(np.float64, copy=False)
            static_bias = static_bias * np.clip(lam, 0.0, 1.0)
        static_bias[0] = 0.0

        bias_v_static = static_bias[pid_v]
        bias_t_static = static_bias[pid_t]
        ab_s = etc._fit_platt_with_offset_irls(y=y_v, x=logit_v_base, offset=bias_v_static)
        if ab_s is not None:
            a_s, b0_s = ab_s
            methods["item_bias_shrinkage_static"] = etc._sigmoid(float(a_s) * logit_t_base + float(b0_s) + bias_t_static)
            meta_out["item_bias_shrinkage_static"] = {"a": float(a_s), "b0": float(b0_s)}

        iso_s = etc._fit_isotonic(y_v, etc._sigmoid(logit_v_base + bias_v_static))
        if iso_s is not None:
            methods["item_bias_shrinkage_static_monotone"] = np.clip(
                iso_s.predict(np.clip(etc._sigmoid(logit_t_base + bias_t_static), 0.0, 1.0)),
                0.0,
                1.0,
            )
            meta_out["item_bias_shrinkage_static_monotone"] = {"link": "isotonic"}

        drift_res = fit_dynamic_item_bias(
            y_calib=y_v,
            p_calib=p_v,
            pid_calib=pid_v,
            count_calib=cnt_v,
            train_pid=np.zeros(1, dtype=np.int64),
            train_count=np.zeros(1, dtype=np.float64),
            n_pid=n_pid,
            cfg=drift_cfg,
            count_eval=cnt_t,
            time_edges_override=time_edges_override,
        )
        if drift_res.bias_table is not None and drift_res.time_bin_index is not None:
            tb_v = etc._digitize_bins(cnt_v, time_edges_override, n_bins=int(drift_cfg.time_bins))
            tb_t = drift_res.time_bin_index.astype(np.int64, copy=False)
            bias_table = drift_res.bias_table.astype(np.float64, copy=False)
            bias_v = bias_table[pid_v, tb_v]
            bias_t = bias_table[pid_t, tb_t]
            ab_dyn = etc._fit_platt_with_offset_irls(y=y_v, x=logit_v_base, offset=bias_v)
            if ab_dyn is not None:
                a_dyn, b0_dyn = ab_dyn
                methods["item_bias_shrinkage_dynamic"] = etc._sigmoid(float(a_dyn) * logit_t_base + float(b0_dyn) + bias_t)
                meta_out["item_bias_shrinkage_dynamic"] = {
                    "a": float(a_dyn),
                    "b0": float(b0_dyn),
                    "process_var": float(drift_cfg.process_var),
                    "obs_min_weight": float(drift_cfg.obs_min_weight),
                }

    (pack_dir / "meta.json").write_text(json.dumps(meta_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    overall_rows = []
    for name, p_fix in methods.items():
        met = etc._metrics(y_t, p_fix, ece_bins=int(args.ece_bins))
        overall_rows.append({"method": name, **met})

    def _fmt(x: object) -> str:
        if isinstance(x, float):
            if math.isnan(x):
                return "nan"
            return f"{x:.6f}"
        return str(x)

    keys = ["method", "n", "auc", "acc", "nll", "brier", "rmse", "ece"]
    csv_lines = [",".join(keys)]
    for r in overall_rows:
        csv_lines.append(",".join(_fmt(r.get(k, "")) for k in keys))
    (tables_dir / "metrics_overall.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    if want_full_pack:
        md = [f"# Overall metrics ({args.dataset} / {args.model})", "", "| method | n | AUC | Acc | NLL | Brier | RMSE | ECE |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in overall_rows:
            md.append(
                f"| {r['method']} | {int(r['n'])} | {_fmt(r['auc'])} | {_fmt(r['acc'])} | {_fmt(r['nll'])} | {_fmt(r['brier'])} | {_fmt(r['rmse'])} | {_fmt(r['ece'])} |"
            )
        (tables_dir / "metrics_overall.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    slice_rows = etc._slice_metrics(y_t, methods, cnt_t, n_slices=int(args.time_slices), ece_bins=int(args.ece_bins))
    if slice_rows:
        skeys = ["slice", "count_lo", "count_hi", "method", "n", "auc", "ece", "nll"]
        out = [",".join(skeys)]
        for r in slice_rows:
            out.append(",".join(_fmt(r.get(k, "")) for k in skeys))
        (tables_dir / "metrics_time_slices.csv").write_text("\n".join(out) + "\n", encoding="utf-8")
    else:
        (tables_dir / "metrics_time_slices.csv").write_text("slice,count_lo,count_hi,method,n,auc,ece,nll\n", encoding="utf-8")

    if want_full_pack:
        (sources_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

        def _safe_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
            try:
                return etc._metrics(y, p, ece_bins=int(args.ece_bins))
            except Exception:
                return {"n": float(y.size), "auc": float("nan"), "acc": float("nan"), "nll": float("nan"), "brier": float("nan"), "rmse": float("nan"), "ece": float("nan")}

        nz_counts = pid_counts[1:][pid_counts[1:] > 0]
        pid_bin = np.full(pid_counts.shape[0], -1, dtype=np.int8)
        bin_defs = []
        if nz_counts.size > 0:
            q1 = float(np.quantile(nz_counts.astype(np.float64), q=1.0 / 3.0, method="linear"))
            q2 = float(np.quantile(nz_counts.astype(np.float64), q=2.0 / 3.0, method="linear"))
            lo1 = float(min(q1, q2))
            hi1 = float(max(q1, q2))
            pid_bin[(pid_counts > 0) & (pid_counts <= lo1)] = 0
            pid_bin[(pid_counts > lo1) & (pid_counts <= hi1)] = 1
            pid_bin[pid_counts > hi1] = 2
            for bi in [0, 1, 2]:
                pids = np.where(pid_bin == bi)[0]
                pids = pids[pids > 0]
                if pids.size <= 0:
                    continue
                cs = pid_counts[pids].astype(np.float64, copy=False)
                bin_defs.append({"pidcount_bin": int(bi), "n_pid": int(pids.size), "pidcount_lo": float(np.min(cs)), "pidcount_hi": float(np.max(cs))})

        dens_rows = []
        for bd in bin_defs:
            bi = int(bd["pidcount_bin"])
            m_bin = (pid_t > 0) & (pid_bin[pid_t] == bi)
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
                        "n_pid": int(bd["n_pid"]),
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

        paper_methods = [m for m in etc.PAPER_METHODS if m in methods]
        paper_label = dict(etc.PAPER_LABELS)
        all_rel_keys = [m for m in dict.fromkeys([*etc.PAPER_METHODS, *etc.APPENDIX_METHODS, "global_sigmoid_time", "item_bias_shrinkage_dynamic"]) if m in methods]

        p_paper = {paper_label.get(k, k): methods[k] for k in paper_methods}
        if p_paper:
            ece_vals = {lbl: etc._ece(y_t, p, n_bins=int(args.ece_bins)) for lbl, p in p_paper.items()}
            etc._plot_reliability_bargap_grid_svg(
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
                    ece_late = {lbl: etc._ece(y_t[m_late], p[m_late], n_bins=int(args.ece_bins)) for lbl, p in p_paper.items()}
                    etc._plot_reliability_bargap_grid_svg(
                        figs_dir / "fig2_reliability_bargap_late.svg",
                        y=y_t[m_late],
                        p_methods={lbl: p[m_late] for lbl, p in p_paper.items()},
                        n_bins=int(args.reliability_bins),
                        ece_values=ece_late,
                    )

        p_rel = {k: methods[k] for k in all_rel_keys}
        if p_rel:
            etc._plot_reliability_svg(
                figs_dir / "fig_reliability_all.svg",
                title=f"Reliability (all test) — {args.dataset} / {args.model}",
                y=y_t,
                p_methods=p_rel,
                n_bins=int(args.reliability_bins),
            )
            m0 = np.isfinite(cnt_t)
            if np.any(m0):
                thr = float(np.quantile(cnt_t[m0], q=float(args.late_q), method="linear"))
                m_late = m0 & (cnt_t >= thr)
                if int(np.sum(m_late)) > 0:
                    etc._plot_reliability_svg(
                        figs_dir / "fig_reliability_late.svg",
                        title=f"Reliability (late q={float(args.late_q):g}) — {args.dataset} / {args.model}",
                        y=y_t[m_late],
                        p_methods={k: v[m_late] for k, v in p_rel.items()},
                        n_bins=int(args.reliability_bins),
                    )

        if slice_rows:
            etc._plot_time_auc_ece_svg(
                figs_dir / "fig1_time_auc_ece.svg",
                title=f"AUC vs ECE over time — {args.dataset} / {args.model}",
                slices=slice_rows,
                methods_auc=paper_methods,
                methods_ece=paper_methods,
                label_map=paper_label,
            )
            etc._plot_time_auc_ece_svg(
                figs_dir / "fig_time_auc_ece.svg",
                title=f"AUC vs ECE over time slices — {args.dataset} / {args.model}",
                slices=slice_rows,
                methods_auc=all_rel_keys,
                methods_ece=all_rel_keys,
            )

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
- NPZ inputs recorded in `meta.json`
"""
        (pack_dir / "index.md").write_text(index, encoding="utf-8")

    (pack_dir / "meta.json").write_text(json.dumps(meta_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[eval_pack_from_logits_npz] ok: {pack_dir}", flush=True)


if __name__ == "__main__":
    main()
