#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

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

DEFAULT_METHODS = [
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

PAPER_METHODS = [
    "raw_score",
    "global_sigmoid",
    "item_bias_mean_monotone",
    "item_bias_shrinkage_static",
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


def _ds_alias(ds: str) -> str:
    ds = str(ds)
    if ds == "assist2017_pid_uid_time_pos":
        return "as17"
    if ds == "assist2009_pid_uid_time":
        return "as09"
    if ds == "algebra_merged_pid_uid_time":
        return "algebra"
    if ds == "eedi_task12_pid_uid_time":
        return "eedi"
    return ds


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(x: object) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return float("nan")


def _safe_name(s: str) -> str:
    s = str(s).strip()
    return "".join(c if (c.isalnum() or c in {"-", "_", "."}) else "_" for c in s) or "x"


def _canonical_method(name: object) -> str:
    method = str(name).strip()
    return METHOD_ALIAS.get(method, method)


def _display_method(name: str) -> str:
    return PAPER_LABELS.get(str(name), str(name))


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


def _plot_metric_time_svg(
    path: Path,
    *,
    title: str,
    slice_ids: list[int],
    series: dict[str, np.ndarray],
    y_label: str,
) -> None:
    width, height = 920, 520
    ml, mr, mt, mb = 90, 30, 60, 80
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    k = len(slice_ids)
    if k <= 0:
        return

    def xmap(i: float) -> float:
        if k <= 1:
            return ml + plot_w / 2.0
        return ml + (float(i) / float(k - 1)) * plot_w

    # Determine y-range from finite values.
    allv = []
    for v in series.values():
        vv = np.asarray(v, dtype=np.float64).reshape(-1)
        allv.append(vv[np.isfinite(vv)])
    allv = np.concatenate(allv, axis=0) if allv else np.array([], dtype=np.float64)
    if allv.size <= 0:
        return
    y_min = float(np.min(allv))
    y_max = float(np.max(allv))
    pad = max(1e-4, 0.10 * (y_max - y_min))
    y_min = y_min - pad
    y_max = max(y_min + 1e-6, y_max + pad)
    if str(y_label).strip().lower() == "ece":
        y_min = max(0.0, y_min)

    def ymap(v: float) -> float:
        if not math.isfinite(v):
            return float("nan")
        t = (float(v) - y_min) / (y_max - y_min)
        return mt + (1.0 - t) * plot_h

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf"]
    methods = list(series.keys())
    color = {m: palette[i % len(palette)] for i, m in enumerate(methods)}

    body = []
    body.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')

    # Axes.
    body.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+plot_h}" stroke="#111" stroke-width="2"/>')
    body.append(f'<line x1="{ml}" y1="{mt+plot_h}" x2="{ml+plot_w}" y2="{mt+plot_h}" stroke="#111" stroke-width="2"/>')

    # X ticks.
    for i, sid in enumerate(slice_ids):
        x = xmap(i)
        body.append(f'<line x1="{x}" y1="{mt+plot_h}" x2="{x}" y2="{mt+plot_h+6}" stroke="#111" stroke-width="2"/>')
        body.append(f'<text x="{x}" y="{mt+plot_h+26}" font-size="13" text-anchor="middle" fill="#111">{sid}</text>')

    # Y ticks.
    for j in range(6):
        t = j / 5.0
        v = y_min + t * (y_max - y_min)
        y = ymap(v)
        body.append(f'<line x1="{ml-6}" y1="{y}" x2="{ml}" y2="{y}" stroke="#111" stroke-width="2"/>')
        body.append(f'<text x="{ml-10}" y="{y+4}" font-size="13" text-anchor="end" fill="#111">{v:.3f}</text>')

    # Labels.
    body.append(f'<text x="{ml}" y="32" font-size="18" text-anchor="start" fill="#111">{title}</text>')
    body.append(
        f'<text x="{ml+plot_w/2}" y="{height-28}" font-size="16" text-anchor="middle" fill="#111">Time slice (quantile index)</text>'
    )
    body.append(
        f'<text x="28" y="{mt+plot_h/2}" font-size="16" text-anchor="middle" fill="#111" transform="rotate(-90 28 {mt+plot_h/2})">{_svg_escape(y_label)}</text>'
    )

    # Lines.
    for m in methods:
        s = np.asarray(series[m], dtype=np.float64).reshape(-1)
        pts = []
        for i in range(k):
            if not math.isfinite(float(s[i])):
                continue
            pts.append((xmap(i), ymap(float(s[i]))))
        if len(pts) >= 2:
            d = "M " + " L ".join(f"{x:.2f} {yy:.2f}" for x, yy in pts)
            body.append(f'<path d="{d}" fill="none" stroke="{color[m]}" stroke-width="3"/>')
        for x, yy in pts:
            body.append(f'<circle cx="{x:.2f}" cy="{yy:.2f}" r="4.0" fill="{color[m]}"/>')

    # Legend.
    leg_x = ml + plot_w - 260
    leg_y = mt - 22
    for i, m in enumerate(methods):
        y0 = leg_y + i * 20
        body.append(f'<rect x="{leg_x}" y="{y0-12}" width="14" height="14" fill="{color[m]}"/>')
        body.append(f'<text x="{leg_x+20}" y="{y0}" font-size="14" text-anchor="start" fill="#111">{m}</text>')

    _write_svg(path, width=width, height=height, body="\n".join(body))


def _plot_ece_time_svg(
    path: Path,
    *,
    title: str,
    slice_ids: list[int],
    series: dict[str, np.ndarray],
) -> None:
    _plot_metric_time_svg(path, title=title, slice_ids=slice_ids, series=series, y_label="ECE")


def _plot_auc_time_svg(
    path: Path,
    *,
    title: str,
    slice_ids: list[int],
    series: dict[str, np.ndarray],
) -> None:
    _plot_metric_time_svg(path, title=title, slice_ids=slice_ids, series=series, y_label="AUC")


def _plot_metric_vs_calib_frac_svg(
    path: Path,
    *,
    title: str,
    fracs: list[float],
    series_mean: dict[str, np.ndarray],
    series_std: dict[str, np.ndarray] | None = None,
    y_label: str,
) -> None:
    width, height = 920, 520
    ml, mr, mt, mb = 90, 30, 60, 80
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    if not fracs or not series_mean:
        return

    # Sort fracs descending for readability (1.0 → 0.1).
    fracs = [float(f) for f in fracs]
    order = np.argsort(-np.asarray(fracs, dtype=np.float64))
    fracs = [fracs[int(i)] for i in order.tolist()]

    k = len(fracs)

    def xmap(i: float) -> float:
        if k <= 1:
            return ml + plot_w / 2.0
        return ml + (float(i) / float(k - 1)) * plot_w

    # y-range from mean±std (if provided) else mean.
    allv = []
    for m, y in series_mean.items():
        ym = np.asarray(y, dtype=np.float64).reshape(-1)[order]
        if series_std and m in series_std:
            ys = np.asarray(series_std[m], dtype=np.float64).reshape(-1)[order]
            allv.append((ym - ys)[np.isfinite(ym - ys)])
            allv.append((ym + ys)[np.isfinite(ym + ys)])
        else:
            allv.append(ym[np.isfinite(ym)])
    allv = np.concatenate(allv, axis=0) if allv else np.array([], dtype=np.float64)
    if allv.size <= 0:
        return
    y_min = float(np.min(allv))
    y_max = float(np.max(allv))
    pad = max(1e-6, 0.10 * (y_max - y_min))
    y_min = y_min - pad
    y_max = y_max + pad
    if y_label.lower() == "ece":
        y_min = max(0.0, y_min)

    def ymap(v: float) -> float:
        if not math.isfinite(v):
            return float("nan")
        t = (float(v) - y_min) / max(1e-9, y_max - y_min)
        return mt + (1.0 - t) * plot_h

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf"]
    methods = list(series_mean.keys())
    color = {m: palette[i % len(palette)] for i, m in enumerate(methods)}

    body = []
    body.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    body.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+plot_h}" stroke="#111" stroke-width="2"/>')
    body.append(f'<line x1="{ml}" y1="{mt+plot_h}" x2="{ml+plot_w}" y2="{mt+plot_h}" stroke="#111" stroke-width="2"/>')

    # X ticks.
    for i, f in enumerate(fracs):
        x = xmap(i)
        body.append(f'<line x1="{x}" y1="{mt+plot_h}" x2="{x}" y2="{mt+plot_h+6}" stroke="#111" stroke-width="2"/>')
        body.append(f'<text x="{x}" y="{mt+plot_h+26}" font-size="13" text-anchor="middle" fill="#111">{f:g}</text>')

    # Y ticks.
    for j in range(6):
        t = j / 5.0
        v = y_min + t * (y_max - y_min)
        y = ymap(v)
        body.append(f'<line x1="{ml-6}" y1="{y}" x2="{ml}" y2="{y}" stroke="#111" stroke-width="2"/>')
        body.append(f'<text x="{ml-10}" y="{y+4}" font-size="13" text-anchor="end" fill="#111">{v:.4f}</text>')

    body.append(f'<text x="{ml}" y="32" font-size="18" text-anchor="start" fill="#111">{title}</text>')
    body.append(
        f'<text x="{ml+plot_w/2}" y="{height-28}" font-size="16" text-anchor="middle" fill="#111">Calibration fraction (valid tail)</text>'
    )
    body.append(
        f'<text x="28" y="{mt+plot_h/2}" font-size="16" text-anchor="middle" fill="#111" transform="rotate(-90 28 {mt+plot_h/2})">{_svg_escape(y_label)}</text>'
    )

    # Lines with optional error bars.
    for m in methods:
        ym = np.asarray(series_mean[m], dtype=np.float64).reshape(-1)[order]
        ys = np.asarray(series_std[m], dtype=np.float64).reshape(-1)[order] if (series_std and m in series_std) else None
        pts = []
        for i in range(k):
            if not math.isfinite(float(ym[i])):
                continue
            x = xmap(i)
            y = ymap(float(ym[i]))
            pts.append((x, y))
            if ys is not None and math.isfinite(float(ys[i])):
                y_lo = ymap(float(ym[i] - ys[i]))
                y_hi = ymap(float(ym[i] + ys[i]))
                body.append(f'<line x1="{x:.2f}" y1="{y_lo:.2f}" x2="{x:.2f}" y2="{y_hi:.2f}" stroke="{color[m]}" stroke-width="2" opacity="0.55"/>')
        if len(pts) >= 2:
            d = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in pts)
            body.append(f'<path d="{d}" fill="none" stroke="{color[m]}" stroke-width="3"/>')
        for x, y in pts:
            body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.0" fill="{color[m]}"/>')

    # Legend.
    leg_x = ml + plot_w - 260
    leg_y = mt - 22
    for i, m in enumerate(methods):
        y0 = leg_y + i * 20
        body.append(f'<rect x="{leg_x}" y="{y0-12}" width="14" height="14" fill="{color[m]}"/>')
        body.append(f'<text x="{leg_x+20}" y="{y0}" font-size="14" text-anchor="start" fill="#111">{_svg_escape(m)}</text>')

    _write_svg(path, width=width, height=height, body="\n".join(body))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Summarize paper packs (overall + time-slice metrics; ECE(t) curves).")
    ap.add_argument("--packs_root", type=str, default="_paper_packs")
    ap.add_argument("--prefix", type=str, required=True, help="Pack dir prefix, e.g. 20260222_201743")
    ap.add_argument("--out_dir", type=str, default="", help="Output directory (default: <packs_root>/<prefix>_summary)")
    ap.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated method order for plots/tables (missing methods are skipped).",
    )
    args = ap.parse_args(argv)

    packs_root = Path(str(args.packs_root))
    if not packs_root.is_absolute():
        packs_root = Path.cwd() / packs_root
    prefix = str(args.prefix).strip()
    if not prefix:
        raise SystemExit("--prefix is required")

    out_dir = Path(str(args.out_dir).strip()) if str(args.out_dir).strip() else (packs_root / f"{_safe_name(prefix)}_summary")
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    want_methods = [_canonical_method(m) for m in str(args.methods).split(",") if m.strip()]

    pack_dirs = []
    for p in sorted(packs_root.iterdir()):
        if not p.is_dir():
            continue
        if not p.name.startswith(prefix + "_"):
            continue
        if not (p / "meta.json").is_file():
            continue
        if not (p / "tables" / "metrics_overall.csv").is_file():
            continue
        if not (p / "tables" / "metrics_time_slices.csv").is_file():
            continue
        pack_dirs.append(p)

    if not pack_dirs:
        raise SystemExit(f"no packs found under {packs_root} with prefix {prefix!r}")

    overall_long: list[dict[str, object]] = []
    time_long: list[dict[str, object]] = []
    pidcount_long: list[dict[str, object]] = []
    for pack in pack_dirs:
        meta = json.loads((pack / "meta.json").read_text(encoding="utf-8"))
        dataset = str(meta.get("dataset", ""))
        model = str(meta.get("model", ""))
        pack_id = str(meta.get("pack_id", pack.name))
        calib_frac = _to_float(meta.get("calib_frac", float("nan")))

        overall_rows = _read_csv(pack / "tables" / "metrics_overall.csv")
        for r in overall_rows:
            method = _canonical_method(r.get("method", ""))
            overall_long.append(
                {
                    "pack": pack_id,
                    "dataset": dataset,
                    "model": model,
                    "calib_frac": calib_frac,
                    "method": method,
                    "n": int(float(r.get("n", "nan"))) if str(r.get("n", "")).strip() else 0,
                    "auc": _to_float(r.get("auc", "nan")),
                    "acc": _to_float(r.get("acc", "nan")),
                    "nll": _to_float(r.get("nll", "nan")),
                    "brier": _to_float(r.get("brier", "nan")),
                    "rmse": _to_float(r.get("rmse", "nan")),
                    "ece": _to_float(r.get("ece", "nan")),
                }
            )
        time_rows = _read_csv(pack / "tables" / "metrics_time_slices.csv")
        for r in time_rows:
            time_long.append(
                {
                    "pack": pack_id,
                    "dataset": dataset,
                    "model": model,
                    "calib_frac": calib_frac,
                    "slice": int(float(r.get("slice", "nan"))) if str(r.get("slice", "")).strip() else -1,
                    "count_lo": _to_float(r.get("count_lo", "nan")),
                    "count_hi": _to_float(r.get("count_hi", "nan")),
                    "method": _canonical_method(r.get("method", "")),
                    "n": int(float(r.get("n", "nan"))) if str(r.get("n", "")).strip() else 0,
                    "auc": _to_float(r.get("auc", "nan")),
                    "ece": _to_float(r.get("ece", "nan")),
                    "nll": _to_float(r.get("nll", "nan")),
                }
            )

        dens_path = pack / "tables" / "metrics_pidcount_slices.csv"
        if dens_path.is_file():
            dens_rows = _read_csv(dens_path)
            for r in dens_rows:
                pidcount_long.append(
                    {
                        "pack": pack_id,
                        "dataset": dataset,
                        "model": model,
                        "calib_frac": calib_frac,
                        "pidcount_bin": int(float(r.get("pidcount_bin", "nan"))) if str(r.get("pidcount_bin", "")).strip() else -1,
                        "pidcount_lo": _to_float(r.get("pidcount_lo", "nan")),
                        "pidcount_hi": _to_float(r.get("pidcount_hi", "nan")),
                        "n_pid": int(float(r.get("n_pid", "nan"))) if str(r.get("n_pid", "")).strip() else 0,
                        "method": _canonical_method(r.get("method", "")),
                        "n": int(float(r.get("n", "nan"))) if str(r.get("n", "")).strip() else 0,
                        "auc": _to_float(r.get("auc", "nan")),
                        "ece": _to_float(r.get("ece", "nan")),
                        "nll": _to_float(r.get("nll", "nan")),
                    }
                )

    # Write long tables.
    def _write_long(path: Path, rows: list[dict[str, object]], keys: list[str]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in keys})

    _write_long(
        tables_dir / "metrics_overall_long.csv",
        overall_long,
        ["pack", "dataset", "model", "calib_frac", "method", "n", "auc", "acc", "nll", "brier", "rmse", "ece"],
    )
    _write_long(
        tables_dir / "metrics_time_slices_long.csv",
        time_long,
        ["pack", "dataset", "model", "calib_frac", "slice", "count_lo", "count_hi", "method", "n", "auc", "ece", "nll"],
    )
    if pidcount_long:
        _write_long(
            tables_dir / "metrics_pidcount_slices_long.csv",
            pidcount_long,
            ["pack", "dataset", "model", "calib_frac", "pidcount_bin", "pidcount_lo", "pidcount_hi", "n_pid", "method", "n", "auc", "ece", "nll"],
        )

    # Mean±std overall metrics (paper table backbone).
    overall_bucket: dict[tuple[str, str, float, str], list[dict[str, float]]] = defaultdict(list)
    for r in overall_long:
        ds = str(r["dataset"])
        model = str(r["model"])
        cf = float(r.get("calib_frac", float("nan")))
        method = str(r["method"])
        overall_bucket[(ds, model, cf, method)].append(
            {
                "auc": float(r.get("auc", float("nan"))),
                "acc": float(r.get("acc", float("nan"))),
                "nll": float(r.get("nll", float("nan"))),
                "brier": float(r.get("brier", float("nan"))),
                "rmse": float(r.get("rmse", float("nan"))),
                "ece": float(r.get("ece", float("nan"))),
            }
        )

    meanstd_rows: list[dict[str, object]] = []
    for (ds, model, cf, method), vals in sorted(overall_bucket.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])):
        arr = {k: np.asarray([v.get(k, float("nan")) for v in vals], dtype=np.float64) for k in ["auc", "acc", "nll", "brier", "rmse", "ece"]}
        meanstd_rows.append(
            {
                "dataset": ds,
                "model": model,
                "calib_frac": float(cf),
                "method": method,
                "n_packs": int(len(vals)),
                "auc_mean": float(np.nanmean(arr["auc"])),
                "auc_std": float(np.nanstd(arr["auc"], ddof=0)),
                "acc_mean": float(np.nanmean(arr["acc"])),
                "acc_std": float(np.nanstd(arr["acc"], ddof=0)),
                "nll_mean": float(np.nanmean(arr["nll"])),
                "nll_std": float(np.nanstd(arr["nll"], ddof=0)),
                "brier_mean": float(np.nanmean(arr["brier"])),
                "brier_std": float(np.nanstd(arr["brier"], ddof=0)),
                "rmse_mean": float(np.nanmean(arr["rmse"])),
                "rmse_std": float(np.nanstd(arr["rmse"], ddof=0)),
                "ece_mean": float(np.nanmean(arr["ece"])),
                "ece_std": float(np.nanstd(arr["ece"], ddof=0)),
            }
        )

    _write_long(
        tables_dir / "metrics_overall_meanstd.csv",
        meanstd_rows,
        [
            "dataset",
            "model",
            "calib_frac",
            "method",
            "n_packs",
            "auc_mean",
            "auc_std",
            "acc_mean",
            "acc_std",
            "nll_mean",
            "nll_std",
            "brier_mean",
            "brier_std",
            "rmse_mean",
            "rmse_std",
            "ece_mean",
            "ece_std",
        ],
    )

    # Mean±std for density slices (if present).
    if pidcount_long:
        dens_bucket: dict[tuple[str, str, float, int, str], list[dict[str, float]]] = defaultdict(list)
        for r in pidcount_long:
            ds = str(r["dataset"])
            model = str(r["model"])
            cf = float(r.get("calib_frac", float("nan")))
            bi = int(r.get("pidcount_bin", -1))
            method = str(r["method"])
            dens_bucket[(ds, model, cf, bi, method)].append(
                {
                    "auc": float(r.get("auc", float("nan"))),
                    "ece": float(r.get("ece", float("nan"))),
                    "nll": float(r.get("nll", float("nan"))),
                }
            )

        dens_meanstd_rows: list[dict[str, object]] = []
        for (ds, model, cf, bi, method), vals in sorted(dens_bucket.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3], x[0][4])):
            auc = np.asarray([v.get("auc", float("nan")) for v in vals], dtype=np.float64)
            ece = np.asarray([v.get("ece", float("nan")) for v in vals], dtype=np.float64)
            nll = np.asarray([v.get("nll", float("nan")) for v in vals], dtype=np.float64)
            dens_meanstd_rows.append(
                {
                    "dataset": ds,
                    "model": model,
                    "calib_frac": float(cf),
                    "pidcount_bin": int(bi),
                    "method": method,
                    "n_packs": int(len(vals)),
                    "auc_mean": float(np.nanmean(auc)),
                    "auc_std": float(np.nanstd(auc, ddof=0)),
                    "ece_mean": float(np.nanmean(ece)),
                    "ece_std": float(np.nanstd(ece, ddof=0)),
                    "nll_mean": float(np.nanmean(nll)),
                    "nll_std": float(np.nanstd(nll, ddof=0)),
                }
            )

        _write_long(
            tables_dir / "metrics_pidcount_slices_meanstd.csv",
            dens_meanstd_rows,
            [
                "dataset",
                "model",
                "calib_frac",
                "pidcount_bin",
                "method",
                "n_packs",
                "auc_mean",
                "auc_std",
                "ece_mean",
                "ece_std",
                "nll_mean",
                "nll_std",
            ],
        )

    # Calib-frac regime curves (if prefix includes multiple calib_frac values).
    meanstd_map: dict[tuple[str, str, float, str], dict[str, float]] = {}
    fracs_by_dm: dict[tuple[str, str], set[float]] = defaultdict(set)
    for r in meanstd_rows:
        ds = str(r["dataset"])
        model = str(r["model"])
        cf = float(r["calib_frac"])
        m = str(r["method"])
        fracs_by_dm[(ds, model)].add(cf)
        meanstd_map[(ds, model, cf, m)] = {
            "auc_mean": float(r["auc_mean"]),
            "auc_std": float(r["auc_std"]),
            "ece_mean": float(r["ece_mean"]),
            "ece_std": float(r["ece_std"]),
        }

    paper_methods = list(PAPER_METHODS)
    for (ds, model), fracs in sorted(fracs_by_dm.items(), key=lambda x: (x[0][0], x[0][1])):
        fracs_sorted = sorted([f for f in fracs if math.isfinite(float(f))], reverse=True)
        if len(fracs_sorted) <= 1:
            continue
        present_methods = [m for m in paper_methods if any((ds, model, cf, m) in meanstd_map for cf in fracs_sorted)]
        if not present_methods:
            continue
        series_auc = {}
        series_auc_std = {}
        series_ece = {}
        series_ece_std = {}
        for m in present_methods:
            auc_mu = np.array([meanstd_map.get((ds, model, cf, m), {}).get("auc_mean", float("nan")) for cf in fracs_sorted], dtype=np.float64)
            auc_sd = np.array([meanstd_map.get((ds, model, cf, m), {}).get("auc_std", float("nan")) for cf in fracs_sorted], dtype=np.float64)
            ece_mu = np.array([meanstd_map.get((ds, model, cf, m), {}).get("ece_mean", float("nan")) for cf in fracs_sorted], dtype=np.float64)
            ece_sd = np.array([meanstd_map.get((ds, model, cf, m), {}).get("ece_std", float("nan")) for cf in fracs_sorted], dtype=np.float64)
            if np.isfinite(auc_mu).any():
                series_auc[m] = auc_mu
                series_auc_std[m] = auc_sd
            if np.isfinite(ece_mu).any():
                series_ece[m] = ece_mu
                series_ece_std[m] = ece_sd

        alias = _safe_name(f"{_ds_alias(ds)}_{model}")
        if series_ece:
            display_ece = {_display_method(m): v for m, v in series_ece.items()}
            display_ece_std = {_display_method(m): v for m, v in series_ece_std.items()}
            _plot_metric_vs_calib_frac_svg(
                figs_dir / f"fig_ece_vs_calib_frac_{alias}.svg",
                title=f"ECE vs calib_frac (mean across packs) — {ds} / {model}",
                fracs=fracs_sorted,
                series_mean=display_ece,
                series_std=display_ece_std,
                y_label="ECE",
            )
        if series_auc:
            display_auc = {_display_method(m): v for m, v in series_auc.items()}
            display_auc_std = {_display_method(m): v for m, v in series_auc_std.items()}
            _plot_metric_vs_calib_frac_svg(
                figs_dir / f"fig_auc_vs_calib_frac_{alias}.svg",
                title=f"AUC vs calib_frac (mean across packs) — {ds} / {model}",
                fracs=fracs_sorted,
                series_mean=display_auc,
                series_std=display_auc_std,
                y_label="AUC",
            )

    # Aggregate metric(t): dataset x model x calib_frac x method x slice.
    ece_bucket: dict[tuple[str, str, float, str, int], list[float]] = defaultdict(list)
    auc_bucket: dict[tuple[str, str, float, str, int], list[float]] = defaultdict(list)
    slice_ids_by_key: dict[tuple[str, str, float], set[int]] = defaultdict(set)
    methods_by_key: dict[tuple[str, str, float], set[str]] = defaultdict(set)
    for r in time_long:
        ds = str(r["dataset"])
        model = str(r["model"])
        cf = float(r.get("calib_frac", float("nan")))
        m = str(r["method"])
        si = int(r["slice"])
        if si < 0:
            continue
        if not m:
            continue

        slice_ids_by_key[(ds, model, cf)].add(si)
        methods_by_key[(ds, model, cf)].add(m)

        e = float(r["ece"])
        if math.isfinite(e):
            ece_bucket[(ds, model, cf, m, si)].append(e)

        a = float(r["auc"])
        if math.isfinite(a):
            auc_bucket[(ds, model, cf, m, si)].append(a)

    ece_rows = []
    first_last_rows = []
    for (ds, model, cf) in sorted(slice_ids_by_key.keys(), key=lambda x: (x[0], x[1], x[2])):
        slice_ids = sorted(slice_ids_by_key[(ds, model, cf)])
        if not slice_ids:
            continue
        s0, s1 = int(slice_ids[0]), int(slice_ids[-1])
        for m in sorted(methods_by_key[(ds, model, cf)]):
            for si in slice_ids:
                vals = np.asarray(ece_bucket.get((ds, model, cf, m, si), []), dtype=np.float64)
                if vals.size <= 0:
                    continue
                ece_rows.append(
                    {
                        "dataset": ds,
                        "model": model,
                        "calib_frac": float(cf),
                        "method": m,
                        "slice": int(si),
                        "n_packs": int(vals.size),
                        "ece_mean": float(np.mean(vals)),
                        "ece_std": float(np.std(vals, ddof=0)),
                    }
                )

            v0 = np.asarray(ece_bucket.get((ds, model, cf, m, s0), []), dtype=np.float64)
            v1 = np.asarray(ece_bucket.get((ds, model, cf, m, s1), []), dtype=np.float64)
            if v0.size > 0 and v1.size > 0:
                first_last_rows.append(
                    {
                        "dataset": ds,
                        "model": model,
                        "calib_frac": float(cf),
                        "method": m,
                        "slice_first": int(s0),
                        "slice_last": int(s1),
                        "ece_first_mean": float(np.mean(v0)),
                        "ece_last_mean": float(np.mean(v1)),
                        "ece_delta_last_minus_first": float(np.mean(v1) - np.mean(v0)),
                    }
                )

    _write_long(
        tables_dir / "ece_time_meanstd.csv",
        sorted(ece_rows, key=lambda r: (str(r["dataset"]), str(r["model"]), float(r["calib_frac"]), str(r["method"]), int(r["slice"]))),
        ["dataset", "model", "calib_frac", "method", "slice", "n_packs", "ece_mean", "ece_std"],
    )
    _write_long(
        tables_dir / "ece_time_first_last.csv",
        sorted(first_last_rows, key=lambda r: (str(r["dataset"]), str(r["model"]), float(r["calib_frac"]), str(r["method"]))),
        ["dataset", "model", "calib_frac", "method", "slice_first", "slice_last", "ece_first_mean", "ece_last_mean", "ece_delta_last_minus_first"],
    )

    # AUC(t) tables.
    auc_rows = []
    auc_first_last_rows = []
    for (ds, model, cf) in sorted(slice_ids_by_key.keys(), key=lambda x: (x[0], x[1], x[2])):
        slice_ids = sorted(slice_ids_by_key[(ds, model, cf)])
        if not slice_ids:
            continue
        s0, s1 = int(slice_ids[0]), int(slice_ids[-1])
        for m in sorted(methods_by_key[(ds, model, cf)]):
            for si in slice_ids:
                vals = np.asarray(auc_bucket.get((ds, model, cf, m, si), []), dtype=np.float64)
                if vals.size <= 0:
                    continue
                auc_rows.append(
                    {
                        "dataset": ds,
                        "model": model,
                        "calib_frac": float(cf),
                        "method": m,
                        "slice": int(si),
                        "n_packs": int(vals.size),
                        "auc_mean": float(np.mean(vals)),
                        "auc_std": float(np.std(vals, ddof=0)),
                    }
                )

            v0 = np.asarray(auc_bucket.get((ds, model, cf, m, s0), []), dtype=np.float64)
            v1 = np.asarray(auc_bucket.get((ds, model, cf, m, s1), []), dtype=np.float64)
            if v0.size > 0 and v1.size > 0:
                auc_first_last_rows.append(
                    {
                        "dataset": ds,
                        "model": model,
                        "calib_frac": float(cf),
                        "method": m,
                        "slice_first": int(s0),
                        "slice_last": int(s1),
                        "auc_first_mean": float(np.mean(v0)),
                        "auc_last_mean": float(np.mean(v1)),
                        "auc_delta_last_minus_first": float(np.mean(v1) - np.mean(v0)),
                    }
                )

    _write_long(
        tables_dir / "auc_time_meanstd.csv",
        sorted(auc_rows, key=lambda r: (str(r["dataset"]), str(r["model"]), float(r["calib_frac"]), str(r["method"]), int(r["slice"]))),
        ["dataset", "model", "calib_frac", "method", "slice", "n_packs", "auc_mean", "auc_std"],
    )
    _write_long(
        tables_dir / "auc_time_first_last.csv",
        sorted(auc_first_last_rows, key=lambda r: (str(r["dataset"]), str(r["model"]), float(r["calib_frac"]), str(r["method"]))),
        ["dataset", "model", "calib_frac", "method", "slice_first", "slice_last", "auc_first_mean", "auc_last_mean", "auc_delta_last_minus_first"],
    )

    # Plot per-dataset mean ECE(t).
    ece_mean_map: dict[tuple[str, str, float, str, int], float] = {}
    for r in ece_rows:
        cfk = float(round(float(r["calib_frac"]), 6)) if math.isfinite(float(r["calib_frac"])) else float("nan")
        ece_mean_map[(str(r["dataset"]), str(r["model"]), cfk, str(r["method"]), int(r["slice"]))] = float(r["ece_mean"])

    for (ds, model, cf) in sorted(slice_ids_by_key.keys(), key=lambda x: (x[0], x[1], x[2])):
        slice_ids = sorted(slice_ids_by_key[(ds, model, cf)])
        if not slice_ids:
            continue
        cfk = float(round(float(cf), 6)) if math.isfinite(float(cf)) else float("nan")
        # method order: wanted first, then the rest.
        present = sorted(methods_by_key[(ds, model, cf)])
        ordered = [m for m in want_methods if m in present] + [m for m in present if m not in want_methods]
        # build series vectors
        series = {}
        for m in ordered:
            s = np.full(len(slice_ids), np.nan, dtype=np.float64)
            for i, si in enumerate(slice_ids):
                s[i] = ece_mean_map.get((ds, model, cfk, m, int(si)), float("nan"))
            if np.isfinite(s).any():
                series[_display_method(m)] = s
        if not series:
            continue
        alias = _safe_name(f"{_ds_alias(ds)}_{model}_cf{cf:g}")
        _plot_ece_time_svg(
            figs_dir / f"fig_ece_time_{alias}.svg",
            title=f"ECE over time slices (mean across packs) — {ds} / {model} (calib_frac={cf:g})",
            slice_ids=[int(x) for x in slice_ids],
            series=series,
        )

    # Plot per-dataset mean AUC(t).
    auc_mean_map: dict[tuple[str, str, float, str, int], float] = {}
    for r in auc_rows:
        cfk = float(round(float(r["calib_frac"]), 6)) if math.isfinite(float(r["calib_frac"])) else float("nan")
        auc_mean_map[(str(r["dataset"]), str(r["model"]), cfk, str(r["method"]), int(r["slice"]))] = float(r["auc_mean"])

    for (ds, model, cf) in sorted(slice_ids_by_key.keys(), key=lambda x: (x[0], x[1], x[2])):
        slice_ids = sorted(slice_ids_by_key[(ds, model, cf)])
        if not slice_ids:
            continue
        cfk = float(round(float(cf), 6)) if math.isfinite(float(cf)) else float("nan")
        present = sorted(methods_by_key[(ds, model, cf)])
        ordered = [m for m in want_methods if m in present] + [m for m in present if m not in want_methods]
        series = {}
        for m in ordered:
            s = np.full(len(slice_ids), np.nan, dtype=np.float64)
            for i, si in enumerate(slice_ids):
                s[i] = auc_mean_map.get((ds, model, cfk, m, int(si)), float("nan"))
            if np.isfinite(s).any():
                series[_display_method(m)] = s
        if not series:
            continue
        alias = _safe_name(f"{_ds_alias(ds)}_{model}_cf{cf:g}")
        _plot_auc_time_svg(
            figs_dir / f"fig_auc_time_{alias}.svg",
            title=f"AUC over time slices (mean across packs) — {ds} / {model} (calib_frac={cf:g})",
            slice_ids=[int(x) for x in slice_ids],
            series=series,
        )

    # Index.
    md = []
    md.append(f"# Paper-pack summary: prefix={prefix}")
    md.append("")
    md.append(f"- packs_root: `{packs_root}`")
    md.append(f"- n_packs: {len(pack_dirs)}")
    md.append(f"- outputs: `{out_dir}`")
    md.append("")
    md.append("## Files")
    md.append("")
    md.append(f"- `tables/metrics_overall_long.csv`")
    md.append(f"- `tables/metrics_overall_meanstd.csv`")
    md.append(f"- `tables/metrics_time_slices_long.csv`")
    md.append(f"- `tables/metrics_pidcount_slices_long.csv` (if present)")
    md.append(f"- `tables/metrics_pidcount_slices_meanstd.csv` (if present)")
    md.append(f"- `tables/ece_time_meanstd.csv`")
    md.append(f"- `tables/ece_time_first_last.csv`")
    md.append(f"- `tables/auc_time_meanstd.csv`")
    md.append(f"- `tables/auc_time_first_last.csv`")
    md.append(f"- `figs/fig_ece_time_<dataset>_<model>_cf*.svg`")
    md.append(f"- `figs/fig_auc_time_<dataset>_<model>_cf*.svg`")
    md.append(f"- `figs/fig_ece_vs_calib_frac_<dataset>_<model>.svg` (if sweep)")
    md.append(f"- `figs/fig_auc_vs_calib_frac_<dataset>_<model>.svg` (if sweep)")
    md.append("")
    md.append("## Packs")
    md.append("")
    for p in pack_dirs:
        md.append(f"- `{p.name}`")
    (out_dir / "index.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[summarize_paper_packs] ok: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
