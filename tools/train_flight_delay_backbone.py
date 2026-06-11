#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier


REPO_ROOT = Path(__file__).resolve().parents[1]


def _extract_from_zip(zip_path: Path, *, member: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(member).name
    if out_path.is_file() and out_path.stat().st_size > 0:
        return out_path

    with zipfile.ZipFile(zip_path, "r") as z:
        try:
            info = z.getinfo(member)
        except KeyError as e:
            raise FileNotFoundError(f"{member} not found in zip: {zip_path}") from e
        with z.open(info) as src, out_path.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    return out_path


def _safe_int_series(s: pd.Series, *, default: int = 0) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce").fillna(default)
    return out.astype(np.int32, copy=False)


def _safe_float_series(s: pd.Series, *, default: float = 0.0) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce").fillna(default)
    return out.astype(np.float32, copy=False)


def _chunk_features(
    df: pd.DataFrame,
    *,
    include_route: bool,
) -> list[dict[str, float]]:
    # Expected columns: carrier, origin, dest, route, dep_hour, dow, month, distance_k
    carrier = df["carrier"].astype("string").fillna("UNK").tolist()
    origin = df["origin"].astype("string").fillna("UNK").tolist()
    dest = df["dest"].astype("string").fillna("UNK").tolist()
    route = df["route"].astype("string").fillna("UNK").tolist()
    dep_hour = df["dep_hour"].astype(np.int32).tolist()
    dow = df["dow"].astype(np.int32).tolist()
    month = df["month"].astype(np.int32).tolist()
    dist_k = df["distance_k"].astype(np.float32).tolist()

    feats: list[dict[str, float]] = []
    feats_extend = feats.append
    for c, o, d, r, h, w, m, dist in zip(carrier, origin, dest, route, dep_hour, dow, month, dist_k):
        dd: dict[str, float] = {
            "bias": 1.0,
            f"carrier={c}": 1.0,
            f"origin={o}": 1.0,
            f"dest={d}": 1.0,
            f"dep_hour={int(h)}": 1.0,
            f"dow={int(w)}": 1.0,
            f"month={int(m)}": 1.0,
            "distance_k": float(dist),
        }
        if include_route:
            dd[f"route={r}"] = 1.0
        feats_extend(dd)
    return feats


def _iter_filtered_chunks(
    csv_path: Path,
    *,
    valid_routes: set[str],
    chunksize: int,
    max_rows: int | None,
) -> tuple[int, pd.DataFrame]:
    usecols = [
        "Year",
        "Month",
        "DayOfWeek",
        "CRSDepTime",
        "Marketing_Airline_Network",
        "Origin",
        "Dest",
        "Distance",
        "Cancelled",
        "Diverted",
        "ArrDel15",
    ]
    dtype = {
        "Year": "int16",
        "Month": "int8",
        "DayOfWeek": "int8",
        "CRSDepTime": "float32",
        "Marketing_Airline_Network": "string",
        "Origin": "string",
        "Dest": "string",
        "Distance": "float32",
        "Cancelled": "int8",
        "Diverted": "int8",
        "ArrDel15": "float32",
    }

    n_rows = 0
    for chunk in pd.read_csv(
        str(csv_path),
        usecols=usecols,
        dtype=dtype,
        chunksize=int(chunksize),
        low_memory=False,
    ):
        n_rows += int(len(chunk))

        m = (
            (chunk["Cancelled"] == 0)
            & (chunk["Diverted"] == 0)
            & chunk["ArrDel15"].notna()
            & chunk["Origin"].notna()
            & chunk["Dest"].notna()
        )
        if not bool(m.any()):
            if max_rows is not None and n_rows >= int(max_rows):
                break
            continue

        sub = chunk.loc[m, ["Year", "Month", "DayOfWeek", "CRSDepTime", "Marketing_Airline_Network", "Origin", "Dest", "Distance", "ArrDel15"]]
        route = (sub["Origin"].astype("string") + "-" + sub["Dest"].astype("string")).astype("string")
        m2 = route.isin(valid_routes)
        if not bool(m2.any()):
            if max_rows is not None and n_rows >= int(max_rows):
                break
            continue
        sub = sub.loc[m2].copy()
        sub["route"] = route.loc[m2].astype("string")

        # Normalize / feature engineering.
        dep = _safe_int_series(sub["CRSDepTime"], default=0)
        dep_hour = (dep // 100).clip(0, 23).astype(np.int16, copy=False)
        dist_k = (_safe_float_series(sub["Distance"], default=0.0) / 1000.0).astype(np.float32, copy=False)

        sub = pd.DataFrame(
            {
                "year": _safe_int_series(sub["Year"], default=0).astype(np.int16, copy=False),
                "month": _safe_int_series(sub["Month"], default=1).clip(1, 12).astype(np.int8, copy=False),
                "dow": _safe_int_series(sub["DayOfWeek"], default=1).clip(1, 7).astype(np.int8, copy=False),
                "dep_hour": dep_hour,
                "carrier": sub["Marketing_Airline_Network"].astype("string").fillna("UNK"),
                "origin": sub["Origin"].astype("string").fillna("UNK"),
                "dest": sub["Dest"].astype("string").fillna("UNK"),
                "route": sub["route"].astype("string").fillna("UNK"),
                "distance_k": dist_k,
                "label": _safe_int_series(sub["ArrDel15"], default=0).clip(0, 1).astype(np.int8, copy=False),
            }
        )

        yield n_rows, sub

        if max_rows is not None and n_rows >= int(max_rows):
            break


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a simple flight-delay backbone (SGD) and export calib/test logits for SLC.")
    ap.add_argument("--zip_path", type=str, default="_raw_data/flight-delay-dataset-20182022.zip")
    ap.add_argument("--raw_dir", type=str, default="dataset/flight_delay_raw")
    ap.add_argument("--data_dir", type=str, default="dataset/flight_delay")
    ap.add_argument("--out_dir", type=str, default="dataset/flight_delay/logits")
    ap.add_argument("--bb", type=str, default="bbA", help="bbA=no-route, bbB=with-route")
    ap.add_argument("--seed", type=int, default=225)
    ap.add_argument("--n_features", type=int, default=262144, help="FeatureHasher dimension (power of 2 recommended).")
    ap.add_argument("--alpha", type=float, default=1e-6, help="SGD L2 regularization strength.")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--max_rows", type=int, default=0, help="Debug: stop after reading N rows per year (0 disables).")
    args = ap.parse_args()

    zip_path = Path(str(args.zip_path))
    if not zip_path.is_absolute():
        zip_path = (REPO_ROOT / zip_path).resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"zip not found: {zip_path}")

    raw_dir = Path(str(args.raw_dir))
    if not raw_dir.is_absolute():
        raw_dir = (REPO_ROOT / raw_dir).resolve()
    data_dir = Path(str(args.data_dir))
    if not data_dir.is_absolute():
        data_dir = (REPO_ROOT / data_dir).resolve()
    out_dir = Path(str(args.out_dir))
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    route_map_path = data_dir / "route_map.json"
    if not route_map_path.is_file():
        raise FileNotFoundError(f"route_map.json not found: {route_map_path} (run preprocess_flight_delay.py first)")
    route_map = json.loads(route_map_path.read_text(encoding="utf-8"))
    routes = [str(r) for r in route_map.get("routes", [])]
    if not routes:
        raise RuntimeError(f"empty routes in: {route_map_path}")
    route_to_id = {r: i + 1 for i, r in enumerate(routes)}
    valid_routes = set(route_to_id.keys())
    n_pid = int(len(routes))

    bb = str(args.bb).strip().lower()
    if bb not in {"bba", "bbb"}:
        raise ValueError("--bb must be bbA or bbB")
    include_route = bb == "bbb"

    seed = int(args.seed)
    n_features = int(args.n_features)
    if n_features <= 0:
        raise ValueError("--n_features must be positive")

    max_rows = int(args.max_rows) if int(args.max_rows) > 0 else None
    chunksize = int(args.chunksize)
    if chunksize <= 0:
        raise ValueError("--chunksize must be positive")

    # Extract only the needed years.
    f2018 = _extract_from_zip(zip_path, member="Combined_Flights_2018.csv", out_dir=raw_dir)
    f2019 = _extract_from_zip(zip_path, member="Combined_Flights_2019.csv", out_dir=raw_dir)

    hasher = FeatureHasher(n_features=n_features, input_type="dict")
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(args.alpha),
        fit_intercept=True,
        learning_rate="optimal",
        random_state=seed,
    )

    print(f"[flight][train] bb={bb} include_route={include_route} seed={seed} n_pid={n_pid} n_features={n_features}", flush=True)

    # ---- Train on 2018 (backbone) ----
    n_fit = 0
    for ep in range(int(max(1, args.epochs))):
        print(f"[flight][train] epoch {ep+1}/{int(max(1,args.epochs))} ...", flush=True)
        first = True
        for n_rows, df in _iter_filtered_chunks(f2018, valid_routes=valid_routes, chunksize=chunksize, max_rows=max_rows):
            y = df["label"].to_numpy(dtype=np.int8, copy=False)
            feats = _chunk_features(df, include_route=include_route)
            X = hasher.transform(feats)
            if first:
                clf.partial_fit(X, y, classes=np.array([0, 1], dtype=np.int8))
                first = False
            else:
                clf.partial_fit(X, y)
            n_fit += int(y.size)
            if n_fit > 0 and (n_fit % 2_000_000) < int(y.size):
                print(f"[flight][train] seen {n_fit/1e6:.2f}M tokens (raw_rows={n_rows/1e6:.2f}M)", flush=True)

    print(f"[flight][train] done. n_fit={n_fit}", flush=True)

    # ---- Export logits for 2019 H1 (calib) and H2 (test) ----
    calib = {"logit": [], "y": [], "pid": [], "count": []}
    test = {"logit": [], "y": [], "pid": [], "count": []}

    base_year = 2018
    for n_rows, df in _iter_filtered_chunks(f2019, valid_routes=valid_routes, chunksize=chunksize, max_rows=max_rows):
        feats = _chunk_features(df, include_route=include_route)
        X = hasher.transform(feats)
        logit = clf.decision_function(X).astype(np.float32, copy=False).reshape(-1)
        y = df["label"].to_numpy(dtype=np.int8, copy=False).reshape(-1)
        route = df["route"].astype("string").tolist()
        pid = np.fromiter((route_to_id.get(str(r), 0) for r in route), dtype=np.int32, count=len(route))
        year = df["year"].to_numpy(dtype=np.int16, copy=False).reshape(-1)
        month = df["month"].to_numpy(dtype=np.int16, copy=False).reshape(-1)
        count = ((year.astype(np.int32) - int(base_year)) * 12 + (month.astype(np.int32) - 1)).astype(np.float32, copy=False)

        is_calib = month <= 6
        is_test = month >= 7

        if bool(np.any(is_calib)):
            m = is_calib
            calib["logit"].append(logit[m])
            calib["y"].append(y[m])
            calib["pid"].append(pid[m])
            calib["count"].append(count[m])
        if bool(np.any(is_test)):
            m = is_test
            test["logit"].append(logit[m])
            test["y"].append(y[m])
            test["pid"].append(pid[m])
            test["count"].append(count[m])

    def _cat(key: str, d: dict[str, list[np.ndarray]], dtype) -> np.ndarray:
        xs = d[key]
        if not xs:
            return np.zeros(0, dtype=dtype)
        return np.concatenate(xs, axis=0).astype(dtype, copy=False)

    out_prefix = f"flight_{bb}_s{seed}"
    calib_npz = out_dir / f"{out_prefix}_calib.npz"
    test_npz = out_dir / f"{out_prefix}_test.npz"

    np.savez_compressed(
        str(calib_npz),
        y=_cat("y", calib, np.int8),
        logit=_cat("logit", calib, np.float32),
        pid=_cat("pid", calib, np.int32),
        count=_cat("count", calib, np.float32),
        n_pid=np.asarray([n_pid], dtype=np.int32),
    )
    np.savez_compressed(
        str(test_npz),
        y=_cat("y", test, np.int8),
        logit=_cat("logit", test, np.float32),
        pid=_cat("pid", test, np.int32),
        count=_cat("count", test, np.float32),
        n_pid=np.asarray([n_pid], dtype=np.int32),
    )

    print(f"[ok] wrote: {calib_npz}", flush=True)
    print(f"[ok] wrote: {test_npz}", flush=True)


if __name__ == "__main__":
    main()
