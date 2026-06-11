#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd


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


def _count_routes(
    csv_path: Path,
    *,
    chunksize: int,
    max_rows: int | None,
) -> tuple[Counter[str], dict[str, int]]:
    usecols = ["Cancelled", "Diverted", "ArrDel15", "Origin", "Dest"]
    dtype = {
        "Cancelled": "int8",
        "Diverted": "int8",
        "ArrDel15": "float32",
        "Origin": "string",
        "Dest": "string",
    }
    counts: Counter[str] = Counter()
    n_rows = 0
    n_kept = 0

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
        sub = chunk.loc[m, ["Origin", "Dest"]]
        n_kept += int(len(sub))
        routes = (sub["Origin"].astype("string") + "-" + sub["Dest"].astype("string")).astype("string")
        vc = routes.value_counts(dropna=True)
        counts.update({str(k): int(v) for k, v in vc.items()})

        if max_rows is not None and n_rows >= int(max_rows):
            break

    return counts, {"n_rows": int(n_rows), "n_kept": int(n_kept)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess flight-delay data: extract 2018/2019 and build valid route map.")
    ap.add_argument("--zip_path", type=str, default="_raw_data/flight-delay-dataset-20182022.zip")
    ap.add_argument("--raw_dir", type=str, default="dataset/flight_delay_raw")
    ap.add_argument("--out_dir", type=str, default="dataset/flight_delay")
    ap.add_argument("--years", type=str, default="2018,2019", help="Comma-separated years, default: 2018,2019")
    ap.add_argument("--min_route_count_per_year", type=int, default=50)
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--max_rows", type=int, default=0, help="Debug: stop after reading N rows per year (0 disables).")
    ap.add_argument("--keep_extracted", type=str, default="on", help="Keep extracted CSVs under raw_dir (on/off).")
    args = ap.parse_args()

    zip_path = Path(str(args.zip_path))
    if not zip_path.is_absolute():
        zip_path = (REPO_ROOT / zip_path).resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"zip not found: {zip_path}")

    raw_dir = Path(str(args.raw_dir))
    if not raw_dir.is_absolute():
        raw_dir = (REPO_ROOT / raw_dir).resolve()
    out_dir = Path(str(args.out_dir))
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    years = [int(x) for x in str(args.years).replace(" ", "").split(",") if x.strip()]
    if not years:
        raise ValueError("--years must be non-empty, e.g. 2018,2019")
    if len(years) != 2:
        raise ValueError("This script currently expects exactly two years (e.g., 2018,2019) for the paper protocol.")
    y0, y1 = int(years[0]), int(years[1])

    max_rows = int(args.max_rows) if int(args.max_rows) > 0 else None
    min_cnt = int(args.min_route_count_per_year)
    if min_cnt <= 0:
        raise ValueError("--min_route_count_per_year must be positive")

    print(f"[flight] zip={zip_path}", flush=True)
    print(f"[flight] years={y0},{y1} min_route_count_per_year={min_cnt} chunksize={int(args.chunksize)}", flush=True)

    f0 = _extract_from_zip(zip_path, member=f"Combined_Flights_{y0}.csv", out_dir=raw_dir)
    f1 = _extract_from_zip(zip_path, member=f"Combined_Flights_{y1}.csv", out_dir=raw_dir)

    c0, s0 = _count_routes(f0, chunksize=int(args.chunksize), max_rows=max_rows)
    c1, s1 = _count_routes(f1, chunksize=int(args.chunksize), max_rows=max_rows)

    valid_routes = sorted([r for r, n in c0.items() if int(n) >= min_cnt and int(c1.get(r, 0)) >= min_cnt])
    routes_path = out_dir / "valid_routes.json"
    routes_path.write_text(json.dumps(valid_routes, indent=2) + "\n", encoding="utf-8")

    route_map = {"n_routes": int(len(valid_routes)), "routes": valid_routes}
    (out_dir / "route_map.json").write_text(json.dumps(route_map, indent=2) + "\n", encoding="utf-8")

    meta = {
        "source": "flight-delay-dataset-20182022.zip",
        "zip_path": str(zip_path.relative_to(REPO_ROOT) if REPO_ROOT in zip_path.parents else zip_path),
        "raw_dir": str(raw_dir.relative_to(REPO_ROOT) if REPO_ROOT in raw_dir.parents else raw_dir),
        "years": [y0, y1],
        "min_route_count_per_year": int(min_cnt),
        "chunksize": int(args.chunksize),
        "pass1": {str(y0): s0, str(y1): s1},
        "n_valid_routes": int(len(valid_routes)),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"[ok] wrote: {routes_path}", flush=True)
    print(f"[ok] n_valid_routes={len(valid_routes)}", flush=True)

    keep = str(args.keep_extracted).strip().lower()
    if keep not in {"on", "off"}:
        raise ValueError("--keep_extracted must be on/off")
    if keep == "off":
        try:
            f0.unlink(missing_ok=True)
            f1.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
