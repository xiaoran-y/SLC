#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = REPO_ROOT / "configs"


def _now_ts() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_name(s: str) -> str:
    s = str(s).strip()
    return "".join(c if (c.isalnum() or c in {"-", "_", "."}) else "_" for c in s)


def _merge_dict(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_config_path(spec: str, *, default_dir: str | None = None) -> Path:
    spec = str(spec).strip()
    if not spec:
        raise ValueError("empty config spec")

    # If spec is a real path (absolute or relative), use it directly.
    p = Path(spec)
    if p.is_absolute() or spec.startswith(".") or "/" in spec:
        if p.suffix != ".json":
            p = p.with_suffix(".json")
        if not p.is_absolute():
            p = REPO_ROOT / p
        return p

    # Otherwise treat as a named config under CONFIG_ROOT.
    if default_dir is None:
        raise ValueError(f"config name {spec!r} needs default_dir")
    name = spec if spec.endswith(".json") else (spec + ".json")
    return CONFIG_ROOT / default_dir / name


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_with_includes(path: Path, *, seen: set[Path]) -> dict:
    path = path.resolve()
    if path in seen:
        raise RuntimeError(f"Config include cycle detected at: {path}")
    seen.add(path)

    cfg_raw = _load_json(path)
    includes = cfg_raw.get("include", [])
    if includes is None:
        includes = []
    if not isinstance(includes, list):
        raise TypeError(f"'include' must be a list in {path}")

    merged: dict = {}
    for inc in includes:
        inc_s = str(inc).strip()
        if not inc_s:
            continue
        inc_p = Path(inc_s)
        if not inc_p.is_absolute():
            # includes are relative to CONFIG_ROOT
            if not inc_s.endswith(".json"):
                inc_s = inc_s + ".json"
            inc_p = (CONFIG_ROOT / inc_s).resolve()
        merged = _merge_dict(merged, _load_with_includes(inc_p, seen=seen))

    # Overlay current file (ignore reserved keys).
    own = {k: v for k, v in cfg_raw.items() if k != "include" and not str(k).startswith("_")}
    merged = _merge_dict(merged, own)
    return merged


def _parse_set_kv(s: str) -> tuple[str, object]:
    if "=" not in s:
        raise ValueError(f"--set expects key=value, got: {s!r}")
    k, v = s.split("=", 1)
    k = k.strip()
    v = v.strip()
    if not k:
        raise ValueError(f"empty key in --set: {s!r}")
    lv = v.lower()
    if lv == "true":
        return k, True
    if lv == "false":
        return k, False
    # number parsing
    try:
        if any(c in v for c in [".", "e", "E"]):
            return k, float(v)
        return k, int(v)
    except Exception:
        return k, v


def _parse_seeds(s: str) -> list[int]:
    s = str(s).strip()
    if not s:
        return []
    parts = []
    for chunk in s.replace(",", " ").split():
        if not chunk:
            continue
        parts.append(int(chunk))
    return parts


def _sanitize_thread_env(env: dict[str, str]) -> dict[str, str]:
    """Sanitize OpenMP/BLAS thread env vars (avoid libgomp warnings on invalid values)."""

    def _normalize_int(v: object) -> str | None:
        s = str(v).strip()
        if not s:
            return None
        # Some environments set lists like "4,1". libgomp may reject these.
        # Take the first component if it is a positive int.
        head = s.split(",", 1)[0].strip()
        try:
            n = int(head)
            return str(n) if n > 0 else None
        except Exception:
            return None

    for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        if k not in env:
            continue
        v = _normalize_int(env.get(k))
        if v is None:
            env[k] = "1"
        else:
            env[k] = v
    return env


def _build_main_cmd(cfg: dict, *, seed: int) -> list[str]:
    cmd = [sys.executable, "-u", str(REPO_ROOT / "main.py")]

    # Required.
    if "dataset" not in cfg:
        raise KeyError("config missing required key: dataset")
    if "model" not in cfg:
        cfg = dict(cfg)
        cfg["model"] = "akt_pid"

    reserved = {"include", "seed", "seeds"}
    for k, v in cfg.items():
        if str(k).startswith("_") or k in reserved:
            continue
        if v is None:
            continue
        if k == "keep_model_files":
            if bool(v):
                cmd.append("--keep_model_files")
            continue
        cmd.extend([f"--{k}", str(v)])

    # Seed override per run.
    cmd.extend(["--seed", str(int(seed))])
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper-facing training entrypoint.")
    ap.add_argument("--exp", type=str, required=True, help="Experiment name under configs/experiments/ (no suffix).")
    ap.add_argument(
        "--configs",
        type=str,
        default="",
        help="Optional extra configs (comma-separated). Paths or names under configs/ (use with care).",
    )
    ap.add_argument("--seeds", type=str, default="", help="Seeds, e.g. '225,226,227'. Default: config.seed or 224.")
    ap.add_argument("--ts", type=str, default="", help="Run timestamp (default: now).")
    ap.add_argument("--out_root", type=str, default="_runs", help="Where to write logs/config snapshots.")
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="Resume-friendly: skip a seed if its expected best.pt already exists under ckpt_root.",
    )
    ap.add_argument("--set", action="append", default=[], help="Override config key=value (repeatable).")
    ap.add_argument("--dry_run", action="store_true", help="Print commands and exit.")
    args = ap.parse_args()

    ts = str(args.ts).strip() or _now_ts()
    exp_name = _safe_name(args.exp)

    exp_path = _resolve_config_path(exp_name, default_dir="experiments")
    cfg = _load_with_includes(exp_path, seen=set())

    extra = [c.strip() for c in str(args.configs).split(",") if c.strip()]
    for c in extra:
        # Support:
        #   - repo-root relative paths (e.g., configs/runtimes/gpu_fast)
        #   - CONFIG_ROOT relative paths (e.g., runtimes/gpu_fast)
        #   - bare names under CONFIG_ROOT (use with care)
        p_in = Path(c)
        if p_in.suffix != ".json":
            p_in = p_in.with_suffix(".json")

        candidates: list[Path] = []
        if p_in.is_absolute():
            candidates.append(p_in)
        else:
            candidates.append((REPO_ROOT / p_in).resolve())
            candidates.append((CONFIG_ROOT / p_in).resolve())

        p = None
        for cand in candidates:
            if cand.is_file():
                p = cand
                break
        if p is None:
            tried = "\n  - ".join(str(x) for x in candidates)
            raise FileNotFoundError(f"extra config not found: {c!r}\nTried:\n  - {tried}")

        cfg = _merge_dict(cfg, _load_with_includes(p, seen=set()))

    # Apply CLI overrides.
    for s in list(args.set):
        k, v = _parse_set_kv(str(s))
        cfg[k] = v

    seeds = _parse_seeds(args.seeds)
    if not seeds:
        if "seeds" in cfg:
            if isinstance(cfg["seeds"], list):
                seeds = [int(x) for x in cfg["seeds"]]
            else:
                seeds = _parse_seeds(str(cfg["seeds"]))
        elif "seed" in cfg:
            seeds = [int(cfg["seed"])]
        else:
            seeds = [224]

    # Ensure save_tag is unique per run and per seed (avoid clobbering checkpoints/results).
    base_tag = str(cfg.get("save_tag", exp_name)).strip() or exp_name
    save_tag_root = f"{base_tag}_{ts}"
    cfg["save_tag"] = save_tag_root

    out_root = Path(str(args.out_root))
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    run_dir = out_root / f"train_{exp_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config_merged.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "exp.json").write_text(exp_path.read_text(encoding="utf-8"), encoding="utf-8")

    cmd_lines: list[str] = []
    seed_save_tags: dict[str, str] = {}
    seed_best_paths: dict[str, str] = {}
    for seed in seeds:
        cfg_seed = dict(cfg)
        cfg_seed["save_tag"] = f"{save_tag_root}_s{int(seed)}"
        seed_save_tags[str(int(seed))] = cfg_seed["save_tag"]
        if bool(cfg_seed.get("keep_model_files", False)):
            ckpt_root = Path(str(cfg_seed.get("ckpt_root", "_ckpts")))
            if not ckpt_root.is_absolute():
                ckpt_root = REPO_ROOT / ckpt_root
            model = str(cfg_seed.get("model", "")).strip()
            dataset = str(cfg_seed.get("dataset", "")).strip()
            save_tag = str(cfg_seed.get("save_tag", "")).strip()
            if model and dataset and save_tag:
                best_path = ckpt_root / model / f"{dataset}_{save_tag}" / "best.pt"
                seed_best_paths[str(int(seed))] = str(best_path)
        cmd = _build_main_cmd(cfg_seed, seed=int(seed))
        cmd_lines.append(" ".join(shlex.quote(x) for x in cmd))

    (run_dir / "commands.sh").write_text("\n".join(cmd_lines) + "\n", encoding="utf-8")
    (run_dir / "save_tags.json").write_text(
        json.dumps(seed_save_tags, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "expected_best_ckpts.json").write_text(
        json.dumps(seed_best_paths, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.dry_run:
        print(f"[dry_run] run_dir={run_dir}", flush=True)
        print("\n".join(cmd_lines), flush=True)
        return

    for seed, cmd_s in zip(seeds, cmd_lines, strict=True):
        if args.skip_existing:
            best_p = seed_best_paths.get(str(int(seed)), "")
            if best_p and Path(best_p).is_file():
                print(f"[skip] exp={exp_name} seed={int(seed)} best_ckpt={best_p}", flush=True)
                continue
        log_dir = run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"s{int(seed)}.train.log"
        print(f"[run] exp={exp_name} seed={int(seed)} log={log_path}", flush=True)
        log_path.write_text(cmd_s + "\n\n", encoding="utf-8")
        with log_path.open("a", encoding="utf-8") as f:
            env = _sanitize_thread_env(os.environ.copy())
            p = subprocess.run(
                cmd_s,
                cwd=str(REPO_ROOT),
                shell=True,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
            )
        if p.returncode != 0:
            raise SystemExit(f"[error] seed={int(seed)} failed (code={p.returncode}). See: {log_path}")

    print(f"[ok] done. run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
