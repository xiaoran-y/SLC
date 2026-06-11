from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

import ckpt
from load_data import DATA, PID_DATA
from run import test, train
from utils import load_model, model_isPid_type


def _apply_seeds(seed: int, *, deterministic: bool) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _is_on(v: object) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _apply_runtime(args) -> None:
    # Fast GPU knobs (safe defaults).
    if torch.cuda.is_available() and _is_on(getattr(args, "fast_gpu", "off")):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Inductor cudagraphs toggle (used by torch.compile modes; force to match args).
    if torch.cuda.is_available():
        cudagraphs = bool(_is_on(getattr(args, "inductor_cudagraphs", "off")))
        try:
            import torch._inductor.config as _inductor_config  # type: ignore

            _inductor_config.triton.cudagraphs = cudagraphs
            _inductor_config.triton.cudagraph_trees = cudagraphs
        except Exception:
            if cudagraphs:
                print("[runtime] warning: failed to enable inductor cudagraphs; ignored.", flush=True)


def _maybe_compile_model(model: torch.nn.Module, args) -> torch.nn.Module:
    mode = str(getattr(args, "torch_compile", "off")).strip()
    # Safety: some backbones contain Python loops / data-dependent control flow
    # which can cause torch.compile to explode in compile-time memory (especially
    # on long sequences). LPKT is a known example (per-step loop over seq_len).
    # Prefer a stable eager path for paper runs.
    try:
        _pid_flag, _model_type = model_isPid_type(str(getattr(args, "model", "")))
    except Exception:
        _model_type = ""
    if str(_model_type).strip().lower() == "lpkt" and mode.lower() not in {"", "off", "false", "0", "none"}:
        print("[runtime] torch.compile=off for lpkt (python loop; high compile memory).", flush=True)
        return model
    if mode.lower() in {"", "off", "false", "0", "none"}:
        return model
    if mode.lower() in {"on", "true", "1"}:
        mode = "default"
    if not hasattr(torch, "compile"):
        print("[runtime] warning: torch.compile not available; ignored.", flush=True)
        return model
    # Some torch.compile modes may flip inductor knobs; force cudagraph settings to match args.
    cudagraphs = bool(_is_on(getattr(args, "inductor_cudagraphs", "off")))
    try:
        try:
            import torch._inductor.config as _inductor_config  # type: ignore

            _inductor_config.triton.cudagraphs = cudagraphs
            _inductor_config.triton.cudagraph_trees = cudagraphs
        except Exception:
            pass
        compiled = torch.compile(model, mode=mode)
        try:
            import torch._inductor.config as _inductor_config  # type: ignore

            _inductor_config.triton.cudagraphs = cudagraphs
            _inductor_config.triton.cudagraph_trees = cudagraphs
        except Exception:
            pass
        print(f"[runtime] torch.compile=on mode={mode}", flush=True)
        return compiled
    except Exception as e:
        msg = str(e)
        if len(msg) > 300:
            msg = msg[:300] + "…"
        print(f"[runtime] warning: torch.compile failed ({type(e).__name__}: {msg}); fallback to eager.", flush=True)
        return model


def _read_meta(dataset_dir: Path) -> dict:
    meta_path = dataset_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _resolve_under_repo(repo_root: Path, p: str) -> Path:
    path = Path(str(p))
    return path if path.is_absolute() else (repo_root / path)


def run_experiment(args) -> None:
    repo_root = Path(__file__).resolve().parent

    data_root = _resolve_under_repo(repo_root, str(getattr(args, "data_root", "dataset")))
    ckpt_root = _resolve_under_repo(repo_root, str(getattr(args, "ckpt_root", "_ckpts")))
    _ = _resolve_under_repo(repo_root, str(getattr(args, "result_root", "_runs/result")))

    dataset = str(getattr(args, "dataset"))
    dataset_dir = data_root / dataset

    meta = _read_meta(dataset_dir)
    args.n_question = int(meta["n_question"])
    args.n_pid = int(meta.get("n_pid", 0))
    args.n_users = int(meta.get("n_users", 0))
    args.seqlen = int(meta.get("seqlen", int(getattr(args, "seqlen", 200))))
    args.data_dir = str(dataset_dir)
    args.data_name = dataset

    is_pid_model, _model_type = model_isPid_type(str(getattr(args, "model", "")))
    if is_pid_model and int(args.n_pid) <= 0:
        raise ValueError(f"PID model requested but meta.n_pid<=0 for dataset={dataset!r}")

    # Loader selection: decide by dataset capability (pid columns), not by backbone.
    has_pid = int(args.n_pid) > 0
    if has_pid:
        dat = PID_DATA(n_question=args.n_question, seqlen=args.seqlen, separate_char=",")
    else:
        dat = DATA(n_question=args.n_question, seqlen=args.seqlen, separate_char=",")

    ts = int(getattr(args, "train_set", 1))
    train_path = dataset_dir / f"{dataset}_train{ts}.csv"
    valid_path = dataset_dir / f"{dataset}_valid{ts}.csv"
    test_path = dataset_dir / f"{dataset}_test{ts}.csv"

    if not train_path.is_file() or not valid_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(f"Missing split files under {dataset_dir} (train/valid/test for train_set={ts}).")

    if has_pid:
        train_q, train_qa, train_pid, train_uid, train_count = dat.load_data(str(train_path))
        valid_q, valid_qa, valid_pid, valid_uid, valid_count = dat.load_data(str(valid_path))
        test_q, test_qa, test_pid, test_uid, test_count = dat.load_data(str(test_path))
    else:
        train_q, train_qa, train_uid, train_count = dat.load_data(str(train_path))
        valid_q, valid_qa, valid_uid, valid_count = dat.load_data(str(valid_path))
        test_q, test_qa, test_uid, test_count = dat.load_data(str(test_path))
        train_pid = valid_pid = test_pid = None

    # Seeds / determinism.
    deterministic = str(getattr(args, "deterministic", "off")).lower() == "on"
    _apply_seeds(int(getattr(args, "seed", 224)), deterministic=deterministic)
    _apply_runtime(args)

    # Print a compact args dump (paper logs are captured by train.py).
    d = vars(args)
    for k in sorted(d.keys()):
        if k.startswith("_"):
            continue
        print("\t", k, "\t", d[k], flush=True)

    print("train_q_data.shape", train_q.shape, flush=True)
    print("train_qa_data.shape", train_qa.shape, flush=True)
    print("valid_q_data.shape", valid_q.shape, flush=True)
    print("valid_qa_data.shape", valid_qa.shape, flush=True)

    # Model / optimizer.
    load_ckpt_path = str(getattr(args, "load_ckpt", "")).strip()
    if load_ckpt_path:
        load_ckpt_path = str(_resolve_under_repo(repo_root, load_ckpt_path))

    model = load_model(args)
    if load_ckpt_path:
        if not Path(load_ckpt_path).is_file():
            raise FileNotFoundError(f"--load_ckpt not found: {load_ckpt_path}")
        ck = ckpt.load(load_ckpt_path, map_location="cpu")
        state = ck.get("model_state_dict", ck)
        ckpt.load_model_state(model, state, strict=True)
        try:
            best_epoch = int(ck.get("epoch", 0))
        except Exception:
            best_epoch = 0
        try:
            best_valid_auc = float(ck.get("best_valid_auc", float("nan")))
        except Exception:
            best_valid_auc = float("nan")
        print(
            f"[eval_only] loaded_ckpt={load_ckpt_path} epoch={ck.get('epoch', 'na')} best_valid_auc={ck.get('best_valid_auc', 'na')}",
            flush=True,
        )

    model = _maybe_compile_model(model, args)
    optimizer = None
    if not load_ckpt_path:
        wd = float(getattr(args, "weight_decay", 0.0))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(getattr(args, "lr", 2e-5)), weight_decay=wd)

    # Train loop.
    max_iter = int(getattr(args, "max_iter", 600))
    patience = int(getattr(args, "early_stop_patience", 40))
    warmup = int(getattr(args, "early_stop_warmup", 0))
    min_delta = float(getattr(args, "early_stop_min_delta", 0.0))

    best_state = None
    if not load_ckpt_path:
        best_valid_auc = float("-inf")
        best_epoch = 0

    save_tag = str(getattr(args, "save_tag", "")).strip() or dataset
    args.save_tag = save_tag

    save_best = bool(getattr(args, "keep_model_files", False))
    best_path = None
    if save_best:
        model_dir = ckpt.resolve_model_dir(ckpt_root=str(ckpt_root), model=str(args.model), dataset=dataset, save_tag=save_tag)
        best_path = model_dir / "best.pt"

    if not load_ckpt_path:
        assert optimizer is not None
        for epoch in range(max_iter):
            train_loss, train_acc, train_auc = train(
                model,
                args,
                optimizer,
                train_q,
                train_qa,
                train_pid,
                uid_data=train_uid,
                count_data=train_count,
                label="Train",
            )
            valid_loss, valid_acc, valid_auc, *_ = test(
                model,
                args,
                None,
                valid_q,
                valid_qa,
                valid_pid,
                uid_data=valid_uid,
                count_data=valid_count,
                label="Valid",
                return_outputs=False,
            )

            ep = epoch + 1
            print("epoch", ep, flush=True)
            print("valid_auc\t", valid_auc, "\ttrain_auc\t", train_auc, flush=True)
            print("valid_accuracy\t", valid_acc, "\ttrain_accuracy\t", train_acc, flush=True)
            print("valid_loss\t", valid_loss, "\ttrain_loss\t", train_loss, flush=True)

            improved = False
            if math.isnan(best_valid_auc) or (valid_auc > best_valid_auc + float(min_delta)):
                best_valid_auc = float(valid_auc)
                best_epoch = ep
                improved = True

            if improved:
                if save_best:
                    assert best_path is not None
                    ckpt.save_best(
                        path=best_path,
                        model=model,
                        optimizer=optimizer,
                        epoch=ep,
                        best_valid_auc=best_valid_auc,
                        args_dict=vars(args).copy(),
                        meta=meta,
                    )
                else:
                    best_state = copy.deepcopy(ckpt.unwrap_model(model).state_dict())

            if ep >= int(warmup) and patience > 0 and best_epoch > 0:
                if (ep - best_epoch) >= patience:
                    break

    # Load best for final test (unless eval-only ckpt already loaded).
    if not load_ckpt_path:
        if save_best:
            assert best_path is not None
            if not best_path.is_file():
                raise FileNotFoundError(f"best checkpoint not found: {best_path}")
            ck = torch.load(str(best_path), map_location="cpu", weights_only=False)
            ckpt.load_model_state(model, ck["model_state_dict"], strict=True)
        else:
            if best_state is not None:
                ckpt.load_model_state(model, best_state, strict=True)

    print("\n\nStart testing ......................\n Best epoch:", best_epoch, flush=True)
    test_loss, test_acc, test_auc, *_ = test(
        model,
        args,
        None,
        test_q,
        test_qa,
        test_pid,
        uid_data=test_uid,
        count_data=test_count,
        label="Test",
        return_outputs=False,
    )
    print("\ntest_auc\t", test_auc, flush=True)
    print("test_accuracy\t", test_acc, flush=True)
    print("test_loss\t", test_loss, flush=True)
