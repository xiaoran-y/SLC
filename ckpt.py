from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    # Unwrap common wrappers so checkpoints are portable across:
    # - eager vs torch.compile (prefix: "_orig_mod.")
    # - DataParallel/DDP (prefix: "module.")
    m: torch.nn.Module = model
    while True:
        # DataParallel / DDP
        if hasattr(m, "module") and isinstance(getattr(m, "module"), torch.nn.Module):
            m = getattr(m, "module")
            continue
        # torch.compile wrapper
        if hasattr(m, "_orig_mod") and isinstance(getattr(m, "_orig_mod"), torch.nn.Module):
            m = getattr(m, "_orig_mod")
            continue
        break
    return m


def normalize_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    # Support common wrappers (possibly nested / different orders), e.g.:
    # - torch.compile: "_orig_mod."
    # - DataParallel/DDP: "module."
    prefixes = ("module.", "_orig_mod.")
    changed = False
    out: dict[str, Any] = {}
    for k, v in state.items():
        if not isinstance(k, str):
            out[k] = v
            continue
        kk = k
        while True:
            stripped = False
            for p in prefixes:
                if kk.startswith(p):
                    kk = kk[len(p) :]
                    stripped = True
                    changed = True
                    break
            if not stripped:
                break
        out[kk] = v
    return out if changed else state


def load_model_state(model: torch.nn.Module, state: dict[str, Any], *, strict: bool = True) -> None:
    target = unwrap_model(model)
    target.load_state_dict(normalize_state_dict(state), strict=bool(strict))


def resolve_model_dir(*, ckpt_root: str, model: str, dataset: str, save_tag: str) -> Path:
    if not save_tag:
        raise ValueError("save_tag must be non-empty to resolve checkpoint directory")
    root = Path(str(ckpt_root))
    return root / str(model) / f"{str(dataset)}_{str(save_tag)}"


def save_best(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    best_valid_auc: float,
    args_dict: dict[str, Any],
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": int(epoch),
        "best_valid_auc": float(best_valid_auc),
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "args": args_dict,
        "meta": meta,
        "torch_version": torch.__version__,
    }
    torch.save(ckpt, str(path))

    # Also keep a tiny JSON for quick inspection without torch.
    try:
        info = {
            "epoch": int(epoch),
            "best_valid_auc": float(best_valid_auc),
            "model": str(args_dict.get("model", "")),
            "dataset": str(args_dict.get("dataset", "")),
            "save_tag": str(args_dict.get("save_tag", "")),
        }
        path.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass


def load(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict:
    # PyTorch >=2.6 defaults torch.load(weights_only=True); our checkpoints are
    # dicts with metadata, so we opt out of weights-only mode.
    return torch.load(str(path), map_location=map_location, weights_only=False)
