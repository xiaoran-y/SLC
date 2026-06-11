from __future__ import annotations

import os
from pathlib import Path

import torch

from model.akt import AKT
from model.dkt import DKT
from model.dkvmn import DKVMN
from model.lpkt import LPKT
from model.mf import MF
from model.ncf import NCF
from model.sakt import SAKT


def try_makedirs(path_: str | Path) -> None:
    path_s = str(path_)
    if not os.path.isdir(path_s):
        os.makedirs(path_s, exist_ok=True)


def model_isPid_type(model_name: str) -> tuple[bool, str]:
    words = str(model_name).split("_")
    is_pid = "pid" in words
    return is_pid, words[0] if words else ""


def _resolve_device(params=None) -> torch.device:
    # Keep behavior simple and stable: use CUDA when available.
    _ = params
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(params) -> torch.nn.Module:
    is_pid, model_type = model_isPid_type(getattr(params, "model", ""))
    model_type = str(model_type).strip().lower()

    n_question = int(getattr(params, "n_question"))
    if n_question <= 0:
        raise ValueError(f"n_question must be >0, got {n_question}")

    n_pid = int(getattr(params, "n_pid", 0)) if is_pid else 0
    if is_pid and n_pid <= 0:
        raise ValueError("PID model requires n_pid > 0 (dataset meta must include n_pid).")
    n_users = int(getattr(params, "n_users", 0))

    # Most PyKT-style backbones assume ids are in [0..n_question] where 0 is padding,
    # hence num_c = n_question + 1.
    num_c = int(n_question) + 1

    if model_type == "akt":
        model = AKT(
            n_question=n_question,
            n_pid=n_pid,
            d_model=int(getattr(params, "d_model", 256)),
            n_blocks=int(getattr(params, "n_block", 1)),
            kq_same=int(getattr(params, "kq_same", 1)),
            dropout=float(getattr(params, "dropout", 0.05)),
            model_type="akt",
            final_fc_dim=int(getattr(params, "final_fc_dim", 512)),
            n_heads=int(getattr(params, "n_head", 8)),
            d_ff=int(getattr(params, "d_ff", 2048)),
            l2=float(getattr(params, "l2", 1e-5)),
            separate_qa=bool(getattr(params, "separate_qa", False)),
        )
        return model.to(_resolve_device(params))

    if model_type == "dkt":
        model = DKT(
            num_c=num_c,
            emb_size=int(getattr(params, "d_model", 256)),
            dropout=float(getattr(params, "dropout", 0.5)),
            emb_type="qid",
        )
        return model.to(_resolve_device(params))

    if model_type == "sakt":
        model = SAKT(
            num_c=num_c,
            seq_len=int(getattr(params, "seqlen", 200)),
            emb_size=int(getattr(params, "d_model", 64)),
            num_attn_heads=int(getattr(params, "n_head", 8)),
            dropout=float(getattr(params, "dropout", 0.5)),
            num_en=int(getattr(params, "n_block", 2)),
            emb_type="qid",
        )
        return model.to(_resolve_device(params))

    if model_type == "dkvmn":
        model = DKVMN(
            num_c=num_c,
            dim_s=int(getattr(params, "d_model", 256)),
            size_m=int(getattr(params, "dkvmn_size_m", 32)),
            dropout=float(getattr(params, "dropout", 0.1)),
            emb_type="qid",
        )
        return model.to(_resolve_device(params))

    if model_type == "lpkt":
        # Paper workspace convention:
        # - Use concept/qid sequences as "exercises" (qid mode).
        # - Time features are optional; by default we pass constant interval bins
        #   (consistent with common LPKT pipelines when timestamps are missing).
        n_at = int(getattr(params, "lpkt_n_at", 16))
        n_it = int(getattr(params, "lpkt_n_it", 16))
        d_a = int(getattr(params, "lpkt_d_a", 64))
        d_e = int(getattr(params, "lpkt_d_e", 64))
        d_k = int(getattr(params, "lpkt_d_k", 64))
        dropout = float(getattr(params, "lpkt_dropout", 0.1))
        gamma = float(getattr(params, "lpkt_gamma", 0.0))
        use_time = str(getattr(params, "lpkt_use_time", "on")).strip().lower() == "on"

        dev = _resolve_device(params)
        # Identity Q-matrix: each concept is treated as its own exercise.
        q_matrix = torch.eye(int(n_question) + 1, dtype=torch.float32, device=dev)
        model = LPKT(
            n_at=n_at,
            n_it=n_it,
            n_exercise=int(n_question),
            n_question=int(n_question),
            d_a=d_a,
            d_e=d_e,
            d_k=d_k,
            gamma=gamma,
            dropout=dropout,
            q_matrix=q_matrix,
            emb_type="qid",
            use_time=use_time,
        )
        return model.to(dev)

    if model_type == "mf":
        if n_users <= 0:
            raise ValueError("MF requires n_users > 0 (dataset meta must include n_users).")
        emb_dim = int(getattr(params, "rec_emb_dim", getattr(params, "d_model", 64)))
        model = MF(n_users=int(n_users), n_items=int(n_pid), emb_dim=int(emb_dim))
        return model.to(_resolve_device(params))

    if model_type == "ncf":
        if n_users <= 0:
            raise ValueError("NCF requires n_users > 0 (dataset meta must include n_users).")
        emb_dim = int(getattr(params, "rec_emb_dim", getattr(params, "d_model", 64)))
        hidden_dims = str(getattr(params, "ncf_hidden_dims", "128,64,32"))
        dropout = float(getattr(params, "ncf_dropout", 0.1))
        model = NCF(
            n_users=int(n_users),
            n_items=int(n_pid),
            emb_dim=int(emb_dim),
            hidden_dims=hidden_dims,
            dropout=float(dropout),
        )
        return model.to(_resolve_device(params))

    raise ValueError(f"Unsupported model_type: {model_type!r} (model={getattr(params, 'model', '')!r})")
