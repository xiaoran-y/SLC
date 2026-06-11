from __future__ import annotations

import argparse

from experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper-facing KT runner.")

    # Core selection.
    p.add_argument("--dataset", type=str, required=True, help="Dataset directory name under --data_root.")
    p.add_argument(
        "--model",
        type=str,
        default="akt_pid",
        help="Model name, e.g. akt_pid / akt / dkt / sakt / dkvmn / lpkt / mf_pid / ncf_pid.",
    )
    p.add_argument("--train_set", type=int, default=1, help="Split id (train1/valid1/test1).")
    p.add_argument("--seed", type=int, default=224)

    # Workspace roots.
    p.add_argument("--data_root", type=str, default="dataset")
    p.add_argument("--ckpt_root", type=str, default="_ckpts")
    p.add_argument("--result_root", type=str, default="_runs/result")
    p.add_argument("--save_tag", type=str, default="", help="Run tag used in ckpt directory naming.")
    p.add_argument("--keep_model_files", action="store_true", help="Save best checkpoint under --ckpt_root.")
    p.add_argument(
        "--load_ckpt",
        type=str,
        default="",
        help="Optional: load a .pt checkpoint and skip training for evaluation only.",
    )

    # Training.
    p.add_argument("--max_iter", type=int, default=600)
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--l2", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--maxgradnorm", type=float, default=-1.0)
    p.add_argument("--early_stop_patience", type=int, default=40)
    p.add_argument("--early_stop_warmup", type=int, default=0)
    p.add_argument("--early_stop_min_delta", type=float, default=0.0)

    # AKT backbone.
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_block", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--kq_same", type=int, default=1)
    p.add_argument("--final_fc_dim", type=int, default=512)
    p.add_argument("--seqlen", type=int, default=200, help="Max sequence length used by the dataset loader.")
    p.add_argument("--separate_qa", type=str, default="off", choices=["off", "on"])

    # DKVMN extras.
    p.add_argument("--dkvmn_size_m", type=int, default=32, help="DKVMN memory slots (size_m).")

    # LPKT backbone.
    p.add_argument("--lpkt_d_a", type=int, default=64)
    p.add_argument("--lpkt_d_e", type=int, default=64)
    p.add_argument("--lpkt_d_k", type=int, default=64)
    p.add_argument("--lpkt_dropout", type=float, default=0.1)
    p.add_argument("--lpkt_gamma", type=float, default=0.0)
    p.add_argument("--lpkt_use_time", type=str, default="on", choices=["off", "on"])
    p.add_argument("--lpkt_n_it", type=int, default=16)
    p.add_argument("--lpkt_n_at", type=int, default=16)

    # RecSys backbones (no-KT control).
    p.add_argument("--rec_emb_dim", type=int, default=64, help="Embedding dim for MF/NCF.")
    p.add_argument("--ncf_hidden_dims", type=str, default="128,64,32", help="NCF MLP dims, e.g. '256,128,64'.")
    p.add_argument("--ncf_dropout", type=float, default=0.1, help="NCF dropout.")

    # Runtime knobs (kept for paper scripts; some are no-ops in this minimal engine).
    p.add_argument("--fast_gpu", type=str, default="off", choices=["off", "on"])
    p.add_argument("--amp", type=str, default="off", choices=["off", "bf16", "fp16"])
    p.add_argument("--torch_compile", type=str, default="off")
    p.add_argument("--inductor_cudagraphs", type=str, default="off")
    p.add_argument("--data_backend", type=str, default="auto")
    p.add_argument("--pin_memory", type=str, default="auto")
    p.add_argument("--deterministic", type=str, default="off", choices=["off", "on"])
    p.add_argument("--metrics_backend", type=str, default="sklearn")

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    # Normalize a few legacy flag styles.
    args.separate_qa = str(args.separate_qa).lower() == "on"
    run_experiment(args)
