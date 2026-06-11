# Code adapted from https://github.com/pykt-team/pykt-toolkit
from __future__ import annotations

from contextlib import nullcontext
import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from utils import model_isPid_type

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def binary_entropy(target: np.ndarray, pred: np.ndarray) -> float:
    target = target.astype(np.float64, copy=False)
    pred = np.clip(pred.astype(np.float64, copy=False), 1e-10, 1.0 - 1e-10)
    loss = target * np.log(pred) + (1.0 - target) * np.log(1.0 - pred)
    return float((-loss).mean())


def compute_auc(target: np.ndarray, pred: np.ndarray) -> float:
    try:
        return float(metrics.roc_auc_score(target, pred))
    except Exception:
        return float("nan")


def compute_accuracy(target: np.ndarray, pred: np.ndarray) -> float:
    pred_label = (pred > 0.5).astype(np.float32)
    return float(metrics.accuracy_score(target.astype(np.float32), pred_label))


def _autocast_context(params):
    amp = str(getattr(params, "amp", "off")).lower()
    if amp in {"off", "none", ""}:
        return nullcontext()
    if not torch.cuda.is_available():
        return nullcontext()
    if amp == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if amp == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _bce_loss_vec(pred_prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # torch.nn.functional.binary_cross_entropy is not safe under autocast.
    # Compute it in fp32 with autocast disabled (CUDA) to support amp=bf16/fp16.
    pred32 = pred_prob.float()
    target32 = target.float()
    if pred32.is_cuda:
        with torch.autocast(device_type="cuda", enabled=False):
            return F.binary_cross_entropy(pred32, target32, reduction="none")
    return F.binary_cross_entropy(pred32, target32, reduction="none")


def _qa_to_target(qa: np.ndarray, *, n_question: int) -> np.ndarray:
    qa = qa.astype(np.int64, copy=False)
    # qa encoding: X = q + a*n_question, with padding qa=0.
    # => target = floor((qa-1)/n_question) gives 0/1 and padding -1.
    return ((qa - 1) // int(n_question)).astype(np.int64, copy=False)


def _flatten_valid_mask(target_2d: np.ndarray) -> np.ndarray:
    # Keep tokens with target in {0,1}. Padding is -1.
    t = target_2d.reshape(-1)
    return t >= 0


def _drop_first_token(target_2d: np.ndarray) -> np.ndarray:
    t = np.asarray(target_2d, dtype=np.int64)
    if t.ndim != 2:
        raise ValueError(f"target must be 2D, got shape={t.shape}")
    if t.shape[1] <= 0:
        return t
    out = t.copy()
    out[:, 0] = -1
    return out


def _target_to_resp(target_2d: np.ndarray) -> np.ndarray:
    # Convert {-1,0,1} -> {0,1} and map padding (-1) to 0 for model inputs.
    t = np.asarray(target_2d, dtype=np.int64)
    return np.clip(t, 0, 1).astype(np.int64, copy=False)


def train(
    net: torch.nn.Module,
    params,
    optimizer: torch.optim.Optimizer,
    q_data: np.ndarray,
    qa_data: np.ndarray,
    pid_data: np.ndarray | None,
    uid_data: np.ndarray | None = None,
    count_data: np.ndarray | None = None,
    *,
    label: str,
) -> tuple[float, float, float]:
    net.train()
    pid_flag, model_type = model_isPid_type(str(getattr(params, "model", "")))
    model_type = str(model_type).strip().lower()

    batch_size = int(getattr(params, "batch_size", 24))
    if batch_size <= 0:
        batch_size = 24

    n_question = int(getattr(params, "n_question"))
    n_seqs = int(q_data.shape[0])
    order = np.arange(n_seqs, dtype=np.int64)
    np.random.shuffle(order)

    pred_list: list[np.ndarray] = []
    target_list: list[np.ndarray] = []

    n_batches = int(math.ceil(n_seqs / float(batch_size)))
    for bi in range(n_batches):
        sl = bi * batch_size
        sr = min(n_seqs, (bi + 1) * batch_size)
        idx = order[sl:sr]

        q_b = q_data[idx]
        qa_b = qa_data[idx]
        pid_b = pid_data[idx] if (pid_data is not None) else None
        uid_b = uid_data[idx] if (uid_data is not None) else None
        count_b = count_data[idx] if (count_data is not None) else None

        target_raw = _qa_to_target(qa_b, n_question=n_question)

        input_q = torch.from_numpy(q_b).long().to(device)
        input_qa = torch.from_numpy(qa_b).long().to(device)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(params):
            if model_type == "akt":
                target_b = _drop_first_token(target_raw)
                valid_mask = _flatten_valid_mask(target_b)
                target_t = torch.from_numpy(target_b).float().to(device)
                if pid_flag:
                    if pid_b is None:
                        raise ValueError("PID model requires pid_data")
                    input_pid = torch.from_numpy(pid_b).long().to(device)
                    loss, pred, _ct = net(input_q, input_qa, target_t, input_pid)
                else:
                    loss, pred, _ct = net(input_q, input_qa, target_t)

                pred_np = pred.detach().float().cpu().numpy().reshape(-1)
                target_np = target_b.reshape(-1).astype(np.int64, copy=False)
                pred_list.append(pred_np[valid_mask])
                target_list.append(target_np[valid_mask].astype(np.float64, copy=False))

            elif model_type == "dkt":
                # Next-step prediction: p(r_t | history up to t-1), for t>=1.
                r_full = _target_to_resp(target_raw)
                q_in = torch.from_numpy(q_b[:, :-1]).long().to(device)
                r_in = torch.from_numpy(r_full[:, :-1]).long().to(device)
                q_next = torch.from_numpy(q_b[:, 1:]).long().to(device)
                target_next_raw = target_raw[:, 1:]
                mask2 = torch.from_numpy((target_next_raw >= 0).astype(np.float32, copy=False)).to(device)
                y_true = torch.from_numpy(_target_to_resp(target_next_raw)).float().to(device)

                y_all = net(q_in, r_in)  # [B, T-1, num_c]
                p_next = torch.gather(y_all, dim=2, index=q_next.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
                loss_vec = _bce_loss_vec(p_next, y_true)
                denom = torch.clamp(mask2.sum(), min=1.0)
                loss = (loss_vec * mask2).sum() / denom

                pred_np = p_next.detach().float().cpu().numpy().reshape(-1)
                target_np = target_next_raw.reshape(-1).astype(np.int64, copy=False)
                valid_mask = target_np >= 0
                pred_list.append(pred_np[valid_mask])
                target_list.append(target_np[valid_mask].astype(np.float64, copy=False))

            elif model_type == "sakt":
                r_full = _target_to_resp(target_raw)
                q_in = torch.from_numpy(q_b[:, :-1]).long().to(device)
                r_in = torch.from_numpy(r_full[:, :-1]).long().to(device)
                q_next = torch.from_numpy(q_b[:, 1:]).long().to(device)
                target_next_raw = target_raw[:, 1:]
                mask2 = torch.from_numpy((target_next_raw >= 0).astype(np.float32, copy=False)).to(device)
                y_true = torch.from_numpy(_target_to_resp(target_next_raw)).float().to(device)

                p_next = net(q_in, r_in, q_next)  # [B, T-1]
                loss_vec = _bce_loss_vec(p_next, y_true)
                denom = torch.clamp(mask2.sum(), min=1.0)
                loss = (loss_vec * mask2).sum() / denom

                pred_np = p_next.detach().float().cpu().numpy().reshape(-1)
                target_np = target_next_raw.reshape(-1).astype(np.int64, copy=False)
                valid_mask = target_np >= 0
                pred_list.append(pred_np[valid_mask])
                target_list.append(target_np[valid_mask].astype(np.float64, copy=False))

            elif model_type == "dkvmn":
                # Predict current response using memory before writing current interaction.
                # Evaluate on t>=1 for consistency with next-step protocols.
                target_b = _drop_first_token(target_raw)
                valid_mask = _flatten_valid_mask(target_b)
                r_full = _target_to_resp(target_raw)
                q_t = torch.from_numpy(q_b).long().to(device)
                r_t = torch.from_numpy(r_full).long().to(device)
                p_all = net(q_t, r_t)  # [B, T]

                y_true = torch.from_numpy(_target_to_resp(target_b)).float().to(device)
                mask2 = torch.from_numpy((target_b >= 0).astype(np.float32, copy=False)).to(device)
                loss_vec = _bce_loss_vec(p_all, y_true)
                denom = torch.clamp(mask2.sum(), min=1.0)
                loss = (loss_vec * mask2).sum() / denom

                pred_np = p_all.detach().float().cpu().numpy().reshape(-1)
                target_np = target_b.reshape(-1).astype(np.int64, copy=False)
                pred_list.append(pred_np[valid_mask])
                target_list.append(target_np[valid_mask].astype(np.float64, copy=False))

            elif model_type == "lpkt":
                # Next-step prediction with concept/qid as exercises.
                r_full = _target_to_resp(target_raw)
                e_t = torch.from_numpy(q_b).long().to(device)
                a_t = torch.from_numpy(r_full).float().to(device)
                target_next_raw = target_raw[:, 1:]
                mask2 = torch.from_numpy((target_next_raw >= 0).astype(np.float32, copy=False)).to(device)
                y_true = torch.from_numpy(_target_to_resp(target_next_raw)).float().to(device)
                use_time = str(getattr(params, "lpkt_use_time", "on")).strip().lower() == "on"
                it_t = None
                if use_time:
                    # If timestamps are missing, use constant interval bins (common in LPKT pipelines).
                    it_np = np.ones_like(q_b, dtype=np.int64)
                    it_t = torch.from_numpy(it_np).long().to(device)
                p_all = net(e_t, a_t, it_t, None)  # [B, T]
                p_next = p_all[:, 1:]
                loss_vec = _bce_loss_vec(p_next, y_true)
                denom = torch.clamp(mask2.sum(), min=1.0)
                loss = (loss_vec * mask2).sum() / denom

                pred_np = p_next.detach().float().cpu().numpy().reshape(-1)
                target_np = target_next_raw.reshape(-1).astype(np.int64, copy=False)
                valid_mask = target_np >= 0
                pred_list.append(pred_np[valid_mask])
                target_list.append(target_np[valid_mask].astype(np.float64, copy=False))

            elif model_type in {"mf", "ncf"}:
                if pid_b is None or uid_b is None:
                    raise ValueError(f"{model_type} requires pid_data and uid_data")

                pid_t = torch.from_numpy(pid_b).long().to(device)  # [B,T]
                uid_s = torch.from_numpy(uid_b).long().to(device)  # [B]
                uid_t = uid_s.unsqueeze(1).expand(-1, pid_t.shape[1])  # [B,T]

                y_raw = target_raw  # {-1,0,1}
                mask2 = torch.from_numpy((y_raw >= 0).astype(np.float32, copy=False)).to(device)
                y_true = torch.from_numpy(_target_to_resp(y_raw)).float().to(device)

                p_all = net(uid_t, pid_t)  # [B,T]
                loss_vec = _bce_loss_vec(p_all, y_true)
                denom = torch.clamp(mask2.sum(), min=1.0)
                loss = (loss_vec * mask2).sum() / denom

                pred_np = p_all.detach().float().cpu().numpy().reshape(-1)
                target_np = y_raw.reshape(-1).astype(np.int64, copy=False)
                valid_mask = target_np >= 0
                pred_list.append(pred_np[valid_mask])
                target_list.append(target_np[valid_mask].astype(np.float64, copy=False))

            else:
                raise ValueError(f"Unsupported model_type={model_type!r} in final/run.py::train()")

        loss.backward()

        maxgradnorm = float(getattr(params, "maxgradnorm", -1.0))
        if maxgradnorm and maxgradnorm > 0.0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=maxgradnorm)
        optimizer.step()

    all_pred = np.concatenate(pred_list, axis=0) if pred_list else np.zeros(0, dtype=np.float64)
    all_target = np.concatenate(target_list, axis=0) if target_list else np.zeros(0, dtype=np.float64)
    loss_m = binary_entropy(all_target, all_pred) if all_pred.size else float("nan")
    auc = compute_auc(all_target, all_pred) if all_pred.size else float("nan")
    acc = compute_accuracy(all_target, all_pred) if all_pred.size else float("nan")
    return loss_m, acc, auc


def test(
    net: torch.nn.Module,
    params,
    optimizer,
    q_data: np.ndarray,
    qa_data: np.ndarray,
    pid_data: np.ndarray | None,
    *,
    uid_data: np.ndarray | None = None,
    count_data: np.ndarray | None = None,
    label: str,
    return_outputs: bool = False,
) -> tuple[float, float, float, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    pid_flag, model_type = model_isPid_type(str(getattr(params, "model", "")))
    model_type = str(model_type).strip().lower()

    net.eval()

    batch_size = int(getattr(params, "eval_batch_size", getattr(params, "batch_size", 24)))
    if batch_size <= 0:
        batch_size = 24

    n_question = int(getattr(params, "n_question"))
    n_seqs = int(q_data.shape[0])
    n_batches = int(math.ceil(n_seqs / float(batch_size)))

    pred_list: list[np.ndarray] = []
    target_list: list[np.ndarray] = []
    pid_list: list[np.ndarray] = []
    count_list: list[np.ndarray] = []

    with torch.no_grad():
        for bi in range(n_batches):
            sl = bi * batch_size
            sr = min(n_seqs, (bi + 1) * batch_size)

            q_b = q_data[sl:sr]
            qa_b = qa_data[sl:sr]
            pid_b = pid_data[sl:sr] if (pid_data is not None) else None
            uid_b = uid_data[sl:sr] if (uid_data is not None) else None
            count_b = count_data[sl:sr] if (count_data is not None) else None

            target_raw = _qa_to_target(qa_b, n_question=n_question)

            input_q = torch.from_numpy(q_b).long().to(device)
            input_qa = torch.from_numpy(qa_b).long().to(device)
            with _autocast_context(params):
                if model_type == "akt":
                    target_b = _drop_first_token(target_raw)
                    valid_mask = _flatten_valid_mask(target_b)
                    target_t = torch.from_numpy(target_b).float().to(device)
                    if pid_flag:
                        if pid_b is None:
                            raise ValueError("PID model requires pid_data")
                        input_pid = torch.from_numpy(pid_b).long().to(device)
                        _loss, pred, _ct = net(input_q, input_qa, target_t, input_pid)
                    else:
                        _loss, pred, _ct = net(input_q, input_qa, target_t)

                    pred_np = pred.detach().float().cpu().numpy().reshape(-1)
                    target_np = target_b.reshape(-1).astype(np.int64, copy=False)
                    pred_list.append(pred_np[valid_mask])
                    target_list.append(target_np[valid_mask].astype(np.float64, copy=False))
                    if return_outputs:
                        if pid_b is not None:
                            pid_flat = pid_b.reshape(-1).astype(np.int64, copy=False)
                            pid_list.append(pid_flat[valid_mask])
                        if count_b is not None:
                            count_flat = count_b.reshape(-1).astype(np.float64, copy=False)
                            count_list.append(count_flat[valid_mask])

                elif model_type == "dkt":
                    r_full = _target_to_resp(target_raw)
                    q_in = torch.from_numpy(q_b[:, :-1]).long().to(device)
                    r_in = torch.from_numpy(r_full[:, :-1]).long().to(device)
                    q_next = torch.from_numpy(q_b[:, 1:]).long().to(device)
                    target_next_raw = target_raw[:, 1:]
                    valid_mask = (target_next_raw.reshape(-1) >= 0)

                    y_all = net(q_in, r_in)
                    p_next = torch.gather(y_all, dim=2, index=q_next.unsqueeze(-1)).squeeze(-1)
                    pred_np = p_next.detach().float().cpu().numpy().reshape(-1)
                    target_np = target_next_raw.reshape(-1).astype(np.int64, copy=False)
                    pred_list.append(pred_np[valid_mask])
                    target_list.append(target_np[valid_mask].astype(np.float64, copy=False))
                    if return_outputs:
                        if pid_b is not None:
                            pid_flat = pid_b[:, 1:].reshape(-1).astype(np.int64, copy=False)
                            pid_list.append(pid_flat[valid_mask])
                        if count_b is not None:
                            count_flat = count_b[:, 1:].reshape(-1).astype(np.float64, copy=False)
                            count_list.append(count_flat[valid_mask])

                elif model_type == "sakt":
                    r_full = _target_to_resp(target_raw)
                    q_in = torch.from_numpy(q_b[:, :-1]).long().to(device)
                    r_in = torch.from_numpy(r_full[:, :-1]).long().to(device)
                    q_next = torch.from_numpy(q_b[:, 1:]).long().to(device)
                    target_next_raw = target_raw[:, 1:]
                    valid_mask = (target_next_raw.reshape(-1) >= 0)

                    p_next = net(q_in, r_in, q_next)
                    pred_np = p_next.detach().float().cpu().numpy().reshape(-1)
                    target_np = target_next_raw.reshape(-1).astype(np.int64, copy=False)
                    pred_list.append(pred_np[valid_mask])
                    target_list.append(target_np[valid_mask].astype(np.float64, copy=False))
                    if return_outputs:
                        if pid_b is not None:
                            pid_flat = pid_b[:, 1:].reshape(-1).astype(np.int64, copy=False)
                            pid_list.append(pid_flat[valid_mask])
                        if count_b is not None:
                            count_flat = count_b[:, 1:].reshape(-1).astype(np.float64, copy=False)
                            count_list.append(count_flat[valid_mask])

                elif model_type == "dkvmn":
                    target_b = _drop_first_token(target_raw)
                    valid_mask = _flatten_valid_mask(target_b)
                    r_full = _target_to_resp(target_raw)
                    q_t = torch.from_numpy(q_b).long().to(device)
                    r_t = torch.from_numpy(r_full).long().to(device)
                    p_all = net(q_t, r_t)
                    pred_np = p_all.detach().float().cpu().numpy().reshape(-1)
                    target_np = target_b.reshape(-1).astype(np.int64, copy=False)
                    pred_list.append(pred_np[valid_mask])
                    target_list.append(target_np[valid_mask].astype(np.float64, copy=False))
                    if return_outputs:
                        if pid_b is not None:
                            pid_flat = pid_b.reshape(-1).astype(np.int64, copy=False)
                            pid_list.append(pid_flat[valid_mask])
                        if count_b is not None:
                            count_flat = count_b.reshape(-1).astype(np.float64, copy=False)
                            count_list.append(count_flat[valid_mask])

                elif model_type == "lpkt":
                    r_full = _target_to_resp(target_raw)
                    e_t = torch.from_numpy(q_b).long().to(device)
                    a_t = torch.from_numpy(r_full).float().to(device)
                    target_next_raw = target_raw[:, 1:]
                    valid_mask = (target_next_raw.reshape(-1) >= 0)
                    use_time = str(getattr(params, "lpkt_use_time", "on")).strip().lower() == "on"
                    it_t = None
                    if use_time:
                        it_np = np.ones_like(q_b, dtype=np.int64)
                        it_t = torch.from_numpy(it_np).long().to(device)
                    p_all = net(e_t, a_t, it_t, None)
                    p_next = p_all[:, 1:]
                    pred_np = p_next.detach().float().cpu().numpy().reshape(-1)
                    target_np = target_next_raw.reshape(-1).astype(np.int64, copy=False)
                    pred_list.append(pred_np[valid_mask])
                    target_list.append(target_np[valid_mask].astype(np.float64, copy=False))
                    if return_outputs:
                        if pid_b is not None:
                            pid_flat = pid_b[:, 1:].reshape(-1).astype(np.int64, copy=False)
                            pid_list.append(pid_flat[valid_mask])
                        if count_b is not None:
                            count_flat = count_b[:, 1:].reshape(-1).astype(np.float64, copy=False)
                            count_list.append(count_flat[valid_mask])

                elif model_type in {"mf", "ncf"}:
                    if pid_b is None or uid_b is None:
                        raise ValueError(f"{model_type} requires pid_data and uid_data")

                    pid_t = torch.from_numpy(pid_b).long().to(device)
                    uid_s = torch.from_numpy(uid_b).long().to(device)
                    uid_t = uid_s.unsqueeze(1).expand(-1, pid_t.shape[1])

                    y_raw = target_raw
                    valid_mask = (y_raw.reshape(-1) >= 0)
                    p_all = net(uid_t, pid_t)
                    pred_np = p_all.detach().float().cpu().numpy().reshape(-1)
                    target_np = y_raw.reshape(-1).astype(np.int64, copy=False)
                    pred_list.append(pred_np[valid_mask])
                    target_list.append(target_np[valid_mask].astype(np.float64, copy=False))
                    if return_outputs:
                        pid_flat = pid_b.reshape(-1).astype(np.int64, copy=False)
                        pid_list.append(pid_flat[valid_mask])
                        if count_b is not None:
                            count_flat = count_b.reshape(-1).astype(np.float64, copy=False)
                            count_list.append(count_flat[valid_mask])

                else:
                    raise ValueError(f"Unsupported model_type={model_type!r} in final/run.py::test()")

    all_pred = np.concatenate(pred_list, axis=0) if pred_list else np.zeros(0, dtype=np.float64)
    all_target = np.concatenate(target_list, axis=0) if target_list else np.zeros(0, dtype=np.float64)
    loss_m = binary_entropy(all_target, all_pred) if all_pred.size else float("nan")
    auc = compute_auc(all_target, all_pred) if all_pred.size else float("nan")
    acc = compute_accuracy(all_target, all_pred) if all_pred.size else float("nan")

    y = all_target if return_outputs else None
    p = all_pred if return_outputs else None
    pid = np.concatenate(pid_list, axis=0) if (return_outputs and pid_list) else None
    cnt = np.concatenate(count_list, axis=0) if (return_outputs and count_list) else None
    _ = uid_data
    return loss_m, acc, auc, y, p, pid, cnt
