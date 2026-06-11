from __future__ import annotations

import torch


class MF(torch.nn.Module):
    def __init__(self, *, n_users: int, n_items: int, emb_dim: int) -> None:
        super().__init__()
        self.user_emb = torch.nn.Embedding(int(n_users) + 1, int(emb_dim))
        self.item_emb = torch.nn.Embedding(int(n_items) + 1, int(emb_dim))
        self.user_bias = torch.nn.Embedding(int(n_users) + 1, 1)
        self.item_bias = torch.nn.Embedding(int(n_items) + 1, 1)
        self.global_bias = torch.nn.Parameter(torch.zeros(()))

        torch.nn.init.normal_(self.user_emb.weight, std=0.02)
        torch.nn.init.normal_(self.item_emb.weight, std=0.02)
        torch.nn.init.zeros_(self.user_bias.weight)
        torch.nn.init.zeros_(self.item_bias.weight)

    def forward(self, uid: torch.Tensor, pid: torch.Tensor) -> torch.Tensor:
        uid = uid.long()
        pid = pid.long()
        u = self.user_emb(uid)
        v = self.item_emb(pid)
        dot = (u * v).sum(dim=-1)
        bu = self.user_bias(uid).squeeze(-1)
        bi = self.item_bias(pid).squeeze(-1)
        logit = dot + bu + bi + self.global_bias
        return torch.sigmoid(logit)

