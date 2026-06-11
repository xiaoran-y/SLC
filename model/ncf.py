from __future__ import annotations

import torch


def _parse_hidden_dims(spec: str) -> list[int]:
    dims: list[int] = []
    for chunk in str(spec).replace(",", " ").split():
        if not chunk:
            continue
        dims.append(int(chunk))
    return dims


class NCF(torch.nn.Module):
    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        emb_dim: int,
        hidden_dims: str = "128,64,32",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.user_emb = torch.nn.Embedding(int(n_users) + 1, int(emb_dim))
        self.item_emb = torch.nn.Embedding(int(n_items) + 1, int(emb_dim))

        dims = [int(emb_dim) * 2, *_parse_hidden_dims(hidden_dims)]
        layers: list[torch.nn.Module] = []
        for din, dout in zip(dims[:-1], dims[1:], strict=True):
            layers.append(torch.nn.Linear(int(din), int(dout)))
            layers.append(torch.nn.ReLU())
            if float(dropout) > 0.0:
                layers.append(torch.nn.Dropout(float(dropout)))
        self.mlp = torch.nn.Sequential(*layers) if layers else torch.nn.Identity()
        last_dim = dims[-1] if dims else int(emb_dim) * 2
        self.out = torch.nn.Linear(int(last_dim), 1)

        torch.nn.init.normal_(self.user_emb.weight, std=0.02)
        torch.nn.init.normal_(self.item_emb.weight, std=0.02)

    def forward(self, uid: torch.Tensor, pid: torch.Tensor) -> torch.Tensor:
        uid = uid.long()
        pid = pid.long()
        u = self.user_emb(uid)
        v = self.item_emb(pid)
        x = torch.cat([u, v], dim=-1)
        h = self.mlp(x)
        logit = self.out(h).squeeze(-1)
        return torch.sigmoid(logit)

