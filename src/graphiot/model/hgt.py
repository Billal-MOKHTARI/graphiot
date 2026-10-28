"""Heterogeneous graph transformer layer (Hu et al., WWW 2020) with edge features.

Per node type: key, query and value projections and a gated residual update;
per relation: an attention matrix, a message matrix and a per-head prior
``mu``, as in the published layer and in PyG's ``HGTConv``. The softmax runs over
all incoming edges of a target node across relations.

Deviation D14 (docs/DEVIATIONS.md): Table II gives every relation edge
features, which the published HGT has no input for. Each relation therefore
has a linear map of its edge features to ``d``, added to the source's key and
value on that edge, as PyG's ``TransformerConv`` does with ``edge_dim``.
Attention weights are kept for explanation (``last_attention``).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .batch import Rel


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax of ``scores`` [m, H] within groups given by ``index`` [m] (n groups)."""
    h = scores.shape[1]
    idx = index.unsqueeze(1).expand(-1, h)
    mx = torch.full((n, h), float("-inf"), dtype=scores.dtype, device=scores.device)
    mx = mx.scatter_reduce(0, idx, scores, reduce="amax", include_self=True)
    ex = (scores - mx.gather(0, idx)).exp()
    den = torch.zeros((n, h), dtype=scores.dtype, device=scores.device).index_add_(0, index, ex)
    return ex / den.gather(0, idx).clamp_min(1e-16)


class HGTLayer(nn.Module):
    def __init__(self, d: int, heads: int, node_types: list[str], relations: list[Rel],
                 edge_dims: dict[Rel, int]):
        super().__init__()
        assert d % heads == 0, "d must be divisible by the number of heads"
        self.d, self.h, self.dk = d, heads, d // heads
        self.node_types, self.relations = node_types, relations
        self.k = nn.ModuleDict({nt: nn.Linear(d, d) for nt in node_types})
        self.q = nn.ModuleDict({nt: nn.Linear(d, d) for nt in node_types})
        self.v = nn.ModuleDict({nt: nn.Linear(d, d) for nt in node_types})
        self.a = nn.ModuleDict({nt: nn.Linear(d, d) for nt in node_types})
        self.skip = nn.ParameterDict({nt: nn.Parameter(torch.ones(1)) for nt in node_types})
        key = self._key
        self.w_att = nn.ParameterDict({key(r): nn.Parameter(torch.empty(heads, self.dk, self.dk)) for r in relations})
        self.w_msg = nn.ParameterDict({key(r): nn.Parameter(torch.empty(heads, self.dk, self.dk)) for r in relations})
        self.mu = nn.ParameterDict({key(r): nn.Parameter(torch.ones(heads)) for r in relations})
        self.e = nn.ModuleDict({key(r): nn.Linear(edge_dims[r], d, bias=False) for r in relations})
        for r in relations:
            nn.init.xavier_uniform_(self.w_att[key(r)])
            nn.init.xavier_uniform_(self.w_msg[key(r)])
        self.last_attention: dict[Rel, torch.Tensor] | None = None

    @staticmethod
    def _key(r: Rel) -> str:
        return "__".join(r)

    def forward(self, h: dict[str, torch.Tensor], edges: dict[Rel, tuple[torch.Tensor, torch.Tensor]],
                keep_attention: bool = False) -> dict[str, torch.Tensor]:
        H, dk = self.h, self.dk
        q = {nt: self.q[nt](x).view(-1, H, dk) for nt, x in h.items() if nt in self.q and x.numel()}
        # Gather per target type: scores, messages and target rows of every incoming edge.
        per_dst: dict[str, list] = {}
        for r in self.relations:
            if r not in edges:
                continue
            s_t, _, d_t = r
            ei, attr = edges[r]
            if ei.shape[1] == 0 or s_t not in q or d_t not in q:
                continue
            kr = self._key(r)
            src, dst = ei[0], ei[1]
            e = self.e[kr](attr).view(-1, H, dk)
            k = self.k[s_t](h[s_t]).view(-1, H, dk)[src] + e
            v = self.v[s_t](h[s_t]).view(-1, H, dk)[src] + e
            k = torch.einsum("mhi,hij->mhj", k, self.w_att[kr])
            msg = torch.einsum("mhi,hij->mhj", v, self.w_msg[kr])
            score = (k * q[d_t][dst]).sum(-1) * self.mu[kr] / math.sqrt(dk)      # [m, H]
            per_dst.setdefault(d_t, []).append((r, dst, score, msg))
        out = dict(h)
        attn: dict[Rel, torch.Tensor] = {}
        for d_t, parts in per_dst.items():
            dst = torch.cat([p[1] for p in parts])
            score = torch.cat([p[2] for p in parts])
            msg = torch.cat([p[3] for p in parts])
            n = h[d_t].shape[0]
            alpha = segment_softmax(score, dst, n)                               # [m, H]
            agg = torch.zeros((n, H, dk), dtype=msg.dtype, device=msg.device)
            agg.index_add_(0, dst, alpha.unsqueeze(-1) * msg)
            upd = self.a[d_t](F.gelu(agg.reshape(n, self.d)))
            beta = torch.sigmoid(self.skip[d_t])
            out[d_t] = beta * upd + (1 - beta) * h[d_t]
            if keep_attention:
                i = 0
                for r, dst_r, _, _ in parts:
                    attn[r] = alpha[i:i + len(dst_r)].detach()
                    i += len(dst_r)
        # Node types without incoming edges keep their input (as in HGT's gated skip with no message).
        self.last_attention = attn if keep_attention else None
        return out
