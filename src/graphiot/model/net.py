"""GRAPHIOT model (paper §IV-B).

Encoder: a linear projection per node type to ``d``; ``L`` HGT layers within each
snapshot; a GRU cell per node across snapshots, m_v^t = GRU(h_v^t, m_v^{t-1}),
keyed by the node's stable key; new nodes start from a zero memory.

Head: for a write command of task tau by agent a, every tag kappa' the task type
may write (the gateway's allow-list) is a candidate. A two-layer perceptron reads
[m_tau, m_a, m_stage(kappa'), m_kappa'] and its output layer has one unit per
(tag, bin) pair; the softmax runs jointly over the allowed (tag, bin) pairs.
Loss: negative log-likelihood of the observed (tag, bin). Score: the surprise
-log p (+inf for a tag outside the allowed set).

Ablations (§V-F) restrict the graph to a subgraph and zero the head inputs that
the subgraph does not contain (VARIANTS).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..graph.schema import EDGE_TYPES, NODE_TYPES
from .batch import Chunk, Rel, reverse
from .hgt import HGTLayer

HEAD_SLOTS = ("task", "agent", "stage", "tag")

# node types kept, head inputs used, tag values masked
VARIANTS: dict[str, dict] = {
    "full": {"nodes": set(NODE_TYPES), "head": HEAD_SLOTS, "mask_tag_values": False},
    "agent_only": {"nodes": {"task", "agent", "gateway"}, "head": ("task", "agent"), "mask_tag_values": False},
    "plant_only": {"nodes": {"plc", "tag", "stage"}, "head": ("stage", "tag"), "mask_tag_values": False},
    # network view: hosts, gateway, PLCs and tags as Modbus endpoints (reads/writes/flows);
    # tag values only through the transactions in the window, no 1 Hz state, no stages
    "network_only": {"nodes": {"host", "gateway", "plc", "tag"}, "head": ("tag",), "mask_tag_values": True},
    "no_task": {"nodes": set(NODE_TYPES) - {"task"}, "head": ("agent", "stage", "tag"), "mask_tag_values": False},
}


class Memory:
    """Per-node GRU memory: one row per stable node key; row 0 is the zero memory."""

    def __init__(self, d: int, device="cpu"):
        self.d, self.device = d, device
        self.index: dict[str, int] = {}
        self.M = torch.zeros((1, d), device=device)

    def rows(self, keys: list[str], create: bool = True) -> torch.Tensor:
        out, new = [], []
        for k in keys:
            i = self.index.get(k)
            if i is None:
                if not create:
                    out.append(0)
                    continue
                i = len(self.index) + 1
                self.index[k] = i
                new.append(k)
            out.append(i)
        if new:
            self.M = torch.cat([self.M, torch.zeros((len(new), self.d), device=self.M.device,
                                                    dtype=self.M.dtype)])
        return torch.as_tensor(out, dtype=torch.long, device=self.M.device)

    def lookup(self, keys: list[str]) -> torch.Tensor:
        return self.M[self.rows(keys, create=False)]

    def update(self, rows: torch.Tensor, values: torch.Tensor) -> None:
        M = self.M.clone()                       # out of place: keeps the autograd graph valid
        M[rows] = values
        self.M = M

    def detach(self) -> None:
        self.M = self.M.detach()

    def state(self) -> dict:
        return {"index": dict(self.index), "M": self.M.detach().cpu().clone()}

    def load(self, st: dict) -> None:
        self.index = dict(st["index"])
        self.M = st["M"].to(self.device).clone()


@dataclass
class HeadBatch:
    """Commands to score, flattened over their candidate tags."""
    task_rows: torch.Tensor        # [n_cmd]
    agent_rows: torch.Tensor       # [n_cmd]
    cand_cmd: torch.Tensor         # [n_cand] command of each candidate
    cand_tag: torch.Tensor         # [n_cand] writable-tag index
    cand_stage_rows: torch.Tensor  # [n_cand]
    cand_tag_rows: torch.Tensor    # [n_cand]
    target_cand: torch.Tensor      # [n_cmd] flat candidate index of the observed tag (-1: not allowed)
    target_bin: torch.Tensor       # [n_cmd]


class GraphiotModel(nn.Module):
    def __init__(self, node_dims: dict[str, int], edge_dims: dict[Rel, int], n_tags: int, q: int,
                 n_bins: list[int], *, d: int = 64, layers: int = 2, heads: int = 4,
                 reverse_edges: bool = True, variant: str = "full"):
        super().__init__()
        self.cfg = dict(node_dims=node_dims, edge_dims={"__".join(k): v for k, v in edge_dims.items()},
                        n_tags=n_tags, q=q, n_bins=list(n_bins), d=d, layers=layers, heads=heads,
                        reverse_edges=reverse_edges, variant=variant)
        self.d, self.q, self.variant = d, q, VARIANTS[variant]
        types = [nt for nt in NODE_TYPES if nt in self.variant["nodes"]]
        rels = [r for r in EDGE_TYPES if r[0] in self.variant["nodes"] and r[2] in self.variant["nodes"]]
        if reverse_edges:
            rels = rels + [reverse(r) for r in rels]
        all_edge_dims = {**edge_dims, **{reverse(r): edge_dims[r] for r in EDGE_TYPES}}
        self.reverse_edges = reverse_edges
        self.proj = nn.ModuleDict({nt: nn.Linear(node_dims[nt], d) for nt in types})
        self.layers = nn.ModuleList([HGTLayer(d, heads, types, rels, all_edge_dims) for _ in range(layers)])
        self.gru = nn.GRUCell(d, d)
        self.head1 = nn.Linear(4 * d, d)
        self.head_w = nn.Parameter(torch.empty(n_tags, q, d))
        self.head_b = nn.Parameter(torch.zeros(n_tags, q))
        nn.init.xavier_uniform_(self.head_w)
        mask = torch.zeros(n_tags, q)
        for i, nb in enumerate(n_bins):
            mask[i, max(1, min(nb, q)):] = float("-inf")       # bins a tag does not have
        self.register_buffer("bin_mask", mask)
        slot_on = torch.tensor([float(s in self.variant["head"]) for s in HEAD_SLOTS])
        self.register_buffer("slot_on", slot_on.repeat_interleave(d))

    # ---- encoder ------------------------------------------------------------
    def encode(self, chunk: Chunk, keep_attention: bool = False) -> dict[str, torch.Tensor]:
        h = {nt: self.proj[nt](x) for nt, x in chunk.x.items() if nt in self.proj and x.numel()}
        attn = []
        for layer in self.layers:
            h = layer(h, chunk.edges, keep_attention=keep_attention)
            attn.append(layer.last_attention)
        self.last_attention = attn if keep_attention else None
        return h

    def step_memory(self, chunk: Chunk, h: dict[str, torch.Tensor], i: int, mem: Memory) -> None:
        """GRU update of the memories of the nodes present in snapshot i of the chunk."""
        keys, hs = [], []
        snap = chunk.snaps[i]
        for nt in h:
            n = snap.num_nodes(nt)
            if n:
                keys += snap.keys[nt]
                hs.append(h[nt][chunk.rows(i, nt)])
        if not keys:
            return
        rows = mem.rows(keys)
        hcat = torch.cat(hs)
        mem.update(rows, self.gru(hcat, mem.M[rows]))

    # ---- head -----------------------------------------------------------------
    def head_logits(self, mem: Memory, hb: HeadBatch) -> torch.Tensor:
        """Logits [n_cand, Q] of every (candidate tag, bin)."""
        M = mem.M
        inp = torch.cat([M[hb.task_rows][hb.cand_cmd], M[hb.agent_rows][hb.cand_cmd],
                         M[hb.cand_stage_rows], M[hb.cand_tag_rows]], dim=1) * self.slot_on
        hid = F.relu(self.head1(inp))
        logits = torch.einsum("cd,cqd->cq", hid, self.head_w[hb.cand_tag]) + self.head_b[hb.cand_tag]
        return logits + self.bin_mask[hb.cand_tag]

    def log_probs(self, mem: Memory, hb: HeadBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """(log p of the observed (tag, bin) per command [n_cmd], log-normaliser per command)."""
        logits = self.head_logits(mem, hb)                                   # [n_cand, Q]
        n_cmd = hb.task_rows.shape[0]
        flat_cmd = hb.cand_cmd.repeat_interleave(self.q)
        flat = logits.reshape(-1)
        mx = torch.full((n_cmd,), float("-inf"), device=flat.device).scatter_reduce(
            0, flat_cmd, flat, reduce="amax", include_self=True)
        lse = torch.zeros(n_cmd, device=flat.device).index_add_(
            0, flat_cmd, (flat - mx[flat_cmd]).exp()).log() + mx
        ok = hb.target_cand >= 0
        tgt = torch.full((n_cmd,), float("-inf"), device=flat.device)
        if ok.any():
            tgt[ok] = logits[hb.target_cand[ok], hb.target_bin[ok]]
        return tgt - lse, lse

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
