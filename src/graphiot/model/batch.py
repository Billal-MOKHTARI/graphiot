"""Snapshots as tensors, and a chunk of snapshots as one disjoint-union graph.

The encoder (projection + HGT) of a snapshot does not depend on the memory, so
the snapshots of a truncated-BPTT chunk are encoded in one call; only the GRU
updates run in time order (``net.GraphiotModel.run_chunk``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..graph.builder import Snapshot
from ..graph.schema import EDGE_TYPES, NODE_TYPES

Rel = tuple[str, str, str]


def reverse(rel: Rel) -> Rel:
    s, r, d = rel
    return (d, f"rev_{r}", s)


@dataclass
class TSnap:
    T: float
    keys: dict[str, list[str]]                          # node type -> stable keys
    x: dict[str, torch.Tensor]                          # node type -> [n, d_in]
    edges: dict[Rel, tuple[torch.Tensor, torch.Tensor]]  # rel -> (edge_index [2, m], attr [m, d_e])

    def num_nodes(self, nt: str) -> int:
        return len(self.keys.get(nt, []))


def to_tensor(snap: Snapshot, node_dims: dict[str, int], edge_dims: dict[Rel, int]) -> TSnap:
    x = {}
    for nt in NODE_TYPES:
        a = snap.x.get(nt)
        x[nt] = torch.as_tensor(a, dtype=torch.float32) if a is not None and a.size \
            else torch.zeros((0, node_dims[nt]), dtype=torch.float32)
    edges = {}
    for et in EDGE_TYPES:
        if et in snap.edges and len(snap.edges[et][0]):
            s, d, a = snap.edges[et]
            ei = torch.as_tensor(np.stack([np.asarray(s, dtype=np.int64), np.asarray(d, dtype=np.int64)]))
            edges[et] = (ei, torch.as_tensor(a, dtype=torch.float32).reshape(len(s), edge_dims[et]))
    return TSnap(T=float(snap.T), keys={nt: list(snap.nodes.get(nt, [])) for nt in NODE_TYPES}, x=x, edges=edges)


def restrict(ts: TSnap, node_types: set[str], mask_tag_values: bool = False) -> TSnap:
    """Ablation view: keep only these node types and the relations between them."""
    x = {nt: (v if nt in node_types else v[:0]) for nt, v in ts.x.items()}
    if mask_tag_values and "tag" in x and x["tag"].numel():
        t = x["tag"].clone()
        t[:, -3:] = 0.0                                  # current value, rolling mean, rolling std
        x["tag"] = t
    keys = {nt: (k if nt in node_types else []) for nt, k in ts.keys.items()}
    edges = {et: e for et, e in ts.edges.items() if et[0] in node_types and et[2] in node_types}
    return TSnap(T=ts.T, keys=keys, x=x, edges=edges)


@dataclass
class Chunk:
    """Disjoint union of consecutive snapshots."""
    x: dict[str, torch.Tensor]
    edges: dict[Rel, tuple[torch.Tensor, torch.Tensor]]
    offsets: list[dict[str, int]]                       # per snapshot: node type -> first row
    snaps: list[TSnap]

    def rows(self, i: int, nt: str) -> slice:
        o = self.offsets[i][nt]
        return slice(o, o + self.snaps[i].num_nodes(nt))

    def to(self, device) -> "Chunk":
        return Chunk(x={k: v.to(device) for k, v in self.x.items()},
                     edges={k: (ei.to(device), a.to(device)) for k, (ei, a) in self.edges.items()},
                     offsets=self.offsets, snaps=self.snaps)


def make_chunk(snaps: list[TSnap], reverse_edges: bool = True) -> Chunk:
    offsets, xs = [], {nt: [] for nt in NODE_TYPES}
    count = {nt: 0 for nt in NODE_TYPES}
    edge_parts: dict[Rel, list] = {}
    for s in snaps:
        offsets.append(dict(count))
        for nt in NODE_TYPES:
            xs[nt].append(s.x[nt])
        for et, (ei, a) in s.edges.items():
            src_t, _, dst_t = et
            shifted = torch.stack([ei[0] + count[src_t], ei[1] + count[dst_t]])
            edge_parts.setdefault(et, []).append((shifted, a))
            if reverse_edges:
                edge_parts.setdefault(reverse(et), []).append((shifted.flip(0), a))
        for nt in NODE_TYPES:
            count[nt] += s.num_nodes(nt)
    x = {nt: torch.cat(v) if v else torch.zeros(0) for nt, v in xs.items()}
    edges = {et: (torch.cat([p[0] for p in parts], 1), torch.cat([p[1] for p in parts]))
             for et, parts in edge_parts.items()}
    return Chunk(x=x, edges=edges, offsets=offsets, snaps=snaps)
