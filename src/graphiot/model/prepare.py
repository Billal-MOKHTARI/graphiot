"""From a GraphDataset to what the model consumes: tensor snapshots of a variant,
the benign sequence for training/validation, and per-command head inputs.

Benign sequence. A snapshot enters training or validation only if its whole window
[T_k - Δ, T_k) lies inside train or validation time, so no red-team event (or
excluded recovery margin) reaches the weights, the early-stopping criterion or the
threshold. The memory carries over the gaps. Each benign command is scored from
the memory after the last benign snapshot with T_k <= t; at test time every
command is scored from the full timeline, snapshot k = last T_k <= t (causal rule).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from ..graph.dataset import GraphDataset
from ..graph.schema import edge_dims, node_dims
from .batch import Chunk, TSnap, make_chunk, restrict, to_tensor
from .net import VARIANTS, HeadBatch, Memory


def _inside(intervals: list[tuple[float, float]], a: float, b: float) -> bool:
    return any(lo <= a and b <= hi for lo, hi in intervals)


@dataclass
class Prepared:
    ds: GraphDataset
    variant: str
    snaps: list[TSnap]
    tag_index: dict[str, int]
    n_bins: list[int]
    benign_pos: np.ndarray                 # snapshot indices of the benign sequence, in time order
    train_end: int                         # benign_pos[:train_end] = training sequence
    cmd: pd.DataFrame                      # commands + head fields
    _chunks: dict = field(default_factory=dict)

    # ---- chunks -------------------------------------------------------------
    def chunk(self, idx: tuple[int, ...], reverse_edges: bool) -> Chunk:
        key = (idx, reverse_edges)
        if key not in self._chunks:
            self._chunks[key] = make_chunk([self.snaps[i] for i in idx], reverse_edges)
        return self._chunks[key]

    # ---- head inputs ----------------------------------------------------------
    def head_batch(self, rows: pd.DataFrame, mem: Memory, device) -> HeadBatch | None:
        if rows.empty:
            return None
        t_rows = mem.rows(list("task:" + rows["task_id"].astype(str)), create=False)
        a_rows = mem.rows(list("agent:" + rows["agent_id"].astype(str)), create=False)
        cand_cmd, cand_tag, cand_stage, cand_tagkey, target = [], [], [], [], []
        for j, (tt, tag) in enumerate(zip(rows["task_type"], rows["tag"])):
            allowed = self.ds.allowed.get(tt, [])
            tgt = -1
            for t in allowed:
                if t == tag:
                    tgt = len(cand_cmd)
                cand_cmd.append(j)
                cand_tag.append(self.tag_index[t])
                cand_stage.append(f"stage:{self.ds.tag_stage[t]}")
                cand_tagkey.append(f"tag:{t}")
            target.append(tgt)
        if not cand_cmd:                              # e.g. monitoring tasks: nothing allowed
            cand_cmd, cand_tag, cand_stage, cand_tagkey = [0], [0], ["-"], ["-"]
        T = lambda v: torch.tensor(np.asarray(v, dtype=np.int64), dtype=torch.long, device=device)
        return HeadBatch(task_rows=t_rows, agent_rows=a_rows, cand_cmd=T(cand_cmd), cand_tag=T(cand_tag),
                         cand_stage_rows=mem.rows(cand_stage, create=False),
                         cand_tag_rows=mem.rows(cand_tagkey, create=False),
                         target_cand=T(target), target_bin=T(rows["bin_m"].to_numpy()))


def prepare(ds: GraphDataset, variant: str = "full") -> Prepared:
    v = VARIANTS[variant]
    nd, ed = node_dims(len(ds.writable_tags)), edge_dims()
    snaps = [restrict(to_tensor(s, nd, ed), v["nodes"], v["mask_tag_values"]) for s in ds.snaps]
    tag_index = {t: i for i, t in enumerate(ds.writable_tags)}
    n_bins = [ds.bins.n_bins(t) for t in ds.writable_tags]

    delta = ds.window_s
    benign = [k for k, T in enumerate(ds.times)
              if _inside(ds.splits.train, T - delta, T) or _inside(ds.splits.val, T - delta, T)]
    train_end_t = max((b for _, b in ds.splits.train), default=-np.inf)
    benign_pos = np.asarray(benign, dtype=np.int64)
    train_end = int(np.searchsorted(ds.times[benign_pos], train_end_t, side="right")) if len(benign_pos) else 0

    cmd = ds.commands.copy()
    cmd["tag_in_vocab"] = cmd["tag"].isin(tag_index)
    # bin index for the head: tags without fitted bins have a single bin 0
    cmd["bin_m"] = [max(0, min(int(b), max(1, nb) - 1)) if t in tag_index else 0
                    for t, b, nb in zip(cmd["tag"], cmd["bin"],
                                        [ds.bins.n_bins(t) for t in cmd["tag"]])]
    # position in the benign sequence (last benign snapshot with T <= t), -1 if none
    bt = ds.times[benign_pos] if len(benign_pos) else np.zeros(0)
    cmd["bpos"] = np.searchsorted(bt, cmd["ts"].to_numpy(), side="right") - 1
    cmd["forwarded"] = cmd["decision"] == "forwarded"
    return Prepared(ds=ds, variant=variant, snaps=snaps, tag_index=tag_index, n_bins=n_bins,
                    benign_pos=benign_pos, train_end=train_end, cmd=cmd)
