"""Re-score an edited episode with the trained model, under the causal rule.

The attacker queries the detector's score on an edited episode (grey-box, §IV-C).
Editing changes the graph snapshots that cover the episode window, so we rebuild
exactly those snapshots from the synthesised monitor view and score every write of
the edited episode, continuing the GRU memory from a checkpoint at the window start
(the memory before the window is unchanged by an edit inside it). This is the same
scoring path as `model.train.score_timeline`, restricted to the window.

``ModelDetector`` is episode-agnostic (it caches the base memory timeline once);
``ModelDetector.scorer(synth)`` binds it to one episode's window.
"""

from __future__ import annotations

from copy import deepcopy

import pandas as pd
import torch

from ..config.schema import ExperimentConfig
from ..data.monitor_view import MonitorView
from ..graph.builder import SnapshotBuilder, Topology, snapshot_index_for
from ..graph.schema import edge_dims, node_dims
from ..model.batch import make_chunk, restrict, to_tensor
from ..model.net import VARIANTS, GraphiotModel, Memory
from ..model.prepare import Prepared
from .synth import Event, WindowSynth

_CKPT_EVERY = 50


class ModelDetector:
    name = "full"

    def __init__(self, model: GraphiotModel, prep: Prepared, cfg: ExperimentConfig, threshold: float, *, device="cpu"):
        self.model, self.prep, self.cfg, self.threshold, self.device = model, prep, cfg, threshold, device
        self.ds = prep.ds
        self.nd, self.ed = node_dims(len(self.ds.writable_tags)), edge_dims()
        self._ckpts = self._memory_timeline()

    @torch.no_grad()
    def _memory_timeline(self) -> list[tuple[int, dict]]:
        """Base GRU memory state every _CKPT_EVERY snapshots (one forward pass)."""
        mem = Memory(self.model.d, self.device)
        ck = [(0, mem.state())]
        B = self.cfg.model.batch_snapshots
        n = len(self.ds.times)
        for s in range(0, n, B):
            idx = tuple(range(s, min(s + B, n)))
            chunk = self.prep.chunk(idx, self.model.reverse_edges).to(self.device)
            h = self.model.encode(chunk)
            for i, k in enumerate(idx):
                self.model.step_memory(chunk, h, i, mem)
                if (k + 1) % _CKPT_EVERY == 0:
                    ck.append((k + 1, mem.state()))
        return ck

    @torch.no_grad()
    def memory_before(self, k: int) -> Memory:
        base = max((c for c in self._ckpts if c[0] <= k), key=lambda c: c[0])
        mem = Memory(self.model.d, self.device)
        mem.load(base[1])
        B = self.cfg.model.batch_snapshots
        for s in range(base[0], k, B):
            idx = tuple(range(s, min(s + B, k)))
            chunk = self.prep.chunk(idx, self.model.reverse_edges).to(self.device)
            h = self.model.encode(chunk)
            for i in range(len(idx)):
                self.model.step_memory(chunk, h, i, mem)
        return mem

    def scorer(self, synth: WindowSynth) -> "ModelScorer":
        return ModelScorer(self, synth)


class ModelScorer:
    def __init__(self, det: ModelDetector, synth: WindowSynth):
        self.det, self.synth = det, synth
        self.name, self.threshold = det.name, det.threshold
        times = det.ds.times
        self.win_k = [k for k, T in enumerate(times) if synth.win_lo < T <= synth.win_hi]
        self.k_start = self.win_k[0] if self.win_k else len(times)
        self._mem0 = det.memory_before(self.k_start)

    @torch.no_grad()
    def max_score(self, view: MonitorView, events: list[Event], task_id: str, agent_id: str) -> float:
        det, times = self.det, self.det.ds.times
        topo = Topology.from_config(view, det.cfg.testbed.network)
        builder = SnapshotBuilder(view, topo, window_s=det.ds.window_s, stride_s=det.ds.stride_s,
                                  rolling_s=det.cfg.graph.rolling_window_s)
        vnodes = VARIANTS["full"]["nodes"]
        snaps = [restrict(to_tensor(det.ds.norm.transform(builder.build(float(times[k]))), det.nd, det.ed),
                          vnodes, VARIANTS["full"]["mask_tag_values"]) for k in self.win_k]
        scores = []
        by_k: dict[int, list[Event]] = {}
        for e in (e for e in events if e.kind == "write"):
            by_k.setdefault(snapshot_index_for(times, e.ts), []).append(e)
        if snaps:
            chunk = make_chunk(snaps, det.model.reverse_edges).to(det.device)
            h = det.model.encode(chunk)
            mem = deepcopy(self._mem0)
            for i, k in enumerate(self.win_k):
                det.model.step_memory(chunk, h, i, mem)
                for e in by_k.get(k, []):
                    scores.append(self._score_one(e, task_id, agent_id, mem))
        for k, evs in by_k.items():                       # writes before the window: pre-window memory
            if k < self.k_start:
                for e in evs:
                    scores.append(self._score_one(e, task_id, agent_id, self._mem0))
        return float(max(scores)) if scores else float("-inf")

    def _score_one(self, e: Event, task_id: str, agent_id: str, mem: Memory) -> float:
        det = self.det
        if e.tag not in det.prep.tag_index:
            return float("inf")                           # outside the task's writable set: always flagged
        nb = det.ds.bins.n_bins(e.tag)
        b = max(0, min(int(det.ds.bins.bin(e.tag, e.value)), max(1, nb) - 1))
        row = pd.DataFrame([{"command_idx": e.command_idx, "task_id": task_id, "agent_id": agent_id,
                             "task_type": self._task_type(task_id), "tag": e.tag, "bin_m": b}])
        hb = det.prep.head_batch(row, mem, det.device)
        lp, _ = det.model.log_probs(mem, hb)
        return float(-lp[0])

    def _task_type(self, task_id: str) -> str:
        m = self.det.prep.ds.commands
        m = m[m["task_id"] == task_id]
        return m["task_type"].iloc[0] if len(m) else "unknown"
