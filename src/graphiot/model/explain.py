"""Explanation of one command's score (GUI: Inference & explainability).

* prediction: the observed (tag, bin), its probability, and the most likely
  allowed (tag, bin) pairs;
* context contribution: the change in surprise when one head input (task, agent,
  stage or tag memory) is replaced by the zero memory — the paper's claim is
  that a context-shifted command is surprising only through task and stage;
* attention: per HGT layer, the attention of every edge of snapshot k (mean over
  heads, and per head), with node keys;
* memory: norm of the memory of the four context nodes over the last snapshots,
  and the cosine similarity between consecutive memories.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from ..config.io import load_config
from .net import HEAD_SLOTS, GraphiotModel, Memory
from .prepare import prepare
from .runner import load_dataset, model_dir
from .train import Trainer

TRAJECTORY = 24


class Explainer:
    def __init__(self, run: Path, seed: int, variant: str = "full"):
        self.run, self.seed, self.variant = Path(run), seed, variant
        self.cfg = load_config(self.run / "config.yaml")
        self.prep = prepare(load_dataset(self.run, self.cfg), variant)
        tr = Trainer(self.prep, self.cfg, seed=seed, out_dir=model_dir(self.run, variant, seed))
        tr.load_best()
        self.model: GraphiotModel = tr.model.eval()
        mp = self.run / "scores" / f"{variant}_seed{seed}_memory.pt"
        self.mem_ckpts = torch.load(mp, weights_only=False) if mp.exists() else [(0, Memory(self.model.d).state())]

    @torch.no_grad()
    def _memory_until(self, k: int, watch: list[str]) -> tuple[Memory, list[dict], dict]:
        """Memory after snapshot k, the trajectory of the watched keys, and attention at k."""
        start_k = max(0, k - TRAJECTORY)
        ck = max((c for c in self.mem_ckpts if c[0] <= start_k), key=lambda c: c[0], default=(0, None))
        mem = Memory(self.model.d)
        if ck[1] is not None:
            mem.load(ck[1])
        traj, prev = [], {}
        attention = {}
        for kk in range(ck[0], k + 1):
            chunk = self.prep.chunk((kk,), self.model.reverse_edges)
            h = self.model.encode(chunk, keep_attention=(kk == k))
            self.model.step_memory(chunk, h, 0, mem)
            if kk >= start_k:
                row = {"k": kk, "T": float(self.prep.ds.times[kk])}
                for key in watch:
                    i = mem.index.get(key)
                    v = mem.M[i] if i is not None else torch.zeros(self.model.d)
                    row[key] = {"norm": float(v.norm()),
                                "cos_prev": float(torch.nn.functional.cosine_similarity(v, prev[key], dim=0))
                                if key in prev and prev[key].norm() > 0 and v.norm() > 0 else None}
                    prev[key] = v.clone()
                traj.append(row)
            if kk == k:
                attention = self._attention(chunk)
        return mem, traj, attention

    def _attention(self, chunk) -> dict:
        snap = chunk.snaps[0]
        out = []
        for layer, att in enumerate(self.model.last_attention or []):
            for rel, alpha in (att or {}).items():
                ei, _ = chunk.edges[rel]
                a = alpha.cpu().numpy()
                for j in range(ei.shape[1]):
                    out.append({"layer": layer, "relation": "__".join(rel),
                                "src": snap.keys[rel[0]][int(ei[0, j])], "dst": snap.keys[rel[2]][int(ei[1, j])],
                                "alpha": float(a[j].mean()), "alpha_heads": [round(float(x), 4) for x in a[j]]})
        return {"edges": out, "nodes": {nt: ks for nt, ks in snap.keys.items() if ks}}

    @torch.no_grad()
    def explain(self, command_idx: int, top: int = 8) -> dict:
        c = self.prep.cmd[self.prep.cmd["command_idx"] == command_idx]
        if c.empty:
            raise KeyError(f"command {command_idx} not found")
        row = c.iloc[0]
        k = int(row["k"])
        stage = self.prep.ds.tag_stage.get(row["tag"], "")
        watch = [f"task:{row['task_id']}", f"agent:{row['agent_id']}", f"stage:{stage}", f"tag:{row['tag']}"]
        if k < 0:
            mem, traj, attention = Memory(self.model.d), [], {"edges": [], "nodes": {}}
        else:
            mem, traj, attention = self._memory_until(k, watch)
        hb = self.prep.head_batch(c, mem, "cpu")
        lp, lse = self.model.log_probs(mem, hb)
        logits = self.model.head_logits(mem, hb)
        probs = (logits - lse[0]).exp()                                         # [n_cand, Q]
        allowed = self.prep.ds.allowed.get(row["task_type"], [])
        flat = [(allowed[i], b, float(probs[i, b])) for i in range(len(allowed)) for b in range(self.model.q)
                if math.isfinite(float(logits[i, b]))]
        flat.sort(key=lambda x: -x[2])
        score = float(-lp[0])
        contrib = {}
        on = self.model.slot_on.clone()
        for si, slot in enumerate(HEAD_SLOTS):
            if on[si * self.model.d] == 0:
                contrib[slot] = None
                continue
            self.model.slot_on[si * self.model.d:(si + 1) * self.model.d] = 0
            contrib[slot] = score - float(-self.model.log_probs(mem, hb)[0][0])   # surprise lost without it
            self.model.slot_on.copy_(on)
        edges = attention["edges"]
        focus = set(watch)
        return {
            "command": {k_: (v.item() if hasattr(v, "item") else v) for k_, v in row[
                ["command_idx", "ts", "task_id", "task_type", "agent_id", "tool", "tag", "value", "bin", "phase",
                 "decision", "split", "k"]].items()},
            "snapshot": {"k": k, "T": float(self.prep.ds.times[k]) if k >= 0 else None},
            "score": score, "in_allowed_set": bool(hb.target_cand[0] >= 0),
            "observed": {"tag": row["tag"], "bin": int(row["bin_m"]),
                         "p": float(math.exp(-score)) if math.isfinite(score) else 0.0},
            "top": [{"tag": t, "bin": b, "p": p} for t, b, p in flat[:top]],
            "context_contribution": contrib,
            "attention_into_context": [e for e in edges if e["dst"] in focus],
            "attention": edges[:4000],
            "nodes": attention["nodes"],
            "memory_trajectory": traj,
            "watch": watch,
        }
