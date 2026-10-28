"""Training, checkpoints with exact resume, threshold and scoring (paper §IV-B, §V-D).

* Truncated BPTT over chunks of ``model.batch_snapshots`` consecutive snapshots of
  the benign training sequence (the paper's "batch of 32 snapshots"); the memory
  is detached between chunks and reset at the start of each epoch.
* Adam (lr 1e-3), gradient clipping, early stopping on the validation negative
  log-likelihood (patience ``model.patience``).
* A checkpoint holds the model, the optimiser, the memory, the position in the
  epoch, the early-stopping state and every RNG state, so a resumed run continues
  exactly where it stopped (``tests/test_checkpoint_resume.py``).
* Threshold: the (1 - fpr_target) quantile of the surprise of benign validation
  commands that the gateway forwarded, computed on the benign sequence only.
"""

from __future__ import annotations

import json
import math
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from ..config.schema import ExperimentConfig
from ..graph.schema import edge_dims, node_dims
from .net import GraphiotModel, Memory
from .prepare import Prepared


@dataclass
class TrainState:
    epoch: int = 0
    chunk: int = 0                     # next chunk of the current epoch
    step: int = 0                      # optimiser steps so far
    best_val: float = math.inf
    best_epoch: int = -1
    bad_epochs: int = 0
    done: bool = False
    stop_reason: str = ""
    history: list = field(default_factory=list)
    seconds: float = 0.0
    epoch_loss_sum: float = 0.0        # running training loss of the current epoch
    epoch_loss_n: int = 0


def _rng_state() -> dict:
    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st: dict) -> None:
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def dataset_fingerprint(prep: Prepared) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(np.asarray(prep.ds.times, dtype=np.float64).tobytes())
    h.update(pd.util.hash_pandas_object(prep.cmd[["ts", "tag", "value"]], index=False).values.tobytes())
    h.update(prep.variant.encode())
    return h.hexdigest()[:16]


class Trainer:
    def __init__(self, prep: Prepared, cfg: ExperimentConfig, *, seed: int, out_dir: str | Path,
                 device: str | torch.device = "cpu", reverse_edges: bool = True):
        self.prep, self.cfg, self.seed = prep, cfg, seed
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        m = cfg.model
        seed_everything(seed)
        self.model = GraphiotModel(node_dims(len(prep.ds.writable_tags)), edge_dims(),
                                   n_tags=len(prep.ds.writable_tags), q=m.q_bins, n_bins=prep.n_bins,
                                   d=m.d, layers=m.hgt_layers, heads=m.heads,
                                   reverse_edges=reverse_edges, variant=prep.variant).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=m.lr, weight_decay=m.weight_decay)
        self.state = TrainState()
        self.mem = Memory(m.d, self.device)
        self.fingerprint = dataset_fingerprint(prep)
        B = m.batch_snapshots
        seq = prep.benign_pos[:prep.train_end]
        self.train_chunks = [tuple(int(k) for k in seq[i:i + B]) for i in range(0, len(seq), B)]
        self.train_offsets = list(range(0, len(seq), B))
        c = prep.cmd
        sel = (c["split"] == "train") & c["forwarded"] & c["tag_in_vocab"] & (c["bpos"] >= 0) \
            & (c["bpos"] < prep.train_end)
        self.train_by_pos = {int(p): g for p, g in c[sel].groupby("bpos")}
        vsel = (c["split"] == "val") & c["forwarded"] & c["tag_in_vocab"] & (c["bpos"] >= 0)
        self.val_cmds = c[vsel]
        self.log_path = self.out / "train_log.jsonl"

    # ---- one optimiser step on one chunk ---------------------------------------
    def _train_chunk(self, ci: int) -> float | None:
        prep, model = self.prep, self.model
        chunk = prep.chunk(self.train_chunks[ci], model.reverse_edges).to(self.device)
        h = model.encode(chunk)
        terms = []
        for i in range(len(chunk.snaps)):
            model.step_memory(chunk, h, i, self.mem)
            rows = self.train_by_pos.get(self.train_offsets[ci] + i)
            if rows is not None:
                hb = prep.head_batch(rows, self.mem, self.device)
                lp, _ = model.log_probs(self.mem, hb)
                terms.append(-lp[hb.target_cand >= 0])
        loss_val = None
        if terms:
            loss = torch.cat(terms).mean()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.model.grad_clip)
            self.opt.step()
            loss_val = float(loss.detach())
        self.mem.detach()
        return loss_val

    # ---- benign sequence pass (validation, threshold) -----------------------------
    @torch.no_grad()
    def benign_scores(self, which: str = "val") -> pd.DataFrame:
        """Surprise of the benign commands of a split, from the benign sequence."""
        prep, model = self.prep, self.model
        model.eval()
        rows_all = self.val_cmds if which == "val" else prep.cmd[
            (prep.cmd["split"] == which) & prep.cmd["forwarded"] & prep.cmd["tag_in_vocab"] & (prep.cmd["bpos"] >= 0)]
        by_pos = {int(p): g for p, g in rows_all.groupby("bpos")}
        last = max(by_pos) if by_pos else -1
        mem = Memory(model.d, self.device)
        B = self.cfg.model.batch_snapshots
        out = []
        for start in range(0, last + 1, B):
            idx = tuple(int(k) for k in prep.benign_pos[start:min(start + B, last + 1)])
            chunk = prep.chunk(idx, model.reverse_edges).to(self.device)
            h = model.encode(chunk)
            for i in range(len(idx)):
                model.step_memory(chunk, h, i, mem)
                rows = by_pos.get(start + i)
                if rows is not None:
                    hb = prep.head_batch(rows, mem, self.device)
                    lp, _ = model.log_probs(mem, hb)
                    out.append(pd.DataFrame({"command_idx": rows["command_idx"].to_numpy(),
                                             "score": (-lp).cpu().numpy()}))
        model.train()
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["command_idx", "score"])

    def validate(self) -> float:
        s = self.benign_scores("val")
        return float(s["score"].mean()) if len(s) else math.nan

    # ---- fit ---------------------------------------------------------------------
    def fit(self, *, max_steps: int | None = None, stop: threading.Event | None = None,
            on_progress: Callable[[dict], None] | None = None) -> TrainState:
        st, m = self.state, self.cfg.model
        t_start = time.monotonic()
        steps_here = 0
        self.model.train()
        while not st.done and st.epoch < m.max_epochs:
            if st.chunk == 0:
                self.mem = Memory(m.d, self.device)
                st.epoch_loss_sum, st.epoch_loss_n = 0.0, 0
            while st.chunk < len(self.train_chunks):
                if (stop is not None and stop.is_set()) or (max_steps is not None and steps_here >= max_steps):
                    st.seconds += time.monotonic() - t_start
                    self.save(self.out / "last.pt")
                    return st
                loss = self._train_chunk(st.chunk)
                st.chunk += 1
                st.step += 1
                steps_here += 1
                if loss is not None:
                    st.epoch_loss_sum += loss
                    st.epoch_loss_n += 1
                if st.step % m.checkpoint_every_steps == 0:
                    self.save(self.out / "last.pt")
            val = self.validate()
            rec = {"epoch": st.epoch, "step": st.step,
                   "train_nll": st.epoch_loss_sum / st.epoch_loss_n if st.epoch_loss_n else None,
                   "val_nll": None if math.isnan(val) else val,
                   "seconds": round(st.seconds + time.monotonic() - t_start, 2)}
            st.history.append(rec)
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            improved = (not math.isnan(val)) and val < st.best_val - 1e-4
            if improved or (math.isnan(val) and st.best_epoch < 0) or math.isnan(val):
                if improved:
                    st.best_val = val
                    st.bad_epochs = 0
                st.best_epoch = st.epoch
                torch.save({"model": self.model.state_dict(), "model_cfg": self.model.cfg,
                            "epoch": st.epoch, "val_nll": None if math.isnan(val) else val},
                           self.out / "best.pt")
            else:
                st.bad_epochs += 1
            st.epoch += 1
            st.chunk = 0
            if st.bad_epochs >= m.patience:
                st.done, st.stop_reason = True, f"early stopping (patience {m.patience})"
            elif st.epoch >= m.max_epochs:
                st.done, st.stop_reason = True, "max_epochs"
            if on_progress:
                on_progress({**rec, "best_val": None if math.isinf(st.best_val) else st.best_val,
                             "best_epoch": st.best_epoch, "done": st.done})
            self.save(self.out / "last.pt")
        st.seconds += time.monotonic() - t_start
        self.save(self.out / "last.pt")
        return st

    # ---- checkpoints -------------------------------------------------------------
    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save({"model": self.model.state_dict(), "model_cfg": self.model.cfg,
                    "optimizer": self.opt.state_dict(), "memory": self.mem.state(),
                    "state": asdict(self.state), "rng": _rng_state(), "seed": self.seed,
                    "variant": self.prep.variant, "fingerprint": self.fingerprint,
                    "config_model": self.cfg.model.model_dump()}, tmp)
        tmp.replace(path)

    def load(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if ck["fingerprint"] != self.fingerprint:
            raise ValueError(f"checkpoint {path} was trained on a different dataset/variant")
        if ck["config_model"] != self.cfg.model.model_dump():
            raise ValueError(f"checkpoint {path} has different model hyperparameters")
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["optimizer"])
        self.mem.load(ck["memory"])
        self.state = TrainState(**ck["state"])
        _set_rng_state(ck["rng"])

    # ---- threshold and scores ------------------------------------------------------
    def load_best(self) -> None:
        ck = torch.load(self.out / "best.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])

    def threshold(self) -> dict:
        from ..baselines.common import threshold_with_fallback
        val = self.benign_scores("val")["score"].to_numpy(dtype=float)
        fpr = self.cfg.model.fpr_target
        v = val[np.isfinite(val)]
        thr, source, n = threshold_with_fallback(val, [], fpr)
        if source != "benign_validation":                 # no validation commands: fall back to benign training
            train = self.benign_scores("train")["score"].to_numpy(dtype=float)
            thr, source, n = threshold_with_fallback(val, train, fpr)
        out = {"threshold": thr, "source": source, "fpr_target": fpr, "n_val": int(len(v)),
               "n_threshold_scores": int(n),
               "val_fpr_at_threshold": float((v > thr).mean()) if len(v) else None}
        (self.out / "threshold.json").write_text(json.dumps(out, indent=1))
        return out


@torch.no_grad()
def score_timeline(model: GraphiotModel, prep: Prepared, *, batch: int = 32, device="cpu",
                   memory_every: int = 50) -> tuple[pd.DataFrame, list]:
    """Score every command from the full timeline (all snapshots, causal rule).
    Also returns memory snapshots every ``memory_every`` snapshots (for explanations)."""
    model.eval()
    cmd = prep.cmd
    by_k = {int(k): g for k, g in cmd.groupby("k")}
    mem = Memory(model.d, device)
    out, mem_ckpts = [], []
    if -1 in by_k:                                   # before the first snapshot: zero memory
        out.append(_score_rows(model, prep, by_k[-1], mem, device))
    n = len(prep.snaps)
    for start in range(0, n, batch):
        idx = tuple(range(start, min(start + batch, n)))
        chunk = prep.chunk(idx, model.reverse_edges).to(device)
        h = model.encode(chunk)
        for i, k in enumerate(idx):
            if k % memory_every == 0:
                mem_ckpts.append((k, mem.state()))
            model.step_memory(chunk, h, i, mem)
            if k in by_k:
                out.append(_score_rows(model, prep, by_k[k], mem, device))
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return res, mem_ckpts


def _score_rows(model, prep: Prepared, rows: pd.DataFrame, mem: Memory, device) -> pd.DataFrame:
    hb = prep.head_batch(rows, mem, device)
    lp, _ = model.log_probs(mem, hb)
    s = (-lp).cpu().numpy()
    return pd.DataFrame({"command_idx": rows["command_idx"].to_numpy(), "k": rows["k"].to_numpy(),
                         "score": s, "in_allowed_set": (hb.target_cand >= 0).cpu().numpy()})
