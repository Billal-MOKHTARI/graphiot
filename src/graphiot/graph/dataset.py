"""Graph dataset of a run: snapshots + the commands they score, normalised.

* Each write command gets its snapshot ``k`` = last snapshot with ``T_k <= t``
  (causal rule), its split (train / val / test / redteam / excluded), its
  quantile bin, and the allowed (tag, bin) support of its task type — the
  allow-list's write tags (paper §IV-B: "softmax over allowed (tag, bin) pairs").
* Normalisation is fitted on training snapshots only: tag values per tag
  (engineering units differ by orders of magnitude), heavy-tailed counters on a
  log scale, every non-binary column z-scored. One-hot columns are left as is.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config.schema import ExperimentConfig
from ..data.monitor_view import MonitorView, load_monitor_view
from ..data.splits import Splits, chronological_splits
from ..eval.bins import QuantileBins
from ..eval.context import attach_context
from ..plant.allow_list import load_allow_list
from .builder import Snapshot, SnapshotBuilder, Topology, snapshot_index_for
from .schema import EDGE_TYPES, NODE_TYPES, TAG_KINDS

_LOG_NODE_COLS = {"agent": [0], "stage": [4], "host": list(range(5, 11))}   # session age, time in phase, flow stats (already log1p)
_TAG_VALUE_COLS = slice(len(TAG_KINDS), len(TAG_KINDS) + 3)               # cur, mean, std


@dataclass
class Normalizer:
    tag_mu: dict[str, float] = field(default_factory=dict)
    tag_sd: dict[str, float] = field(default_factory=dict)
    node_mu: dict[str, np.ndarray] = field(default_factory=dict)
    node_sd: dict[str, np.ndarray] = field(default_factory=dict)
    node_binary: dict[str, np.ndarray] = field(default_factory=dict)
    edge_mu: dict[tuple, np.ndarray] = field(default_factory=dict)
    edge_sd: dict[tuple, np.ndarray] = field(default_factory=dict)
    edge_binary: dict[tuple, np.ndarray] = field(default_factory=dict)

    # -- tag-relative scaling ---------------------------------------------------
    def _tag_scale(self, snap: Snapshot) -> tuple[dict, dict]:
        x = {k: v.copy() for k, v in snap.x.items()}
        names = [k.split(":", 1)[1] for k in snap.nodes["tag"]]
        mu = np.array([self.tag_mu.get(n, 0.0) for n in names], dtype=np.float32)
        sd = np.array([self.tag_sd.get(n, 1.0) for n in names], dtype=np.float32)
        t = x["tag"]
        t[:, len(TAG_KINDS)] = (t[:, len(TAG_KINDS)] - mu) / sd
        t[:, len(TAG_KINDS) + 1] = (t[:, len(TAG_KINDS) + 1] - mu) / sd
        t[:, len(TAG_KINDS) + 2] = t[:, len(TAG_KINDS) + 2] / sd
        for nt, cols in _LOG_NODE_COLS.items():
            if nt in x and x[nt].size and nt != "host":
                x[nt][:, cols] = np.log1p(np.clip(x[nt][:, cols], 0, None))
        edges = {}
        for et, (s, d, a) in snap.edges.items():
            a = a.copy()
            if et[1] in ("writes", "reads") and len(d):
                tmu, tsd = mu[d], sd[d]
                a[:, -2] = (a[:, -2] - tmu) / tsd           # value
                a[:, -1] = a[:, -1] / tsd                   # change
            elif et[1] == "assigned" and len(a):
                a = np.log1p(np.clip(a, 0, None))
            edges[et] = (s, d, a)
        return x, edges

    def fit(self, snaps: list[Snapshot], tags: pd.DataFrame) -> "Normalizer":
        for tag, g in tags.groupby("tag"):
            v = g["value"].to_numpy(dtype=float)
            self.tag_mu[tag] = float(v.mean())
            self.tag_sd[tag] = float(v.std()) if v.std() > 1e-6 else 1.0
        acc_x: dict[str, list] = {}
        acc_e: dict[tuple, list] = {}
        for s in snaps:
            x, e = self._tag_scale(s)
            for nt, arr in x.items():
                if arr.size:
                    acc_x.setdefault(nt, []).append(arr)
            for et, (_, _, a) in e.items():
                if len(a):
                    acc_e.setdefault(et, []).append(a)
        for nt, parts in acc_x.items():
            m = np.concatenate(parts)
            self.node_binary[nt] = np.all(np.isin(m, (0.0, 1.0)), axis=0)
            self.node_mu[nt], self.node_sd[nt] = m.mean(0), np.where(m.std(0) > 1e-6, m.std(0), 1.0)
        for et, parts in acc_e.items():
            m = np.concatenate(parts)
            self.edge_binary[et] = np.all(np.isin(m, (0.0, 1.0)), axis=0)
            self.edge_mu[et], self.edge_sd[et] = m.mean(0), np.where(m.std(0) > 1e-6, m.std(0), 1.0)
        return self

    def transform(self, snap: Snapshot) -> Snapshot:
        x, e = self._tag_scale(snap)
        for nt, arr in x.items():
            if arr.size and nt in self.node_mu:
                z = (arr - self.node_mu[nt]) / self.node_sd[nt]
                x[nt] = np.where(self.node_binary[nt], arr, z).astype(np.float32)
        out_e = {}
        for et, (s, d, a) in e.items():
            if len(a) and et in self.edge_mu:
                z = (a - self.edge_mu[et]) / self.edge_sd[et]
                a = np.where(self.edge_binary[et], a, z).astype(np.float32)
            out_e[et] = (s, d, np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0))
        return Snapshot(T=snap.T, nodes=snap.nodes,
                        x={k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in x.items()},
                        edges=out_e)


@dataclass
class GraphDataset:
    times: np.ndarray
    snaps: list[Snapshot]                 # normalised
    commands: pd.DataFrame
    splits: Splits
    bins: QuantileBins
    norm: Normalizer
    allowed: dict[str, list[str]]         # task type -> writable tags allowed by the allow-list
    writable_tags: list[str]
    tag_stage: dict[str, str]
    snap_split: np.ndarray                # split of each snapshot (by T)
    window_s: float = 10.0                # Δ
    stride_s: float = 5.0

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "GraphDataset":
        with open(path, "rb") as fh:
            return pickle.load(fh)


def build_dataset(run_dir: str | Path, cfg: ExperimentConfig, *, root: str | Path = ".",
                  view: MonitorView | None = None) -> GraphDataset:
    run, root = Path(run_dir), Path(root)
    view = view or load_monitor_view(run, tag_map=root / cfg.paths.tag_map)
    topo = Topology.from_config(view, cfg.testbed.network)
    builder = SnapshotBuilder(view, topo, window_s=cfg.graph.window_s, stride_s=cfg.graph.stride_s,
                              rolling_s=cfg.graph.rolling_window_s)
    times, raw = builder.build_all()

    # Commands of the monitor view. Oracle labels are NOT part of the dataset: training,
    # bins, normalisation and the threshold never use them; evaluation joins them later
    # on command_idx (the row order of capture/commands.parquet).
    cmds = view.commands.reset_index().rename(columns={"index": "command_idx"})
    cmds = attach_context(cmds, view.tags, view.tasks)

    periods = pd.read_parquet(run / "groundtruth" / "agent_periods.parquet") \
        if (run / "groundtruth" / "agent_periods.parquet").exists() else None
    if periods is not None and len(periods):
        sp = chronological_splits(periods, train=cfg.splits.train, val=cfg.splits.val,
                                  exclude_after_redteam_s=cfg.splits.exclude_after_redteam_s)
    else:                                                    # no ground truth: treat all as benign time
        sp = chronological_splits(pd.DataFrame({"start_ts": [times[0] - cfg.graph.stride_s],
                                                "end_ts": [times[-1]], "actor": ["operator"]}),
                                  train=cfg.splits.train, val=cfg.splits.val)
    cmds["split"] = [sp.assign(t) for t in cmds["ts"]]
    cmds["k"] = [snapshot_index_for(times, t) for t in cmds["ts"]]
    snap_split = np.array([sp.assign(T - 1e-6) for T in times])

    # Bins on the writes of benign training time (+ 1 Hz series fallback, Q19). Benign time
    # is operator time by construction of the splits; no oracle label is used.
    benign_train = cmds[(cmds["split"] == "train") & (cmds["decision"] == "forwarded")]
    t_train_end = max((b for _, b in sp.train), default=times[0])
    series = view.tags[(view.tags["ts"] < t_train_end) & view.tags["tag"].isin(topo.writable_tags)]
    bins = QuantileBins(q=cfg.model.q_bins).fit(benign_train[["tag", "value"]], series[["tag", "value"]])
    cmds["bin"] = [bins.bin(t, v) for t, v in zip(cmds["tag"], cmds["value"])]

    norm = Normalizer().fit([s for s, sp_ in zip(raw, snap_split) if sp_ == "train"],
                            view.tags[view.tags["ts"] < t_train_end])
    snaps = [norm.transform(s) for s in raw]
    al = load_allow_list(root / cfg.paths.allow_list)
    allowed = {tt: [t for t in pol.write_tags if t in topo.writable_tags] for tt, pol in al.task_types.items()}
    tag_stage = dict(zip(topo.tags["name"], topo.tags["stage"]))
    return GraphDataset(times=times, snaps=snaps, commands=cmds, splits=sp, bins=bins, norm=norm,
                        allowed=allowed, writable_tags=topo.writable_tags, tag_stage=tag_stage,
                        snap_split=snap_split, window_s=cfg.graph.window_s, stride_s=cfg.graph.stride_s)
