"""B2 — same-feature tabular model (paper §V-E): XGBoost predicts the same (tag, bin)
target from the same features, flattened per command, and scores the same surprise.

Features of a command = snapshot k (the one our model uses, last T_k <= t), flattened:
its task node and agent node, the gateway, every stage, PLC and tag node, the hosts
with a fixed role (unknown hosts summed), and per relation the edge count and mean
attributes. The class is the (tag, bin) pair; the probability is renormalised over
the pairs the task type may write, like the model's softmax. A pair never seen in
training gets probability 1e-12 (finite, above every seen pair).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..graph.dataset import GraphDataset
from ..graph.schema import EDGE_TYPES, HOST_ROLES, NODE_TYPES, edge_dims, node_dims

P_FLOOR = 1e-12


def _host_block(snap, nd: int) -> np.ndarray:
    x, keys = snap.x.get("host"), snap.nodes.get("host", [])
    out = np.zeros((len(HOST_ROLES), nd), dtype=np.float32)
    for key, row in zip(keys, x if x is not None else []):
        role = int(np.argmax(row[:len(HOST_ROLES)])) if row[:len(HOST_ROLES)].sum() else len(HOST_ROLES) - 1
        out[role] += row
    return out.ravel()


def flatten(ds: GraphDataset, k: int, task_id: str, agent_id: str) -> np.ndarray:
    nd, ed = node_dims(len(ds.writable_tags)), edge_dims()
    if k < 0:
        snap = None
    else:
        snap = ds.snaps[k]
    parts = []
    for nt, key in (("task", f"task:{task_id}"), ("agent", f"agent:{agent_id}")):
        v = np.zeros(nd[nt], dtype=np.float32)
        if snap is not None and key in snap.nodes.get(nt, []):
            v = snap.x[nt][snap.nodes[nt].index(key)]
        parts.append(v)
    for nt in ("gateway", "stage", "plc", "tag"):
        n_fixed = {"gateway": 1, "stage": 3, "plc": 3, "tag": len(ds.tag_stage)}[nt]
        v = np.zeros((n_fixed, nd[nt]), dtype=np.float32)
        if snap is not None and snap.x.get(nt) is not None and snap.x[nt].size:
            v[:len(snap.x[nt])] = snap.x[nt][:n_fixed]
        parts.append(v.ravel())
    parts.append(_host_block(snap, nd["host"]) if snap is not None else np.zeros(len(HOST_ROLES) * nd["host"], np.float32))
    for et in EDGE_TYPES:
        cnt, mean = 0.0, np.zeros(ed[et], dtype=np.float32)
        if snap is not None and et in snap.edges and len(snap.edges[et][0]):
            a = snap.edges[et][2]
            cnt, mean = float(len(a)), a.mean(0)
        parts.append(np.concatenate([[math.log1p(cnt)], mean]).astype(np.float32))
    return np.concatenate(parts)


class SameFeatureXGB:
    def __init__(self, ds: GraphDataset, *, n_estimators: int, max_depth: int, learning_rate: float, seed: int = 0):
        self.ds = ds
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                           random_state=seed, tree_method="hist", n_jobs=4)
        self.classes: list[tuple[str, int]] = []

    def _xy(self, cmds: pd.DataFrame):
        X = np.stack([flatten(self.ds, int(r.k), r.task_id, r.agent_id) for r in cmds.itertuples()]) \
            if len(cmds) else np.zeros((0, 1))
        pairs = list(zip(cmds["tag"], cmds["bin_m"]))
        return X, pairs

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> "SameFeatureXGB":
        from xgboost import XGBClassifier
        Xtr, ptr = self._xy(train)
        self.classes = sorted(set(ptr))
        idx = {c: i for i, c in enumerate(self.classes)}
        ytr = np.array([idx[p] for p in ptr])
        Xva, pva = self._xy(val)
        keep = [i for i, p in enumerate(pva) if p in idx]
        if len(self.classes) == 1:                         # degenerate: one class only
            self.model = None
            return self
        kw = dict(self.params, objective="multi:softprob", eval_metric="mlogloss")
        if keep:
            kw["early_stopping_rounds"] = 20
        self.model = XGBClassifier(**kw)
        # XGBoost needs labels 0..K-1 all present: map to the classes seen in training
        fit_kw = {"verbose": False}
        if keep:
            fit_kw["eval_set"] = [(Xva[keep], np.array([idx[pva[i]] for i in keep]))]
        self.model.fit(Xtr, ytr, **fit_kw)
        return self

    def score(self, cmds: pd.DataFrame) -> np.ndarray:
        X, pairs = self._xy(cmds)
        if self.model is None:
            P = np.ones((len(cmds), 1))
        else:
            P = self.model.predict_proba(X)
        idx = {c: i for i, c in enumerate(self.classes)}
        out = np.empty(len(cmds))
        for j, (r, pair) in enumerate(zip(cmds.itertuples(), pairs)):
            allowed = set(self.ds.allowed.get(r.task_type, []))
            if r.tag not in allowed:
                out[j] = math.inf
                continue
            cols = [i for i, (t, _) in enumerate(self.classes) if t in allowed]
            n_unseen = sum(self.ds.bins.n_bins(t) for t in allowed) - len(cols)
            z = P[j, cols].sum() + max(n_unseen, 0) * P_FLOOR
            p = P[j, idx[pair]] if pair in idx else P_FLOOR
            out[j] = -math.log(max(p, P_FLOOR) / max(z, P_FLOOR))
        return out
