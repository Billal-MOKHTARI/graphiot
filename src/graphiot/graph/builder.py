"""Snapshot builder for the task–plant graph (paper §IV-A, Table II).

Snapshot ``k`` is taken at ``T_k = t0 + k * stride`` and contains the events with
timestamps in ``[T_k - window, T_k)``. Node features that describe a *state*
(tag value, rolling mean/std, stage phase, time in phase, session age, time since
assignment) use only samples strictly before ``T_k``; flows enter a snapshot only
once complete (``flow_end < T_k``). A command at time ``t`` is scored from the last
snapshot with ``T_k <= t`` (:func:`snapshot_index_for`) plus its own fields, so no
event at or after ``t`` can reach its score — the causal scoring rule, tested in
``tests/test_causal_rule.py``.

Inputs come exclusively from :class:`graphiot.data.monitor_view.MonitorView` and
the static topology (tag -> PLC/stage, host IP -> role). Values are kept in
engineering units; normalisation is fitted later on the training split only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.monitor_view import MonitorView
from .schema import (EDGE_TYPES, FLOW_FEATURES, FUNCTION_CODES, HOST_FLOW_STATS, HOST_ROLES,
                     PHASES, STAGES, TAG_KINDS, TASK_TYPES, TOOLS)

_FC_TABLE = {1: "coil", 5: "coil", 15: "coil", 2: "discrete", 3: "holding", 6: "holding",
             16: "holding", 4: "input"}
_STAGE_PREFIX = {"mixing": "mix", "coating": "coat", "slitting": "slit"}


@dataclass
class Snapshot:
    T: float
    nodes: dict[str, list[str]]                      # node type -> stable keys
    x: dict[str, np.ndarray]                         # node type -> features [n, d]
    edges: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=dict)
    # edge type -> (src index, dst index, attributes [m, d])


@dataclass
class Topology:
    tags: pd.DataFrame                               # name, plc, stage, kind, table, modbus_address
    host_roles: dict[str, str]                       # ip -> role
    plc_ips: dict[str, str]                          # PLC name -> ip
    writable_tags: list[str]

    @classmethod
    def from_config(cls, view: MonitorView, net) -> "Topology":
        roles = {net.gateway_ip: "gateway", net.hmi_ip: "hmi", net.ews_ip: "ews"}
        roles.update({ip: "plc" for ip in net.plc_ips.values()})
        topo = view.topology
        writable = sorted(topo.loc[topo["kind"].isin(["setpoint", "command", "alarm_ack"]), "name"])
        return cls(tags=topo, host_roles=roles, plc_ips=dict(net.plc_ips), writable_tags=writable)


def snapshot_times(t_start: float, t_end: float, stride: float) -> np.ndarray:
    return np.arange(t_start + stride, t_end + 1e-9, stride)


def snapshot_index_for(times: np.ndarray, t: float) -> int:
    """Index of the last snapshot with T_k <= t (-1 if none): the causal rule."""
    return int(np.searchsorted(times, t, side="right")) - 1


class SnapshotBuilder:
    def __init__(self, view: MonitorView, topo: Topology, *, window_s: float = 10.0,
                 stride_s: float = 5.0, rolling_s: float = 60.0):
        self.v, self.topo = view, topo
        self.window, self.stride, self.rolling = window_s, stride_s, rolling_s
        self.tag_names = list(topo.tags["name"])
        self.tag_idx = {n: i for i, n in enumerate(self.tag_names)}
        self.tag_meta = topo.tags.set_index("name")
        self.plc_names = sorted(topo.plc_ips)
        self.ip_to_plc = {ip: p for p, ip in topo.plc_ips.items()}
        self.addr_tag = {(r.plc, r.table, int(r.modbus_address)): r.name for r in topo.tags.itertuples()}
        # Register codecs (scale, signedness): static device configuration, like any SCADA's.
        from ..plant.tags import catalog_by_name
        self._codec = catalog_by_name()
        self._prepare()

    # ---- pre-sorted event arrays -------------------------------------------
    def _prepare(self) -> None:
        v = self.v
        self.tags_df = v.tags.sort_values("ts").reset_index(drop=True) if not v.tags.empty else v.tags
        self.series = {t: (g["ts"].to_numpy(), g["value"].to_numpy(dtype=float))
                       for t, g in self.tags_df.groupby("tag")} if not v.tags.empty else {}
        self.calls = v.gateway_log.sort_values("ts").reset_index(drop=True) if not v.gateway_log.empty else v.gateway_log
        self.tasks = v.tasks.sort_values("ts").reset_index(drop=True) if not v.tasks.empty else v.tasks
        self.flows = v.flows.sort_values("flow_end").reset_index(drop=True) if not v.flows.empty else v.flows
        self.modbus = v.modbus.sort_values("ts").reset_index(drop=True) if not v.modbus.empty else v.modbus
        self.commands = v.commands.sort_values("ts").reset_index(drop=True) if not v.commands.empty else v.commands
        # An agent exists from its first tool call or its first task assignment, whichever is earlier.
        first: dict[str, float] = {}
        for df in (self.calls, self.tasks):
            if not df.empty:
                for ag, t in df.groupby("agent_id")["ts"].min().items():
                    first[ag] = min(t, first.get(ag, t))
        self.agent_first = first

    @staticmethod
    def _rows(df: pd.DataFrame, col: str, a: float, b: float) -> pd.DataFrame:
        if df.empty:
            return df
        ts = df[col].to_numpy()
        return df.iloc[np.searchsorted(ts, a, "left"):np.searchsorted(ts, b, "left")]

    def _last_before(self, tag: str, T: float) -> float | None:
        s = self.series.get(tag)
        if s is None:
            return None
        i = int(np.searchsorted(s[0], T, "left")) - 1
        return float(s[1][i]) if i >= 0 else None

    def _rolling(self, tag: str, T: float) -> tuple[float, float]:
        s = self.series.get(tag)
        if s is None:
            return 0.0, 0.0
        lo, hi = np.searchsorted(s[0], T - self.rolling, "left"), np.searchsorted(s[0], T, "left")
        w = s[1][lo:hi]
        return (float(w.mean()), float(w.std())) if len(w) else (0.0, 0.0)

    # ---- one snapshot ------------------------------------------------------
    def build(self, T: float) -> Snapshot:
        a = T - self.window
        nodes: dict[str, list[str]] = {}
        x: dict[str, np.ndarray] = {}
        edges: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        # tags (static set)
        nodes["tag"] = [f"tag:{n}" for n in self.tag_names]
        tx = []
        for n in self.tag_names:
            kind = self.tag_meta.at[n, "kind"]
            cur = self._last_before(n, T)
            mu, sd = self._rolling(n, T)
            tx.append([float(kind == k) for k in TAG_KINDS] + [cur if cur is not None else 0.0, mu, sd])
        x["tag"] = np.asarray(tx, dtype=np.float32)

        # stages (phase from the stage's PLC phase tag, strictly before T)
        nodes["stage"] = [f"stage:{s}" for s in STAGES]
        sx = []
        for s in STAGES:
            p = _STAGE_PREFIX[s]
            ph = self._last_before(f"{p}.phase", T)
            tip = self._last_before(f"{p}.time_in_phase", T)
            sx.append([float(ph is not None and int(ph) == i) for i in range(len(PHASES))] + [tip or 0.0])
        x["stage"] = np.asarray(sx, dtype=np.float32)

        # PLCs: Modbus request rate in the window
        mb = self._rows(self.modbus, "ts", a, T)
        nodes["plc"] = [f"plc:{p}" for p in self.plc_names]
        x["plc"] = np.asarray([[float((mb["dst_ip"] == self.topo.plc_ips[p]).sum()) / self.window
                                if not mb.empty else 0.0] for p in self.plc_names], dtype=np.float32)

        # gateway: call rate, error rate
        calls = self._rows(self.calls, "ts", a, T)
        nodes["gateway"] = ["gateway:0"]
        n_calls = len(calls)
        n_err = int((calls["decision"] != "allowed").sum()) if n_calls else 0
        x["gateway"] = np.asarray([[n_calls / self.window, n_err / max(1, n_calls)]], dtype=np.float32)

        # agents: seen before T; session age, tool-call rate
        agents = sorted(ag for ag, t0 in self.agent_first.items() if t0 < T)
        nodes["agent"] = [f"agent:{ag}" for ag in agents]
        x["agent"] = np.asarray([[T - self.agent_first[ag],
                                  float((calls["agent_id"] == ag).sum()) / self.window if n_calls else 0.0]
                                 for ag in agents], dtype=np.float32).reshape(-1, 2)

        # tasks: the latest task of each agent assigned before T, plus tasks with calls in the window
        tasks = self.tasks[self.tasks["ts"] < T] if not self.tasks.empty else self.tasks
        active = set()
        if not tasks.empty:
            active |= set(tasks.groupby("agent_id")["task_id"].last())
            if n_calls:
                active |= set(calls["task_id"]) & set(tasks["task_id"])
        trows = tasks[tasks["task_id"].isin(active)] if active else tasks.iloc[0:0]
        nodes["task"] = [f"task:{t}" for t in trows["task_id"]]
        wt = self.topo.writable_tags
        x["task"] = np.asarray(
            [[float(r.task_type == k) for k in TASK_TYPES] + [float(r.target_stage == s) for s in STAGES]
             + [float(t in set(r.requested_tags)) for t in wt] for r in trows.itertuples()],
            dtype=np.float32).reshape(-1, len(TASK_TYPES) + len(STAGES) + len(wt))

        # hosts: static roles always present + unknown IPs seen in the window
        fl = self._rows(self.flows, "flow_end", a, T)
        ips = set(self.topo.host_roles)
        for df, cols in ((fl, ("src_ip", "dst_ip")), (mb, ("src_ip", "dst_ip"))):
            if not df.empty:
                for c in cols:
                    ips |= set(df[c])
        hosts = sorted(ips)
        nodes["host"] = [f"host:{ip}" for ip in hosts]
        hx = []
        for ip in hosts:
            role = self.topo.host_roles.get(ip, "unknown")
            if fl.empty:
                stats = [0.0] * len(HOST_FLOW_STATS)
            else:
                # IN_* = client->server, OUT_* = server->client (capture/pcap.py).
                c, s = fl[fl["src_ip"] == ip], fl[fl["dst_ip"] == ip]   # ip as client / as server
                stats = [len(c), len(s),
                         float(c["IN_BYTES"].sum() + s["OUT_BYTES"].sum()),    # bytes sent
                         float(c["OUT_BYTES"].sum() + s["IN_BYTES"].sum()),    # bytes received
                         float(c["IN_PKTS"].sum() + s["OUT_PKTS"].sum()),      # packets sent
                         float(c["OUT_PKTS"].sum() + s["IN_PKTS"].sum())]      # packets received
            hx.append([float(role == r) for r in HOST_ROLES] + [float(np.log1p(s)) for s in stats])
        x["host"] = np.asarray(hx, dtype=np.float32)

        hidx = {ip: i for i, ip in enumerate(hosts)}
        tidx = {k.split(":", 1)[1]: i for i, k in enumerate(nodes["task"])}
        aidx = {k.split(":", 1)[1]: i for i, k in enumerate(nodes["agent"])}

        # assigned: task -> agent (time since assignment)
        if len(trows):
            edges[("task", "assigned", "agent")] = (
                np.asarray([tidx[r.task_id] for r in trows.itertuples()]),
                np.asarray([aidx[r.agent_id] for r in trows.itertuples()]),
                np.asarray([[T - r.ts] for r in trows.itertuples()], dtype=np.float32))

        # invokes: agent -> gateway, one edge per tool call in the window
        if n_calls:
            ok = calls[calls["agent_id"].isin(aidx)]
            edges[("agent", "invokes", "gateway")] = (
                np.asarray([aidx[ag] for ag in ok["agent_id"]]), np.zeros(len(ok), dtype=int),
                np.asarray([[float(tool == t) for t in TOOLS] for tool in ok["tool"]], dtype=np.float32))

        # writes / reads from Modbus transactions (gateway or host -> tag)
        self._rw_edges(mb, T, hidx, edges)

        # flow: host -> host, completed flows only
        if not fl.empty:
            feats = np.log1p(fl[FLOW_FEATURES].to_numpy(dtype=float).clip(min=0)).astype(np.float32)
            edges[("host", "flow", "host")] = (np.asarray([hidx[s] for s in fl["src_ip"]]),
                                               np.asarray([hidx[d] for d in fl["dst_ip"]]), feats)

        # static: controls (PLC -> tag), belongs (tag -> stage)
        pidx = {p: i for i, p in enumerate(self.plc_names)}
        edges[("plc", "controls", "tag")] = (
            np.asarray([pidx[self.tag_meta.at[n, "plc"]] for n in self.tag_names]),
            np.arange(len(self.tag_names)), np.ones((len(self.tag_names), 1), dtype=np.float32))
        sidx = {s: i for i, s in enumerate(STAGES)}
        edges[("tag", "belongs", "stage")] = (
            np.arange(len(self.tag_names)),
            np.asarray([sidx[self.tag_meta.at[n, "stage"]] for n in self.tag_names]),
            np.ones((len(self.tag_names), 1), dtype=np.float32))
        return Snapshot(T=T, nodes=nodes, x=x, edges=edges)

    def _rw_edges(self, mb: pd.DataFrame, T: float, hidx: dict, edges: dict) -> None:
        if mb.empty:
            return
        gw_ip = next((ip for ip, r in self.topo.host_roles.items() if r == "gateway"), None)
        buckets: dict[tuple, list] = {}
        for r in mb.itertuples():
            plc = self.ip_to_plc.get(r.dst_ip)
            fc = int(r.function_code)
            if plc is None or fc not in _FC_TABLE or r.exception:
                continue
            is_write = fc in (5, 6, 15, 16)
            vals = list(r.write_values) if is_write else list(r.read_values)
            for i, raw in enumerate(vals):
                tag = self.addr_tag.get((plc, _FC_TABLE[fc], int(r.address) + i))
                if tag is None:
                    continue
                raw = int(raw)
                if self._codec[tag].dtype.value == "int" and raw > 32767:
                    raw -= 65536                                   # signed 16-bit register
                src_type = "gateway" if r.src_ip == gw_ip else "host"
                key = (src_type, "writes" if is_write else "reads", r.src_ip, tag, fc)
                buckets.setdefault(key, []).append((r.ts, raw))
        per_type: dict[tuple, list] = {}
        for (src_type, rel, ip, tag, fc), evs in buckets.items():
            evs.sort()
            rows = evs if rel == "writes" else evs[-1:]           # aggregate reads, keep writes
            for ts, raw in rows:
                prev = self._last_before(tag, ts)
                val = self._codec[tag].decode(raw)
                change = val - prev if prev is not None else 0.0
                src = 0 if src_type == "gateway" else hidx[ip]
                per_type.setdefault((src_type, rel, "tag"), []).append(
                    (src, self.tag_idx[tag], [float(fc == f) for f in FUNCTION_CODES] + [val, change]))
        for et, rows in per_type.items():
            edges[et] = (np.asarray([r[0] for r in rows]), np.asarray([r[1] for r in rows]),
                         np.asarray([r[2] for r in rows], dtype=np.float32))

    def build_all(self, t_start: float | None = None, t_end: float | None = None) -> tuple[np.ndarray, list[Snapshot]]:
        candidates = [df["ts"].min() for df in (self.calls, self.tags_df, self.modbus) if not df.empty]
        t0 = t_start if t_start is not None else float(min(candidates))
        t1 = t_end if t_end is not None else float(max(df["ts"].max() for df in (self.calls, self.tags_df, self.modbus)
                                                        if not df.empty))
        times = snapshot_times(t0, t1, self.stride)
        return times, [self.build(T) for T in times]
