"""Synthesiser: regenerate the monitor's streams for an edited episode (paper §IV-C).

An edit changes the agent's event stream (an inserted read, an interleaved benign
write, a split setpoint change, a time shift). The detector scores from the monitor
view, so to query its score on an edited episode we must regenerate what the monitor
would have recorded:

* ``tags`` (1 Hz) — the plant trajectory under the *edited writes*, produced by the
  same deterministic twin the oracle uses (so plant-state features are consistent);
* ``commands`` / ``modbus`` / ``gateway_log`` — one row per edited tool call;
* ``flows`` / ``tasks`` — kept from the base capture (flows are 5–15 s aggregates;
  the task is unchanged — the attacker keeps its identity and task, T1).

Only the episode window is rebuilt; events outside it keep the base capture, so
agent session ages and cross-window context stay correct. The twin is advanced from
a checkpoint at the window start (built once per episode), so each candidate edit
costs one short replay, not a full-run replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..config.schema import ExperimentConfig
from ..data.monitor_view import MonitorView
from ..plant.tags import Table, Tag
from ..sim.plant import PlantSimulator

_WRITE_FC = {Table.COIL: 5, Table.HOLDING: 6}
_READ_FC = {Table.COIL: 1, Table.HOLDING: 3, Table.DISCRETE: 2, Table.INPUT: 4}


@dataclass
class Event:
    ts: float
    tag: str
    value: float          # for writes; ignored for reads
    kind: str             # "write" | "read"
    command_idx: int = -1  # the base command this event implements (-1 = inserted)
    target: bool = False   # a harmful write whose score the attacker wants to lower

    def copy(self, **kw):
        return replace(self, **kw)


@dataclass
class EditedEpisode:
    events: list[Event]
    cost: int = 0
    trail: list[str] = field(default_factory=list)


class WindowSynth:
    def __init__(self, run: Path, cfg: ExperimentConfig, base: MonitorView, tags: dict[str, Tag],
                 *, episode_start: float, episode_end: float, settle_s: float):
        self.cfg, self.base, self.tags = cfg, base, tags
        self.rolling = cfg.graph.rolling_window_s
        self.win_lo = episode_start - self.rolling - cfg.graph.window_s
        self.win_hi = episode_end + settle_s
        self.net = cfg.testbed.network
        self.plc_ip = dict(self.net.plc_ips)
        self.gw_ip = self.net.gateway_ip
        self.monitored = [n for n, t in tags.items() if t.agent_readable or t.kind.value in
                          ("setpoint", "command", "phase", "status", "alarm", "alarm_ack")]
        self.tick_s = cfg.testbed.plc_scan_ms / 1000.0
        # Base writes before the window set the plant; checkpoint the twin at win_lo.
        from ..oracle.core import load_writes, sim_start_ts
        self.all_writes, _, _ = load_writes(run, cfg, tags, source="auto")
        self.t0 = sim_start_ts(run, self.tick_s)
        self._sim = PlantSimulator(tag_map_path=Path(cfg.paths.tag_map) if Path(cfg.paths.tag_map).is_absolute()
                                   else (run.parents[1] / cfg.paths.tag_map),
                                   plc_src_dir=(run.parents[1] / cfg.paths.plc_src_dir),
                                   seed=cfg.testbed.sim_seed, scan_s=self.tick_s)
        self._checkpoint = self._build_checkpoint()
        # Base-view rows outside the window, reused for every candidate.
        self._base_outside = {name: self._outside(getattr(base, name))
                              for name in ("tags", "commands", "gateway_log", "modbus", "flows")}

    def close(self):
        self._sim.close()

    def _tick(self, ts: float) -> int:
        return int(np.ceil((ts - self.t0) / self.tick_s - 1e-9))

    def _build_checkpoint(self) -> dict:
        sim = self._sim
        pending = sorted((w for w in self.all_writes if w.ts < self.win_lo), key=lambda w: w.ts)
        pi, end = 0, self._tick(self.win_lo)
        for k in range(0, end):
            while pi < len(pending) and self._tick(pending[pi].ts) <= k:
                sim.apply_command(pending[pi].tag, pending[pi].value)
                pi += 1
            sim.step()
        return sim.save_state()

    def _outside(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or "ts" not in df:
            return df
        return df[(df["ts"] < self.win_lo) | (df["ts"] >= self.win_hi)]

    # ---- plant trajectory under edited writes ---------------------------------
    def regen_tags(self, writes_in_window: list[tuple[float, str, float]]) -> pd.DataFrame:
        sim = self._sim
        sim.restore_state(self._checkpoint)
        pending = sorted(writes_in_window, key=lambda w: w[0])
        pi, rows = 0, []
        per_sample = max(1, int(round(1.0 / self.tick_s)))
        for k in range(self._tick(self.win_lo), self._tick(self.win_hi)):
            while pi < len(pending) and self._tick(pending[pi][0]) <= k:
                sim.apply_command(pending[pi][1], pending[pi][2])
                pi += 1
            sim.step()
            if k % per_sample == 0:
                ts = self.t0 + (k + 1) * self.tick_s
                for name in self.monitored:
                    t = self.tags[name]
                    rows.append({"ts": ts, "plc": t.plc.value, "tag": name, "value": sim.read(name)})
        return pd.DataFrame(rows)

    def regen_outputs(self, writes_in_window: list[tuple[float, str, float]]) -> pd.DataFrame:
        """Noise-free process outputs under the edited writes (the oracle's twin trace)."""
        from ..oracle.core import SENSOR_STATE
        sim = self._sim
        sim.restore_state(self._checkpoint)
        pending = sorted(writes_in_window, key=lambda w: w[0])
        pi, rows = 0, []
        per_sample = max(1, int(round(1.0 / self.tick_s)))
        for k in range(self._tick(self.win_lo), self._tick(self.win_hi)):
            while pi < len(pending) and self._tick(pending[pi][0]) <= k:
                sim.apply_command(pending[pi][1], pending[pi][2])
                pi += 1
            sim.step()
            if k % per_sample == 0:
                st = sim.physics.true_state()
                rows.append({"ts": self.t0 + (k + 1) * self.tick_s,
                             **{s: st[key] for s, key in SENSOR_STATE.items()}})
        return pd.DataFrame(rows)

    # ---- event rows -----------------------------------------------------------
    def _modbus_row(self, ev: Event, call_id: int, task_id: str) -> dict:
        t = self.tags[ev.tag]
        fc = _WRITE_FC[t.table] if ev.kind == "write" else _READ_FC[t.table]
        raw = t.encode(ev.value) if ev.kind == "write" else 0
        return {"ts": ev.ts, "ts_response": ev.ts + 0.02, "src_ip": self.gw_ip, "src_port": 40000 + call_id % 2000,
                "dst_ip": self.plc_ip[t.plc.value], "dst_port": 502, "transaction_id": call_id, "unit_id": 1,
                "function_code": fc, "address": t.modbus_address, "count": 1,
                "write_values": [raw] if ev.kind == "write" else [], "read_values": [] if ev.kind == "write" else [raw],
                "exception": 0, "request_bytes": 12, "response_bytes": 12 if ev.kind == "write" else 11}

    def build_view(self, events: list[Event], task_id: str, agent_id: str) -> MonitorView:
        writes_win = [(e.ts, e.tag, e.value) for e in events if e.kind == "write"]
        tags = pd.concat([self._base_outside["tags"], self.regen_tags(writes_win)], ignore_index=True).sort_values("ts")
        cmd_rows, gw_rows, mb_rows = [], [], []
        for i, e in enumerate(sorted(events, key=lambda e: e.ts)):
            cid = 900000 + i                       # synthetic call ids, disjoint from base
            t = self.tags[e.tag]
            tool = "write_tag" if e.kind == "write" else "read_tag"
            gw_rows.append({"ts": e.ts, "ts_response": e.ts + 0.05, "call_id": cid, "task_id": task_id,
                            "agent_id": agent_id, "tool": tool, "tag": e.tag, "stage": None,
                            "decision": "allowed", "error": None, "n_modbus": 1,
                            "fcs": [self._modbus_row(e, cid, task_id)["function_code"]]})
            mb_rows.append(self._modbus_row(e, cid, task_id))
            if e.kind == "write":
                cmd_rows.append({"ts": e.ts, "call_id": cid, "task_id": task_id, "agent_id": agent_id,
                                 "tool": "write_tag", "tag": e.tag, "value": e.value, "raw": t.encode(e.value),
                                 "function_code": self._modbus_row(e, cid, task_id)["function_code"],
                                 "decision": "forwarded", "command_idx": e.command_idx})
        mk = lambda base, rows, cols: pd.concat(
            [base, pd.DataFrame(rows, columns=cols if not rows else None)], ignore_index=True).sort_values("ts") \
            if rows else base
        return MonitorView(
            topology=self.base.topology, tasks=self.base.tasks, flows=self.base.flows,
            tags=tags.reset_index(drop=True),
            commands=mk(self._base_outside["commands"], cmd_rows, self.base.commands.columns),
            gateway_log=mk(self._base_outside["gateway_log"], gw_rows, self.base.gateway_log.columns),
            modbus=mk(self._base_outside["modbus"], mb_rows, self.base.modbus.columns))
