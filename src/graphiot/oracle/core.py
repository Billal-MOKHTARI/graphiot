"""The oracle on process outputs (paper §V-B) — reads limits, ground truth, twin.

Primary oracle (twin replay)
    The twin (compiled ST + physics, :class:`PlantSimulator`) replays every
    process-relevant write from the simulator's start, tick-aligned with the live
    noise stream. A process output violates when its active exceedance is at
    least τ of the limit width for at least T_min seconds (1 Hz samples of the
    noise-free twin state).

    * Episode label, paper-literal: a violation occurs in [start, end + settle].
    * Episode label, causal (default; the paper speaks of episodes that *cause* a
      violation): a violation occurs in the window AND disappears when all of the
      episode's writes are removed and the window is replayed.
    * Command label (leave-one-out): a forwarded write of a harmful episode is
      harmful if removing it alone and replaying brings the outputs the episode
      caused back inside the episode criterion (``command_rule="episode_criterion"``)
      or strictly inside the limits (``"strict"``) — Q10.

    Every counterfactual starts from a checkpoint taken at the episode start in
    one factual pass, so a replay costs only its window.

Fallback oracle (logged tags)
    For each red-team episode: harmful if a logged process tag leaves its limits
    for at least T_min seconds within ``fallback_window_s`` of one of its
    commands; the harmful command is the last write to a tag of the same stage
    before the violation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config.schema import ExperimentConfig
from ..plant.limits import LimitBand, OperatingLimits, load_limits
from ..plant.tags import Kind, Plc, Tag, Table, load_tag_map
from ..sim.plant import PlantSimulator

# Limited sensor tag -> key of PlantPhysics.true_state() (noise-free twin output).
SENSOR_STATE = {
    "mix.viscosity": "mix.viscosity", "mix.solids_pct": "mix.solids", "mix.temp": "mix.temp",
    "mix.air_entrain": "mix.air", "mix.speed_actual": "mix.speed",
    "coat.dryer_z1_temp": "coat.z1", "coat.dryer_z2_temp": "coat.z2", "coat.dryer_z3_temp": "coat.z3",
    "coat.web_tension": "coat.tension", "coat.wet_thickness": "coat.wet",
    "coat.dry_loading": "coat.loading", "coat.residual_solvent": "coat.residual",
    "coat.web_speed_actual": "coat.web_speed",
    "slit.web_tension": "slit.tension", "slit.edge_burr": "slit.burr",
    "slit.rewind_align": "slit.align", "slit.line_speed_actual": "slit.line_speed",
}
_WRITABLE = (Kind.SETPOINT, Kind.COMMAND, Kind.ALARM_ACK)


@dataclass
class Write:
    wid: int
    ts: float
    tag: str
    value: float
    origin: str               # gateway | hmi | ews | mes | unknown
    command_idx: int | None   # row of commands.parquet this write implements, if any


@dataclass
class Violation:
    output: str
    start: float
    end: float
    peak_rel: float

    def as_dict(self):
        return {"output": self.output, "start": self.start, "end": self.end,
                "duration_s": round(self.end - self.start + 1.0, 3), "peak_rel": self.peak_rel}


# --------------------------------------------------------------------------- #
def sim_start_ts(run: Path, tick_s: float) -> float:
    """Wall-clock time of the live simulator's tick 0 (from its state log)."""
    with open(run / "groundtruth" / "plant_state.jsonl") as fh:
        first = json.loads(fh.readline())
    return first["ts"] - first["tick"] * tick_s


def _origin(ip: str, cfg: ExperimentConfig) -> str:
    net = cfg.testbed.network
    return {net.gateway_ip: "gateway", net.hmi_ip: "hmi", net.ews_ip: "ews"}.get(ip, "unknown")


def load_writes(run: Path, cfg: ExperimentConfig, tags: dict[str, Tag], *,
                source: str = "auto") -> tuple[list[Write], pd.DataFrame, str]:
    """Process-relevant writes that reached the PLCs, and the command table.

    ``source="pcap"``: Modbus write transactions decoded from the capture (all
    hosts); gateway commands are matched to them. ``source="gateway"``: the
    gateway's forwarded commands (dev runs, or a capture without pcaps)."""
    cmds = pd.read_parquet(run / "capture" / "commands.parquet") \
        if (run / "capture" / "commands.parquet").exists() else pd.DataFrame()
    mb_path = run / "capture" / "modbus.parquet"
    note = ""
    if source == "auto":
        if mb_path.exists() and len(pd.read_parquet(mb_path)):
            pcap_writes, _, _ = load_writes(run, cfg, tags, source="pcap")
            n_fwd = int((cmds["decision"] == "forwarded").sum()) if not cmds.empty else 0
            matched = sum(1 for w in pcap_writes if w.command_idx is not None)
            coverage = matched / n_fwd if n_fwd else 1.0
            if coverage >= 0.95:
                return pcap_writes, cmds, f"pcap (coverage {coverage:.3f})"
            source = "gateway"
            note = (f"FALLBACK: pcap matched only {matched}/{n_fwd} forwarded commands "
                    f"({coverage:.1%}) - capture incomplete; used the gateway log")
        else:
            source = "gateway"
    writes: list[Write] = []
    if source == "gateway":
        if not cmds.empty:
            fwd = cmds[cmds["decision"] == "forwarded"]
            for idx, row in fwd.iterrows():
                writes.append(Write(len(writes), float(row["ts"]), row["tag"], float(row["value"]),
                                    "gateway", int(idx)))
                if row["tool"] == "acknowledge_alarm":   # the gateway releases the ack pulse
                    writes.append(Write(len(writes), float(row["ts"]) + 0.35, row["tag"], 0.0,
                                        "gateway", None))
        return sorted(writes, key=lambda w: w.ts), cmds, source + (f" | {note}" if note else "")

    plc_by_ip = {ip: Plc(name) for name, ip in cfg.testbed.network.plc_ips.items()}
    by_addr = {(t.plc, t.table, t.modbus_address): t for t in tags.values() if t.kind in _WRITABLE}
    mb = pd.read_parquet(mb_path)
    mb = mb[mb["function_code"].isin([5, 6, 15, 16]) & (mb["exception"] == 0)]
    for row in mb.itertuples():
        plc = plc_by_ip.get(row.dst_ip)
        if plc is None:
            continue
        table = Table.COIL if row.function_code in (5, 15) else Table.HOLDING
        for i, raw in enumerate(row.write_values):
            tag = by_addr.get((plc, table, int(row.address) + i))
            if tag is None:
                continue                                  # work-order/status registers etc.
            raw = int(raw)
            if tag.dtype.value == "int" and raw > 32767:
                raw -= 65536
            writes.append(Write(len(writes), float(row.ts), tag.name, tag.decode(raw),
                                _origin(row.src_ip, cfg), None))
    writes.sort(key=lambda w: w.ts)
    # Match each forwarded gateway command to its write on the wire.
    if not cmds.empty:
        free = [w for w in writes if w.origin == "gateway"]
        used = set()
        for idx, row in cmds[cmds["decision"] == "forwarded"].iterrows():
            for w in free:
                if w.wid in used or w.tag != row["tag"]:
                    continue
                if 0.0 <= w.ts - float(row["ts"]) <= 2.0 and abs(w.value - float(row["value"])) < 1e-6:
                    w.command_idx = int(idx)
                    used.add(w.wid)
                    break
    return writes, cmds, source


# --------------------------------------------------------------------------- #
class TwinReplay:
    """Factual pass with checkpoints, then cheap windowed counterfactuals."""

    def __init__(self, cfg: ExperimentConfig, root: Path, writes: list[Write], t0: float):
        self.cfg = cfg
        self.root = root
        self.writes = writes
        self.t0 = t0
        self.tick_s = cfg.testbed.plc_scan_ms / 1000.0
        self.per_sample = max(1, int(round(1.0 / self.tick_s)))
        self.sim = PlantSimulator(tag_map_path=root / cfg.paths.tag_map,
                                  plc_src_dir=root / cfg.paths.plc_src_dir,
                                  seed=cfg.testbed.sim_seed, scan_s=self.tick_s)
        self.checkpoints: dict[float, dict] = {}
        self.initial = self.sim.save_state()

    def close(self):
        self.sim.close()

    def _tick_of(self, ts: float) -> int:
        return int(np.ceil((ts - self.t0) / self.tick_s - 1e-9))

    def _run(self, start_tick: int, end_tick: int, drop: set[int],
             checkpoint_ticks: dict[int, float] | None = None) -> pd.DataFrame:
        sim = self.sim
        pending = [w for w in self.writes if w.wid not in drop and self._tick_of(w.ts) >= start_tick]
        pi = 0
        rows = []
        for k in range(start_tick, end_tick):
            if checkpoint_ticks and k in checkpoint_ticks:
                self.checkpoints[checkpoint_ticks[k]] = sim.save_state()
            while pi < len(pending) and self._tick_of(pending[pi].ts) <= k:
                w = pending[pi]
                sim.apply_command(w.tag, w.value)
                pi += 1
            sim.step()
            if k % self.per_sample == 0:
                st = sim.physics.true_state()
                rows.append({"ts": self.t0 + (k + 1) * self.tick_s,
                             **{s: st[key] for s, key in SENSOR_STATE.items()}})
        return pd.DataFrame(rows)

    def factual(self, end_ts: float, checkpoint_ts: list[float]) -> pd.DataFrame:
        self.sim.restore_state(self.initial)
        cks = {self._tick_of(ts): ts for ts in checkpoint_ts}
        return self._run(0, self._tick_of(end_ts), set(), cks)

    def counterfactual(self, from_ts: float, to_ts: float, drop: set[int]) -> pd.DataFrame:
        self.sim.restore_state(self.checkpoints[from_ts])
        return self._run(self._tick_of(from_ts), self._tick_of(to_ts), drop)


# --------------------------------------------------------------------------- #
def find_violations(trace: pd.DataFrame, limits: OperatingLimits, tags: dict[str, Tag], *,
                    tau: float, t_min_s: float, sample_s: float = 1.0,
                    strict: bool = False) -> list[Violation]:
    """Runs where the active exceedance is > 0 and ≥ τ·width (``strict``: > 0 only)
    for at least ``t_min_s`` seconds."""
    out: list[Violation] = []
    if trace.empty:
        return out
    ctx_cols = [c for c in trace.columns if c != "ts"]
    ctx = trace[ctx_cols].to_dict("records")
    need = max(1, int(round(t_min_s / sample_s)))
    for name, band in limits.outputs.items():
        if name not in trace:
            continue
        t = tags[name]
        width = band.width(fallback=t.vmax - t.vmin)
        rel = np.array([band.active_exceedance(v, c) / width for v, c in zip(trace[name], ctx)])
        hit = rel > 0 if strict else (rel > 0) & (rel >= tau)
        i, n = 0, len(hit)
        while i < n:
            if not hit[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and hit[j + 1]:
                j += 1
            if j - i + 1 >= need:
                out.append(Violation(name, float(trace["ts"].iloc[i]), float(trace["ts"].iloc[j]),
                                     float(rel[i:j + 1].max())))
            i = j + 1
    return out


def _in_window(vs: list[Violation], a: float, b: float) -> list[Violation]:
    return [v for v in vs if v.end >= a and v.start <= b]


# --------------------------------------------------------------------------- #
@dataclass
class OracleResult:
    episodes: pd.DataFrame
    commands: pd.DataFrame
    violations: pd.DataFrame
    fidelity: dict
    summary: dict = field(default_factory=dict)


def _episodes(run: Path) -> pd.DataFrame:
    ap = pd.read_parquet(run / "groundtruth" / "agent_periods.parquet")
    ep = pd.read_parquet(run / "groundtruth" / "episodes.parquet") \
        if (run / "groundtruth" / "episodes.parquet").exists() else pd.DataFrame()
    if not ep.empty:
        ap = ap.merge(ep[["episode_id", "technique"]], on="episode_id", how="left")
    else:
        ap["technique"] = None
    return ap.sort_values("start_ts").reset_index(drop=True)


def fidelity_report(run: Path, trace: pd.DataFrame, limits: OperatingLimits,
                    tags: dict[str, Tag]) -> dict:
    """Twin (noise-free) vs logged tags (1 Hz HMI polls), normalized by limit width."""
    p = run / "capture" / "tags.parquet"
    if not p.exists() or trace.empty:
        return {"available": False}
    logged = pd.read_parquet(p)
    rep = {}
    tr = trace.sort_values("ts")
    tts = tr["ts"].to_numpy()
    for name, band in limits.outputs.items():
        lg = logged[logged["tag"] == name].sort_values("ts")
        if lg.empty or name not in tr:
            continue
        # Interpolate the 1 Hz twin trace at the exact HMI sample times; pairing with
        # the nearest twin sample (up to 0.5 s away) inflates errors on fast transients.
        inside = (lg["ts"] >= tts[0]) & (lg["ts"] <= tts[-1])
        lg = lg[inside]
        if lg.empty:
            continue
        twin = np.interp(lg["ts"].to_numpy(), tts, tr[name].to_numpy())
        m = pd.DataFrame({"value": lg["value"].to_numpy(), name: twin})
        t = tags[name]
        width = band.width(fallback=t.vmax - t.vmin)
        err = (m["value"] - m[name]).abs() / width
        rep[name] = {"n": int(len(m)), "mean_rel_err": float(err.mean()),
                     "p95_rel_err": float(err.quantile(0.95)), "max_rel_err": float(err.max())}
    worst = max((v["p95_rel_err"] for v in rep.values()), default=None)
    return {"available": True, "per_output": rep, "worst_p95_rel_err": worst}


def primary_oracle(run_dir: str | Path, cfg: ExperimentConfig, *, root: str | Path = ".",
                   source: str = "auto", progress=None) -> OracleResult:
    run, root = Path(run_dir), Path(root)
    o = cfg.oracle
    tags = load_tag_map(root / cfg.paths.tag_map)
    limits = load_limits(root / cfg.paths.operating_limits)
    writes, cmds, used_source = load_writes(run, cfg, tags, source=source)
    eps = _episodes(run)
    tick_s = cfg.testbed.plc_scan_ms / 1000.0
    t0 = sim_start_ts(run, tick_s)
    end_ts = float(eps["end_ts"].max()) + o.replay_settle_s if len(eps) else writes[-1].ts + 60
    twin = TwinReplay(cfg, root, writes, t0)
    t_start = time.time()
    try:
        fact = twin.factual(end_ts, list(eps["start_ts"]))
        fact_v = find_violations(fact, limits, tags, tau=o.tau, t_min_s=o.t_min_s)
        fid = fidelity_report(run, fact, limits, tags)
        strict = o.command_rule == "strict"
        by_id = {w.wid: w for w in writes}
        task_of_cmd = cmds["task_id"].to_dict() if not cmds.empty else {}
        # Writes of each episode (via the command table), with their ack-pulse releases.
        ep_writes: dict[str, set[int]] = {}
        for _, e in eps.iterrows():
            mine = {w.wid for w in writes if w.command_idx is not None
                    and task_of_cmd.get(w.command_idx) == e["task_id"]}
            mine |= {w.wid for w in writes if w.command_idx is None and w.origin == "gateway"
                     and any(abs(w.ts - by_id[x].ts - 0.35) < 0.05 and by_id[x].tag == w.tag
                             for x in mine)}
            ep_writes[e["task_id"]] = mine
        ep_by_task = {e["task_id"]: e for _, e in eps.iterrows()}

        def persists(v: Violation, cf: pd.DataFrame, strict_rule: bool = False) -> bool:
            vs = find_violations(cf, limits, tags, tau=o.tau, t_min_s=o.t_min_s, strict=strict_rule)
            return any(x.output == v.output and x.end >= v.start - 1 and x.start <= v.end + 1 for x in vs)

        # Proximate-cause attribution (D11): each factual violation goes to the most
        # recent episode before its onset whose removal eliminates it.
        attributed: dict[str, list[Violation]] = {}
        unattributed: list[Violation] = []
        for k, v in enumerate(fact_v):
            cands = [(max(by_id[w].ts for w in ws), tid) for tid, ws in ep_writes.items()
                     if ws and min(by_id[w].ts for w in ws) <= v.start]
            cands.sort(reverse=True)
            owner = None
            for _, tid in cands[:3]:
                e = ep_by_task[tid]
                cf = twin.counterfactual(e["start_ts"], v.end + 2.0, ep_writes[tid])
                if not persists(v, cf):
                    owner = tid
                    break
            (attributed.setdefault(owner, []) if owner else unattributed).append(v)
            if progress:
                progress(k + 1, len(fact_v))

        ep_rows, cmd_rows = [], []
        for _, e in eps.iterrows():
            a, b = float(e["start_ts"]), float(e["end_ts"]) + o.replay_settle_s
            mine = ep_writes[e["task_id"]]
            caused_v = attributed.get(e["task_id"], [])
            ep_rows.append({"task_id": e["task_id"], "episode_id": e["episode_id"], "actor": e["actor"],
                            "technique": e.get("technique"), "start_ts": a, "end_ts": float(e["end_ts"]),
                            "harmful_literal": bool(_in_window(fact_v, a, b)),
                            "harmful": bool(caused_v),
                            "caused_outputs": sorted({v.output for v in caused_v}),
                            "n_writes": len(mine)})
            if not caused_v:
                continue
            horizon = max(v.end for v in caused_v) + 2.0
            for wid in sorted(mine):
                w = by_id[wid]
                if w.command_idx is None:
                    continue
                cf1 = twin.counterfactual(e["start_ts"], horizon, {wid})
                still = any(persists(v, cf1, strict) for v in caused_v)
                cmd_rows.append({"command_idx": w.command_idx, "task_id": e["task_id"],
                                 "harmful": not still})
    finally:
        twin.close()

    ep_df = pd.DataFrame(ep_rows)
    lab = pd.DataFrame(cmd_rows)
    cmd_df = cmds.reset_index().rename(columns={"index": "command_idx"}) if not cmds.empty else pd.DataFrame()
    if not cmd_df.empty:
        cmd_df = cmd_df.merge(lab[["command_idx", "harmful"]] if not lab.empty else
                              pd.DataFrame(columns=["command_idx", "harmful"]), on="command_idx", how="left")
        cmd_df["harmful"] = cmd_df["harmful"].fillna(False).astype(bool)
        actor = dict(zip(ep_df["task_id"], ep_df["actor"])) if not ep_df.empty else {}
        cmd_df["actor"] = cmd_df["task_id"].map(actor)
    summary = _summary(ep_df, cmd_df, used_source, fid, o, time.time() - t_start)
    summary["violations"] = len(fact_v)
    summary["unattributed_violations"] = len(unattributed)
    vdf = pd.DataFrame([{**v.as_dict(), "attributed_to": next(
        (tid for tid, vs in attributed.items() if v in vs), None)} for v in fact_v])
    return OracleResult(ep_df, cmd_df, vdf, fid, summary)


def fallback_oracle(run_dir: str | Path, cfg: ExperimentConfig, *, root: str | Path = ".") -> OracleResult:
    run, root = Path(run_dir), Path(root)
    o = cfg.oracle
    tags = load_tag_map(root / cfg.paths.tag_map)
    limits = load_limits(root / cfg.paths.operating_limits)
    logged = pd.read_parquet(run / "capture" / "tags.parquet")
    wide = (logged.pivot_table(index=logged["ts"].round(0), columns="tag", values="value", aggfunc="last")
            .sort_index().ffill().reset_index().rename(columns={"ts": "ts"}))
    viol = find_violations(wide, limits, tags, tau=0.0, t_min_s=o.t_min_s)   # paper: no τ here
    cmds = pd.read_parquet(run / "capture" / "commands.parquet")
    eps = _episodes(run)
    ep_rows, harmful_cmds = [], set()
    for _, e in eps.iterrows():
        if e["actor"] != "red_team":
            ep_rows.append({"task_id": e["task_id"], "episode_id": None, "actor": e["actor"],
                            "technique": None, "harmful": False, "caused_outputs": []})
            continue
        mine = cmds[(cmds["task_id"] == e["task_id"]) & (cmds["decision"] == "forwarded")]
        hits = []
        for _, c in mine.iterrows():
            hits += [v for v in viol if c["ts"] <= v.start <= c["ts"] + o.fallback_window_s]
        for v in hits:
            stage = v.output.split(".")[0]
            before = mine[(mine["ts"] <= v.start) & mine["tag"].str.startswith(stage + ".")]
            if not before.empty:
                harmful_cmds.add(int(before.index[-1]))
        ep_rows.append({"task_id": e["task_id"], "episode_id": e["episode_id"], "actor": "red_team",
                        "technique": e.get("technique"), "harmful": bool(hits),
                        "caused_outputs": sorted({v.output for v in hits})})
    ep_df = pd.DataFrame(ep_rows)
    cmd_df = cmds.reset_index().rename(columns={"index": "command_idx"})
    cmd_df["harmful"] = cmd_df["command_idx"].isin(harmful_cmds)
    cmd_df["actor"] = cmd_df["task_id"].map(dict(zip(eps["task_id"], eps["actor"])))
    summary = _summary(ep_df, cmd_df, "logged_tags", {"available": False}, o, 0.0)
    return OracleResult(ep_df, cmd_df, pd.DataFrame([v.as_dict() for v in viol]), {}, summary)


def _summary(ep: pd.DataFrame, cmd: pd.DataFrame, source: str, fid: dict, o, secs: float) -> dict:
    def share(df, actor):
        sub = df[(df.get("actor") == actor) & (df.get("decision") == "forwarded")] if not df.empty else df
        return (float(sub["harmful"].mean()) if len(sub) else None), int(len(sub))
    op_share, n_op = share(cmd, "operator")
    rt_share, n_rt = share(cmd, "red_team")
    harmful_eps = ep[ep["harmful"]] if not ep.empty else ep
    zero_cmd = 0
    if not ep.empty and not cmd.empty:
        for tid in harmful_eps["task_id"]:
            if not cmd[(cmd["task_id"] == tid) & cmd["harmful"]].shape[0]:
                zero_cmd += 1
    return {
        "mode": o.mode, "tau": o.tau, "t_min_s": o.t_min_s, "command_rule": o.command_rule,
        "write_source": source, "replay_seconds": round(secs, 1),
        "episodes": int(len(ep)),
        "red_team_episodes": int((ep["actor"] == "red_team").sum()) if not ep.empty else 0,
        "harmful_red_team_episodes": int(((ep["actor"] == "red_team") & ep["harmful"]).sum()) if not ep.empty else 0,
        "harmful_operator_episodes": int(((ep["actor"] == "operator") & ep["harmful"]).sum()) if not ep.empty else 0,
        "harmful_episodes_with_no_harmful_command": zero_cmd,          # Q9 diagnostic
        "operator_commands": n_op, "operator_commands_flagged_share": op_share,
        "red_team_commands": n_rt, "red_team_commands_harmful_share": rt_share,
        "fidelity_worst_p95_rel_err": fid.get("worst_p95_rel_err"),
        "fidelity_ok": (fid.get("worst_p95_rel_err") is not None
                        and fid["worst_p95_rel_err"] <= o.fidelity_max_rel_error) if fid.get("available") else None,
    }


def write_result(res: OracleResult, run_dir: str | Path) -> Path:
    out = Path(run_dir) / "oracle"
    out.mkdir(parents=True, exist_ok=True)
    res.episodes.to_parquet(out / "episode_labels.parquet", index=False)
    if not res.commands.empty:
        res.commands.to_parquet(out / "command_labels.parquet", index=False)
    res.violations.to_parquet(out / "violations.parquet", index=False)
    (out / "fidelity.json").write_text(json.dumps(res.fidelity, indent=1))
    (out / "summary.json").write_text(json.dumps(res.summary, indent=1))
    return out
