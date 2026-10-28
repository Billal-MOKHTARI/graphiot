"""Turn a run's raw logs and pcaps into the documented Parquet tables.

Monitor-visible tables (``capture/``), read only through
:mod:`graphiot.data.monitor_view`:

    gateway_log.parquet  one row per tool call (no result payloads)
    commands.parquet     one row per tag-write attempt (forwarded or denied)
    tasks.parquet        task type, target stage, requested tags (no values)
    tags.parquet         1 Hz tag values polled by the HMI (long format)
    modbus.parquet       Modbus/TCP transactions decoded from the pcaps
    flows.parquet        NetFlow-style flow records from the pcaps

Ground truth (``groundtruth/``), never read by the monitor:

    tasks_full, agent_periods, episodes, plant_state, llm_turns (+ llm_log.jsonl)

See docs/DATA_FORMAT.md for the column-level schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..capture.pcap import flows as decode_flows
from ..capture.pcap import modbus_transactions

COMMAND_COLUMNS = ["ts", "call_id", "task_id", "agent_id", "tool", "tag", "value", "raw",
                   "function_code", "decision"]
TASK_COLUMNS = ["ts", "task_id", "task_type", "target_stage", "requested_tags", "agent_id",
                "work_order_id"]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass                          # a torn last line from a killed writer
    return rows


def _write(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def build_monitor_tables(run_dir: str | Path, *, idle_timeout_s: float = 15.0,
                         active_timeout_s: float = 5.0) -> dict[str, Path]:
    run = Path(run_dir)
    cap = run / "capture"
    out: dict[str, Path] = {}

    gl = pd.DataFrame(_read_jsonl(cap / "gateway_log.jsonl"))
    out["gateway_log"] = _write(gl, cap / "gateway_log.parquet")

    cmds = pd.DataFrame(_read_jsonl(cap / "commands.jsonl"))
    if not cmds.empty:
        cmds = cmds[[c for c in COMMAND_COLUMNS if c in cmds]]
    out["commands"] = _write(cmds, cap / "commands.parquet")

    tasks = pd.DataFrame(_read_jsonl(cap / "tasks.jsonl"))
    if not tasks.empty:
        tasks = tasks[[c for c in TASK_COLUMNS if c in tasks]]
    out["tasks"] = _write(tasks, cap / "tasks.parquet")

    rows = []
    for rec in _read_jsonl(cap / "tags.jsonl"):
        for tag, val in rec["values"].items():
            rows.append((rec["ts"], rec["plc"], tag, float(val)))
    tags = pd.DataFrame(rows, columns=["ts", "plc", "tag", "value"])
    out["tags"] = _write(tags, cap / "tags.parquet")

    pcaps = sorted((cap / "pcap").glob("*.pcap")) if (cap / "pcap").exists() else []
    out["modbus"] = _write(pd.DataFrame(modbus_transactions(pcaps)), cap / "modbus.parquet")
    out["flows"] = _write(pd.DataFrame(decode_flows(pcaps, idle_timeout_s=idle_timeout_s,
                                                    active_timeout_s=active_timeout_s)),
                          cap / "flows.parquet")
    return out


def build_groundtruth_tables(run_dir: str | Path) -> dict[str, Path]:
    run = Path(run_dir)
    gt = run / "groundtruth"
    out: dict[str, Path] = {}
    for name in ("tasks_full", "agent_periods", "episodes"):
        out[name] = _write(pd.DataFrame(_read_jsonl(gt / f"{name}.jsonl")), gt / f"{name}.parquet")
    states = _read_jsonl(gt / "plant_state.jsonl")
    if states:
        ps = pd.DataFrame([{"ts": r["ts"], "tick": r["tick"], **r["state"],
                            **{f"act:{k}": v for k, v in r.get("actuators", {}).items()}}
                           for r in states])
        out["plant_state"] = _write(ps, gt / "plant_state.parquet")
    turns = [r for r in _read_jsonl(gt / "llm_log.jsonl") if r.get("event") in ("turn", "tool", "end")]
    if turns:
        df = pd.DataFrame(turns)
        for col in ("tool_calls", "usage", "meta", "args", "result"):
            if col in df:
                df[col] = df[col].map(lambda v: json.dumps(v, default=str) if v is not None else None)
        out["llm_turns"] = _write(df, gt / "llm_turns.parquet")
    return out
