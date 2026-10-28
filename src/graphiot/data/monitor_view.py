"""The monitor's view of a run — the ONLY way detectors read data.

Paper §III-B: the monitor observes the gateway log, the Modbus transactions,
flow records of the cell network and PLC tag values sampled at 1 Hz, and knows
the static topology (which PLC controls which tag, which tag belongs to which
stage). It does NOT receive recipes, operating limits or process outputs
computed by the twin.

This loader opens a fixed whitelist of files under ``capture/`` and drops any
column that is not on that table's whitelist. Nothing under ``groundtruth/``,
``oracle/`` or ``plant/{recipes,operating_limits}.yaml`` is reachable from here;
``tests/test_no_leakage.py`` enforces it (including a file-access audit).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..plant.tags import load_tag_map

MONITOR_FILES = {
    "gateway_log": "capture/gateway_log.parquet",
    "commands": "capture/commands.parquet",
    "tasks": "capture/tasks.parquet",
    "tags": "capture/tags.parquet",
    "modbus": "capture/modbus.parquet",
    "flows": "capture/flows.parquet",
}

ALLOWED_COLUMNS = {
    "gateway_log": {"ts", "ts_response", "call_id", "task_id", "agent_id", "tool", "tag", "stage",
                    "decision", "error", "n_modbus", "fcs"},
    "commands": {"ts", "call_id", "task_id", "agent_id", "tool", "tag", "value", "raw",
                 "function_code", "decision"},
    "tasks": {"ts", "task_id", "task_type", "target_stage", "requested_tags", "agent_id",
              "work_order_id"},
    "tags": {"ts", "plc", "tag", "value"},
    "modbus": {"ts", "ts_response", "src_ip", "src_port", "dst_ip", "dst_port", "transaction_id",
               "unit_id", "function_code", "address", "count", "write_values", "read_values",
               "exception", "request_bytes", "response_bytes"},
    "flows": None,   # NetFlow-style counters; checked against FORBIDDEN_SUBSTRINGS only
}

# Column-name fragments that must never reach the monitor, whatever the table.
FORBIDDEN_SUBSTRINGS = ("limit", "recipe", "twin", "true:", "act:", "actor", "technique",
                        "episode", "red_team", "operator_", "llm", "prompt", "harm", "label",
                        "score", "alarm_flag", "text", "planned_phase", "kind")

# Static topology fields the monitor may know (no ranges, no limits).
TOPOLOGY_FIELDS = ("name", "plc", "stage", "kind", "table", "modbus_address", "dtype", "unit")


@dataclass
class MonitorView:
    gateway_log: pd.DataFrame
    commands: pd.DataFrame
    tasks: pd.DataFrame
    tags: pd.DataFrame
    modbus: pd.DataFrame
    flows: pd.DataFrame
    topology: pd.DataFrame           # static tag -> PLC/stage map


def _clean(name: str, df: pd.DataFrame) -> pd.DataFrame:
    allowed = ALLOWED_COLUMNS[name]
    keep = []
    for c in df.columns:
        lc = str(c).lower()
        if any(s in lc for s in FORBIDDEN_SUBSTRINGS):
            continue
        if allowed is not None and c not in allowed:
            continue
        keep.append(c)
    return df[keep]


def load_monitor_view(run_dir: str | Path, *, tag_map: str | Path = "plant/tag_map.csv") -> MonitorView:
    run = Path(run_dir)
    frames = {}
    for name, rel in MONITOR_FILES.items():
        p = run / rel
        frames[name] = _clean(name, pd.read_parquet(p)) if p.exists() else pd.DataFrame()
    tags = load_tag_map(tag_map)
    topo = pd.DataFrame([{f: (getattr(t, f).value if hasattr(getattr(t, f), "value") else getattr(t, f))
                          for f in TOPOLOGY_FIELDS} for t in tags.values()])
    return MonitorView(topology=topo, **frames)
