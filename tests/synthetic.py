"""Small synthetic monitor view + ground truth for model tests (no testbed needed).

40 minutes of a coating cell: 1 Hz tag samples, a task every 60 s with 2-3 writes
(gateway call, command, Modbus write), HMI reads, flows, and one red-team period
in the middle. Values depend on the task type and the coating phase, so a model
that uses the context can beat the marginal distribution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from graphiot.data.monitor_view import MonitorView
from graphiot.plant.tags import catalog_by_name

T0 = 1_000_000.0
GW, HMI, COAT, MIX = "10.77.2.10", "10.77.2.20", "10.77.2.102", "10.77.2.101"
FLOW_COLS = ("IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS",
             "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT", "LONGEST_FLOW_PKT",
             "SHORTEST_FLOW_PKT", "NUM_PKTS_UP_TO_128_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT")


def topology_df() -> pd.DataFrame:
    return pd.DataFrame([{"name": t.name, "plc": t.plc.value, "stage": t.stage.value, "kind": t.kind.value,
                          "table": t.table.value, "modbus_address": t.modbus_address,
                          "dtype": t.dtype.value, "unit": t.unit} for t in catalog_by_name().values()])


def make_view(duration_s: float = 2400.0, seed: int = 0) -> tuple[MonitorView, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    cat = catalog_by_name()
    ts = np.arange(T0, T0 + duration_s, 1.0)
    phase = ((ts - T0) // 300 % 3).astype(float)            # idle/ramp/steady cycling every 5 min
    speed = np.where(phase == 2, 2.0, 0.5)
    tags = pd.concat([
        pd.DataFrame({"ts": ts, "plc": "PLC_COAT", "tag": "coat.phase", "value": phase}),
        pd.DataFrame({"ts": ts, "plc": "PLC_COAT", "tag": "coat.time_in_phase", "value": (ts - T0) % 300}),
        pd.DataFrame({"ts": ts, "plc": "PLC_COAT", "tag": "coat.web_speed_sp", "value": speed}),
        pd.DataFrame({"ts": ts, "plc": "PLC_COAT", "tag": "coat.dryer_z1_temp", "value": 60 + rng.normal(0, 1, len(ts))}),
        pd.DataFrame({"ts": ts, "plc": "PLC_MIX", "tag": "mix.phase", "value": 0.0}),
    ], ignore_index=True)
    calls, cmds, tasks, mb, flows = [], [], [], [], []
    call_id = 0
    for i, t in enumerate(np.arange(T0 + 20, T0 + duration_s - 30, 60.0)):
        ttype = ["coating_speed_adjust", "dryer_tuning", "coating_operation"][i % 3]
        tid = f"T-{i:04d}"
        tasks.append({"ts": t, "task_id": tid, "task_type": ttype, "target_stage": "coating",
                      "requested_tags": ["coat.web_speed_sp"], "agent_id": "operator-1", "work_order_id": 42})
        ph = phase[int(t - T0)]
        for j in range(2 + i % 2):
            tc = t + 8 + 9 * j
            if ttype == "dryer_tuning":
                tag, val = "coat.dryer_z1_temp_sp", float(60 + 10 * ph + rng.integers(0, 3))
            else:
                tag, val = "coat.web_speed_sp", float(0.5 + 0.75 * ph + 0.1 * rng.integers(0, 3))
            tg = cat[tag]
            raw = int(round(val * tg.scale))
            call_id += 1
            calls.append({"ts": tc, "ts_response": tc + 0.05, "call_id": call_id, "task_id": tid,
                          "agent_id": "operator-1", "tool": "write_tag", "tag": tag, "stage": None,
                          "decision": "allowed", "error": None, "n_modbus": 1, "fcs": [6]})
            cmds.append({"ts": tc + 0.01, "call_id": call_id, "task_id": tid, "agent_id": "operator-1",
                         "tool": "write_tag", "tag": tag, "value": val, "raw": raw, "function_code": 6,
                         "decision": "forwarded"})
            mb.append({"ts": tc + 0.01, "ts_response": tc + 0.02, "src_ip": GW, "src_port": 40000, "dst_ip": COAT,
                       "dst_port": 502, "transaction_id": call_id, "unit_id": 1, "function_code": 6,
                       "address": tg.modbus_address, "count": 1, "write_values": [raw], "read_values": [],
                       "exception": 0, "request_bytes": 12, "response_bytes": 12})
            flows.append({"flow_start": tc, "flow_end": tc + 2.0, "end_reason": "idle", "src_ip": GW,
                          "src_port": 40000, "dst_ip": COAT, "dst_port": 502, "protocol": 6,
                          **{c: float(rng.integers(50, 200)) for c in FLOW_COLS}})
    for t in np.arange(T0 + 0.5, T0 + duration_s, 5.0):        # HMI polls, aggregated as reads
        tg = cat["coat.dryer_z1_temp"]
        mb.append({"ts": t, "ts_response": t + 0.01, "src_ip": HMI, "src_port": 41000, "dst_ip": COAT,
                   "dst_port": 502, "transaction_id": 0, "unit_id": 1, "function_code": 4,
                   "address": tg.modbus_address, "count": 1, "write_values": [], "read_values": [600],
                   "exception": 0, "request_bytes": 12, "response_bytes": 11})
    view = MonitorView(topology=topology_df(), tags=tags, gateway_log=pd.DataFrame(calls),
                       commands=pd.DataFrame(cmds), tasks=pd.DataFrame(tasks),
                       modbus=pd.DataFrame(mb).sort_values("ts").reset_index(drop=True),
                       flows=pd.DataFrame(flows))
    mid = T0 + duration_s / 2
    periods = pd.DataFrame({"start_ts": [T0, mid, mid + 120], "end_ts": [mid, mid + 120, T0 + duration_s],
                            "actor": ["operator", "red_team", "operator"]})
    return view, periods


def write_run(run_dir: Path, periods: pd.DataFrame) -> Path:
    (run_dir / "groundtruth").mkdir(parents=True, exist_ok=True)
    periods.to_parquet(run_dir / "groundtruth" / "agent_periods.parquet", index=False)
    return run_dir
