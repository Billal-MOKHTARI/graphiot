"""Causal scoring rule (paper §IV-A): the score of a command at time t uses only
events strictly before t plus its own fields. The snapshot a command uses (last
T_k <= t) must not change when ANY event at or after T_k is added or altered —
tag samples, Modbus writes (the command's own effect), flows, tool calls, tasks."""

import numpy as np
import pandas as pd
import pytest

from graphiot.config.io import load_config
from graphiot.data.monitor_view import MonitorView
from graphiot.graph.builder import SnapshotBuilder, Topology, snapshot_index_for
from graphiot.plant.tags import catalog_by_name

T0 = 1_000.0
GW, HMI, COAT = "10.77.2.10", "10.77.2.20", "10.77.2.102"


def _topology_df():
    return pd.DataFrame([{"name": t.name, "plc": t.plc.value, "stage": t.stage.value, "kind": t.kind.value,
                          "table": t.table.value, "modbus_address": t.modbus_address,
                          "dtype": t.dtype.value, "unit": t.unit} for t in catalog_by_name().values()])


def _view(extra=None):
    rng = np.random.default_rng(0)
    ts = np.arange(T0, T0 + 120, 1.0)
    tags = pd.concat([
        pd.DataFrame({"ts": ts, "plc": "PLC_COAT", "tag": "coat.dryer_z1_temp", "value": 60 + rng.normal(0, 1, len(ts))}),
        pd.DataFrame({"ts": ts, "plc": "PLC_COAT", "tag": "coat.phase", "value": 1.0}),
        pd.DataFrame({"ts": ts, "plc": "PLC_COAT", "tag": "coat.time_in_phase", "value": ts - T0}),
        pd.DataFrame({"ts": ts, "plc": "PLC_COAT", "tag": "coat.web_speed_sp", "value": 1.0})])
    calls = pd.DataFrame({"ts": T0 + np.array([10.0, 25.0, 40.0]), "ts_response": T0 + np.array([10.1, 25.1, 40.1]),
                          "call_id": [1, 2, 3], "task_id": "T-1", "agent_id": "operator-1",
                          "tool": ["read_stage_status", "write_tag", "read_tag"], "tag": [None, "coat.web_speed_sp", "coat.dryer_z1_temp"],
                          "stage": ["coating", None, None], "decision": "allowed", "error": None, "n_modbus": 1, "fcs": [[3]] * 3})
    tasks = pd.DataFrame({"ts": [T0 + 5.0], "task_id": ["T-1"], "task_type": ["coating_speed_adjust"],
                          "target_stage": ["coating"], "requested_tags": [["coat.web_speed_sp", "coat.pump_flow_sp"]],
                          "agent_id": ["operator-1"], "work_order_id": [42]})
    tag = catalog_by_name()["coat.web_speed_sp"]
    modbus = pd.DataFrame({"ts": [T0 + 25.05, T0 + 30.0], "ts_response": [T0 + 25.06, T0 + 30.01],
                           "src_ip": [GW, HMI], "src_port": [40000, 41000], "dst_ip": [COAT, COAT], "dst_port": [502, 502],
                           "transaction_id": [1, 2], "unit_id": [1, 1], "function_code": [6, 3],
                           "address": [tag.modbus_address, tag.modbus_address], "count": [1, 1],
                           "write_values": [[60], []], "read_values": [[], [60]], "exception": [0, 0],
                           "request_bytes": [12, 12], "response_bytes": [12, 11]})
    flows = pd.DataFrame({"flow_start": [T0 + 20.0], "flow_end": [T0 + 25.0], "end_reason": ["active"],
                          "src_ip": [GW], "src_port": [40000], "dst_ip": [COAT], "dst_port": [502], "protocol": [6],
                          **{c: [100.0] for c in ("IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS",
                                                  "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT", "LONGEST_FLOW_PKT",
                                                  "SHORTEST_FLOW_PKT", "NUM_PKTS_UP_TO_128_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT")}})
    v = dict(tags=tags, gateway_log=calls, tasks=tasks, modbus=modbus, flows=flows,
             commands=pd.DataFrame(columns=["ts"]))
    if extra:
        for k, df in extra.items():
            v[k] = pd.concat([v[k], df], ignore_index=True)
    return MonitorView(topology=_topology_df(), **v)


def _future_events(T):
    """Every kind of event, all at or after T (the scored command's own write included)."""
    tag = catalog_by_name()["coat.web_speed_sp"]
    return {
        "tags": pd.DataFrame({"ts": [T, T + 1], "plc": "PLC_COAT", "tag": "coat.dryer_z1_temp", "value": [999.0, -999.0]}),
        "modbus": pd.DataFrame({"ts": [T], "ts_response": [T + .01], "src_ip": [GW], "src_port": [40001],
                                "dst_ip": [COAT], "dst_port": [502], "transaction_id": [9], "unit_id": [1],
                                "function_code": [6], "address": [tag.modbus_address], "count": [1],
                                "write_values": [[250]], "read_values": [[]], "exception": [0],
                                "request_bytes": [12], "response_bytes": [12]}),
        "flows": pd.DataFrame({"flow_start": [T - 3], "flow_end": [T + 0.5], "end_reason": ["idle"], "src_ip": ["10.77.2.99"],
                               "src_port": [1], "dst_ip": [COAT], "dst_port": [502], "protocol": [6],
                               **{c: [1e6] for c in ("IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS",
                                                     "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT", "LONGEST_FLOW_PKT",
                                                     "SHORTEST_FLOW_PKT", "NUM_PKTS_UP_TO_128_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT")}}),
        "gateway_log": pd.DataFrame({"ts": [T], "ts_response": [T + .1], "call_id": [99], "task_id": ["T-2"],
                                     "agent_id": ["operator-1"], "tool": ["write_tag"], "tag": ["coat.web_speed_sp"],
                                     "stage": [None], "decision": ["denied"], "error": ["denied"], "n_modbus": [0], "fcs": [[]]}),
        "tasks": pd.DataFrame({"ts": [T], "task_id": ["T-2"], "task_type": ["dryer_tuning"], "target_stage": ["coating"],
                               "requested_tags": [["coat.dryer_z1_temp_sp"]], "agent_id": ["operator-1"], "work_order_id": [43]}),
    }


def _same(a, b) -> bool:
    if a.nodes != b.nodes or a.x.keys() != b.x.keys() or a.edges.keys() != b.edges.keys():
        return False
    return (all(np.array_equal(a.x[k], b.x[k]) for k in a.x)
            and all(all(np.array_equal(p, q) for p, q in zip(a.edges[k], b.edges[k])) for k in a.edges))


@pytest.fixture
def net(root):
    return load_config(root / "configs" / "pilot.yaml").testbed.network


@pytest.mark.parametrize("k", [3, 6, 10])
def test_future_events_never_reach_past_snapshots(net, k):
    base = _view()
    b0 = SnapshotBuilder(base, Topology.from_config(base, net))
    times, snaps = b0.build_all(T0, T0 + 110)
    T = float(times[k])
    pert = _view(_future_events(T))
    b1 = SnapshotBuilder(pert, Topology.from_config(pert, net))
    _, snaps2 = b1.build_all(T0, T0 + 110)
    for j in range(k + 1):
        assert _same(snaps[j], snaps2[j]), f"event at/after T leaked into snapshot {j} (T_j={times[j]})"
    assert any(not _same(snaps[j], snaps2[j]) for j in range(k + 1, len(snaps))), "sanity: future must change later snapshots"


def test_command_uses_last_snapshot_at_or_before_t(net):
    base = _view()
    times, _ = SnapshotBuilder(base, Topology.from_config(base, net)).build_all(T0, T0 + 110)
    for t in (times[4], times[4] + 0.01, times[5] - 0.01):
        assert times[snapshot_index_for(times, t)] <= t
    assert snapshot_index_for(times, times[0] - 0.1) == -1
