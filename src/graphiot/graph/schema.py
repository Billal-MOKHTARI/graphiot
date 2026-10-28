"""Task–plant graph schema — exactly Table II of the paper.

Node types and features
    Task     task type (one-hot), target stage (one-hot), requested tag set (multi-hot); no value
    Agent    session age, tool-call rate
    Gateway  call rate, error rate
    Host     role (hmi, ews, gateway, plc, unknown — 'plc' added, Q11/D8), flow statistics
    PLC      request rate
    Tag      type (kind one-hot), current value, rolling mean and standard deviation
    Stage    phase (idle, ramp, steady, transition), time in phase

Edge types and features
    assigned  Task -> Agent        time since assignment
    invokes   Agent -> Gateway     tool identifier (one-hot)
    writes    Gateway|Host -> Tag  Modbus function code (one-hot), value, change from previous value
    reads     Gateway|Host -> Tag  Modbus function code (one-hot), value, change from previous value
    flow      Host -> Host         NetFlow-style features
    controls  PLC -> Tag           static
    belongs   Tag -> Stage         static

No feature carries a recipe value, an operating limit or a twin output (Table II
caption); every feature is computed from the monitor view.
"""

from __future__ import annotations

TASK_TYPES = ["mixing_operation", "coating_operation", "coating_speed_adjust", "dryer_tuning",
              "slitting_operation", "line_supervisor", "monitoring"]
STAGES = ["mixing", "coating", "slitting"]
PHASES = ["idle", "ramp", "steady", "transition"]
HOST_ROLES = ["hmi", "ews", "gateway", "plc", "unknown"]
TOOLS = ["read_tag", "write_tag", "read_work_order", "read_stage_status", "load_recipe",
         "acknowledge_alarm"]
TAG_KINDS = ["setpoint", "sensor", "output", "phase", "status", "command", "interlock", "alarm",
             "alarm_ack"]
FUNCTION_CODES = [1, 2, 3, 4, 5, 6, 15, 16]
FLOW_FEATURES = ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS",
                 "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT", "LONGEST_FLOW_PKT",
                 "SHORTEST_FLOW_PKT", "NUM_PKTS_UP_TO_128_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT"]
HOST_FLOW_STATS = ["flows_out", "flows_in", "bytes_out", "bytes_in", "pkts_out", "pkts_in"]

NODE_TYPES = ["task", "agent", "gateway", "host", "plc", "tag", "stage"]
EDGE_TYPES = [                                  # (src type, relation, dst type)
    ("task", "assigned", "agent"),
    ("agent", "invokes", "gateway"),
    ("gateway", "writes", "tag"), ("host", "writes", "tag"),
    ("gateway", "reads", "tag"), ("host", "reads", "tag"),
    ("host", "flow", "host"),
    ("plc", "controls", "tag"),
    ("tag", "belongs", "stage"),
]


def node_dims(n_writable_tags: int) -> dict[str, int]:
    return {
        "task": len(TASK_TYPES) + len(STAGES) + n_writable_tags,
        "agent": 2,
        "gateway": 2,
        "host": len(HOST_ROLES) + len(HOST_FLOW_STATS),
        "plc": 1,
        "tag": len(TAG_KINDS) + 3,
        "stage": len(PHASES) + 1,
    }


def edge_dims() -> dict[tuple[str, str, str], int]:
    rw = len(FUNCTION_CODES) + 2
    return {
        ("task", "assigned", "agent"): 1,
        ("agent", "invokes", "gateway"): len(TOOLS),
        ("gateway", "writes", "tag"): rw, ("host", "writes", "tag"): rw,
        ("gateway", "reads", "tag"): rw, ("host", "reads", "tag"): rw,
        ("host", "flow", "host"): len(FLOW_FEATURES),
        ("plc", "controls", "tag"): 1,
        ("tag", "belongs", "stage"): 1,
    }


def feature_names(writable_tags: list[str]) -> tuple[dict[str, list[str]], dict[tuple, list[str]]]:
    """Human-readable names of every node and edge feature column (GUI, explanations)."""
    nodes = {
        "task": [f"type={t}" for t in TASK_TYPES] + [f"stage={s}" for s in STAGES]
                + [f"requests {t}" for t in writable_tags],
        "agent": ["session age (s)", "tool-call rate (1/s)"],
        "gateway": ["call rate (1/s)", "error rate"],
        "host": [f"role={r}" for r in HOST_ROLES] + [f"log1p {s}" for s in HOST_FLOW_STATS],
        "plc": ["request rate (1/s)"],
        "tag": [f"kind={k}" for k in TAG_KINDS] + ["value", "rolling mean", "rolling std"],
        "stage": [f"phase={p}" for p in PHASES] + ["time in phase (s)"],
    }
    rw = [f"FC{fc}" for fc in FUNCTION_CODES] + ["value", "change"]
    edges = {
        ("task", "assigned", "agent"): ["time since assignment (s)"],
        ("agent", "invokes", "gateway"): [f"tool={t}" for t in TOOLS],
        ("gateway", "writes", "tag"): rw, ("host", "writes", "tag"): rw,
        ("gateway", "reads", "tag"): rw, ("host", "reads", "tag"): rw,
        ("host", "flow", "host"): [f"log1p {f}" for f in FLOW_FEATURES],
        ("plc", "controls", "tag"): ["static"],
        ("tag", "belongs", "stage"): ["static"],
    }
    return nodes, edges
