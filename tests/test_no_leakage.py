"""The monitor never receives recipes, operating limits or twin outputs
(paper §III-B, Table II caption). Four checks on the only loader detectors use:
static source scan, runtime file-access audit, column pollution, and a
metamorphic test (rewrite every forbidden source -> identical monitor inputs)."""

import builtins
import inspect
import shutil

import pandas as pd
import pytest

import graphiot.data.monitor_view as mv_mod
from graphiot.data.monitor_view import load_monitor_view

FORBIDDEN_PATH_PARTS = ("groundtruth", "oracle", "recipes.yaml", "operating_limits.yaml",
                        "plant_state", "llm_log")


def _fake_run(tmp_path):
    run = tmp_path / "run"
    (run / "capture").mkdir(parents=True)
    (run / "groundtruth").mkdir()
    (run / "oracle").mkdir()
    pd.DataFrame({"ts": [1.0, 2.0], "call_id": [1, 2], "task_id": ["T-1", "T-1"], "agent_id": ["a", "a"],
                  "tool": ["write_tag"] * 2, "tag": ["coat.web_speed_sp"] * 2, "value": [12.0, 1.0],
                  "raw": [120, 10], "function_code": [6, 6], "decision": ["forwarded"] * 2,
                  "technique": ["T0836", None], "score": [9.0, 1.0]}          # forbidden columns
                 ).to_parquet(run / "capture" / "commands.parquet")
    pd.DataFrame({"ts": [0.5], "task_id": ["T-1"], "task_type": ["coating_operation"],
                  "target_stage": ["coating"], "requested_tags": [["coat.web_speed_sp"]],
                  "agent_id": ["a"], "work_order_id": [42], "text": ["set speed to 12"]}
                 ).to_parquet(run / "capture" / "tasks.parquet")
    pd.DataFrame({"ts": [1.0], "plc": ["PLC_COAT"], "tag": ["coat.dryer_z1_temp"], "value": [70.0]}
                 ).to_parquet(run / "capture" / "tags.parquet")
    pd.DataFrame({"ts": [1.0], "true:coat.residual": [3.3], "episode": ["E-1"]}
                 ).to_parquet(run / "groundtruth" / "plant_state.parquet")
    (run / "oracle" / "summary.json").write_text("{}")
    return run


def test_loader_imports_and_whitelist():
    """AST check: the loader imports no recipe/limit loader and every file it may
    open lives under capture/."""
    import ast
    tree = ast.parse(inspect.getsource(mv_mod))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported |= {f"{node.module}.{a.name}" for a in node.names}
        elif isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
    for bad in ("limits", "recipes", "oracle", "sim.", "campaign", "agents"):
        assert not any(bad in name for name in imported), f"monitor_view imports {bad}: {imported}"
    assert all(path.startswith("capture/") for path in mv_mod.MONITOR_FILES.values())


def test_runtime_file_access_audit(tmp_path, root, monkeypatch):
    run = _fake_run(tmp_path)
    opened = []
    real_open = builtins.open
    real_rp = pd.read_parquet
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (opened.append(str(f)), real_open(f, *a, **k))[1])
    monkeypatch.setattr(pd, "read_parquet", lambda f, *a, **k: (opened.append(str(f)), real_rp(f, *a, **k))[1])
    load_monitor_view(run, tag_map=root / "plant" / "tag_map.csv")
    assert opened, "audit saw no file access"
    for path in opened:
        assert not any(part in path for part in FORBIDDEN_PATH_PARTS), f"monitor opened {path}"


def test_forbidden_columns_dropped(tmp_path, root):
    v = load_monitor_view(_fake_run(tmp_path), tag_map=root / "plant" / "tag_map.csv")
    assert "technique" not in v.commands and "score" not in v.commands
    assert "text" not in v.tasks
    assert {"tag", "value", "tool", "decision"} <= set(v.commands.columns)
    assert not {"vmin", "vmax"} & set(v.topology.columns), "register ranges are not topology"


def test_metamorphic_forbidden_sources_do_not_matter(tmp_path, root):
    run = _fake_run(tmp_path)
    a = load_monitor_view(run, tag_map=root / "plant" / "tag_map.csv")
    # Rewrite every forbidden source: ground truth, oracle output, and copies of the
    # plant's recipe book / limits next to the run.
    pd.DataFrame({"ts": [1.0], "true:coat.residual": [99.0]}).to_parquet(run / "groundtruth" / "plant_state.parquet")
    (run / "oracle" / "summary.json").write_text('{"harmful": 1}')
    plant = tmp_path / "plant"
    shutil.copytree(root / "plant", plant)
    (plant / "recipes.yaml").write_text("products: {}\n")
    (plant / "operating_limits.yaml").write_text("outputs: {}\n")
    b = load_monitor_view(run, tag_map=plant / "tag_map.csv")
    for name in ("commands", "tasks", "tags", "gateway_log", "modbus", "flows", "topology"):
        pd.testing.assert_frame_equal(getattr(a, name), getattr(b, name), check_like=True)
