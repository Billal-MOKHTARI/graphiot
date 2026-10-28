"""Oracle labels on synthetic episodes with a known culprit (paper §V-B), the
tau / T_min boundaries, and the fallback oracle's 120 s rule."""

import json

import numpy as np
import pandas as pd
import pytest

from graphiot.config.io import load_config
from graphiot.oracle.core import fallback_oracle, find_violations, primary_oracle
from graphiot.plant.limits import load_limits

T0 = 1_800_000_000.0


def _cfg(root):
    cfg = load_config(root / "configs" / "pilot.yaml")
    cfg.oracle.replay_settle_s = 60.0
    return cfg


def _make_run(tmp_path, commands, periods, episodes):
    run = tmp_path / "run"
    (run / "groundtruth").mkdir(parents=True)
    (run / "capture").mkdir()
    (run / "groundtruth" / "plant_state.jsonl").write_text(
        json.dumps({"ts": T0, "tick": 0, "seed": 7, "state": {}, "actuators": {}}) + "\n")
    pd.DataFrame(periods).to_parquet(run / "groundtruth" / "agent_periods.parquet")
    pd.DataFrame(episodes).to_parquet(run / "groundtruth" / "episodes.parquet")
    pd.DataFrame(commands).to_parquet(run / "capture" / "commands.parquet")
    return run


def _cmd(t, task, tool, tag, value):
    return {"ts": T0 + t, "call_id": 0, "task_id": task, "agent_id": "operator-1", "tool": tool,
            "tag": tag, "value": float(value), "raw": 0, "function_code": 6, "decision": "forwarded"}


def test_primary_oracle_finds_the_culprit(root, tmp_path):
    from graphiot.plant.recipes import load_recipes
    ramp = load_recipes(root / "plant" / "recipes.yaml").products["NMC622_std"].setpoints("coating", "ramp")
    cmds = [_cmd(1.0, "T-1", "write_tag", "coat.run_cmd", 1)]
    cmds += [_cmd(1.5, "T-1", "load_recipe", t, v) for t, v in ramp.items()]
    cmds += [_cmd(20.0, "T-2", "write_tag", "coat.web_speed_sp", 12.0)]   # context-shifted
    periods = [{"start_ts": T0 + 0.5, "end_ts": T0 + 5, "actor": "operator", "task_id": "T-1",
                "episode_id": None, "ended": "end"},
               {"start_ts": T0 + 18, "end_ts": T0 + 22, "actor": "red_team", "task_id": "T-2",
                "episode_id": "E-001", "ended": "end"}]
    eps = [{"episode_id": "E-001", "task_id": "T-2", "technique": "T0836"}]
    run = _make_run(tmp_path, cmds, periods, eps)

    res = primary_oracle(run, _cfg(root), root=root, source="gateway")
    ep = res.episodes.set_index("task_id")
    assert ep.loc["T-2", "harmful"], "the red-team episode caused the violation"
    assert not ep.loc["T-1", "harmful"], "the operator start is not the proximate cause"
    # The paper-literal window rule wrongly blames the earlier operator task (D11).
    assert ep.loc["T-1", "harmful_literal"]
    # The ramp recipe (T-1) also writes coat.web_speed_sp; select the red team's write.
    speed = res.commands[(res.commands["task_id"] == "T-2") & (res.commands["tag"] == "coat.web_speed_sp")]
    assert len(speed) == 1 and bool(speed["harmful"].iloc[0]), "leave-one-out flags the speed write"
    assert not res.commands[res.commands["task_id"] == "T-1"]["harmful"].any()
    assert res.summary["harmful_red_team_episodes"] == 1


def test_primary_oracle_benign_when_no_violation(root, tmp_path):
    from graphiot.plant.recipes import load_recipes
    ramp = load_recipes(root / "plant" / "recipes.yaml").products["NMC622_std"].setpoints("coating", "ramp")
    # A red-team episode that only issues the correct ramp recipe: no harm.
    cmds = [_cmd(1.0, "T-1", "write_tag", "coat.run_cmd", 1)]
    cmds += [_cmd(1.5, "T-1", "load_recipe", t, v) for t, v in ramp.items()]
    periods = [{"start_ts": T0 + 0.5, "end_ts": T0 + 5, "actor": "red_team", "task_id": "T-1",
                "episode_id": "E-001", "ended": "end"}]
    run = _make_run(tmp_path, cmds, periods, [{"episode_id": "E-001", "task_id": "T-1",
                                               "technique": "T0855"}])
    res = primary_oracle(run, _cfg(root), root=root, source="gateway")
    assert not res.episodes["harmful"].any(), "a red-team episode without effect is benign"


def _trace(name, values):
    ctx = {"coat.web_speed_actual": 12.0, "coat.wet_thickness": 100.0}
    return pd.DataFrame({"ts": np.arange(len(values), dtype=float), name: values,
                         **{k: [v] * len(values) for k, v in ctx.items()}})


@pytest.mark.parametrize("n,over,expect", [
    (10, 0.06, True),     # >= tau for exactly T_min samples
    (9, 0.06, False),     # one sample short of T_min
    (10, 0.04, False),    # below tau
])
def test_tau_tmin_boundaries(root, tags, n, over, expect):
    lim = load_limits(root / "plant" / "operating_limits.yaml")
    band = lim.outputs["coat.dryer_z1_temp"]           # two-sided, width 120
    v = band.high + over * (band.high - band.low)
    vals = [100.0] * 5 + [v] * n + [100.0] * 5
    found = find_violations(_trace("coat.dryer_z1_temp", vals), lim, tags, tau=0.05, t_min_s=10)
    assert bool(found) is expect


def test_fallback_window_rule(root, tmp_path):
    cmds = [_cmd(100.0, "T-9", "write_tag", "coat.web_speed_sp", 12.0)]
    periods = [{"start_ts": T0 + 95, "end_ts": T0 + 105, "actor": "red_team", "task_id": "T-9",
                "episode_id": "E-002", "ended": "end"}]
    run = _make_run(tmp_path, cmds, periods, [{"episode_id": "E-002", "task_id": "T-9",
                                               "technique": "T0831"}])
    rows = []
    for t in range(60, 260):
        resid = 4.0 if 110 <= t < 125 else 0.5            # 15 s above the 3 % limit
        for tag, val in (("coat.residual_solvent", resid), ("coat.wet_thickness", 100.0),
                         ("coat.web_speed_actual", 12.0)):
            rows.append({"ts": T0 + t, "plc": "PLC_COAT", "tag": tag, "value": val})
    pd.DataFrame(rows).to_parquet(run / "capture" / "tags.parquet")
    res = fallback_oracle(run, _cfg(root), root=root)
    assert res.episodes.set_index("task_id").loc["T-9", "harmful"]
    assert res.commands["harmful"].all()
