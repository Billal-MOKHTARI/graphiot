"""The physical premise of the paper, checked for every product and stage:
a correct recipe run stays inside the operating limits, and the same recipe with
ONE steady-phase setpoint issued during the ramp phase (a context-shifted
command: its value is benign elsewhere) violates them under the oracle's own
criterion (τ, T_min)."""

import pandas as pd
import pytest

from graphiot.oracle.core import SENSOR_STATE, find_violations
from graphiot.plant.limits import load_limits
from graphiot.plant.recipes import load_recipes
from graphiot.sim.plant import PlantSimulator, ScheduledCommand as SC

PREFIX = {"mixing": "mix", "coating": "coat", "slitting": "slit"}
KEY_TAG = {"mixing": "mix.speed_sp", "coating": "coat.web_speed_sp", "slitting": "slit.line_speed_sp"}


def _run(root, stage, product, shifted: bool):
    book = load_recipes(root / "plant" / "recipes.yaml")
    prod = book.products[product]
    p = PREFIX[stage]
    cmds = [SC(0.0, f"{p}.run_cmd", 1)]
    cmds += [SC(1.0, t, v) for t, v in prod.setpoints(stage, "ramp").items()]
    if stage == "slitting":
        cmds.append(SC(40.0, "slit.knife_engage_cmd", 1))
    if shifted:
        cmds.append(SC(20.0, KEY_TAG[stage], prod.setpoints(stage, "steady")[KEY_TAG[stage]]))
    cmds += [SC(150.0, t, v) for t, v in prod.setpoints(stage, "steady").items()]
    with PlantSimulator(tag_map_path=root / "plant/tag_map.csv", plc_src_dir=root / "plc/src",
                        seed=3) as sim:
        tr = sim.run_schedule(cmds, 260.0, include_true_state=True)
    return pd.DataFrame({"ts": tr["t"], **{s: tr["true:" + k] for s, k in SENSOR_STATE.items()}})


@pytest.mark.parametrize("product", ["NMC622_std", "LFP_thin"])
@pytest.mark.parametrize("stage", ["mixing", "coating", "slitting"])
def test_correct_in_spec_and_context_shift_violates(root, tags, stage, product):
    lim = load_limits(root / "plant" / "operating_limits.yaml")
    good = find_violations(_run(root, stage, product, False), lim, tags, tau=0.05, t_min_s=10)
    bad = find_violations(_run(root, stage, product, True), lim, tags, tau=0.05, t_min_s=10)
    assert not good, f"correct {product}/{stage} run violates: {[v.output for v in good]}"
    assert bad, f"context-shifted {KEY_TAG[stage]} during ramp did not violate"
