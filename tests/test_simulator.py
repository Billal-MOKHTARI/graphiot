"""Simulator determinism/replay and the context-shift physics that the whole
paper rests on, including the counterfactual (leave-one-out) property the
primary oracle uses."""

import pytest

from graphiot.sim.plant import PlantSimulator, ScheduledCommand as SC
from graphiot.plant.recipes import load_recipes


def _coat_run(root, mistimed: bool):
    book = load_recipes(root / "plant" / "recipes.yaml")
    prod = book.products["NMC622_std"]
    ramp = prod.setpoints("coating", "ramp")
    steady = prod.setpoints("coating", "steady")
    cmds = [SC(0.0, "coat.run_cmd", 1)]
    cmds += [SC(1.0, tag, v) for tag, v in ramp.items()]
    if mistimed:
        # paper's example: STEADY web speed issued during the RAMP warm-up
        cmds.append(SC(20.0, "coat.web_speed_sp", steady["coat.web_speed_sp"]))
    cmds += [SC(140.0, tag, v) for tag, v in steady.items()]
    return cmds


def _max_window(trace, name, t0, t1):
    return max(v for t, v in zip(trace["t"], trace[name]) if t0 <= t < t1)


def test_determinism(root):
    cmds = _coat_run(root, mistimed=False)
    il = {"coat.web_present": True}
    with PlantSimulator(seed=7) as a:
        ta = a.run_schedule(cmds, 60, interlocks=il)
    with PlantSimulator(seed=7) as b:
        tb = b.run_schedule(cmds, 60, interlocks=il)
    assert ta == tb


def test_seed_changes_noise(root):
    cmds = _coat_run(root, mistimed=False)
    il = {"coat.web_present": True}
    with PlantSimulator(seed=7) as a:
        ta = a.run_schedule(cmds, 60, interlocks=il)
    with PlantSimulator(seed=8) as b:
        tb = b.run_schedule(cmds, 60, interlocks=il)
    assert ta["coat.residual_solvent"] != tb["coat.residual_solvent"]


def test_correct_recipe_in_spec(root, tags):
    from graphiot.plant.limits import load_limits
    lim = load_limits(root / "plant" / "operating_limits.yaml")
    cmds = _coat_run(root, mistimed=False)
    with PlantSimulator(seed=1) as sim:
        tr = sim.run_schedule(cmds, 200, interlocks={"coat.web_present": True})
    # After warm-up, residual solvent stays within its (one-sided) limit.
    resid = _max_window(tr, "coat.residual_solvent", 150, 200)
    assert resid <= lim.outputs["coat.residual_solvent"].high


def test_context_shift_violates_then_counterfactual_restores(root, tags):
    """The mistimed steady-speed command during warm-up pushes residual solvent
    out of limits; removing just that command (replay) brings it back in. This
    is the leave-one-out signal the primary oracle relies on."""
    from graphiot.plant.limits import load_limits
    lim = load_limits(root / "plant" / "operating_limits.yaml")
    hi = lim.outputs["coat.residual_solvent"].high
    il = {"coat.web_present": True}

    with PlantSimulator(seed=1) as bad:
        tb = bad.run_schedule(_coat_run(root, mistimed=True), 90, interlocks=il)
    with PlantSimulator(seed=1) as good:
        tg = good.run_schedule(_coat_run(root, mistimed=False), 90, interlocks=il)

    bad_peak = _max_window(tb, "coat.residual_solvent", 25, 70)
    good_peak = _max_window(tg, "coat.residual_solvent", 25, 70)
    assert bad_peak > hi, f"expected violation, got {bad_peak:.2f} <= {hi}"
    assert good_peak <= hi, f"counterfactual should be in spec, got {good_peak:.2f}"
