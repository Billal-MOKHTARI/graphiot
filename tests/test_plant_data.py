"""Plant data files load, validate against the tag map, and keep the
oracle/task-generator split that the paper requires."""

import pytest

from graphiot.plant.tags import Kind
from graphiot.plant.limits import load_limits
from graphiot.plant.recipes import load_recipes
from graphiot.plant.allow_list import load_allow_list


def test_limits_reference_sensors_only(root, tags):
    names = set(tags)
    sensors = {n for n, t in tags.items() if t.kind is Kind.SENSOR}
    lim = load_limits(root / "plant" / "operating_limits.yaml", tag_names=names)
    assert lim.tags() <= sensors, "operating limits must be on process outputs (sensors)"
    assert lim.outputs, "no limits defined"


def test_limit_band_exceedance(root, tags):
    lim = load_limits(root / "plant" / "operating_limits.yaml")
    b = lim.outputs["coat.dryer_z1_temp"]
    assert b.exceedance(b.high + 5.0) == pytest.approx(5.0)
    assert b.exceedance((b.low + b.high) / 2) == 0.0
    assert b.relative_exceedance(b.high + (b.high - b.low) * 0.1) == pytest.approx(0.1)


def test_recipes_valid_and_context_shift_source(root, tags):
    setp = {n for n, t in tags.items() if t.kind is Kind.SETPOINT}
    ranges = {n: (t.vmin, t.vmax) for n, t in tags.items()}
    book = load_recipes(root / "plant" / "recipes.yaml", setpoint_tags=setp, tag_ranges=ranges)
    assert book.products
    # The same setpoint takes different values across phases -> context-shift exists.
    for name, prod in book.products.items():
        ramp = prod.setpoints("coating", "ramp").get("coat.web_speed_sp")
        steady = prod.setpoints("coating", "steady").get("coat.web_speed_sp")
        assert ramp is not None and steady is not None and steady > ramp, name


def test_recipes_carry_no_limit_bands(root):
    """Keep the files independent: recipes must not embed operating-limit data.
    The loaded recipe book has only product setpoints and overrides — no band
    fields (low/high/basis) and no `outputs:` section."""
    import yaml
    data = yaml.safe_load((root / "plant" / "recipes.yaml").read_text())
    assert set(data) <= {"meta", "products", "overrides"}
    assert "outputs" not in data

    def walk(node):
        if isinstance(node, dict):
            assert not ({"low", "high", "basis"} & set(node)), "limit band leaked into recipes"
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(data["products"])


def test_allow_list(root, tags):
    names = set(tags)
    al = load_allow_list(root / "plant" / "allow_list.yaml", tag_names=names)
    # A write can pass the allow-list yet be context-shifted (speed adjust task).
    ok, _ = al.decide("coating_speed_adjust", "write_tag", "coat.web_speed_sp")
    assert ok
    # ...but the same task may not touch the dryer.
    bad, reason = al.decide("coating_speed_adjust", "write_tag", "coat.dryer_z1_temp_sp")
    assert not bad and "not in allow-list" in reason
    # monitoring is read-only.
    assert not al.decide("monitoring", "write_tag", "mix.speed_sp")[0]
    # unknown task type is denied.
    assert not al.decide("nope", "read_tag", "mix.viscosity")[0]
