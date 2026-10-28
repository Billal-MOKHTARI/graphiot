"""Recipe book loader — TASK GENERATOR ONLY.

Loads ``plant/recipes.yaml``. Imported only by the task generator; the oracle
and the monitor must not import it. Provides per-product, per-stage, per-phase
setpoints and the list of legitimate overrides.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .tags import Stage

_PHASES = {"idle", "ramp", "steady", "transition"}


class Override(BaseModel):
    name: str
    stage: Stage
    tag: str
    phase: str
    set: float
    rationale: str = ""


class Product(BaseModel):
    label: str
    work_orders: list[int]
    # stages[stage][phase][tag] = setpoint value
    stages: dict[str, dict[str, dict[str, float]]]

    def setpoints(self, stage: str, phase: str) -> dict[str, float]:
        return dict(self.stages.get(stage, {}).get(phase, {}))


class RecipeBook(BaseModel):
    meta: dict = Field(default_factory=dict)
    products: dict[str, Product]
    overrides: list[Override] = Field(default_factory=list)

    def product_for_work_order(self, wo: int) -> tuple[str, Product] | None:
        for name, p in self.products.items():
            if wo in p.work_orders:
                return name, p
        return None

    def all_setpoint_values(self) -> dict[str, list[float]]:
        """Every value each setpoint tag takes across all products/phases — used
        to sanity-check that recipe values fall inside the tag's declared range
        and to design the benign value distribution."""
        out: dict[str, list[float]] = {}
        for p in self.products.values():
            for phases in p.stages.values():
                for sp in phases.values():
                    for tag, v in sp.items():
                        out.setdefault(tag, []).append(v)
        return out


def load_recipes(
    path: str | Path,
    *,
    setpoint_tags: set[str] | None = None,
    tag_ranges: dict[str, tuple[float, float]] | None = None,
) -> RecipeBook:
    """Load and validate recipes. If ``setpoint_tags`` is given, every referenced
    tag must be a writable setpoint; if ``tag_ranges`` is given, every recipe
    value must be inside the tag's engineering range."""
    data = yaml.safe_load(Path(path).read_text())
    book = RecipeBook.model_validate(data)

    for pname, p in book.products.items():
        for stage, phases in p.stages.items():
            if stage not in {s.value for s in Stage}:
                raise ValueError(f"{pname}: unknown stage {stage!r}")
            for phase, sp in phases.items():
                if phase not in _PHASES:
                    raise ValueError(f"{pname}.{stage}: unknown phase {phase!r}")
                for tag, v in sp.items():
                    if setpoint_tags is not None and tag not in setpoint_tags:
                        raise ValueError(f"{pname}.{stage}.{phase}: {tag!r} is not a setpoint tag")
                    if tag_ranges is not None and tag in tag_ranges:
                        lo, hi = tag_ranges[tag]
                        if not (lo <= v <= hi):
                            raise ValueError(
                                f"{pname}.{stage}.{phase}: {tag}={v} outside range [{lo},{hi}]")
    for ov in book.overrides:
        if ov.phase not in _PHASES:
            raise ValueError(f"override {ov.name!r}: unknown phase {ov.phase!r}")
        if setpoint_tags is not None and ov.tag not in setpoint_tags:
            raise ValueError(f"override {ov.name!r}: {ov.tag!r} is not a setpoint tag")
    return book
