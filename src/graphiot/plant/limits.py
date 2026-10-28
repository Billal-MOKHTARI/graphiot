"""Operating limits loader — ORACLE ONLY.

Loads ``plant/operating_limits.yaml`` into validated models. Imported only by
the oracle; importing it from the monitor/graph/detector code is what the
no-leak test forbids. Keeps τ and T_min out of this module — those live in the
experiment YAML and are applied by the oracle over time; this module only says
how far a value is outside its band and how wide the band is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class Condition(BaseModel):
    """A bound applies only while ``tag`` satisfies ``op value`` (e.g. a dryer's
    low-temperature bound only while the web is moving)."""
    tag: str
    op: Literal["gt", "lt", "ge", "le"]
    value: float

    def holds(self, values: dict[str, float]) -> bool:
        v = values.get(self.tag)
        if v is None:
            return False
        return {"gt": v > self.value, "lt": v < self.value,
                "ge": v >= self.value, "le": v <= self.value}[self.op]


class LimitBand(BaseModel):
    low: float | None = None
    high: float | None = None
    unit: str
    basis: Literal["equipment", "quality"]
    low_when: Condition | None = None     # None: the low bound always applies
    high_when: Condition | None = None    # None: the high bound always applies

    def active_exceedance(self, value: float, context: dict[str, float]) -> float:
        """Like :meth:`exceedance`, but only for the bounds whose condition holds."""
        if self.high is not None and value > self.high and \
                (self.high_when is None or self.high_when.holds(context)):
            return value - self.high
        if self.low is not None and value < self.low and \
                (self.low_when is None or self.low_when.holds(context)):
            return self.low - value
        return 0.0

    @model_validator(mode="after")
    def _one_bound(self):
        if self.low is None and self.high is None:
            raise ValueError("limit band needs at least one of low/high")
        if self.low is not None and self.high is not None and self.low >= self.high:
            raise ValueError("low must be < high")
        return self

    def exceedance(self, value: float) -> float:
        """Engineering amount by which ``value`` is outside the band (0 if inside)."""
        if self.high is not None and value > self.high:
            return value - self.high
        if self.low is not None and value < self.low:
            return self.low - value
        return 0.0

    def width(self, fallback: float | None = None) -> float:
        """Scale used to normalise τ ("relative to the limit width", §V-B).
        Two-sided: high − low. One-sided: the bound's magnitude (distance from the
        physical zero, e.g. 3 % residual solvent, 15 µm burr) — the register range
        would make τ far too lax. ``fallback`` is used only for a zero bound."""
        if self.low is not None and self.high is not None:
            return self.high - self.low
        bound = self.high if self.high is not None else self.low
        if bound:
            return abs(bound)
        return fallback if fallback and fallback > 0 else 1.0

    def relative_exceedance(self, value: float, fallback_width: float | None = None) -> float:
        w = self.width(fallback_width)
        return self.exceedance(value) / w if w > 0 else 0.0


class OperatingLimits(BaseModel):
    meta: dict = Field(default_factory=dict)
    outputs: dict[str, LimitBand]

    def tags(self) -> set[str]:
        return set(self.outputs)


def load_limits(path: str | Path, *, tag_names: set[str] | None = None) -> OperatingLimits:
    """Load and validate the limits file. If ``tag_names`` is given, every output
    must be a known tag (defensive: catches typos against the tag map)."""
    data = yaml.safe_load(Path(path).read_text())
    lim = OperatingLimits.model_validate(data)
    if tag_names is not None:
        referenced = lim.tags() | {c.tag for b in lim.outputs.values()
                                   for c in (b.low_when, b.high_when) if c is not None}
        unknown = referenced - tag_names
        if unknown:
            raise ValueError(f"operating_limits references unknown tags: {sorted(unknown)}")
    return lim
