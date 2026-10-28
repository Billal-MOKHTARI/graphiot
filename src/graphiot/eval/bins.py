"""Per-tag quantile bins learned on benign data (paper §III-D: bin(v), Q bins).

Fitted on the values written to each tag in the benign *training* commands; a
tag with fewer than ``2·Q`` distinct written values is fitted on its 1 Hz series
over the training period instead (Q19). Interior edges sit at the j/Q
quantiles; duplicate edges collapse (a tag with few distinct values gets fewer
bins); the outer bins are open-ended, as quantile bins are. The benign
[min, max] per tag is kept for the out-of-range diagnostic of Q8.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class QuantileBins:
    q: int = 8
    edges: dict[str, list[float]] = field(default_factory=dict)
    benign_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    source: dict[str, str] = field(default_factory=dict)

    def fit(self, writes: pd.DataFrame, tag_series: pd.DataFrame | None = None) -> "QuantileBins":
        """``writes``: columns tag, value (benign training commands).
        ``tag_series``: columns tag, value (1 Hz samples, training period)."""
        qs = [j / self.q for j in range(1, self.q)]
        tags = set(writes["tag"]) | (set(tag_series["tag"]) if tag_series is not None else set())
        for tag in sorted(tags):
            w = writes.loc[writes["tag"] == tag, "value"].to_numpy(dtype=float)
            vals, src = w, "writes"
            if len(np.unique(w)) < 2 * self.q and tag_series is not None:
                s = tag_series.loc[tag_series["tag"] == tag, "value"].to_numpy(dtype=float)
                if len(s):
                    vals, src = np.concatenate([w, s]), "writes+series"
            if not len(vals):
                continue
            e = np.unique(np.quantile(vals, qs)) if len(np.unique(vals)) > 1 else np.array([])
            self.edges[tag] = [float(x) for x in e]
            self.benign_range[tag] = (float(vals.min()), float(vals.max()))
            self.source[tag] = src
        return self

    def bin(self, tag: str, value: float) -> int:
        e = self.edges.get(tag)
        if e is None:
            return -1                                   # tag never seen in benign data
        return int(np.searchsorted(np.asarray(e), value, side="right"))

    def n_bins(self, tag: str) -> int:
        return len(self.edges.get(tag, [])) + 1

    def in_benign_range(self, tag: str, value: float) -> bool:
        lo_hi = self.benign_range.get(tag)
        return lo_hi is not None and lo_hi[0] <= value <= lo_hi[1]

    def to_json(self) -> str:
        return json.dumps({"q": self.q, "edges": self.edges, "benign_range": self.benign_range,
                           "source": self.source})

    @classmethod
    def from_json(cls, s: str) -> "QuantileBins":
        d = json.loads(s)
        return cls(q=d["q"], edges=d["edges"],
                   benign_range={k: tuple(v) for k, v in d["benign_range"].items()},
                   source=d.get("source", {}))
