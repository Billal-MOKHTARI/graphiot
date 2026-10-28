"""B0 — phase-aware value table (paper §V-E).

For each (tag, stage phase): the value bins seen in benign training and their
frequencies. Score of a command = −log(frequency of its bin) when seen, else the
unseen score. The paper writes "score 1 if the bin is unseen", which ranks an
unseen bin *below* a seen bin of frequency < e⁻¹ (Q7); the default ``"max"``
scores unseen bins +∞, ``"literal"`` reproduces the text.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import pandas as pd

from ..eval.bins import QuantileBins


class PhaseValueTable:
    def __init__(self, bins: QuantileBins, unseen_score: str = "max"):
        self.bins = bins
        self.unseen = math.inf if unseen_score == "max" else 1.0
        self.freq: dict[tuple[str, str], dict[int, float]] = {}

    def fit(self, benign: pd.DataFrame) -> "PhaseValueTable":
        """``benign``: columns tag, value, phase (benign training commands)."""
        counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
        for r in benign.itertuples():
            counts[(r.tag, r.phase)][self.bins.bin(r.tag, r.value)] += 1
        for key, c in counts.items():
            n = sum(c.values())
            self.freq[key] = {b: k / n for b, k in c.items()}
        return self

    def score(self, tag: str, phase: str, value: float) -> float:
        f = self.freq.get((tag, phase), {}).get(self.bins.bin(tag, value))
        return -math.log(f) if f else self.unseen

    def score_frame(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series([self.score(r.tag, r.phase, r.value) for r in df.itertuples()], index=df.index)
