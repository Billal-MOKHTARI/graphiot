"""Chronological splits (paper §V-D): never random.

Benign time = the capture timeline minus every red-team episode interval (plus an
optional recovery margin after each, Q16). Training takes the first ``train``
fraction of benign time, validation the next ``val`` fraction, test the rest —
and every red-team episode goes to test, wherever it falls in time. Thresholds
and hyperparameters may only see the train and validation intervals.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Splits:
    train: list[tuple[float, float]]
    val: list[tuple[float, float]]
    test: list[tuple[float, float]]
    redteam: list[tuple[float, float]]      # all in test
    benign_seconds: float

    def assign(self, ts: float) -> str:
        for name in ("redteam", "train", "val", "test"):
            for a, b in getattr(self, name):
                if a <= ts < b:
                    return name
        return "excluded"


def _subtract(a: float, b: float, holes: list[tuple[float, float]]) -> list[tuple[float, float]]:
    segs = [(a, b)]
    for h0, h1 in sorted(holes):
        nxt = []
        for s0, s1 in segs:
            if h1 <= s0 or h0 >= s1:
                nxt.append((s0, s1))
            else:
                if h0 > s0:
                    nxt.append((s0, h0))
                if h1 < s1:
                    nxt.append((h1, s1))
        segs = nxt
    return segs


def _cut(segs: list[tuple[float, float]], start: float, stop: float) -> list[tuple[float, float]]:
    """Sub-intervals covering cumulative benign seconds [start, stop)."""
    out, acc = [], 0.0
    for s0, s1 in segs:
        d = s1 - s0
        lo, hi = max(start - acc, 0.0), min(stop - acc, d)
        if hi > lo:
            out.append((s0 + lo, s0 + hi))
        acc += d
    return out


def chronological_splits(periods: pd.DataFrame, *, train: float, val: float,
                         exclude_after_redteam_s: float = 0.0) -> Splits:
    t0, t1 = float(periods["start_ts"].min()), float(periods["end_ts"].max())
    rt = periods[periods["actor"] == "red_team"]
    red = [(float(a), float(b)) for a, b in zip(rt["start_ts"], rt["end_ts"])]
    holes = [(a, b + exclude_after_redteam_s) for a, b in red]
    benign = _subtract(t0, t1 + 1e-6, holes)
    total = sum(b - a for a, b in benign)
    c1, c2 = train * total, (train + val) * total
    return Splits(train=_cut(benign, 0.0, c1), val=_cut(benign, c1, c2),
                  test=_cut(benign, c2, total + 1.0), redteam=red, benign_seconds=total)
