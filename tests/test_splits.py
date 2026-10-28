"""Chronological splits: ordered, 70/10/20 of benign time, red team only in test."""

import pandas as pd
import pytest

from graphiot.data.splits import chronological_splits


@pytest.fixture
def periods():
    rows, t = [], 0.0
    for i in range(40):
        actor = "red_team" if i % 7 == 3 else "operator"
        rows.append({"start_ts": t, "end_ts": t + 60.0, "actor": actor, "task_id": f"T-{i}"})
        t += 90.0                                      # 30 s benign gap after each task
    return pd.DataFrame(rows)


def test_order_and_fractions(periods):
    s = chronological_splits(periods, train=0.7, val=0.1)
    dur = lambda segs: sum(b - a for a, b in segs)
    assert dur(s.train) == pytest.approx(0.7 * s.benign_seconds, rel=1e-6)
    assert dur(s.val) == pytest.approx(0.1 * s.benign_seconds, rel=1e-6)
    assert dur(s.test) == pytest.approx(0.2 * s.benign_seconds, rel=1e-3)
    assert max(b for _, b in s.train) <= min(a for a, _ in s.val)
    assert max(b for _, b in s.val) <= min(a for a, _ in s.test)


def test_red_team_never_in_train_or_val(periods):
    s = chronological_splits(periods, train=0.7, val=0.1)
    rt = periods[periods["actor"] == "red_team"]
    for a, b in zip(rt["start_ts"], rt["end_ts"]):
        for t in (a, (a + b) / 2, b - 1e-3):
            assert s.assign(t) == "redteam"            # counted as test
    for a, b in s.train + s.val:
        for ra, rb in zip(rt["start_ts"], rt["end_ts"]):
            assert b <= ra or a >= rb, "a red-team interval overlaps train/val"


def test_recovery_margin_excluded(periods):
    s = chronological_splits(periods, train=0.7, val=0.1, exclude_after_redteam_s=20.0)
    rt = periods[periods["actor"] == "red_team"].iloc[0]
    assert s.assign(rt["end_ts"] + 10.0) == "excluded"
