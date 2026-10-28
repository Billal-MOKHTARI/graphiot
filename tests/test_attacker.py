"""Adaptive attacker (paper §IV-C): edit generation, the greedy search, evasion."""

import random

import pandas as pd
import pytest

from graphiot.attacker.core import Attacker, Episode, harmful_episodes
from graphiot.attacker.synth import Event


def _bare_attacker(edits=("insert_read", "interleave_benign", "split_change", "time_shift")):
    a = object.__new__(Attacker)
    a.rng = random.Random(0)
    a.edits = set(edits)
    a.max_shift = 30.0
    a.benign_pool = [("coat.web_speed_sp", 1.0)]
    return a


def _episode_events():
    return [Event(100.0, "coat.web_speed_sp", 3.0, "write", 0, target=True),
            Event(130.0, "coat.pump_flow_sp", 5.0, "write", 1, target=True)]


def test_candidates_cover_all_edit_types_and_split_preserves_value():
    a = _bare_attacker()
    evs = _episode_events()
    kinds = {k for k, _ in [c for _ in range(20) for c in a._candidates(evs)]}
    assert {"insert_read", "interleave_benign", "split_change", "time_shift"} <= kinds
    # a split of the first write reaches the same final value in two steps
    split = next(cand for k, cand in a._candidates(evs) if k == "split_change")
    w = [e for e in split if e.tag == "coat.web_speed_sp" and e.kind == "write"]
    assert len(w) == 2 and abs(max(e.value for e in w) - 3.0) < 1e-9
    assert len(split) == len(evs) + 1          # one added event


def test_insert_read_and_interleave_add_one_event():
    a = _bare_attacker()
    evs = _episode_events()
    for kind in ("insert_read", "interleave_benign"):
        cand = next(c for k, c in a._candidates(evs) if k == kind)
        assert len(cand) == len(evs) + 1
    # time shift keeps the count, moves a write within the budget
    shift = next(c for k, c in a._candidates(evs) if k == "time_shift")
    assert len(shift) == len(evs)
    assert {round(e.ts, 3) for e in shift} != {round(e.ts, 3) for e in evs}


def test_greedy_lowers_score_within_budget_and_respects_harm():
    a = _bare_attacker()
    base = _episode_events()
    base_len = len(base)
    # a detector whose score drops by 1 per added event (so adding events evades)
    score = lambda evs: 10.0 - (len(evs) - base_len)
    budgets = [0, 4, 8]
    scores, trail = a._greedy([e.copy() for e in base], score, lambda evs: True, budgets)
    assert scores[0] == 10.0
    assert scores[8] < scores[4] < scores[0]   # monotone improvement
    assert len(trail) <= 8                      # budget respected

    # if no edit preserves the effect, nothing changes
    scores2, trail2 = a._greedy([e.copy() for e in base], score, lambda evs: len(evs) == base_len, budgets)
    assert {b: scores2[b] for b in budgets} == {0: 10.0, 4: 10.0, 8: 10.0} and trail2 == []


def test_harmful_episodes_parsing(tmp_path):
    run = tmp_path
    (run / "oracle").mkdir()
    (run / "capture").mkdir()
    (run / "groundtruth").mkdir()
    pd.DataFrame([{"task_id": "T-1", "episode_id": "E-1", "actor": "red_team", "technique": "T0836",
                   "harmful": True, "caused_outputs": ["coat.residual_solvent"]},
                  {"task_id": "T-2", "episode_id": "E-2", "actor": "red_team", "technique": "T0855",
                   "harmful": False, "caused_outputs": []}]).to_parquet(run / "oracle" / "episode_labels.parquet")
    pd.DataFrame([{"ts": 10.0, "task_id": "T-1", "agent_id": "operator-1", "tool": "write_tag",
                   "tag": "coat.web_speed_sp", "value": 4.0, "decision": "forwarded"},
                  {"ts": 12.0, "task_id": "T-1", "agent_id": "operator-1", "tool": "read_tag",
                   "tag": "coat.web_speed_sp", "value": 4.0, "decision": "allowed"}]).to_parquet(
        run / "capture" / "commands.parquet")
    from graphiot.config.io import load_config
    cfg = load_config("configs/dev_smoke.yaml")
    eps = harmful_episodes(run, cfg)
    assert len(eps) == 1 and eps[0].task_id == "T-1"
    assert len(eps[0].writes) == 1 and eps[0].writes[0].tag == "coat.web_speed_sp"
    assert eps[0].violated_outputs == ["coat.residual_solvent"]
