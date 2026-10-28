"""Evaluation (M8) on a synthetic run with known answers, and the statistics."""

import json
import math
import shutil

import numpy as np
import pandas as pd
import pytest

from graphiot.eval.metrics import episode_recall, holm, mcnemar_exact

from synthetic import make_view, write_run


def test_mcnemar_and_holm_known_values():
    a = np.array([True] * 10 + [False] * 5)
    b = np.array([False] * 10 + [False] * 5)
    r = mcnemar_exact(a, b)
    assert r["a_only"] == 10 and r["b_only"] == 0
    assert abs(r["p"] - 2 * 0.5 ** 10) < 1e-12
    adj = holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert adj == {"x": 0.03, "z": 0.06, "y": 0.06}          # step-down, monotone
    rec, n, hit = episode_recall(np.array(["e1", "e1", "e2", "e3"]), np.array([1, 0, 1, 0]),
                                 np.array([True, False, False, True]))
    assert n == 2 and rec == 0.5 and hit == {"e1"}           # e3 has no positive: not counted


@pytest.fixture(scope="module")
def synth_run(tmp_path_factory, root):
    run = tmp_path_factory.mktemp("evalrun")
    view, periods = make_view(1800.0)
    write_run(run, periods)
    (run / "capture").mkdir()
    for name in ("gateway_log", "commands", "tasks", "tags", "modbus", "flows"):
        getattr(view, name).to_parquet(run / "capture" / f"{name}.parquet", index=False)
    shutil.copy(root / "configs" / "dev_smoke.yaml", run / "config.yaml")
    cmds = view.commands.reset_index().rename(columns={"index": "command_idx"})
    mid = periods["start_ts"][1]
    red = (cmds["ts"] >= mid) & (cmds["ts"] < mid + 120)
    labels = cmds[["command_idx"]].copy()
    labels["harmful"] = red.to_numpy()
    labels["actor"] = np.where(red, "red_team", "operator")
    (run / "oracle").mkdir()
    labels.to_parquet(run / "oracle" / "command_labels.parquet", index=False)
    (run / "scores").mkdir()
    rng = np.random.default_rng(0)
    for seed in (11, 22, 33):                                  # perfect detector, three seeds
        s = labels["harmful"].astype(float) * 10 + rng.normal(0, 0.1, len(labels))
        pd.DataFrame({"command_idx": labels["command_idx"], "score": s, "alarm": s > 5}).to_parquet(
            run / "scores" / f"full_seed{seed}.parquet", index=False)
        (run / "models" / f"full_seed{seed}").mkdir(parents=True)
        (run / "models" / f"full_seed{seed}" / "threshold.json").write_text(json.dumps({"threshold": 5.0}))
    s = rng.normal(0, 1, len(labels))                          # uninformative detector
    pd.DataFrame({"command_idx": labels["command_idx"], "score": s, "alarm": s > 3}).to_parquet(
        run / "scores" / "b0.parquet", index=False)
    (run / "scores" / "b0.json").write_text(json.dumps({"threshold": 3.0}))
    return run, int(red.sum())


def test_evaluate_known_answers(synth_run):
    from graphiot.eval.evaluate import evaluate
    run, n_red = synth_run
    res = evaluate(run, resamples=60)
    assert res["info"]["n_harmful"] == n_red > 0
    full, b0 = res["point"]["full"]["all"], res["point"]["b0"]["all"]
    assert full["auroc"] == 1.0 and full["recall"] == 1.0 and full["alarms_per_h"] == 0.0
    assert b0["recall"] < 1.0
    lo, hi = res["ci"]["full"]["all"]["recall"]
    assert lo == hi == 1.0
    mc = res["mcnemar"]["all"]["vs"]["b0"]
    assert mc["a_only"] >= 1 and mc["b_only"] == 0 and mc["p_holm"] >= mc["p"]
    assert (run / "results" / "main_table.csv").exists() and (run / "results" / "figures" / "recall_ci.pdf").exists()
    assert res["h1"]["missing_baselines"] == ["b2", "b3", "b4", "b5"]
