"""Shared output format of every detector (our model and the baselines).

    runs/<run>/scores/<detector>.parquet   command_idx, score, alarm
    runs/<run>/scores/<detector>.json      threshold, how it was set, metadata

Every detector scores every write command of the run under the causal rule. The
threshold is the (1 - fpr_target) quantile of its scores on the benign validation
commands the gateway forwarded ("higher" interpolation; alarm iff score > threshold),
the same rule for all. Binary detectors (B1, B4) use their own decision instead.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def val_threshold(val_scores, fpr: float) -> float:
    s = np.asarray(val_scores, dtype=float)
    s = s[np.isfinite(s)]
    return float(np.quantile(s, 1.0 - fpr, method="higher")) if len(s) else math.inf


def threshold_with_fallback(val_scores, train_scores, fpr: float) -> tuple[float, str, int]:
    """(1 - fpr) quantile on benign *validation* scores (paper §IV-B). When a run has no
    benign validation commands — e.g. a short pilot where the operator acted only early,
    so the validation time window holds no forwarded operator command — fall back to the
    benign *training* scores so the detector still has a finite threshold and can fire.
    Both sources are benign (never red-team data), so this stays paper-compliant; the
    source is recorded. Returns (threshold, source, n_scores_used)."""
    v = np.asarray([x for x in val_scores], dtype=float)
    v = v[np.isfinite(v)]
    if len(v):
        return float(np.quantile(v, 1.0 - fpr, method="higher")), "benign_validation", len(v)
    t = np.asarray([x for x in train_scores], dtype=float)
    t = t[np.isfinite(t)]
    if len(t):
        return float(np.quantile(t, 1.0 - fpr, method="higher")), "benign_training_fallback", len(t)
    return math.inf, "unset_no_benign_scores", 0


def write_scores(run: Path, name: str, scores: pd.DataFrame, threshold: float, **meta) -> Path:
    out = run / "scores"
    out.mkdir(exist_ok=True)
    df = scores[["command_idx", "score"]].copy()
    df["score"] = df["score"].astype(float)
    df["alarm"] = df["score"] > threshold
    df.to_parquet(out / f"{name}.parquet", index=False)
    (out / f"{name}.json").write_text(json.dumps({"detector": name, "threshold": threshold, **meta},
                                                  indent=1, default=str))
    return out / f"{name}.parquet"


def benign_val(cmds: pd.DataFrame) -> pd.Series:
    """Mask of benign validation commands that the gateway forwarded."""
    return (cmds["split"] == "val") & (cmds["decision"] == "forwarded")
