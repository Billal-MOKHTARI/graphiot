"""Detection metrics and statistics (paper §V-F).

Pure functions over arrays; the run-level assembly is in eval/evaluate.py.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom
from sklearn.metrics import average_precision_score, roc_auc_score


def finite_scores(s: np.ndarray) -> np.ndarray:
    """+inf -> above the largest finite score (ranking is preserved); NaN kept."""
    s = np.asarray(s, dtype=float).copy()
    fin = s[np.isfinite(s)]
    top = (fin.max() if len(fin) else 0.0) + 1.0
    s[np.isposinf(s)] = top
    s[np.isneginf(s)] = (fin.min() if len(fin) else 0.0) - 1.0
    return s


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    m = ~np.isnan(s)
    y, s = y[m], s[m]
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else math.nan


def auprc(y: np.ndarray, s: np.ndarray) -> float:
    m = ~np.isnan(s)
    y, s = y[m], s[m]
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else math.nan


def episode_recall(episode: np.ndarray, y: np.ndarray, alarm: np.ndarray) -> tuple[float, int, set]:
    """Share of episodes (with >= 1 positive command) where a positive command alarms."""
    eps = set(episode[y == 1])
    if not eps:
        return math.nan, 0, set()
    hit = set(episode[(y == 1) & alarm])
    return len(hit) / len(eps), len(eps), hit


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired exact McNemar test on boolean decisions a, b over the same episodes."""
    n01 = int((a & ~b).sum())          # a detects, b misses
    n10 = int((~a & b).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2.0 * binom.cdf(min(n01, n10), n, 0.5))
    return {"a_only": n01, "b_only": n10, "p": float(p)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted((p, k) for k, p in pvals.items() if p == p)
    m, out, running = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    for k, p in pvals.items():
        out.setdefault(k, math.nan)
    return out


def bootstrap_ci(values: list[float], ci: float) -> tuple[float, float]:
    v = np.asarray([x for x in values if x == x], dtype=float)
    if not len(v):
        return (math.nan, math.nan)
    a = (1 - ci) / 2
    return float(np.quantile(v, a)), float(np.quantile(v, 1 - a))
