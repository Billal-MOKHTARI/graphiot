"""Pilot go/no-go test (paper §V-C) and the pilot table (§VI-A).

Go criterion, both parts required:
1. at least ``pilot.min_context_shifted`` context-shifted commands (Definition 1:
   (tool, tag, bin) seen in benign training data, permitted by the allow-list,
   labelled harmful by the oracle);
2. a depth-``pilot.tree_depth`` decision tree on (tool, tag, bin) + stage phase +
   task type stays below ``pilot.tree_auprc_max`` AUPRC on them, with a
   chronological split of the pilot.

Population for the tree (and B0): commands that passed the gateway, either
context-shifted (positive) or not harmful (negative). Harmful commands outside
the slice are excluded from this separability test. Benign training data for
the bins, C_benign and B0 = commands before the split that the oracle did not
flag. Phase is read causally (last 1 Hz sample strictly before the command).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from ..baselines.b0 import PhaseValueTable
from ..config.schema import ExperimentConfig
from ..data.monitor_view import load_monitor_view
from .bins import QuantileBins
from .context import attach_context

FEATURES = ["tool", "tag", "bin", "phase", "task_type"]


def _auprc_ci(y: np.ndarray, s: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0 or y[i].sum() == len(i):
            continue
        vals.append(average_precision_score(y[i], s[i]))
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))) if vals else (math.nan, math.nan)


def run_pilot(run_dir: str | Path, cfg: ExperimentConfig, *, root: str | Path = ".") -> dict:
    run, root = Path(run_dir), Path(root)
    p = cfg.pilot
    mv = load_monitor_view(run, tag_map=root / cfg.paths.tag_map)
    labels = pd.read_parquet(run / "oracle" / "command_labels.parquet")       # oracle output
    episodes = pd.read_parquet(run / "oracle" / "episode_labels.parquet")
    cmds = attach_context(labels, mv.tags, mv.tasks)
    cmds["forwarded"] = cmds["decision"] == "forwarded"
    t_lo, t_hi = cmds["ts"].min(), cmds["ts"].max()
    t_split = t_lo + p.train_fraction * (t_hi - t_lo)
    cmds["split"] = np.where(cmds["ts"] < t_split, "train", "test")

    benign_train = cmds[(cmds["split"] == "train") & cmds["forwarded"] & ~cmds["harmful"]
                        & (cmds["actor"] == "operator")]
    series = mv.tags[(mv.tags["ts"] < t_split) & mv.tags["tag"].isin(set(cmds["tag"]))]
    bins = QuantileBins(q=cfg.model.q_bins).fit(benign_train[["tag", "value"]], series[["tag", "value"]])
    cmds["bin"] = [bins.bin(t, v) for t, v in zip(cmds["tag"], cmds["value"])]
    c_benign = set(zip(benign_train["tool"], benign_train["tag"],
                       [bins.bin(t, v) for t, v in zip(benign_train["tag"], benign_train["value"])]))
    cmds["in_c_benign"] = [(a, k, b) in c_benign for a, k, b in zip(cmds["tool"], cmds["tag"], cmds["bin"])]
    cmds["context_shifted"] = cmds["in_c_benign"] & cmds["forwarded"] & cmds["harmful"]
    cmds["in_benign_range"] = [bins.in_benign_range(t, v) for t, v in zip(cmds["tag"], cmds["value"])]
    cmds["context_shifted_strict"] = cmds["context_shifted"] & cmds["in_benign_range"]   # Q8 diagnostic

    pop = cmds[cmds["forwarded"] & (cmds["context_shifted"] | ~cmds["harmful"])].copy()
    pop["y"] = pop["context_shifted"].astype(int)
    tr, te = pop[pop["split"] == "train"], pop[pop["split"] == "test"]
    res: dict = {"n_context_shifted": int(cmds["context_shifted"].sum()),
                 "n_context_shifted_strict": int(cmds["context_shifted_strict"].sum()),
                 "n_train": int(len(tr)), "n_test": int(len(te)),
                 "pos_train": int(tr["y"].sum()), "pos_test": int(te["y"].sum())}
    if tr["y"].nunique() == 2 and te["y"].nunique() == 2:
        enc = OneHotEncoder(handle_unknown="ignore")
        Xtr = enc.fit_transform(tr[FEATURES].astype(str))
        Xte = enc.transform(te[FEATURES].astype(str))
        tree = DecisionTreeClassifier(max_depth=p.tree_depth, random_state=0).fit(Xtr, tr["y"])
        s_tree = tree.predict_proba(Xte)[:, 1]
        res["tree_auprc"] = float(average_precision_score(te["y"], s_tree))
        res["tree_auprc_ci95"] = _auprc_ci(te["y"].to_numpy(), s_tree)
        b0 = PhaseValueTable(bins, cfg.baselines.b0_unseen_score).fit(benign_train)
        s_b0 = b0.score_frame(te).to_numpy(dtype=float)
        finite = s_b0[np.isfinite(s_b0)]
        s_b0[~np.isfinite(s_b0)] = (finite.max() + 1.0) if len(finite) else 1.0
        res["b0_auprc"] = float(average_precision_score(te["y"], s_b0))
        res["b0_auprc_ci95"] = _auprc_ci(te["y"].to_numpy(), s_b0)
        res["prevalence_test"] = float(te["y"].mean())
    else:
        res["tree_auprc"] = None
        res["note"] = "not enough positives and negatives on both sides of the split"
    res["go"] = bool(res["n_context_shifted"] >= p.min_context_shifted
                     and res.get("tree_auprc") is not None and res["tree_auprc"] < p.tree_auprc_max)

    # ---- pilot table (§VI-A) --------------------------------------------
    gt = pd.read_parquet(run / "groundtruth" / "agent_periods.parquet")
    benign_h = float((gt.loc[gt["actor"] == "operator", "end_ts"] - gt.loc[gt["actor"] == "operator",
                                                                          "start_ts"]).sum()) / 3600.0
    rt = episodes[episodes["actor"] == "red_team"]
    ctx_by_task = cmds[cmds["context_shifted"]].groupby("task_id").size()
    per_tech = {}
    for tech, grp in rt.groupby("technique"):
        per_tech[tech] = {"episodes": int(len(grp)), "harmful": int(grp["harmful"].sum()),
                          "context_shifted": int(ctx_by_task.reindex(grp["task_id"]).fillna(0).sum())}
    op = cmds[(cmds["actor"] == "operator") & cmds["forwarded"]]
    table = {
        "benign_task_hours": round(benign_h, 3),
        "commands": int(len(cmds)), "commands_forwarded": int(cmds["forwarded"].sum()),
        "red_team_episodes": int(len(rt)), "harmful_red_team_episodes": int(rt["harmful"].sum()),
        "harmful_operator_episodes": int(((episodes["actor"] == "operator") & episodes["harmful"]).sum()),
        "context_shifted_commands": res["n_context_shifted"],
        "context_shifted_commands_in_benign_range": res["n_context_shifted_strict"],
        "per_technique": per_tech,
        "operator_commands_flagged_share": float(op["harmful"].mean()) if len(op) else None,
        "tree_auprc": res.get("tree_auprc"), "b0_auprc": res.get("b0_auprc"),
        "decision": "GO" if res["go"] else "NO-GO",
    }
    out = run / "results"
    out.mkdir(exist_ok=True)
    (out / "pilot.json").write_text(json.dumps({"test": res, "table": table}, indent=1, default=str))
    cmds.drop(columns=["prefix"]).to_parquet(out / "pilot_commands.parquet", index=False)
    rows = [(k, v) for k, v in table.items() if k != "per_technique"]
    rows += [(f"technique {t}: episodes / harmful / ctx", f"{d['episodes']} / {d['harmful']} / {d['context_shifted']}")
             for t, d in sorted(per_tech.items())]
    pd.DataFrame(rows, columns=["quantity", "value"]).to_csv(out / "pilot_table.csv", index=False)
    (out / "pilot_table.tex").write_text(_latex(rows))
    return {"test": res, "table": table}


def _latex(rows) -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v).replace("_", r"\_")
    body = "\n".join(f"{fmt(k)} & {fmt(v)} \\\\" for k, v in rows)
    return ("\\begin{tabular}{lr}\n\\toprule\nQuantity & Value \\\\\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}\n")
