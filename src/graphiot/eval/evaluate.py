"""Run-level evaluation (paper §V-F, H1-H3).

Test set: the last 20 % of benign time (split ``test``) plus every red-team
episode (split ``redteam``). Labels come from the oracle; detectors never saw them.

Slices (positives / negatives):
* ``all``   every test command; positive = harmful;
* ``allow`` commands the allow-list permits (B1's policy); positive = harmful;
* ``ctx``   permitted commands that are context-shifted (Definition 1: (tool, tag,
  bin) in C_benign, permitted, harmful) — positives — or not harmful — negatives.
  C_benign = triples of the benign training writes the oracle did not flag.

Episode recall at the detector's threshold (1 % FPR on benign validation): an
episode (task) with >= 1 positive command in the slice is detected if one of those
commands raises an alarm. Alarms per hour: alarms on slice negatives during benign
test time / benign test hours. Seeds: metrics are averaged over seeds; 95 % CIs by
bootstrap over episodes (all test tasks, benign and red-team), seeds averaged in
every resample. McNemar (exact) on episode decisions, majority vote over seeds,
Holm-corrected across the baselines.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..config.io import load_config
from ..model.prepare import prepare
from ..model.runner import ROOT, load_dataset
from ..plant.allow_list import load_allow_list
from .metrics import auprc, auroc, bootstrap_ci, episode_recall, finite_scores, holm, mcnemar_exact

LABELS = {"full": "GRAPHIOT", "agent_only": "Ours: agent subgraph", "plant_only": "Ours: plant subgraph",
          "network_only": "Ours: network subgraph", "no_task": "Ours: no task node",
          "b0": "B0 phase-aware value table", "b1": "B1 allow-list", "b2": "B2 same-feature XGBoost",
          "b3": "B3 E-GraphSAGE", "b4": "B4 AIRGuard", "b5": "B5 LLM judge"}
ABLATIONS = ("agent_only", "plant_only", "network_only", "no_task")
H1_BASELINES = ("b0", "b2", "b3", "b4", "b5")
BINARY = {"b1", "b4"}                         # decisions, no continuous score (Q6)
SLICES = ("all", "allow", "ctx")


@dataclass
class Det:
    name: str
    seed: int | None
    score: np.ndarray          # aligned to the evaluation table
    alarm: np.ndarray
    threshold: float


def _threshold(run: Path, name: str, seed: int | None) -> float:
    if seed is not None and name in LABELS and name not in {"b0", "b1", "b2", "b3", "b4", "b5"}:
        p = run / "models" / f"{name}_seed{seed}" / "threshold.json"
    else:
        p = run / "scores" / (f"{name}_seed{seed}.json" if seed is not None else f"{name}.json")
    return float(json.loads(p.read_text())["threshold"]) if p.exists() else math.inf


def collect(run: Path, table: pd.DataFrame) -> dict[str, list[Det]]:
    dets: dict[str, list[Det]] = {}
    for p in sorted((run / "scores").glob("*.parquet")):
        stem = p.stem
        name, seed = (stem.rsplit("_seed", 1)[0], int(stem.rsplit("_seed", 1)[1])) if "_seed" in stem else (stem, None)
        if name not in LABELS:
            continue
        df = pd.read_parquet(p)[["command_idx", "score", "alarm"]]
        m = table[["command_idx"]].merge(df, on="command_idx", how="left")
        dets.setdefault(name, []).append(Det(name, seed, m["score"].to_numpy(dtype=float),
                                             m["alarm"].fillna(False).to_numpy(dtype=bool), _threshold(run, name, seed)))
    return dets


def evaluation_table(run: Path, cfg) -> tuple[pd.DataFrame, dict]:
    ds = load_dataset(run, cfg)
    cmds = prepare(ds, "full").cmd
    labels = pd.read_parquet(run / "oracle" / "command_labels.parquet")[["command_idx", "harmful", "actor"]]
    cmds = cmds.merge(labels, on="command_idx", how="left")
    cmds["harmful"] = cmds["harmful"].fillna(False).astype(bool)
    al = load_allow_list(ROOT / cfg.paths.allow_list)

    def permitted(r) -> bool:
        ok, _ = al.decide(r.task_type, r.tool, r.tag if r.tool == "write_tag" else None)
        return ok and (r.tool == "write_tag" or al.task_types[r.task_type].may_write(r.tag))
    cmds["allowed"] = [permitted(r) for r in cmds.itertuples()]
    benign_train = cmds[(cmds["split"] == "train") & (cmds["decision"] == "forwarded") & ~cmds["harmful"]]
    c_benign = set(zip(benign_train["tool"], benign_train["tag"], benign_train["bin"]))
    cmds["in_c_benign"] = [(a, k, b) in c_benign for a, k, b in zip(cmds["tool"], cmds["tag"], cmds["bin"])]
    cmds["context_shifted"] = cmds["in_c_benign"] & cmds["allowed"] & cmds["harmful"]
    test = cmds[cmds["split"].isin(["test", "redteam"])].reset_index(drop=True)
    hours = sum(b - a for a, b in ds.splits.test) / 3600.0
    info = {"benign_test_hours": hours, "n_test_commands": int(len(test)),
            "n_harmful": int(test["harmful"].sum()), "n_context_shifted": int(test["context_shifted"].sum()),
            "n_train_triples": len(c_benign)}
    return test, info


def slice_masks(t: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    h, ok, cs = t["harmful"].to_numpy(), t["allowed"].to_numpy(), t["context_shifted"].to_numpy()
    return {"all": (np.ones(len(t), bool), h),
            "allow": (ok, h),
            "ctx": (ok & (cs | ~h), cs)}


def _metrics(t: pd.DataFrame, rows: np.ndarray, mask: np.ndarray, y: np.ndarray, d: Det, hours: float,
             binary: bool) -> dict:
    sel = rows[mask[rows]]
    yy, ss, aa = y[sel].astype(int), finite_scores(d.score[sel]), d.alarm[sel]
    ep = t["task_id"].to_numpy()[sel]
    rec, n_ep, _ = episode_recall(ep, yy, aa)
    neg_test = (yy == 0) & (t["split"].to_numpy()[sel] == "test")
    return {"auroc": math.nan if binary else auroc(yy, ss), "auprc": math.nan if binary else auprc(yy, ss),
            "recall": rec, "episodes": n_ep,
            "alarms_per_h": float(aa[neg_test].sum() / hours) if hours > 0 else math.nan,
            "fpr_test": float(aa[neg_test].mean()) if neg_test.any() else math.nan,
            "n": int(len(sel)), "pos": int(yy.sum())}


def _avg(ms: list[dict]) -> dict:
    keys = ms[0].keys()
    return {k: float(np.nanmean([m[k] for m in ms])) if any(m[k] == m[k] for m in ms) else math.nan for k in keys}


def evaluate(run: Path, *, resamples: int | None = None, on_log: Callable[[str], None] = lambda s: None) -> dict:
    run = Path(run)
    cfg = load_config(run / "config.yaml")
    ev = cfg.evaluation
    t, info = evaluation_table(run, cfg)
    dets = collect(run, t)
    if not dets:
        raise FileNotFoundError("no detector scores in scores/ — train the model and run the baselines first")
    masks = slice_masks(t)
    hours = info["benign_test_hours"]
    all_rows = np.arange(len(t))
    point = {name: {sl: _avg([_metrics(t, all_rows, *masks[sl], d, hours, name in BINARY) for d in runs])
                    for sl in SLICES} for name, runs in dets.items()}

    # bootstrap over episodes (tasks), seeds averaged in each resample
    B = resamples or ev.bootstrap_resamples
    rng = np.random.default_rng(cfg.seeds[0])
    tasks = t["task_id"].to_numpy()
    groups = {k: np.flatnonzero(tasks == k) for k in pd.unique(tasks)}
    keys = list(groups)
    boot = {name: {sl: {m: [] for m in ("auroc", "auprc", "recall", "alarms_per_h")} for sl in SLICES} for name in dets}
    for b in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        rows = np.concatenate([groups[keys[i]] for i in pick])
        tb = t.iloc[rows].reset_index(drop=True)
        # resampled episodes must stay distinct episodes: tag duplicates
        tb["task_id"] = [f"{k}#{j}" for j, i in enumerate(pick) for k in [keys[i]] * len(groups[keys[i]])]
        mb = {sl: (masks[sl][0][rows], masks[sl][1][rows]) for sl in SLICES}
        scale = hours * (tb["split"].eq("test").sum() / max(1, t["split"].eq("test").sum()))
        for name, runs in dets.items():
            for sl in SLICES:
                ms = [_metrics(tb, np.arange(len(tb)), *mb[sl], Det(d.name, d.seed, d.score[rows], d.alarm[rows],
                                                                   d.threshold), scale, name in BINARY) for d in runs]
                a = _avg(ms)
                for m in boot[name][sl]:
                    boot[name][sl][m].append(a[m])
        if on_log and b % 200 == 0:
            on_log(f"bootstrap {b}/{B}")
    ci = {name: {sl: {m: bootstrap_ci(v, ev.ci) for m, v in d.items()} for sl, d in s.items()} for name, s in boot.items()}

    # McNemar on episode decisions (majority vote over seeds), Holm across baselines
    tests = {}
    if "full" in dets:
        for sl in SLICES:
            mask, y = masks[sl]
            eps = sorted(set(tasks[mask & (y == 1)]))
            if not eps:
                continue

            def detected(runs):
                votes = []
                for d in runs:
                    hit = set(tasks[mask & (y == 1) & d.alarm])
                    votes.append(np.array([e in hit for e in eps]))
                return np.sum(votes, axis=0) * 2 > len(votes)
            ours = detected(dets["full"])
            raw = {}
            for bname in H1_BASELINES:
                if bname in dets:
                    raw[bname] = mcnemar_exact(ours, detected(dets[bname]))
            adj = holm({k: v["p"] for k, v in raw.items()}) if ev.holm else {k: v["p"] for k, v in raw.items()}
            for k, v in raw.items():
                v["p_holm"] = adj[k]
                v["recall_diff_points"] = 100 * (point["full"][sl]["recall"] - point[k][sl]["recall"])
            tests[sl] = {"episodes": len(eps), "vs": raw}

    h1 = None
    if "ctx" in tests:
        vs = tests["ctx"]["vs"]
        h1 = {"supported": bool(vs) and all(v["recall_diff_points"] >= 10 and v["p_holm"] < 0.05 for v in vs.values()),
              "baselines_compared": sorted(vs), "missing_baselines": [b for b in H1_BASELINES if b not in vs]}
    res = {"info": info, "detectors": {n: [d.seed for d in r] for n, r in dets.items()},
           "thresholds": {n: [d.threshold for d in r] for n, r in dets.items()},
           "point": point, "ci": ci, "mcnemar": tests, "h1": h1, "bootstrap_resamples": B}
    out = run / "results"
    out.mkdir(exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(res, indent=1, default=_json))
    write_tables(res, out)
    from .figures import pr_curves, recall_ci
    res["figures"] = [str(p.relative_to(run)) for p in recall_ci(res, out / "figures", LABELS)
                      + pr_curves(t, dets, masks, out / "figures", LABELS)]
    t.to_parquet(out / "evaluation_commands.parquet", index=False)
    return res


def _json(o):
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def _fmt(v: float, lo: float | None = None, hi: float | None = None, pct: bool = False) -> str:
    if v is None or v != v:
        return "n/a"
    f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.3f}")
    s = f(v)
    if lo is not None and lo == lo:
        s += f" [{f(lo)}, {f(hi)}]"
    return s


def write_tables(res: dict, out: Path) -> None:
    def table(names: list[str], slices: tuple[str, ...]) -> pd.DataFrame:
        rows = []
        for n in names:
            if n not in res["point"]:
                continue
            r = {"detector": LABELS.get(n, n), "seeds": len(res["detectors"][n])}
            for sl in slices:
                p, c = res["point"][n][sl], res["ci"][n][sl]
                r[f"{sl} AUROC"] = _fmt(p["auroc"], *c["auroc"])
                r[f"{sl} AUPRC"] = _fmt(p["auprc"], *c["auprc"])
                r[f"{sl} recall@1%FPR (%)"] = _fmt(p["recall"], *c["recall"], pct=True)
                r[f"{sl} alarms/h"] = _fmt(p["alarms_per_h"])
            rows.append(r)
        return pd.DataFrame(rows)
    main = table(["full", "b0", "b1", "b2", "b3", "b4", "b5"], SLICES)
    abl = table(["full", *ABLATIONS], SLICES)
    for name, df in (("main_table", main), ("ablation_table", abl)):
        df.to_csv(out / f"{name}.csv", index=False)
        (out / f"{name}.tex").write_text(df.to_latex(index=False, escape=True))
    rows = []
    for sl, d in res["mcnemar"].items():
        for b, v in d["vs"].items():
            rows.append({"slice": sl, "baseline": LABELS.get(b, b), "episodes": d["episodes"],
                         "only GRAPHIOT": v["a_only"], "only baseline": v["b_only"],
                         "recall diff (points)": round(v["recall_diff_points"], 1),
                         "p": v["p"], "p (Holm)": v["p_holm"]})
    mc = pd.DataFrame(rows, columns=["slice", "baseline", "episodes", "only GRAPHIOT", "only baseline",
                                     "recall diff (points)", "p", "p (Holm)"])
    mc.to_csv(out / "mcnemar.csv", index=False)
    (out / "mcnemar.tex").write_text(mc.to_latex(index=False, escape=True, float_format="%.4f") if len(mc) else "")
