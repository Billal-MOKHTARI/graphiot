"""Run the baselines of a run and write their scores (baselines/common.py format).

B0, B1 and B2 run in seconds. B5 calls the operator's LLM once per validation and
test command (cached, resumable). B3 and B4 have their own modules (public code).
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Callable

import pandas as pd

from ..config.io import load_config
from ..data.monitor_view import load_monitor_view
from ..model.prepare import prepare
from ..model.runner import ROOT, load_dataset
from ..plant.tags import catalog_by_name
from .b0 import PhaseValueTable
from .b1 import score_b1
from .b2 import SameFeatureXGB
from .common import benign_val, threshold_with_fallback, val_threshold, write_scores


def run_baselines(run: Path, which: list[str] | None = None, *, stop: threading.Event | None = None,
                  train_run: Path | None = None, on_log: Callable[[str], None] = lambda s: None) -> dict:
    run = Path(run)
    cfg = load_config(run / "config.yaml")
    b = cfg.baselines
    which = which or [n for n in ("b0", "b1", "b2", "b5") if getattr(b, n)]
    ds = load_dataset(run, cfg)
    cmds = prepare(ds, "full").cmd
    fpr = cfg.model.fpr_target
    val = benign_val(cmds)
    train_mask = (cmds["split"] == "train") & (cmds["decision"] == "forwarded")
    train_fwd = cmds[train_mask]
    out = {}

    def auto_threshold(df):
        """Threshold on benign validation scores, falling back to benign training (common.py)."""
        return threshold_with_fallback(df.loc[val.to_numpy(), "score"],
                                       df.loc[train_mask.to_numpy(), "score"], fpr)

    def finish(name, df, thr, *, source="benign_validation", n_threshold=None, **meta):
        write_scores(run, name, df, thr, n_val=int(val.sum()), fpr_target=fpr,
                     threshold_source=source, n_threshold_scores=n_threshold, **meta)
        v = df.loc[val.to_numpy(), "score"].astype(float)
        out[name] = {"threshold": thr, "threshold_source": source,
                     "val_fpr": float((v > thr).mean()) if len(v) else None, **meta}
        on_log(f"{name}: threshold {thr} ({source})")

    if "b0" in which:
        t = PhaseValueTable(ds.bins, b.b0_unseen_score).fit(train_fwd)
        df = pd.DataFrame({"command_idx": cmds["command_idx"].to_numpy(), "score": t.score_frame(cmds).to_numpy()})
        thr, src, n = auto_threshold(df)
        finish("b0", df, thr, source=src, n_threshold=n, unseen_score=b.b0_unseen_score, table_size=len(t.freq))
    if "b1" in which:
        df = score_b1(cmds, ROOT / cfg.paths.allow_list)
        finish("b1", df, 0.5, rule="allow-list decision (binary)")
    if "b2" in which:
        val_fwd = cmds[val]
        m = SameFeatureXGB(ds, n_estimators=b.b2_n_estimators, max_depth=b.b2_max_depth,
                           learning_rate=b.b2_learning_rate, seed=cfg.seeds[0]).fit(train_fwd, val_fwd)
        df = pd.DataFrame({"command_idx": cmds["command_idx"].to_numpy(), "score": m.score(cmds)})
        thr, src, n = auto_threshold(df)
        finish("b2", df, thr, source=src, n_threshold=n, classes=len(m.classes),
               best_iteration=getattr(getattr(m, "model", None), "best_iteration", None))
    if "b5" in which:
        from .b5 import judge, phases_before, prompt_hash, task_texts
        view = load_monitor_view(run, tag_map=ROOT / cfg.paths.tag_map)
        sel = cmds[cmds["split"].isin(["val", "test", "redteam"])]
        prompt = ROOT / b.b5_prompt_path
        (run / "baselines").mkdir(exist_ok=True)
        df = asyncio.run(judge(sel, phases_before(sel, view.tags), task_texts(run),
                               {n: t.unit for n, t in catalog_by_name().items()}, cfg.llm.operator, prompt,
                               run / "baselines" / "b5_cache.jsonl", stop=stop, on_log=on_log))
        full = cmds[["command_idx"]].merge(df, on="command_idx", how="left")
        n_missing = int(full.loc[full["command_idx"].isin(sel["command_idx"]), "score"].isna().sum())
        thr, src, n = auto_threshold(full)                # B5 scores val; no train scores, so no training fallback
        finish("b5", full, thr, source=src, n_threshold=n,
               prompt_sha256=prompt_hash(prompt), model=cfg.llm.operator.model, unparsed_or_missing=n_missing,
               scored_splits=["val", "test", "redteam"])
    if "b3" in which:
        if train_run is None:
            on_log("B3 skipped: a labelled training run (the pilot, Q5) is required; pass train_run")
        else:
            try:
                from .b3_egraphsage import run_b3
                sc = run_b3(run, Path(train_run), seed=cfg.seeds[0], cfg=cfg, on_log=on_log)
                full = cmds[["command_idx"]].merge(sc, on="command_idx", how="left")
                thr, src, n = auto_threshold(full)        # B3 scores val+test+redteam; no benign training fallback
                finish("b3", full, thr, source=src, n_threshold=n,
                       train_run=str(Path(train_run).name), scored=int(sc["score"].notna().sum()))
            except Exception as exc:                 # a missing env or incomplete capture: skip, don't abort
                on_log(f"B3 skipped: {exc}")
                out["b3"] = {"skipped": str(exc)}
    if "b4" in which:
        from .b4_airguard import score_b4
        df, meta = score_b4(run, cfg=cfg, on_log=on_log)
        full = cmds[["command_idx"]].merge(df, on="command_idx", how="left")
        full["score"] = full["score"].fillna(0.0)
        finish("b4", full, 0.5, **{k: meta[k] for k in ("binary", "denied", "total", "note")})
    (run / "results").mkdir(exist_ok=True)
    prev = json.loads((run / "results" / "baselines.json").read_text()) if (run / "results" / "baselines.json").exists() else {}
    prev.update(out)
    (run / "results" / "baselines.json").write_text(json.dumps(prev, indent=1, default=str))
    return out
