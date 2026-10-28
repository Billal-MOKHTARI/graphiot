"""Run-level model steps: dataset cache, train (with resume), threshold, scores.

    runs/<run>/graph/dataset.pkl                    normalised snapshots + commands (+ inputs hash)
    runs/<run>/models/<variant>_seed<s>/last.pt     resumable checkpoint
    runs/<run>/models/<variant>_seed<s>/best.pt     best validation epoch
    runs/<run>/models/<variant>_seed<s>/threshold.json
    runs/<run>/scores/<variant>_seed<s>.parquet     surprise of every command (full timeline)
    runs/<run>/scores/full_seed<s>_memory.pt        memory every 50 snapshots (explanations)
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Callable

import pandas as pd
import torch

from ..config.io import load_config, resolve_device
from ..config.schema import ExperimentConfig
from ..data.monitor_view import MONITOR_FILES
from ..graph.dataset import GraphDataset, build_dataset
from .prepare import Prepared, prepare
from .train import Trainer, score_timeline

ROOT = Path(__file__).resolve().parents[3]


def _inputs_hash(run: Path, cfg: ExperimentConfig) -> str:
    h = hashlib.sha256()
    for rel in MONITOR_FILES.values():
        p = run / rel
        h.update(f"{rel}:{p.stat().st_size if p.exists() else -1}:{p.stat().st_mtime_ns if p.exists() else 0}".encode())
    p = run / "groundtruth" / "agent_periods.parquet"          # splits only
    h.update(str(p.stat().st_mtime_ns if p.exists() else 0).encode())
    h.update(json.dumps([cfg.graph.model_dump(), cfg.splits.model_dump(), cfg.model.q_bins], sort_keys=True).encode())
    return h.hexdigest()[:16]


def load_dataset(run: Path, cfg: ExperimentConfig | None = None, *, rebuild: bool = False) -> GraphDataset:
    cfg = cfg or load_config(run / "config.yaml")
    path = run / "graph" / "dataset.pkl"
    key = _inputs_hash(run, cfg)
    if path.exists() and not rebuild:
        ds = GraphDataset.load(path)
        if getattr(ds, "inputs_hash", None) == key:
            return ds
    ds = build_dataset(run, cfg, root=ROOT)
    ds.inputs_hash = key
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save(path)
    return ds


def model_dir(run: Path, variant: str, seed: int) -> Path:
    return run / "models" / f"{variant}_seed{seed}"


def train_one(run: Path, variant: str, seed: int, *, cfg: ExperimentConfig | None = None,
              resume: bool = True, stop: threading.Event | None = None,
              on_progress: Callable[[dict], None] | None = None, prep: Prepared | None = None) -> dict:
    cfg = cfg or load_config(run / "config.yaml")
    device = resolve_device(cfg)
    prep = prep or prepare(load_dataset(run, cfg), variant)
    out = model_dir(run, variant, seed)
    tr = Trainer(prep, cfg, seed=seed, out_dir=out, device=device)
    if resume and (out / "last.pt").exists():
        tr.load(out / "last.pt")
    st = tr.fit(stop=stop, on_progress=on_progress)
    summary = {"variant": variant, "seed": seed, "device": device, "params": tr.model.n_params(),
               "epochs": st.epoch, "steps": st.step, "best_epoch": st.best_epoch,
               "best_val_nll": None if st.best_val == float("inf") else st.best_val,
               "done": st.done, "stop_reason": st.stop_reason, "train_seconds": round(st.seconds, 1)}
    if st.done:
        tr.load_best()
        summary["threshold"] = tr.threshold()
        scores, mem = score_timeline(tr.model, prep, batch=cfg.model.batch_snapshots, device=device)
        thr = summary["threshold"]["threshold"]
        scores["alarm"] = scores["score"] > thr
        (run / "scores").mkdir(exist_ok=True)
        scores.to_parquet(run / "scores" / f"{variant}_seed{seed}.parquet", index=False)
        if variant == "full":
            torch.save(mem, run / "scores" / f"full_seed{seed}_memory.pt")
        summary["model_kb"] = round((out / "best.pt").stat().st_size / 1024, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def train_all(run: Path, *, variants: list[str] | None = None, seeds: list[int] | None = None,
              stop: threading.Event | None = None, on_progress: Callable[[dict], None] | None = None) -> list[dict]:
    cfg = load_config(run / "config.yaml")
    variants = variants or ["full", *cfg.baselines.ablations]
    seeds = seeds or list(cfg.seeds)
    ds = load_dataset(run, cfg)
    out = []
    for variant in variants:
        prep = prepare(ds, variant)
        for seed in seeds:
            if stop is not None and stop.is_set():
                return out
            prog = (lambda r, v=variant, s=seed: on_progress({**r, "variant": v, "seed": s})) if on_progress else None
            out.append(train_one(run, variant, seed, cfg=cfg, stop=stop, on_progress=prog, prep=prep))
    (run / "results").mkdir(exist_ok=True)
    (run / "results" / "training.json").write_text(json.dumps(out, indent=1))
    return out


def scores_frame(run: Path) -> pd.DataFrame:
    """All model score files of a run, long format (variant, seed, command_idx, score, alarm)."""
    parts = []
    for p in sorted((run / "scores").glob("*_seed*.parquet")):
        variant, seed = p.stem.rsplit("_seed", 1)
        df = pd.read_parquet(p)
        df["variant"], df["seed"] = variant, int(seed)
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
