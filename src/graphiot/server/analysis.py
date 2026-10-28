"""GUI endpoints for M4-M8: graph snapshots, training, scores and explanations,
edge cost, baselines and evaluation. Long steps run as jobs (see app.State)."""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel


class TrainReq(BaseModel):
    variants: list[str] | None = None
    seeds: list[int] | None = None
    resume: bool = True


class BaselineReq(BaseModel):
    which: list[str] | None = None
    train_run: str | None = None       # labelled run for B3 (the pilot, Q5)


class EvalReq(BaseModel):
    resamples: int | None = None


class AttackReq(BaseModel):
    detectors: list[str] | None = None     # "full", "b0"
    seed: int | None = None


def _clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, (np.floating, np.integer)):
        return _clean(v.item())
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _deep(v):
    """JSON-safe copy: non-finite floats -> None, numpy scalars -> Python."""
    if isinstance(v, dict):
        return {str(k): _deep(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_deep(x) for x in v]
    return _clean(v)


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")]


def register(app, S, run_dir) -> None:
    from ..config.io import load_config
    from ..graph.schema import feature_names

    stops: dict[str, threading.Event] = {}
    explainers: dict[tuple, object] = {}
    raw_builders: dict[str, object] = {}

    def dataset(run: Path):
        from ..graph.dataset import GraphDataset
        p = run / "graph" / "dataset.pkl"
        if not p.exists():
            raise HTTPException(409, "graph dataset not built yet (Graph tab: Build)")
        return GraphDataset.load(p)

    # ---- graph -----------------------------------------------------------------------
    @app.get("/api/runs/{run_id}/graph/meta")
    def graph_meta(run_id: str):
        run = run_dir(run_id)
        p = run / "graph" / "dataset.pkl"
        if not p.exists():
            return {"available": False, "has_capture": (run / "capture" / "commands.parquet").exists()}
        ds = dataset(run)
        c = ds.commands
        return {"available": True, "n": len(ds.times), "t0": float(ds.times[0]) if len(ds.times) else None,
                "stride_s": ds.stride_s, "window_s": ds.window_s,
                "splits": [str(s) for s in ds.snap_split],
                "commands_per_snapshot": np.bincount(c["k"][c["k"] >= 0], minlength=len(ds.times)).tolist(),
                "n_commands": int(len(c)), "writable_tags": ds.writable_tags,
                "split_counts": {str(k): int(v) for k, v in c["split"].value_counts().items()}}

    @app.post("/api/runs/{run_id}/graph/build")
    def graph_build(run_id: str):
        run = run_dir(run_id)
        job = S.new_job("graph_build", run_id)

        def go():
            from ..model.runner import load_dataset
            S.job_log(job, "building snapshots (Δ, stride from the config), normaliser on the training split")
            ds = load_dataset(run, rebuild=True)
            return {"snapshots": len(ds.times), "commands": int(len(ds.commands))}
        S.run_thread(job, go)
        return {"job_id": job.id}

    @app.get("/api/runs/{run_id}/graph/snapshot/{k}")
    def graph_snapshot(run_id: str, k: int, raw: bool = False):
        run = run_dir(run_id)
        ds = dataset(run)
        if not 0 <= k < len(ds.times):
            raise HTTPException(404, "snapshot index out of range")
        if raw:
            if run_id not in raw_builders:
                from ..data.monitor_view import load_monitor_view
                from ..graph.builder import SnapshotBuilder, Topology
                cfg = load_config(run / "config.yaml")
                view = load_monitor_view(run, tag_map=Path(__file__).resolve().parents[3] / cfg.paths.tag_map)
                raw_builders.clear()
                raw_builders[run_id] = SnapshotBuilder(view, Topology.from_config(view, cfg.testbed.network),
                                                       window_s=cfg.graph.window_s, stride_s=cfg.graph.stride_s,
                                                       rolling_s=cfg.graph.rolling_window_s)
            snap = raw_builders[run_id].build(float(ds.times[k]))
        else:
            snap = ds.snaps[k]
        fn, fe = feature_names(ds.writable_tags)
        nodes, edges = [], []
        for nt, keys in snap.nodes.items():
            x = snap.x.get(nt)
            for i, key in enumerate(keys):
                feats = {n: _clean(float(v)) for n, v in zip(fn[nt], x[i])} if x is not None and len(x) else {}
                nodes.append({"id": key, "type": nt, "label": key.split(":", 1)[1], "features": feats})
        for et, (s, d, a) in snap.edges.items():
            for j in range(len(s)):
                edges.append({"id": f"{et[1]}:{j}:{et[0]}{s[j]}-{et[2]}{d[j]}", "relation": et[1],
                              "source": snap.nodes[et[0]][int(s[j])], "target": snap.nodes[et[2]][int(d[j])],
                              "features": {n: _clean(float(v)) for n, v in zip(fe[et], a[j])}})
        c = ds.commands
        cmds = c[c["k"] == k][["command_idx", "ts", "task_id", "tool", "tag", "value", "decision"]]
        return {"k": k, "T": float(ds.times[k]), "split": str(ds.snap_split[k]), "raw": raw,
                "nodes": nodes, "edges": edges, "commands_scored_here": _records(cmds)}

    # ---- training --------------------------------------------------------------------
    @app.get("/api/runs/{run_id}/models")
    def models(run_id: str):
        run = run_dir(run_id)
        out = []
        for d in sorted((run / "models").glob("*_seed*")):
            variant, seed = d.name.rsplit("_seed", 1)
            hist = [json.loads(l) for l in (d / "train_log.jsonl").read_text().splitlines()] \
                if (d / "train_log.jsonl").exists() else []
            summ = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else None
            thr = json.loads((d / "threshold.json").read_text()) if (d / "threshold.json").exists() else None
            out.append({"variant": variant, "seed": int(seed), "history": hist, "summary": summ, "threshold": thr,
                        "has_checkpoint": (d / "last.pt").exists(), "has_best": (d / "best.pt").exists()})
        return _deep({"models": out, "training": run_id in stops and not stops[run_id].is_set()})

    @app.post("/api/runs/{run_id}/train")
    def train(run_id: str, req: TrainReq):
        run = run_dir(run_id)
        if run_id in stops and not stops[run_id].is_set():
            raise HTTPException(409, "training already running for this run")
        stop = stops[run_id] = threading.Event()
        job = S.new_job("train", run_id)

        def progress(r):
            S.emit({"type": "train", "run_id": run_id, "progress": {k: _clean(v) for k, v in r.items()}})
            S.job_log(job, f"{r.get('variant')} seed {r.get('seed')} epoch {r['epoch']} "
                           f"train {r.get('train_nll')} val {r.get('val_nll')}")

        def go():
            from ..model.runner import train_all
            try:
                if not req.resume:
                    import shutil
                    for v in (req.variants or []):
                        for s in (req.seeds or []):
                            shutil.rmtree(run / "models" / f"{v}_seed{s}", ignore_errors=True)
                res = train_all(run, variants=req.variants, seeds=req.seeds, stop=stop, on_progress=progress)
                return {"trained": [{k: r[k] for k in ("variant", "seed", "done", "epochs", "best_val_nll")} for r in res],
                        "stopped": stop.is_set()}
            finally:
                stop.set()
        S.run_thread(job, go)
        return {"job_id": job.id}

    @app.post("/api/runs/{run_id}/train/stop")
    def train_stop(run_id: str):
        ev = stops.get(run_id)
        if ev is None or ev.is_set():
            raise HTTPException(409, "no training running")
        ev.set()                                     # the trainer checkpoints and returns
        return {"ok": True}

    # ---- scores & explanations ----------------------------------------------------------
    @app.get("/api/runs/{run_id}/detectors")
    def detectors(run_id: str):
        run = run_dir(run_id)
        names = sorted(p.stem for p in (run / "scores").glob("*.parquet")) if (run / "scores").exists() else []
        return {"detectors": names}

    @app.get("/api/runs/{run_id}/scores/{name}")
    def scores(run_id: str, name: str):
        run = run_dir(run_id)
        p = run / "scores" / f"{name}.parquet"
        if not p.exists():
            raise HTTPException(404, "no such score file")
        ds = dataset(run)
        df = ds.commands[["command_idx", "ts", "task_id", "task_type", "tool", "tag", "value", "phase", "split",
                          "decision", "k"]].merge(pd.read_parquet(p)[["command_idx", "score", "alarm"]],
                                                  on="command_idx", how="left")
        ev = run / "results" / "evaluation_commands.parquet"
        if ev.exists():
            lab = pd.read_parquet(ev)[["command_idx", "harmful", "context_shifted"]]
            df = df.merge(lab, on="command_idx", how="left")
        if "_seed" in name and name.split("_seed")[0] in ("full", "agent_only", "plant_only", "network_only", "no_task"):
            tp = run / "models" / name / "threshold.json"
        else:
            tp = run / "scores" / f"{name}.json"
        thr = json.loads(tp.read_text()).get("threshold") if tp.exists() else None
        return {"threshold": _clean(thr), "commands": _records(df.sort_values("ts"))}

    @app.get("/api/runs/{run_id}/explain/{command_idx}")
    def explain(run_id: str, command_idx: int, seed: int | None = None):
        run = run_dir(run_id)
        seeds = sorted(int(p.name.rsplit("_seed", 1)[1]) for p in (run / "models").glob("full_seed*")
                       if (p / "best.pt").exists())
        if not seeds:
            raise HTTPException(409, "train the model first (Training tab)")
        seed = seed if seed in seeds else seeds[0]
        key = (run_id, seed)
        if key not in explainers:
            from ..model.explain import Explainer
            explainers.clear()
            explainers[key] = Explainer(run, seed)
        try:
            e = explainers[key].explain(command_idx)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return _deep(e)

    @app.post("/api/runs/{run_id}/edge_cost")
    def edge_cost(run_id: str, seed: int | None = None):
        run = run_dir(run_id)
        cfg = load_config(run / "config.yaml")
        seed = seed or cfg.seeds[0]
        job = S.new_job("edge_cost", run_id)

        def go():
            from ..model.edge_cost import measure_limited
            S.job_log(job, f"measuring at {cfg.edge_cost.cpus} CPUs / {cfg.edge_cost.mem_gb} GB on test commands")
            return measure_limited(run, seed)
        S.run_thread(job, go)
        return {"job_id": job.id}

    # ---- baselines & evaluation ----------------------------------------------------------
    @app.post("/api/runs/{run_id}/baselines")
    def baselines(run_id: str, req: BaselineReq):
        run = run_dir(run_id)
        stop = threading.Event()
        job = S.new_job("baselines", run_id)

        train_run = run_dir(req.train_run) if req.train_run else None

        def go():
            from ..baselines.runner import run_baselines
            return run_baselines(run, req.which, stop=stop, train_run=train_run, on_log=lambda s: S.job_log(job, s))
        S.run_thread(job, go)
        return {"job_id": job.id}

    @app.post("/api/runs/{run_id}/evaluate")
    def evaluate(run_id: str, req: EvalReq):
        run = run_dir(run_id)
        if not (run / "oracle" / "command_labels.parquet").exists():
            raise HTTPException(409, "run the oracle first (Experiment tab: Oracle + pilot test)")
        job = S.new_job("evaluate", run_id)

        def go():
            from ..eval.evaluate import evaluate as ev
            res = ev(run, resamples=req.resamples, on_log=lambda s: S.job_log(job, s))
            return {"h1": res.get("h1"), "figures": res.get("figures")}
        S.run_thread(job, go)
        return {"job_id": job.id}

    @app.post("/api/runs/{run_id}/attacker")
    def attacker(run_id: str, req: AttackReq):
        run = run_dir(run_id)
        if not (run / "oracle" / "episode_labels.parquet").exists():
            raise HTTPException(409, "run the oracle first; the attacker needs harmful episodes")
        job = S.new_job("attacker", run_id)

        def go():
            from ..attacker.core import run_attacker
            return run_attacker(run, detectors=req.detectors, seed=req.seed,
                                on_log=lambda s: S.job_log(job, s))
        S.run_thread(job, go)
        return {"job_id": job.id}
