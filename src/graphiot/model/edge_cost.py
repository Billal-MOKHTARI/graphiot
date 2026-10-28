"""Edge cost of the monitor (paper §IV-B "Edge deployment", task §6).

The online monitor keeps the current snapshot and the memory up to date and scores
each write before forwarding it. Measured on the run's test commands, in order:

* ``update_ms``: one snapshot update from the monitor's streams — build the snapshot
  (window Δ), normalise, encode (HGT), GRU memory update;
* ``score_ms``: scoring one command from the current memory (the head);
* ``latency_ms`` per command = the update of its snapshot + its scoring, i.e. the
  worst case of a write arriving just after a snapshot boundary (conservative);
* peak resident memory of the monitor process and the model size.

``measure_limited`` runs the measurement in a container capped at ``edge_cost.cpus``
CPUs and ``edge_cost.mem_gb`` GB (the paper's gateway VM), reusing the project's
virtual environment read-only; without Docker it measures in-process with the
thread count capped, and says so.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def measure(run: Path, seed: int, n_commands: int, threads: int) -> dict:
    import torch

    from ..config.io import load_config
    from ..data.monitor_view import load_monitor_view
    from ..graph.builder import SnapshotBuilder, Topology
    from ..graph.schema import edge_dims, node_dims
    from .batch import make_chunk, restrict, to_tensor
    from .net import VARIANTS, Memory
    from .prepare import prepare
    from .runner import load_dataset, model_dir
    from .train import Trainer

    torch.set_num_threads(threads)
    cfg = load_config(run / "config.yaml")
    ds = load_dataset(run, cfg)
    prep = prepare(ds, "full")
    tr = Trainer(prep, cfg, seed=seed, out_dir=model_dir(run, "full", seed))
    tr.load_best()
    model = tr.model.eval()
    view = load_monitor_view(run, tag_map=ROOT / cfg.paths.tag_map)
    builder = SnapshotBuilder(view, Topology.from_config(view, cfg.testbed.network), window_s=cfg.graph.window_s,
                              stride_s=cfg.graph.stride_s, rolling_s=cfg.graph.rolling_window_s)
    nd, ed = node_dims(len(ds.writable_tags)), edge_dims()
    v = VARIANTS["full"]
    cmds = prep.cmd[prep.cmd["split"].isin(["test", "redteam"]) & (prep.cmd["k"] >= 0)].sort_values("ts")
    cmds = cmds.head(n_commands)
    if cmds.empty:
        raise ValueError("no test commands to measure")
    by_k = {int(k): g for k, g in cmds.groupby("k")}
    k_first, k_last = int(cmds["k"].min()), int(cmds["k"].max())
    mem = Memory(model.d)
    update_ms, score_ms, latency_ms = {}, [], []
    with torch.no_grad():
        # warm-up of the memory before the first measured command (not timed)
        for k in range(0, k_first):
            ch = prep.chunk((k,), model.reverse_edges)
            model.step_memory(ch, model.encode(ch), 0, mem)
        for k in range(k_first, k_last + 1):
            t0 = time.perf_counter()
            snap = ds.norm.transform(builder.build(float(ds.times[k])))
            ts = restrict(to_tensor(snap, nd, ed), v["nodes"], v["mask_tag_values"])
            ch = make_chunk([ts], model.reverse_edges)
            model.step_memory(ch, model.encode(ch), 0, mem)
            update_ms[k] = (time.perf_counter() - t0) * 1e3
            g = by_k.get(k)
            for j in range(0 if g is None else len(g)):
                t1 = time.perf_counter()
                hb = prep.head_batch(g.iloc[[j]], mem, "cpu")
                model.log_probs(mem, hb)
                s = (time.perf_counter() - t1) * 1e3
                score_ms.append(s)
                latency_ms.append(update_ms[k] + s)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    model_bytes = sum(p.numel() * p.element_size() for p in model.state_dict().values())
    pct = lambda a, q: float(np.percentile(a, q))
    return {"n_commands": len(latency_ms), "n_snapshots": len(update_ms), "threads": threads,
            "latency_ms_median": pct(latency_ms, 50), "latency_ms_p99": pct(latency_ms, 99),
            "score_ms_median": pct(score_ms, 50), "score_ms_p99": pct(score_ms, 99),
            "update_ms_median": pct(list(update_ms.values()), 50),
            "update_ms_p99": pct(list(update_ms.values()), 99),
            "peak_rss_mb": round(peak_mb, 1), "model_kb": round(model_bytes / 1024, 1),
            "params": model.n_params(), "seed": seed}


def measure_limited(run: Path, seed: int) -> dict:
    """Measure inside a CPU/memory-capped container; fall back to in-process."""
    from ..config.io import load_config
    from ..testbed import docker
    from ..testbed.compose import services_image_tag

    run = Path(run).resolve()
    ec = load_config(run / "config.yaml").edge_cost
    image = services_image_tag()
    out_file = run / "results" / "edge_cost.json"
    out_file.parent.mkdir(exist_ok=True)
    have_docker = docker.available()
    if have_docker and docker.run(["image", "inspect", image], check=False).returncode == 0:
        uid, gid = os.getuid(), os.getgid()
        cmd = ["run", "--rm", f"--cpus={ec.cpus}", f"--memory={ec.mem_gb}g", f"--memory-swap={ec.mem_gb}g",
               "--network", "none", "--user", f"{uid}:{gid}", "-e", "HOME=/tmp", "-e", f"PYTHONPATH={ROOT / 'src'}",
               "-v", f"{ROOT}:{ROOT}", "-w", str(ROOT), image,
               str(ROOT / ".venv" / "bin" / "python"), "-m", "graphiot.model.edge_cost", "--inside",
               "--run", str(run), "--seed", str(seed), "--n", str(ec.n_commands), "--threads", str(ec.cpus)]
        p = docker.run(cmd, check=False, timeout=3600)
        if p.returncode == 0:
            res = json.loads(p.stdout.strip().splitlines()[-1])
            res.update(environment=f"container --cpus={ec.cpus} --memory={ec.mem_gb}g ({image})")
            out_file.write_text(json.dumps(res, indent=1))
            return res
        err = (p.stderr or "")[-2000:]
    else:
        err = "Docker or the services image is not available"
    res = measure(run, seed, ec.n_commands, ec.cpus)
    res.update(environment=f"host process, {ec.cpus} threads, memory not capped ({err.strip()[:300]})")
    out_file.write_text(json.dumps(res, indent=1))
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inside", action="store_true")
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    res = measure(Path(a.run), a.seed, a.n, a.threads)
    sys.stdout.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
