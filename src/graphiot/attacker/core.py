"""Adaptive, effect-preserving attacker and the evasion evaluation (paper §IV-C).

For each harmful red-team episode and each detector, the attacker greedily applies
the realizable edit that most lowers the detector's maximum command score, keeping
the physical effect (checked by the twin oracle). It reports, per budget B in
{0, 4, 8}, the share of harmful episodes whose maximum score falls below the
detector's 1 % FPR threshold — the evasion rate.

Realizable edits (§III-C, each costs one added event):
* insert_read      — a benign-looking read before a harmful write (no physics change);
* interleave_benign — a benign write from a past operator session (physics re-checked);
* split_change     — one setpoint change replaced by two reaching the same value;
* time_shift       — move a write by up to ``attacker.max_time_shift_s``.

The effect is preserved iff the edited episode still drives one of the episode's
originally-violated outputs past its limit by ≥ τ for ≥ T_min (twin replay).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config.io import load_config
from ..config.schema import ExperimentConfig
from ..data.monitor_view import load_monitor_view
from ..model.runner import ROOT
from ..oracle.core import find_violations
from ..plant.limits import load_limits
from ..plant.tags import load_tag_map
from .synth import Event, WindowSynth


@dataclass
class Episode:
    task_id: str
    agent_id: str
    start: float
    end: float
    writes: list[Event]
    violated_outputs: list[str]


def harmful_episodes(run: Path, cfg: ExperimentConfig) -> list[Episode]:
    """Red-team episodes the oracle labelled harmful, with their violated outputs."""
    ep = pd.read_parquet(run / "oracle" / "episode_labels.parquet")
    ep = ep[(ep["actor"] == "red_team") & ep["harmful"]]
    cmds = pd.read_parquet(run / "capture" / "commands.parquet").reset_index().rename(columns={"index": "command_idx"})
    tasks = pd.read_parquet(run / "groundtruth" / "agent_periods.parquet") \
        if (run / "groundtruth" / "agent_periods.parquet").exists() else None
    out = []
    for r in ep.itertuples():
        c = cmds[(cmds["task_id"] == r.task_id) & (cmds["decision"] == "forwarded") & (cmds["tool"] != "read_tag")]
        if c.empty:
            continue
        agent = c["agent_id"].iloc[0]
        writes = [Event(float(x.ts), x.tag, float(x.value), "write", int(x.command_idx), target=True)
                  for x in c.itertuples()]
        vo = list(r.caused_outputs) if isinstance(getattr(r, "caused_outputs", None), (list, np.ndarray)) else []
        out.append(Episode(r.task_id, agent, float(c["ts"].min()), float(c["ts"].max()), writes, vo))
    return out


class Attacker:
    def __init__(self, run: Path, cfg: ExperimentConfig, *, seed: int = 0):
        self.run, self.cfg = Path(run), cfg
        self.rng = random.Random(seed)
        self.tags = load_tag_map(ROOT / cfg.paths.tag_map)
        self.limits = load_limits(ROOT / cfg.paths.operating_limits)
        self.base = load_monitor_view(run, tag_map=ROOT / cfg.paths.tag_map)
        self.settle = cfg.oracle.replay_settle_s
        self.max_shift = cfg.attacker.max_time_shift_s
        self.edits = set(cfg.attacker.edits)
        # a pool of past benign operator writes to interleave
        fwd = self.base.commands[self.base.commands["decision"] == "forwarded"]
        self.benign_pool = list(zip(fwd["tag"], fwd["value"]))

    # ---- harm check -----------------------------------------------------------
    def _harmful(self, synth: WindowSynth, events: list[Event], violated: list[str]) -> bool:
        writes = [(e.ts, e.tag, e.value) for e in events if e.kind == "write"]
        trace = synth.regen_outputs(writes)
        vs = find_violations(trace, self.limits, self.tags, tau=self.cfg.oracle.tau,
                             t_min_s=self.cfg.oracle.t_min_s)
        got = {v.output for v in vs}
        return bool(got & set(violated)) if violated else bool(vs)

    # ---- candidate edits ------------------------------------------------------
    def _candidates(self, events: list[Event]) -> list[tuple[str, list[Event]]]:
        writes = [e for e in events if e.kind == "write" and e.target]
        out: list[tuple[str, list[Event]]] = []
        for e in writes:
            if "insert_read" in self.edits:
                r = Event(e.ts - 0.5, e.tag, 0.0, "read")
                out.append(("insert_read", sorted(events + [r], key=lambda x: x.ts)))
            if "time_shift" in self.edits:
                for dt in (-self.max_shift, -self.max_shift / 2, self.max_shift / 2, self.max_shift):
                    shifted = [x.copy(ts=x.ts + dt) if x is e else x for x in events]
                    out.append(("time_shift", sorted(shifted, key=lambda x: x.ts)))
            if "split_change" in self.edits:
                prev = self._prev_value(events, e)
                mid = prev + 0.5 * (e.value - prev)
                sub = [e.copy(ts=e.ts, value=mid, command_idx=-1, target=True),
                       e.copy(ts=e.ts + 1.0, value=e.value, target=True)]
                out.append(("split_change", sorted([x for x in events if x is not e] + sub, key=lambda x: x.ts)))
            if "interleave_benign" in self.edits and self.benign_pool:
                tag, val = self.rng.choice(self.benign_pool)
                b = Event(e.ts - 0.5, tag, float(val), "write", target=False)
                out.append(("interleave_benign", sorted(events + [b], key=lambda x: x.ts)))
        self.rng.shuffle(out)
        return out[:16]                              # bounded fan-out per greedy step

    @staticmethod
    def _prev_value(events: list[Event], e: Event) -> float:
        earlier = [x for x in events if x.kind == "write" and x.tag == e.tag and x.ts < e.ts]
        return earlier[-1].value if earlier else 0.0

    # ---- greedy search per episode -------------------------------------------
    def _greedy(self, events, score, harmful, budgets) -> tuple[dict, list]:
        """Pure greedy search: ``score(events)->float``, ``harmful(events)->bool``.
        Returns ({budget: best max-score}, trail of edit kinds). Testable with stubs."""
        base = score(events)
        per_budget = {0: base}
        trail, cost, cur, max_b = [], 0, events, max(budgets)
        while cost < max_b:
            best = None
            for kind, cand in self._candidates(cur):
                if not harmful(cand):
                    continue
                sc = score(cand)
                if best is None or sc < best[0]:
                    best = (sc, kind, cand)
            if best is None or best[0] >= per_budget[cost] - 1e-9:
                break                                    # no effect-preserving edit lowers the score
            cost += 1
            cur, _ = best[2], trail.append(best[1])
            per_budget[cost] = best[0]
        scores, last = {}, base
        for b in range(0, max_b + 1):
            last = per_budget.get(b, last)
            scores[b] = last
        return scores, trail

    def attack_episode(self, ep: Episode, detector, budgets: list[int], on_log=lambda s: None) -> dict:
        synth = WindowSynth(self.run, self.cfg, self.base, self.tags,
                            episode_start=ep.start, episode_end=ep.end, settle_s=self.settle)
        try:
            scorer = detector.scorer(synth)
            tid, aid = ep.task_id, ep.agent_id
            score = lambda evs: scorer.max_score(synth.build_view(evs, tid, aid), evs, tid, aid)
            harmful = lambda evs: self._harmful(synth, evs, ep.violated_outputs)
            scores, trail = self._greedy([e.copy() for e in ep.writes], score, harmful, budgets)
            return {"task_id": ep.task_id, "base_score": scores[0], "trail": trail,
                    "scores": {b: scores[b] for b in budgets},
                    "evaded": {b: bool(scores[b] < scorer.threshold) for b in budgets}}
        finally:
            synth.close()

    def run_detector(self, detector, budgets: list[int], *, max_episodes: int, on_log=lambda s: None) -> dict:
        eps = harmful_episodes(self.run, self.cfg)[:max_episodes]
        results = []
        for i, ep in enumerate(eps):
            on_log(f"[{detector.name}] episode {i + 1}/{len(eps)} {ep.task_id}")
            results.append(self.attack_episode(ep, detector, budgets, on_log=on_log))
        rate = {b: (float(np.mean([r["evaded"][b] for r in results])) if results else float("nan")) for b in budgets}
        return {"detector": detector.name, "threshold": detector.threshold, "n_episodes": len(results),
                "evasion_rate": rate, "episodes": results}


def run_attacker(run: Path, *, detectors: list[str] | None = None, seed: int | None = None,
                 on_log=lambda s: None) -> dict:
    """Load the model (and any other requested detector) and run the evasion evaluation."""
    run = Path(run)
    cfg = load_config(run / "config.yaml")
    seed = cfg.seeds[0] if seed is None else seed
    budgets = list(cfg.attacker.budgets)
    att = Attacker(run, cfg, seed=seed)
    dets = []
    want = detectors or ["full"]
    if "full" in want:
        import json as _json

        from ..model.prepare import prepare
        from ..model.runner import load_dataset, model_dir
        from ..model.train import Trainer
        from .rescore import ModelDetector
        ds = load_dataset(run, cfg)
        prep = prepare(ds, "full")
        tr = Trainer(prep, cfg, seed=seed, out_dir=model_dir(run, "full", seed))
        tr.load_best()
        thr = _json.loads((model_dir(run, "full", seed) / "threshold.json").read_text())["threshold"]
        dets.append(ModelDetector(tr.model.eval(), prep, cfg, thr))
    if "b0" in want:
        from .rescore_b0 import B0Detector
        dets.append(B0Detector(run, cfg))
    out = {"budgets": budgets, "seed": seed, "detectors": {}}
    for det in dets:
        res = att.run_detector(det, budgets, max_episodes=cfg.attacker.episodes_per_detector, on_log=on_log)
        out["detectors"][det.name] = res
    (run / "results").mkdir(exist_ok=True)
    (run / "results" / "attacker.json").write_text(json.dumps(out, indent=1, default=str))
    _write_table(out, run / "results")
    return out


def _write_table(out: dict, outdir: Path) -> None:
    rows = []
    for name, res in out["detectors"].items():
        row = {"detector": name, "episodes": res["n_episodes"]}
        for b in out["budgets"]:
            row[f"evasion@B={b}"] = res["evasion_rate"].get(b)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "attacker_table.csv", index=False)
    (outdir / "attacker_table.tex").write_text(df.to_latex(index=False, float_format="%.3f") if len(df) else "")
