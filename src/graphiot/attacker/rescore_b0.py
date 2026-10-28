"""B0 (phase-aware value table) as an attacker-facing detector, for the evasion
comparison. B0 scores a write from (tag, stage phase, value bin); the attacker can
lower it by splitting a change into bins B0 has seen, or shifting it into another
phase. Phases are read from the synthesised tags (causal: last 1 Hz sample < t)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..baselines.b0 import PhaseValueTable
from ..baselines.common import val_threshold
from ..config.schema import ExperimentConfig
from ..data.monitor_view import MonitorView
from ..eval.bins import QuantileBins
from ..model.prepare import prepare
from ..model.runner import load_dataset
from ..plant.tags import PHASES
from .synth import Event, WindowSynth

_PREFIX = {"mix": "mixing", "coat": "coating", "slit": "slitting"}


class B0Detector:
    name = "b0"

    def __init__(self, run: Path, cfg: ExperimentConfig):
        self.cfg = cfg
        cmds = prepare(load_dataset(run, cfg), "full").cmd
        train = cmds[(cmds["split"] == "train") & (cmds["decision"] == "forwarded")]
        self.bins = QuantileBins(q=cfg.model.q_bins).fit(train[["tag", "value"]])
        self.table = PhaseValueTable(self.bins, cfg.baselines.b0_unseen_score).fit(train)
        p = run / "scores" / "b0.json"
        if p.exists():
            self.threshold = float(json.loads(p.read_text())["threshold"])
        else:
            val = cmds[(cmds["split"] == "val") & (cmds["decision"] == "forwarded")]
            s = [self.table.score(r.tag, r.phase, r.value) for r in val.itertuples()]
            self.threshold = val_threshold(s, cfg.model.fpr_target)

    def scorer(self, synth: WindowSynth) -> "B0Scorer":
        return B0Scorer(self, synth)


class B0Scorer:
    def __init__(self, det: B0Detector, synth: WindowSynth):
        self.det, self.synth = det, synth
        self.name, self.threshold = det.name, det.threshold

    def _phase_at(self, view: MonitorView, prefix: str, ts: float) -> str:
        ph = view.tags[(view.tags["tag"] == f"{prefix}.phase") & (view.tags["ts"] < ts)]
        if ph.empty:
            return "unknown"
        return PHASES.get(int(ph.sort_values("ts")["value"].iloc[-1]), "unknown")

    def max_score(self, view: MonitorView, events: list[Event], task_id: str, agent_id: str) -> float:
        scores = []
        for e in events:
            if e.kind != "write":
                continue
            prefix = e.tag.split(".")[0]
            phase = self._phase_at(view, prefix, e.ts)
            s = self.det.table.score(e.tag, phase, e.value)
            scores.append(s if math.isfinite(s) else 1e9)
        return float(max(scores)) if scores else float("-inf")
