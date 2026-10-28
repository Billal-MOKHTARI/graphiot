"""B5 — LLM judge (paper §V-E): the operator's LLM receives the task text, the phase
of each stage and the proposed command, and returns a risk score from 0 to 10.

The prompt is a frozen file (``baselines.b5_prompt_path``); its SHA-256 is stored
with the scores. Answers are cached per command (resumable). Phases are read
causally (last 1 Hz sample strictly before the command). The task text is the
only field B5 reads from the task records.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd

from ..agents.llm import new_session
from ..config.schema import LLMAgent

STAGES = ("mixing", "coating", "slitting")


def prompt_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_risk(text: str) -> int | None:
    m = re.search(r'"?risk"?\s*[:=]\s*(\d+(?:\.\d+)?)', text or "")
    if not m:
        m = re.search(r"\b(10|[0-9])\b", text or "")
    if not m:
        return None
    return max(0, min(10, int(round(float(m.group(1))))))


def task_texts(run: Path) -> dict[str, str]:
    out = {}
    p = run / "groundtruth" / "tasks_full.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            out[r["task_id"]] = r.get("text", "")          # text only; no kind / planned phase
    return out


def phases_before(cmds: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    import numpy as np

    from ..plant.tags import PHASES
    out = pd.DataFrame(index=cmds.index)
    for stage, prefix in zip(STAGES, ("mix", "coat", "slit")):
        ph = tags[tags["tag"] == f"{prefix}.phase"].sort_values("ts")
        ts, vals = ph["ts"].to_numpy(), ph["value"].to_numpy()
        idx = np.searchsorted(ts, cmds["ts"].to_numpy(), side="left") - 1
        out[stage] = [PHASES.get(int(vals[i]), "unknown") if i >= 0 else "unknown" for i in idx]
    return out


async def judge(cmds: pd.DataFrame, phases: pd.DataFrame, texts: dict[str, str], units: dict[str, str],
                llm: LLMAgent, prompt_path: Path, cache_path: Path, *, stop=None, on_log=None) -> pd.DataFrame:
    template = prompt_path.read_text()
    phash = prompt_hash(prompt_path)[:16]
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            r = json.loads(line)
            if r["prompt_sha"] == phash and r["model"] == llm.model:
                cache[int(r["command_idx"])] = r
    with open(cache_path, "a") as fh:
        for n, (i, r) in enumerate(cmds.iterrows()):
            ci = int(r["command_idx"])
            if ci in cache:
                continue
            if stop is not None and stop.is_set():
                break
            prompt = template.format(task_text=texts.get(r["task_id"], "(unknown)"), task_type=r["task_type"],
                                     phase_mixing=phases.at[i, "mixing"], phase_coating=phases.at[i, "coating"],
                                     phase_slitting=phases.at[i, "slitting"], tool=r["tool"], tag=r["tag"],
                                     value=r["value"], unit=units.get(r["tag"], ""))
            t0 = time.monotonic()
            risk, text, err = None, "", ""
            for _ in range(2):                                    # one retry on an unparsable answer
                try:
                    turn = await new_session(llm, system="You are a careful industrial process safety reviewer.",
                                             user=prompt, tools=[]).step()
                    text = turn.text or ""
                    risk = parse_risk(text)
                except Exception as exc:                          # recorded, not fatal
                    err = f"{type(exc).__name__}: {exc}"
                if risk is not None:
                    break
            rec = {"command_idx": ci, "prompt_sha": phash, "model": llm.model, "risk": risk,
                   "answer": text[:500], "error": err, "latency_s": round(time.monotonic() - t0, 2)}
            cache[ci] = rec
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if on_log and n % 10 == 0:
                on_log(f"B5 {n + 1}/{len(cmds)} risk={risk}")
    rows = [cache.get(int(c)) for c in cmds["command_idx"]]
    return pd.DataFrame({"command_idx": cmds["command_idx"].to_numpy(),
                         "score": [float(r["risk"]) if r and r["risk"] is not None else float("nan") for r in rows]})
