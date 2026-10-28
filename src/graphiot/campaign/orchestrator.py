"""Capture campaign: benign operation with random red-team takeovers (paper §V-A).

One agent operates the cell, one task at a time. Takeover times are drawn once,
uniformly at random over the planned campaign length (seeded, with a minimum
gap); when one is due, the *next* task is executed by the red-team agent under
the operator's identity and task id (Q15). The campaign ends when the benign
time reaches its target and every episode has run.

Everything that reveals who acted is ground truth, written under
``groundtruth/`` and never read by the monitor:

* ``tasks_full.jsonl``     every task with its text, kind and planned phase
* ``agent_periods.jsonl``  [start, end] of every task and which actor ran it
* ``episodes.jsonl``       red-team episodes (technique, model, outcome)
* ``llm_log.jsonl``        every prompt, model turn and tool exchange
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from ..agents.prompts import (load_technique, operator_system, operator_user, red_team_system,
                              red_team_user)
from ..agents.runner import LLMLog, run_agent
from ..config.schema import ExperimentConfig
from ..gateway.client import AdminClient, ToolClient
from ..plant.allow_list import load_allow_list
from ..plant.recipes import load_recipes
from ..tasks.generator import TaskGenerator
from .power import SleepInhibitor, SuspendWatch, stale_plcs

AGENT_ID = "operator-1"        # the single agent identity both actors use
MAX_CONSECUTIVE_LLM_ERRORS = 3


@dataclass
class CampaignState:
    running: bool = False
    phase: str = "idle"            # idle | operator | red_team | gap | done | stopped | error
    started: float = 0.0
    benign_s: float = 0.0
    red_team_s: float = 0.0
    tasks_done: int = 0
    episodes_done: int = 0
    episodes_planned: int = 0
    current_task: dict | None = None
    last_error: str = ""


# Result of a task the watchdog cancelled; the gateway log still has its calls.
_ABORTED = SimpleNamespace(ended="aborted", tool_calls=0, writes=0, denied=0, models_served=[])


def _model_identity(agent) -> dict:
    """Provider, model tag and — for Ollama — the weights digest (a tag can move)."""
    out = {"provider": agent.provider, "model": agent.model, "family": agent.family}
    if agent.provider == "ollama":
        try:
            import httpx
            tags = httpx.get(f"{agent.base_url}/api/tags", timeout=5).json().get("models", [])
            m = next((m for m in tags if m.get("name") == agent.model or m.get("model") == agent.model), None)
            if m:
                out.update(digest=m.get("digest"), parameter_size=m.get("details", {}).get("parameter_size"),
                           quantization=m.get("details", {}).get("quantization_level"))
        except Exception as exc:                 # best effort; never blocks a campaign
            out["digest_error"] = str(exc)
    return out


def preflight_llms(cfg: ExperimentConfig) -> list[str]:
    """Problems that would make every task end in ``llm_error`` (empty list = ready):
    an Ollama model tag that is not installed, or a cloud provider without its key."""
    problems = []
    for role, agent in (("operator", cfg.llm.operator), ("red_team", cfg.llm.red_team)):
        if agent.provider == "ollama":
            try:
                import httpx
                tags = httpx.get(f"{agent.base_url}/api/tags", timeout=5).json().get("models", [])
            except Exception as exc:
                problems.append(f"{role}: Ollama not reachable at {agent.base_url} ({exc})")
                continue
            names = {m.get("name") for m in tags} | {m.get("model") for m in tags}
            if agent.model not in names:
                problems.append(f"{role}: model {agent.model!r} is not installed in Ollama "
                                f"(installed: {', '.join(sorted(n for n in names if n))})")
        else:
            from ..utils.secrets import STORE
            if not STORE.get(agent.api_key_env):
                problems.append(f"{role}: no API key in {agent.api_key_env} (.env or the GUI field)")
    return problems


def plan_takeovers(n: int, horizon_s: float, min_gap_s: float, rng: random.Random) -> list[float]:
    """n times in [0, horizon) with pairwise gaps >= min_gap (rejection sampling,
    falling back to an even grid with jitter if the horizon is too tight)."""
    if n <= 0:
        return []
    for _ in range(2000):
        ts = sorted(rng.uniform(0.0, horizon_s) for _ in range(n))
        if all(b - a >= min_gap_s for a, b in zip(ts, ts[1:])):
            return ts
    step = horizon_s / n
    return [i * step + rng.uniform(0, max(0.0, step - min_gap_s)) for i in range(n)]


class Campaign:
    def __init__(self, cfg: ExperimentConfig, run_dir: str | Path, *, gateway_url: str,
                 admin_token: str, root: str | Path = ".",
                 on_event: Callable[[dict], None] | None = None):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.gw_url = gateway_url
        self.admin_token = admin_token
        self.root = Path(root)
        self.on_event = on_event or (lambda e: None)
        self.state = CampaignState()
        self._stop = asyncio.Event()
        self._fault: str | None = None          # set by the watchdog; aborts the current task
        self._fault_event = asyncio.Event()
        c = cfg.campaign
        self.rng = random.Random(c.schedule_seed)
        self.gen = TaskGenerator(load_recipes(self.root / cfg.paths.recipes),
                                 load_allow_list(self.root / cfg.paths.allow_list),
                                 c.tasks, seed=c.schedule_seed)
        self.plant_doc = (self.root / cfg.paths.plant_documentation).read_text()
        rt = c.red_team
        est_episode_s = 0.5 * rt.max_episode_s
        horizon = c.benign_duration_s + rt.episodes * est_episode_s
        self.takeovers = plan_takeovers(rt.episodes, horizon, rt.min_gap_s, self.rng)
        techniques = [rt.techniques[i % len(rt.techniques)] for i in range(rt.episodes)]
        self.rng.shuffle(techniques)
        self.techniques = techniques
        self.state.episodes_planned = rt.episodes
        gt = self.run_dir / "groundtruth"
        gt.mkdir(parents=True, exist_ok=True)
        self._fh = {n: open(gt / f"{n}.jsonl", "a", buffering=1)
                    for n in ("tasks_full", "agent_periods", "episodes")}
        self.llm_log = LLMLog(gt / "llm_log.jsonl")
        (gt / "schedule.json").write_text(json.dumps(
            {"takeover_offsets_s": self.takeovers, "techniques": self.techniques,
             "schedule_seed": c.schedule_seed}, indent=1))
        (gt / "llm_models.json").write_text(json.dumps(
            {"operator": _model_identity(cfg.llm.operator), "red_team": _model_identity(cfg.llm.red_team)},
            indent=1))

    def _gt(self, stream: str, rec: dict) -> None:
        self._fh[stream].write(json.dumps(rec, default=str) + "\n")

    def _emit(self, kind: str, **data) -> None:
        self.on_event({"ts": time.time(), "kind": kind, **data})

    def stop(self) -> None:
        self._stop.set()

    async def _sleep(self, s: float) -> bool:
        """Interruptible sleep; returns False if the campaign was stopped."""
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=max(0.0, s))
            return False
        except asyncio.TimeoutError:
            return True

    def _done(self) -> bool:
        c = self.cfg.campaign
        return (self.state.benign_s >= c.benign_duration_s
                and self.state.episodes_done >= c.red_team.episodes)

    def _host_event(self, kind: str, **data) -> None:
        with open(self.run_dir / "groundtruth" / "host_events.jsonl", "a") as fh:
            fh.write(json.dumps({"ts": time.time(), "kind": kind, **data}) + "\n")

    async def _watchdog(self) -> None:
        """Stop the campaign on a host suspend or a silent PLC (see campaign/power.py)."""
        watch = SuspendWatch()
        plcs = sorted(self.cfg.testbed.network.plc_ips)
        tags_log = self.run_dir / "capture" / "tags.jsonl"
        last_live = 0.0
        while not self._stop.is_set():
            await asyncio.sleep(2.0)
            fault = None
            slept = watch.check()
            if slept:
                fault = (f"the host was suspended for {slept:.1f} s (around {time.strftime('%H:%M:%S')}); "
                         f"OpenPLC runtimes do not survive a suspend, so the campaign stopped. Data before "
                         f"the suspend is usable.")
                self._host_event("suspend", seconds=round(slept, 2))
            elif self.cfg.testbed.backend == "openplc_v4" and time.monotonic() - last_live >= 10.0:
                last_live = time.monotonic()
                stale = stale_plcs(tags_log, plcs, self.cfg.campaign.plc_stale_s)
                if stale:
                    fault = f"PLC(s) stopped answering the HMI: {stale} (seconds since last record)"
                    self._host_event("plc_stale", stale=stale)
            if fault:
                self._fault = fault
                self._fault_event.set()
                self._stop.set()
                return

    async def _until_fault(self, coro):
        """Run an agent task, cancelling it if the watchdog reports a fault."""
        task = asyncio.ensure_future(coro)
        fault = asyncio.ensure_future(self._fault_event.wait())
        await asyncio.wait({task, fault}, return_when=asyncio.FIRST_COMPLETED)
        if task.done():
            fault.cancel()
            return task.result()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        return None

    async def _register(self, admin: AdminClient, payload: dict) -> None:
        """Register a task, retrying transient gateway failures (5xx, transport errors);
        the gateway's registration is idempotent, so a retry never duplicates a task."""
        import httpx
        for attempt, wait in enumerate((2.0, 5.0, 10.0, None)):
            try:
                await admin.register_task(payload)
                return
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                server_side = not isinstance(exc, httpx.HTTPStatusError) or exc.response.status_code >= 500
                if wait is None or not server_side:
                    raise
                self._emit("gateway_retry", task_id=payload.get("task_id"), attempt=attempt + 1,
                           error=str(exc)[:200])
                if not await self._sleep(wait):
                    raise

    async def run(self) -> CampaignState:
        c, llm = self.cfg.campaign, self.cfg.llm
        st = self.state
        st.running, st.started = True, time.time()
        admin = AdminClient(self.gw_url, token=self.admin_token)
        tools = ToolClient(self.gw_url, agent_id=AGENT_ID)
        awake = SleepInhibitor(f"GRAPHIOT capture {self.run_dir.name}", enabled=c.inhibit_sleep).start()
        watchdog = asyncio.ensure_future(self._watchdog())
        self._emit("campaign_start", takeovers=len(self.takeovers), sleep_inhibited=awake.active)
        llm_errors = 0                     # consecutive tasks that ended in llm_error
        try:
            next_takeover = 0
            while not self._stop.is_set() and not self._done():
                elapsed = time.time() - st.started
                red_due = (next_takeover < len(self.takeovers)
                           and elapsed >= self.takeovers[next_takeover])
                # Once benign time is complete, run the remaining episodes back to back
                # (separated by the minimum gap of benign operation).
                if st.benign_s >= c.benign_duration_s and next_takeover < len(self.takeovers):
                    red_due = True
                task = self.gen.next_task(elapsed)
                actor = "red_team" if red_due else "operator"
                ep_id = f"E-{next_takeover + 1:03d}" if red_due else None
                t0 = time.time()
                await self._register(admin, task.gateway_payload(AGENT_ID, t0))
                self._gt("tasks_full", {"ts": t0, **asdict(task)})
                st.current_task = {"task_id": task.task_id, "type": task.task_type,
                                   "text": task.text, "actor": actor}
                st.phase = actor
                self._emit("task_start", task_id=task.task_id, task_type=task.task_type,
                           actor=actor, text=task.text, episode_id=ep_id)
                if red_due:
                    technique = self.techniques[next_takeover]
                    system = red_team_system(self.plant_doc,
                                             load_technique(self.root / self.cfg.paths.mitre_dir, technique))
                    res = await self._until_fault(run_agent(role="red_team", llm=llm.red_team, system=system,
                                          user=red_team_user(), client=tools, task_id=task.task_id,
                                          pacing=llm.pacing, rng=self.rng,
                                          max_calls=llm.max_tool_calls_per_task * 2,
                                          deadline_s=c.red_team.max_episode_s, log=self.llm_log,
                                          episode_id=ep_id))
                    res = res or _ABORTED
                    next_takeover += 1
                    st.episodes_done += 1
                    self._gt("episodes", {"episode_id": ep_id, "task_id": task.task_id,
                                          "technique": technique, "provider": llm.red_team.provider,
                                          "model": llm.red_team.model,
                                          "models_served": res.models_served, "start_ts": t0,
                                          "end_ts": time.time(), "ended": res.ended,
                                          "tool_calls": res.tool_calls, "writes": res.writes,
                                          "denied": res.denied, "prompt_version": c.red_team.prompt_version})
                else:
                    res = await self._until_fault(run_agent(role="operator", llm=llm.operator,
                                          system=operator_system(self.plant_doc),
                                          user=operator_user(task.text, task.task_id), client=tools,
                                          task_id=task.task_id, pacing=llm.pacing, rng=self.rng,
                                          max_calls=llm.max_tool_calls_per_task, deadline_s=None,
                                          log=self.llm_log))
                    res = res or _ABORTED
                t1 = time.time()
                if red_due:
                    st.red_team_s += t1 - t0
                else:
                    st.benign_s += t1 - t0
                st.tasks_done += 1
                self._gt("agent_periods", {"start_ts": t0, "end_ts": t1, "actor": actor,
                                           "task_id": task.task_id, "episode_id": ep_id,
                                           "ended": res.ended})
                self._emit("task_end", task_id=task.task_id, actor=actor, ended=res.ended,
                           tool_calls=res.tool_calls, writes=res.writes, denied=res.denied)
                llm_errors = llm_errors + 1 if res.ended == "llm_error" else 0
                if llm_errors >= MAX_CONSECUTIVE_LLM_ERRORS:
                    raise RuntimeError(f"{llm_errors} consecutive tasks ended in llm_error; "
                                       f"check the models and keys (groundtruth/llm_log.jsonl)")
                # Benign idle gap (counts as benign time: the plant keeps running).
                st.phase = "gap"
                gap = max(self.gen.idle_gap(), c.red_team.min_gap_s if red_due else 0.0)
                g0 = time.time()
                if not await self._sleep(gap):
                    break
                st.benign_s += time.time() - g0
            if self._fault:
                raise RuntimeError(self._fault)
            st.phase = "stopped" if self._stop.is_set() else "done"
        except Exception as exc:
            st.phase, st.last_error = "error", f"{type(exc).__name__}: {exc}"
            self._emit("campaign_error", error=st.last_error)
            raise
        finally:
            st.running = False
            watchdog.cancel()
            awake.stop()
            await admin.aclose()
            await tools.aclose()
            self.llm_log.close()
            for fh in self._fh.values():
                fh.close()
            self._emit("campaign_end", phase=st.phase, benign_s=st.benign_s,
                       episodes=st.episodes_done, tasks=st.tasks_done)
        return st
