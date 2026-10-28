"""GUI backend: FastAPI + WebSocket over the experiment pipeline.

Everything the GUI does goes through here: configs (schema, validate, save),
runs, testbed up/down/health, campaign start/stop, processing (oracle + pilot),
live tails of the run's streams, agent transcripts, oracle and results views,
downloads, and an in-memory API-key store (values never leave this process).

Long operations run as background jobs; their progress and the campaign's
events are pushed to every connected WebSocket client.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import threading
import time
import traceback
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from .. import pipeline
from ..config.io import ConfigError, config_hash, dump_yaml, json_schema, load_config, parse_config
from ..plant.limits import load_limits
from ..plant.tags import load_tag_map
from ..testbed.manager import Testbed, project_name
from ..utils.secrets import STORE, RedactingFilter

ROOT = pipeline.ROOT
RUNS = ROOT / "runs"
CONFIGS = ROOT / "configs"
DIST = ROOT / "gui" / "dist"
log = logging.getLogger("graphiot.server")


# --------------------------------------------------------------------------- #
@dataclass
class Job:
    id: str
    kind: str
    run_id: str | None
    status: str = "running"          # running | done | error | stopped
    started: float = field(default_factory=time.time)
    finished: float | None = None
    log: list[str] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None

    def public(self) -> dict:
        return {k: getattr(self, k) for k in ("id", "kind", "run_id", "status", "started",
                                               "finished", "error")} | {"log": self.log[-200:]}


class State:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.clients: set[WebSocket] = set()
        self.testbeds: dict[str, object] = {}      # run_id -> Testbed
        self.campaigns: dict[str, object] = {}     # run_id -> Campaign
        self.loop: asyncio.AbstractEventLoop | None = None

    async def _broadcast(self, msg: dict) -> None:
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    def emit(self, msg: dict) -> None:
        """Thread-safe broadcast."""
        if self.loop:
            asyncio.run_coroutine_threadsafe(self._broadcast(msg), self.loop)

    def new_job(self, kind: str, run_id: str | None) -> Job:
        job = Job(id=uuid.uuid4().hex[:10], kind=kind, run_id=run_id)
        self.jobs[job.id] = job
        self.emit({"type": "job", "job": job.public()})
        return job

    def job_log(self, job: Job, line: str) -> None:
        job.log.append(f"[{time.strftime('%H:%M:%S')}] {line}")
        self.emit({"type": "job_log", "job_id": job.id, "line": job.log[-1]})

    def finish(self, job: Job, *, result=None, error=None) -> None:
        job.status = "error" if error else "done"
        job.result, job.error, job.finished = result, error, time.time()
        self.emit({"type": "job", "job": job.public()})

    def run_thread(self, job: Job, fn) -> None:
        def target():
            try:
                self.finish(job, result=fn())
            except Exception as exc:
                self.job_log(job, traceback.format_exc(limit=3))
                self.finish(job, error=f"{type(exc).__name__}: {exc}")
        threading.Thread(target=target, daemon=True, name=f"job-{job.kind}").start()


S = State()


_PROJECTS_CACHE: dict = {"t": 0.0, "projects": set()}


def _running_projects() -> set[str]:
    """Compose projects of GRAPHIOT testbeds that Docker reports as running — including
    ones this server did not start (a script, or a server that was restarted)."""
    if time.monotonic() - _PROJECTS_CACHE["t"] < 3.0:
        return _PROJECTS_CACHE["projects"]
    from ..testbed import docker as dk
    try:
        out = dk.run(["ps", "--filter", "label=com.docker.compose.project", "--format",
                      "{{.Label \"com.docker.compose.project\"}}"], check=False, timeout=10).stdout
        projects = {p for p in out.split() if p.startswith("gi-")}
    except Exception:
        projects = set()
    _PROJECTS_CACHE.update(t=time.monotonic(), projects=projects)
    return projects


def _run_dir(run_id: str) -> Path:
    p = (RUNS / run_id).resolve()
    if not str(p).startswith(str(RUNS.resolve())) or not p.is_dir():
        raise HTTPException(404, f"run {run_id!r} not found")
    return p


def _read_jsonl_from(path: Path, offset: int, limit: int = 2000) -> tuple[list[dict], int]:
    if not path.exists():
        return [], offset
    out = []
    with open(path, "rb") as fh:
        fh.seek(offset)
        for line in fh:
            if not line.endswith(b"\n"):
                break                                  # partial last line: next poll
            offset += len(line)
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
            if len(out) >= limit:
                break
    return out, offset


def _df_records(df: pd.DataFrame, limit: int | None = None) -> list[dict]:
    if limit:
        df = df.tail(limit)
    return json.loads(df.to_json(orient="records", default_handler=str))


# --------------------------------------------------------------------------- #
class ConfigText(BaseModel):
    text: str


class NewRun(BaseModel):
    config_name: str | None = None
    text: str | None = None


class Secret(BaseModel):
    name: str
    value: str


def create_app() -> FastAPI:
    app = FastAPI(title="GRAPHIOT", version="0.1")
    for h in logging.getLogger().handlers:
        h.addFilter(RedactingFilter())
    STORE.load_dotenv()

    @app.on_event("startup")
    async def _startup():
        S.loop = asyncio.get_running_loop()

    # ---- configs ---------------------------------------------------------
    @app.get("/api/schema")
    def schema():
        return json_schema()

    @app.get("/api/configs")
    def configs():
        return [{"name": p.stem, "size": p.stat().st_size} for p in sorted(CONFIGS.glob("*.yaml"))]

    @app.get("/api/configs/{name}")
    def get_config(name: str):
        p = CONFIGS / f"{name}.yaml"
        if not p.exists():
            raise HTTPException(404, "no such config")
        return {"name": name, "text": p.read_text()}

    def _validate_text(text: str) -> dict:
        try:
            data = yaml.safe_load(text)
            cfg = parse_config(data)
        except yaml.YAMLError as exc:
            return {"ok": False, "errors": [{"loc": "yaml", "msg": str(exc)}]}
        except ValidationError as exc:
            return {"ok": False, "errors": [{"loc": ".".join(str(x) for x in e["loc"]), "msg": e["msg"]}
                                            for e in exc.errors()]}
        except (ValueError, ConfigError) as exc:
            return {"ok": False, "errors": [{"loc": "config", "msg": str(exc)}]}
        return {"ok": True, "hash": config_hash(cfg), "normalized": json.loads(cfg.model_dump_json()),
                "yaml": dump_yaml(cfg)}

    @app.post("/api/config/validate")
    def validate(body: ConfigText):
        return _validate_text(body.text)

    @app.post("/api/configs/{name}")
    def save_config(name: str, body: ConfigText):
        if not name.replace("_", "").replace("-", "").isalnum():
            raise HTTPException(400, "config name: letters, digits, - and _ only")
        v = _validate_text(body.text)
        if not v["ok"]:
            return JSONResponse(v, status_code=422)
        (CONFIGS / f"{name}.yaml").write_text(body.text)
        return {"ok": True, "hash": v["hash"]}

    # ---- secrets (names only ever leave the process) ---------------------
    @app.get("/api/secrets")
    def secrets():
        return {"names": STORE.names()}

    @app.post("/api/secrets")
    def set_secret(s: Secret):
        if not s.name.isidentifier():
            raise HTTPException(400, "invalid variable name")
        STORE.set(s.name, s.value)
        return {"ok": True, "names": STORE.names()}

    @app.delete("/api/secrets/{name}")
    def del_secret(name: str):
        STORE.clear(name)
        return {"ok": True, "names": STORE.names()}

    # ---- runs ------------------------------------------------------------
    @app.get("/api/runs")
    def runs():
        out = []
        for d in sorted(RUNS.glob("*/meta.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            meta = json.loads(d.read_text())
            rid = d.parent.name
            out.append({**{k: meta.get(k) for k in ("run_id", "created_utc", "testbed_backend",
                                                    "for_paper", "config_sha256", "git_commit")},
                        "run_id": rid,
                        "testbed_up": rid in S.testbeds,
                        "testbed_running": project_name(d.parent) in _running_projects(),
                        "campaign_running": bool(getattr(getattr(S.campaigns.get(rid), "state", None),
                                                         "running", False)),
                        "has_oracle": (d.parent / "oracle" / "summary.json").exists(),
                        "has_pilot": (d.parent / "results" / "pilot.json").exists()})
        return out

    @app.post("/api/runs")
    def new_run(body: NewRun):
        text = body.text if body.text else (CONFIGS / f"{body.config_name}.yaml").read_text()
        v = _validate_text(text)
        if not v["ok"]:
            return JSONResponse(v, status_code=422)
        run = pipeline.new_run(parse_config(yaml.safe_load(text)))
        return {"run_id": run.name}

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str):
        run = _run_dir(run_id)
        meta = json.loads((run / "meta.json").read_text())
        files = {}
        for sub in ("oracle", "results"):
            p = run / sub
            files[sub] = sorted(str(f.relative_to(run)) for f in p.rglob("*") if f.is_file()) if p.exists() else []
        camp = S.campaigns.get(run_id)
        return {"meta": meta, "config": (run / "config.yaml").read_text(), "files": files,
                "testbed_up": run_id in S.testbeds,
                "testbed_running": project_name(run) in _running_projects(),
                "campaign": camp.state.__dict__ if camp else None}

    # ---- testbed ---------------------------------------------------------
    @app.post("/api/runs/{run_id}/testbed/up")
    def tb_up(run_id: str):
        run = _run_dir(run_id)
        if run_id in S.testbeds:
            return {"ok": True, "already": True}
        job = S.new_job("testbed_up", run_id)

        def go():
            S.job_log(job, "bringing the testbed up (compose, deploy PLCs, capture)")
            tb = pipeline.testbed_up(run)
            S.testbeds[run_id] = tb
            for line in tb.log:
                S.job_log(job, line)
            S.emit({"type": "testbed", "run_id": run_id, "up": True})
            return json.loads((run / "testbed" / "up_report.json").read_text())
        S.run_thread(job, go)
        return {"job_id": job.id}

    @app.post("/api/runs/{run_id}/testbed/down")
    def tb_down(run_id: str):
        tb = S.testbeds.pop(run_id, None)
        if tb is None:
            run = _run_dir(run_id)
            if project_name(run) not in _running_projects():
                raise HTTPException(409, "testbed not up for this run")
            # Running but not started by this server session: stop it from its compose file.
            tb = Testbed(load_config(run / "config.yaml"), run)
            tb.compose_file = run / "testbed" / "docker-compose.yml"
        job = S.new_job("testbed_down", run_id)

        def go():
            camp = S.campaigns.get(run_id)
            if camp and camp.state.running:
                camp.stop()
            pipeline.testbed_down(tb)
            _PROJECTS_CACHE["t"] = 0.0
            S.emit({"type": "testbed", "run_id": run_id, "up": False})
            return {"ok": True}
        S.run_thread(job, go)
        return {"job_id": job.id}

    @app.get("/api/runs/{run_id}/testbed/health")
    def tb_health(run_id: str):
        tb = S.testbeds.get(run_id)
        if tb is None:
            return {"up": False}
        return {"up": True, **tb.health()}

    # ---- campaign --------------------------------------------------------
    @app.get("/api/runs/{run_id}/preflight")
    def preflight(run_id: str):
        from ..campaign.orchestrator import preflight_llms
        problems = preflight_llms(load_config(_run_dir(run_id) / "config.yaml"))
        return {"ok": not problems, "problems": problems}

    @app.post("/api/runs/{run_id}/campaign/start")
    async def camp_start(run_id: str):
        run = _run_dir(run_id)
        tb = S.testbeds.get(run_id)
        if tb is None:
            if project_name(run) in _running_projects():
                raise HTTPException(409, "this run's testbed was started outside this server session "
                                         "(its gateway admin token is unknown here): bring it down and up again")
            raise HTTPException(409, "bring the testbed up first")
        if run_id in S.campaigns and S.campaigns[run_id].state.running:
            raise HTTPException(409, "campaign already running")
        from ..campaign.orchestrator import Campaign, preflight_llms
        cfg = load_config(run / "config.yaml")
        problems = preflight_llms(cfg)
        if problems:
            raise HTTPException(409, "LLM preflight failed: " + "; ".join(problems))
        job = S.new_job("campaign", run_id)

        def on_event(e):
            S.emit({"type": "campaign", "run_id": run_id, "event": e})
            if e["kind"] in ("task_start", "task_end", "campaign_end", "campaign_error"):
                S.job_log(job, json.dumps({k: v for k, v in e.items() if k not in ("text", "ts")}))
        camp = Campaign(cfg, run, gateway_url=tb.gateway_url, admin_token=tb.admin_token,
                        root=ROOT, on_event=on_event)
        S.campaigns[run_id] = camp

        async def go():
            try:
                st = await camp.run()
                S.finish(job, result=st.__dict__)
            except Exception as exc:
                S.finish(job, error=f"{type(exc).__name__}: {exc}")
        asyncio.create_task(go())
        return {"job_id": job.id}

    @app.post("/api/runs/{run_id}/campaign/stop")
    def camp_stop(run_id: str):
        camp = S.campaigns.get(run_id)
        if not camp:
            raise HTTPException(409, "no campaign for this run")
        camp.stop()
        return {"ok": True}

    # ---- processing (oracle + pilot) --------------------------------------
    @app.post("/api/runs/{run_id}/process")
    def do_process(run_id: str, mode: str | None = None):
        run = _run_dir(run_id)
        job = S.new_job("process", run_id)

        def go():
            S.job_log(job, f"building tables, oracle ({mode or 'config'}), pilot go/no-go")
            out = pipeline.process(run, oracle_mode=mode)
            S.job_log(job, f"done in {out['seconds']} s")
            return out
        S.run_thread(job, go)
        return {"job_id": job.id}

    # ---- live streams (byte-offset tails) --------------------------------
    @app.get("/api/runs/{run_id}/live/{stream}")
    def live(run_id: str, stream: str, offset: int = 0, limit: int = 2000):
        files = {"tags": "capture/tags.jsonl", "gateway": "capture/gateway_full.jsonl",
                 "commands": "capture/commands.jsonl", "tasks": "groundtruth/tasks_full.jsonl",
                 "periods": "groundtruth/agent_periods.jsonl", "llm": "groundtruth/llm_log.jsonl",
                 "plant": "groundtruth/plant_state.jsonl"}
        if stream not in files:
            raise HTTPException(404, "unknown stream")
        rows, new = _read_jsonl_from(_run_dir(run_id) / files[stream], offset, limit)
        return {"rows": rows, "offset": new}

    @app.get("/api/plant")
    def plant():
        tags = load_tag_map(ROOT / "plant" / "tag_map.csv")
        lim = load_limits(ROOT / "plant" / "operating_limits.yaml")
        return {"tags": [{"name": t.name, "plc": t.plc.value, "stage": t.stage.value, "kind": t.kind.value,
                          "unit": t.unit, "vmin": t.vmin, "vmax": t.vmax, "iec": t.iec_address,
                          "table": t.table.value, "address": t.modbus_address} for t in tags.values()],
                "limits": {k: b.model_dump() for k, b in lim.outputs.items()}}

    # ---- oracle & results --------------------------------------------------
    @app.get("/api/runs/{run_id}/oracle")
    def oracle(run_id: str):
        run = _run_dir(run_id)
        o = run / "oracle"
        if not (o / "summary.json").exists():
            return {"available": False}
        out = {"available": True, "summary": json.loads((o / "summary.json").read_text()),
               "fidelity": json.loads((o / "fidelity.json").read_text())}
        for name in ("episode_labels", "violations", "command_labels"):
            p = o / f"{name}.parquet"
            out[name] = _df_records(pd.read_parquet(p)) if p.exists() else []
        return out

    @app.get("/api/runs/{run_id}/results")
    def results(run_id: str):
        run = _run_dir(run_id)
        r = run / "results"
        load = lambda name: json.loads((r / name).read_text()) if (r / name).exists() else None
        tables = {}
        for name in ("main_table", "ablation_table", "mcnemar", "attacker_table"):
            if (r / f"{name}.csv").exists():
                try:
                    tables[name] = _df_records(pd.read_csv(r / f"{name}.csv"))
                except pd.errors.EmptyDataError:
                    tables[name] = []
        files = sorted(str(f.relative_to(run)) for f in r.rglob("*") if f.is_file()) if r.exists() else []
        from .analysis import _deep
        return _deep({"pilot": load("pilot.json"), "evaluation": load("evaluation.json"),
                      "baselines": load("baselines.json"), "training": load("training.json"),
                      "edge_cost": load("edge_cost.json"), "attacker": load("attacker.json"),
                      "tables": tables, "files": files})

    @app.get("/api/runs/{run_id}/file")
    def get_file(run_id: str, path: str):
        run = _run_dir(run_id)
        f = (run / path).resolve()
        allowed = ("results/", "oracle/", "scores/", "config.yaml", "meta.json", "testbed/docker-compose.yml")
        rel = str(f.relative_to(run)) if str(f).startswith(str(run)) else ""
        if not rel or not any(rel == a or rel.startswith(a) for a in allowed) or not f.is_file():
            raise HTTPException(404, "not downloadable")
        if f.suffix == ".parquet":
            buf = io.StringIO()
            pd.read_parquet(f).to_csv(buf, index=False)
            return Response(buf.getvalue(), media_type="text/csv",
                            headers={"Content-Disposition": f'attachment; filename="{f.stem}.csv"'})
        return FileResponse(f, filename=f.name)

    @app.get("/api/runs/{run_id}/bundle.zip")
    def bundle(run_id: str):
        run = _run_dir(run_id)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in run.rglob("*"):
                rel = f.relative_to(run)
                if f.is_file() and (rel.parts[0] in ("results", "oracle")
                                    or (rel.parts[0] == "scores" and f.suffix != ".pt")
                                    or str(rel) in ("config.yaml", "meta.json")):
                    zf.write(f, f"{run_id}/{rel}")
        buf.seek(0)
        return StreamingResponse(buf, media_type="application/zip",
                                 headers={"Content-Disposition": f'attachment; filename="{run_id}.zip"'})

    # ---- jobs & websocket ---------------------------------------------------
    @app.get("/api/jobs")
    def jobs():
        return [j.public() for j in sorted(S.jobs.values(), key=lambda j: j.started, reverse=True)[:50]]

    @app.get("/api/system")
    def system():
        from ..testbed import docker as dk
        try:
            import torch
            dev = {"torch": torch.__version__, "cuda": torch.cuda.is_available(),
                   "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
        except Exception as exc:
            dev = {"torch": None, "error": str(exc)}
        return {"docker": dk.available(), "secrets": STORE.names(), "compute": dev}

    from .analysis import register as _register_analysis
    _register_analysis(app, S, _run_dir)

    @app.websocket("/api/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        S.clients.add(sock)
        try:
            while True:
                await sock.receive_text()        # keep-alive pings from the client
        except WebSocketDisconnect:
            S.clients.discard(sock)

    if DIST.exists():
        app.mount("/", StaticFiles(directory=DIST, html=True), name="gui")
    return app


app = create_app()
