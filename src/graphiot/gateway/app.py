"""FastAPI surface of the gateway.

* ``GET  /tools``              — the six tool specifications
* ``POST /tools/{tool}``       — agent tool call: {task_id, agent_id, args}
* ``POST /admin/tasks``        — task system registers a task (admin token)
* ``GET  /admin/health``       — liveness + counters (admin token)

The admin token travels in the ``X-Admin-Token`` header. It is read from the
process environment and never logged.
"""

from __future__ import annotations

import hmac
import time

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .core import Gateway, GatewayError, TaskRecord
from .tools_spec import TOOL_NAMES, TOOL_SPECS


class ToolCall(BaseModel):
    task_id: str
    agent_id: str
    args: dict = Field(default_factory=dict)


class TaskIn(BaseModel):
    task_id: str
    task_type: str
    target_stage: str
    requested_tags: list[str] = Field(default_factory=list)
    work_order: dict = Field(default_factory=dict)
    agent_id: str
    assigned_ts: float | None = None


def build_app(gateway: Gateway, *, admin_token: str) -> FastAPI:
    if not admin_token:
        raise ValueError("gateway admin token is empty")
    app = FastAPI(title="GRAPHIOT edge gateway", version="1")
    started = time.time()

    def _admin(token: str | None):
        if not token or not hmac.compare_digest(token, admin_token):
            raise HTTPException(status_code=403, detail="admin token required")

    @app.get("/tools")
    def tools():
        return {"tools": TOOL_SPECS}

    @app.post("/tools/{tool}")
    async def call(tool: str, body: ToolCall):
        if tool not in TOOL_NAMES:
            raise HTTPException(status_code=404, detail=f"unknown tool {tool!r}")
        return await gateway.call(tool=tool, task_id=body.task_id, agent_id=body.agent_id,
                                  args=body.args)

    @app.post("/admin/tasks")
    async def register(task: TaskIn, x_admin_token: str | None = Header(default=None)):
        _admin(x_admin_token)
        rec = TaskRecord(task_id=task.task_id, task_type=task.task_type,
                         target_stage=task.target_stage, requested_tags=task.requested_tags,
                         work_order=task.work_order, agent_id=task.agent_id,
                         assigned_ts=task.assigned_ts or time.time())
        try:
            gateway.register_task(rec)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if "id" in task.work_order:
            try:
                await gateway.publish_work_order(task.target_stage, int(task.work_order["id"]))
            except GatewayError as exc:
                # The task is registered; the caller may retry (registration is idempotent).
                raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"ok": True}

    @app.get("/admin/health")
    def health(x_admin_token: str | None = Header(default=None)):
        _admin(x_admin_token)
        return {"ok": True, "uptime_s": round(time.time() - started, 1),
                "tasks": len(gateway.tasks)}

    return app
