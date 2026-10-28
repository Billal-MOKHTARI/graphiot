"""HTTP clients for the gateway.

:class:`ToolClient` is what *both* agents use — identical code, headers and
identity, so the gateway (and the monitor) cannot tell the operator from the
red team by anything but their tool calls (paper §V-A).

:class:`AdminClient` is the task system's client; it carries the admin token.
"""

from __future__ import annotations

import httpx


class ToolClient:
    def __init__(self, base_url: str, *, agent_id: str, timeout_s: float = 30.0):
        self.agent_id = agent_id
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout_s,
                                       headers={"User-Agent": "graphiot-agent/1"})

    async def call(self, tool: str, *, task_id: str, args: dict) -> dict:
        r = await self._http.post(f"/tools/{tool}",
                                  json={"task_id": task_id, "agent_id": self.agent_id, "args": args})
        if r.status_code == 404:
            return {"ok": False, "error": f"unknown tool {tool!r}", "code": "unknown_tool"}
        r.raise_for_status()
        return r.json()

    async def aclose(self) -> None:
        await self._http.aclose()


class AdminClient:
    def __init__(self, base_url: str, *, token: str, timeout_s: float = 30.0):
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout_s,
                                       headers={"X-Admin-Token": token})

    async def register_task(self, task: dict) -> dict:
        r = await self._http.post("/admin/tasks", json=task)
        r.raise_for_status()
        return r.json()

    async def health(self) -> dict:
        r = await self._http.get("/admin/health")
        r.raise_for_status()
        return r.json()

    async def aclose(self) -> None:
        await self._http.aclose()
