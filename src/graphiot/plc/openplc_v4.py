"""REST client for OpenPLC Runtime v4 (deploys a program.zip headlessly).

Follows the runtime's documented API (docs/API.md, v4.3.1): create the first
user, log in for a JWT, upload the bundle, poll the compilation, start the PLC.
The runtime serves a self-signed certificate, so TLS verification is off for
this client only (testbed traffic on a private bridge network).

Credentials come from the caller (the testbed config / .env), never from YAML
results or logs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import httpx


class DeployError(RuntimeError):
    pass


@dataclass
class OpenPLCV4Client:
    base_url: str               # e.g. https://10.77.2.101:8443
    username: str
    password: str
    timeout_s: float = 30.0

    def __post_init__(self):
        self._http = httpx.Client(base_url=self.base_url, verify=False, timeout=self.timeout_s)
        self._token: str | None = None

    # -- auth ---------------------------------------------------------------
    def ensure_user(self) -> None:
        """Create the first user if the runtime has none (idempotent)."""
        r = self._http.post("/api/create-user", json={
            "username": self.username, "password": self.password, "role": "admin"})
        if r.status_code not in (200, 201, 401, 409):
            raise DeployError(f"create-user: HTTP {r.status_code} {r.text[:200]}")

    def login(self) -> None:
        r = self._http.post("/api/login", json={"username": self.username,
                                                "password": self.password})
        if r.status_code != 200:
            raise DeployError(f"login: HTTP {r.status_code} {r.text[:200]}")
        self._token = r.json()["access_token"]
        self._http.headers["Authorization"] = f"Bearer {self._token}"

    # -- control ------------------------------------------------------------
    def wait_ready(self, timeout_s: float = 120.0) -> None:
        """Wait until the REST server answers (the container boots in ~10-30 s)."""
        deadline = time.monotonic() + timeout_s
        last = None
        while time.monotonic() < deadline:
            try:
                r = self._http.post("/api/login", json={"username": "_", "password": "_"})
                if r.status_code in (200, 401, 400, 500):
                    return
            except httpx.HTTPError as exc:
                last = exc
            time.sleep(1.0)
        raise DeployError(f"runtime at {self.base_url} not ready: {last}")

    def status(self) -> dict:
        return self._http.get("/api/status").json()

    def start(self) -> dict:
        return self._http.get("/api/start-plc").json()

    def stop(self) -> dict:
        return self._http.get("/api/stop-plc").json()

    def logs(self) -> dict:
        return self._http.get("/api/runtime-logs").json()

    def upload(self, zip_path: str | Path) -> dict:
        with open(zip_path, "rb") as fh:
            r = self._http.post("/api/upload-file",
                                files={"file": ("program.zip", fh, "application/zip")})
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if r.status_code >= 400 or body.get("UploadFileFail"):
            raise DeployError(f"upload: HTTP {r.status_code} {body or r.text[:300]}")
        return body

    def wait_compiled(self, timeout_s: float = 300.0, poll_s: float = 2.0) -> dict:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            st = self._http.get("/api/compilation-status").json()
            if st.get("status") in ("SUCCESS", "FAILED"):
                if st["status"] == "FAILED":
                    tail = "\n".join(st.get("logs", [])[-25:])
                    raise DeployError(f"compilation FAILED:\n{tail}")
                return st
            time.sleep(poll_s)
        raise DeployError("compilation timed out")

    # -- one-shot -----------------------------------------------------------
    def deploy(self, zip_path: str | Path, *, start: bool = True) -> dict:
        self.wait_ready()
        self.ensure_user()
        self.login()
        self.upload(zip_path)
        comp = self.wait_compiled()
        out = {"compilation": comp.get("status"), "log_tail": comp.get("logs", [])[-6:]}
        if start:
            out["start"] = self.start()
            time.sleep(1.0)
            out["status"] = self.status()
        return out

    def close(self) -> None:
        self._http.close()
