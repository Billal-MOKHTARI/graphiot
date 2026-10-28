"""Run docker / docker compose, whether or not this session has the docker group.

After ``usermod -aG docker`` the group only appears in new login sessions. Until
then ``sg docker -c ...`` runs a command with the group. This helper detects
which form works once and uses it for every call, so the GUI and start.sh work
in both situations.
"""

from __future__ import annotations

import shlex
import subprocess
from functools import lru_cache


class DockerUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _mode() -> str:
    def ok(cmd: list[str]) -> bool:
        try:
            return subprocess.run(cmd, capture_output=True, timeout=20).returncode == 0
        except Exception:
            return False
    if ok(["docker", "info"]):
        return "direct"
    if ok(["sg", "docker", "-c", "docker info"]):
        return "sg"
    raise DockerUnavailable(
        "Docker is not reachable. Install Docker and add your user to the 'docker' group: "
        "sudo usermod -aG docker $USER (then log out and back in).")


def available() -> bool:
    try:
        _mode()
        return True
    except DockerUnavailable:
        return False


def run(args: list[str], *, check: bool = True, capture: bool = True,
        timeout: float | None = None, env: dict | None = None,
        cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run ``docker <args>``; returns the CompletedProcess (text mode)."""
    if _mode() == "direct":
        cmd = ["docker", *args]
    else:
        cmd = ["sg", "docker", "-c", shlex.join(["docker", *args])]
    proc = subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout,
                          env=env, cwd=cwd)
    if check and proc.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args[:3])} failed ({proc.returncode}):\n"
                           f"{(proc.stderr or '')[-2000:]}")
    return proc


def compose(project: str, compose_file: str, args: list[str], **kw) -> subprocess.CompletedProcess:
    return run(["compose", "-p", project, "-f", compose_file, *args], **kw)
