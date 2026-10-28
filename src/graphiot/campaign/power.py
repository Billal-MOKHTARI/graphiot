"""Keep the host awake during a capture, and notice when it was not.

A laptop suspend freezes every container. On 26/10/2026 a 7-second suspend (lid
closed at 10:31:49) left two OpenPLC v4 runtimes unresponsive; the campaign only
failed three minutes later, on a gateway write. Campaigns therefore

* hold a systemd ``sleep:handle-lid-switch`` inhibitor while they run, and
* watch for suspends that happen anyway: ``CLOCK_BOOTTIME`` counts time spent
  suspended, ``CLOCK_MONOTONIC`` does not, so their difference grows by exactly
  the suspended time;
* check PLC liveness *passively*, from the age of each PLC's last record in the
  HMI's 1 Hz log. Probing the PLCs from the host would add an unknown host to the
  captured cell traffic.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path


class SleepInhibitor:
    """Blocks system sleep and lid-close suspend while active (no-op without systemd)."""

    def __init__(self, why: str, enabled: bool = True):
        self.why, self.enabled = why, enabled
        self._proc: subprocess.Popen | None = None
        self.active = False

    def start(self) -> "SleepInhibitor":
        exe = shutil.which("systemd-inhibit")
        if self.enabled and exe and self._proc is None:
            self._proc = subprocess.Popen(
                # Held by `tail --pid`: released when this process exits, even if it is killed.
                [exe, "--what=sleep:handle-lid-switch", "--who=GRAPHIOT", f"--why={self.why}",
                 "--mode=block", "tail", f"--pid={os.getpid()}", "-f", "/dev/null"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.2)
            self.active = self._proc.poll() is None
        return self

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        self.active = False

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def suspended_seconds() -> float:
    """Total time this machine has spent suspended since boot."""
    return time.clock_gettime(time.CLOCK_BOOTTIME) - time.monotonic()


class SuspendWatch:
    def __init__(self, threshold_s: float = 1.0):
        self.threshold_s = threshold_s
        self.base = suspended_seconds()

    def check(self) -> float:
        """Seconds suspended since the watch started (0.0 below the threshold)."""
        d = suspended_seconds() - self.base
        return d if d >= self.threshold_s else 0.0


def hmi_last_seen(tags_jsonl: str | Path, tail_bytes: int = 262_144) -> dict[str, float]:
    """Timestamp of the most recent HMI record of each PLC (from the end of the file)."""
    p = Path(tags_jsonl)
    if not p.exists():
        return {}
    with open(p, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - tail_bytes))
        lines = fh.read().splitlines()[1:] if size > tail_bytes else fh.read().splitlines()
    seen: dict[str, float] = {}
    for line in lines:
        try:
            r = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if "plc" in r and "ts" in r:
            seen[r["plc"]] = max(seen.get(r["plc"], 0.0), float(r["ts"]))
    return seen


def stale_plcs(tags_jsonl: str | Path, plcs: list[str], max_age_s: float, now: float | None = None) -> dict[str, float]:
    """PLCs whose last HMI record is older than ``max_age_s`` (name -> age in s)."""
    now = time.time() if now is None else now
    seen = hmi_last_seen(tags_jsonl)
    if not seen:
        return {}                       # HMI not started yet / no HMI on this backend
    return {p: round(now - seen.get(p, 0.0), 1) for p in plcs if now - seen.get(p, 0.0) > max_age_s}
