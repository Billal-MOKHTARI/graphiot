"""End-to-end steps of an experiment, shared by the GUI backend and the CLI.

    run = new_run(config)            create runs/<name>_<utc>/ with config + meta
    tb  = testbed_up(run)            compose up, deploy PLCs, start capture
    await run_campaign(run, tb)      benign operation + red-team takeovers
    testbed_down(tb)                 stop capture and containers
    process(run)                     Parquet tables -> oracle -> pilot go/no-go
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .campaign.orchestrator import Campaign
from .config.io import create_run, load_config
from .config.schema import ExperimentConfig
from .data.convert import build_groundtruth_tables, build_monitor_tables
from .oracle.core import fallback_oracle, primary_oracle, write_result
from .testbed import docker
from .testbed.manager import Testbed
from .utils.secrets import redact

ROOT = Path(__file__).resolve().parents[2]


def new_run(cfg: ExperimentConfig | str | Path) -> Path:
    if not isinstance(cfg, ExperimentConfig):
        cfg = load_config(cfg)
    return create_run(cfg)


def testbed_up(run: Path) -> Testbed:
    cfg = load_config(run / "config.yaml")
    tb = Testbed(cfg, run)
    tb.up()
    start_capture(tb)
    return tb


def start_capture(tb: Testbed, timeout_s: float = 30.0) -> None:
    """Start tcpdump and return only once it is capturing: `compose up` returns when
    the container starts, before tcpdump has opened the interface, and a campaign that
    starts in that gap loses its first writes (observed: 12/13 commands on the wire)."""
    if not tb.cfg.campaign.capture.pcap:
        return
    pcap_dir = tb.run_dir / "capture" / "pcap"
    docker.compose(tb.project, str(tb.compose_file), ["up", "-d", "capture"], timeout=120, env=tb._env())
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if any(p.stat().st_size > 0 for p in pcap_dir.glob("*.pcap")):
            time.sleep(1.0)                      # tcpdump writes the header first; let it settle
            return
        time.sleep(0.5)
    raise RuntimeError(f"capture did not start within {timeout_s} s (no pcap in {pcap_dir})")


def testbed_down(tb: Testbed) -> None:
    # Stop capture first (SIGTERM lets tcpdump flush and hand files back), then keep the
    # containers' logs (a gateway traceback is otherwise lost with the containers).
    if tb.compose_file:
        docker.compose(tb.project, str(tb.compose_file), ["stop", "-t", "10", "capture"],
                       timeout=120, check=False)
        logs = docker.compose(tb.project, str(tb.compose_file),
                              ["logs", "--no-color", "--timestamps", "--tail", "5000"],
                              timeout=120, check=False)
        (tb.run_dir / "logs").mkdir(exist_ok=True)
        (tb.run_dir / "logs" / "compose.log").write_text(redact(logs.stdout or ""))
    tb.down()


async def run_campaign(run: Path, tb: Testbed, *, on_event=None, cfg: ExperimentConfig | None = None):
    cfg = cfg or load_config(run / "config.yaml")
    camp = Campaign(cfg, run, gateway_url=tb.gateway_url, admin_token=tb.admin_token, root=ROOT,
                    on_event=on_event)
    return camp, await camp.run()


def process(run: Path, *, oracle_mode: str | None = None, pilot: bool = True) -> dict:
    from .eval.pilot import run_pilot
    cfg = load_config(run / "config.yaml")
    t0 = time.time()
    build_monitor_tables(run, idle_timeout_s=cfg.campaign.capture.flow_idle_timeout_s,
                         active_timeout_s=cfg.campaign.capture.flow_active_timeout_s)
    build_groundtruth_tables(run)
    mode = oracle_mode or cfg.oracle.mode
    res = primary_oracle(run, cfg, root=ROOT) if mode == "primary" else fallback_oracle(run, cfg, root=ROOT)
    write_result(res, run)
    out = {"oracle": res.summary, "seconds": round(time.time() - t0, 1)}
    if pilot:
        out["pilot"] = run_pilot(run, cfg, root=ROOT)
    (run / "results" / "process.json").write_text(json.dumps(out, indent=1, default=str))
    return out
