"""Bring the Docker testbed up, deploy the PLC programs, check health, tear down.

The OpenPLC runtime password comes from the environment variable named in the
config (``testbed.openplc.password_env``); if it is unset a random one is
generated for this testbed instance and kept in memory only. The runtime data
directory is not persisted, so a fresh ``up`` always starts from clean PLCs.
"""

from __future__ import annotations

import json
import os
import secrets
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from ..config.schema import ExperimentConfig
from ..plant.tags import Plc, load_tag_map
from ..plc.bundle import FieldIO, build_bundle
from ..plc.openplc_v4 import OpenPLCV4Client
from . import docker
from .compose import PLC_SERVICES, SERVICE_DOCKERFILE, services_image_tag, write_compose

_ROOT = Path(__file__).resolve().parents[3]
PLC_ST = {Plc.MIX: "mixing.st", Plc.COAT: "coating.st", Plc.SLIT: "slitting.st"}


def yaml_services(compose_file: Path) -> list[str]:
    import yaml
    return list((yaml.safe_load(compose_file.read_text().split("\n", 1)[1]) or {})
                .get("services", {}))


def project_name(run_dir: Path) -> str:
    """Compose project name: lowercase alnum/dash only."""
    base = "".join(ch if ch.isalnum() else "-" for ch in run_dir.name.lower()).strip("-")
    return f"gi-{base}"[:60]


@dataclass
class Testbed:
    cfg: ExperimentConfig
    run_dir: Path
    log: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.run_dir = Path(self.run_dir).resolve()
        self.project = project_name(self.run_dir)
        self.compose_file: Path | None = None
        self._password = os.environ.get(self.cfg.testbed.openplc.password_env) or \
            secrets.token_urlsafe(18)
        # Task-system token for the gateway's admin API; in memory and in the
        # compose process environment only.
        self.admin_token = os.environ.get(self.cfg.gateway.admin_token_env) or \
            secrets.token_urlsafe(24)

    def _env(self) -> dict:
        return {**os.environ, self.cfg.gateway.admin_token_env: self.admin_token}

    @property
    def gateway_url(self) -> str:
        net = self.cfg.testbed.network
        return f"http://{net.gateway_north_ip}:{self.cfg.gateway.north_port}"

    def _say(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log.append(f"[{stamp}] {msg}")

    # ---- images ----------------------------------------------------------
    def ensure_images(self) -> None:
        tag = services_image_tag()
        have = docker.run(["image", "inspect", tag], check=False).returncode == 0
        if not have:
            self._say(f"building {tag} (first run only)")
            docker.run(["build", "-t", tag, "-f", str(SERVICE_DOCKERFILE), str(_ROOT)],
                       timeout=1800)
        o = self.cfg.testbed.openplc
        ref = f"{o.image}@{o.digest}"
        if docker.run(["image", "inspect", ref], check=False).returncode != 0:
            self._say(f"pulling {ref}")
            docker.run(["pull", ref], timeout=1800)

    # ---- lifecycle -------------------------------------------------------
    def up(self) -> dict:
        if self.cfg.testbed.backend != "openplc_v4":
            raise ValueError("Testbed.up() is for the openplc_v4 backend")
        self._refuse_if_other_testbed_running()
        self._prune_stale_networks()
        self.ensure_images()
        if not (self.run_dir / "config.yaml").exists():
            raise FileNotFoundError(f"{self.run_dir}/config.yaml missing; create the run with "
                                    f"graphiot.config.io.create_run first")
        self.compose_file = write_compose(self.cfg, self.run_dir, project=self.project)
        rendered = yaml_services(self.compose_file)
        plant_side = [s for s in rendered if s in PLC_SERVICES.values() or s == "field_io"]
        hosts_side = [s for s in rendered if s in ("gateway", "hmi", "ews")]
        # Phase 1: controllers and the plant; deploy the programs.
        self._say(f"compose up plant side ({self.project}): {plant_side}")
        docker.compose(self.project, str(self.compose_file),
                       ["up", "-d", "--remove-orphans", *plant_side], timeout=600, env=self._env())
        deploy = self.deploy_plcs()
        health = self.health(wait_s=30)
        # Phase 2: cell hosts (gateway, HMI, EWS) once the PLCs answer Modbus.
        if hosts_side:
            self._say(f"compose up cell hosts: {hosts_side}")
            docker.compose(self.project, str(self.compose_file), ["up", "-d", *hosts_side],
                           timeout=300, env=self._env())
            health["gateway"] = self.wait_gateway() if "gateway" in hosts_side else None
        report = {"project": self.project, "compose": str(self.compose_file),
                  "deploy": deploy, "health": health}
        (self.run_dir / "testbed" / "up_report.json").write_text(json.dumps(report, indent=2))
        return report

    def wait_gateway(self, timeout_s: float = 60.0) -> dict:
        import httpx
        deadline = time.monotonic() + timeout_s
        last = None
        while time.monotonic() < deadline:
            try:
                r = httpx.get(f"{self.gateway_url}/admin/health",
                              headers={"X-Admin-Token": self.admin_token}, timeout=2.0)
                if r.status_code == 200:
                    return r.json()
                last = f"HTTP {r.status_code}"
            except httpx.HTTPError as exc:
                last = str(exc)
            time.sleep(1.0)
        return {"ok": False, "error": last}

    def down(self) -> None:
        compose_file = self.compose_file or (self.run_dir / "testbed" / "docker-compose.yml")
        if compose_file.exists():
            self._say("compose down")
            docker.compose(self.project, str(compose_file), ["down", "--remove-orphans", "-t", "5"],
                           timeout=300, check=False)

    def _prune_stale_networks(self) -> None:
        """Remove GRAPHIOT compose networks with no attached container. A testbed brought
        down after a half-finished ``up`` (or stopped outside compose) can leave its
        ``gi-*`` networks behind; they keep reserving the fixed cell/field/north subnets,
        so the next ``up`` fails with 'Pool overlaps with other one on this address space'.
        Only empty networks are removed (a running testbed keeps its own)."""
        ids = docker.run(["network", "ls", "--filter", "name=gi-", "-q"], check=False).stdout.split()
        for nid in ids:
            info = docker.run(["network", "inspect", nid, "--format",
                               "{{.Name}}\t{{len .Containers}}"], check=False).stdout.strip()
            if not info:
                continue
            name, _, n = info.partition("\t")
            if n.strip() in ("0", "") and name.startswith("gi-"):
                r = docker.run(["network", "rm", nid], check=False)
                if r.returncode == 0:
                    self._say(f"removed stale network {name}")

    def _refuse_if_other_testbed_running(self) -> None:
        out = docker.run(["ps", "--filter", "label=graphiot.role=plc", "--format",
                          "{{.Label \"com.docker.compose.project\"}}"], check=False).stdout
        others = {p for p in out.split() if p and p != self.project}
        if others:
            raise RuntimeError(f"another GRAPHIOT testbed is running ({', '.join(sorted(others))}); "
                               f"stop it first (the cell bridge name is shared)")

    # ---- PLC deployment --------------------------------------------------
    def deploy_plcs(self) -> dict:
        net = self.cfg.testbed.network
        tags = load_tag_map(_ROOT / self.cfg.paths.tag_map)
        bundle_dir = self.run_dir / "testbed" / "bundles"
        results = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # self-signed runtime certificate
            for plc, st in PLC_ST.items():
                field_io = FieldIO(net.field_io_ip, net.field_io_ports[plc.value],
                                   cycle_time_ms=self.cfg.testbed.plc_scan_ms)
                manifest = build_bundle(_ROOT / self.cfg.paths.plc_src_dir / st, plc, tags, field_io,
                                        bundle_dir / f"{plc.value}.zip",
                                        modbus_port=self.cfg.testbed.openplc.modbus_port)
                client = OpenPLCV4Client(f"https://{net.plc_ips[plc.value]}:8443",
                                         self.cfg.testbed.openplc.admin_user, self._password)
                self._say(f"deploying {plc.value} ({manifest['md5'][:8]})")
                try:
                    out = client.deploy(bundle_dir / f"{plc.value}.zip")
                finally:
                    client.close()
                results[plc.value] = {"md5": manifest["md5"], "compilation": out["compilation"],
                                      "status": out.get("status", {}).get("status")}
        return results

    # ---- health ----------------------------------------------------------
    def health(self, wait_s: float = 0.0) -> dict:
        """Agent-facing Modbus reachability and field-loop liveness. Each probe is a
        sensor that is > 0 at rest (slurry temperature, dryer air temperature,
        edge-burr baseline), so a non-zero read through the PLC proves its Modbus
        master is receiving the simulator's sensors."""
        from pymodbus.client import ModbusTcpClient

        net = self.cfg.testbed.network
        tags = load_tag_map(_ROOT / self.cfg.paths.tag_map)
        probe = {Plc.MIX: "mix.temp", Plc.COAT: "coat.dryer_z1_temp", Plc.SLIT: "slit.edge_burr"}
        deadline = time.monotonic() + wait_s
        out: dict = {}
        while True:
            out = {}
            for plc in PLC_ST:
                ip = net.plc_ips[plc.value]
                entry = {"ip": ip}
                cl = ModbusTcpClient(ip, port=self.cfg.testbed.openplc.modbus_port, timeout=2)
                try:
                    entry["modbus"] = cl.connect()
                    if entry["modbus"]:
                        t = tags[probe[plc]]
                        r = cl.read_input_registers(t.modbus_address, count=1)
                        entry["field_loop"] = (not r.isError()) and r.registers[0] > 0
                        ph = cl.read_holding_registers(tags[f"{probe[plc].split('.')[0]}.phase"].modbus_address, count=1)
                        entry["phase"] = None if ph.isError() else ph.registers[0]
                except Exception as exc:  # report, don't raise
                    entry["error"] = str(exc)
                finally:
                    cl.close()
                out[plc.value] = entry
            healthy = all(e.get("modbus") and e.get("field_loop") for e in out.values())
            if healthy or time.monotonic() >= deadline:
                break
            time.sleep(2.0)
        ps = docker.compose(self.project, str(self.compose_file or self.run_dir / "testbed/docker-compose.yml"),
                            ["ps", "--format", "json"], check=False).stdout
        containers = []
        for line in ps.splitlines():
            try:
                c = json.loads(line)
                containers.append({"service": c.get("Service"), "state": c.get("State"),
                                   "status": c.get("Status")})
            except json.JSONDecodeError:
                pass
        return {"plcs": out, "containers": containers,
                "healthy": all(e.get("modbus") and e.get("field_loop") for e in out.values())}
