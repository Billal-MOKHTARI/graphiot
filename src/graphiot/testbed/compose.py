"""Render the testbed's docker-compose.yml from an experiment config.

Topology (Purdue-style; see docs/PLAN.md §2):

* ``north``  agents/host ↔ gateway northbound API (not captured)
* ``cell``   gateway, HMI, EWS, the three PLCs — the network the monitor sees;
             its Linux bridge has a fixed name so the capture container can
             tcpdump it
* ``field``  PLCs ↔ process simulator (remote I/O); ``internal`` (no egress)

The rendered file is written into each run folder, so a run always stores the
exact topology it ran on. Services whose module does not exist yet are left
out, so ``up`` never starts a half-built service.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from ..config.schema import ExperimentConfig

_ROOT = Path(__file__).resolve().parents[3]
SERVICE_DOCKERFILE = _ROOT / "testbed" / "docker" / "Dockerfile.services"
SERVICE_REQS = _ROOT / "requirements" / "services.txt"

PLC_SERVICES = {"PLC_MIX": "plc_mix", "PLC_COAT": "plc_coat", "PLC_SLIT": "plc_slit"}

# service name -> python module it runs (included only when the module exists)
PY_SERVICES = {
    "field_io": "graphiot.sim.field_io",
    "gateway": "graphiot.gateway.service",
    "hmi": "graphiot.hmi.service",
    "ews": "graphiot.ews.service",
    "capture": "graphiot.capture.service",
}


def services_image_tag() -> str:
    h = hashlib.sha256(SERVICE_DOCKERFILE.read_bytes() + SERVICE_REQS.read_bytes()).hexdigest()[:12]
    return f"graphiot-services:{h}"


def _module_exists(module: str) -> bool:
    return (_ROOT / "src" / Path(*module.split("."))).with_suffix(".py").exists()


def render(cfg: ExperimentConfig, run_dir: str | Path, *, project: str) -> dict:
    net = cfg.testbed.network
    run_dir = Path(run_dir).resolve()
    src_mounts = [
        f"{_ROOT / 'src'}:/app/src:ro",
        f"{_ROOT / 'plant'}:/app/plant:ro",
        f"{_ROOT / 'configs'}:/app/configs:ro",
        f"{run_dir}:/data",
    ]
    plc_image = f"{cfg.testbed.openplc.image}@{cfg.testbed.openplc.digest}"
    svc_image = services_image_tag()

    services: dict[str, dict] = {}
    for plc, svc in PLC_SERVICES.items():
        services[svc] = {
            "image": plc_image,
            "hostname": svc.replace("_", "-"),
            "cap_add": ["SYS_NICE", "SYS_RESOURCE", "IPC_LOCK"],
            "ulimits": {"memlock": {"soft": -1, "hard": -1}},
            "networks": {
                "cell": {"ipv4_address": net.plc_ips[plc]},
                "field": {"ipv4_address": net.plc_field_ips[plc]},
            },
            "labels": {"graphiot.role": "plc", "graphiot.plc": plc},
            "restart": "no",
        }

    ports = net.field_io_ports
    py: dict[str, dict] = {
        "field_io": {
            "command": ["python", "-m", "graphiot.sim.field_io",
                        "--seed", str(cfg.testbed.sim_seed), "--host", "0.0.0.0",
                        "--ports", f"{ports['PLC_MIX']},{ports['PLC_COAT']},{ports['PLC_SLIT']}",
                        "--tick", str(cfg.testbed.plc_scan_ms / 1000.0),
                        "--state-log", "/data/groundtruth/plant_state.jsonl"],
            "networks": {"field": {"ipv4_address": net.field_io_ip}},
        },
        "gateway": {
            "command": ["python", "-m", "graphiot.gateway.service", "--config", "/data/config.yaml"],
            # Variable NAME only: compose takes the value from the environment of
            # the process running `compose up`, so the token never lands in this file.
            "environment": [cfg.gateway.admin_token_env],
            "networks": {"north": {"ipv4_address": net.gateway_north_ip},
                         "cell": {"ipv4_address": net.gateway_ip}},
            "cpus": cfg.testbed.gateway_cpus,
            "mem_limit": f"{int(cfg.testbed.gateway_mem_gb * 1024)}m",
        },
        "hmi": {
            "command": ["python", "-m", "graphiot.hmi.service", "--config", "/data/config.yaml"],
            "networks": {"cell": {"ipv4_address": net.hmi_ip}},
        },
        "ews": {
            "command": ["python", "-m", "graphiot.ews.service", "--config", "/data/config.yaml"],
            "networks": {"cell": {"ipv4_address": net.ews_ip}},
        },
        "capture": {
            "command": ["python", "-m", "graphiot.capture.service", "--config", "/data/config.yaml"],
            "network_mode": "host",
            "cap_add": ["NET_RAW", "NET_ADMIN"],
        },
    }
    for name, spec in py.items():
        if not _module_exists(PY_SERVICES[name]):
            continue
        services[name] = {
            "image": svc_image,
            "volumes": src_mounts,
            "labels": {"graphiot.role": name},
            "restart": "no",
            **spec,
        }

    return {
        "name": project,
        "networks": {
            "north": {"driver": "bridge",
                      "ipam": {"config": [{"subnet": net.north_subnet}]}},
            "cell": {"driver": "bridge",
                     "driver_opts": {"com.docker.network.bridge.name": net.cell_bridge},
                     "ipam": {"config": [{"subnet": net.cell_subnet}]}},
            "field": {"driver": "bridge", "internal": True,
                      "ipam": {"config": [{"subnet": net.field_subnet}]}},
        },
        "services": services,
    }


def write_compose(cfg: ExperimentConfig, run_dir: str | Path, *, project: str) -> Path:
    doc = render(cfg, run_dir, project=project)
    out = Path(run_dir) / "testbed" / "docker-compose.yml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Rendered by graphiot.testbed.compose — do not edit; edit the experiment YAML.\n"
                   + yaml.safe_dump(doc, sort_keys=False))
    return out
