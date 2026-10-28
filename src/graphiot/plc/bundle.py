"""Build an OpenPLC Runtime v4 ``program.zip`` from a ``.st`` file, headlessly.

Bundle layout (extracted by the runtime into ``core/generated/``):

    generated.hpp, configuration.cpp, pou_*.cpp   STruC++ output
    generated_debug.cpp, debug-map.json            debugger tables (the shim needs them)
    defines.h                                      PROGRAM_MD5 for the runtime shim
    strucpp_runtime/include/*.hpp                  header-only STruC++ runtime
    conf/modbus_slave.json                         agent-facing Modbus TCP server (:502)
    conf/modbus_master.json                        field I/O polling of the process simulator

The runtime enables a plugin when ``conf/<plugin>.json`` is present. The field
I/O map is derived from the tag map, so the PLC reads exactly the sensors the
simulator serves and writes exactly the actuators it consumes:

    FC4  sim input registers 0..n-1   -> %IW0..   (sensors)
    FC2  sim discrete inputs 0..k-1   -> %IX0.0.. (interlocks)
    FC16 %QW0..m-1 -> sim holding registers 0..m-1 (actuators)
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ..plant.tags import Kind, Plc, Tag
from .build import _ROOT, strucpp_version

NODE_BIN = _ROOT / ".tools" / "node" / "bin" / "node"
COMPILE_SCRIPT = _ROOT / "plc" / "tools" / "compile_v4.mjs"


@dataclass(frozen=True)
class FieldIO:
    """Where this PLC's remote I/O (the process simulator) listens."""
    host: str
    port: int
    slave_id: int = 1
    cycle_time_ms: int = 100
    timeout_ms: int = 1000


def program_md5(st_source: str) -> str:
    h = hashlib.md5()
    h.update(st_source.encode())
    h.update(strucpp_version().encode())
    return h.hexdigest()


def _contiguous_words(tags: list[Tag], area: str) -> int:
    idx = sorted(int(t.iec_address[3:]) for t in tags)
    if idx and idx != list(range(len(idx))):
        raise ValueError(f"{area} block not contiguous from 0: {idx}")
    return len(idx)


def _contiguous_bits(tags: list[Tag]) -> int:
    pos = sorted(int(t.iec_address[3:].split(".")[0]) * 8 + int(t.iec_address.split(".")[1])
                 for t in tags)
    if pos and pos != list(range(len(pos))):
        raise ValueError(f"%IX block not contiguous from 0.0: {pos}")
    return len(pos)


def master_config(plc: Plc, tags: dict[str, Tag], field: FieldIO) -> list[dict]:
    mine = [t for t in tags.values() if t.plc is plc]
    sensors = [t for t in mine if t.kind is Kind.SENSOR]
    interlocks = [t for t in mine if t.kind is Kind.INTERLOCK]
    actuators = [t for t in mine if t.kind is Kind.OUTPUT]
    n_s = _contiguous_words(sensors, "%IW")
    n_a = _contiguous_words(actuators, "%QW")
    n_i = _contiguous_bits(interlocks)
    # The runtime's config model requires offsets as non-empty strings ("0x0000").
    points = []
    if n_s:
        points.append({"fc": 4, "offset": "0x0000", "iec_location": "%IW0", "len": n_s,
                       "cycle_time_ms": field.cycle_time_ms})
    if n_i:
        points.append({"fc": 2, "offset": "0x0000", "iec_location": "%IX0.0", "len": n_i,
                       "cycle_time_ms": field.cycle_time_ms})
    if n_a:
        points.append({"fc": 16, "offset": "0x0000", "iec_location": "%QW0", "len": n_a,
                       "cycle_time_ms": field.cycle_time_ms})
    return [{
        "name": f"{plc.value}_field_io",
        "protocol": "MODBUS",
        "config": {
            "type": "SLAVE", "transport": "tcp",
            "host": field.host, "port": field.port, "slave_id": field.slave_id,
            "cycle_time_ms": field.cycle_time_ms, "timeout_ms": field.timeout_ms,
            "io_points": points,
        },
    }]


def slave_config(port: int = 502) -> dict:
    # The segmented map must be explicit: without `buffer_mapping` the plugin
    # uses a "legacy" layout in which %MW is empty (observed on v4.3.1:
    # "%MW=1024-1023"), so setpoint writes would have nowhere to land. This is
    # the v3-compatible map plant/tag_map.csv assumes.
    return {
        "network_configuration": {"host": "0.0.0.0", "port": port},
        "buffer_mapping": {
            "holding_registers": {"qw_count": 1024, "mw_count": 1024,
                                  "md_count": 0, "ml_count": 0},
            "coils": {"qx_bits": 8192, "mx_bits": 0},
            "discrete_inputs": {"ix_bits": 8192},
            "input_registers": {"iw_count": 1024},
        },
    }


def build_bundle(st_path: str | Path, plc: Plc, tags: dict[str, Tag], field: FieldIO,
                 out_zip: str | Path, *, modbus_port: int = 502) -> dict:
    """Compile ``st_path`` and write ``out_zip``. Returns a manifest dict."""
    st_path = Path(st_path).resolve()   # the Node step runs with cwd=plc/tools
    source = st_path.read_text()
    md5 = program_md5(source)
    if not NODE_BIN.exists():
        raise FileNotFoundError(f"project Node not found at {NODE_BIN}; run ./start.sh")
    with tempfile.TemporaryDirectory(prefix="v4bundle_") as tmp:
        stage = Path(tmp) / "bundle"
        proc = subprocess.run(
            [str(NODE_BIN), str(COMPILE_SCRIPT), "--st", str(st_path),
             "--out", str(stage), "--md5", md5],
            capture_output=True, text=True, cwd=COMPILE_SCRIPT.parent,
        )
        try:
            summary = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            summary = {"ok": False, "raw": proc.stdout}
        if proc.returncode != 0 or not summary.get("ok"):
            raise RuntimeError(f"STruC++ compile failed for {st_path.name}: "
                               f"{summary.get('errors') or proc.stderr or proc.stdout}")
        (stage / "defines.h").write_text(
            f'#pragma once\n#define PROGRAM_MD5 "{md5}"\n')
        conf = stage / "conf"
        conf.mkdir()
        (conf / "modbus_slave.json").write_text(json.dumps(slave_config(modbus_port), indent=1))
        (conf / "modbus_master.json").write_text(
            json.dumps(master_config(plc, tags, field), indent=1))
        out_zip = Path(out_zip)
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(stage.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(stage).as_posix())
    return {"zip": str(out_zip), "md5": md5, "strucpp": summary.get("strucpp"),
            "files": summary.get("files"), "plc": plc.value,
            "field_io": {"host": field.host, "port": field.port}}
