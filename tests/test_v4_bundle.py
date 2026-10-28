"""OpenPLC Runtime v4 bundle: STruC++ output, defines.h, and plugin configs
derived from the tag map (the format verified live against v4.3.1)."""

import json
import zipfile

import pytest

from graphiot.plc.bundle import NODE_BIN, FieldIO, build_bundle, master_config, slave_config
from graphiot.plant.tags import Plc

pytestmark = pytest.mark.skipif(not NODE_BIN.exists(), reason="project Node not installed (run ./start.sh)")


@pytest.mark.parametrize("plc,src", [(Plc.MIX, "mixing.st"), (Plc.COAT, "coating.st"),
                                     (Plc.SLIT, "slitting.st")])
def test_bundle_layout(tmp_path, root, tags, plc, src):
    out = tmp_path / f"{plc.value}.zip"
    m = build_bundle(root / "plc" / "src" / src, plc, tags, FieldIO("10.77.3.50", 5022), out)
    names = set(zipfile.ZipFile(out).namelist())
    for required in ("generated.hpp", "defines.h", "generated_debug.cpp",
                     "conf/modbus_slave.json", "conf/modbus_master.json"):
        assert required in names, required
    assert any(n.startswith("strucpp_runtime/include/") for n in names)
    assert any(n.endswith(".cpp") and n.startswith("pou_") for n in names)
    defines = zipfile.ZipFile(out).read("defines.h").decode()
    assert m["md5"] in defines and "PROGRAM_MD5" in defines


def test_slave_config_exposes_mw_segment():
    # Without an explicit segmented map v4.3.1 leaves %MW empty (observed live).
    hr = slave_config()["buffer_mapping"]["holding_registers"]
    assert hr["qw_count"] == 1024 and hr["mw_count"] == 1024


def test_master_config_from_tag_map(tags):
    dev = master_config(Plc.COAT, tags, FieldIO("10.77.3.50", 5022))[0]
    pts = {p["fc"]: p for p in dev["config"]["io_points"]}
    assert pts[4]["iec_location"] == "%IW0" and pts[4]["len"] == 8      # 8 coating sensors
    assert pts[2]["iec_location"] == "%IX0.0" and pts[2]["len"] == 1    # web_present
    assert pts[16]["iec_location"] == "%QW0" and pts[16]["len"] == 8    # 8 actuators
    assert all(isinstance(p["offset"], str) and p["offset"] for p in pts.values())
    assert dev["config"]["host"] == "10.77.3.50" and dev["config"]["port"] == 5022
