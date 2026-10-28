"""The three PLC programs compile under STruC++ and run under simulated time,
and two instances of one program stay independent (needed for the oracle twin)."""

import pytest

from graphiot.plc.compiled import CompiledPLC, parse_iec_address, modbus_to_iec

PROGRAMS = ["mixing.st", "coating.st", "slitting.st"]


@pytest.mark.parametrize("prog", PROGRAMS)
def test_compiles_and_steps(root, prog):
    plc = CompiledPLC(root / "plc" / "src" / prog)
    try:
        assert len(plc.located()) > 0
        for _ in range(10):
            plc.step(0.1)
    finally:
        plc.close()


def test_address_parsing():
    assert parse_iec_address("%MW0") == (2, 0, 0, 0)
    assert parse_iec_address("%IW3") == (0, 3, 0, 0)
    assert parse_iec_address("%QX1.0") == (1, 1, 0, 1)
    assert modbus_to_iec("holding", 1024) == (2, 0, 0, 0)   # %MW0
    assert modbus_to_iec("holding", 10) == (1, 10, 0, 0)    # %QW10
    assert modbus_to_iec("input", 3) == (0, 3, 0, 0)        # %IW3
    assert modbus_to_iec("coil", 16) == (1, 2, 0, 1)        # %QX2.0


def test_coating_overtemp_trip_and_phase(root, tags):
    plc = CompiledPLC(root / "plc" / "src" / "coating.st")

    def W(name, eng):
        t = tags[name]; plc.write_iec(t.iec_address, t.encode(eng))

    def R(name):
        t = tags[name]; return t.decode(plc.read_iec(t.iec_address))

    try:
        W("coat.run_cmd", 1); W("coat.web_present", 1)
        # Drive a zone past the trip; expect alarm set and that zone's heater cut.
        W("coat.dryer_z2_temp", 180.0)
        plc.step(0.1)
        assert R("coat.alarm") == 1.0
        assert R("coat.z2_heat_cmd") == 0.0
    finally:
        plc.close()


def test_instance_isolation(root):
    """Both instances share the generated global locatedVars[] only at
    construction; after the per-instance snapshot they must not interfere."""
    so = CompiledPLC(root / "plc" / "src" / "slitting.st").so_path
    a = CompiledPLC(so_path=so)
    b = CompiledPLC(so_path=so)
    try:
        a.write_iec("%QX0.0", 1)   # a.run_cmd = 1
        b.write_iec("%QX0.0", 0)   # b.run_cmd = 0
        for _ in range(60):
            a.step(0.1); b.step(0.1)
        # a advanced out of idle, b stayed in idle (phase %QW10)
        assert a.read_iec("%QW10") >= 1
        assert b.read_iec("%QW10") == 0
    finally:
        a.close(); b.close()
