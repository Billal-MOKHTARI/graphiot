"""Register/tag map for the battery-electrode cell.

One authoritative catalogue of every Modbus tag, authored here and emitted to
``plant/tag_map.csv`` (the CSV the paper requires "used by every component").
Every component loads the CSV through :func:`load_tag_map`; the CSV is the
contract, this module is how it is authored. ``tests/test_tag_map.py`` fails if
the committed CSV drifts from this source.

Modbus addressing follows OpenPLC Runtime v4 (``simple_modbus.py``):

====================  ===================  ============================  ======
IEC located area      Modbus table         Modbus address               access
====================  ===================  ============================  ======
``%IW`` int_input     input register        ``n``                        R (FC4)
``%QW`` int_output    holding register      ``n`` (0..1023)              R/W
``%MW`` int_memory    holding register      ``1024 + n``                 R/W
``%QX`` bool_output   coil                  ``a*8 + b`` for ``%QXa.b``   R/W
``%IX`` bool_input    discrete input        ``a*8 + b`` for ``%IXa.b``   R (FC2)
====================  ===================  ============================  ======

Register values are 16-bit integers, so an engineering value ``v`` is stored as
``round(v * scale)`` and read back as ``raw / scale`` (see :meth:`Tag.encode` /
:meth:`Tag.decode`). ``scale`` lives in the CSV so every component packs values
the same way. ``%IW`` sensors are written by the PLC's own Modbus-master I/O
polling the process twin (a separate field network), never by the agent
(DEVIATIONS D3).
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path


class Stage(str, Enum):
    MIXING = "mixing"
    COATING = "coating"
    SLITTING = "slitting"


class Plc(str, Enum):
    MIX = "PLC_MIX"
    COAT = "PLC_COAT"
    SLIT = "PLC_SLIT"


class Kind(str, Enum):
    """Role of a tag in the process (not the same as Modbus table)."""

    SETPOINT = "setpoint"   # agent-written target (%MW)
    SENSOR = "sensor"       # process measurement into the PLC (%IW)
    OUTPUT = "output"       # actuator command the PLC drives (%QW)
    PHASE = "phase"         # stage phase code (%QW), PLC-written
    STATUS = "status"       # other PLC status word (%QW)
    COMMAND = "command"     # agent-written boolean command (%QX)
    INTERLOCK = "interlock"  # plant condition into the PLC (%IX)
    ALARM = "alarm"         # PLC-raised boolean alarm (%QX)
    ALARM_ACK = "alarm_ack"  # agent-written acknowledge (%QX)


class Table(str, Enum):
    HOLDING = "holding"
    INPUT = "input"
    COIL = "coil"
    DISCRETE = "discrete"


class DType(str, Enum):
    INT = "int"
    BOOL = "bool"


# Phase codes shared by every stage (Stage node feature in Table II of the paper).
PHASES = {0: "idle", 1: "ramp", 2: "steady", 3: "transition"}
PHASE_CODE = {v: k for k, v in PHASES.items()}


@dataclass(frozen=True)
class Tag:
    """One Modbus tag. ``name`` is the stable identifier used everywhere."""

    name: str
    plc: Plc
    stage: Stage
    kind: Kind
    iec_address: str          # e.g. "%MW0", "%IW3", "%QX1.0"
    table: Table
    modbus_address: int       # address within that Modbus table
    dtype: DType
    scale: float              # engineering value = raw / scale
    unit: str
    vmin: float               # engineering lower bound (sensor/setpoint range)
    vmax: float               # engineering upper bound
    agent_readable: bool
    agent_writable: bool
    description: str

    def encode(self, value: float) -> int:
        """Engineering value -> 16-bit register word."""
        if self.dtype is DType.BOOL:
            return 1 if value else 0
        raw = int(round(value * self.scale))
        return max(-32768, min(32767, raw))

    def decode(self, raw: int) -> float:
        """Register word -> engineering value."""
        if self.dtype is DType.BOOL:
            return 1.0 if raw else 0.0
        return raw / self.scale


# IEC address helpers --------------------------------------------------------

def _iec_word(area: str, n: int) -> str:
    return f"%{area}{n}"


def _iec_bit(area: str, byte: int, bit: int) -> str:
    return f"%{area}{byte}.{bit}"


def modbus_address_of(table: Table, iec_address: str) -> int:
    """Map an IEC located address to its Modbus table address (OpenPLC v4)."""
    body = iec_address[1:]  # drop '%'
    area, rest = body[:2], body[2:]
    if area in ("IW", "QW", "MW"):
        n = int(rest)
        if area == "MW":
            return 1024 + n
        return n  # IW -> input register n ; QW -> holding register n
    if area in ("QX", "IX"):
        byte_s, bit_s = rest.split(".")
        return int(byte_s) * 8 + int(bit_s)
    raise ValueError(f"unsupported IEC address {iec_address!r}")


# ---------------------------------------------------------------------------
# Catalogue authoring helpers
# ---------------------------------------------------------------------------

_TAGS: list[Tag] = []


def _add(
    *, name, plc, stage, kind, iec_address, dtype, scale, unit, vmin, vmax,
    agent_readable, agent_writable, description,
):
    area = iec_address[1:3]
    table = {
        "IW": Table.INPUT, "QW": Table.HOLDING, "MW": Table.HOLDING,
        "QX": Table.COIL, "IX": Table.DISCRETE,
    }[area]
    _TAGS.append(Tag(
        name=name, plc=plc, stage=stage, kind=kind, iec_address=iec_address,
        table=table, modbus_address=modbus_address_of(table, iec_address),
        dtype=dtype, scale=scale, unit=unit, vmin=vmin, vmax=vmax,
        agent_readable=agent_readable, agent_writable=agent_writable,
        description=description,
    ))


def _stage_block(plc: Plc, stage: Stage, setpoints, sensors, outputs, specials):
    """Lay out one stage's tags into contiguous register blocks.

    ``setpoints`` -> %MW0.. (agent R/W), ``sensors`` -> %IW0.. (agent R, PLC
    gets them from field I/O), ``outputs`` -> %QW0.. (agent R), plus fixed
    status/phase and coil layout. Each PLC has its own Modbus address space.
    """
    for i, (nm, unit, scale, lo, hi, desc) in enumerate(setpoints):
        _add(name=f"{nm}", plc=plc, stage=stage, kind=Kind.SETPOINT,
             iec_address=_iec_word("MW", i), dtype=DType.INT, scale=scale,
             unit=unit, vmin=lo, vmax=hi, agent_readable=True,
             agent_writable=True, description=desc)
    for i, (nm, unit, scale, lo, hi, desc) in enumerate(sensors):
        _add(name=f"{nm}", plc=plc, stage=stage, kind=Kind.SENSOR,
             iec_address=_iec_word("IW", i), dtype=DType.INT, scale=scale,
             unit=unit, vmin=lo, vmax=hi, agent_readable=True,
             agent_writable=False, description=desc)
    for i, (nm, unit, scale, lo, hi, desc) in enumerate(outputs):
        _add(name=f"{nm}", plc=plc, stage=stage, kind=Kind.OUTPUT,
             iec_address=_iec_word("QW", i), dtype=DType.INT, scale=scale,
             unit=unit, vmin=lo, vmax=hi, agent_readable=True,
             agent_writable=False, description=desc)
    # Fixed status words: phase at %QW10, time-in-phase at %QW11, work order at %QW12.
    _add(name=f"{plc.value.split('_')[1].lower()}.phase", plc=plc, stage=stage,
         kind=Kind.PHASE, iec_address=_iec_word("QW", 10), dtype=DType.INT,
         scale=1, unit="code", vmin=0, vmax=3, agent_readable=True,
         agent_writable=False, description="stage phase: 0 idle,1 ramp,2 steady,3 transition")
    _add(name=f"{plc.value.split('_')[1].lower()}.time_in_phase", plc=plc, stage=stage,
         kind=Kind.STATUS, iec_address=_iec_word("QW", 11), dtype=DType.INT,
         scale=1, unit="s", vmin=0, vmax=32767, agent_readable=True,
         agent_writable=False, description="seconds elapsed in the current phase")
    _add(name=f"{plc.value.split('_')[1].lower()}.work_order", plc=plc, stage=stage,
         kind=Kind.STATUS, iec_address=_iec_word("QW", 12), dtype=DType.INT,
         scale=1, unit="id", vmin=0, vmax=32767, agent_readable=True,
         agent_writable=False, description="active work-order id loaded into the stage")
    # Coils / discrete inputs declared per stage.
    for nm, kind, iec, ar, aw, desc in specials:
        area = iec[1:3]
        _add(name=nm, plc=plc, stage=stage, kind=kind, iec_address=iec,
             dtype=DType.BOOL, scale=1, unit="bool", vmin=0, vmax=1,
             agent_readable=ar, agent_writable=aw, description=desc)


# ---------------------------------------------------------------------------
# The catalogue (see docs/PROCESS_DESIGN.md for the engineering rationale)
# ---------------------------------------------------------------------------

_stage_block(
    Plc.MIX, Stage.MIXING,
    setpoints=[
        ("mix.speed_sp", "rpm", 1, 0, 3500, "mixer/disperser speed setpoint"),
        ("mix.jacket_temp_sp", "degC", 10, 0, 60, "jacket temperature setpoint"),
        ("mix.dose_rate_sp", "pct_per_s", 100, 0, 10, "solids dosing rate setpoint"),
        ("mix.vacuum_sp", "kPa_abs", 10, 0, 101, "de-aeration vacuum setpoint"),
    ],
    sensors=[
        ("mix.viscosity", "mPa_s", 1, 0, 12000, "slurry viscosity at reference shear"),
        ("mix.solids_pct", "pct", 10, 0, 100, "slurry solids content"),
        ("mix.temp", "degC", 10, 0, 80, "slurry temperature"),
        ("mix.air_entrain", "pct", 10, 0, 100, "entrained-air proxy"),
        ("mix.speed_actual", "rpm", 1, 0, 3500, "measured mixer speed"),
    ],
    outputs=[
        ("mix.motor_cmd", "rpm", 1, 0, 3500, "motor speed command to drive"),
        ("mix.heater_cmd", "pct", 10, 0, 100, "jacket heater duty command"),
        ("mix.dose_cmd", "pct_per_s", 100, 0, 10, "solids feeder rate command"),
        ("mix.vacuum_cmd", "kPa_abs", 10, 0, 101, "vacuum pump/valve pressure command"),
    ],
    specials=[
        ("mix.run_cmd", Kind.COMMAND, "%QX0.0", True, True, "start/stop the mixer batch"),
        ("mix.vessel_closed", Kind.INTERLOCK, "%IX0.0", True, False, "vessel lid closed interlock"),
        ("mix.alarm", Kind.ALARM, "%QX1.0", True, False, "stage alarm raised by the PLC"),
        ("mix.alarm_ack", Kind.ALARM_ACK, "%QX2.0", True, True, "acknowledge the stage alarm"),
    ],
)

_stage_block(
    Plc.COAT, Stage.COATING,
    setpoints=[
        ("coat.web_speed_sp", "m_per_min", 10, 0, 25, "line/web speed setpoint"),
        ("coat.pump_flow_sp", "mL_per_min", 1, 0, 450, "slurry pump flow setpoint"),
        ("coat.gap_sp", "um", 1, 0, 450, "slot-die to web gap setpoint"),
        ("coat.dryer_z1_temp_sp", "degC", 10, 0, 180, "dryer zone 1 temperature setpoint"),
        ("coat.dryer_z2_temp_sp", "degC", 10, 0, 180, "dryer zone 2 temperature setpoint"),
        ("coat.dryer_z3_temp_sp", "degC", 10, 0, 180, "dryer zone 3 temperature setpoint"),
        ("coat.airflow_sp", "m_per_s", 10, 0, 18, "dryer nozzle airflow setpoint"),
        ("coat.tension_sp", "N", 1, 0, 350, "web tension setpoint"),
    ],
    sensors=[
        ("coat.dryer_z1_temp", "degC", 10, 0, 200, "dryer zone 1 air temperature"),
        ("coat.dryer_z2_temp", "degC", 10, 0, 200, "dryer zone 2 air temperature"),
        ("coat.dryer_z3_temp", "degC", 10, 0, 200, "dryer zone 3 air temperature"),
        ("coat.web_tension", "N", 1, 0, 400, "measured web tension"),
        ("coat.wet_thickness", "um", 1, 0, 400, "wet-film thickness"),
        ("coat.dry_loading", "mg_per_cm2", 100, 0, 50, "dry areal mass loading"),
        ("coat.residual_solvent", "pct", 10, 0, 100, "residual solvent in the film"),
        ("coat.web_speed_actual", "m_per_min", 10, 0, 25, "measured web speed"),
    ],
    outputs=[
        ("coat.drive_cmd", "m_per_min", 10, 0, 25, "web drive speed command"),
        ("coat.pump_cmd", "mL_per_min", 1, 0, 450, "pump flow command"),
        ("coat.z1_heat_cmd", "pct", 10, 0, 100, "zone 1 heater duty"),
        ("coat.z2_heat_cmd", "pct", 10, 0, 100, "zone 2 heater duty"),
        ("coat.z3_heat_cmd", "pct", 10, 0, 100, "zone 3 heater duty"),
        ("coat.fan_cmd", "m_per_s", 10, 0, 18, "dryer fan / nozzle airflow command"),
        ("coat.tension_cmd", "N", 1, 0, 350, "unwind brake / dancer tension command"),
        ("coat.gap_cmd", "um", 1, 0, 450, "slot-die gap actuator command"),
    ],
    specials=[
        ("coat.run_cmd", Kind.COMMAND, "%QX0.0", True, True, "start/stop coating"),
        ("coat.web_present", Kind.INTERLOCK, "%IX0.0", True, False, "web threaded interlock"),
        ("coat.alarm", Kind.ALARM, "%QX1.0", True, False, "stage alarm raised by the PLC"),
        ("coat.alarm_ack", Kind.ALARM_ACK, "%QX2.0", True, True, "acknowledge the stage alarm"),
    ],
)

_stage_block(
    Plc.SLIT, Stage.SLITTING,
    setpoints=[
        ("slit.line_speed_sp", "m_per_min", 10, 0, 45, "slitting line speed setpoint"),
        ("slit.tension_sp", "N", 1, 0, 300, "web tension setpoint"),
        ("slit.rewind_torque_sp", "Nm", 10, 0, 60, "rewind torque setpoint"),
    ],
    sensors=[
        ("slit.web_tension", "N", 1, 0, 350, "measured web tension"),
        ("slit.edge_burr", "um", 1, 0, 200, "edge-burr proxy"),
        ("slit.rewind_align", "mm", 100, 0, 20, "rewind telescoping/alignment proxy"),
        ("slit.line_speed_actual", "m_per_min", 10, 0, 45, "measured line speed"),
    ],
    outputs=[
        ("slit.drive_cmd", "m_per_min", 10, 0, 45, "line drive speed command"),
        ("slit.rewind_cmd", "Nm", 10, 0, 60, "rewind torque command"),
        ("slit.tension_cmd", "N", 1, 0, 300, "unwind tension command"),
        ("slit.knife_cmd", "bool", 1, 0, 1, "knife actuator command (interlocked in the PLC)"),
    ],
    specials=[
        ("slit.run_cmd", Kind.COMMAND, "%QX0.0", True, True, "start/stop slitting"),
        ("slit.knife_engage_cmd", Kind.COMMAND, "%QX0.1", True, True, "engage the slitting knives"),
        ("slit.tension_ok", Kind.INTERLOCK, "%IX0.0", True, False, "tension-established interlock"),
        ("slit.alarm", Kind.ALARM, "%QX1.0", True, False, "stage alarm raised by the PLC"),
        ("slit.alarm_ack", Kind.ALARM_ACK, "%QX2.0", True, True, "acknowledge the stage alarm"),
    ],
)

CATALOG: tuple[Tag, ...] = tuple(_TAGS)

CSV_FIELDS = [
    "name", "plc", "stage", "kind", "iec_address", "table", "modbus_address",
    "dtype", "scale", "unit", "vmin", "vmax", "agent_readable",
    "agent_writable", "description",
]


def _row(tag: Tag) -> dict:
    d = asdict(tag)
    for k in ("plc", "stage", "kind", "table", "dtype"):
        d[k] = d[k].value if isinstance(d[k], Enum) else d[k]
    d["agent_readable"] = int(tag.agent_readable)
    d["agent_writable"] = int(tag.agent_writable)
    return {k: d[k] for k in CSV_FIELDS}


def to_csv() -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS, lineterminator="\n")
    w.writeheader()
    for tag in CATALOG:
        w.writerow(_row(tag))
    return buf.getvalue()


def write_csv(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_csv())
    return path


def load_tag_map(path: str | Path) -> dict[str, Tag]:
    """Load the CSV into ``{name: Tag}`` — the component-facing entry point."""
    out: dict[str, Tag] = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            tag = Tag(
                name=r["name"], plc=Plc(r["plc"]), stage=Stage(r["stage"]),
                kind=Kind(r["kind"]), iec_address=r["iec_address"],
                table=Table(r["table"]), modbus_address=int(r["modbus_address"]),
                dtype=DType(r["dtype"]), scale=float(r["scale"]), unit=r["unit"],
                vmin=float(r["vmin"]), vmax=float(r["vmax"]),
                agent_readable=bool(int(r["agent_readable"])),
                agent_writable=bool(int(r["agent_writable"])),
                description=r["description"],
            )
            out[tag.name] = tag
    return out


def catalog_by_name() -> dict[str, Tag]:
    return {t.name: t for t in CATALOG}


if __name__ == "__main__":  # python -m graphiot.plant.tags > plant/tag_map.csv
    import sys
    sys.stdout.write(to_csv())
