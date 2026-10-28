"""Edge IIoT gateway: the six tools, the allow-list, Modbus southbound, logging.

Paper §III-A. Northbound the gateway exposes exactly six tools; southbound it
turns each call into Modbus transactions to the three PLCs. Every call is logged
with its task identifier, and every tag write is checked against the static
allow-list of the task's type before it is sent (baseline B1).

Two logs are written, on purpose:

* ``gateway_full.jsonl`` — audit log with full results (work-order contents,
  recipe names). Never read by the monitor.
* ``gateway_log.jsonl`` / ``commands.jsonl`` — the monitor's view: one record per
  call and one per write command, with the fields the graph needs (tool, tag,
  value, Modbus function code, decision) and no result payloads.

A scorer hook sees each write command *before* it is forwarded (the monitor's
edge deployment, §IV-B). It records a score and an alarm and never blocks.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Protocol

from ..plant.allow_list import AllowList
from ..plant.recipes import RecipeBook
from ..plant.tags import Kind, Plc, Stage, Tag, Table

TOOLS = ("read_tag", "write_tag", "read_work_order", "read_stage_status",
         "load_recipe", "acknowledge_alarm")

STAGE_PREFIX = {Stage.MIXING: "mix", Stage.COATING: "coat", Stage.SLITTING: "slit"}
PREFIX_STAGE = {v: k for k, v in STAGE_PREFIX.items()}
ACK_PULSE_S = 0.35   # > 3 PLC scans so the program sees the acknowledge


class ModbusPort(Protocol):
    """Southbound transport to one PLC (pymodbus client or a test double)."""
    async def read(self, table: str, address: int, count: int) -> list[int]: ...
    async def write(self, table: str, address: int, values: list[int]) -> None: ...


@dataclass
class TaskRecord:
    task_id: str
    task_type: str
    target_stage: str
    requested_tags: list[str]
    work_order: dict
    assigned_ts: float
    agent_id: str


@dataclass
class Command:
    """One tag write the gateway issued (what the model scores)."""
    ts: float
    call_id: int
    task_id: str
    agent_id: str
    tool: str
    tag: str
    value: float
    raw: int
    function_code: int
    decision: str            # forwarded | denied
    reason: str = ""
    score: float | None = None
    alarm: bool | None = None


class GatewayError(Exception):
    def __init__(self, message: str, *, code: str = "error"):
        super().__init__(message)
        self.code = code


Scorer = Callable[[Command], Awaitable[tuple[float, bool]] | tuple[float, bool]]


class Gateway:
    def __init__(self, *, tags: dict[str, Tag], allow_list: AllowList, recipes: RecipeBook,
                 ports: dict[Plc, ModbusPort], log_dir: str | Path | None = None,
                 scorer: Scorer | None = None, clock: Callable[[], float] = time.time):
        self.tags = tags
        self.allow = allow_list
        self.recipes = recipes
        self.ports = ports
        self.scorer = scorer
        self.clock = clock
        self.tasks: dict[str, TaskRecord] = {}
        self._call_ids = itertools.count(1)
        self._lock = asyncio.Lock()      # one southbound sequence at a time
        self.log_dir = Path(log_dir) if log_dir else None
        self._fh = {}
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            for name in ("gateway_full", "gateway_log", "commands", "tasks"):
                self._fh[name] = open(self.log_dir / f"{name}.jsonl", "a", buffering=1)

    # ------------------------------------------------------------------ admin
    def register_task(self, rec: TaskRecord) -> bool:
        """Called by the task system (MES role) — never by an agent. Idempotent: a
        retried registration of the same task is not logged twice (returns False)."""
        self.allow.policy(rec.task_type)   # raises on unknown task type
        if self.tasks.get(rec.task_id) == rec:
            return False
        self.tasks[rec.task_id] = rec
        # The monitor's task view: type, target stage, requested tags; NO values.
        self._emit("tasks", {"ts": rec.assigned_ts, "task_id": rec.task_id,
                             "task_type": rec.task_type, "target_stage": rec.target_stage,
                             "requested_tags": rec.requested_tags, "agent_id": rec.agent_id,
                             "work_order_id": rec.work_order.get("id")})
        return True

    async def publish_work_order(self, stage: str, wo_id: int) -> None:
        """MES action: show the active work order on the stage PLC (%QW12). Not an
        agent command, so it is logged in the audit log only."""
        tag = self.tags[f"{STAGE_PREFIX[self._stage(stage)]}.work_order"]
        rec = {"call_id": 0, "modbus": []}
        try:
            async with self._lock:
                await self._write_raw(tag, int(wo_id), rec)
        finally:   # failed attempts are audited too
            self._emit("gateway_full", {"ts": self.clock(), "admin": "publish_work_order",
                                        "stage": stage, "work_order": wo_id, "modbus": rec["modbus"]})

    # ------------------------------------------------------------------ tools
    async def call(self, *, tool: str, task_id: str, agent_id: str, args: dict) -> dict:
        call_id = next(self._call_ids)
        t0 = self.clock()
        rec = {"ts": t0, "call_id": call_id, "task_id": task_id, "agent_id": agent_id,
               "tool": tool, "args": args, "decision": "allowed", "reason": "ok",
               "modbus": [], "commands": []}
        try:
            if tool not in TOOLS:
                raise GatewayError(f"unknown tool {tool!r}", code="unknown_tool")
            task = self.tasks.get(task_id)
            if task is None:
                raise GatewayError(f"unknown task {task_id!r}", code="unknown_task")
            ok, reason = self.allow.decide(task.task_type, tool,
                                           args.get("name") if tool in ("read_tag", "write_tag") else None)
            if not ok:
                rec["decision"], rec["reason"] = "denied", reason
                if tool == "write_tag":
                    self._deny_command(rec, task, tool, args.get("name"), args.get("value"), reason)
                raise GatewayError(reason, code="denied")
            async with self._lock:
                result = await getattr(self, f"_t_{tool}")(task, args, rec)
            rec["result"] = result
            return {"ok": True, "call_id": call_id, "result": result}
        except GatewayError as exc:
            if rec["decision"] == "allowed":
                rec["decision"] = "error"
            rec["reason"] = str(exc)
            rec["error_code"] = exc.code
            return {"ok": False, "call_id": call_id, "error": str(exc), "code": exc.code}
        finally:
            rec["ts_response"] = self.clock()
            rec["latency_ms"] = round(1000 * (rec["ts_response"] - t0), 3)
            self._emit("gateway_full", rec)
            self._emit("gateway_log", self._monitor_projection(rec))

    # ---- tool implementations -------------------------------------------
    def _tag(self, name: str | None) -> Tag:
        if not name or name not in self.tags:
            raise GatewayError(f"unknown tag {name!r}", code="unknown_tag")
        return self.tags[name]

    async def _read_raw(self, tag: Tag, rec: dict) -> int:
        fc = {Table.COIL: 1, Table.DISCRETE: 2, Table.HOLDING: 3, Table.INPUT: 4}[tag.table]
        t = self.clock()
        try:
            vals = await self.ports[tag.plc].read(tag.table.value, tag.modbus_address, 1)
        except Exception as exc:
            rec["modbus"].append({"t": t, "plc": tag.plc.value, "fc": fc,
                                  "address": tag.modbus_address, "ok": False, "error": str(exc)})
            raise GatewayError(f"Modbus read failed for {tag.name}: {exc}", code="modbus") from exc
        raw = vals[0]
        if tag.dtype.value == "int" and raw > 32767:
            raw -= 65536
        rec["modbus"].append({"t": t, "plc": tag.plc.value, "fc": fc,
                              "address": tag.modbus_address, "count": 1, "ok": True})
        return raw

    async def _write_raw(self, tag: Tag, raw: int, rec: dict) -> int:
        fc = 5 if tag.table is Table.COIL else 6
        t = self.clock()
        try:
            await self.ports[tag.plc].write(tag.table.value, tag.modbus_address, [raw & 0xFFFF])
        except Exception as exc:
            rec["modbus"].append({"t": t, "plc": tag.plc.value, "fc": fc,
                                  "address": tag.modbus_address, "ok": False, "error": str(exc)})
            raise GatewayError(f"Modbus write failed for {tag.name}: {exc}", code="modbus") from exc
        rec["modbus"].append({"t": t, "plc": tag.plc.value, "fc": fc,
                              "address": tag.modbus_address, "values": [raw & 0xFFFF], "ok": True})
        return fc

    async def _issue_write(self, task: TaskRecord, tool: str, tag: Tag, value: float,
                           rec: dict) -> dict:
        if not (tag.vmin <= value <= tag.vmax):
            reason = (f"{tag.name}={value} outside the register range "
                      f"[{tag.vmin}, {tag.vmax}] {tag.unit}")
            rec["decision"], rec["reason"] = "denied", reason
            self._deny_command(rec, task, tool, tag.name, value, reason)
            raise GatewayError(reason, code="range")
        raw = tag.encode(value)
        cmd = Command(ts=self.clock(), call_id=rec["call_id"], task_id=task.task_id,
                      agent_id=rec["agent_id"], tool=tool, tag=tag.name, value=float(value),
                      raw=raw, function_code=5 if tag.table is Table.COIL else 6,
                      decision="forwarded")
        if self.scorer is not None:          # score BEFORE forwarding; never block
            out = self.scorer(cmd)
            if asyncio.iscoroutine(out):
                out = await out
            cmd.score, cmd.alarm = out
        await self._write_raw(tag, raw, rec)
        rec["commands"].append(asdict(cmd))
        self._emit("commands", asdict(cmd))
        return {"tag": tag.name, "value": tag.decode(raw), "unit": tag.unit}

    def _deny_command(self, rec: dict, task: TaskRecord, tool: str, name, value, reason: str):
        """A blocked write is still a command attempt (B1 scores it)."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = float("nan")
        tag = self.tags.get(name) if name else None
        raw = 0
        if tag and v == v:
            raw = tag.encode(v)   # encode() saturates to int16, safe for out-of-range values
        cmd = Command(ts=self.clock(), call_id=rec["call_id"], task_id=task.task_id,
                      agent_id=rec["agent_id"], tool=tool, tag=name or "",
                      value=v, raw=raw,
                      function_code=(5 if tag and tag.table is Table.COIL else 6),
                      decision="denied", reason=reason)
        rec["commands"].append(asdict(cmd))
        self._emit("commands", asdict(cmd))

    async def _t_read_tag(self, task: TaskRecord, args: dict, rec: dict) -> dict:
        tag = self._tag(args.get("name"))
        if not tag.agent_readable:
            raise GatewayError(f"{tag.name} is not readable", code="denied")
        raw = await self._read_raw(tag, rec)
        return {"tag": tag.name, "value": tag.decode(raw), "unit": tag.unit}

    async def _t_write_tag(self, task: TaskRecord, args: dict, rec: dict) -> dict:
        tag = self._tag(args.get("name"))
        if not tag.agent_writable:
            raise GatewayError(f"{tag.name} is not writable", code="denied")
        try:
            value = float(args.get("value"))
        except (TypeError, ValueError):
            raise GatewayError("write_tag needs a numeric 'value'", code="bad_args")
        return await self._issue_write(task, "write_tag", tag, value, rec)

    async def _t_read_work_order(self, task: TaskRecord, args: dict, rec: dict) -> dict:
        wo_id = args.get("id")
        if wo_id is None or str(wo_id) == str(task.work_order.get("id")):
            return dict(task.work_order)
        # Only the work order of a task registered with the gateway is visible.
        for other in self.tasks.values():
            if str(other.work_order.get("id")) == str(wo_id):
                return dict(other.work_order)
        raise GatewayError(f"unknown work order {wo_id!r}", code="not_found")

    def _stage(self, value: str | None) -> Stage:
        try:
            return Stage(value) if value in {s.value for s in Stage} else PREFIX_STAGE[value]
        except KeyError:
            raise GatewayError(f"unknown stage {value!r}; use mixing|coating|slitting",
                               code="bad_args") from None

    async def _t_read_stage_status(self, task: TaskRecord, args: dict, rec: dict) -> dict:
        stage = self._stage(args.get("stage"))
        p = STAGE_PREFIX[stage]
        phase = await self._read_raw(self.tags[f"{p}.phase"], rec)
        tip = await self._read_raw(self.tags[f"{p}.time_in_phase"], rec)
        wo = await self._read_raw(self.tags[f"{p}.work_order"], rec)
        alarm = await self._read_raw(self.tags[f"{p}.alarm"], rec)
        from ..plant.tags import PHASES
        return {"stage": stage.value, "phase": PHASES.get(phase, str(phase)),
                "time_in_phase_s": tip, "work_order": wo, "alarm": bool(alarm)}

    async def _t_load_recipe(self, task: TaskRecord, args: dict, rec: dict) -> dict:
        product = args.get("product")
        stage = self._stage(args.get("stage"))
        phase = args.get("phase", "steady")
        prod = self.recipes.products.get(product)
        if prod is None:
            raise GatewayError(f"unknown product {product!r}", code="not_found")
        setpoints = prod.setpoints(stage.value, phase)
        if not setpoints:
            raise GatewayError(f"no recipe for {product}/{stage.value}/{phase}", code="not_found")
        pol = self.allow.policy(task.task_type)
        blocked = [t for t in setpoints if not pol.may_write(t)]
        if blocked:  # atomic: all or nothing
            rec["decision"], rec["reason"] = "denied", f"recipe writes {blocked} not in allow-list"
            for name in blocked:
                self._deny_command(rec, task, "load_recipe", name, setpoints[name], rec["reason"])
            raise GatewayError(rec["reason"], code="denied")
        written = [await self._issue_write(task, "load_recipe", self.tags[t], v, rec)
                   for t, v in setpoints.items()]
        return {"product": product, "stage": stage.value, "phase": phase, "written": written}

    async def _t_acknowledge_alarm(self, task: TaskRecord, args: dict, rec: dict) -> dict:
        stage = self._stage(args.get("stage"))
        tag = self.tags[f"{STAGE_PREFIX[stage]}.alarm_ack"]
        pol = self.allow.policy(task.task_type)
        if not pol.may_write(tag.name):
            rec["decision"], rec["reason"] = "denied", f"{tag.name} not in allow-list"
            self._deny_command(rec, task, "acknowledge_alarm", tag.name, 1, rec["reason"])
            raise GatewayError(rec["reason"], code="denied")
        await self._issue_write(task, "acknowledge_alarm", tag, 1, rec)
        await asyncio.sleep(ACK_PULSE_S)
        await self._write_raw(tag, 0, rec)   # release the pulse (not a separate command)
        alarm = await self._read_raw(self.tags[f"{STAGE_PREFIX[stage]}.alarm"], rec)
        return {"stage": stage.value, "alarm_active": bool(alarm)}

    # ------------------------------------------------------------------ logs
    @staticmethod
    def _monitor_projection(rec: dict) -> dict:
        """What the monitor may see of a call: no args payloads beyond the tag,
        no results (work orders, recipe names)."""
        args = rec.get("args") or {}
        return {"ts": rec["ts"], "ts_response": rec.get("ts_response"), "call_id": rec["call_id"],
                "task_id": rec["task_id"], "agent_id": rec["agent_id"], "tool": rec["tool"],
                "tag": args.get("name"), "stage": args.get("stage"),
                "decision": rec["decision"], "error": rec.get("error_code"),
                "n_modbus": len(rec["modbus"]),
                "fcs": sorted({m["fc"] for m in rec["modbus"]})}

    def _emit(self, stream: str, record: dict) -> None:
        fh = self._fh.get(stream)
        if fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def close(self) -> None:
        for fh in self._fh.values():
            fh.close()
        self._fh.clear()
