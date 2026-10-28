"""Lockstep plant: the three compiled PLCs coupled to the physics model.

This is the dev backend's plant *and* the oracle's twin — one deterministic
engine. Each tick (default 100 ms = one PLC scan):

1. draw this tick's sensor noise;
2. write the (noisy) sensors and the interlock states into each PLC's input
   image (%IW / %IX), exactly what remote I/O would deliver;
3. step each PLC one scan (it computes its %QW actuator outputs);
4. read the actuator outputs and integrate the physics one dt.

Sensors therefore lag actuators by one scan (a real transport delay, and it
breaks the algebraic loop). Setpoints (%MW) and boolean commands (%QX) arrive
through :meth:`apply_command` — from the gateway when live, or from a logged
command sequence in :meth:`run_schedule` (replay and counterfactual replay).

Determinism: same seed + same command sequence ⇒ bit-identical traces. Noise is
counter-based, so removing a command changes only the physics it caused.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..plant.tags import Kind, Plc, Tag, load_tag_map
from ..plc.compiled import CompiledPLC
from .model import PlantPhysics, actuator_names
from .noise import NoiseField
from .physics import PhysicsParams

PLC_SOURCES = {
    Plc.MIX: "mixing.st",
    Plc.COAT: "coating.st",
    Plc.SLIT: "slitting.st",
}

_WRITABLE_KINDS = (Kind.SETPOINT, Kind.COMMAND, Kind.ALARM_ACK)


@dataclass(frozen=True)
class ScheduledCommand:
    """A timed setpoint/command for a deterministic run."""
    t: float          # seconds from the start of the run
    tag: str          # writable tag (setpoint, command or alarm acknowledge)
    value: float      # engineering value (booleans as 0/1)


class PlantSimulator:
    def __init__(self, *, tag_map_path: str | Path = "plant/tag_map.csv",
                 plc_src_dir: str | Path = "plc/src", seed: int = 0,
                 scan_s: float = 0.1, params: PhysicsParams | None = None,
                 actuator_delay_ticks: int = 0):
        self.tags: dict[str, Tag] = load_tag_map(tag_map_path)
        self.params = params or PhysicsParams()
        self.scan_s = scan_s
        self.seed = seed
        self.physics = PlantPhysics(self.params, NoiseField(seed), self.tags)
        src = Path(plc_src_dir)
        self.plcs: dict[Plc, CompiledPLC] = {
            plc: CompiledPLC(src / fname, scan_s=scan_s) for plc, fname in PLC_SOURCES.items()
        }
        self._actuators = actuator_names(self.tags)
        self._forced_interlocks: dict[str, bool] = {}
        # Extra PLC->plant latency of the live remote-I/O path (Modbus master polling),
        # in scan ticks; 0 = ideal lockstep. Calibrated by the oracle's fidelity check.
        self.actuator_delay = int(actuator_delay_ticks)
        self._act_queue: list[dict[str, float]] = []
        self.t = 0.0
        self.tick_i = 0

    # ---- tag access ------------------------------------------------------
    def read(self, name: str) -> float:
        t = self.tags[name]
        return t.decode(self.plcs[t.plc].read_iec(t.iec_address))

    def _write(self, name: str, eng: float) -> None:
        t = self.tags[name]
        self.plcs[t.plc].write_iec(t.iec_address, t.encode(eng))

    def apply_command(self, tag: str, value: float) -> None:
        """An agent/gateway write: setpoint (%MW) or boolean command (%QX)."""
        t = self.tags[tag]
        if t.kind not in _WRITABLE_KINDS:
            raise ValueError(f"{tag} ({t.kind.value}) is not a writable command tag")
        self._write(tag, value)

    def force_interlock(self, tag: str, value: bool | None) -> None:
        """Override a physics-computed interlock (scenario/test use); None clears it."""
        if value is None:
            self._forced_interlocks.pop(tag, None)
        else:
            self._forced_interlocks[tag] = bool(value)

    # ---- main tick -------------------------------------------------------
    def step(self) -> None:
        self.physics.tick_noise()
        for name, val in self.physics.sensor_values().items():
            self._write(name, val)
        interlocks = self.physics.interlock_values()
        interlocks.update(self._forced_interlocks)
        for name, val in interlocks.items():
            self._write(name, 1.0 if val else 0.0)
        for plc in self.plcs.values():
            plc.step(self.scan_s)
        act = {name: self.read(name) for name in self._actuators}
        if self.actuator_delay:
            self._act_queue.append(act)
            act = (self._act_queue.pop(0) if len(self._act_queue) > self.actuator_delay
                   else {name: 0.0 for name in self._actuators})
        self.physics.integrate(act, self.scan_s)
        self.t += self.scan_s
        self.tick_i += 1

    # ---- observation -----------------------------------------------------
    def snapshot(self, *, agent_only: bool = False) -> dict[str, float]:
        """Every tag (engineering units), as a Modbus client would read it."""
        return {name: self.read(name) for name, t in self.tags.items()
                if not agent_only or t.agent_readable}

    def true_state(self) -> dict[str, float]:
        """Noise-free physical state — an oracle-only twin output."""
        return self.physics.true_state()

    # ---- checkpoints (oracle counterfactual replay) ----------------------
    def save_state(self) -> dict:
        import copy
        ph = self.physics
        return {
            "plcs": {plc: p.save_state() for plc, p in self.plcs.items()},
            "physics": copy.deepcopy({"mix": ph.mix, "coat": ph.coat, "slit": ph.slit,
                                      "web_broken": ph.web_broken,
                                      "over_tension_s": ph._over_tension_s}),
            "noise": {"rng": {n: copy.deepcopy(r.bit_generator.state)
                              for n, r in ph.noise._rng.items()},
                      "cache": dict(ph.noise._cache), "k": ph.noise._k},
            "t": self.t, "tick_i": self.tick_i,
            "forced": dict(self._forced_interlocks),
            "act_queue": copy.deepcopy(self._act_queue),
        }

    def restore_state(self, s: dict) -> None:
        import copy
        import numpy as np
        for plc, st in s["plcs"].items():
            self.plcs[plc].restore_state(st)
        ph = self.physics
        p = copy.deepcopy(s["physics"])
        ph.mix, ph.coat, ph.slit = p["mix"], p["coat"], p["slit"]
        ph.web_broken, ph._over_tension_s = p["web_broken"], p["over_tension_s"]
        nz = ph.noise
        for name, state in s["noise"]["rng"].items():
            g = nz._rng.get(name) or np.random.Generator(np.random.PCG64())
            g.bit_generator.state = copy.deepcopy(state)
            nz._rng[name] = g
        nz._cache, nz._k = dict(s["noise"]["cache"]), s["noise"]["k"]
        self.t, self.tick_i = s["t"], s["tick_i"]
        self._forced_interlocks = dict(s["forced"])
        self._act_queue = copy.deepcopy(s.get("act_queue", []))

    # ---- deterministic driven run (twin / counterfactual) ----------------
    def run_schedule(self, commands: list[ScheduledCommand], duration_s: float, *,
                     sample_hz: float = 1.0, include_true_state: bool = False,
                     interlocks: dict[str, bool] | None = None) -> dict[str, list]:
        """Apply timed commands for ``duration_s``; return a sampled trace.

        A command at time t is applied before the first scan whose start time is
        ≥ t, matching how a Modbus write lands before the PLC's next scan."""
        for tag, val in (interlocks or {}).items():
            self.force_interlock(tag, val)
        cmds = sorted(commands, key=lambda c: c.t)
        ci = 0
        n_ticks = int(round(duration_s / self.scan_s))
        every = max(1, int(round((1.0 / sample_hz) / self.scan_s)))
        trace: dict[str, list] = {"t": []}
        for k in range(n_ticks):
            now = k * self.scan_s
            while ci < len(cmds) and cmds[ci].t <= now + 1e-9:
                self.apply_command(cmds[ci].tag, cmds[ci].value)
                ci += 1
            self.step()
            if k % every == 0:
                trace["t"].append(round(self.t, 3))
                for name, val in self.snapshot().items():
                    trace.setdefault(name, []).append(val)
                if include_true_state:
                    for name, val in self.true_state().items():
                        trace.setdefault("true:" + name, []).append(val)
        return trace

    def close(self) -> None:
        for plc in self.plcs.values():
            plc.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
