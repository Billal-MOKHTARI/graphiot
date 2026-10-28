"""Pure process physics, driven only by actuator commands.

The model sees exactly what real field I/O would see: the PLC's actuator
outputs (``Kind.OUTPUT`` tags, ``%QW0..``) come in, sensor values (``%IW``) and
discrete interlock states (``%IX``) go out. It never reads a setpoint, so the
same model serves both couplings:

* the dev backend and the oracle twin (:class:`graphiot.sim.plant.PlantSimulator`),
  which step the compiled ST logic in lockstep;
* the Docker field-I/O server (:mod:`graphiot.sim.field_io`), where OpenPLC
  runs the logic and polls this model over Modbus.

Units are engineering units throughout; packing to registers happens at the
I/O boundary via the tag map's ``scale``.
"""

from __future__ import annotations

import math

from ..plant.tags import Kind, Plc, Tag
from .noise import NoiseField
from .physics import PhysicsParams


def actuator_names(tags: dict[str, Tag], plc: Plc | None = None) -> list[str]:
    return [n for n, t in tags.items() if t.kind is Kind.OUTPUT and (plc is None or t.plc is plc)]


def sensor_names(tags: dict[str, Tag], plc: Plc | None = None) -> list[str]:
    return [n for n, t in tags.items() if t.kind is Kind.SENSOR and (plc is None or t.plc is plc)]


def interlock_names(tags: dict[str, Tag], plc: Plc | None = None) -> list[str]:
    return [n for n, t in tags.items() if t.kind is Kind.INTERLOCK and (plc is None or t.plc is plc)]


class PlantPhysics:
    """State + one-step integration of the three stages."""

    WEB_BREAK_N = 340.0       # coating web breaks above this tension ...
    WEB_BREAK_DWELL_S = 1.0   # ... sustained for this long

    def __init__(self, params: PhysicsParams, noise: NoiseField, tags: dict[str, Tag]):
        self.p = params
        self.noise = noise
        self._sensor_names = sensor_names(tags)
        self.mix: dict[str, float] = {}
        self.coat: dict[str, float] = {}
        self.slit: dict[str, float] = {}
        self.web_broken = False
        self._over_tension_s = 0.0
        self.reset()

    def reset(self) -> None:
        p = self.p
        # Initial state must be a fixed point both with no PLC program loaded (all
        # actuators 0: e.g. vacuum_cmd 0 = full vacuum) and with the idle program
        # (vented): a fresh mixer holds no entrained air. Found by the fidelity check.
        self.mix = dict(speed=0.0, temp=p.mix.ambient, solids=58.0, air=0.0,
                        viscosity=p.mix.visc_base)
        self.coat = dict(web_speed=0.0, z1=p.coat.ambient, z2=p.coat.ambient,
                         z3=p.coat.ambient, tension=0.0, wet=0.0, loading=0.0,
                         residual=0.0)
        self.slit = dict(line_speed=0.0, tension=0.0, burr=p.slit.burr_base, align=0.0)
        self.web_broken = False
        self._over_tension_s = 0.0

    # ---- noise ----------------------------------------------------------
    def tick_noise(self) -> None:
        """Draw this tick's noise for every sensor (exactly once per tick)."""
        self.noise.tick(self._sensor_names)

    def _n(self, name: str) -> float:
        return self.noise.value(name, self.p.noise_sigma.get(name, 0.0))

    # ---- outputs of the model -------------------------------------------
    def sensor_values(self) -> dict[str, float]:
        m, c, s = self.mix, self.coat, self.slit
        n = self._n
        return {
            "mix.viscosity": m["viscosity"] + n("mix.viscosity"),
            "mix.solids_pct": m["solids"] + n("mix.solids_pct"),
            "mix.temp": m["temp"] + n("mix.temp"),
            "mix.air_entrain": m["air"] + n("mix.air_entrain"),
            "mix.speed_actual": m["speed"] + n("mix.speed_actual"),
            "coat.dryer_z1_temp": c["z1"] + n("coat.dryer_z1_temp"),
            "coat.dryer_z2_temp": c["z2"] + n("coat.dryer_z2_temp"),
            "coat.dryer_z3_temp": c["z3"] + n("coat.dryer_z3_temp"),
            "coat.web_tension": c["tension"] + n("coat.web_tension"),
            "coat.wet_thickness": c["wet"] + n("coat.wet_thickness"),
            "coat.dry_loading": c["loading"] + n("coat.dry_loading"),
            "coat.residual_solvent": c["residual"] + n("coat.residual_solvent"),
            "coat.web_speed_actual": c["web_speed"] + n("coat.web_speed_actual"),
            "slit.web_tension": s["tension"] + n("slit.web_tension"),
            "slit.edge_burr": s["burr"] + n("slit.edge_burr"),
            "slit.rewind_align": s["align"] + n("slit.rewind_align"),
            "slit.line_speed_actual": s["line_speed"] + n("slit.line_speed_actual"),
        }

    def interlock_values(self) -> dict[str, bool]:
        return {
            "mix.vessel_closed": True,
            "coat.web_present": not self.web_broken,
            "slit.tension_ok": self.slit["tension"] >= 25.0,
        }

    def true_state(self) -> dict[str, float]:
        """Noise-free physical state (oracle-only twin output)."""
        out = {}
        for stage, d in (("mix", self.mix), ("coat", self.coat), ("slit", self.slit)):
            for k, v in d.items():
                out[f"{stage}.{k}"] = float(v)
        out["coat.web_broken"] = float(self.web_broken)
        return out

    # ---- integration ----------------------------------------------------
    def integrate(self, act: dict[str, float], dt: float) -> None:
        self._mix(act, dt)
        self._coat(act, dt)
        self._slit(act, dt)

    def _mix(self, a: dict[str, float], dt: float) -> None:
        p, st = self.p.mix, self.mix
        st["speed"] += (a["mix.motor_cmd"] - st["speed"]) * dt / p.tau_speed
        frac = st["speed"] / 3500.0
        st["temp"] += (p.k_heat * a["mix.heater_cmd"] + p.k_shear_heat * frac * frac
                       - p.k_cool * (st["temp"] - p.ambient)) * dt
        if st["solids"] < p.dose_target_solids:
            st["solids"] = min(p.dose_target_solids, st["solids"] + a["mix.dose_cmd"] * dt)
        # Lower absolute pressure = stronger de-aeration (vented when the PLC stops).
        vac = max(0.0, min(1.0, (101.0 - a["mix.vacuum_cmd"]) / 101.0))
        # Air is entrained only above a critical shear (vortex), removed under vacuum.
        gen = p.air_gen * 2.0 * max(0.0, frac - p.air_crit_frac)
        st["air"] += (gen * (1.0 - vac) - p.air_remove * vac * st["air"]) * dt
        st["air"] = max(0.0, st["air"])
        visc_target = (p.visc_base
                       * math.exp(p.visc_solids_gain * (st["solids"] - 60.0))
                       * (1.0 - p.visc_temp_gain * max(0.0, st["temp"] - p.ambient))
                       * (1.0 + p.visc_air_gain * st["air"]))
        st["viscosity"] += (visc_target - st["viscosity"]) * dt / p.tau_visc
        st["viscosity"] = max(0.0, st["viscosity"])

    def _coat(self, a: dict[str, float], dt: float) -> None:
        p, st = self.p.coat, self.coat
        prev = st["web_speed"]
        st["web_speed"] += (a["coat.drive_cmd"] - st["web_speed"]) * dt / p.tau_speed
        accel = abs(st["web_speed"] - prev) / dt
        for z, h in (("z1", "coat.z1_heat_cmd"), ("z2", "coat.z2_heat_cmd"), ("z3", "coat.z3_heat_cmd")):
            st[z] += (p.k_heat * a[h] - p.k_loss * (st[z] - p.ambient)) * dt
            st[z] = max(p.ambient, st[z])

        # Tension follows the brake/dancer command with a bump on acceleration;
        # a sustained over-tension breaks the web (latched until the line stops).
        if self.web_broken:
            tgt = 0.0
        else:
            tgt = a["coat.tension_cmd"] + p.accel_tension * accel
        st["tension"] += (tgt - st["tension"]) * dt / p.tau_tension
        st["tension"] = max(0.0, st["tension"])
        if st["tension"] > self.WEB_BREAK_N:
            self._over_tension_s += dt
            if self._over_tension_s >= self.WEB_BREAK_DWELL_S:
                self.web_broken = True
        else:
            self._over_tension_s = 0.0
        if self.web_broken and a["coat.drive_cmd"] <= 0.0 and st["web_speed"] < 0.05:
            self.web_broken = False   # line stopped and re-threaded

        # Wet film: mass balance, capped by the slot-die gap (bead cannot sustain
        # a film much thicker than the gap -> starved/under-loaded coating).
        if a["coat.pump_cmd"] > 0.0 and st["web_speed"] > 0.2 and not self.web_broken:
            wet = p.wet_gain * a["coat.pump_cmd"] / st["web_speed"]
            gap = max(a["coat.gap_cmd"], 1.0)
            wet = min(wet, 1.0 * gap, 400.0)
        else:
            wet = 0.0
        st["wet"] += (wet - st["wet"]) * dt / 1.0
        st["loading"] = p.loading_gain * st["wet"]

        # Residual solvent: drying capacity x residence time (the context-shift output).
        cap = sum(max(0.0, st[z] - p.dry_floor_temp) for z in ("z1", "z2", "z3")) / 3.0 / 100.0
        cap *= (0.5 + a["coat.fan_cmd"] / 20.0)
        residence = p.oven_length / max(st["web_speed"], 0.2)
        dry = p.dry_k * cap * residence
        resid_target = p.solvent_in * math.exp(-max(dry, 0.0)) if st["wet"] > 1.0 else 0.0
        st["residual"] += (resid_target - st["residual"]) * dt / 2.0

    def _slit(self, a: dict[str, float], dt: float) -> None:
        p, st = self.p.slit, self.slit
        prev = st["line_speed"]
        st["line_speed"] += (a["slit.drive_cmd"] - st["line_speed"]) * dt / p.tau_speed
        accel = abs(st["line_speed"] - prev) / dt
        tgt = a["slit.tension_cmd"] + p.accel_tension * accel
        st["tension"] += (tgt - st["tension"]) * dt / p.tau_tension
        st["tension"] = max(0.0, st["tension"])
        burr_tgt = p.burr_base
        if a["slit.knife_cmd"] >= 0.5 and st["line_speed"] > 1.0:
            burr_tgt += p.burr_speed_gain * max(0.0, st["line_speed"] - p.burr_speed_opt)
            # A slack web tears rather than shears; the faster the line, the worse.
            burr_tgt += (p.burr_tension_gain * max(0.0, p.burr_tension_floor - st["tension"])
                         * (st["line_speed"] / 10.0))
        st["burr"] += (burr_tgt - st["burr"]) * dt / p.tau_burr
        nominal = 0.12 * st["tension"]
        align_tgt = p.align_gain * abs(a["slit.rewind_cmd"] - nominal)
        st["align"] += (align_tgt - st["align"]) * dt / p.tau_align
