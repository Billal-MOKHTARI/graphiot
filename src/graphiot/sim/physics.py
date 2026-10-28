"""Process physics parameters for the three stages.

Simple first-order dynamics with transport delay, calibrated so that correct
recipes keep process outputs inside the operating limits and mistimed commands
(a steady-phase value applied during ramp) drive an output out of limits. The
equations live in :mod:`graphiot.sim.plant`; this module only holds the tunable
constants, so the pilot can re-tune without touching the coupling logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MixParams:
    ambient: float = 22.0
    tau_speed: float = 1.5          # s, motor speed lag
    k_heat: float = 0.03            # degC per (% duty * s)
    k_shear_heat: float = 0.05      # degC/s at full speed from shear
    k_cool: float = 0.012           # 1/s toward ambient
    dose_target_solids: float = 70.0
    tau_visc: float = 8.0           # s, viscosity relaxation
    visc_base: float = 2600.0       # mPa.s nominal at reference
    visc_solids_gain: float = 0.06  # per % solids above 60
    visc_temp_gain: float = 0.010   # per degC above ambient (thinning, negative effect)
    visc_air_gain: float = 0.020    # per % entrained air (thickening/instability)
    air_gen: float = 9.0            # %/s per unit of shear above the vortex threshold
    air_crit_frac: float = 0.4      # entrainment starts above this fraction of max speed
    air_remove: float = 1.0         # 1/s removal rate under full vacuum


@dataclass
class CoatParams:
    ambient: float = 22.0
    tau_speed: float = 0.8          # s, web speed lag
    k_heat: float = 0.055           # degC per (% duty * s)
    k_loss: float = 0.011           # 1/s zone heat loss
    tau_tension: float = 1.2        # s
    accel_tension: float = 6.0      # N per (m/min/s) of web acceleration
    wet_gain: float = 7.5           # um * (m/min) / (mL/min)
    loading_gain: float = 0.12      # mg/cm2 per um wet (incl. solids fraction)
    oven_length: float = 10.0       # m
    solvent_in: float = 8.0         # % solvent load entering the oven
    dry_k: float = 5.2              # drying rate constant
    dry_floor_temp: float = 40.0    # degC below which a zone does little drying


@dataclass
class SlitParams:
    ambient: float = 22.0
    tau_speed: float = 1.0
    tau_tension: float = 1.0
    accel_tension: float = 5.0
    burr_base: float = 3.0
    burr_speed_gain: float = 0.5    # um per (m/min) above the optimum
    burr_speed_opt: float = 32.0
    burr_tension_gain: float = 0.25  # um per N of tension shortfall below band
    burr_tension_floor: float = 110.0
    tau_burr: float = 2.0
    align_gain: float = 0.03        # mm per Nm torque error
    tau_align: float = 3.0


@dataclass
class PhysicsParams:
    mix: MixParams = field(default_factory=MixParams)
    coat: CoatParams = field(default_factory=CoatParams)
    slit: SlitParams = field(default_factory=SlitParams)
    # Sensor noise standard deviations, in engineering units, per sensor tag.
    noise_sigma: dict[str, float] = field(default_factory=lambda: {
        "mix.viscosity": 15.0, "mix.solids_pct": 0.1, "mix.temp": 0.1,
        "mix.air_entrain": 0.15, "mix.speed_actual": 2.0,
        "coat.dryer_z1_temp": 0.3, "coat.dryer_z2_temp": 0.3, "coat.dryer_z3_temp": 0.3,
        "coat.web_tension": 0.5, "coat.wet_thickness": 0.5, "coat.dry_loading": 0.05,
        "coat.residual_solvent": 0.03, "coat.web_speed_actual": 0.05,
        "slit.web_tension": 0.5, "slit.edge_burr": 0.1, "slit.rewind_align": 0.01,
        "slit.line_speed_actual": 0.05,
    })
