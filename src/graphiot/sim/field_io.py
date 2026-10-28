"""The process simulator as the PLCs' remote I/O (Docker / OpenPLC mode).

One process runs the physics model in real time and serves one Modbus TCP
device per PLC on the field network. The OpenPLC Modbus-master plugin of each
PLC polls its device every 100 ms:

    input registers 0..n-1   <- sensors      (PLC reads FC4 into %IW0..)
    discrete inputs 0..k-1   <- interlocks   (PLC reads FC2 into %IX0.0..)
    holding registers 0..m-1 -> actuators    (PLC writes FC16 from %QW0..)

Each tick: read the actuators the PLCs last wrote, integrate the physics one
dt, draw the tick's noise, publish sensors and interlocks. The noise-free state
is logged at 1 Hz to a JSONL file under the run's ``groundtruth/`` — an
oracle-only twin output the monitor never reads. The same :class:`PlantPhysics`
drives the dev backend and the oracle twin, so live and replayed physics share
one code path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from ..modbus.server import ModbusTcpServer, RegisterBank
from ..plant.tags import Plc, Tag, load_tag_map
from .model import PlantPhysics, actuator_names, interlock_names, sensor_names
from .noise import NoiseField
from .physics import PhysicsParams

log = logging.getLogger("graphiot.field_io")

DEFAULT_PORTS = {Plc.MIX: 5021, Plc.COAT: 5022, Plc.SLIT: 5023}


def _s16(raw: int) -> int:
    return raw - 65536 if raw > 32767 else raw


class FieldIOServer:
    def __init__(self, *, tags: dict[str, Tag], seed: int, params: PhysicsParams | None = None,
                 host: str = "0.0.0.0", ports: dict[Plc, int] | None = None,
                 tick_s: float = 0.1, state_log: str | Path | None = None,
                 state_log_hz: float = 1.0):
        self.tags = tags
        self.host = host
        self.ports = ports or DEFAULT_PORTS
        self.tick_s = tick_s
        self.seed = seed
        self.physics = PlantPhysics(params or PhysicsParams(), NoiseField(seed), tags)
        self.layout: dict[Plc, tuple[list[str], list[str], list[str]]] = {}
        self.banks: dict[Plc, RegisterBank] = {}
        for plc in self.ports:
            s, i, a = sensor_names(tags, plc), interlock_names(tags, plc), actuator_names(tags, plc)
            self.layout[plc] = (s, i, a)
            # Sensors and interlocks are read-only on the wire: only the plant sets them.
            self.banks[plc] = RegisterBank(
                {"input": max(1, len(s)), "discrete": max(1, len(i)),
                 "holding": max(1, len(a)), "coil": 8},
                read_only={"input", "discrete"})
        self.servers = {plc: ModbusTcpServer(self.banks[plc], host, port)
                        for plc, port in self.ports.items()}
        self._state_log = Path(state_log) if state_log else None
        self._state_every = max(1, int(round((1.0 / state_log_hz) / tick_s)))
        self.tick_i = 0
        self._stop = asyncio.Event()

    def read_actuators(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for plc, (_, _, acts) in self.layout.items():
            raws = self.banks[plc].tables["holding"][:len(acts)]
            for name, raw in zip(acts, raws):
                out[name] = self.tags[name].decode(_s16(raw))
        return out

    def publish(self) -> None:
        sens = self.physics.sensor_values()
        il = self.physics.interlock_values()
        for plc, (s_names, i_names, _) in self.layout.items():
            bank = self.banks[plc]
            if s_names:
                bank.set_local("input", 0, [self.tags[n].encode(sens[n]) & 0xFFFF for n in s_names])
            if i_names:
                bank.set_local("discrete", 0, [1 if il[n] else 0 for n in i_names])

    def tick(self) -> dict[str, float]:
        act = self.read_actuators()
        self.physics.integrate(act, self.tick_s)
        self.physics.tick_noise()
        self.publish()
        self.tick_i += 1
        return act

    async def run(self) -> None:
        for srv in self.servers.values():
            await srv.start()
        log.info("field I/O on %s: %s", self.host, {p.value: v for p, v in self.ports.items()})
        fh = None
        if self._state_log:
            self._state_log.parent.mkdir(parents=True, exist_ok=True)
            fh = open(self._state_log, "a", buffering=1)
        self.physics.tick_noise()
        self.publish()
        next_t = time.monotonic()
        try:
            while not self._stop.is_set():
                act = self.tick()
                if fh and self.tick_i % self._state_every == 0:
                    fh.write(json.dumps({"ts": time.time(), "tick": self.tick_i, "seed": self.seed,
                                         "state": self.physics.true_state(),
                                         "actuators": act}) + "\n")
                next_t += self.tick_s
                await asyncio.sleep(max(0.0, next_t - time.monotonic()))
        finally:
            if fh:
                fh.close()
            for srv in self.servers.values():
                await srv.stop()

    def stop(self) -> None:
        self._stop.set()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Process simulator as Modbus remote I/O")
    ap.add_argument("--tag-map", default="plant/tag_map.csv")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--ports", default="5021,5022,5023", help="mix,coat,slit")
    ap.add_argument("--tick", type=float, default=0.1)
    ap.add_argument("--state-log", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p_mix, p_coat, p_slit = (int(x) for x in args.ports.split(","))
    srv = FieldIOServer(tags=load_tag_map(args.tag_map), seed=args.seed, host=args.host,
                        ports={Plc.MIX: p_mix, Plc.COAT: p_coat, Plc.SLIT: p_slit},
                        tick_s=args.tick, state_log=args.state_log)
    asyncio.run(srv.run())


if __name__ == "__main__":
    main()
