"""HMI emulator: polls every PLC at the configured rate (1 Hz) and logs the values.

Its log, ``/data/capture/tags.jsonl``, is the "PLC tag values sampled at 1 Hz"
stream the monitor observes (paper §III-B) — values exactly as a SCADA client
reads them from the PLCs, never from the simulator. One record per PLC per poll:
``{"ts", "plc", "values": {tag: engineering value}}``.

Background writes: at a low random rate the operator "confirms" a setpoint from
the HMI by re-writing its current value. These are real Modbus writes from the
HMI host (Host→Tag write edges in the graph) that never change the process.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
from pathlib import Path

from pymodbus.client import AsyncModbusTcpClient

from ..config.io import load_config
from ..modbus.poller import plan_blocks, read_blocks
from ..plant.tags import Kind, Plc, load_tag_map

log = logging.getLogger("graphiot.hmi")


async def run_hmi(*, tags, hosts: dict[Plc, tuple[str, int]], poll_hz: float, out: Path,
                  confirm_rate_per_h: float, seed: int, stop: asyncio.Event | None = None) -> None:
    rng = random.Random(seed)
    stop = stop or asyncio.Event()
    clients = {p: AsyncModbusTcpClient(h, port=port, timeout=1.0) for p, (h, port) in hosts.items()}
    blocks = {p: plan_blocks(tags, p) for p in hosts}
    setpoints = [t for t in tags.values() if t.kind is Kind.SETPOINT]
    period = 1.0 / poll_hz
    out.parent.mkdir(parents=True, exist_ok=True)
    next_t = time.monotonic()
    with open(out, "a", buffering=1) as fh:
        while not stop.is_set():
            ts = time.time()
            for plc, cl in clients.items():
                try:
                    if not cl.connected:
                        await cl.connect()
                    values = await read_blocks(cl, blocks[plc])
                except Exception as exc:  # keep polling; a gap is visible in the data
                    log.warning("poll %s failed: %s", plc.value, exc)
                    continue
                fh.write(json.dumps({"ts": ts, "plc": plc.value, "values": values}) + "\n")
            # Poisson-rate setpoint confirmation (writes the value already there).
            if setpoints and rng.random() < confirm_rate_per_h * period / 3600.0:
                t = rng.choice(setpoints)
                cl = clients[t.plc]
                try:
                    r = await cl.read_holding_registers(t.modbus_address, count=1, device_id=1)
                    if not r.isError():
                        await cl.write_register(t.modbus_address, r.registers[0], device_id=1)
                except Exception as exc:
                    log.warning("confirm write failed: %s", exc)
            next_t += period
            await asyncio.sleep(max(0.0, next_t - time.monotonic()))
    for cl in clients.values():
        cl.close()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default="/app")
    ap.add_argument("--out", default="/data/capture/tags.jsonl")
    ap.add_argument("--plc-host", default=None)
    ap.add_argument("--plc-ports", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(args.config)
    tags = load_tag_map(Path(args.root) / cfg.paths.tag_map)
    if args.plc_host:
        ports = [int(p) for p in args.plc_ports.split(",")]
        hosts = {p: (args.plc_host, port) for p, port in zip(Plc, ports)}
    else:
        hosts = {p: (cfg.testbed.network.plc_ips[p.value], cfg.testbed.openplc.modbus_port) for p in Plc}
    asyncio.run(run_hmi(tags=tags, hosts=hosts, poll_hz=cfg.campaign.capture.tag_poll_hz,
                        out=Path(args.out), confirm_rate_per_h=2.0,
                        seed=cfg.campaign.schedule_seed + 1))


if __name__ == "__main__":
    main()
