"""Engineering-workstation emulator: periodic diagnostic block reads.

Every 30–90 s (uniform, seeded) the EWS reads the full register image of one
PLC in a few block reads — the background traffic of an engineering tool
checking a controller. It writes nothing. Its own transaction log
(``/data/capture/ews.jsonl``) is kept for audit; the monitor sees this traffic
on the wire like any other host's.
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
from ..plant.tags import Plc, load_tag_map

log = logging.getLogger("graphiot.ews")


async def run_ews(*, tags, hosts: dict[Plc, tuple[str, int]], out: Path, seed: int,
                  interval_s: tuple[float, float] = (30.0, 90.0),
                  stop: asyncio.Event | None = None) -> None:
    rng = random.Random(seed)
    stop = stop or asyncio.Event()
    blocks = {p: plan_blocks(tags, p, readable_only=False) for p in hosts}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", buffering=1) as fh:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=rng.uniform(*interval_s))
            except asyncio.TimeoutError:
                pass
            if stop.is_set():
                break
            plc = rng.choice(list(hosts))
            host, port = hosts[plc]
            cl = AsyncModbusTcpClient(host, port=port, timeout=1.0)
            try:
                await cl.connect()
                vals = await read_blocks(cl, blocks[plc])
                fh.write(json.dumps({"ts": time.time(), "plc": plc.value, "n_tags": len(vals)}) + "\n")
            except Exception as exc:
                log.warning("diagnostic read of %s failed: %s", plc.value, exc)
            finally:
                cl.close()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default="/app")
    ap.add_argument("--out", default="/data/capture/ews.jsonl")
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
    asyncio.run(run_ews(tags=tags, hosts=hosts, out=Path(args.out),
                        seed=cfg.campaign.schedule_seed + 2))


if __name__ == "__main__":
    main()
