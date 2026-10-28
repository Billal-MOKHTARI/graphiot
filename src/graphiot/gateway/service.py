"""Gateway container entry point: ``python -m graphiot.gateway.service --config ...``.

Reads the run's config, connects to the three PLCs on the cell network, serves
the tool API on the north network only, and logs to ``/data/capture/``.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import uvicorn

from ..config.io import load_config
from ..plant.allow_list import load_allow_list
from ..plant.recipes import load_recipes
from ..plant.tags import Plc, load_tag_map
from .app import build_app
from .core import Gateway
from .southbound import PymodbusPort


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default="/app", help="where plant/ is mounted")
    ap.add_argument("--log-dir", default="/data/capture")
    ap.add_argument("--bind", default=None, help="override the bind address")
    ap.add_argument("--plc-host", default=None, help="dev: all PLCs on this host")
    ap.add_argument("--plc-ports", default=None, help="dev: mix,coat,slit Modbus ports")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    cfg = load_config(args.config)
    root = Path(args.root)
    tags = load_tag_map(root / cfg.paths.tag_map)
    net = cfg.testbed.network
    if args.plc_host:
        mix, coat, slit = (int(p) for p in args.plc_ports.split(","))
        ports = {Plc.MIX: PymodbusPort(args.plc_host, mix), Plc.COAT: PymodbusPort(args.plc_host, coat),
                 Plc.SLIT: PymodbusPort(args.plc_host, slit)}
    else:
        ports = {plc: PymodbusPort(net.plc_ips[plc.value], cfg.testbed.openplc.modbus_port,
                                   timeout_s=cfg.gateway.modbus_timeout_s) for plc in Plc}
    gateway = Gateway(tags=tags, allow_list=load_allow_list(root / cfg.paths.allow_list),
                      recipes=load_recipes(root / cfg.paths.recipes), ports=ports,
                      log_dir=args.log_dir)
    token = os.environ.get(cfg.gateway.admin_token_env, "")
    app = build_app(gateway, admin_token=token)
    uvicorn.run(app, host=args.bind or net.gateway_north_ip, port=cfg.gateway.north_port,
                log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
