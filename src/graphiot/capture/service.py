"""Capture container entry point: tcpdump on the cell bridge, rotated pcaps.

Runs in the host network namespace with NET_RAW/NET_ADMIN (see
testbed/compose.py) so it can open the cell network's Linux bridge
(``testbed.network.cell_bridge``). Writes ``/data/capture/pcap/cell-*.pcap``,
rotated every ``campaign.capture.rotate_s``. On stop, hands the files back to the
owner of the run folder (tcpdump runs as root in the container).
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

from ..config.io import load_config

log = logging.getLogger("graphiot.capture")


def _chown_like(target: Path, ref: Path) -> None:
    st = ref.stat()
    for p in [target, *target.rglob("*")]:
        try:
            os.chown(p, st.st_uid, st.st_gid)
        except OSError:
            pass


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="/data/capture/pcap")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(args.config)
    cap = cfg.campaign.capture
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # -Z root: tcpdump otherwise drops to the 'tcpdump' user after opening the first
    # file and can no longer create the rotated files (-G) in the bind-mounted folder —
    # observed: only the first 60 s were kept. The container is the isolation boundary.
    cmd = ["tcpdump", "-i", cfg.testbed.network.cell_bridge, "-n", "-U", "-Z", "root",
           "-s", str(cap.snaplen), "-G", str(cap.rotate_s),
           "-w", str(out / "cell-%Y%m%dT%H%M%S.pcap"), "ip"]
    log.info("capturing: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd)

    def _stop(*_):
        proc.terminate()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    code = proc.wait()
    _chown_like(out, out.parent.parent)          # /data/capture/pcap -> owner of /data
    log.info("tcpdump exited with %s", code)
    sys.exit(0 if code in (0, -15, 143) else code)


if __name__ == "__main__":
    main()
