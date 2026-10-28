"""Decode cell-network pcaps into Modbus transactions and NetFlow-style flows.

* **Modbus/TCP transactions** — TCP payloads to/from port 502 are reassembled per
  direction of each connection, split into MBAP frames, and requests are matched
  to responses by transaction id. One row per request: function code, address,
  quantity, written values (or read results), exception code, sizes, timing.
* **Flows** — bidirectional 5-tuple flows (the initiator is the client), with
  NetFlow v9-style counters after Sarhan et al. (2022): bytes, packets, duration,
  TCP flags per side, TTL and packet-length extremes, throughput, small/large
  packet histogram. A flow record is exported when the flow has been idle for
  ``idle_timeout_s`` and, for long-lived connections, every ``active_timeout_s``
  (interim records), so a snapshot can only use flows that ended before it.

Pure Python on ``dpkt``; no libpcap needed (pcaps are written by tcpdump in the
capture container).
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass, field
from pathlib import Path

import dpkt

MODBUS_PORT = 502


def _ip(b: bytes) -> str:
    return socket.inet_ntoa(b)


def iter_packets(paths: list[Path]):
    """Yield (ts, ip, tcp_or_udp_or_None) over pcaps in time order."""
    for p in sorted(paths):
        with open(p, "rb") as fh:
            try:
                reader = dpkt.pcap.Reader(fh)
            except (ValueError, dpkt.dpkt.NeedData):
                continue                          # empty or truncated rotation file
            link = reader.datalink()
            for ts, buf in reader:
                try:
                    if link == dpkt.pcap.DLT_EN10MB:
                        eth = dpkt.ethernet.Ethernet(buf)
                        ip = eth.data
                    elif link == dpkt.pcap.DLT_LINUX_SLL:
                        ip = dpkt.sll.SLL(buf).data
                    else:
                        ip = dpkt.ip.IP(buf)
                except (dpkt.dpkt.UnpackError, dpkt.dpkt.NeedData):
                    continue
                if not isinstance(ip, dpkt.ip.IP):
                    continue
                l4 = ip.data if isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)) else None
                yield ts, ip, l4


# --------------------------------------------------------------------------- #
@dataclass
class _Pending:
    ts: float
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    tid: int
    unit: int
    fc: int
    address: int
    count: int
    values: list[int]
    req_len: int


def _parse_request(pdu: bytes) -> tuple[int, int, int, list[int]]:
    """(fc, address, count, written values) of a request PDU."""
    fc = pdu[0]
    if fc in (1, 2, 3, 4):
        addr, qty = struct.unpack(">HH", pdu[1:5])
        return fc, addr, qty, []
    if fc == 5:
        addr, raw = struct.unpack(">HH", pdu[1:5])
        return fc, addr, 1, [1 if raw == 0xFF00 else 0]
    if fc == 6:
        addr, raw = struct.unpack(">HH", pdu[1:5])
        return fc, addr, 1, [raw]
    if fc == 15:
        addr, qty, n = struct.unpack(">HHB", pdu[1:6])
        bits = [(pdu[6 + i // 8] >> (i % 8)) & 1 for i in range(qty)]
        return fc, addr, qty, bits
    if fc == 16:
        addr, qty, n = struct.unpack(">HHB", pdu[1:6])
        return fc, addr, qty, list(struct.unpack(f">{qty}H", pdu[6:6 + 2 * qty]))
    return fc, 0, 0, []


def _parse_response(fc: int, count: int, pdu: bytes) -> tuple[int, list[int]]:
    """(exception code, read values) of a response PDU."""
    if pdu[0] & 0x80:
        return pdu[1], []
    if fc in (1, 2):
        n = pdu[1]
        data = pdu[2:2 + n]
        return 0, [(data[i // 8] >> (i % 8)) & 1 for i in range(count)]
    if fc in (3, 4):
        n = pdu[1]
        return 0, list(struct.unpack(f">{n // 2}H", pdu[2:2 + n]))
    return 0, []


def modbus_transactions(paths: list[Path]) -> list[dict]:
    """Reassemble Modbus/TCP over every connection to port 502."""
    buffers: dict[tuple, bytearray] = {}
    pending: dict[tuple, _Pending] = {}
    out: list[dict] = []
    for ts, ip, l4 in iter_packets(paths):
        if not isinstance(l4, dpkt.tcp.TCP) or not l4.data:
            continue
        if MODBUS_PORT not in (l4.sport, l4.dport):
            continue
        src, dst = _ip(ip.src), _ip(ip.dst)
        to_server = l4.dport == MODBUS_PORT
        conn = (src, l4.sport, dst, l4.dport) if to_server else (dst, l4.dport, src, l4.sport)
        key = (conn, to_server)
        buf = buffers.setdefault(key, bytearray())
        buf += bytes(l4.data)
        while len(buf) >= 7:
            tid, proto, length, unit = struct.unpack(">HHHB", buf[:7])
            if proto != 0 or length < 2 or length > 254:
                buf.clear()                       # lost sync: drop this direction's buffer
                break
            if len(buf) < 6 + length:
                break
            pdu = bytes(buf[7:6 + length])
            frame_len = 6 + length
            del buf[:frame_len]
            if not pdu:
                continue
            if to_server:
                try:
                    fc, addr, cnt, vals = _parse_request(pdu)
                except (struct.error, IndexError):
                    continue
                pending[(conn, tid)] = _Pending(ts, conn[0], conn[1], conn[2], conn[3], tid,
                                                unit, fc, addr, cnt, vals, frame_len)
            else:
                req = pending.pop((conn, tid), None)
                if req is None:
                    continue
                try:
                    exc, read_vals = _parse_response(req.fc, req.count, pdu)
                except (struct.error, IndexError):
                    exc, read_vals = 0, []
                out.append({
                    "ts": req.ts, "ts_response": ts, "src_ip": req.src_ip, "src_port": req.src_port,
                    "dst_ip": req.dst_ip, "dst_port": req.dst_port, "transaction_id": tid,
                    "unit_id": req.unit, "function_code": req.fc, "address": req.address,
                    "count": req.count, "write_values": req.values, "read_values": read_vals,
                    "exception": exc, "request_bytes": req.req_len, "response_bytes": frame_len,
                })
    for req in pending.values():                  # requests never answered
        out.append({"ts": req.ts, "ts_response": None, "src_ip": req.src_ip, "src_port": req.src_port,
                    "dst_ip": req.dst_ip, "dst_port": req.dst_port, "transaction_id": req.tid,
                    "unit_id": req.unit, "function_code": req.fc, "address": req.address,
                    "count": req.count, "write_values": req.values, "read_values": [],
                    "exception": -1, "request_bytes": req.req_len, "response_bytes": 0})
    out.sort(key=lambda r: r["ts"])
    return out


# --------------------------------------------------------------------------- #
@dataclass
class _Flow:
    key: tuple                 # (client_ip, client_port, server_ip, server_port, proto)
    first: float
    last: float
    export_start: float
    c_bytes: int = 0
    s_bytes: int = 0
    c_pkts: int = 0
    s_pkts: int = 0
    c_flags: int = 0
    s_flags: int = 0
    min_ttl: int = 255
    max_ttl: int = 0
    min_len: int = 1 << 30
    max_len: int = 0
    pkts_le_128: int = 0
    pkts_128_512: int = 0
    pkts_gt_512: int = 0
    tcp_win_c: int = 0
    tcp_win_s: int = 0
    retrans_c: int = 0
    seen_seq: set = field(default_factory=set)

    def record(self, end: float, reason: str) -> dict:
        dur = max(end - self.export_start, 1e-6)
        ci, cp, si, sp, proto = self.key
        return {
            "flow_start": self.export_start, "flow_end": end, "end_reason": reason,
            "src_ip": ci, "src_port": cp, "dst_ip": si, "dst_port": sp, "protocol": proto,
            "IN_BYTES": self.c_bytes, "OUT_BYTES": self.s_bytes,
            "IN_PKTS": self.c_pkts, "OUT_PKTS": self.s_pkts,
            "FLOW_DURATION_MILLISECONDS": round(1000 * (end - self.export_start), 3),
            "TCP_FLAGS": self.c_flags | self.s_flags, "CLIENT_TCP_FLAGS": self.c_flags,
            "SERVER_TCP_FLAGS": self.s_flags,
            "MIN_TTL": self.min_ttl if self.c_pkts + self.s_pkts else 0, "MAX_TTL": self.max_ttl,
            "SHORTEST_FLOW_PKT": self.min_len if self.c_pkts + self.s_pkts else 0,
            "LONGEST_FLOW_PKT": self.max_len,
            "SRC_TO_DST_SECOND_BYTES": self.c_bytes / dur, "DST_TO_SRC_SECOND_BYTES": self.s_bytes / dur,
            "SRC_TO_DST_AVG_THROUGHPUT": 8 * self.c_bytes / dur,
            "DST_TO_SRC_AVG_THROUGHPUT": 8 * self.s_bytes / dur,
            "NUM_PKTS_UP_TO_128_BYTES": self.pkts_le_128,
            "NUM_PKTS_128_TO_512_BYTES": self.pkts_128_512,
            "NUM_PKTS_OVER_512_BYTES": self.pkts_gt_512,
            "TCP_WIN_MAX_IN": self.tcp_win_c, "TCP_WIN_MAX_OUT": self.tcp_win_s,
            "RETRANSMITTED_IN_PKTS": self.retrans_c,
        }

    def reset_counters(self, at: float) -> None:
        keep = (self.key, self.first, self.last)
        self.__init__(key=keep[0], first=keep[1], last=keep[2], export_start=at)


def flows(paths: list[Path], *, idle_timeout_s: float = 15.0, active_timeout_s: float = 5.0) -> list[dict]:
    active: dict[tuple, _Flow] = {}
    out: list[dict] = []
    last_sweep = 0.0
    for ts, ip, l4 in iter_packets(paths):
        if ts - last_sweep >= 1.0:                # expire idle flows / emit interim records
            for k, f in list(active.items()):
                if ts - f.last >= idle_timeout_s:
                    out.append(f.record(f.last, "idle"))
                    del active[k]
                elif ts - f.export_start >= active_timeout_s:
                    out.append(f.record(f.export_start + active_timeout_s, "active"))
                    f.reset_counters(f.export_start + active_timeout_s)
            last_sweep = ts
        proto = ip.p
        sport = getattr(l4, "sport", 0)
        dport = getattr(l4, "dport", 0)
        src, dst = _ip(ip.src), _ip(ip.dst)
        fwd = (src, sport, dst, dport, proto)
        rev = (dst, dport, src, sport, proto)
        if fwd in active:
            f, from_client = active[fwd], True
        elif rev in active:
            f, from_client = active[rev], False
        else:
            # Initiator = sender of a SYN without ACK; otherwise the lower port is the server.
            syn_only = isinstance(l4, dpkt.tcp.TCP) and (l4.flags & dpkt.tcp.TH_SYN) and \
                not (l4.flags & dpkt.tcp.TH_ACK)
            client_first = bool(syn_only) or dport < sport
            key = fwd if client_first else rev
            f = _Flow(key=key, first=ts, last=ts, export_start=ts)
            active[key] = f
            from_client = client_first
        n = ip.len
        f.last = ts
        f.min_ttl, f.max_ttl = min(f.min_ttl, ip.ttl), max(f.max_ttl, ip.ttl)
        f.min_len, f.max_len = min(f.min_len, n), max(f.max_len, n)
        if n <= 128:
            f.pkts_le_128 += 1
        elif n <= 512:
            f.pkts_128_512 += 1
        else:
            f.pkts_gt_512 += 1
        if from_client:
            f.c_bytes += n
            f.c_pkts += 1
        else:
            f.s_bytes += n
            f.s_pkts += 1
        if isinstance(l4, dpkt.tcp.TCP):
            if from_client:
                f.c_flags |= l4.flags
                f.tcp_win_c = max(f.tcp_win_c, l4.win)
                if l4.data:
                    seq_key = (l4.seq, len(l4.data))
                    if seq_key in f.seen_seq:
                        f.retrans_c += 1
                    elif len(f.seen_seq) < 4096:
                        f.seen_seq.add(seq_key)
            else:
                f.s_flags |= l4.flags
                f.tcp_win_s = max(f.tcp_win_s, l4.win)
    for f in active.values():
        out.append(f.record(f.last, "end_of_capture"))
    out.sort(key=lambda r: r["flow_end"])
    return out
