"""Minimal asyncio Modbus TCP server with a pluggable register backend.

Implements the function codes the testbed uses — FC1/2/3/4 reads, FC5/6
single writes, FC15/16 multiple writes — with standard exception responses.
Addresses are the 0-based wire addresses (no hidden offset). A backend answers
reads/writes by (table, address); tables are "coil", "discrete", "holding",
"input". Registers travel as unsigned 16-bit words, bits as 0/1.

Two backends use it: the process simulator's field I/O (registers backed by
the physics) and the dev soft-PLC (registers backed by the compiled PLC image,
with OpenPLC v4's map). An optional ``on_transaction`` hook receives one record
per request/response — in the dev backend, where there is no packet capture,
this is the source of the Modbus transaction stream.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

log = logging.getLogger("graphiot.modbus")

ILLEGAL_FUNCTION, ILLEGAL_ADDRESS, ILLEGAL_VALUE, DEVICE_FAILURE = 1, 2, 3, 4

READ_BITS = {1: "coil", 2: "discrete"}
READ_REGS = {3: "holding", 4: "input"}
MAX_READ_BITS, MAX_READ_REGS = 2000, 125
MAX_WRITE_BITS, MAX_WRITE_REGS = 1968, 123


class ModbusBackend(Protocol):
    def read(self, table: str, address: int, count: int) -> list[int] | None: ...
    def write(self, table: str, address: int, values: list[int]) -> bool: ...


@dataclass
class Transaction:
    """One served request (dev-mode substitute for a pcap-decoded transaction)."""
    ts: float
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    transaction_id: int
    unit_id: int
    function_code: int
    address: int
    count: int
    values: list[int] = field(default_factory=list)  # written values, or read result
    exception: int = 0
    request_bytes: int = 0       # MBAP + PDU on the wire
    response_bytes: int = 0


def _pack_bits(bits: list[int]) -> bytes:
    out = bytearray((len(bits) + 7) // 8)
    for i, b in enumerate(bits):
        if b:
            out[i // 8] |= 1 << (i % 8)
    return bytes(out)


def _unpack_bits(data: bytes, count: int) -> list[int]:
    return [(data[i // 8] >> (i % 8)) & 1 for i in range(count)]


class ModbusTcpServer:
    def __init__(self, backend: ModbusBackend, host: str, port: int, *,
                 unit_ids: set[int] | None = None,
                 on_transaction: Callable[[Transaction], None] | None = None):
        self.backend = backend
        self.host, self.port = host, port
        self.unit_ids = unit_ids        # None = answer any unit id (like OpenPLC)
        self.on_transaction = on_transaction
        self._server: asyncio.base_events.Server | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, self.host, self.port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def serve_forever(self) -> None:
        await self.start()
        async with self._server:
            await self._server.serve_forever()

    # ---- one connection --------------------------------------------------
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer = writer.get_extra_info("peername") or ("?", 0)
        local = writer.get_extra_info("sockname") or (self.host, self.port)
        try:
            while True:
                header = await reader.readexactly(7)
                tid, proto, length, unit = struct.unpack(">HHHB", header)
                if proto != 0 or length < 2 or length > 254:
                    break
                pdu = await reader.readexactly(length - 1)
                if self.unit_ids is not None and unit not in self.unit_ids:
                    continue  # ignore other unit ids before touching the backend
                resp_pdu, rec = self._dispatch(pdu)
                resp = struct.pack(">HHHB", tid, 0, len(resp_pdu) + 1, unit) + resp_pdu
                writer.write(resp)
                await writer.drain()
                if self.on_transaction and rec is not None:
                    rec.ts = time.time()
                    rec.src_ip, rec.src_port = peer[0], peer[1]
                    rec.dst_ip, rec.dst_port = local[0], local[1]
                    rec.transaction_id, rec.unit_id = tid, unit
                    rec.request_bytes = 7 + len(pdu)
                    rec.response_bytes = len(resp)
                    try:
                        self.on_transaction(rec)
                    except Exception:  # never let logging break the server
                        log.exception("transaction hook failed")
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()

    # ---- PDU dispatch ----------------------------------------------------
    def _exc(self, fc: int, code: int, rec: Transaction) -> tuple[bytes, Transaction]:
        rec.exception = code
        return bytes([fc | 0x80, code]), rec

    def _dispatch(self, pdu: bytes) -> tuple[bytes, Transaction | None]:
        fc = pdu[0]
        rec = Transaction(0.0, "", 0, "", 0, 0, 0, fc, 0, 0)
        try:
            if fc in READ_BITS or fc in READ_REGS:
                addr, count = struct.unpack(">HH", pdu[1:5])
                rec.address, rec.count = addr, count
                bits = fc in READ_BITS
                if count < 1 or count > (MAX_READ_BITS if bits else MAX_READ_REGS):
                    return self._exc(fc, ILLEGAL_VALUE, rec)
                table = READ_BITS.get(fc) or READ_REGS[fc]
                vals = self.backend.read(table, addr, count)
                if vals is None:
                    return self._exc(fc, ILLEGAL_ADDRESS, rec)
                rec.values = list(vals)
                if bits:
                    data = _pack_bits(vals)
                else:
                    data = struct.pack(f">{count}H", *[v & 0xFFFF for v in vals])
                return bytes([fc, len(data)]) + data, rec
            if fc == 5:
                addr, raw = struct.unpack(">HH", pdu[1:5])
                rec.address, rec.count = addr, 1
                if raw not in (0x0000, 0xFF00):
                    return self._exc(fc, ILLEGAL_VALUE, rec)
                rec.values = [1 if raw == 0xFF00 else 0]
                if not self.backend.write("coil", addr, rec.values):
                    return self._exc(fc, ILLEGAL_ADDRESS, rec)
                return pdu[:5], rec
            if fc == 6:
                addr, raw = struct.unpack(">HH", pdu[1:5])
                rec.address, rec.count, rec.values = addr, 1, [raw]
                if not self.backend.write("holding", addr, [raw]):
                    return self._exc(fc, ILLEGAL_ADDRESS, rec)
                return pdu[:5], rec
            if fc in (15, 16):
                addr, count, nbytes = struct.unpack(">HHB", pdu[1:6])
                rec.address, rec.count = addr, count
                payload = pdu[6:6 + nbytes]
                if fc == 15:
                    if count < 1 or count > MAX_WRITE_BITS or nbytes != (count + 7) // 8:
                        return self._exc(fc, ILLEGAL_VALUE, rec)
                    vals = _unpack_bits(payload, count)
                    table = "coil"
                else:
                    if count < 1 or count > MAX_WRITE_REGS or nbytes != 2 * count:
                        return self._exc(fc, ILLEGAL_VALUE, rec)
                    vals = list(struct.unpack(f">{count}H", payload))
                    table = "holding"
                rec.values = vals
                if not self.backend.write(table, addr, vals):
                    return self._exc(fc, ILLEGAL_ADDRESS, rec)
                return struct.pack(">BHH", fc, addr, count), rec
            return self._exc(fc, ILLEGAL_FUNCTION, rec)
        except (struct.error, IndexError):
            return self._exc(fc, ILLEGAL_VALUE, rec)
        except Exception:
            log.exception("backend failure on FC%d", fc)
            return self._exc(fc, DEVICE_FAILURE, rec)


class RegisterBank:
    """A simple in-memory backend: fixed-size tables, optional read-only tables."""

    def __init__(self, sizes: dict[str, int], read_only: set[str] = frozenset()):
        self.tables = {t: [0] * n for t, n in sizes.items()}
        self.read_only = set(read_only)

    def read(self, table: str, address: int, count: int) -> list[int] | None:
        tab = self.tables.get(table)
        if tab is None or address < 0 or address + count > len(tab):
            return None
        return tab[address:address + count]

    def write(self, table: str, address: int, values: list[int]) -> bool:
        if table in self.read_only:
            return False
        tab = self.tables.get(table)
        if tab is None or address < 0 or address + len(values) > len(tab):
            return False
        tab[address:address + len(values)] = [int(v) for v in values]
        return True

    # local (non-wire) access for the owner of the bank
    def set_local(self, table: str, address: int, values: list[int]) -> None:
        tab = self.tables[table]
        tab[address:address + len(values)] = [int(v) for v in values]
