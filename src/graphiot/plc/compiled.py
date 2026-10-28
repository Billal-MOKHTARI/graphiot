"""ctypes wrapper around a compiled PLC program.

A :class:`CompiledPLC` holds one running instance of a PLC program. It reads and
writes located variables by IEC address or by Modbus (table, address), and steps
the program one scan at a time under caller-controlled simulated time. The live
soft-PLC and the oracle twin both use this; the twin creates a second instance
for counterfactual replay.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from .build import compile_st

# LocatedArea / LocatedSize mirror strucpp_abi.hpp.
AREA_INPUT, AREA_OUTPUT, AREA_MEMORY = 0, 1, 2
SIZE_BIT, SIZE_WORD = 0, 2


def parse_iec_address(iec: str) -> tuple[int, int, int, int]:
    """'%MW3' / '%QX1.0' -> (area, byte_index, bit_index, is_bit)."""
    body = iec[1:]
    area_code = body[:2]
    rest = body[2:]
    area = {"IW": AREA_INPUT, "IX": AREA_INPUT,
            "QW": AREA_OUTPUT, "QX": AREA_OUTPUT,
            "MW": AREA_MEMORY}[area_code]
    if area_code.endswith("X"):
        byte_s, bit_s = rest.split(".")
        return area, int(byte_s), int(bit_s), 1
    return area, int(rest), 0, 0


def modbus_to_iec(table: str, address: int) -> tuple[int, int, int, int]:
    """Modbus (table, address) -> (area, byte_index, bit_index, is_bit), per the
    OpenPLC v4 map: holding 0..1023 = %QW, 1024.. = %MW; input reg = %IW; coil =
    %QX; discrete = %IX."""
    if table == "input":
        return AREA_INPUT, address, 0, 0
    if table == "holding":
        if address >= 1024:
            return AREA_MEMORY, address - 1024, 0, 0
        return AREA_OUTPUT, address, 0, 0
    if table == "coil":
        return AREA_OUTPUT, address // 8, address % 8, 1
    if table == "discrete":
        return AREA_INPUT, address // 8, address % 8, 1
    raise ValueError(f"unknown Modbus table {table!r}")


class CompiledPLC:
    """One running PLC instance. Not thread-safe; drive it from one thread."""

    def __init__(self, st_path: str | Path | None = None, *,
                 so_path: str | Path | None = None, scan_s: float = 0.1):
        if so_path is None:
            if st_path is None:
                raise ValueError("provide st_path or so_path")
            so_path = compile_st(st_path)
        self.so_path = Path(so_path)
        self.scan_s = scan_s
        self._lib = ctypes.CDLL(str(self.so_path))
        self._bind()
        self._h = self._lib.plc_new()
        if not self._h:
            raise RuntimeError("plc_new returned NULL")

    def _bind(self) -> None:
        L = self._lib
        L.plc_new.restype = ctypes.c_void_p
        L.plc_free.argtypes = [ctypes.c_void_p]
        L.plc_step.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        L.plc_count.argtypes = [ctypes.c_void_p]
        L.plc_count.restype = ctypes.c_int
        L.plc_meta.argtypes = [ctypes.c_void_p, ctypes.c_int] + [ctypes.POINTER(ctypes.c_int)] * 4
        L.plc_read.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 4
        L.plc_read.restype = ctypes.c_int32
        L.plc_write.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 4 + [ctypes.c_int32]
        L.plc_write.restype = ctypes.c_int
        L.plc_state_size.restype = ctypes.c_int
        L.plc_save.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        L.plc_save.restype = ctypes.c_int64
        L.plc_restore.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]

    # --- snapshot / restore (oracle counterfactual replay) ---------------
    def save_state(self) -> tuple[bytes, int]:
        n = self._lib.plc_state_size()
        buf = ctypes.create_string_buffer(n)
        t_ns = self._lib.plc_save(self._h, buf)
        return buf.raw, int(t_ns)

    def restore_state(self, state: tuple[bytes, int]) -> None:
        blob, t_ns = state
        buf = ctypes.create_string_buffer(blob, len(blob))
        self._lib.plc_restore(self._h, buf, int(t_ns))

    # --- lifecycle -------------------------------------------------------
    def close(self) -> None:
        if getattr(self, "_h", None):
            self._lib.plc_free(self._h)
            self._h = None

    def __del__(self):  # best-effort
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- stepping --------------------------------------------------------
    def step(self, dt_s: float | None = None) -> None:
        dt = self.scan_s if dt_s is None else dt_s
        self._lib.plc_step(self._h, int(round(dt * 1e9)))

    # --- raw located access ---------------------------------------------
    def _read(self, area: int, byte_index: int, bit_index: int, is_bit: int) -> int:
        return int(self._lib.plc_read(self._h, area, byte_index, bit_index, is_bit))

    def _write(self, area: int, byte_index: int, bit_index: int, is_bit: int, value: int) -> bool:
        return bool(self._lib.plc_write(self._h, area, byte_index, bit_index, is_bit, int(value)))

    # --- by IEC address --------------------------------------------------
    def read_iec(self, iec: str) -> int:
        return self._read(*parse_iec_address(iec))

    def write_iec(self, iec: str, value: int) -> bool:
        area, byte_index, bit_index, is_bit = parse_iec_address(iec)
        return self._write(area, byte_index, bit_index, is_bit, value)

    # --- by Modbus (table, address): what the soft-PLC Modbus server uses -
    def read_modbus(self, table: str, address: int) -> int:
        return self._read(*modbus_to_iec(table, address))

    def write_modbus(self, table: str, address: int, value: int) -> bool:
        area, byte_index, bit_index, is_bit = modbus_to_iec(table, address)
        return self._write(area, byte_index, bit_index, is_bit, value)

    # --- introspection ---------------------------------------------------
    def located(self) -> list[tuple[int, int, int, int]]:
        n = self._lib.plc_count(self._h)
        out = []
        a, s, by, bi = (ctypes.c_int() for _ in range(4))
        for i in range(n):
            self._lib.plc_meta(self._h, i, ctypes.byref(a), ctypes.byref(s),
                               ctypes.byref(by), ctypes.byref(bi))
            out.append((a.value, s.value, by.value, bi.value))
        return out
