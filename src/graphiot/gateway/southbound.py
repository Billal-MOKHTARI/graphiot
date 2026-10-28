"""Southbound Modbus TCP: one persistent pymodbus client per PLC."""

from __future__ import annotations

import asyncio

from pymodbus.client import AsyncModbusTcpClient

_READ = {"coil": "read_coils", "discrete": "read_discrete_inputs",
         "holding": "read_holding_registers", "input": "read_input_registers"}


class PymodbusPort:
    """ModbusPort over an AsyncModbusTcpClient, reconnecting on demand."""

    def __init__(self, host: str, port: int = 502, *, timeout_s: float = 1.0, unit: int = 1):
        self.host, self.port, self.unit = host, port, unit
        self.timeout_s = timeout_s
        # Created lazily: pymodbus binds a client to the event loop running at
        # construction time, and the service builds its ports before uvicorn's loop.
        self._client: AsyncModbusTcpClient | None = None
        self._lock: asyncio.Lock | None = None

    async def _ensure(self) -> None:
        if self._client is None:
            self._client = AsyncModbusTcpClient(self.host, port=self.port,
                                                timeout=self.timeout_s, retries=1)
        if not self._client.connected:
            ok = await self._client.connect()
            if not ok:
                raise ConnectionError(f"cannot connect to {self.host}:{self.port}")

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _drop(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None

    async def _request(self, fn):
        """Run one request; on a transport failure drop the connection and retry once
        on a fresh one. A connection the PLC closed while idle passes ``connected``
        and then fails the next request (pilot run of 26/10: HTTP 500 on a work-order
        publish after six idle minutes). A Modbus exception response is not retried."""
        async with self._get_lock():
            for attempt in (0, 1):
                try:
                    await self._ensure()
                    r = await fn(self._client)
                except Exception:
                    self._drop()
                    if attempt:
                        raise
                    continue
                if r.isError():
                    raise IOError(f"exception {getattr(r, 'exception_code', '?')}")
                return r

    async def read(self, table: str, address: int, count: int) -> list[int]:
        r = await self._request(lambda c: getattr(c, _READ[table])(address, count=count, device_id=self.unit))
        return [int(b) for b in r.bits[:count]] if table in ("coil", "discrete") else list(r.registers)

    async def write(self, table: str, address: int, values: list[int]) -> None:
        if table == "coil":
            await self._request(lambda c: c.write_coil(address, bool(values[0]), device_id=self.unit))
        elif table == "holding":
            await self._request(lambda c: c.write_register(address, values[0], device_id=self.unit))
        else:
            raise ValueError(f"table {table!r} is not writable")

    def close(self) -> None:
        self._drop()
