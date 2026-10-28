"""The gateway's southbound port survives a connection the PLC closed while idle.

Pilot run of 26/10: after six idle minutes a work-order publish to PLC_MIX failed,
the gateway answered HTTP 500 and the campaign stopped. The port now drops a
failed connection and retries the request once on a fresh one.
"""

import asyncio

from graphiot.gateway.southbound import PymodbusPort
from graphiot.modbus.server import ModbusTcpServer, RegisterBank


class _ClosableServer(ModbusTcpServer):
    """Records server-side connections so the test can close them (idle timeout)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.writers = []

    async def _handle(self, reader, writer):
        self.writers.append(writer)
        await super()._handle(reader, writer)

    def drop_all(self):
        for w in self.writers:
            w.close()
        self.writers.clear()


def test_port_reconnects_after_server_side_close():
    async def main():
        bank = RegisterBank({"holding": 16, "input": 4, "coil": 8, "discrete": 8})
        srv = _ClosableServer(bank, "127.0.0.1", 0)
        await srv.start()
        port = PymodbusPort("127.0.0.1", srv._server.sockets[0].getsockname()[1], timeout_s=1.0)
        try:
            await port.write("holding", 12, [42])
            assert await port.read("holding", 12, 1) == [42]
            srv.drop_all()                                  # the PLC closes the idle connection
            await asyncio.sleep(0.2)
            await port.write("holding", 12, [43])           # must not raise
            assert await port.read("holding", 12, 1) == [43]
            await port.write("coil", 3, [1])
            assert (await port.read("coil", 3, 1)) == [1]
        finally:
            port.close()
            srv.drop_all()
            await srv.stop()
    asyncio.run(asyncio.wait_for(main(), timeout=30))


def test_modbus_exception_is_not_retried():
    async def main():
        bank = RegisterBank({"holding": 4, "input": 4, "coil": 8, "discrete": 8})
        srv = _ClosableServer(bank, "127.0.0.1", 0)
        await srv.start()
        port = PymodbusPort("127.0.0.1", srv._server.sockets[0].getsockname()[1], timeout_s=1.0)
        try:
            try:
                await port.write("holding", 100, [1])       # outside the table -> exception 2
                raise AssertionError("expected an IOError")
            except IOError as exc:
                assert "exception 2" in str(exc)
        finally:
            port.close()
            srv.drop_all()
            await srv.stop()
    asyncio.run(asyncio.wait_for(main(), timeout=30))
