"""Custom Modbus TCP server (0-based addressing, exceptions, read-only tables)
and the field-I/O server, exercised with a real pymodbus client."""

import asyncio

import pytest
from pymodbus.client import AsyncModbusTcpClient

from graphiot.modbus.server import ModbusTcpServer, RegisterBank
from graphiot.plant.tags import Plc
from graphiot.sim.field_io import FieldIOServer


def _run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=20))


def test_server_roundtrip_and_exceptions(unused_port_factory=None):
    async def main():
        bank = RegisterBank({"holding": 10, "input": 4, "coil": 16, "discrete": 8},
                            read_only={"input", "discrete"})
        bank.set_local("input", 0, [11, 22, 33, 44])
        log = []
        srv = ModbusTcpServer(bank, "127.0.0.1", 0, on_transaction=log.append)
        await srv.start()
        port = srv._server.sockets[0].getsockname()[1]
        cl = AsyncModbusTcpClient("127.0.0.1", port=port)
        await cl.connect()
        try:
            await cl.write_registers(2, [7, 8, 9])
            r = await cl.read_holding_registers(0, count=6)
            assert r.registers == [0, 0, 7, 8, 9, 0]          # exact 0-based addressing
            assert (await cl.read_input_registers(1, count=2)).registers == [22, 33]
            await cl.write_coil(5, True)
            assert (await cl.read_coils(0, count=8)).bits[5] is True
            # out of range -> ILLEGAL DATA ADDRESS (2)
            ex = await cl.read_holding_registers(8, count=5)
            assert ex.isError() and ex.exception_code == 2
            # input registers are read-only on the wire (no FC writes them); a
            # write to holding beyond the table is rejected
            ex2 = await cl.write_register(50, 1)
            assert ex2.isError() and ex2.exception_code == 2
        finally:
            cl.close()
            await srv.stop()
        fcs = [t.function_code for t in log]
        assert 16 in fcs and 3 in fcs and 4 in fcs and 5 in fcs
        assert any(t.exception == 2 for t in log)
        assert all(t.request_bytes >= 12 and t.response_bytes >= 9 for t in log)

    _run(main())


def test_field_io_maps_actuators_and_sensors(tags):
    async def main():
        srv = FieldIOServer(tags=tags, seed=3, host="127.0.0.1",
                            ports={Plc.MIX: 0, Plc.COAT: 0, Plc.SLIT: 0})
        task = asyncio.create_task(srv.run())
        await asyncio.sleep(0.3)
        port = srv.servers[Plc.COAT]._server.sockets[0].getsockname()[1]
        cl = AsyncModbusTcpClient("127.0.0.1", port=port)
        await cl.connect()
        try:
            # What the OpenPLC master writes with FC16 from %QW0..%QW7.
            await cl.write_registers(0, [50, 200, 0, 0, 0, 40, 150, 180])
            await asyncio.sleep(0.25)
            a = srv.read_actuators()
            assert a["coat.drive_cmd"] == 5.0 and a["coat.pump_cmd"] == 200
            assert a["coat.fan_cmd"] == 4.0 and a["coat.tension_cmd"] == 150
            assert a["coat.gap_cmd"] == 180
            # Sensors are served read-only at FC4 0.. in tag-map order.
            r = await cl.read_input_registers(0, count=8)
            assert r.registers == srv.banks[Plc.COAT].tables["input"][:8]
            # The physics responds to the actuators in real time.
            await asyncio.sleep(1.5)
            assert srv.physics.coat["web_speed"] > 3.0
        finally:
            cl.close()
            srv.stop()
            await asyncio.sleep(0.2)
            task.cancel()

    _run(main())
