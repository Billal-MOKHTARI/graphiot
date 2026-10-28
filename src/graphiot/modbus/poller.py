"""Block-read a PLC's tags the way an HMI or engineering tool does.

From the tag map, build a handful of contiguous block reads per PLC (one per
Modbus table/segment) instead of one request per tag, then decode every tag
from the block results into engineering units.
"""

from __future__ import annotations

from dataclasses import dataclass

from pymodbus.client import AsyncModbusTcpClient

from ..plant.tags import Plc, Table, Tag

_FC = {Table.COIL: 1, Table.DISCRETE: 2, Table.HOLDING: 3, Table.INPUT: 4}
_READ = {1: "read_coils", 2: "read_discrete_inputs", 3: "read_holding_registers",
         4: "read_input_registers"}


@dataclass(frozen=True)
class Block:
    fc: int
    start: int
    count: int
    tags: tuple[Tag, ...]


def plan_blocks(tags: dict[str, Tag], plc: Plc, *, readable_only: bool = True) -> list[Block]:
    """One block per (table, segment); holding registers split at 1024 (%QW vs %MW)."""
    groups: dict[tuple[int, int], list[Tag]] = {}
    for t in tags.values():
        if t.plc is not plc or (readable_only and not t.agent_readable):
            continue
        fc = _FC[t.table]
        segment = 1 if (t.table is Table.HOLDING and t.modbus_address >= 1024) else 0
        groups.setdefault((fc, segment), []).append(t)
    blocks = []
    for (fc, _), ts in sorted(groups.items()):
        lo = min(t.modbus_address for t in ts)
        hi = max(t.modbus_address for t in ts)
        blocks.append(Block(fc, lo, hi - lo + 1, tuple(sorted(ts, key=lambda t: t.modbus_address))))
    return blocks


def decode_block(block: Block, values: list[int]) -> dict[str, float]:
    out = {}
    for t in block.tags:
        raw = int(values[t.modbus_address - block.start])
        if t.dtype.value == "int" and raw > 32767:
            raw -= 65536
        out[t.name] = t.decode(raw)
    return out


async def read_blocks(client: AsyncModbusTcpClient, blocks: list[Block]) -> dict[str, float]:
    """Read every block; tags of a failed block are simply absent from the result."""
    out: dict[str, float] = {}
    for b in blocks:
        r = await getattr(client, _READ[b.fc])(b.start, count=b.count, device_id=1)
        if r.isError():
            continue
        vals = [int(x) for x in r.bits[:b.count]] if b.fc in (1, 2) else list(r.registers)
        out.update(decode_block(b, vals))
    return out
