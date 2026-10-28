"""The committed tag_map.csv is the contract; it must match the Python source,
and the catalogue must satisfy the Modbus-addressing and role invariants."""

from collections import defaultdict

from graphiot.plant import tags as T


def test_csv_matches_source(tag_map_path):
    assert tag_map_path.read_text() == T.to_csv(), (
        "plant/tag_map.csv is stale; regenerate with "
        "`python -m graphiot.plant.tags > plant/tag_map.csv`")


def test_names_unique():
    names = [t.name for t in T.CATALOG]
    assert len(names) == len(set(names))


def test_modbus_addresses_unique_per_plc_table():
    seen = defaultdict(set)
    for t in T.CATALOG:
        key = (t.plc, t.table)
        assert t.modbus_address not in seen[key], f"address clash for {t.name}"
        seen[key].add(t.modbus_address)


def test_roles_consistent():
    for t in T.CATALOG:
        if t.agent_writable:
            assert t.kind in (T.Kind.SETPOINT, T.Kind.COMMAND, T.Kind.ALARM_ACK), t.name
        if t.kind is T.Kind.SENSOR:
            assert t.iec_address.startswith("%IW") and not t.agent_writable, t.name
        if t.kind is T.Kind.SETPOINT:
            assert t.iec_address.startswith("%MW") and t.agent_writable, t.name


def test_encode_decode_roundtrip():
    for t in T.CATALOG:
        if t.dtype is T.DType.INT:
            v = (t.vmin + t.vmax) / 2.0
            assert abs(t.decode(t.encode(v)) - v) <= 1.0 / t.scale + 1e-9, t.name


def test_modbus_address_mapping():
    # %MW holding offset by 1024; %QW holding at n; %IW input at n; coil bit math.
    assert T.modbus_address_of(T.Table.HOLDING, "%MW0") == 1024
    assert T.modbus_address_of(T.Table.HOLDING, "%QW10") == 10
    assert T.modbus_address_of(T.Table.INPUT, "%IW3") == 3
    assert T.modbus_address_of(T.Table.COIL, "%QX2.0") == 16
    assert T.modbus_address_of(T.Table.DISCRETE, "%IX0.0") == 0


def test_all_three_plcs_present():
    plcs = {t.plc for t in T.CATALOG}
    assert plcs == set(T.Plc)
