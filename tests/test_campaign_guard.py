"""Campaign guard: passive PLC liveness from the HMI log, and suspend detection."""

import json

from graphiot.campaign import power


def test_stale_plcs_from_hmi_log(tmp_path):
    f = tmp_path / "tags.jsonl"
    rows = [{"ts": 100.0 + i, "plc": p, "values": {}} for i in range(5) for p in ("PLC_MIX", "PLC_SLIT")]
    rows += [{"ts": 160.0, "plc": "PLC_SLIT", "values": {}}]
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    stale = power.stale_plcs(f, ["PLC_MIX", "PLC_SLIT", "PLC_COAT"], max_age_s=30, now=165.0)
    assert set(stale) == {"PLC_MIX", "PLC_COAT"}          # COAT never seen, MIX silent for 61 s
    assert power.stale_plcs(tmp_path / "missing.jsonl", ["PLC_MIX"], 30) == {}


def test_suspend_watch(monkeypatch):
    t = {"v": 1000.0}
    monkeypatch.setattr(power, "suspended_seconds", lambda: t["v"])
    w = power.SuspendWatch(threshold_s=1.0)
    t["v"] += 0.3
    assert w.check() == 0.0                               # clock jitter is not a suspend
    t["v"] += 7.0
    assert abs(w.check() - 7.3) < 1e-9
