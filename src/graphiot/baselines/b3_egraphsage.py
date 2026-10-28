"""B3 — E-GraphSAGE (Lo et al., NOMS 2022), the authors' public code run unchanged.

Network view (paper §V-E): the detector sees flows and Modbus transactions. Each
Modbus transaction is an edge ip:port -> ip:port with network-level features (no
decoded engineering value); every scored write command is mapped to its write
transaction, and B3's score for the command is that edge's P(attack).

The model is supervised, so (Q5, D15) B3 is trained on a *labelled* run — the pilot
campaign, disjoint from the main campaign — and scores the main run's commands. It
runs in its own environment ``.venv-b3`` because DGL cannot share the main venv's
torch; this module only builds the edge tables and reads the scores back.

Transductivity note (D15): E-GraphSAGE scores an edge from the whole test-period
graph, so a command's score can draw on later transactions between the same hosts.
That is how the public code works; a strict causal prefix would be an architectural
change, which the protocol forbids. Node features are all ones (the paper's Eq. 1),
so the per-edge signal is dominated by the transaction's own features.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..config.io import load_config
from ..config.schema import ExperimentConfig
from ..data.monitor_view import MonitorView, load_monitor_view
from ..model.prepare import prepare
from ..model.runner import ROOT, load_dataset

WRITE_FCS = (5, 6, 15, 16)
FEATURES = ["f_fc", "f_address", "f_count", "f_nwrite", "f_nread", "f_req_bytes", "f_resp_bytes"]
B3_PY = ROOT / ".venv-b3" / "bin" / "python"
WORKER = Path(__file__).resolve().parent / "_b3_worker.py"


def edge_table(view: MonitorView, cmd_labels: pd.DataFrame | None) -> pd.DataFrame:
    """One row per Modbus transaction (the network edges B3 classifies).

    ``cmd_labels``: command_idx, ts, tag, value, harmful (oracle) for the commands to
    map onto write transactions. Each write command is matched to the nearest write
    transaction to the same PLC (same second); that edge carries its command_idx and,
    when labelling, the oracle label. Unmatched transactions are benign edges.
    """
    mb = view.modbus.copy()
    if mb.empty:
        return pd.DataFrame(columns=["src", "dst", "label", "command_idx", *FEATURES])
    mb["src"] = mb["src_ip"].astype(str) + ":" + mb["src_port"].astype(str)
    mb["dst"] = mb["dst_ip"].astype(str) + ":" + mb["dst_port"].astype(str)
    mb["f_fc"] = mb["function_code"].astype(float)
    mb["f_address"] = mb["address"].astype(float)
    mb["f_count"] = mb["count"].astype(float)
    mb["f_nwrite"] = mb["write_values"].apply(lambda v: len(v) if isinstance(v, (list, np.ndarray)) else 0).astype(float)
    mb["f_nread"] = mb["read_values"].apply(lambda v: len(v) if isinstance(v, (list, np.ndarray)) else 0).astype(float)
    mb["f_req_bytes"] = mb.get("request_bytes", 0.0).astype(float)
    mb["f_resp_bytes"] = mb.get("response_bytes", 0.0).astype(float)
    mb = mb.sort_values("ts").reset_index(drop=True)
    mb["command_idx"] = -1
    mb["label"] = 0

    if cmd_labels is not None and len(cmd_labels):
        writes = mb[mb["function_code"].isin(WRITE_FCS)].copy()
        for r in cmd_labels.itertuples():
            cand = writes[(writes["dst_ip"] == getattr(r, "dst_ip", None)) if "dst_ip" in cmd_labels else
                          np.ones(len(writes), bool)]
            pool = writes if cand.empty else cand
            if pool.empty:
                continue
            j = (pool["ts"] - r.ts).abs().idxmin()
            if abs(mb.at[j, "ts"] - r.ts) <= 2.0 and mb.at[j, "command_idx"] == -1:
                mb.at[j, "command_idx"] = int(r.command_idx)
                mb.at[j, "label"] = int(getattr(r, "harmful", 0) or 0)
    return mb[["src", "dst", "label", "command_idx", *FEATURES]]


def _command_frame(run: Path, cfg: ExperimentConfig, splits: list[str] | None, labelled: bool) -> pd.DataFrame:
    """Commands to map onto edges, with dst PLC ip and (if labelled) the oracle label."""
    cmds = prepare(load_dataset(run, cfg), "full").cmd
    if splits:
        cmds = cmds[cmds["split"].isin(splits)]
    cmds = cmds[cmds["decision"] == "forwarded"]
    cols = {"command_idx": cmds["command_idx"], "ts": cmds["ts"]}
    net = cfg.testbed.network
    prefix_ip = {"mix": net.plc_ips.get("PLC_MIX"), "coat": net.plc_ips.get("PLC_COAT"),
                 "slit": net.plc_ips.get("PLC_SLIT")}
    cols["dst_ip"] = cmds["tag"].str.split(".").str[0].map(prefix_ip)
    out = pd.DataFrame(cols)
    if labelled:
        lab = pd.read_parquet(run / "oracle" / "command_labels.parquet")[["command_idx", "harmful"]]
        out = out.merge(lab, on="command_idx", how="left")
        out["harmful"] = out["harmful"].fillna(False).astype(int)
    return out


def run_b3(main_run: Path, train_run: Path, *, seed: int = 0, epochs: int = 1500,
           cfg: ExperimentConfig | None = None, on_log=lambda s: None) -> pd.DataFrame:
    """Train B3 on ``train_run`` (labelled) and score ``main_run``'s test commands.
    Returns command_idx, score. Raises if the B3 environment is missing."""
    if not B3_PY.exists():
        raise RuntimeError("the B3 environment is missing; create it with requirements/b3.txt "
                           "(see README: baselines). It cannot share the main venv (DGL vs torch 2.14).")
    main_run, train_run = Path(main_run), Path(train_run)
    cfg = cfg or load_config(main_run / "config.yaml")
    tcfg = load_config(train_run / "config.yaml")
    train_view = load_monitor_view(train_run, tag_map=ROOT / tcfg.paths.tag_map)
    main_view = load_monitor_view(main_run, tag_map=ROOT / cfg.paths.tag_map)
    train_edges = edge_table(train_view, _command_frame(train_run, tcfg, None, labelled=True))
    test_edges = edge_table(main_view, _command_frame(main_run, cfg, ["val", "test", "redteam"], labelled=False))
    n_scored = int((test_edges["command_idx"] >= 0).sum())
    on_log(f"B3 edges: {len(train_edges)} train ({int(train_edges['label'].sum())} attack), "
           f"{len(test_edges)} test ({n_scored} scored commands)")
    if n_scored == 0:
        n_wr = int(main_view.modbus["function_code"].isin(WRITE_FCS).sum()) if not main_view.modbus.empty else 0
        raise RuntimeError(
            "B3 found no write transaction on the wire to score (the Modbus capture has "
            f"{n_wr} write transactions). B3 is a network detector; it needs a capture with "
            "gateway writes (pcap coverage ≥ 0.95). This run's capture is incomplete — re-run "
            "the campaign with capture, or exclude B3 for this run.")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        train_edges.to_parquet(tmp / "train.parquet", index=False)
        test_edges.to_parquet(tmp / "test.parquet", index=False)
        (tmp / "meta.json").write_text(json.dumps({"seed": seed, "epochs": epochs}))
        proc = subprocess.run([str(B3_PY), str(WORKER), str(tmp / "train.parquet"),
                               str(tmp / "test.parquet"), str(tmp / "out.parquet"), str(tmp / "meta.json")],
                              capture_output=True, text=True, timeout=3600, cwd=str(ROOT))
        if proc.returncode != 0:
            raise RuntimeError(f"B3 worker failed:\n{proc.stderr[-2000:]}")
        scores = pd.read_parquet(tmp / "out.parquet")
        info = json.loads((tmp / "meta_out.json").read_text()) if (tmp / "meta_out.json").exists() else {}
    on_log(f"B3 trained: {info.get('epochs')} epochs, train acc {info.get('train_acc'):.3f}" if info else "B3 done")
    return scores
