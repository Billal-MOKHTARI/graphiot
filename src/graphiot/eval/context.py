"""Attach the monitor-visible context to each command, causally.

* stage phase: the last 1 Hz sample of the command's stage phase tag with a
  timestamp *strictly before* the command (causal scoring rule, §IV-A);
* task type: from the monitor's task table (task id -> type).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..plant.tags import PHASES

STAGE_OF_PREFIX = {"mix": "mixing", "coat": "coating", "slit": "slitting"}


def attach_context(commands: pd.DataFrame, tags: pd.DataFrame, tasks: pd.DataFrame) -> pd.DataFrame:
    df = commands.copy()
    df["prefix"] = df["tag"].str.split(".").str[0]
    df["stage"] = df["prefix"].map(STAGE_OF_PREFIX)
    df["phase"] = "unknown"
    for prefix in STAGE_OF_PREFIX:
        ph = tags[tags["tag"] == f"{prefix}.phase"].sort_values("ts")
        if ph.empty:
            continue
        ts = ph["ts"].to_numpy()
        vals = ph["value"].to_numpy()
        m = df["prefix"] == prefix
        # index of the last sample strictly before the command time
        idx = np.searchsorted(ts, df.loc[m, "ts"].to_numpy(), side="left") - 1
        df.loc[m, "phase"] = [PHASES.get(int(vals[i]), "unknown") if i >= 0 else "unknown" for i in idx]
    ttype = dict(zip(tasks["task_id"], tasks["task_type"])) if not tasks.empty else {}
    df["task_type"] = df["task_id"].map(ttype).fillna("unknown")
    return df
