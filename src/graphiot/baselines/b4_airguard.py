"""B4 — AIRGuard (Qin et al., 2026), the authors' public code run unchanged.

Agent view (paper §V-E): AIRGuard is configured with our six tool specifications
and decides, per tool call, whether to allow it. We call its own ``check_action``
5-layer pipeline (``third_party/airguard/repo``) verbatim, building the Resource /
AuthorityContext / Action exactly as the authors' MCP proxy does (same default
untrusted resource and user-authority context, same tool-name normalisation). The
only unused import of the proxy is the ``mcp`` server package; we stub it so the
authors' ``_normalize`` loads, and change no AIRGuard logic.

AIRGuard returns an enforcement decision, not a continuous score (Q6): B4 gives a
binary alarm (its ``should_deny_outcome``), so AUROC/AUPRC are n/a and only recall
and FPR at its single operating point are reported. Finding to report: a fieldbus
``write_tag`` normalises to ``file.write``, which AIRGuard routes to its heuristic
fallback, not its LLM — so configuring B4 with the operator's LLM does not change
its decisions on setpoint writes. AIRGuard targets prompt injection via untrusted
tool *outputs*, a different threat than context-shifted setpoints.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd

from ..config.io import load_config
from ..config.schema import ExperimentConfig
from ..model.prepare import prepare
from ..model.runner import ROOT, load_dataset

AIRGUARD_SRC = ROOT / "third_party" / "airguard" / "repo" / "src"


def _load_airguard():
    """Import the authors' modules; stub the proxy's unused ``mcp`` dependency."""
    if str(AIRGUARD_SRC) not in sys.path:
        sys.path.insert(0, str(AIRGUARD_SRC))
    for name in ("mcp", "mcp.server", "mcp.server.fastmcp", "mcp.types"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
    sys.modules["mcp.server.fastmcp"].FastMCP = object
    sys.modules["mcp.types"].Tool = object
    sys.modules["mcp.types"].TextContent = object
    from airguard import guard, types as at  # noqa: E402
    from airguard.audit_containment import LedgerStore  # noqa: E402
    from airguard.integrations import mcp_proxy  # noqa: E402  (for _normalize, _resource, _authority)
    from airguard.output_guard import should_deny_outcome  # noqa: E402
    return guard, at, LedgerStore, mcp_proxy, should_deny_outcome


# AIRGuard treats a tool call as an action on a resource; our gateway tools map to its
# tool names so its own _normalize classifies them (write_tag -> file.write, etc.).
TOOL_ALIAS = {"write_tag": "write_tag", "load_recipe": "write_recipe", "acknowledge_alarm": "write_ack",
              "read_tag": "read_tag", "read_work_order": "read_work_order", "read_stage_status": "read_status"}


def score_b4(run: Path, *, cfg: ExperimentConfig | None = None, on_log=lambda s: None) -> tuple[pd.DataFrame, dict]:
    """Decide every forwarded write command with AIRGuard. Returns (command_idx, score in
    {0,1}) where 1 = AIRGuard denies (its alarm), and a metadata dict (binary detector)."""
    run = Path(run)
    cfg = cfg or load_config(run / "config.yaml")
    guard, at, LedgerStore, mcp_proxy, should_deny_outcome = _load_airguard()
    cmds = prepare(load_dataset(run, cfg), "full").cmd
    cmds = cmds[cmds["tool"].isin(["write_tag", "load_recipe", "acknowledge_alarm"])]

    ledger = LedgerStore()
    rows, denied, normals = [], 0, {}
    for i, r in enumerate(cmds.itertuples()):
        tool = TOOL_ALIAS.get(r.tool, r.tool)
        na = mcp_proxy._normalize(tool)                           # authors' tool-name classifier
        normals[na.value] = normals.get(na.value, 0) + 1
        action = at.Action(action_id=f"cmd_{r.command_idx}", name=tool,
                           args={"name": r.tag, "value": float(r.value)},
                           source_resource_id="gateway-tool",
                           required_capabilities=[na.value.split(".")[0]], normalized_action=na)
        decision = guard.check_action(resource=mcp_proxy._resource, authority_ctx=mcp_proxy._authority,
                                      action=action, history=[], llm_client=None, ledger=ledger)
        ov = decision.outcome.value if hasattr(decision.outcome, "value") else str(decision.outcome)
        deny = bool(should_deny_outcome(ov))
        denied += deny
        rows.append({"command_idx": int(r.command_idx), "score": 1.0 if deny else 0.0})
    on_log(f"B4 AIRGuard: {len(rows)} commands, {denied} denied; normalized actions {normals}")
    meta = {"binary": True, "denied": denied, "total": len(rows), "normalized_actions": normals,
            "note": "AIRGuard decision (allow/audit = pass, else deny); AUROC n/a (Q6). "
                    "write_tag -> file.write -> heuristic fallback, LLM not consulted for setpoint writes."}
    return pd.DataFrame(rows), meta
