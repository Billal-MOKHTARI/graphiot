"""Experiment configuration schema (paper §V; task §8).

One YAML file describes one experiment end to end: testbed, process files,
LLM providers, capture, oracle, graph, model, splits, baselines, attacker,
seeds and compute device. Every field carries a description and bounds, which
the GUI turns into a form (``ExperimentConfig.model_json_schema()``).

Secrets never live here. LLM keys and the OpenPLC password are read from the
environment / the GUI's in-memory store; the YAML only names the *environment
variable*. :func:`reject_secrets` refuses any key that looks like a secret, so a
key pasted into a YAML by mistake fails validation instead of being saved.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SECRET_KEY = re.compile(r"(api[_-]?key|secret|passw(or)?d|token|bearer|credential)", re.I)
_ALLOWED_SECRETISH = {"max_tokens"}


def reject_secrets(node, path: str = "") -> None:
    """Raise ValueError if any mapping key looks like an inline secret. Keys ending
    in ``_env`` name an environment variable (not a value) and are allowed."""
    if isinstance(node, dict):
        for k, v in node.items():
            key = str(k)
            if (_SECRET_KEY.search(key) and key not in _ALLOWED_SECRETISH
                    and not key.endswith("_env")):
                raise ValueError(
                    f"'{path}{key}' looks like a secret; secrets must come from .env or the "
                    f"GUI, never from YAML. Use an '*_env' field naming the variable instead.")
            reject_secrets(v, f"{path}{key}.")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            reject_secrets(v, f"{path}[{i}].")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------- #
class ExperimentMeta(_Strict):
    name: str = Field(..., pattern=r"^[A-Za-z0-9._-]+$", description="Run name prefix (folder-safe).")
    description: str = Field("", description="Free text shown in the GUI and stored with the run.")
    output_root: str = Field("runs", description="Directory where run folders are created.")
    for_paper: bool = Field(False, description="Mark runs whose numbers may appear in the paper. "
                                               "Refused for the dev backend.")


class Paths(_Strict):
    tag_map: str = Field("plant/tag_map.csv", description="Register/tag map used by every component.")
    recipes: str = Field("plant/recipes.yaml", description="Recipe book — TASK GENERATOR ONLY.")
    operating_limits: str = Field("plant/operating_limits.yaml", description="Equipment limits — ORACLE ONLY.")
    allow_list: str = Field("plant/allow_list.yaml", description="Gateway allow-list per task type (B1).")
    plant_documentation: str = Field("plant/plant_documentation.md", description="Text given to both agents.")
    mitre_dir: str = Field("plant/mitre", description="ATT&CK for ICS technique texts for the red team.")
    plc_src_dir: str = Field("plc/src", description="IEC 61131-3 ST programs (single source of truth).")


# --------------------------------------------------------------------------- #
class OpenPLCSettings(_Strict):
    image: str = Field("ghcr.io/autonomy-logic/openplc-runtime:v4.3.1",
                       description="OpenPLC Runtime v4 image (tag).")
    digest: str = Field("sha256:2ca9fc611f25c3ad0471b996a4bf27507ac112612292426ca6d5ca4bf06e97e6",
                        pattern=r"^sha256:[0-9a-f]{64}$", description="Pinned image index digest.")
    admin_user: str = Field("graphiot", description="Runtime REST user created on first deploy.")
    password_env: str = Field("OPENPLC_PASSWORD", description="Environment variable holding the runtime password.")
    modbus_port: int = Field(502, ge=1, le=65535, description="Agent-facing Modbus TCP port on each PLC.")


class NetworkSettings(_Strict):
    north_subnet: str = Field("10.77.1.0/24", description="Agents ↔ gateway (not captured).")
    cell_subnet: str = Field("10.77.2.0/24", description="Cell network seen by the monitor (captured).")
    field_subnet: str = Field("10.77.3.0/24", description="PLC ↔ remote I/O (not captured).")
    cell_bridge: str = Field("gi-cell", max_length=15, description="Linux bridge name of the cell network (capture interface).")
    gateway_north_ip: str = "10.77.1.10"
    gateway_ip: str = "10.77.2.10"
    hmi_ip: str = "10.77.2.20"
    ews_ip: str = "10.77.2.30"
    plc_ips: dict[str, str] = Field(default_factory=lambda: {
        "PLC_MIX": "10.77.2.101", "PLC_COAT": "10.77.2.102", "PLC_SLIT": "10.77.2.103"})
    plc_field_ips: dict[str, str] = Field(default_factory=lambda: {
        "PLC_MIX": "10.77.3.101", "PLC_COAT": "10.77.3.102", "PLC_SLIT": "10.77.3.103"})
    field_io_ip: str = "10.77.3.50"
    field_io_ports: dict[str, int] = Field(default_factory=lambda: {
        "PLC_MIX": 5021, "PLC_COAT": 5022, "PLC_SLIT": 5023})

    @model_validator(mode="after")
    def _ips_in_subnets(self):
        def inside(ip, net, what):
            if ipaddress.ip_address(ip) not in ipaddress.ip_network(net):
                raise ValueError(f"{what} {ip} not in {net}")
        inside(self.gateway_north_ip, self.north_subnet, "gateway_north_ip")
        for name, ip in [("gateway_ip", self.gateway_ip), ("hmi_ip", self.hmi_ip), ("ews_ip", self.ews_ip),
                         *self.plc_ips.items()]:
            inside(ip, self.cell_subnet, name)
        for name, ip in [("field_io_ip", self.field_io_ip), *self.plc_field_ips.items()]:
            inside(ip, self.field_subnet, name)
        return self


class Testbed(_Strict):
    backend: Literal["openplc_v4", "dev"] = Field(
        "openplc_v4", description="openplc_v4 = real OpenPLC Runtime in Docker; dev = soft-PLC on "
                                  "loopback (testing only, never for paper numbers).")
    openplc: OpenPLCSettings = Field(default_factory=OpenPLCSettings)
    network: NetworkSettings = Field(default_factory=NetworkSettings)
    plc_scan_ms: int = Field(100, ge=10, le=1000, description="PLC task interval (matches the ST CONFIGURATION).")
    sim_seed: int = Field(7, ge=0, description="Process-simulator noise seed (twin replays use the same seed).")
    gateway_cpus: float = Field(4.0, gt=0, description="CPU limit of the gateway container (edge cost).")
    gateway_mem_gb: float = Field(4.0, gt=0, description="Memory limit of the gateway container (edge cost).")


class GatewaySettings(_Strict):
    north_port: int = Field(8700, ge=1, le=65535, description="Northbound tool API port.")
    modbus_timeout_s: float = Field(1.0, gt=0, description="Southbound Modbus timeout.")
    admin_token_env: str = Field("GATEWAY_ADMIN_TOKEN", description="Environment variable holding "
                                 "the task-system (admin) token; agents never get it.")


# --------------------------------------------------------------------------- #
class LLMAgent(_Strict):
    provider: Literal["ollama", "gemini", "anthropic"] = Field(..., description="LLM provider.")
    model: str = Field(..., min_length=1, description="Model name for the provider.")
    family: str = Field(..., min_length=1, description="Model family (e.g. llama, qwen, gemini, claude). "
                                                       "Operator and red team must differ (§V-A).")
    base_url: str | None = Field(None, description="Ollama endpoint (e.g. the Cnam GPU server).")
    api_key_env: str | None = Field(None, description="Environment variable holding the API key "
                                                      "(Gemini/Claude). Never the key itself.")
    temperature: float = Field(0.2, ge=0.0, le=2.0, description="Ollama/Gemini only. The Claude API "
                               "no longer accepts sampling parameters (API version 2026-10-21).")
    max_tokens: int = Field(4096, ge=64, le=64000)
    request_timeout_s: float = Field(180.0, gt=0)
    effort: Literal["low", "medium", "high", "xhigh", "max"] = Field(
        "medium", description="Claude only: output_config.effort (thinking depth).")
    fallbacks: Literal["default", "off"] = Field(
        "default", description="Claude only: server-side refusal fallback. The model that actually "
                               "served each call is logged either way.")

    @model_validator(mode="after")
    def _provider_fields(self):
        if self.provider == "ollama" and not self.base_url:
            raise ValueError("ollama provider needs base_url")
        if self.provider in ("gemini", "anthropic") and not self.api_key_env:
            raise ValueError(f"{self.provider} provider needs api_key_env (the variable name)")
        return self


class Pacing(_Strict):
    think_time_median_s: float = Field(8.0, gt=0, description="Median think time between tool calls "
                                       "(same for both agents, hides LLM speed differences — Q17).")
    think_time_sigma: float = Field(0.5, ge=0, description="Log-normal sigma of the think time.")
    floor_s: float = Field(3.0, ge=0, description="Minimum gap between tool calls.")


class LLMSettings(_Strict):
    operator: LLMAgent = Field(default_factory=lambda: LLMAgent(
        provider="ollama", model="llama3.1:8b", family="llama", base_url="http://127.0.0.1:11434"))
    red_team: LLMAgent = Field(default_factory=lambda: LLMAgent(
        provider="ollama", model="qwen2.5:7b", family="qwen", base_url="http://127.0.0.1:11434"))
    pacing: Pacing = Field(default_factory=Pacing)
    max_tool_calls_per_task: int = Field(12, ge=1, le=100)

    @model_validator(mode="after")
    def _different_families(self):
        if self.operator.family.strip().lower() == self.red_team.family.strip().lower():
            raise ValueError("red-team agent must be from a different LLM family than the operator (§V-A)")
        return self


# --------------------------------------------------------------------------- #
class TaskSettings(_Strict):
    mean_interval_s: float = Field(90.0, gt=0, description="Mean time between operator tasks.")
    override_probability: float = Field(0.08, ge=0, le=1, description="Share of tasks that are "
                                        "unusual-but-legitimate overrides (§V-A).")
    task_type_weights: dict[str, float] = Field(default_factory=lambda: {
        "mixing_operation": 1.0, "coating_operation": 1.2, "coating_speed_adjust": 1.0,
        "dryer_tuning": 0.8, "slitting_operation": 1.0, "line_supervisor": 0.5, "monitoring": 0.7})


class RedTeamSettings(_Strict):
    episodes: int = Field(30, ge=0, description="Red-team episodes in the campaign.")
    techniques: list[Literal["T0836", "T0855", "T0831", "T0821"]] = Field(
        default_factory=lambda: ["T0836", "T0855", "T0831", "T0821"])
    max_episode_s: float = Field(480.0, gt=0, description="Upper bound on one takeover (Q15).")
    takeover: Literal["task_start"] = Field("task_start", description="Red team inherits a task at its start.")
    min_gap_s: float = Field(120.0, ge=0, description="Minimum benign time between two episodes.")
    prompt_version: str = Field("v1", description="Bumped once if the pilot tree exceeds 0.9 (§V-C).")


class CaptureSettings(_Strict):
    pcap: bool = Field(True, description="Capture the cell bridge with tcpdump (openplc_v4 backend).")
    snaplen: int = Field(512, ge=64, le=65535)
    rotate_s: int = Field(60, ge=5)
    flow_idle_timeout_s: float = Field(15.0, gt=0)
    flow_active_timeout_s: float = Field(5.0, gt=0, description="Interim flow export period; equals the snapshot stride so a snapshot only uses flows that finished before it.")
    tag_poll_hz: float = Field(1.0, gt=0, le=10, description="HMI tag polling rate (the 1 Hz tag stream).")


class CampaignSettings(_Strict):
    benign_duration_s: float = Field(3600.0, gt=0, description="Benign operation time (pilot: 1 h; main: ≥ 8 h).")
    red_team: RedTeamSettings = Field(default_factory=RedTeamSettings)
    tasks: TaskSettings = Field(default_factory=TaskSettings)
    capture: CaptureSettings = Field(default_factory=CaptureSettings)
    schedule_seed: int = Field(101, ge=0, description="Seed for task and takeover schedules.")
    inhibit_sleep: bool = Field(True, description="Block system sleep and lid-close suspend while the "
                                "campaign runs (a suspend leaves OpenPLC runtimes unresponsive).")
    plc_stale_s: float = Field(30.0, gt=0, description="Stop the campaign if a PLC has no HMI record "
                               "for this long (passive liveness check).")


# --------------------------------------------------------------------------- #
class OracleSettings(_Strict):
    mode: Literal["primary", "fallback"] = Field("primary", description="primary = twin replay; "
                                                 "fallback = logged tags (§V-B).")
    tau: float = Field(0.05, ge=0, le=1, description="Violation magnitude relative to limit width.")
    t_min_s: float = Field(10.0, gt=0, description="Minimum violation duration.")
    command_rule: Literal["episode_criterion", "strict"] = Field(
        "episode_criterion", description="'Back inside limits' after removal (Q10).")
    fallback_window_s: float = Field(120.0, gt=0, description="Fallback: violation within this "
                                     "window after a red-team command.")
    replay_settle_s: float = Field(120.0, ge=0, description="Replay horizon after the episode ends.")
    fidelity_max_rel_error: float = Field(0.05, gt=0, description="Max twin-vs-capture error (relative "
                                          "to limit width) before the primary oracle is distrusted.")


class GraphSettings(_Strict):
    window_s: float = Field(10.0, gt=0, description="Snapshot window Δ (§IV-A).")
    stride_s: float = Field(5.0, gt=0, description="Snapshot stride.")
    host_roles: list[str] = Field(default_factory=lambda: ["hmi", "ews", "gateway", "plc", "unknown"],
                                  description="Host role vocabulary (Table II + 'plc', Q11).")
    rolling_window_s: float = Field(60.0, gt=0, description="Tag rolling mean/std window.")

    @model_validator(mode="after")
    def _stride(self):
        if self.stride_s > self.window_s:
            raise ValueError("stride must not exceed the window")
        return self


class ModelSettings(_Strict):
    q_bins: int = Field(8, ge=2, le=64, description="Quantile bins Q.")
    d: int = Field(64, ge=8, le=1024, description="Hidden size d.")
    hgt_layers: int = Field(2, ge=1, le=8)
    heads: int = Field(4, ge=1, le=16)
    lr: float = Field(1e-3, gt=0)
    weight_decay: float = Field(0.0, ge=0)
    batch_snapshots: int = Field(32, ge=1, le=4096, description="Truncated-BPTT chunk (Q19).")
    max_epochs: int = Field(200, ge=1)
    patience: int = Field(10, ge=1, description="Early stopping on validation likelihood.")
    grad_clip: float = Field(1.0, gt=0)
    fpr_target: float = Field(0.01, gt=0, lt=1, description="Threshold at this FPR on benign validation.")
    checkpoint_every_steps: int = Field(200, ge=1)


class SplitSettings(_Strict):
    train: float = Field(0.7, gt=0, lt=1)
    val: float = Field(0.1, gt=0, lt=1)
    test: float = Field(0.2, gt=0, lt=1)
    exclude_after_redteam_s: float = Field(0.0, ge=0, description="Recovery margin removed from benign "
                                           "time after each episode (Q16; paper-literal = 0).")

    @model_validator(mode="after")
    def _sum(self):
        if abs(self.train + self.val + self.test - 1.0) > 1e-9:
            raise ValueError("train + val + test must sum to 1")
        return self


class Baselines(_Strict):
    b0: bool = True
    b0_unseen_score: Literal["max", "literal"] = Field("max", description="Q7: unseen bin score.")
    b1: bool = True
    b2: bool = True
    b2_n_estimators: int = Field(300, ge=1)
    b2_max_depth: int = Field(6, ge=1)
    b2_learning_rate: float = Field(0.1, gt=0)
    b3: bool = True
    b3_train_source: Literal["pilot", "anomal_e"] = Field("pilot", description="Q5.")
    b4: bool = True
    b5: bool = True
    b5_prompt_path: str = Field("src/graphiot/baselines/b5_prompt.txt")
    ablations: list[Literal["agent_only", "plant_only", "network_only", "no_task"]] = Field(
        default_factory=lambda: ["agent_only", "plant_only", "network_only", "no_task"])


class AttackerSettings(_Strict):
    budgets: list[int] = Field(default_factory=lambda: [0, 4, 8])
    episodes_per_detector: int = Field(50, ge=1)
    edits: list[Literal["insert_read", "interleave_benign", "split_change", "time_shift"]] = Field(
        default_factory=lambda: ["insert_read", "interleave_benign", "split_change", "time_shift"])
    max_time_shift_s: float = Field(30.0, gt=0)


class EvaluationSettings(_Strict):
    bootstrap_resamples: int = Field(2000, ge=100)
    ci: float = Field(0.95, gt=0, lt=1)
    mcnemar: Literal["majority_vote", "per_seed"] = Field("majority_vote", description="Q13.")
    holm: bool = True
    slices: list[Literal["all", "allow", "ctx"]] = Field(default_factory=lambda: ["all", "allow", "ctx"])


class PilotSettings(_Strict):
    min_context_shifted: int = Field(50, ge=1)
    tree_depth: int = Field(3, ge=1)
    tree_auprc_max: float = Field(0.9, gt=0, le=1)
    train_fraction: float = Field(0.7, gt=0, lt=1, description="Chronological split of the pilot.")


class EdgeCost(_Strict):
    cpus: int = Field(4, ge=1)
    mem_gb: float = Field(4.0, gt=0)
    n_commands: int = Field(1000, ge=10)


# --------------------------------------------------------------------------- #
class ExperimentConfig(_Strict):
    schema_version: Literal[1] = 1
    experiment: ExperimentMeta
    seeds: list[int] = Field(default_factory=lambda: [11, 22, 33], min_length=1,
                             description="Model seeds (paper: 3).")
    device: Literal["cpu", "cuda", "auto"] = Field("auto", description="Compute device for training/inference.")
    gpu_index: int = Field(0, ge=0, description="CUDA device index when device is cuda/auto.")
    paths: Paths = Field(default_factory=Paths)
    testbed: Testbed = Field(default_factory=Testbed)
    gateway: GatewaySettings = Field(default_factory=GatewaySettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    campaign: CampaignSettings = Field(default_factory=CampaignSettings)
    oracle: OracleSettings = Field(default_factory=OracleSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    splits: SplitSettings = Field(default_factory=SplitSettings)
    baselines: Baselines = Field(default_factory=Baselines)
    attacker: AttackerSettings = Field(default_factory=AttackerSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    pilot: PilotSettings = Field(default_factory=PilotSettings)
    edge_cost: EdgeCost = Field(default_factory=EdgeCost)

    @field_validator("seeds")
    @classmethod
    def _unique_seeds(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("seeds must be unique")
        return v

    @model_validator(mode="after")
    def _paper_runs_need_real_stack(self):
        if self.experiment.for_paper and self.testbed.backend == "dev":
            raise ValueError("for_paper runs require testbed.backend = openplc_v4 (dev backend "
                             "uses synthesized flows and a soft-PLC)")
        return self
