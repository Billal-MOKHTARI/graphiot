"""Load / validate / save / hash experiment configs, and create run folders.

A run folder stores the exact YAML it was started with, its SHA-256, the git
commit (plus a dirty flag and a hash of the source tree, so a run made before
the first commit is still traceable), the seeds, the device actually used and
package versions (task §8).
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import yaml

from .schema import ExperimentConfig, reject_secrets

_ROOT = Path(__file__).resolve().parents[3]


class ConfigError(ValueError):
    pass


def parse_config(data: dict) -> ExperimentConfig:
    if not isinstance(data, dict):
        raise ConfigError("config root must be a mapping")
    reject_secrets(data)
    return ExperimentConfig.model_validate(data)


def load_config(path: str | Path) -> ExperimentConfig:
    text = Path(path).read_text()
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML parse error: {exc}") from exc
    return parse_config(data)


def config_to_dict(cfg: ExperimentConfig) -> dict:
    return cfg.model_dump(mode="json")


def dump_yaml(cfg: ExperimentConfig) -> str:
    return yaml.safe_dump(config_to_dict(cfg), sort_keys=False, allow_unicode=True)


def save_config(cfg: ExperimentConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_yaml(cfg))
    return path


def config_hash(cfg: ExperimentConfig) -> str:
    """SHA-256 of the canonical (sorted-key JSON) form of the *validated* config,
    so formatting or key order in the YAML never changes the hash."""
    canonical = json.dumps(config_to_dict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def json_schema() -> dict:
    """JSON Schema of the config — the GUI generates its form from this."""
    return ExperimentConfig.model_json_schema()


def resolve_device(cfg: ExperimentConfig) -> str:
    """'cpu' | 'cuda:<i>' — resolves 'auto' against what this machine has."""
    if cfg.device == "cpu":
        return "cpu"
    try:
        import torch  # optional until M5
        has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > cfg.gpu_index
    except Exception:
        has_cuda = False
    if cfg.device == "cuda" and not has_cuda:
        raise ConfigError(f"device=cuda requested but CUDA device {cfg.gpu_index} is unavailable")
    return f"cuda:{cfg.gpu_index}" if has_cuda else "cpu"


# --------------------------------------------------------------------------- #
def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def source_tree_hash() -> str:
    """Hash of every tracked-source file under src/, plc/src/, plant/ (content +
    path), so a run is traceable even without a commit."""
    h = hashlib.sha256()
    for base in ("src", "plc/src", "plant"):
        for f in sorted((_ROOT / base).rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts and f.suffix not in (".pyc",):
                h.update(str(f.relative_to(_ROOT)).encode())
                h.update(f.read_bytes())
    return h.hexdigest()


def _versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ("numpy", "pandas", "pyarrow", "pydantic", "pymodbus", "torch", "torch-geometric",
                "xgboost-cpu", "xgboost", "scikit-learn", "fastapi"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def create_run(cfg: ExperimentConfig, *, run_id: str | None = None,
               device: str | None = None) -> Path:
    """Create ``<output_root>/<name>_<timestamp>/`` with config.yaml + meta.json."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = run_id or f"{cfg.experiment.name}_{stamp}"
    run_dir = (_ROOT / cfg.experiment.output_root / run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    for sub in ("capture", "groundtruth", "oracle", "graph", "models", "scores", "results", "logs"):
        (run_dir / sub).mkdir()
    save_config(cfg, run_dir / "config.yaml")
    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    meta = {
        "run_id": run_id,
        "created_utc": stamp,
        "config_sha256": config_hash(cfg),
        "git_commit": commit,
        "git_dirty": bool(dirty) if dirty is not None else None,
        "source_tree_sha256": source_tree_hash(),
        "seeds": cfg.seeds,
        "sim_seed": cfg.testbed.sim_seed,
        "schedule_seed": cfg.campaign.schedule_seed,
        "device_requested": cfg.device,
        "device_resolved": device or resolve_device(cfg),
        "testbed_backend": cfg.testbed.backend,
        "for_paper": cfg.experiment.for_paper,
        "versions": _versions(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return run_dir
