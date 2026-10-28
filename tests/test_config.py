"""Experiment config schema: validation rules the paper and the task impose."""

import copy

import pytest
import yaml
from pydantic import ValidationError

from graphiot.config.io import (ConfigError, config_hash, dump_yaml, load_config, parse_config)


@pytest.fixture
def pilot_dict(root):
    return yaml.safe_load((root / "configs" / "pilot.yaml").read_text())


@pytest.mark.parametrize("name", ["pilot", "main", "dev_smoke"])
def test_shipped_configs_validate(root, name):
    load_config(root / "configs" / f"{name}.yaml")


def test_secret_in_yaml_is_rejected(pilot_dict):
    d = copy.deepcopy(pilot_dict)
    d["llm"]["operator"]["api_key"] = "sk-live-123"
    with pytest.raises(ValueError, match="looks like a secret"):
        parse_config(d)


def test_env_var_name_is_allowed(pilot_dict):
    d = copy.deepcopy(pilot_dict)
    d["llm"]["red_team"] = {"provider": "gemini", "model": "gemini-x", "family": "gemini",
                            "api_key_env": "GEMINI_API_KEY"}
    cfg = parse_config(d)
    assert cfg.llm.red_team.api_key_env == "GEMINI_API_KEY"
    assert "GEMINI_API_KEY" in dump_yaml(cfg) and "sk-" not in dump_yaml(cfg)


def test_same_family_rejected(pilot_dict):
    d = copy.deepcopy(pilot_dict)
    d["llm"]["red_team"]["family"] = "llama"
    with pytest.raises(ValidationError, match="different LLM family"):
        parse_config(d)


def test_unknown_key_rejected(pilot_dict):
    d = copy.deepcopy(pilot_dict)
    d["oracle"]["tau_typo"] = 0.1
    with pytest.raises(ValidationError):
        parse_config(d)


def test_splits_must_sum_to_one(pilot_dict):
    d = copy.deepcopy(pilot_dict)
    d["splits"] = {"train": 0.7, "val": 0.2, "test": 0.2}
    with pytest.raises(ValidationError, match="sum to 1"):
        parse_config(d)


def test_paper_runs_refuse_dev_backend(pilot_dict):
    d = copy.deepcopy(pilot_dict)
    d["testbed"] = {"backend": "dev"}
    with pytest.raises(ValidationError, match="for_paper"):
        parse_config(d)


def test_ip_outside_subnet_rejected(pilot_dict):
    d = copy.deepcopy(pilot_dict)
    d["testbed"]["network"] = {"hmi_ip": "10.99.0.5"}
    with pytest.raises(ValidationError, match="not in"):
        parse_config(d)


def test_hash_is_canonical(pilot_dict):
    a = parse_config(copy.deepcopy(pilot_dict))
    # Same content, different key order and explicit defaults -> same hash.
    d = copy.deepcopy(pilot_dict)
    d = {k: d[k] for k in reversed(list(d))}
    d["graph"] = {"window_s": 10.0, "stride_s": 5.0}
    b = parse_config(d)
    assert config_hash(a) == config_hash(b)
    d["graph"]["stride_s"] = 4.0
    assert config_hash(parse_config(d)) != config_hash(a)


def test_roundtrip_yaml(pilot_dict, tmp_path):
    from graphiot.config.io import save_config
    cfg = parse_config(pilot_dict)
    p = save_config(cfg, tmp_path / "x.yaml")
    assert config_hash(load_config(p)) == config_hash(cfg)


def test_bad_yaml_raises_config_error(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("experiment: [unclosed")
    with pytest.raises(ConfigError):
        load_config(p)
