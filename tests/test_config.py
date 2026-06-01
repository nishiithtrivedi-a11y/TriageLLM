"""RouterConfig YAML loading + defaults."""
from pathlib import Path

import router_hook
from router_hook import CloudEscalationConfig, RouterConfig, load_config


def test_load_config_defaults_when_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.yaml"
    cfg = load_config(missing)
    assert cfg == RouterConfig()
    assert cfg.cloud_escalation == CloudEscalationConfig()


def test_load_config_reads_route_llm_block(tmp_path: Path) -> None:
    yaml_text = """
model_list: []
route_llm:
  use_llm_classifier: false
  llm_classifier_min_chars: 500
  critic_pass_threshold: 3
  cloud_escalation:
    enabled: true
    model: openai/gpt-5
    api_key_env: OPENAI_KEY
    timeout_s: 90
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml_text, encoding="utf-8")

    cfg = load_config(cfg_path)
    assert cfg.use_llm_classifier is False
    assert cfg.llm_classifier_min_chars == 500
    assert cfg.critic_pass_threshold == 3
    # Untouched key falls through to default
    assert cfg.critic_timeout_s == RouterConfig.critic_timeout_s
    # Cloud block
    assert cfg.cloud_escalation.enabled is True
    assert cfg.cloud_escalation.model == "openai/gpt-5"
    assert cfg.cloud_escalation.api_key_env == "OPENAI_KEY"
    assert cfg.cloud_escalation.timeout_s == 90


def test_load_config_handles_empty_yaml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("", encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg == RouterConfig()


def test_load_config_handles_missing_route_llm_block(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("model_list: []\n", encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg == RouterConfig()


def test_real_config_yaml_is_loadable() -> None:
    """The repo's own config.yaml must round-trip cleanly."""
    cfg = load_config(router_hook.CONFIG_PATH)
    assert isinstance(cfg, RouterConfig)
    # Sanity: cloud escalation off by default in committed config
    assert cfg.cloud_escalation.enabled is False
