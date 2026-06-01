"""Unit tests for init.py (smart-defaults config generator, #14).

Pure-heavy: synthetic /api/tags model dicts, monkeypatched _fetch_tags,
tmp_path for writes. No live Ollama.
"""
import init


def _model(name, param_size=None, size_bytes=None):
    """Build a synthetic /api/tags model dict."""
    m = {"name": name, "size": size_bytes or 0}
    if param_size is not None:
        m["details"] = {"parameter_size": param_size}
    return m


def test_module_constants_present():
    assert init._CLASSIFIER_MAX == 1.5
    assert init._S_MAX == 3.0
    assert init._M_MAX == 14.0
    assert init._L_MAX == 28.0
    assert init._DEFAULT_CLASSIFIER == "qwen2.5:0.5b"


def test_parse_params_billions():
    assert init._parse_params(_model("a", "7.6B")) == 7.6
    assert init._parse_params(_model("b", "1.5B")) == 1.5


def test_parse_params_millions():
    assert init._parse_params(_model("c", "500M")) == 0.5


def test_parse_params_byte_fallback_when_field_absent():
    # 7e9 bytes / 0.7e9 ~= 10.0 billion params
    got = init._parse_params(_model("d", param_size=None, size_bytes=7_000_000_000))
    assert abs(got - 10.0) < 0.01


def test_parse_params_zero_when_nothing_known():
    assert init._parse_params(_model("e", param_size=None, size_bytes=0)) == 0.0


def test_assign_tiers_five_family_acceptance():
    models = [
        _model("qwen2.5-coder:1.5b", "1.5B"),
        _model("deepseek-coder-v2:16b", "16B"),
        _model("llama3:8b", "8B"),
        _model("mistral:7b", "7B"),
        _model("gemma2:27b", "27B"),
        _model("qwen2.5:0.5b", "500M"),
    ]
    a = init.assign_tiers(models)
    assert a["classifier"] == "qwen2.5:0.5b"
    assert a["classifier_is_default"] is False
    assert a["tiers"]["S"] == "qwen2.5-coder:1.5b"
    assert a["tiers"]["M"] == "llama3:8b"
    assert a["tiers"]["L"] == "deepseek-coder-v2:16b"
    assert a["tiers"]["XL"] == "gemma2:27b"
    assert a["borrowed"] == {}
    assert a["model_count"] == 6


def test_assign_tiers_coder_preference_within_band():
    models = [
        _model("qwen2.5:0.5b", "500M"),
        _model("qwen2.5-coder:7b", "7B"),
        _model("llama3:13b", "13B"),
    ]
    a = init.assign_tiers(models)
    assert a["tiers"]["M"] == "qwen2.5-coder:7b"


def test_assign_tiers_empty_band_collapses_and_comments():
    models = [
        _model("qwen2.5:0.5b", "500M"),
        _model("qwen2.5-coder:1.5b", "1.5B"),
        _model("llama3:8b", "8B"),
        _model("qwen3-coder:20b", "20B"),
    ]
    a = init.assign_tiers(models)
    assert a["tiers"]["XL"] == "qwen3-coder:20b"
    assert "XL" in a["borrowed"]
    assert "qwen3-coder:20b" in a["borrowed"]["XL"]


def test_assign_tiers_no_tiny_model_flags_default():
    models = [
        _model("llama3:8b", "8B"),
        _model("gemma2:27b", "27B"),
    ]
    a = init.assign_tiers(models)
    assert a["classifier"] == init._DEFAULT_CLASSIFIER
    assert a["classifier_is_default"] is True


def test_assign_tiers_no_workers_raises():
    import pytest
    with pytest.raises(ValueError):
        init.assign_tiers([_model("qwen2.5:0.5b", "500M")])


def test_assign_tiers_skips_embedding_models():
    # Embedding models report tiny param sizes and would wrongly win the
    # "smallest <= 1.5B" classifier slot -- but they cannot chat/critique.
    # They must be dropped from both classifier and worker selection.
    models = [
        _model("nomic-embed-text:latest", "137M"),
        _model("qwen2.5:0.5b", "500M"),
        _model("qwen2.5-coder:1.5b", "1.5B"),
        _model("deepseek-coder-v2:16b", "16B"),
    ]
    a = init.assign_tiers(models)
    assert a["classifier"] == "qwen2.5:0.5b"        # NOT the embed model
    assert a["tiers"]["S"] == "qwen2.5-coder:1.5b"  # 0.5b is classifier-excluded
    # the embed model appears in no tier slot
    assert all("embed" not in (v or "") for v in a["tiers"].values())
    # model_count still reflects detected models (count is informational)
    assert a["model_count"] == 4


def test_assign_tiers_overflow_spreads_stacked_band():
    # Two models land in the L band (14 < p <= 28) with XL empty. Rather than
    # pick only the largest for L and drop the 16B, the overflow pass promotes
    # the larger (27B) up to the empty XL so distinct models spread across
    # tiers. No borrowing should occur.
    models = [
        _model("qwen2.5:0.5b", "500M"),
        _model("qwen2.5-coder:1.5b", "1.5B"),
        _model("llama3:8b", "8B"),
        _model("deepseek-coder-v2:16b", "16B"),
        _model("gemma2:27b", "27B"),
    ]
    a = init.assign_tiers(models)
    assert a["tiers"]["L"] == "deepseek-coder-v2:16b"
    assert a["tiers"]["XL"] == "gemma2:27b"
    assert a["borrowed"] == {}


def test_assign_tiers_overflow_noop_when_single_in_lower_band():
    # Real-config shape: empty M band, but the lower S band holds a single
    # model -> no promotion, M borrows. (Guards the overflow against firing
    # when there is nothing spare to spread.)
    models = [
        _model("qwen2.5:0.5b", "500M"),
        _model("qwen2.5-coder:1.5b", "1.5B"),   # S (only worker <= 3B)
        _model("deepseek-coder-v2:16b", "16B"),  # L
        _model("gemma2:27b", "27B"),             # L -> overflow to XL
    ]
    a = init.assign_tiers(models)
    assert a["tiers"]["S"] == "qwen2.5-coder:1.5b"
    assert "M" in a["borrowed"]      # M empty, S has one model -> borrow, not promote


import yaml


def _sample_assignment():
    return init.assign_tiers([
        _model("qwen2.5-coder:1.5b", "1.5B"),
        _model("deepseek-coder-v2:16b", "16B"),
        _model("llama3:8b", "8B"),
        _model("gemma2:27b", "27B"),
        _model("qwen2.5:0.5b", "500M"),
    ])


def test_build_config_yaml_safe_load_clean():
    text = init.build_config(_sample_assignment())
    doc = yaml.safe_load(text)
    assert isinstance(doc, dict)
    assert "model_list" in doc
    assert "router_settings" in doc
    assert "litellm_settings" in doc
    assert "general_settings" in doc
    assert "route_llm" in doc


def test_build_config_places_models_in_tier_slots():
    a = _sample_assignment()
    doc = yaml.safe_load(init.build_config(a))
    by_alias = {m["model_name"]: m for m in doc["model_list"]}
    assert "local-s" in by_alias
    assert "local-m" in by_alias
    assert "local-l" in by_alias
    assert "local-xl" in by_alias
    assert "local-auto" in by_alias
    assert a["tiers"]["S"] in by_alias["local-s"]["litellm_params"]["model"]
    assert a["tiers"]["M"] in by_alias["local-auto"]["litellm_params"]["model"]


def test_build_config_standard_settings():
    doc = yaml.safe_load(init.build_config(_sample_assignment()))
    assert doc["general_settings"]["master_key"] == "sk-local-dev"
    assert doc["litellm_settings"]["callbacks"] == "router_hook.tier_router_instance"
    assert doc["router_settings"]["fallbacks"] == [
        {"local-s": ["local-m"]},
        {"local-m": ["local-l"]},
        {"local-l": ["local-xl"]},
    ]


def test_build_config_classifier_and_critic_are_tiny_model():
    a = _sample_assignment()
    doc = yaml.safe_load(init.build_config(a))
    assert doc["route_llm"]["classifier_model"] == a["classifier"]
    assert doc["route_llm"]["critic_model"] == a["classifier"]


def test_build_config_header_has_count_not_date():
    a = _sample_assignment()
    text = init.build_config(a)
    assert "Generated by init.py" in text
    assert str(a["model_count"]) in text.split("\n", 1)[0]


def test_build_config_borrowed_tier_carries_comment():
    a = init.assign_tiers([
        _model("qwen2.5:0.5b", "500M"),
        _model("qwen2.5-coder:1.5b", "1.5B"),
        _model("llama3:8b", "8B"),
        _model("qwen3-coder:20b", "20B"),
    ])
    text = init.build_config(a)
    assert "no model in this size band was installed" in text


def test_write_config_default_writes_generated_file(tmp_path):
    out = tmp_path / "config.generated.yaml"
    path = init.write_config(str(out), "hello: world\n", do_write=False, force=False)
    assert path == str(out)
    assert out.read_text(encoding="ascii") == "hello: world\n"


def test_write_config_write_mode_backs_up_existing(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("old: config\n", encoding="ascii")
    path = init.write_config(str(cfg), "new: config\n", do_write=True, force=False)
    assert path == str(cfg)
    assert cfg.read_text(encoding="ascii") == "new: config\n"
    bak = tmp_path / "config.yaml.bak"
    assert bak.read_text(encoding="ascii") == "old: config\n"


def test_write_config_write_mode_no_existing_no_bak(tmp_path):
    cfg = tmp_path / "config.yaml"
    init.write_config(str(cfg), "new: config\n", do_write=True, force=False)
    assert cfg.read_text(encoding="ascii") == "new: config\n"
    assert not (tmp_path / "config.yaml.bak").exists()


def test_run_default_writes_generated_and_exits_zero(tmp_path, monkeypatch, capsys):
    out = tmp_path / "config.generated.yaml"
    fake = [
        _model("qwen2.5:0.5b", "500M"),
        _model("qwen2.5-coder:1.5b", "1.5B"),
        _model("llama3:8b", "8B"),
        _model("deepseek-coder-v2:16b", "16B"),
        _model("gemma2:27b", "27B"),
    ]

    async def _fake_fetch():
        return fake
    monkeypatch.setattr(init, "_fetch_tags", _fake_fetch)

    rc = init.run(init.build_parser().parse_args(["--output", str(out)]))
    assert rc == 0
    assert out.exists()
    import yaml
    assert yaml.safe_load(out.read_text(encoding="ascii"))
    captured = capsys.readouterr().out
    assert "--write" in captured


def test_run_write_mode_writes_target(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"

    async def _fake_fetch():
        return [
            _model("qwen2.5:0.5b", "500M"),
            _model("qwen2.5-coder:1.5b", "1.5B"),
            _model("llama3:8b", "8B"),
            _model("gemma2:27b", "27B"),
        ]
    monkeypatch.setattr(init, "_fetch_tags", _fake_fetch)

    rc = init.run(init.build_parser().parse_args(["--write", "--output", str(cfg)]))
    assert rc == 0
    assert cfg.exists()


def test_run_no_models_exits_nonzero(monkeypatch, capsys):
    async def _empty():
        return []
    monkeypatch.setattr(init, "_fetch_tags", _empty)
    rc = init.run(init.build_parser().parse_args([]))
    assert rc != 0
    assert "pull" in capsys.readouterr().out.lower()


def test_run_unreachable_ollama_exits_nonzero(monkeypatch, capsys):
    async def _boom():
        raise RuntimeError("connection refused")
    monkeypatch.setattr(init, "_fetch_tags", _boom)
    rc = init.run(init.build_parser().parse_args([]))
    assert rc != 0
    assert "ollama" in capsys.readouterr().out.lower()
