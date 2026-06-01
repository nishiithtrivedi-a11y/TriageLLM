"""Tests for capability advisory surfacing (#18 v0.3 live piece)."""
import os
import router_hook


def test_mode_default_is_shadow_and_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("TRIAGELLM_CAPABILITY_MODE", raising=False)
    monkeypatch.delenv("TRIAGELLM_CAPABILITY_ROUTING_ENABLED", raising=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("route_llm:\n  critic_pass_threshold: 4\n", encoding="ascii")
    c = router_hook.load_config(cfg)
    assert c.capability_routing.mode == "shadow"
    assert c.capability_routing.enabled is False


def test_env_mode_beats_config_and_forces_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("TRIAGELLM_CAPABILITY_MODE", "advisory")
    monkeypatch.delenv("TRIAGELLM_CAPABILITY_ROUTING_ENABLED", raising=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "route_llm:\n  capability_routing:\n    enabled: false\n    mode: shadow\n",
        encoding="ascii")
    c = router_hook.load_config(cfg)
    assert c.capability_routing.mode == "advisory"
    assert c.capability_routing.enabled is True


def test_config_advisory_forces_enabled(monkeypatch, tmp_path):
    monkeypatch.delenv("TRIAGELLM_CAPABILITY_MODE", raising=False)
    monkeypatch.delenv("TRIAGELLM_CAPABILITY_ROUTING_ENABLED", raising=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "route_llm:\n  capability_routing:\n    mode: advisory\n",
        encoding="ascii")
    c = router_hook.load_config(cfg)
    assert c.capability_routing.enabled is True


def test_unknown_mode_falls_back_to_shadow_with_warning(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("TRIAGELLM_CAPABILITY_MODE", "bananas")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("route_llm: {}\n", encoding="ascii")
    c = router_hook.load_config(cfg)
    assert c.capability_routing.mode == "shadow"
    out = capsys.readouterr().out
    assert "capability" in out.lower() and "bananas" in out


def _sample_cap(agree=1, conf=0.72, cat="structured_output", suggested="M"):
    return {
        "cap_category": cat,
        "cap_recommended_tier": suggested,
        "cap_reason_code": "rule",
        "cap_signals": "json",
        "cap_confidence": conf,
        "cap_classifier_used": False,
        "cap_pack": "coder",
        "cap_agrees_with_tier": agree,
    }


def test_format_advisory_line_disagree():
    line = router_hook._format_advisory_line(_sample_cap(agree=0), actual_tier="L")
    assert line == "[advisory] cap=structured_output suggested=M actual=L agree=no conf=0.72"


def test_format_advisory_line_agree_and_conf_rounding():
    line = router_hook._format_advisory_line(_sample_cap(agree=1, conf=0.7), actual_tier="M")
    assert line == "[advisory] cap=structured_output suggested=M actual=M agree=yes conf=0.70"


def test_format_advisory_line_is_ascii():
    line = router_hook._format_advisory_line(_sample_cap(), actual_tier="L")
    line.encode("ascii")


def test_advisory_headers_disagree():
    h = router_hook._advisory_headers(_sample_cap(agree=0, cat="quick_question", suggested="S"), actual_tier="M")
    assert h == {
        "x-triagellm-cap-category": "quick_question",
        "x-triagellm-cap-suggested-tier": "S",
        "x-triagellm-cap-actual-tier": "M",
        "x-triagellm-cap-agrees": "false",
    }


def test_advisory_headers_agree_true():
    h = router_hook._advisory_headers(_sample_cap(agree=1), actual_tier="M")
    assert h["x-triagellm-cap-agrees"] == "true"


def test_advisory_headers_values_are_ascii_strings():
    h = router_hook._advisory_headers(_sample_cap(), actual_tier="L")
    for k, v in h.items():
        assert isinstance(v, str)
        k.encode("ascii"); v.encode("ascii")


import pytest
from unittest.mock import AsyncMock, patch


def _advisory_router(mode="advisory", enabled=True):
    """A TierRouter with capability enabled and the given mode."""
    import router_hook
    from router_hook import CapabilityRoutingConfig, CapabilityPacksConfig
    with patch.object(router_hook.TierRouter, "_warmup", AsyncMock(return_value=None)):
        r = router_hook.TierRouter()
    r.config.capability_routing = CapabilityRoutingConfig(
        enabled=enabled, mode=mode, use_llm_tiebreaker=False,
        confidence_threshold=0.6,
        packs=CapabilityPacksConfig(coder=True, writing=False, analyst=False))
    return r


@pytest.mark.asyncio
async def test_advisory_mode_emits_log_line(capsys):
    import router_hook
    r = _advisory_router(mode="advisory")
    allow = AsyncMock(return_value=(True, "ok"))
    with patch.object(router_hook._ollama_circuit, "preflight", allow):
        data = {"model": "local-auto",
                "messages": [{"role": "user", "content": "return JSON with field name"}]}
        await r.async_pre_call_hook(None, None, data, "completion")
    out = capsys.readouterr().out
    assert "[advisory] cap=structured_output" in out
    assert "suggested=" in out and "actual=" in out


@pytest.mark.asyncio
async def test_shadow_mode_does_not_emit_log_line(capsys):
    import router_hook
    r = _advisory_router(mode="shadow")
    allow = AsyncMock(return_value=(True, "ok"))
    with patch.object(router_hook._ollama_circuit, "preflight", allow):
        data = {"model": "local-auto",
                "messages": [{"role": "user", "content": "return JSON with field name"}]}
        await r.async_pre_call_hook(None, None, data, "completion")
    assert "[advisory]" not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_disabled_does_not_emit_log_line(capsys):
    import router_hook
    r = _advisory_router(mode="shadow", enabled=False)
    allow = AsyncMock(return_value=(True, "ok"))
    with patch.object(router_hook._ollama_circuit, "preflight", allow):
        data = {"model": "local-auto",
                "messages": [{"role": "user", "content": "return JSON with field name"}]}
        await r.async_pre_call_hook(None, None, data, "completion")
    assert "[advisory]" not in capsys.readouterr().out


class _FakeResp:
    def __init__(self):
        self._hidden_params = {}


def test_attach_advisory_headers_advisory_merges():
    r = _advisory_router(mode="advisory")
    resp = _FakeResp()
    state = {"initial_tier": "L", "capability": _sample_cap(agree=0, suggested="M")}
    r._attach_advisory_headers(resp, state)
    hdrs = resp._hidden_params["additional_headers"]
    assert hdrs["x-triagellm-cap-category"] == "structured_output"
    assert hdrs["x-triagellm-cap-suggested-tier"] == "M"
    assert hdrs["x-triagellm-cap-actual-tier"] == "L"
    assert hdrs["x-triagellm-cap-agrees"] == "false"


def test_attach_advisory_headers_preserves_existing():
    r = _advisory_router(mode="advisory")
    resp = _FakeResp()
    resp._hidden_params["additional_headers"] = {"x-existing": "1"}
    state = {"initial_tier": "M", "capability": _sample_cap()}
    r._attach_advisory_headers(resp, state)
    hdrs = resp._hidden_params["additional_headers"]
    assert hdrs["x-existing"] == "1"
    assert "x-triagellm-cap-category" in hdrs


def test_attach_advisory_headers_shadow_is_noop():
    r = _advisory_router(mode="shadow", enabled=True)
    resp = _FakeResp()
    state = {"initial_tier": "M", "capability": _sample_cap()}
    r._attach_advisory_headers(resp, state)
    assert "additional_headers" not in resp._hidden_params


def test_attach_advisory_headers_no_capability_is_noop():
    r = _advisory_router(mode="advisory")
    resp = _FakeResp()
    r._attach_advisory_headers(resp, {"initial_tier": "M"})
    assert "additional_headers" not in resp._hidden_params


def test_attach_advisory_headers_swallows_failure():
    r = _advisory_router(mode="advisory")
    state = {"initial_tier": "M", "capability": _sample_cap()}

    class _Boom:
        @property
        def _hidden_params(self):
            raise RuntimeError("no hidden params here")
    r._attach_advisory_headers(_Boom(), state)


@pytest.mark.asyncio
async def test_advisory_does_not_change_routing():
    """Enabling advisory must not change data['model'] or the routed tier
    vs capability disabled, for the same prompt."""
    import router_hook
    prompt = "return JSON with field name"
    allow = AsyncMock(return_value=(True, "ok"))

    r_off = _advisory_router(mode="shadow", enabled=False)
    with patch.object(router_hook._ollama_circuit, "preflight", allow):
        d_off = {"model": "local-auto", "messages": [{"role": "user", "content": prompt}]}
        await r_off.async_pre_call_hook(None, None, d_off, "completion")

    r_adv = _advisory_router(mode="advisory", enabled=True)
    with patch.object(router_hook._ollama_circuit, "preflight", allow):
        d_adv = {"model": "local-auto", "messages": [{"role": "user", "content": prompt}]}
        await r_adv.async_pre_call_hook(None, None, d_adv, "completion")

    assert d_off["model"] == d_adv["model"]
    assert d_off["_router_state"]["initial_tier"] == d_adv["_router_state"]["initial_tier"]
