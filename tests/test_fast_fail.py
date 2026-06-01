"""DEF-004 fast-fail / circuit-breaker unit tests (no real Ollama; mocked).

Covers: CLOSED start, open-on-failure, OPEN fast-fails WITHOUT probing,
cooldown -> HALF_OPEN probe, probe-success closes, probe-failure reopens,
explicit + cp1252-safe reason strings, disabled passthrough, and the pre-call
hook raising fast (+ writing an observable ledger row) when the circuit denies.
"""
import sqlite3
from unittest.mock import AsyncMock, patch

import pytest

import router_hook
from router_hook import OllamaCircuitBreaker, TierRouter


def _cb(**kw):
    kw.setdefault("base_url", "http://localhost:11434")
    kw.setdefault("cooldown", 10.0)
    return OllamaCircuitBreaker(**kw)


def _router():
    with patch.object(TierRouter, "_warmup", AsyncMock(return_value=None)):
        return TierRouter()


@pytest.mark.asyncio
async def test_circuit_starts_closed():
    assert _cb().state == "CLOSED"


@pytest.mark.asyncio
async def test_connection_failure_opens_circuit():
    cb = _cb()
    with patch.object(cb, "_probe", AsyncMock(side_effect=ConnectionError("refused"))):
        allowed, reason = await cb.preflight()
    assert allowed is False
    assert cb.state == "OPEN"
    assert "unreachable" in reason.lower()
    assert cb.open_count == 1


@pytest.mark.asyncio
async def test_open_circuit_fast_fails_without_probing():
    """An OPEN circuit within cooldown must NOT wait on a probe at all."""
    cb = _cb(cooldown=100.0)
    with patch.object(cb, "_probe", AsyncMock(side_effect=ConnectionError("x"))):
        await cb.preflight()
    assert cb.state == "OPEN"
    probe = AsyncMock(return_value=True)
    with patch.object(cb, "_probe", probe):
        allowed, reason = await cb.preflight()
    assert allowed is False
    probe.assert_not_called()          # <-- the fast-fail: no network wait
    assert "cooldown" in reason.lower()


@pytest.mark.asyncio
async def test_cooldown_elapsed_moves_to_halfopen_and_probes():
    cb = _cb(cooldown=5.0)
    with patch.object(cb, "_probe", AsyncMock(side_effect=ConnectionError("x"))):
        await cb.preflight()
    probe = AsyncMock(return_value=True)
    with patch.object(cb, "_probe", probe):
        allowed, reason = await cb.preflight(now=cb.opened_at + 6.0)
    probe.assert_called_once()
    assert allowed is True
    assert cb.state == "CLOSED"
    assert "recovered" in reason.lower()


@pytest.mark.asyncio
async def test_successful_probe_closes_circuit():
    cb = _cb()
    with patch.object(cb, "_probe", AsyncMock(return_value=True)):
        allowed, _ = await cb.preflight()
    assert allowed is True
    assert cb.state == "CLOSED"


@pytest.mark.asyncio
async def test_failed_probe_reopens_from_halfopen():
    cb = _cb(cooldown=5.0)
    with patch.object(cb, "_probe", AsyncMock(side_effect=ConnectionError("x"))):
        await cb.preflight()
    with patch.object(cb, "_probe", AsyncMock(side_effect=ConnectionError("y"))):
        allowed, _ = await cb.preflight(now=cb.opened_at + 6.0)
    assert allowed is False
    assert cb.state == "OPEN"
    assert cb.open_count == 2


@pytest.mark.asyncio
async def test_failure_reason_is_explicit_and_cp1252_safe():
    cb = _cb()
    with patch.object(cb, "_probe", AsyncMock(side_effect=TimeoutError("t"))):
        allowed, reason = await cb.preflight()
    assert allowed is False
    assert "Ollama unreachable" in reason
    reason.encode("cp1252")   # must not raise (DEF-003 must not return)


@pytest.mark.asyncio
async def test_disabled_breaker_always_allows_without_probing():
    cb = _cb(enabled=False)
    probe = AsyncMock(side_effect=ConnectionError("x"))
    with patch.object(cb, "_probe", probe):
        allowed, _ = await cb.preflight()
    assert allowed is True
    probe.assert_not_called()


@pytest.mark.asyncio
async def test_pre_call_hook_fast_fails_and_logs_ledger_row(tmp_path, monkeypatch):
    """When the circuit denies, async_pre_call_hook must raise (fast) AND write
    an observable ledger row (classifier='ollama-down-fastfail', handoff=1)."""
    monkeypatch.setattr(router_hook, "DB_PATH", tmp_path / "rt.sqlite")
    router_hook._init_db()
    r = _router()
    deny = AsyncMock(return_value=(False, "circuit OPEN: Ollama unreachable (ConnectError); fast-fail"))
    monkeypatch.setattr(router_hook._ollama_circuit, "preflight", deny)

    data = {"model": "local-auto", "messages": [{"role": "user", "content": "hi"}]}
    with pytest.raises(Exception):
        await r.async_pre_call_hook(None, None, data, "completion")

    with sqlite3.connect(tmp_path / "rt.sqlite") as c:
        rows = c.execute("SELECT classifier, handoff FROM decisions").fetchall()
    assert rows, "fast-fail must leave an observable ledger row"
    assert rows[-1][0] == "ollama-down-fastfail"
    assert rows[-1][1] == 1


@pytest.mark.asyncio
async def test_pre_call_hook_passes_through_when_circuit_allows(monkeypatch):
    """Happy path: when the circuit allows, the hook proceeds and rewrites the
    model to a tier alias (no raise)."""
    r = _router()
    allow = AsyncMock(return_value=(True, "ok"))
    monkeypatch.setattr(router_hook._ollama_circuit, "preflight", allow)
    # keep it rules-only (short prompt) so no real classifier call is made
    data = {"model": "local-auto", "messages": [{"role": "user", "content": "rename foo to bar"}]}
    out = await r.async_pre_call_hook(None, None, data, "completion")
    assert out["model"].startswith("local-")
    assert "_router_state" in out


@pytest.mark.asyncio
async def test_external_model_skips_preflight(monkeypatch):
    """Genuinely-external models (gpt-4, claude-3, ...) still skip the breaker
    entirely -- they are not TriageLLM's concern. (#3: only local-* now triggers it.)"""
    r = _router()
    probe = AsyncMock(side_effect=AssertionError("preflight must not run for external models"))
    monkeypatch.setattr(router_hook._ollama_circuit, "preflight", probe)
    data = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
    out = await r.async_pre_call_hook(None, None, data, "completion")
    assert out["model"] == "gpt-4"
    assert "_router_state" not in out


@pytest.mark.asyncio
async def test_explicit_pin_fast_fails_when_circuit_open(tmp_path, monkeypatch):
    """#3: an explicit local-* pin now ALSO fast-fails (raises + writes a ledger
    row) when the circuit denies -- it no longer bypasses the breaker."""
    monkeypatch.setattr(router_hook, "DB_PATH", tmp_path / "rt.sqlite")
    router_hook._init_db()
    r = _router()
    deny = AsyncMock(return_value=(False, "circuit OPEN: Ollama unreachable (ConnectError); fast-fail"))
    monkeypatch.setattr(router_hook._ollama_circuit, "preflight", deny)

    data = {"model": "local-m", "messages": [{"role": "user", "content": "hi"}]}
    with pytest.raises(Exception):
        await r.async_pre_call_hook(None, None, data, "completion")

    with sqlite3.connect(tmp_path / "rt.sqlite") as c:
        rows = c.execute("SELECT classifier, handoff FROM decisions").fetchall()
    assert rows, "explicit-pin fast-fail must leave an observable ledger row"
    assert rows[-1][0] == "ollama-down-fastfail"
    assert rows[-1][1] == 1


@pytest.mark.asyncio
async def test_explicit_pin_passes_through_when_circuit_allows(monkeypatch):
    """#3: an explicit local-* pin runs the breaker now, but when it ALLOWS,
    passes through UNCHANGED -- no classification, no tier rewrite, no routing
    state (the escape-hatch is preserved)."""
    r = _router()
    allow = AsyncMock(return_value=(True, "ok"))
    monkeypatch.setattr(router_hook._ollama_circuit, "preflight", allow)
    data = {"model": "local-m", "messages": [{"role": "user", "content": "hi"}]}
    out = await r.async_pre_call_hook(None, None, data, "completion")
    assert out["model"] == "local-m"       # unchanged -- no tier rewrite
    assert "_router_state" not in out      # escape hatch -- no routing
