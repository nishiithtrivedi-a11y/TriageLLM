"""Issue #29: per-attempt cost tracking (was_warm / vram_mb / cost_usd)."""
import importlib

import pytest

import router_hook


def test_attempt_has_defaulted_cost_fields():
    """The 3 new fields default so existing call sites + old rows keep working."""
    a = router_hook.Attempt(
        tier="S", model="x", prompt_tokens=1, completion_tokens=1,
        duration_s=0.1, critic_score=None, preview="",
    )
    assert a.was_warm is False
    assert a.vram_mb is None
    assert a.cost_usd is None


def test_was_warm_tracker_flips_cold_to_warm():
    """First time a model is seen this process -> cold; subsequent -> warm."""
    # Reset the module-level tracker for a clean assertion.
    router_hook._warm_models.clear()
    m = "ollama_chat/test-model:1b"
    assert router_hook._mark_and_check_warm(m) is False   # first call: cold
    assert router_hook._mark_and_check_warm(m) is True    # second: warm
    assert router_hook._mark_and_check_warm(m) is True    # still warm
    # A different model is independently cold on first sight.
    assert router_hook._mark_and_check_warm("ollama_chat/other:1b") is False


def test_cost_usd_from_litellm_best_effort(monkeypatch):
    """cost_usd is populated from litellm.completion_cost; None on failure."""
    import litellm

    class _Resp:  # minimal stand-in
        pass

    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.0123)
    assert router_hook._safe_cost_usd(_Resp()) == 0.0123

    def _boom(completion_response):
        raise ValueError("no pricing for local model")
    monkeypatch.setattr(litellm, "completion_cost", _boom)
    assert router_hook._safe_cost_usd(_Resp()) is None


@pytest.mark.asyncio
async def test_fetch_vram_mb_parses_api_ps(monkeypatch):
    """_fetch_vram_mb maps a model's size_vram (bytes) to MB from /api/ps."""
    import httpx

    sample = {"models": [
        {"name": "qwen2.5-coder:1.5b", "size_vram": 1610612736},  # 1536 MB
        {"name": "deepseek-coder-v2:16b", "size_vram": 9000000000},
    ]}

    class _Resp:
        status_code = 200
        def json(self):
            return sample

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url): return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    # ollama_chat/ prefix should be stripped to match Ollama's bare name.
    mb = await router_hook._fetch_vram_mb("ollama_chat/qwen2.5-coder:1.5b")
    assert mb == 1536


@pytest.mark.asyncio
async def test_fetch_vram_mb_returns_none_on_failure(monkeypatch):
    """Any /api/ps failure -> None, never raises."""
    import httpx

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url): raise httpx.ConnectError("ollama down")

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    assert await router_hook._fetch_vram_mb("ollama_chat/anything:1b") is None


@pytest.mark.asyncio
async def test_vram_mb_nonblocking_caches_after_background_fetch(monkeypatch):
    """First call returns None (cache miss) but schedules a background fetch;
    after the task runs, subsequent calls return the cached value. The call
    itself never awaits the fetch (non-blocking)."""
    import asyncio as _asyncio
    # Reset module state
    router_hook._vram_cache.clear()
    router_hook._vram_inflight.clear()

    async def _fake_fetch(model):
        return 2048
    monkeypatch.setattr(router_hook, "_fetch_vram_mb", _fake_fetch)

    m = "ollama_chat/test:1b"
    # First call: cache miss -> None, but a background task is scheduled.
    assert router_hook._vram_mb_nonblocking(m) is None
    # Let the event loop run the scheduled task.
    await _asyncio.sleep(0)
    # Now cached.
    assert router_hook._vram_mb_nonblocking(m) == 2048


def test_vram_mb_nonblocking_no_loop_returns_none(monkeypatch):
    """With no running event loop, the wrapper returns None and doesn't crash."""
    router_hook._vram_cache.clear()
    router_hook._vram_inflight.clear()
    # Called from a sync context (no running loop) -> None, no exception.
    assert router_hook._vram_mb_nonblocking("ollama_chat/x:1b") is None


@pytest.mark.asyncio
async def test_first_attempt_carries_cost_fields_and_was_warm_flips(monkeypatch, tmp_path):
    """Regression for the smoke-caught wiring bug: the first_attempt (the only
    attempt logged for a non-escalated tier-S request) must carry was_warm/
    cost_usd, and was_warm must flip False->True across two same-model requests.
    The mocked helper tests passed while this was broken because the populated
    Attempt was not the logged one."""
    import time as _time

    # Isolate DB + reset trackers
    monkeypatch.setattr(router_hook, "DB_PATH", tmp_path / "fa.db")
    router_hook._warm_models.clear()
    router_hook._vram_cache.clear()
    router_hook._vram_inflight.clear()
    router_hook._init_db()

    # Capture what _log_decision receives.
    logged = []
    monkeypatch.setattr(router_hook, "_log_decision",
                        lambda state, attempts, *a, **k: logged.append(attempts))

    inst = router_hook.tier_router_instance
    # Tier S short-circuits orchestration (no critique, no escalation), so the
    # logged ledger is exactly [first_attempt].
    state = {
        "initial_tier": "S", "alias": "local-s", "requested": "local-auto",
        "tokens": 5, "score": 0, "signals": [], "classifier": "rules",
        "started_at": _time.time(),
    }

    class _Resp:
        # minimal LiteLLM-ish response _extract_usage/_extract_answer tolerate
        class _U:
            prompt_tokens = 5
            completion_tokens = 3
        usage = _U()
        choices = [type("C", (), {"message": type("M", (), {"content": "hi"})()})()]

    # _safe_cost_usd will try litellm.completion_cost on _Resp and fail -> None
    # (fine; this test asserts was_warm, not cost). Fire twice.
    data = {"_router_state": state, "messages": [{"role": "user", "content": "hi"}]}
    await inst.async_post_call_success_hook(data, None, _Resp())
    await inst.async_post_call_success_hook(dict(data, _router_state=dict(state)), None, _Resp())

    assert len(logged) == 2
    first_call_attempt = logged[0][0]
    second_call_attempt = logged[1][0]
    assert first_call_attempt.was_warm is False   # cold on first sight
    assert second_call_attempt.was_warm is True    # warm on second
