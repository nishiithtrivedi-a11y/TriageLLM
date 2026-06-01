"""Issue #20b: runtime model-not-found + cold-load hints. Mock-based, no real I/O."""
from types import SimpleNamespace

import router_hook


def test_is_model_not_found_status_404():
    assert router_hook._is_model_not_found(SimpleNamespace(status_code=404)) is True


def test_is_model_not_found_message():
    exc = Exception("model 'qwen2.5-coder:1.5b' not found, try pulling it first")
    assert router_hook._is_model_not_found(exc) is True


def test_is_model_not_found_false_on_unrelated():
    assert router_hook._is_model_not_found(Exception("connection timeout")) is False
    assert router_hook._is_model_not_found(SimpleNamespace(status_code=500)) is False


def test_model_name_from_error_extracts_real_tag():
    exc = Exception("model 'qwen2.5-coder:1.5b' not found, try pulling it first")
    assert router_hook._model_name_from_error(exc, "local-s") == "qwen2.5-coder:1.5b"


def test_model_name_from_error_falls_back():
    assert router_hook._model_name_from_error(Exception("boom"), "local-s") == "local-s"


def test_model_not_found_hint_strips_prefix_and_is_ascii():
    hint = router_hook._model_not_found_hint("ollama_chat/qwen2.5-coder:1.5b")
    assert "ollama pull qwen2.5-coder:1.5b" in hint
    hint.encode("ascii")  # raises if non-ASCII slipped in


def test_maybe_warn_cold_load_logs_when_slow(capsys):
    router_hook._maybe_warn_cold_load("ollama_chat/qwen3-coder:30b", 35.0)
    out = capsys.readouterr().out
    assert "qwen3-coder:30b" in out and "cold load" in out


def test_maybe_warn_cold_load_silent_when_fast(capsys):
    router_hook._maybe_warn_cold_load("ollama_chat/qwen3-coder:30b", 2.0)
    assert capsys.readouterr().out == ""


from unittest.mock import patch, AsyncMock  # noqa: E402

import pytest  # noqa: E402

from router_hook import TierRouter  # noqa: E402


@pytest.fixture
def router() -> TierRouter:
    with patch.object(TierRouter, "_warmup", AsyncMock(return_value=None)):
        r = TierRouter()
    r.config.cloud_escalation.enabled = False
    return r


@pytest.mark.asyncio
async def test_failure_hook_logs_hint_with_real_model(router, capsys):
    exc = Exception("model 'qwen2.5-coder:1.5b' not found, try pulling it first")
    await router.async_post_call_failure_hook({"model": "local-s"}, exc, None)
    out = capsys.readouterr().out
    assert "ollama pull qwen2.5-coder:1.5b" in out   # real tag, NOT the alias


@pytest.mark.asyncio
async def test_failure_hook_silent_on_unrelated_error(router, capsys):
    await router.async_post_call_failure_hook({"model": "local-s"},
                                              Exception("connection reset"), None)
    assert capsys.readouterr().out == ""


@pytest.mark.asyncio
async def test_failure_hook_never_raises(router):
    # Even if detection blows up, the hook must swallow it (must not mask the
    # original error). Pass an object whose str() raises.
    class Boom:
        def __str__(self):
            raise RuntimeError("nope")
    await router.async_post_call_failure_hook({"model": "local-s"}, Boom(), None)


@pytest.mark.asyncio
async def test_call_tier_404_logs_hint_and_reraises(router, capsys):
    import litellm
    exc = Exception("model 'deepseek-coder-v2:16b' not found, try pulling it first")
    with patch.object(router_hook._ollama_circuit, "preflight",
                      AsyncMock(return_value=(True, "ok"))), \
         patch.object(litellm, "acompletion", AsyncMock(side_effect=exc)):
        raised = False
        try:
            await router._call_tier([{"role": "user", "content": "hi"}], "M")
        except Exception:
            raised = True
    assert raised   # the original error still propagates (C-5 ledger path intact)
    assert "ollama pull deepseek-coder-v2:16b" in capsys.readouterr().out
