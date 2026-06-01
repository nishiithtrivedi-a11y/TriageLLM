"""Multi-step escalation + cloud step + handoff orchestration (litellm mocked)."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from router_hook import Attempt, TierRouter


def _resp(text: str, prompt_t: int = 50, completion_t: int = 100):
    """Build a fake ModelResponse-shaped object."""
    msg = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=msg, delta=SimpleNamespace(content=text), finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=prompt_t, completion_tokens=completion_t)
    return SimpleNamespace(choices=[choice], usage=usage)


@pytest.fixture
def router() -> TierRouter:
    r = TierRouter()
    r.config.cloud_escalation.enabled = False
    return r


@pytest.mark.asyncio
async def test_orchestrate_stops_when_critic_passes(router: TierRouter) -> None:
    """Critic on the first attempt scores 5/5 → no escalation."""
    data = {"messages": [{"role": "user", "content": "refactor this"}]}
    first_resp = _resp("a perfect answer")
    first_attempt = Attempt(
        tier="L", model="ollama_chat/qwen3-coder:30b",
        prompt_tokens=50, completion_tokens=100, duration_s=1.0,
        critic_score=None, preview="",
    )
    with patch.object(router, "_critique", AsyncMock(return_value=5)):
        final, attempts, cloud, handoff = await router._orchestrate(data, first_resp, first_attempt)
    assert final is first_resp
    assert len(attempts) == 1
    assert attempts[0].critic_score == 5
    assert cloud is False
    assert handoff is False


@pytest.mark.asyncio
async def test_orchestrate_escalates_until_xl(router: TierRouter) -> None:
    """All tiers fail → ledger has all attempts L,XL and handoff fires."""
    data = {"messages": [{"role": "user", "content": "refactor this"}]}
    first_resp = _resp("L's weak answer")
    first_attempt = Attempt(
        tier="L", model="ollama_chat/qwen3-coder:30b",
        prompt_tokens=50, completion_tokens=100, duration_s=1.0,
        critic_score=None, preview="",
    )
    xl_resp = _resp("XL's weak answer too")
    xl_attempt = Attempt(
        tier="XL", model="ollama_chat/qwen3.6:35b",
        prompt_tokens=50, completion_tokens=100, duration_s=2.0,
        critic_score=None, preview="",
    )

    with patch.object(router, "_critique", AsyncMock(return_value=2)), \
         patch.object(router, "_call_tier", AsyncMock(return_value=(xl_resp, xl_attempt))):
        final, attempts, cloud, handoff = await router._orchestrate(data, first_resp, first_attempt)

    tiers = [a.tier for a in attempts]
    assert tiers == ["L", "XL"]
    assert handoff is True
    assert cloud is False
    # The final response's content was rewritten to the handoff message.
    assert "LOCAL STACK EXHAUSTED" in final.choices[0].message.content


@pytest.mark.asyncio
async def test_orchestrate_skips_critic_for_tier_S(router: TierRouter) -> None:
    data = {"messages": [{"role": "user", "content": "rename foo"}]}
    first_resp = _resp("renamed")
    first_attempt = Attempt(
        tier="S", model="ollama_chat/qwen2.5-coder:1.5b",
        prompt_tokens=10, completion_tokens=5, duration_s=0.5,
        critic_score=None, preview="",
    )
    with patch.object(router, "_critique", AsyncMock(return_value=1)) as critique:
        final, attempts, cloud, handoff = await router._orchestrate(data, first_resp, first_attempt)
    assert critique.call_count == 0  # never invoked
    assert len(attempts) == 1
    assert attempts[0].critic_score is None
    assert handoff is False


@pytest.mark.asyncio
async def test_orchestrate_uses_cloud_when_enabled_and_succeeds(router: TierRouter, monkeypatch) -> None:
    router.config.cloud_escalation.enabled = True
    router.config.cloud_escalation.api_key_env = "TEST_KEY"
    monkeypatch.setenv("TEST_KEY", "sk-test")

    data = {"messages": [{"role": "user", "content": "design a system"}]}
    xl_resp = _resp("XL weak")
    xl_attempt = Attempt(
        tier="XL", model="ollama_chat/qwen3.6:35b",
        prompt_tokens=50, completion_tokens=100, duration_s=2.0,
        critic_score=None, preview="",
    )
    cloud_resp = _resp("brilliant cloud answer")
    cloud_attempt = Attempt(
        tier="CLOUD", model="anthropic/claude-sonnet-4-6",
        prompt_tokens=50, completion_tokens=200, duration_s=4.0,
        critic_score=None, preview="",
    )

    critique_scores = iter([2, 5])  # XL fails, cloud passes
    async def fake_critique(*_args, **_kw):
        return next(critique_scores)

    with patch.object(router, "_critique", side_effect=fake_critique), \
         patch.object(router, "_call_cloud", AsyncMock(return_value=(cloud_resp, cloud_attempt))):
        final, attempts, cloud, handoff = await router._orchestrate(data, xl_resp, xl_attempt)

    assert [a.tier for a in attempts] == ["XL", "CLOUD"]
    assert attempts[-1].critic_score == 5
    assert cloud is True
    assert handoff is False
    assert final is cloud_resp


@pytest.mark.asyncio
async def test_orchestrate_handoff_when_cloud_disabled(router: TierRouter) -> None:
    router.config.cloud_escalation.enabled = False
    data = {"messages": [{"role": "user", "content": "design a system"}]}
    xl_resp = _resp("XL weak")
    xl_attempt = Attempt(
        tier="XL", model="ollama_chat/qwen3.6:35b",
        prompt_tokens=50, completion_tokens=100, duration_s=2.0,
        critic_score=None, preview="",
    )
    with patch.object(router, "_critique", AsyncMock(return_value=2)):
        _final, attempts, cloud, handoff = await router._orchestrate(data, xl_resp, xl_attempt)
    assert cloud is False
    assert handoff is True
    assert [a.tier for a in attempts] == ["XL"]


@pytest.mark.asyncio
async def test_call_cloud_skipped_when_api_key_missing(router: TierRouter, monkeypatch) -> None:
    router.config.cloud_escalation.enabled = True
    router.config.cloud_escalation.api_key_env = "DEFINITELY_NOT_SET_XYZ"
    monkeypatch.delenv("DEFINITELY_NOT_SET_XYZ", raising=False)
    result = await router._call_cloud([{"role": "user", "content": "hi"}])
    assert result is None
