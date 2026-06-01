"""Pattern 3: tier-aware soft-pass on critic failure (eviction-cascade defence).

When the critic returns None (timeout / error), tiers in soft_pass_tiers should
ship the answer instead of cascading through every bigger tier uselessly.
Tiers NOT in soft_pass_tiers should still escalate — this protects high-stakes
work where critique matters more.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from router_hook import Attempt, TierRouter


def _resp(text: str, prompt_t: int = 50, completion_t: int = 100):
    msg = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=msg, delta=SimpleNamespace(content=text), finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=prompt_t, completion_tokens=completion_t)
    return SimpleNamespace(choices=[choice], usage=usage)


def _attempt(tier: str) -> Attempt:
    return Attempt(
        tier=tier, model=f"ollama_chat/{tier.lower()}",
        prompt_tokens=50, completion_tokens=100, duration_s=1.0,
        critic_score=None, preview="",
    )


@pytest.fixture
def router() -> TierRouter:
    # Stub out _warmup so __init__ doesn't fire a real ~30s Ollama call.
    # (Bare TierRouter() would otherwise hit Ollama and block under pytest's
    # event loop — and with a real Ollama up, repeatedly load the critic.)
    with patch.object(TierRouter, "_warmup", AsyncMock(return_value=None)):
        r = TierRouter()
    r.config.cloud_escalation.enabled = False
    return r


# ─── Soft-pass triggers on tier in soft_pass_tiers ────────────────────────

@pytest.mark.asyncio
async def test_soft_pass_on_M_when_critic_fails(router: TierRouter) -> None:
    """Default config has 'M' in soft_pass_tiers. M critic timeout → ship, don't escalate."""
    router.config.soft_pass_tiers = ("S", "M")

    with patch.object(router, "_critique", AsyncMock(return_value=None)) as critique, \
         patch.object(router, "_call_tier", AsyncMock()) as call_tier:
        final, attempts, cloud, handoff = await router._orchestrate(
            {"messages": [{"role": "user", "content": "write a function"}]},
            _resp("M's answer"),
            _attempt("M"),
        )

    assert len(attempts) == 1
    assert attempts[0].tier == "M"
    assert handoff is False
    assert cloud is False
    call_tier.assert_not_called()  # critically: did NOT escalate
    critique.assert_called_once()


@pytest.mark.asyncio
async def test_soft_pass_NOT_triggered_on_L_when_critic_fails(router: TierRouter) -> None:
    """L is NOT in default soft_pass_tiers. Critic failure → still escalate."""
    router.config.soft_pass_tiers = ("S", "M")
    xl_resp = _resp("XL's answer")
    xl_attempt = _attempt("XL")

    with patch.object(router, "_critique", AsyncMock(return_value=None)), \
         patch.object(router, "_call_tier", AsyncMock(return_value=(xl_resp, xl_attempt))):
        final, attempts, cloud, handoff = await router._orchestrate(
            {"messages": [{"role": "user", "content": "design a system"}]},
            _resp("L's answer"),
            _attempt("L"),
        )

    assert [a.tier for a in attempts] == ["L", "XL"]
    assert handoff is True  # both failed → handoff


@pytest.mark.asyncio
async def test_soft_pass_can_be_widened_to_all_tiers(router: TierRouter) -> None:
    """User can soft-pass everywhere by configuring soft_pass_tiers = (S, M, L, XL)."""
    router.config.soft_pass_tiers = ("S", "M", "L", "XL")

    with patch.object(router, "_critique", AsyncMock(return_value=None)), \
         patch.object(router, "_call_tier", AsyncMock()) as call_tier:
        _final, attempts, cloud, handoff = await router._orchestrate(
            {"messages": [{"role": "user", "content": "design a system"}]},
            _resp("L's answer"),
            _attempt("L"),
        )

    assert len(attempts) == 1
    assert handoff is False
    call_tier.assert_not_called()


@pytest.mark.asyncio
async def test_soft_pass_can_be_disabled_entirely(router: TierRouter) -> None:
    """Empty soft_pass_tiers → original (pre-Pattern-3) behavior."""
    router.config.soft_pass_tiers = ()
    xl_resp = _resp("xl")
    xl_attempt = _attempt("XL")

    with patch.object(router, "_critique", AsyncMock(return_value=None)), \
         patch.object(router, "_call_tier", AsyncMock(return_value=(xl_resp, xl_attempt))):
        _final, attempts, cloud, handoff = await router._orchestrate(
            {"messages": [{"role": "user", "content": "write a function"}]},
            _resp("m"),
            _attempt("M"),
        )

    # Without soft-pass, M → L → XL → handoff
    assert [a.tier for a in attempts] == ["M", "XL"]  # mock returns XL each escalation


# ─── Soft-pass does NOT override a real low score ─────────────────────────

@pytest.mark.asyncio
async def test_soft_pass_does_not_override_real_low_score(router: TierRouter) -> None:
    """Critic returns 2 (real low score) on M → escalate normally, ignore soft_pass list."""
    router.config.soft_pass_tiers = ("S", "M")

    # _call_tier mock MUST advance the tier pointer — orchestrator reads
    # attempts[-1].tier to decide the next step. A static return value
    # causes infinite escalation (see commit history / journey doc).
    async def fake_call_tier(messages, tier):
        return _resp(tier.lower()), _attempt(tier)

    with patch.object(router, "_critique", AsyncMock(return_value=2)), \
         patch.object(router, "_call_tier", AsyncMock(side_effect=fake_call_tier)):
        _final, attempts, cloud, handoff = await router._orchestrate(
            {"messages": [{"role": "user", "content": "write a function"}]},
            _resp("m"),
            _attempt("M"),
        )

    # Real 2/5 → still escalate (soft-pass only applies to None scores).
    # Chain walks M → L → XL → handoff (cloud disabled in fixture).
    assert [a.tier for a in attempts] == ["M", "L", "XL"]
    assert attempts[0].critic_score == 2
    assert handoff is True
