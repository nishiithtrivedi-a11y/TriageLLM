"""Regression: router_hook's runtime prints must be ASCII / cp1252-safe.

Found via the forced-escalation stress test: `print(f"[router] escalating
{tier} -> {nxt}")` originally used a Unicode arrow, which raised
UnicodeEncodeError on Windows' default cp1252 console and aborted the
orchestration loop mid-escalation. The proxy masked it via `chcp 65001`,
but any other importer (scripts, tests) crashed. These tests pin the
runtime print paths to encodable output.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from router_hook import Attempt, TierRouter


def _resp(text):
    msg = SimpleNamespace(content=text)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, delta=SimpleNamespace(content=text), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
    )


def _attempt(tier):
    return Attempt(tier=tier, model=f"ollama_chat/{tier.lower()}", prompt_tokens=10,
                   completion_tokens=20, duration_s=1.0, critic_score=None, preview="")


@pytest.fixture
def router():
    with patch.object(TierRouter, "_warmup", AsyncMock(return_value=None)):
        r = TierRouter()
    r.config.cloud_escalation.enabled = False
    r.config.soft_pass_tiers = ()  # force escalation (not soft-pass) so the print fires
    return r


@pytest.mark.asyncio
async def test_escalation_prints_are_cp1252_safe(router, capsys):
    """Forcing M -> L -> XL escalation must emit only cp1252-encodable output."""
    async def advancing(_messages, tier):
        return _resp(tier), _attempt(tier)

    with patch.object(router, "_critique", AsyncMock(return_value=2)), \
         patch.object(router, "_call_tier", side_effect=advancing):
        await router._orchestrate(
            {"messages": [{"role": "user", "content": "x"}]}, _resp("m"), _attempt("M"))

    out = capsys.readouterr().out
    assert "escalating" in out, "expected the escalation print to fire"
    # Raises UnicodeEncodeError if any non-ASCII (e.g. a Unicode arrow) slipped
    # back into a runtime print — which is exactly the crash this guards against.
    out.encode("cp1252")


def test_config_banner_is_cp1252_safe(capsys):
    """TierRouter.__init__ prints a config banner; it must be cp1252-safe too."""
    with patch.object(TierRouter, "_warmup", AsyncMock(return_value=None)):
        TierRouter()
    out = capsys.readouterr().out
    assert "config loaded" in out
    out.encode("cp1252")
