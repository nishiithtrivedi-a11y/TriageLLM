"""Streaming critic hook: accumulates chunks, appends handoff on weak score."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from router_hook import TierRouter


def _chunk(delta_text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            delta=SimpleNamespace(content=delta_text),
            finish_reason=None,
        )]
    )


async def _async_iter(items):
    for item in items:
        yield item


@pytest.fixture
def router() -> TierRouter:
    return TierRouter()


@pytest.mark.asyncio
async def test_streaming_passes_chunks_through_for_tier_S(router: TierRouter) -> None:
    chunks = [_chunk("hello "), _chunk("world")]
    request_data = {
        "messages": [{"role": "user", "content": "rename foo"}],
        "_router_state": {
            "initial_tier": "S", "started_at": 0.0,
            "requested": "local-auto", "alias": "local-s",
            "tokens": 4, "score": -2, "signals": ["s_kw=1"],
            "classifier": "rules",
        },
    }
    with patch.object(router, "_critique", AsyncMock(return_value=2)) as critique:
        out: list = []
        async for c in router.async_post_call_streaming_iterator_hook(
            user_api_key_dict=None,
            response=_async_iter(chunks),
            request_data=request_data,
        ):
            out.append(c)
    # S is never critiqued; no extra chunk appended
    assert len(out) == 2
    assert critique.call_count == 0


@pytest.mark.asyncio
async def test_streaming_appends_handoff_when_critic_weak(router: TierRouter) -> None:
    router.config.critic_pass_threshold = 4
    chunks = [_chunk("partial answer "), _chunk("incomplete")]
    request_data = {
        "messages": [{"role": "user", "content": "refactor this module"}],
        "_router_state": {
            "initial_tier": "L", "started_at": 0.0,
            "requested": "local-auto", "alias": "local-l",
            "tokens": 4, "score": 2, "signals": [],
            "classifier": "rules",
        },
    }
    with patch.object(router, "_critique", AsyncMock(return_value=2)):
        out: list = []
        async for c in router.async_post_call_streaming_iterator_hook(
            user_api_key_dict=None,
            response=_async_iter(chunks),
            request_data=request_data,
        ):
            out.append(c)
    # 2 upstream chunks + 1 appended handoff chunk
    assert len(out) == 3
    appended_content = out[-1].choices[0].delta.content
    assert "LOCAL STACK EXHAUSTED" in appended_content


@pytest.mark.asyncio
async def test_streaming_no_handoff_when_critic_passes(router: TierRouter) -> None:
    chunks = [_chunk("great answer")]
    request_data = {
        "messages": [{"role": "user", "content": "refactor this"}],
        "_router_state": {
            "initial_tier": "L", "started_at": 0.0,
            "requested": "local-auto", "alias": "local-l",
            "tokens": 4, "score": 2, "signals": [],
            "classifier": "rules",
        },
    }
    with patch.object(router, "_critique", AsyncMock(return_value=5)):
        out: list = []
        async for c in router.async_post_call_streaming_iterator_hook(
            user_api_key_dict=None,
            response=_async_iter(chunks),
            request_data=request_data,
        ):
            out.append(c)
    assert len(out) == 1  # no extra chunk
