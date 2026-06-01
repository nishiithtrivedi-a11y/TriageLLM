"""LLM classifier + critic — httpx mocked so no Ollama needed."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from router_hook import classify_llm, critic_score


def _mock_response(text: str) -> MagicMock:
    r = MagicMock()
    r.json.return_value = {"response": text}
    return r


@pytest.mark.asyncio
async def test_classify_llm_parses_tier_letter() -> None:
    with patch("router_hook.httpx.AsyncClient") as ClientCls:
        client = ClientCls.return_value.__aenter__.return_value
        client.post = AsyncMock(return_value=_mock_response("XL"))
        result = await classify_llm("design a distributed system")
    assert result == "XL"


@pytest.mark.asyncio
async def test_classify_llm_handles_xl_before_l() -> None:
    """'XL' must be matched before 'L' so the substring doesn't steal the match."""
    with patch("router_hook.httpx.AsyncClient") as ClientCls:
        client = ClientCls.return_value.__aenter__.return_value
        client.post = AsyncMock(return_value=_mock_response(" XL "))
        result = await classify_llm("design a distributed system")
    assert result == "XL"


@pytest.mark.asyncio
async def test_classify_llm_returns_none_on_failure() -> None:
    with patch("router_hook.httpx.AsyncClient") as ClientCls:
        client = ClientCls.return_value.__aenter__.return_value
        client.post = AsyncMock(side_effect=RuntimeError("connection refused"))
        result = await classify_llm("anything")
    assert result is None


@pytest.mark.asyncio
async def test_critic_score_extracts_digit() -> None:
    with patch("router_hook.httpx.AsyncClient") as ClientCls:
        client = ClientCls.return_value.__aenter__.return_value
        client.post = AsyncMock(return_value=_mock_response("Score: 4"))
        score = await critic_score("task", "answer")
    assert score == 4


@pytest.mark.asyncio
async def test_critic_score_returns_none_on_unparseable() -> None:
    with patch("router_hook.httpx.AsyncClient") as ClientCls:
        client = ClientCls.return_value.__aenter__.return_value
        client.post = AsyncMock(return_value=_mock_response("hello world"))
        score = await critic_score("task", "answer")
    assert score is None


@pytest.mark.asyncio
async def test_critic_score_handles_network_error() -> None:
    with patch("router_hook.httpx.AsyncClient") as ClientCls:
        client = ClientCls.return_value.__aenter__.return_value
        client.post = AsyncMock(side_effect=TimeoutError("slow"))
        score = await critic_score("task", "answer")
    assert score is None
