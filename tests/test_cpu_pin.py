"""CPU pinning + warmup behavior."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from router_hook import classify_llm, critic_score, RouterConfig, TierRouter, load_config


def _mock_response(text: str) -> MagicMock:
    r = MagicMock()
    r.json.return_value = {"response": text}
    return r


@pytest.mark.asyncio
async def test_critic_score_sends_num_gpu_0_when_cpu_only_true() -> None:
    posted_json = {}
    with patch("router_hook.httpx.AsyncClient") as ClientCls:
        client = ClientCls.return_value.__aenter__.return_value
        async def capture_post(url, json):
            posted_json.update(json)
            return _mock_response("3")
        client.post = AsyncMock(side_effect=capture_post)
        await critic_score("t", "a", cpu_only=True)

    assert posted_json["options"].get("num_gpu") == 0
    assert posted_json.get("keep_alive") == -1


@pytest.mark.asyncio
async def test_critic_score_omits_num_gpu_when_cpu_only_false() -> None:
    posted_json = {}
    with patch("router_hook.httpx.AsyncClient") as ClientCls:
        client = ClientCls.return_value.__aenter__.return_value
        async def capture_post(url, json):
            posted_json.update(json)
            return _mock_response("3")
        client.post = AsyncMock(side_effect=capture_post)
        await critic_score("t", "a", cpu_only=False)

    assert "num_gpu" not in posted_json["options"]


@pytest.mark.asyncio
async def test_classify_llm_also_cpu_pinned_by_default() -> None:
    """Classifier is in the same eviction risk as the critic — pin it too."""
    posted_json = {}
    with patch("router_hook.httpx.AsyncClient") as ClientCls:
        client = ClientCls.return_value.__aenter__.return_value
        async def capture_post(url, json):
            posted_json.update(json)
            return _mock_response("S")
        client.post = AsyncMock(side_effect=capture_post)
        await classify_llm("hello world")

    assert posted_json["options"].get("num_gpu") == 0
    assert posted_json.get("keep_alive") == -1


def test_router_config_defaults_have_cpu_only_on_and_softpass_sm() -> None:
    cfg = RouterConfig()
    assert cfg.critic_cpu_only is True
    assert cfg.soft_pass_tiers == ("S", "M")
    assert cfg.warmup_on_startup is True
    assert cfg.critic_timeout_s == 30.0


def test_load_config_reads_cpu_only_and_softpass_overrides(tmp_path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
route_llm:
  critic_cpu_only: false
  soft_pass_tiers: ["S"]
  warmup_on_startup: false
""",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.critic_cpu_only is False
    assert cfg.soft_pass_tiers == ("S",)
    assert cfg.warmup_on_startup is False


@pytest.mark.asyncio
async def test_warmup_calls_critic_once() -> None:
    """Verify _warmup() actually sends a critic call."""
    with patch("router_hook.critic_score", AsyncMock(return_value=3)) as critic:
        r = TierRouter.__new__(TierRouter)
        r.config = RouterConfig()
        await r._warmup()
    assert critic.call_count == 1
    _, kwargs = critic.call_args
    assert kwargs.get("cpu_only") is True
