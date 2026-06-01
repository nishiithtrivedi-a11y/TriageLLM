"""Attempt ledger + handoff message rendering."""
from router_hook import Attempt, render_handoff


def _a(tier: str, model: str, score: int | None, dur: float = 1.0) -> Attempt:
    return Attempt(
        tier=tier, model=model,
        prompt_tokens=100, completion_tokens=200,
        duration_s=dur, critic_score=score,
        preview="...",
    )


def test_handoff_contains_every_attempt_row() -> None:
    attempts = [
        _a("S",  "ollama_chat/qwen2.5-coder:1.5b", 2),
        _a("M",  "ollama_chat/deepseek-coder-v2:16b", 3),
        _a("L",  "ollama_chat/qwen3-coder:30b", 3),
        _a("XL", "ollama_chat/qwen3.6:35b", 3),
    ]
    msg = render_handoff(attempts, threshold=4, cloud_attempted=False, best_answer="draft")
    for tier in ("S", "M", "L", "XL"):
        assert f"Tier {tier}" in msg, f"expected tier {tier} in handoff"
    assert "draft" in msg
    assert "LOCAL STACK EXHAUSTED" in msg


def test_handoff_marks_best_attempt() -> None:
    attempts = [
        _a("S",  "ollama_chat/qwen2.5-coder:1.5b", 1),
        _a("M",  "ollama_chat/deepseek-coder-v2:16b", 3),  # best
        _a("L",  "ollama_chat/qwen3-coder:30b", 2),
    ]
    msg = render_handoff(attempts, threshold=4, cloud_attempted=False, best_answer="d")
    assert "◄ best" in msg
    # The "◄ best" line should be the tier-M one
    best_line = [ln for ln in msg.splitlines() if "◄ best" in ln][0]
    assert "Tier M" in best_line


def test_handoff_totals_sum_tokens_and_time() -> None:
    attempts = [
        _a("L",  "ollama_chat/qwen3-coder:30b", 2, dur=10.0),
        _a("XL", "ollama_chat/qwen3.6:35b", 3, dur=30.0),
    ]
    msg = render_handoff(attempts, threshold=4, cloud_attempted=False, best_answer="d")
    # Each attempt is 100+200=300 tokens, two attempts → 600 tokens, 40.0s
    assert "600 tokens" in msg
    assert "40.0s" in msg


def test_handoff_mentions_cloud_state() -> None:
    attempts = [_a("XL", "m", 2)]
    enabled = render_handoff(attempts, threshold=4, cloud_attempted=True, best_answer="d")
    disabled = render_handoff(attempts, threshold=4, cloud_attempted=False, best_answer="d")
    assert "Cloud escalation was attempted" in enabled
    assert "disabled or not configured" in disabled
