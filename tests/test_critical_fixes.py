"""Regression tests for the 5 critical bugs caught in the multi-agent audit.

Each test is named after the audit's bug id (C-1 .. C-5). Each one:
  - Fails on the pre-fix code path (verified manually before commit)
  - Passes on the fix
  - Is intentionally narrow: it asserts the specific bug behavior, not adjacent
    refactoring concerns. If a future change re-introduces the bug, the test
    fails with a message that points back to the audit id.
"""
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from router_hook import Attempt, TierRouter


def _resp(text: str, prompt_t: int = 50, completion_t: int = 100):
    """Fake ModelResponse-shaped object with mutable message/delta."""
    msg = SimpleNamespace(content=text)
    delta = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=msg, delta=delta, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=prompt_t, completion_tokens=completion_t)
    return SimpleNamespace(choices=[choice], usage=usage)


def _attempt(tier: str, score: int | None = None) -> Attempt:
    return Attempt(
        tier=tier, model=f"ollama_chat/{tier.lower()}-model",
        prompt_tokens=50, completion_tokens=100, duration_s=1.0,
        critic_score=score, preview="",
    )


@pytest.fixture
def router() -> TierRouter:
    # Stub _warmup so the fixture doesn't fire a real Ollama call.
    with patch.object(TierRouter, "_warmup", AsyncMock(return_value=None)):
        r = TierRouter()
    r.config.cloud_escalation.enabled = False
    return r


# ─── C-1: handoff draft uses highest-scoring response, not last ────────────

@pytest.mark.asyncio
async def test_c1_handoff_uses_best_scoring_response(router: TierRouter) -> None:
    """When M scores higher than L and XL, the handoff draft must contain M's
    actual answer text — not whatever response is `current_resp` at the end."""
    router.config.critic_pass_threshold = 4

    m_resp = _resp("M's GOOD answer — should appear as best draft")
    l_resp = _resp("L's mediocre answer")
    xl_resp = _resp("XL's WORST answer")

    # _call_tier returns L then XL in sequence; each one different so we can
    # tell which response's text ended up in the handoff message.
    call_seq = iter([
        (l_resp, _attempt("L")),
        (xl_resp, _attempt("XL")),
    ])
    async def fake_call_tier(_messages, _tier):
        return next(call_seq)

    # Critic scores M=3, L=2, XL=1 — so M is the highest non-passing score.
    critique_seq = iter([3, 2, 1])
    async def fake_critique(_task, _ans):
        return next(critique_seq)

    with patch.object(router, "_critique", side_effect=fake_critique), \
         patch.object(router, "_call_tier", side_effect=fake_call_tier):
        final, attempts, _cloud, handoff = await router._orchestrate(
            {"messages": [{"role": "user", "content": "write me a function"}]},
            m_resp,
            _attempt("M"),
        )

    # Whole chain ran.
    assert [a.tier for a in attempts] == ["M", "L", "XL"]
    assert handoff is True

    # Handoff message must contain M's answer (the actually-best draft),
    # NOT XL's (the most-recent but lowest-scoring).
    final_content = final.choices[0].message.content
    assert "M's GOOD answer" in final_content, (
        "C-1 regression: best draft is not M's text. Bug: best_answer always "
        "reads current_resp instead of the highest-scoring response."
    )
    assert "XL's WORST answer" not in final_content, (
        "C-1 regression: handoff contains the WORST response's text. "
        "Audit trail is lying to the user."
    )


# ─── C-2: streaming handoff doesn't mutate already-yielded chunks ──────────

@pytest.mark.asyncio
async def test_c2_streaming_handoff_does_not_mutate_yielded_chunk(
    router: TierRouter,
) -> None:
    """Streaming handoff must deepcopy last_chunk before mutating it.
    Otherwise the previously-yielded chunk's content gets rewritten under
    the client's feet."""
    router.config.critic_pass_threshold = 4

    chunk1 = SimpleNamespace(
        choices=[SimpleNamespace(
            delta=SimpleNamespace(content="UPSTREAM_ORIGINAL_TEXT"),
            finish_reason=None,
        )]
    )

    async def _async_iter(items):
        for it in items:
            yield it

    request_data = {
        "messages": [{"role": "user", "content": "refactor this module"}],
        "_router_state": {
            "initial_tier": "L", "started_at": 0.0,
            "requested": "local-auto", "alias": "local-l",
            "tokens": 4, "score": 2, "signals": [],
            "classifier": "rules",
        },
    }

    with patch.object(router, "_critique", AsyncMock(return_value=1)):
        collected: list = []
        async for c in router.async_post_call_streaming_iterator_hook(
            user_api_key_dict=None,
            response=_async_iter([chunk1]),
            request_data=request_data,
        ):
            collected.append(c)

    # We expect 2 chunks: the original + the appended handoff note.
    assert len(collected) == 2

    # The ORIGINAL chunk (index 0) must still hold its original text.
    # Pre-fix, the in-place mutation would have rewritten chunk1.delta.content
    # to the handoff note, so collected[0].delta.content would NOT contain
    # the original.
    assert collected[0].choices[0].delta.content == "UPSTREAM_ORIGINAL_TEXT", (
        "C-2 regression: streaming handoff mutated the already-yielded chunk "
        "in place. Use copy.deepcopy on last_chunk before mutating."
    )
    # And the new chunk has the handoff note.
    assert "LOCAL STACK EXHAUSTED" in collected[1].choices[0].delta.content


# ─── C-3: streaming handoff yield failure clears handoff flag ──────────────

@pytest.mark.asyncio
async def test_c3_streaming_handoff_failure_clears_handoff_flag(
    router: TierRouter, tmp_path, monkeypatch,
) -> None:
    """If yielding the handoff chunk raises, the SQLite row must say
    handoff=0 — otherwise audit log claims a handoff was sent when the
    client never received it."""
    import router_hook
    monkeypatch.setattr(router_hook, "DB_PATH", tmp_path / "rt.sqlite")
    # Bootstrap the schema at the new path (otherwise _log_decision would
    # fail silently with "no such table" — this is H-4 in the audit, the
    # log-swallow bug — but for THIS test we just need the table to exist).
    router_hook._init_db()
    router.config.critic_pass_threshold = 4

    # A chunk whose .delta is read-only — assigning to its .content raises,
    # which forces the streaming handoff yield path into the except branch.
    class FrozenDelta:
        content = "frozen original"
        def __setattr__(self, name, value):
            raise AttributeError(f"FrozenDelta has no setter for {name}")

    frozen_chunk = SimpleNamespace(
        choices=[SimpleNamespace(delta=FrozenDelta(), finish_reason=None)]
    )

    async def _async_iter(items):
        for it in items:
            yield it

    request_data = {
        "messages": [{"role": "user", "content": "refactor"}],
        "_router_state": {
            "initial_tier": "L", "started_at": 0.0,
            "requested": "local-auto", "alias": "local-l",
            "tokens": 4, "score": 2, "signals": [],
            "classifier": "rules",
        },
    }

    with patch.object(router, "_critique", AsyncMock(return_value=1)):
        collected: list = []
        async for c in router.async_post_call_streaming_iterator_hook(
            user_api_key_dict=None,
            response=_async_iter([frozen_chunk]),
            request_data=request_data,
        ):
            collected.append(c)

    # Only the original chunk made it through — handoff yield failed.
    assert len(collected) == 1

    # Crucial assertion: the SQLite row reflects reality (no handoff sent).
    with sqlite3.connect(tmp_path / "rt.sqlite") as c:
        rows = c.execute("SELECT handoff FROM decisions").fetchall()
    assert rows == [(0,)], (
        "C-3 regression: handoff=1 logged to DB even though the handoff chunk "
        "yield raised. Audit log diverges from what the client actually saw."
    )


# ─── C-4: handoff injection failure is logged, not silently swallowed ──────

@pytest.mark.asyncio
async def test_c4_handoff_injection_failure_is_logged(
    router: TierRouter, capsys,
) -> None:
    """If assigning the handoff message to current_resp.choices[0].message.content
    raises (e.g. due to a future LiteLLM response-shape change), the failure
    must produce a CRITICAL log line — not be silently swallowed."""
    router.config.critic_pass_threshold = 4

    # Build a response whose .message has a read-only `content` attribute.
    class FrozenMessage:
        content = "original"
        def __setattr__(self, name, value):
            raise AttributeError(f"FrozenMessage cannot set {name}")

    frozen_choice = SimpleNamespace(
        message=FrozenMessage(),
        delta=SimpleNamespace(content="x"),
        finish_reason="stop",
    )
    frozen_resp = SimpleNamespace(
        choices=[frozen_choice],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
    )

    with patch.object(router, "_critique", AsyncMock(return_value=1)):
        _final, _attempts, _cloud, handoff = await router._orchestrate(
            {"messages": [{"role": "user", "content": "design something"}]},
            frozen_resp,
            _attempt("XL"),
        )

    assert handoff is True

    # The critical log line must have fired.
    captured = capsys.readouterr().out
    assert "CRITICAL: failed to inject handoff" in captured, (
        "C-4 regression: handoff injection failure was silently swallowed. "
        "Add a log line so operators can detect response-shape regressions."
    )


# ─── C-5: failed escalation appears in the ledger ──────────────────────────

@pytest.mark.asyncio
async def test_c5_failed_escalation_appears_in_ledger(router: TierRouter) -> None:
    """When _call_tier raises during escalation, the failed tier must show up
    in the attempts list — otherwise the handoff message lies about the chain
    actually being walked, and dashboard analytics undercount failures."""
    router.config.critic_pass_threshold = 4

    async def crashing_call_tier(_messages, _tier):
        raise RuntimeError("Ollama OOM on tier escalation")

    with patch.object(router, "_critique", AsyncMock(return_value=2)), \
         patch.object(router, "_call_tier", side_effect=crashing_call_tier):
        _final, attempts, _cloud, handoff = await router._orchestrate(
            {"messages": [{"role": "user", "content": "refactor"}]},
            _resp("M's weak answer"),
            _attempt("M"),
        )

    # The failed L tier MUST be in the ledger.
    tiers = [a.tier for a in attempts]
    assert tiers == ["M", "L"], (
        f"C-5 regression: ledger missing failed escalation. Got {tiers}, "
        "expected ['M', 'L'] (M with score, L marked as failed)."
    )
    assert attempts[1].critic_score is None
    assert "escalation failed" in attempts[1].preview.lower()
    assert "RuntimeError" in attempts[1].preview or "OOM" in attempts[1].preview
    assert handoff is True  # nowhere else to go
