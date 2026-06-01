"""Hermetic tests for capability_router (no real Ollama).

Spec: the design notes
"""
from dataclasses import asdict
import pytest

import capability_router as cr


def test_recommendation_has_all_expected_fields():
    rec = cr.Recommendation(
        category="default",
        tier_min=None,
        tier_max=None,
        tier_prefer=None,
        reason_code="default:no-rule-fired",
        signals=[],
        confidence=0.0,
        classifier_used="default",
        pack="coder",
    )
    d = asdict(rec)
    assert set(d.keys()) == {
        "category", "tier_min", "tier_max", "tier_prefer",
        "reason_code", "signals", "confidence",
        "classifier_used", "pack",
    }


def test_module_exposes_public_surface():
    import inspect
    assert inspect.iscoroutinefunction(cr.classify_capability)   # async by contract
    assert callable(cr.recommend_tier)
    assert callable(cr.shadow_columns)
    assert hasattr(cr, "Recommendation")


# Helpers
import asyncio


def _mk_config(enabled=True, packs=None, threshold=0.6, use_tiebreaker=False):
    """Build a minimal config object the classifier reads from."""
    class CapPacks:
        def __init__(self, coder=True, writing=False, analyst=False):
            self.coder = coder
            self.writing = writing
            self.analyst = analyst

    class CapCfg:
        def __init__(self, enabled, packs, threshold, use_tiebreaker):
            self.enabled = enabled
            self.mode = "shadow"
            self.use_llm_tiebreaker = use_tiebreaker
            self.confidence_threshold = threshold
            self.packs = CapPacks(**(packs or {}))

    class RootCfg:
        def __init__(self, cap):
            self.capability_routing = cap
            self.classifier_model = "qwen2.5:0.5b"
            self.llm_classifier_timeout_s = 3.0
            self.llm_classifier_min_chars = 250

    return RootCfg(CapCfg(enabled, packs or {}, threshold, use_tiebreaker))


def _classify(text):
    """Sync wrapper for the async classifier (rules-only path is fast)."""
    cfg = _mk_config()
    return asyncio.run(cr.classify_capability(text, [{"role": "user", "content": text}], cfg))


# Category positive tests (one per category)
@pytest.mark.parametrize("prompt, expected_category", [
    ("rename foo to bar",                            "quick_question"),
    ("explain how Python decorators work",           "explanation_or_summary"),
    ("return the result as JSON with field name",    "structured_output"),
    ("compare these two designs and evaluate",       "analytical_task"),
    ("write a short story about a cat",              "creative_generation"),
    ("refactor this function to use a list comprehension", "modification_or_edit"),
    ("plan the migration steps for moving to v3",    "multi_step_or_planning"),
    ("review this auth code for SQL injection risk", "high_risk"),
])
def test_category_positive(prompt, expected_category):
    rec = _classify(prompt)
    assert rec.category == expected_category, f"got {rec.category} for {prompt!r}"


def test_high_risk_precedence_overrides_other_signals():
    # Intent: a high-risk credential-action signal must beat the
    # "rename" quick_question signal. The tightened patterns (Issue #6)
    # require an actionable verb near the credential noun, so use
    # "rotate" + "API secret" to fire credential-action.
    rec = _classify("rename the helper, then rotate the API secret on prod")
    assert rec.category == "high_risk"
    assert rec.classifier_used == "high-risk-precedence"
    assert rec.tier_min == "XL"


# ----- Issue #6 benchmark: false-positive set -----------------------
# These prompts MUST classify as something OTHER THAN high_risk after the
# Option-A pattern tightening lands. They were over-matching under the
# original word-boundary-only patterns because keywords like "billing",
# "financial", "MFA", "authentication" fire on any topical mention.
_HIGH_RISK_FALSE_POSITIVES = [
    "add a billing field to the user model",
    "compare financial sectors for Q3 portfolio review",
    "how do I write an authentication form in React",
    "pseudo SSO migration helper docs",
    "render a chart of financial performance",
    "name a billing-friendly subscription tier",
    "explain MFA in plain English",
    "draft an onboarding email about our authentication options",
    "what is OAuth, in simple terms",
    "summarize a tutorial on payment gateways",
    "explain what a password manager is",
    "what is an API key in simple terms",
    "what is an access token",
    "review the access control list",
    "explain security policies in plain English",
    "what is CSRF protection",
    "explain SQL injection in plain English",
    "tutorial on SQLi for beginners",
    "draft an email explaining what XSS means",
    "compare authentication strategies and CSRF defenses for a blog post",
    "explain SSRF to a junior developer",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", _HIGH_RISK_FALSE_POSITIVES)
async def test_high_risk_no_longer_false_positives(prompt, monkeypatch):
    """Issue #6: prompts that merely mention a risk-adjacent topic word
    without an actionable verb or a system-noun context must NOT be
    classified as high_risk. Word-boundary-only matching was producing
    inflated XL recommendations in the shadow ledger."""
    cfg = _mk_config()
    rec = await cr.classify_capability(
        prompt, [{"role": "user", "content": prompt}], cfg
    )
    assert rec.category != "high_risk", (
        f"Expected non-high_risk for {prompt!r}, "
        f"got high_risk with reason_code={rec.reason_code!r}"
    )


# ----- Issue #6 benchmark: true-positive set ------------------------
# These prompts MUST continue to classify as high_risk after the
# pattern tightening. Each one is an unambiguous risk operation
# (destructive SQL/shell, credential leak, auth system work, etc.)
_HIGH_RISK_TRUE_POSITIVES = [
    "DROP TABLE users in production",
    "rm -rf / on the live server",
    "delete from accounts where id = 1",
    "truncate table transactions",
    "rotate the API key for the prod database",
    "leaked credential in the logs, what now?",
    "production db schema migration plan",
    "is this SQL injection exploitable",
    "design an authentication system for the API gateway",
    "fix the OAuth bypass in our SSO middleware",
    "fix the payment-gateway race condition in production",
    "hardcoded passwords in the config.py file",
    "rotating the API keys for our microservices",
    "disable security checks",
    "bypass the access control list",
    "skip input validation in dev",
    "strip CORS headers from production responses",
    "weaken the security policy for staging",
    "TLS verification was disabled in the new build",
    "found an XSS in our login form",
    "vulnerable to CSRF in production",
    "patch the SQLi bug we found yesterday",
    "detect SSRF in the file upload endpoint",
    "exploit the SQL injection on /admin",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", _HIGH_RISK_TRUE_POSITIVES)
async def test_high_risk_true_positives_still_fire(prompt, monkeypatch):
    """Issue #6: tightening the patterns must NOT lose recall on
    genuine high-risk prompts. Each of these should still classify
    as high_risk via one of the existing reason codes."""
    cfg = _mk_config()
    rec = await cr.classify_capability(
        prompt, [{"role": "user", "content": prompt}], cfg
    )
    assert rec.category == "high_risk", (
        f"Expected high_risk for {prompt!r}, "
        f"got category={rec.category!r} reason_code={rec.reason_code!r}"
    )
    assert rec.reason_code.startswith("high-risk:"), (
        f"reason_code should start with 'high-risk:', got {rec.reason_code!r}"
    )


def test_high_risk_reason_code_is_first_pattern_not_last():
    """When multiple high-risk patterns fire, reason_code must be the FIRST
    one in _HIGH_RISK_PATTERNS (deterministic), not whichever happened last."""
    # 'rotate ... API key' (credential-action is pattern[0]) +
    # 'review ... auth flow' (auth-system action-verb form is later).
    rec = _classify("rotate the api key and review our oauth flow")
    assert rec.category == "high_risk"
    assert rec.reason_code == "high-risk:credential-action"
    # Both signals should accumulate
    assert "credential-action" in rec.signals
    assert "auth-system" in rec.signals


def test_long_context_overrides_other_signals_when_prompt_huge():
    big = ("explain this code: " + "x = 1\n" * 3000)
    rec = _classify(big)
    assert rec.category == "long_context"
    assert rec.tier_max == "L"


def test_default_when_nothing_matches():
    rec = _classify("hi")
    assert rec.category == "default"
    assert rec.tier_min is None and rec.tier_max is None


def test_coder_pack_default_on_fires_coder_signals():
    rec = _classify("refactor this function")
    assert "modification_or_edit" == rec.category
    assert "coder" in rec.pack


def test_coder_pack_off_does_not_fire_coder_signals():
    cfg = _mk_config(packs={"coder": False})
    rec = asyncio.run(cr.classify_capability("refactor this function", [], cfg))
    assert "coder" not in rec.pack


def test_writing_pack_classifies_creative_writing():
    cfg = _mk_config(packs={"coder": False, "writing": True})
    rec = asyncio.run(cr.classify_capability(
        "write a short narrative about an astronaut", [], cfg))
    assert rec.category == "creative_generation"
    assert "writing" in rec.pack


def test_runtime_strings_are_cp1252_safe():
    samples = [
        _classify("refactor this"),
        _classify("return JSON"),
        _classify("hi"),
        _classify("review auth code for sql injection"),
    ]
    for rec in samples:
        rec.reason_code.encode("cp1252")
        for s in rec.signals:
            s.encode("cp1252")


# --- Task 3: LLM tie-breaker tests -------------------------------------------

from unittest.mock import AsyncMock, MagicMock, patch


def _mock_llm_response(category_label: str):
    """Build a mock httpx response that returns Ollama-shaped JSON."""
    resp = MagicMock()
    resp.json.return_value = {"response": category_label}
    return resp


@pytest.mark.asyncio
async def test_llm_tiebreaker_fires_on_long_ambiguous_prompt():
    cfg = _mk_config(use_tiebreaker=True)
    text = ("can you do something with this text " * 10).strip()
    with patch("capability_router.httpx.AsyncClient") as ClientCls:
        client = ClientCls.return_value.__aenter__.return_value
        async def fake_post(url, json):
            return _mock_llm_response("analytical_task")
        client.post = AsyncMock(side_effect=fake_post)
        rec = await cr.classify_capability(text, [], cfg)
    assert rec.category == "analytical_task"
    assert rec.classifier_used == "rules+llm"


@pytest.mark.asyncio
async def test_llm_tiebreaker_does_not_fire_for_short_prompt():
    cfg = _mk_config(use_tiebreaker=True)
    with patch("capability_router.httpx.AsyncClient") as ClientCls:
        client = ClientCls.return_value.__aenter__.return_value
        client.post = AsyncMock(side_effect=AssertionError("LLM must not be called for short prompts"))
        rec = await cr.classify_capability("hi", [], cfg)
    assert rec.category == "default"
    assert rec.classifier_used == "default"


@pytest.mark.asyncio
async def test_llm_tiebreaker_does_not_override_high_risk_precedence():
    cfg = _mk_config(use_tiebreaker=True)
    with patch("capability_router.httpx.AsyncClient") as ClientCls:
        client = ClientCls.return_value.__aenter__.return_value
        client.post = AsyncMock(side_effect=AssertionError("LLM must not be called when high-risk wins"))
        rec = await cr.classify_capability("delete from users where 1=1", [], cfg)
    assert rec.category == "high_risk"
    assert rec.classifier_used == "high-risk-precedence"


@pytest.mark.asyncio
async def test_llm_tiebreaker_failure_falls_back_to_default():
    cfg = _mk_config(use_tiebreaker=True)
    text = "x " * 200
    with patch("capability_router.httpx.AsyncClient") as ClientCls:
        client = ClientCls.return_value.__aenter__.return_value
        client.post = AsyncMock(side_effect=Exception("Ollama unreachable"))
        rec = await cr.classify_capability(text, [], cfg)
    assert rec.category == "default"
    assert rec.classifier_used == "default"


@pytest.mark.asyncio
async def test_llm_tiebreaker_disabled_in_config():
    cfg = _mk_config(use_tiebreaker=False)
    text = "x " * 200
    with patch("capability_router.httpx.AsyncClient") as ClientCls:
        client = ClientCls.return_value.__aenter__.return_value
        client.post = AsyncMock(side_effect=AssertionError("LLM must not be called when disabled"))
        rec = await cr.classify_capability(text, [], cfg)
    assert rec.classifier_used == "default"


@pytest.mark.asyncio
async def test_tiebreaker_path_runtime_strings_are_cp1252_safe():
    """When the LLM tie-breaker fires, reason_code + signals must still be ASCII.

    Uses neutral text that clears the min_chars threshold but fires no rule
    patterns, so the rules path falls through and the tie-breaker is invoked.
    """
    cfg = _mk_config(use_tiebreaker=True)
    text = ("can you do something useful with this input " * 8).strip()
    with patch("capability_router.httpx.AsyncClient") as ClientCls:
        client = ClientCls.return_value.__aenter__.return_value
        client.post = AsyncMock(return_value=_mock_llm_response("analytical_task"))
        rec = await cr.classify_capability(text, [], cfg)
    assert rec.classifier_used == "rules+llm"
    rec.reason_code.encode("cp1252")   # raises if non-ASCII
    for s in rec.signals:
        s.encode("cp1252")


# --- Task 4: recommend_tier + shadow_columns tests ----------------------------

# Helper to build a Recommendation quickly
def _rec(cat, tmin=None, tmax=None, tprefer=None):
    rules = cr.TIER_RULES[cat]
    return cr.Recommendation(
        category=cat,
        tier_min=tmin if tmin is not None else rules["min"],
        tier_max=tmax if tmax is not None else rules["max"],
        tier_prefer=tprefer if tprefer is not None else rules["prefer"],
        reason_code=f"{cat}:test",
        signals=[],
        confidence=0.8,
        classifier_used="rules",
        pack="coder",
    )


def test_recommend_tier_structured_output_raises_floor_from_s_to_m():
    rec = _rec("structured_output")  # min=M
    out = cr.recommend_tier(rec, tier_router_choice="S", config=_mk_config())
    # The Recommendation's tier_* fields are bookkeeping; the SELECTED tier
    # the dashboard records is derived. We expose this via shadow_columns.
    cols = cr.shadow_columns(out, tier_router_choice="S")
    assert cols["cap_recommended_tier"] == "M"


def test_recommend_tier_long_context_caps_xl_to_l():
    rec = _rec("long_context")  # max=L
    cols = cr.shadow_columns(cr.recommend_tier(rec, "XL", _mk_config()), "XL")
    assert cols["cap_recommended_tier"] == "L"


def test_recommend_tier_high_risk_min_xl():
    rec = _rec("high_risk")  # min=XL
    cols = cr.shadow_columns(cr.recommend_tier(rec, "S", _mk_config()), "S")
    assert cols["cap_recommended_tier"] == "XL"


def test_recommend_tier_default_preserves_router_choice():
    rec = _rec("default")
    cols = cr.shadow_columns(cr.recommend_tier(rec, "M", _mk_config()), "M")
    assert cols["cap_recommended_tier"] == "M"
    assert cols["cap_agrees_with_tier"] == 1


def test_recommend_tier_prefer_used_when_no_min_max_override():
    rec = _rec("quick_question")  # prefer=S
    cols = cr.shadow_columns(cr.recommend_tier(rec, "M", _mk_config()), "M")
    assert cols["cap_recommended_tier"] == "S"
    assert cols["cap_agrees_with_tier"] == 0


def test_shadow_columns_returns_all_eight_keys():
    rec = _rec("structured_output")
    cols = cr.shadow_columns(rec, "S")
    assert set(cols.keys()) == {
        "cap_category", "cap_recommended_tier", "cap_reason_code",
        "cap_signals", "cap_confidence", "cap_classifier_used",
        "cap_pack", "cap_agrees_with_tier",
    }


def test_shadow_columns_serializes_signals_as_comma_string():
    rec = cr.Recommendation(
        category="quick_question", tier_min=None, tier_max=None, tier_prefer="S",
        reason_code="quick_question:test", signals=["a", "b", "c"],
        confidence=0.8, classifier_used="rules", pack="coder",
    )
    cols = cr.shadow_columns(cr.recommend_tier(rec, "M", _mk_config()), "M")
    assert cols["cap_signals"] == "a,b,c"


def test_apply_tier_rules_unknown_tier_degrades_to_no_override():
    """Defensive: unknown tier_router_choice (not S/M/L/XL) must return as-is,
    not silently produce a misleading min-raise/max-cap result."""
    rec = _rec("structured_output")  # min=M would normally raise from S
    cols = cr.shadow_columns(cr.recommend_tier(rec, "weird-tier", _mk_config()), "weird-tier")
    # No override applied -> cap_recommended_tier passes through.
    assert cols["cap_recommended_tier"] == "weird-tier"
    assert cols["cap_agrees_with_tier"] == 1


# --- Task 6: Hook integration tests ------------------------------------------

def _make_router_with_capability(enabled=True):
    """Build a TierRouter with capability_routing enabled/disabled in its config."""
    import router_hook
    from router_hook import CapabilityRoutingConfig, CapabilityPacksConfig
    with patch.object(router_hook.TierRouter, "_warmup", AsyncMock(return_value=None)):
        r = router_hook.TierRouter()
    r.config.capability_routing = CapabilityRoutingConfig(
        enabled=enabled,
        mode="shadow",
        use_llm_tiebreaker=False,
        confidence_threshold=0.6,
        packs=CapabilityPacksConfig(coder=True, writing=False, analyst=False),
    )
    return r


@pytest.mark.asyncio
async def test_pre_call_hook_disabled_does_not_populate_capability_state():
    import router_hook
    r = _make_router_with_capability(enabled=False)
    allow = AsyncMock(return_value=(True, "ok"))
    with patch.object(router_hook._ollama_circuit, "preflight", allow):
        data = {"model": "local-auto", "messages": [{"role": "user", "content": "rename foo"}]}
        await r.async_pre_call_hook(None, None, data, "completion")
    state = data["_router_state"]
    assert state.get("capability") is None


@pytest.mark.asyncio
async def test_pre_call_hook_enabled_populates_capability_state():
    import router_hook
    r = _make_router_with_capability(enabled=True)
    allow = AsyncMock(return_value=(True, "ok"))
    with patch.object(router_hook._ollama_circuit, "preflight", allow):
        data = {"model": "local-auto", "messages": [{"role": "user", "content": "return JSON with field name"}]}
        await r.async_pre_call_hook(None, None, data, "completion")
    cap = data["_router_state"].get("capability")
    assert cap is not None
    assert cap["cap_category"] == "structured_output"
    assert cap["cap_recommended_tier"] in {"S", "M", "L", "XL"}


@pytest.mark.asyncio
async def test_pre_call_hook_capability_failure_does_not_break_request():
    """If classify_capability raises, the hook must still complete and route normally."""
    import router_hook
    r = _make_router_with_capability(enabled=True)
    allow = AsyncMock(return_value=(True, "ok"))
    import capability_router
    with patch.object(router_hook._ollama_circuit, "preflight", allow):
        with patch.object(capability_router, "classify_capability",
                          AsyncMock(side_effect=RuntimeError("boom"))):
            data = {"model": "local-auto", "messages": [{"role": "user", "content": "hi"}]}
            await r.async_pre_call_hook(None, None, data, "completion")
    # Existing routing fields populated as usual
    assert data["model"] in {"local-s", "local-m", "local-l", "local-xl"}
    assert data["_router_state"]["initial_tier"] in {"S", "M", "L", "XL"}
    # capability state is NOT present (failed)
    assert data["_router_state"].get("capability") is None
