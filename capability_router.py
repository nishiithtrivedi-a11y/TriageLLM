"""TriageLLM Capability Routing v0.2 -- sidecar classifier + recommender.

SHADOW MODE ONLY in v0.2: this module classifies the request and records
what tier capability routing WOULD have chosen. The actual routing is
unchanged. See the design notes
for the full design (10 categories, pack system, hard constraints).

All runtime strings are ASCII (cp1252-safe -- DEF-003 must not return).
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

# Pre-staged for Task 3 LLM tie-breaker; intentionally imported now to keep
# the import block stable across the 8-task implementation.
import httpx


# --- Recommendation: the audit-trail unit -------------------------------------

@dataclass
class Recommendation:
    """Structured output of the capability classifier. Persists to the
    ledger as 8 additive columns; consumed by the dashboard + future
    advisory/active modes. Stable string codes only -- never prose."""
    category: str                       # one of 10 categories
    tier_min: Optional[str]             # "S"|"M"|"L"|"XL"|None
    tier_max: Optional[str]             # likewise
    tier_prefer: Optional[str]          # likewise
    reason_code: str                    # stable ID, e.g. "high-risk:credentials-mention"
    signals: list[str] = field(default_factory=list)
    confidence: float = 0.0             # 0.0..1.0
    classifier_used: str = "default"    # "rules"|"rules+llm"|"high-risk-precedence"|"default"
    pack: str = ""                      # active pack(s), e.g. "coder" or "coder+writing"


# --- Tier rules per category (spec section 3) ---------------------------------

TIER_RULES: dict[str, dict[str, Optional[str]]] = {
    "quick_question":           {"min": None, "max": None, "prefer": "S"},
    "explanation_or_summary":   {"min": None, "max": "M",  "prefer": None},
    "structured_output":        {"min": "M",  "max": None, "prefer": None},
    "analytical_task":          {"min": None, "max": None, "prefer": "M"},
    "creative_generation":      {"min": None, "max": None, "prefer": "M"},
    "modification_or_edit":     {"min": "L",  "max": None, "prefer": None},
    "multi_step_or_planning":   {"min": None, "max": None, "prefer": "L"},
    "high_risk":                {"min": "XL", "max": None, "prefer": None},
    "long_context":             {"min": None, "max": "L",  "prefer": None},
    "default":                  {"min": None, "max": None, "prefer": None},
}


# --- High-risk pre-check (precedence rule from spec section 3) ----------------

# High-risk patterns. Each requires either an actionable verb (rotate, fix,
# implement, bypass) near the risk keyword, or a system-noun context (system,
# gateway, middleware, flow) -- so a topical mention like "explain MFA" or
# "compare financial sectors" does NOT fire. See Issue #6 / spec section 3.
_HIGH_RISK_PATTERNS = [
    # Credential operations: rotation, leak, hardcode, exposure, reset --
    # requires either an action verb near the noun, or a noun-context
    # like "in production" / "in the logs" / "compromised".
    (re.compile(
        r"\b(rotat(e|ed|ing|ion)|leak(ed|ing|s)?|expos(e|ed|ing|ure)|"
        r"hardcoded?|hard-coded|reset|(steal|stole|stolen|stealing)|"
        r"compromis(e|ed|ing))\b"
        r".{0,40}\b(api[\s_-]?keys?|access[\s_-]?tokens?|secrets?|"
        r"credentials?|passwords?)\b",
        re.I),
     "high-risk:credential-action"),
    (re.compile(
        r"\b(api[\s_-]?keys?|access[\s_-]?tokens?|secrets?|credentials?|"
        r"passwords?)\b"
        r".{0,40}\b(rotat(e|ed|ing|ion)|leak(ed)?|expos(ed|ure)|"
        r"hardcoded?|in\s+production|in\s+the\s+logs?|"
        r"(steal|stole|stolen|stealing)|compromis(e|ed|ing))\b",
        re.I),
     "high-risk:credential-action"),
    # Auth system context: requires a system-noun (bypass/flow/server/...)
    # near the auth keyword, OR an action verb (implement/build/secure/...).
    (re.compile(
        r"\b(auth(entication|orization)?|jwt|oauth\d?|sso|mfa|2fa)[\s-]+"
        r"(bypass|breach|vulnerab(le|ility)|flow|system|server|service|"
        r"provider|middleware|integration|endpoint|gateway|module|"
        r"library|implementation)\b",
        re.I),
     "high-risk:auth-system"),
    (re.compile(
        r"\b(implement(ing)?|build(ing)?|design(ing)?|fix(ing)?|"
        r"patch(ing)?|update|secur(e|ing|ity)|harden(ing)?|review|"
        r"audit|break|bypass|hack|exploit)\b"
        r".{0,40}\b(auth(entication|orization)?|jwt|oauth\d?|sso|"
        r"mfa|2fa)\b",
        re.I),
     "high-risk:auth-system"),
    # Security control bypass: action verb near a security/control compound
    # noun. Bare topical mention of "security" or "access" without a risk
    # verb does NOT fire. Designed for prompts like "disable security checks"
    # or "strip CSRF headers" -- not for "explain security policies" or
    # "review the access control list".
    (re.compile(
        r"\b(disable|disabl(ing|ed)|turn\s+off|turning\s+off|"
        r"skip(ping|ped)?|bypass(ing|ed)?|remov(e|ing|ed)|"
        r"strip(ping|ped)?|circumvent(ing|ed)?|weaken(ing|ed)?|"
        r"opt[\s-]?out\s+of)\b"
        r".{0,40}\b(security|access|authorization|input|csrf|cors|csp|tls|ssl)"
        r"[\s-]+(check|control|gate|guard|filter|policy|policies|header|"
        r"middleware|validation|verification|enforcement|limit|limiting)s?\b",
        re.I),
     "high-risk:security-control-bypass"),
    (re.compile(
        r"\b(security|access|authorization|input|csrf|cors|csp|tls|ssl)"
        r"[\s-]+(check|control|gate|guard|filter|policy|policies|header|"
        r"middleware|validation|verification|enforcement|limit|limiting)s?\b"
        r".{0,40}\b(disabled|skipped|bypassed|removed|stripped|"
        r"circumvented|weakened|turned\s+off)\b",
        re.I),
     "high-risk:security-control-bypass"),
    # Injection / web-vulnerability patterns. Bare keyword (sqli/xss/csrf/ssrf)
    # is NOT enough -- requires either an action verb (find, patch, exploit,
    # detect, vulnerable to) near the keyword, or a direct vulnerability/attack
    # noun after it. Educational prompts ("what is CSRF protection", "explain
    # SQL injection") no longer fire. Spec section 3 (Issue #11, 2026-05-26).
    (re.compile(
        r"\b(exploit(ing|ed)?|find(ing)?|found|patch(ing|ed)?|"
        r"fix(ing|ed)?|audit(ing|ed)?|detect(ing|ed)?|trigger(ing|ed)?|"
        r"abus(e|ed|ing)|attack(ing|ed)?|"
        r"vulnerable\s+to|susceptible\s+to|payload\s+for)\b"
        r".{0,40}\b(sql[\s-]?injection|sqli|xss|csrf|ssrf)\b",
        re.I),
     "high-risk:injection-exploit"),
    (re.compile(
        r"\b(sql[\s-]?injection|sqli|xss|csrf|ssrf)\b"
        r"[\s-]+(vulnerab(le|ility|ilities)|bug|bugs|exploit(s|able)?|"
        r"attack|attacks|vector|vectors|payload|payloads|in\s+production|"
        r"in\s+prod|in\s+the\s+wild|on\s+/\w+|breach)\b",
        re.I),
     "high-risk:injection-exploit"),
    (re.compile(r"\binjection\s+vuln(erab(le|ility|ilities))?\b", re.I),
     "high-risk:injection-exploit"),
    # Destructive shell/SQL idioms -- already specific, no change.
    (re.compile(r"\b(drop\s+table|truncate\s+table|delete\s+from|rm\s+-rf)\b", re.I),
     "high-risk:destructive-op"),
    # DB migration -- already specific, no change.
    (re.compile(r"\b(database\s+migration|schema\s+migration|prod(uction)?\s+db)\b", re.I),
     "high-risk:db-migration"),
    # Financial/payment SYSTEM context: requires a system noun, not just
    # the bare topical word. "billing field" / "financial sectors" stop
    # matching; "billing system" / "payment gateway" / "payment-gateway"
    # still match (whitespace OR hyphen between topic and noun).
    (re.compile(
        r"\b(financial|trading|payment|billing|invoicing)[\s-]+"
        r"(system|service|api|integration|pipeline|engine|module|"
        r"logic|gateway|workflow|process|database|backend|"
        r"infrastructure|settlement|reconciliation)\b",
        re.I),
     "high-risk:financial-system"),
    (re.compile(r"\brisk[\s-]?logic\b", re.I),
     "high-risk:financial-system"),
]


# --- Pack-based keyword rules (spec section 4) --------------------------------
# Generic rules apply regardless of pack selection.
# Pack-specific rules are merged in only when that pack is active.

_GENERIC_RULES: dict[str, list[tuple[re.Pattern, str]]] = {
    "quick_question": [
        (re.compile(r"^\s*(what|how|when|why|where|who|which)\b", re.I), "wh-question"),
        (re.compile(r"\b(what does|what is|define)\b", re.I), "definition-q"),
        (re.compile(r"\brename\b", re.I), "rename"),
    ],
    "explanation_or_summary": [
        (re.compile(r"\b(explain|summari[sz]e|describe|walk me through)\b", re.I), "explain"),
        (re.compile(r"\b(what does this code do|in simple terms)\b", re.I), "in-simple-terms"),
        (re.compile(r"\b(documentation|docstring|readme|comments?)\b", re.I), "doc-related"),
    ],
    "structured_output": [
        (re.compile(r"\bjson\b", re.I), "json"),
        (re.compile(r"\bschema\b", re.I), "schema"),
        (re.compile(r"return.{0,30}as.{0,30}(json|table|list)", re.I), "return-as-format"),
        (re.compile(r"\b(yaml|toml|xml|csv)\b", re.I), "structured-format"),
        (re.compile(r"\bfields?:|\bkeys?:|\bcolumns?:", re.I), "field-list"),
    ],
    "analytical_task": [
        (re.compile(r"\b(compare|contrast|evaluate|assess|analy[sz]e)\b", re.I), "analyze"),
        (re.compile(r"\b(pros and cons|trade-?offs|differences)\b", re.I), "tradeoffs"),
        (re.compile(r"\b(classif|categori|extract)\b", re.I), "classify"),
    ],
    "creative_generation": [
        (re.compile(r"\b(story|narrative|poem|chapter|character|dialogue|prose)\b", re.I), "creative-prose"),
        (re.compile(r"\b(draft (an?|the))\s+(email|letter|memo|article|post)\b", re.I), "draft-comm"),
    ],
    "multi_step_or_planning": [
        (re.compile(r"\b(plan|roadmap|steps to|project plan|milestone)\b", re.I), "plan"),
        (re.compile(r"\b(design (a|the|an|this))\b", re.I), "design-task"),
    ],
}

_CODER_RULES: dict[str, list[tuple[re.Pattern, str]]] = {
    "quick_question": [
        (re.compile(r"\b(syntax|one[\s-]?liner|format(ting)?)\b", re.I), "code-quick"),
    ],
    "modification_or_edit": [
        (re.compile(r"\b(refactor|edit|modify|patch|diff)\b", re.I), "code-edit"),
        (re.compile(r"\bchange\s+this\b", re.I), "code-change"),
    ],
    "creative_generation": [
        (re.compile(
            r"\b(write|implement|create|generate)\s+(a |an |the )?"
            r"(function|class|module|script|method)\b", re.I), "code-gen"),
    ],
    "multi_step_or_planning": [
        (re.compile(r"\b(architect(ure)?|system design|api design)\b", re.I), "code-design"),
    ],
    "explanation_or_summary": [
        (re.compile(r"\b(traceback|stack[\s-]?trace|exception)\b", re.I), "traceback-explain"),
    ],
}

_WRITING_RULES: dict[str, list[tuple[re.Pattern, str]]] = {
    "modification_or_edit": [
        (re.compile(r"\b(revise|proofread|edit (this|my|the))\b", re.I), "text-edit"),
        (re.compile(r"\btranslate\b", re.I), "translate"),
    ],
}

_ANALYST_RULES: dict[str, list[tuple[re.Pattern, str]]] = {
    "analytical_task": [
        (re.compile(r"\b(trend|metric|kpi|insight|correlation)\b", re.I), "analysis-term"),
        (re.compile(r"\b(market|customer|revenue|growth) analysis\b", re.I), "business-analysis"),
    ],
}


# --- LLM tie-breaker (Task 3) -------------------------------------------------
# Used only when rules are ambiguous + prompt is long + explicitly enabled.

_CAP_PROMPT = """You are a task-type classifier. Read the user request and reply
with ONLY one label from this list, no explanation:

quick_question, explanation_or_summary, structured_output, analytical_task,
creative_generation, modification_or_edit, multi_step_or_planning,
high_risk, long_context, default

REQUEST:
{text}

LABEL:"""

_VALID_LABELS = {
    "quick_question", "explanation_or_summary", "structured_output",
    "analytical_task", "creative_generation", "modification_or_edit",
    "multi_step_or_planning", "high_risk", "long_context", "default",
}


async def _llm_tiebreaker(text: str, config: Any) -> Optional[str]:
    """Returns one of _VALID_LABELS (excluding "default") or None (on failure / non-conforming reply)."""
    # Note: reuses the *existing* classifier knobs on RouterConfig
    # (classifier_model, llm_classifier_timeout_s, llm_classifier_min_chars).
    # Only the capability-specific switch lives under config.capability_routing.
    timeout_s = getattr(config, "llm_classifier_timeout_s", 3.0)
    model = getattr(config, "classifier_model", "qwen2.5:0.5b")
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as c:
            r = await c.post(
                f"{base}/api/generate",
                json={
                    "model": model,
                    "prompt": _CAP_PROMPT.format(text=text[:2000]),
                    "stream": False,
                    "options": {"num_predict": 8, "temperature": 0.0, "num_gpu": 0},
                    "keep_alive": -1,
                },
            )
        out = (r.json().get("response") or "").strip().lower()
        for label in _VALID_LABELS:
            if label in out:
                # "default" from the LLM is a non-decision — fall through to
                # the rules-based default fallback instead of recording it
                # as a rules+llm classification.
                return None if label == "default" else label
    except Exception as e:
        print(f"[capability] tie-breaker failed: {type(e).__name__}: {e}")
    return None


def _active_packs(config: Any) -> list[str]:
    """Return ordered list of active pack names, honoring env-var override."""
    env = os.environ.get("TRIAGELLM_CAPABILITY_PACKS")
    if env:
        return [p.strip() for p in env.split(",") if p.strip()]
    packs = []
    if getattr(config.capability_routing.packs, "coder", True):
        packs.append("coder")
    if getattr(config.capability_routing.packs, "writing", False):
        packs.append("writing")
    if getattr(config.capability_routing.packs, "analyst", False):
        packs.append("analyst")
    return packs


def _combine_rules(
    packs: list[str],
) -> dict[str, list[tuple[re.Pattern, str]]]:
    """Merge generic rules with rules from active packs."""
    out: dict[str, list[tuple[re.Pattern, str]]] = {
        k: list(v) for k, v in _GENERIC_RULES.items()
    }
    pack_map = {
        "coder": _CODER_RULES,
        "writing": _WRITING_RULES,
        "analyst": _ANALYST_RULES,
    }
    for pack_name in packs:
        pack_rules = pack_map.get(pack_name, {})
        for cat, rules in pack_rules.items():
            out.setdefault(cat, []).extend(rules)
    return out


def _check_high_risk(text: str) -> Optional[tuple[str, list[str]]]:
    """Return (reason_code, signals) on a high-risk hit, else None.

    reason_code comes from the FIRST pattern that fires (primary signal,
    deterministic by listing order). All matched signal labels accumulate."""
    first_code: Optional[str] = None
    signals: list[str] = []
    for pat, code in _HIGH_RISK_PATTERNS:
        if pat.search(text):
            if first_code is None:
                first_code = code
            signals.append(code.split(":", 1)[1])
    return (first_code, signals) if first_code else None


def _approx_tokens(text: str) -> int:
    """Rough token estimate: 4 chars per token."""
    return len(text) // 4


async def classify_capability(text: str, messages: list, config: Any) -> Recommendation:
    """Classify the request into one of 10 categories.

    Flow (spec section 4):
      1. High-risk precedence -- any hit returns high_risk immediately.
      2. Long-context check -- > 4000 tokens forces long_context.
      3. Rules scoring across active packs.
      4. (Task 3) LLM tie-breaker only if rules ambiguous + long prompt.
      5. Fallback to "default" if nothing reaches confidence threshold.
    """
    pack_list = _active_packs(config)
    pack_label = "+".join(pack_list) if pack_list else "none"

    # 1. High-risk precedence -- fires before any other check.
    hr = _check_high_risk(text)
    if hr is not None:
        code, sigs = hr
        rules = TIER_RULES["high_risk"]
        return Recommendation(
            category="high_risk",
            tier_min=rules["min"],
            tier_max=rules["max"],
            tier_prefer=rules["prefer"],
            reason_code=code,
            signals=sigs,
            confidence=0.9,
            classifier_used="high-risk-precedence",
            pack=pack_label,
        )

    # 2. Long-context bypass -- fires before rules scoring.
    if _approx_tokens(text) > 4000:
        rules = TIER_RULES["long_context"]
        return Recommendation(
            category="long_context",
            tier_min=rules["min"],
            tier_max=rules["max"],
            tier_prefer=rules["prefer"],
            reason_code="long-context:prompt-tokens>4000",
            signals=["long-prompt"],
            confidence=0.9,
            classifier_used="rules",
            pack=pack_label,
        )

    # 3. Rules scoring.
    combined = _combine_rules(pack_list)
    scores: dict[str, list[str]] = {}
    for cat, patterns in combined.items():
        for pat, label in patterns:
            if pat.search(text):
                scores.setdefault(cat, []).append(label)

    if scores:
        top_cat = max(scores.keys(), key=lambda c: (len(scores[c]), c))
        signals = scores[top_cat]
        confidence = min(0.9, 0.4 + 0.2 * len(signals))
        threshold = getattr(config.capability_routing, "confidence_threshold", 0.6)
        if confidence >= threshold:
            rules = TIER_RULES[top_cat]
            return Recommendation(
                category=top_cat,
                tier_min=rules["min"],
                tier_max=rules["max"],
                tier_prefer=rules["prefer"],
                reason_code=f"{top_cat}:rule-match",
                signals=signals,
                confidence=confidence,
                classifier_used="rules",
                pack=pack_label,
            )

    # 4. LLM tie-breaker: only when rules ambiguous + prompt long + enabled + not short-circuited above.
    use_tb = getattr(config.capability_routing, "use_llm_tiebreaker", False)
    min_chars = getattr(config, "llm_classifier_min_chars", 250)
    if use_tb and len(text) >= min_chars:
        label = await _llm_tiebreaker(text, config)
        if label is not None:
            # _llm_tiebreaker guarantees label in _VALID_LABELS and label != "default"
            rules = TIER_RULES[label]
            return Recommendation(
                category=label, tier_min=rules["min"], tier_max=rules["max"],
                tier_prefer=rules["prefer"], reason_code=f"{label}:llm-tiebreaker",
                signals=["llm-classified"], confidence=0.65,
                classifier_used="rules+llm", pack=pack_label,
            )

    # 5. Fallback default.
    rules = TIER_RULES["default"]
    return Recommendation(
        category="default",
        tier_min=rules["min"],
        tier_max=rules["max"],
        tier_prefer=rules["prefer"],
        reason_code="default:no-rule-fired",
        signals=[],
        confidence=0.0,
        classifier_used="default",
        pack=pack_label,
    )


# --- recommend_tier / shadow_columns (Task 4) ---------------------------------

_TIER_RANK = {"S": 0, "M": 1, "L": 2, "XL": 3}


def _apply_tier_rules(tier_router_choice: str, rec: Recommendation) -> str:
    """Derive cap_recommended_tier from the tier router's choice + this
    category's min/max/prefer rules (spec section 3 worked examples)."""
    # Defensive: if caller passes a tier we don't recognize, degrade gracefully
    # to "no override" rather than silently raising/capping incorrectly.
    if tier_router_choice not in _TIER_RANK:
        return tier_router_choice
    cand = tier_router_choice
    # 1. Raise to min (if any)
    if rec.tier_min and _TIER_RANK.get(cand, 0) < _TIER_RANK[rec.tier_min]:
        cand = rec.tier_min
    # 2. Cap at max (if any)
    if rec.tier_max and _TIER_RANK.get(cand, 3) > _TIER_RANK[rec.tier_max]:
        cand = rec.tier_max
    # 3. Use prefer only when neither min nor max changed the value
    if cand == tier_router_choice and rec.tier_prefer:
        cand = rec.tier_prefer
    return cand


def recommend_tier(rec: Recommendation, tier_router_choice: str, config: Any) -> Recommendation:
    """Pure pass-through in v0.2. Kept so the public surface stays stable for
    advisory/active modes in v0.3+, where this function will mutate `rec` to
    actually steer routing. The derivation lives in `shadow_columns` for now."""
    return rec


def shadow_columns(rec: Recommendation, tier_router_choice: str) -> dict[str, Any]:
    """Build the 8 additive columns persisted to the `decisions` table."""
    cap_tier = _apply_tier_rules(tier_router_choice, rec)
    return {
        "cap_category": rec.category,
        "cap_recommended_tier": cap_tier,
        "cap_reason_code": rec.reason_code,
        "cap_signals": ",".join(rec.signals),
        "cap_confidence": rec.confidence,
        "cap_classifier_used": rec.classifier_used,
        "cap_pack": rec.pack,
        "cap_agrees_with_tier": 1 if cap_tier == tier_router_choice else 0,
    }
