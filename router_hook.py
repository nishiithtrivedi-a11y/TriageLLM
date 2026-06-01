"""
Tier-routing pre/post-call hook for LiteLLM.

PRE-CALL : classify request → rewrite model to local-{s,m,l,xl}.
POST-CALL: critique non-streaming responses; on weak score walk the
           escalation chain (S→M→L→XL) until critic passes or XL is
           exhausted. If XL fails and cloud_escalation is configured,
           call the cloud model once. If everything fails, return a
           structured handoff ledger as the assistant message.
STREAM   : same critique + handoff logic, but the handoff (if any) is
           appended as one final chunk after the upstream stream ends.
           No mid-stream re-routing.

Caller sends model="local-auto". Explicit local-{s,m,l,xl} pins pass through
without classification/routing, but still get the DEF-004 fast-fail preflight (#3).

Behavioural details:
  - Short prompts (< llm_classifier_min_chars): rule-based classifier only.
  - Longer prompts: rules + LLM classifier with a "rules-floor" safety so
    the small classifier can only escalate, never downgrade.
  - Tier S is never critiqued (latency cost > benefit on trivial prompts).
  - All settings live in the `route_llm:` block of config.yaml.
"""

import asyncio
import copy
import json
import os
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Literal, Optional

import httpx
import yaml
from litellm.integrations.custom_logger import CustomLogger

# Optional: used to fast-fail a request with a clean 503 from the pre-call hook
# (DEF-004). Available wherever the LiteLLM proxy runs; absent in bare unit
# imports, in which case we fall back to a plain RuntimeError.
try:
    from fastapi import HTTPException as _HTTPException
except Exception:
    _HTTPException = None

Tier = Literal["S", "M", "L", "XL"]

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
# M-6 (SEC-003): refuse a non-localhost Ollama target unless explicitly allowed.
# Otherwise a user-level process that set OLLAMA_BASE_URL could redirect every
# prompt (full user content) + every critic call to an attacker host. Local-only
# by default; opt out with ALLOW_REMOTE_OLLAMA=1 for legitimate remote setups.
if not os.environ.get("ALLOW_REMOTE_OLLAMA"):
    if not OLLAMA_BASE.startswith(("http://localhost", "http://127.0.0.1")):
        print(f"[router] WARNING: ignoring non-localhost OLLAMA_BASE_URL={OLLAMA_BASE!r} "
              f"(set ALLOW_REMOTE_OLLAMA=1 to allow). Using http://localhost:11434.")
        OLLAMA_BASE = "http://localhost:11434"
CONFIG_PATH = Path(__file__).parent / "config.yaml"
DB_PATH = Path(__file__).parent / "router_decisions.sqlite"


def _redact(text: Any) -> str:
    """Strip anything that looks like an API key before logging (M-5 / SEC-002).
    Some provider errors echo the request — including the auth header — in their
    message; this prevents a real key from landing in proxy.log."""
    return re.sub(r"sk-[A-Za-z0-9_\-]{6,}", "sk-***REDACTED***", str(text))

# Aliases declared in config.yaml — used to set data["model"] in pre-call.
TIER_TO_ALIAS: dict[Tier, str] = {
    "S":  "local-s",
    "M":  "local-m",
    "L":  "local-l",
    "XL": "local-xl",
}

# Reverse map for #3: identify an explicit local-{s,m,l,xl} pin. Derived from
# TIER_TO_ALIAS so pin-detection follows automatically if an alias ever changes.
_ALIAS_TO_TIER: dict[str, Tier] = {alias: tier for tier, alias in TIER_TO_ALIAS.items()}

# Underlying model names for direct litellm.acompletion calls during escalation
# (acompletion doesn't know about proxy aliases). These are the built-in
# fallbacks; the live map is derived from config.yaml at import (see below).
_TIER_TO_MODEL_DEFAULTS: dict[Tier, str] = {
    "S":  "ollama_chat/qwen2.5-coder:1.5b",
    "M":  "ollama_chat/deepseek-coder-v2:16b",
    "L":  "ollama_chat/qwen3-coder:30b",
    "XL": "ollama_chat/qwen3.6:35b",
}


def _load_tier_models(path: Path = CONFIG_PATH) -> dict:
    """M-2: derive tier->model from config.yaml's model_list so escalation uses
    the SAME model the user configured for each alias. Previously this was a
    second hardcoded copy that silently drifted when a user edited config.yaml
    (first-tier routing used the new model, escalation kept the old one). Falls
    back to the built-in default for any alias not present in config."""
    models = dict(_TIER_TO_MODEL_DEFAULTS)
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        by_alias = {}
        for entry in raw.get("model_list") or []:
            name = entry.get("model_name")
            model = (entry.get("litellm_params") or {}).get("model")
            if name and model:
                by_alias[name] = model
        for tier, alias in TIER_TO_ALIAS.items():
            if alias in by_alias:
                models[tier] = by_alias[alias]
    except Exception as e:
        print(f"[router] could not derive tier models from config ({e}); using defaults")
    return models


TIER_TO_MODEL: dict[Tier, str] = _load_tier_models()

NEXT_TIER: dict[Tier, Optional[Tier]] = {"S": "M", "M": "L", "L": "XL", "XL": None}


# ─── Config loading ───────────────────────────────────────────────────────

@dataclass
class CloudEscalationConfig:
    enabled: bool = False
    model: str = "anthropic/claude-sonnet-4-6"
    api_key_env: str = "ANTHROPIC_API_KEY"
    timeout_s: float = 60.0


@dataclass
class CapabilityPacksConfig:
    coder: bool = True
    writing: bool = False
    analyst: bool = False


@dataclass
class CapabilityRoutingConfig:
    enabled: bool = False                # OPT-IN; default off
    mode: str = "shadow"
    use_llm_tiebreaker: bool = True
    confidence_threshold: float = 0.6
    packs: CapabilityPacksConfig = field(default_factory=CapabilityPacksConfig)


@dataclass
class RouterConfig:
    use_llm_classifier: bool = True
    llm_classifier_min_chars: int = 250
    llm_classifier_timeout_s: float = 3.0
    classifier_model: str = "qwen2.5:0.5b"
    critic_model: str = "qwen2.5:0.5b"
    critic_timeout_s: float = 30.0          # raised from 8s — see JOURNEY.md (eviction-cascade fix)
    critic_pass_threshold: int = 4
    critic_cpu_only: bool = True            # CPU-pin critic; the cascade fix
    soft_pass_tiers: tuple = ("S", "M")     # tier-aware soft-pass — ship without escalation on critic failure
    warmup_on_startup: bool = True          # pre-load critic so first request isn't cold
    cloud_escalation: CloudEscalationConfig = field(default_factory=CloudEscalationConfig)
    capability_routing: CapabilityRoutingConfig = field(default_factory=CapabilityRoutingConfig)


def load_config(path: Path = CONFIG_PATH) -> RouterConfig:
    """Read the route_llm block from config.yaml. Missing keys fall back to defaults."""
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return RouterConfig()

    block = raw.get("route_llm") or {}
    cloud_block = block.get("cloud_escalation") or {}
    # YAML lists round-trip as lists; we store as tuple for hashability.
    soft_pass = block.get("soft_pass_tiers")
    soft_pass_t = tuple(soft_pass) if soft_pass is not None else RouterConfig.soft_pass_tiers
    cap_block = block.get("capability_routing") or {}
    packs_block = cap_block.get("packs") or {}
    env_enabled = os.environ.get("TRIAGELLM_CAPABILITY_ROUTING_ENABLED")
    if env_enabled is not None:
        cap_enabled = env_enabled.strip().lower() in ("1", "true", "yes", "on")
    else:
        cap_enabled = cap_block.get("enabled", CapabilityRoutingConfig.enabled)

    # Mode: env override beats config; validate to the known set.
    env_mode = os.environ.get("TRIAGELLM_CAPABILITY_MODE")
    raw_mode = env_mode if env_mode is not None else cap_block.get("mode", CapabilityRoutingConfig.mode)
    cap_mode = str(raw_mode).strip().lower()
    if cap_mode not in ("shadow", "advisory"):
        print("[capability] unknown mode '" + str(raw_mode)
              + "'; falling back to 'shadow'.")
        cap_mode = "shadow"
    # Advisory implies enabled -- you cannot surface what you did not classify.
    if cap_mode == "advisory":
        cap_enabled = True

    capability_routing = CapabilityRoutingConfig(
        enabled=cap_enabled,
        mode=cap_mode,
        use_llm_tiebreaker=cap_block.get("use_llm_tiebreaker", CapabilityRoutingConfig.use_llm_tiebreaker),
        confidence_threshold=cap_block.get("confidence_threshold", CapabilityRoutingConfig.confidence_threshold),
        packs=CapabilityPacksConfig(
            coder=packs_block.get("coder", CapabilityPacksConfig.coder),
            writing=packs_block.get("writing", CapabilityPacksConfig.writing),
            analyst=packs_block.get("analyst", CapabilityPacksConfig.analyst),
        ),
    )
    return RouterConfig(
        use_llm_classifier=block.get("use_llm_classifier", RouterConfig.use_llm_classifier),
        llm_classifier_min_chars=block.get("llm_classifier_min_chars", RouterConfig.llm_classifier_min_chars),
        llm_classifier_timeout_s=block.get("llm_classifier_timeout_s", RouterConfig.llm_classifier_timeout_s),
        classifier_model=block.get("classifier_model", RouterConfig.classifier_model),
        critic_model=block.get("critic_model", RouterConfig.critic_model),
        critic_timeout_s=block.get("critic_timeout_s", RouterConfig.critic_timeout_s),
        critic_pass_threshold=block.get("critic_pass_threshold", RouterConfig.critic_pass_threshold),
        critic_cpu_only=block.get("critic_cpu_only", RouterConfig.critic_cpu_only),
        soft_pass_tiers=soft_pass_t,
        warmup_on_startup=block.get("warmup_on_startup", RouterConfig.warmup_on_startup),
        cloud_escalation=CloudEscalationConfig(
            enabled=cloud_block.get("enabled", CloudEscalationConfig.enabled),
            model=cloud_block.get("model", CloudEscalationConfig.model),
            api_key_env=cloud_block.get("api_key_env", CloudEscalationConfig.api_key_env),
            timeout_s=cloud_block.get("timeout_s", CloudEscalationConfig.timeout_s),
        ),
        capability_routing=capability_routing,
    )


# ─── Ollama fast-fail circuit breaker (DEF-004) ───────────────────────────
# When Ollama is unreachable, a normal worker call hangs ~32s (LiteLLM connect
# timeout + retry). This breaker does a lightweight preflight (GET /api/tags)
# so a down backend fails in ~probe_timeout, keeps the proxy alive, and recovers
# automatically once Ollama returns. Env-configurable, matching the existing
# env style (TRIAGELLM_SKIP_WARMUP / ALLOW_REMOTE_OLLAMA). All strings ASCII.

def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return default


class OllamaCircuitBreaker:
    """Minimal CLOSED/OPEN/HALF_OPEN breaker around Ollama reachability.

    preflight() returns (allowed: bool, reason: str) and NEVER raises — the
    caller decides how to fail (the pre-call hook raises a 503; _call_tier
    raises so the C-5 ledger path records it). Reason strings are ASCII so they
    can't reintroduce the cp1252 console crash (DEF-003).
    """

    CLOSED, OPEN, HALF_OPEN = "CLOSED", "OPEN", "HALF_OPEN"

    def __init__(self, base_url: str, enabled: bool = True,
                 connect_timeout: float = 2.0, cooldown: float = 10.0,
                 probe_timeout: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.enabled = enabled
        self.connect_timeout = connect_timeout
        self.cooldown = cooldown
        self.probe_timeout = probe_timeout
        self.state = self.CLOSED
        self.opened_at = 0.0
        self.last_error: Optional[str] = None
        self.last_success = 0.0
        self.open_count = 0

    async def _probe(self) -> bool:
        """Lightweight liveness check. Never loads a model."""
        timeout = httpx.Timeout(connect=self.connect_timeout, read=self.probe_timeout,
                                write=self.probe_timeout, pool=self.probe_timeout)
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(f"{self.base_url}/api/tags")
        return r.status_code == 200

    async def preflight(self, now: Optional[float] = None) -> tuple[bool, str]:
        if not self.enabled:
            return True, "fast-fail disabled"
        now = time.time() if now is None else now
        # OPEN: fail instantly during cooldown (no probe at all).
        if self.state == self.OPEN:
            if now - self.opened_at < self.cooldown:
                remaining = round(self.cooldown - (now - self.opened_at), 1)
                return False, (f"circuit OPEN: Ollama unreachable ({self.last_error}); "
                               f"fast-fail, cooldown {remaining}s left")
            self.state = self.HALF_OPEN
        # CLOSED or HALF_OPEN: probe.
        try:
            ok = await self._probe()
            err = None if ok else "non-200 from /api/tags"
        except Exception as e:
            ok, err = False, type(e).__name__
        if ok:
            recovered = self.state == self.HALF_OPEN
            self.state = self.CLOSED
            self.last_success = now
            self.last_error = None
            return True, ("recovered: Ollama probe ok" if recovered else "ok")
        self.state = self.OPEN
        self.opened_at = now
        self.open_count += 1
        self.last_error = err
        return False, f"circuit OPEN: Ollama unreachable ({err}); fast-fail"

    def snapshot(self) -> dict:
        return {"state": self.state, "open_count": self.open_count,
                "last_error": self.last_error}


_FAST_FAIL_ENABLED = _env_bool("TRIAGELLM_OLLAMA_FAST_FAIL_ENABLED", True)
_ollama_circuit = OllamaCircuitBreaker(
    base_url=OLLAMA_BASE,
    enabled=_FAST_FAIL_ENABLED,
    connect_timeout=_env_float("TRIAGELLM_OLLAMA_CONNECT_TIMEOUT_SECONDS", 2.0),
    cooldown=_env_float("TRIAGELLM_OLLAMA_CIRCUIT_COOLDOWN_SECONDS", 10.0),
    probe_timeout=_env_float("TRIAGELLM_OLLAMA_PROBE_TIMEOUT_SECONDS", 2.0),
)


# ─── Rule-based classifier ────────────────────────────────────────────────

KEYWORDS_XL = [
    r"\barchitect", r"\bdesign\s+(doc|review)",
    r"\bsecurity\b", r"\bvulnerab", r"\bauth(entication|orization)\b",
    r"\bconcurren", r"\brace\s+condition", r"\bdeadlock",
    r"\bmigration\b", r"\bschema\s+change", r"\bbackwards?\s+compatib",
    r"\bdistributed\b", r"\bconsisten(cy|t)\b",
]
KEYWORDS_L = [
    r"\brefactor", r"\bdebug", r"\bperformance\b", r"\boptim(ize|isation)",
    r"\bmulti-?file\b", r"\bacross\s+\w+\s+files",
    r"\bmemory\s+leak", r"\bprofile\b", r"\bbenchmark",
    r"\btraceback", r"\bstack\s*trace", r"\bexception\b",
]
KEYWORDS_M = [
    r"\bimplement\b", r"\bwrite\s+a\s+(function|class|module|test)",
    r"\badd\s+a\s+(feature|endpoint|method)", r"\btest(s|ing)?\b",
    r"\bfix\s+", r"\bbug\b",
]
KEYWORDS_S = [
    r"\brename\b", r"\btypo\b", r"\bformat\b", r"\bone[- ]liner",
    r"\bautocomplete", r"\bcomplete\s+this\b", r"\bregex\b",
    r"\bwhat\s+is\b", r"\bsyntax\s+for\b",
]

CODEBLOCK_RE = re.compile(r"```")
FILEPATH_RE  = re.compile(
    r"[\w./\\-]+\.(py|ts|tsx|js|jsx|go|rs|java|cpp|c|h|hpp|cs|rb|php|swift|kt|sql|yaml|yml|toml|json)\b"
)
TRACEBACK_RE = re.compile(
    r"(Traceback \(most recent call last\)|^\s*at\s+\w+.*:\d+|panic:|Error:.*\n\s*at )",
    re.MULTILINE,
)


def _extract_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in messages or []:
        content = m.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    parts.append(blk.get("text", ""))
    return "\n".join(parts)


def classify_rules(text: str) -> tuple[Tier, int, list[str]]:
    text_lc = text.lower()
    approx_tokens = len(text) // 4
    score = 0
    signals: list[str] = []

    if approx_tokens > 8000:   score += 4; signals.append("tokens>8000")
    elif approx_tokens > 3000: score += 3; signals.append("tokens>3000")
    elif approx_tokens > 1000: score += 2; signals.append("tokens>1000")
    elif approx_tokens > 300:  score += 1; signals.append("tokens>300")

    codeblocks = len(CODEBLOCK_RE.findall(text)) // 2
    filepaths  = len(FILEPATH_RE.findall(text))
    if codeblocks >= 3: score += 2; signals.append(f"codeblocks={codeblocks}")
    elif codeblocks:    score += 1; signals.append(f"codeblocks={codeblocks}")
    if filepaths >= 3:  score += 2; signals.append(f"filepaths={filepaths}")
    elif filepaths:     score += 1; signals.append(f"filepaths={filepaths}")
    if TRACEBACK_RE.search(text):
        score += 2; signals.append("traceback")

    def hits(patterns: list[str]) -> int:
        return sum(1 for p in patterns if re.search(p, text_lc))

    xl_hits, l_hits = hits(KEYWORDS_XL), hits(KEYWORDS_L)
    m_hits,  s_hits = hits(KEYWORDS_M),  hits(KEYWORDS_S)
    if xl_hits: score += 3 + 2 * xl_hits; signals.append(f"xl_kw={xl_hits}")
    if l_hits:  score += 2 + l_hits;      signals.append(f"l_kw={l_hits}")
    if m_hits:  score += m_hits;          signals.append(f"m_kw={m_hits}")
    if s_hits:  score -= 2 * s_hits;      signals.append(f"s_kw={s_hits}")

    if   score >= 7: tier: Tier = "XL"
    elif score >= 4: tier        = "L"
    elif score >= 1: tier        = "M"
    else:            tier        = "S"
    return tier, score, signals


def classify(messages: list[dict[str, Any]]) -> tuple[Tier, int, list[str]]:
    """Back-compat shim for test_classifier.py."""
    return classify_rules(_extract_text(messages))


# ─── LLM classifier ───────────────────────────────────────────────────────

_CLASSIFIER_PROMPT = """You are a tier classifier for a coding assistant. Given a user request, output exactly one of: S, M, L, or XL.

S  = trivial (rename, format, syntax question, one-liner)
M  = normal coding (implement a function, write a test, fix a small bug)
L  = complex (multi-file refactor, debug a stack trace, performance work)
XL = architecture (security, concurrency, distributed systems, schema migration)

Reply with ONLY the letters. No explanation.

REQUEST:
{text}

TIER:"""


async def classify_llm(
    text: str,
    model: str = "qwen2.5:0.5b",
    timeout_s: float = 3.0,
    cpu_only: bool = True,
) -> Optional[Tier]:
    """Classify tier via a small LLM.

    cpu_only=True pins inference to CPU (num_gpu=0). On low-VRAM machines this
    prevents the classifier from being evicted whenever a big worker model
    loads — see docs/JOURNEY.md for the cascade story.
    """
    options = {"num_predict": 4, "temperature": 0.0}
    if cpu_only:
        options["num_gpu"] = 0
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as c:
            r = await c.post(
                f"{OLLAMA_BASE}/api/generate",
                json={
                    "model": model,
                    "prompt": _CLASSIFIER_PROMPT.format(text=text[:2000]),
                    "stream": False,
                    "options": options,
                    "keep_alive": -1,   # never unload — see JOURNEY.md §"the eviction cascade"
                },
            )
        out = (r.json().get("response") or "").strip().upper()
        for t in ("XL", "L", "M", "S"):       # XL first so "L" doesn't substring-match
            if t in out:
                return t  # type: ignore[return-value]
        # H-3: ran fine but emitted no tier letter — distinct from a crash.
        print(f"[classifier-llm] ran but no tier letter in response: {out!r}")
    except httpx.TimeoutException:
        print(f"[classifier-llm] TIMEOUT after {timeout_s}s (model cold/evicted?)")
    except httpx.ConnectError:
        print(f"[classifier-llm] CONNECT FAILED (Ollama unreachable at {OLLAMA_BASE})")
    except Exception as e:
        print(f"[classifier-llm] ERROR {type(e).__name__}: {e}")
    return None


# ─── Critic ───────────────────────────────────────────────────────────────

_CRITIC_PROMPT = """Rate this assistant's answer for the user's request on a 1-5 scale.
1=poor (wrong/incomplete/off-topic), 5=excellent (correct, complete, addresses the actual ask).
Reply with ONLY a single digit 1-5. No explanation.

REQUEST:
{task}

ANSWER:
{answer}

SCORE:"""


async def critic_score(
    task: str,
    answer: str,
    model: str = "qwen2.5:0.5b",
    timeout_s: float = 30.0,
    cpu_only: bool = True,
) -> Optional[int]:
    """Score an assistant answer 1-5.

    cpu_only=True pins inference to CPU (num_gpu=0). This is the central fix
    for the eviction cascade on low-VRAM machines: the critic lives in regular
    RAM, untouched by whatever worker model is hogging the GPU. See
    docs/JOURNEY.md for the full story.
    """
    options = {"num_predict": 3, "temperature": 0.0}
    if cpu_only:
        options["num_gpu"] = 0
    # M-7 (SEC-004): neutralize the critic's stop-token inside the worker's
    # answer. A malicious/confused worker could emit "SCORE: 5" to force a pass
    # and bypass escalation. Defang any "SCORE:" the answer tries to inject.
    safe_answer = re.sub(r"(?i)score\s*:", "score_", answer[:2500])
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as c:
            r = await c.post(
                f"{OLLAMA_BASE}/api/generate",
                json={
                    "model": model,
                    "prompt": _CRITIC_PROMPT.format(task=task[:1500], answer=safe_answer),
                    "stream": False,
                    "options": options,
                    "keep_alive": -1,   # never unload — keeps critic warm for the whole proxy session
                },
            )
        out = (r.json().get("response") or "").strip()
        for ch in out:
            if ch in "12345":
                return int(ch)
        # H-2: ran fine but produced no usable score — distinct from a transport
        # failure. Previously this fell through to a silent None.
        print(f"[critic] ran but no 1-5 digit in response: {out!r}")
    except httpx.TimeoutException:
        # H-2: timeout is the eviction-cascade signature — name it explicitly.
        print(f"[critic] TIMEOUT after {timeout_s}s (critic model cold/evicted?)")
    except httpx.ConnectError:
        print(f"[critic] CONNECT FAILED (Ollama unreachable at {OLLAMA_BASE})")
    except Exception as e:
        print(f"[critic] ERROR {type(e).__name__}: {e}")
    return None


# ─── Attempt ledger ───────────────────────────────────────────────────────

@dataclass
class Attempt:
    tier: str                          # "S" | "M" | "L" | "XL" | "CLOUD"
    model: str                         # ollama_chat/... or anthropic/...
    prompt_tokens: int
    completion_tokens: int
    duration_s: float
    critic_score: Optional[int]        # None if S (not critiqued) or critic failed
    preview: str                       # first 200 chars of the answer
    # Issue #29: cost tracking. Defaulted -> backward-compatible with the
    # existing construction sites and with old attempts_json rows (which
    # lack these keys). See the cost-tracking change.
    was_warm: bool = False             # model warm at call time (process-set heuristic)
    vram_mb: Optional[int] = None      # VRAM footprint (MB); gated fetch, Task 2
    cost_usd: Optional[float] = None   # cloud spend via LiteLLM; None for local


def _extract_usage(response: Any) -> tuple[int, int]:
    """Return (prompt_tokens, completion_tokens) from a LiteLLM response."""
    try:
        u = response.usage
        return int(getattr(u, "prompt_tokens", 0) or 0), int(getattr(u, "completion_tokens", 0) or 0)
    except (AttributeError, TypeError, ValueError) as e:
        # H-6: narrow the catch + log. A bare `except: return 0,0` would hide a
        # LiteLLM response-shape change (e.g. usage moving) behind silent zeros,
        # making token accounting wrong with no signal.
        print(f"[extract] usage extraction failed ({type(e).__name__}); "
              f"response={repr(response)[:200]}")
        return 0, 0


# Issue #29: which models we've already called this process. First sight of a
# model => cold load (was_warm=False); afterwards => warm. Heuristic only --
# does NOT account for Ollama evicting a model after keep_alive expires.
_warm_models: set[str] = set()


def _mark_and_check_warm(model: str) -> bool:
    """Return True if this model was already called this process (warm),
    False on first sight (cold). Records the model either way."""
    warm = model in _warm_models
    _warm_models.add(model)
    return warm


_COLD_LOAD_WARN_S = 30.0  # a successful attempt slower than this logs a cold-load hint

# Ollama's 404 body names the real model: model "qwen2.5-coder:1.5b" not found.
_MODEL_IN_ERR = re.compile(r"model\s+['\"]?([\w.:\-/]+)['\"]?\s+not found", re.I)


def _is_model_not_found(exc: Any) -> bool:
    """True if exc looks like an Ollama 'model not pulled' (404) error."""
    if getattr(exc, "status_code", None) == 404:
        return True
    msg = str(exc).lower()
    if "try pulling" in msg:
        return True
    return "not found" in msg and ("model" in msg or "pull" in msg)


def _model_name_from_error(exc: Any, fallback: str) -> str:
    """Pull the real model name out of the Ollama error body; else fallback."""
    m = _MODEL_IN_ERR.search(str(exc))
    return m.group(1) if m else fallback


def _model_not_found_hint(model: str) -> str:
    """ASCII actionable hint. Strips any provider prefix for the bare pull name."""
    bare = (model or "").split("/", 1)[-1]
    return ("Model " + bare + " is not installed on this Ollama daemon. "
            "Run: ollama pull " + bare
            + " (or rebuild it if it is a custom Modelfile build).")


def _maybe_warn_cold_load(model: str, duration_s: float) -> None:
    """Log a cold-load hint when a SUCCESSFUL attempt took a long time. Log-only."""
    if duration_s >= _COLD_LOAD_WARN_S:
        bare = (model or "").split("/", 1)[-1]
        print("[router] Model " + bare + " took " + str(round(duration_s, 1))
              + "s (cold load?); it stays warm via keep_alive after the first call.")


def _safe_cost_usd(response: Any) -> Optional[float]:
    """Best-effort cloud cost via LiteLLM. None for local models (no pricing)
    or on any failure -- never raises into the request path."""
    try:
        import litellm
        cost = litellm.completion_cost(completion_response=response)
        return float(cost) if cost else None
    except Exception:
        return None


async def _fetch_vram_mb(model: str) -> Optional[int]:
    """Best-effort VRAM footprint (MB) for a loaded model via Ollama /api/ps.
    Strips the 'ollama_chat/' (or similar) provider prefix to match Ollama's
    bare model name. None on any failure -- never raises into the request path."""
    bare = model.split("/", 1)[-1]
    try:
        timeout = httpx.Timeout(connect=2.0, read=2.0, write=2.0, pool=2.0)
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(f"{OLLAMA_BASE.rstrip('/')}/api/ps")
        if r.status_code != 200:
            return None
        for m in r.json().get("models", []):
            if m.get("name") == bare and m.get("size_vram"):
                return int(m["size_vram"]) // (1024 * 1024)
        return None
    except Exception:
        return None


# Issue #29: VRAM is captured fire-and-forget so it NEVER blocks the response
# path (capability routing must not change behaviour, only observe). First
# request for a given model returns None and kicks off a one-time background
# fetch; subsequent requests read the cached value. Fetched once per model
# per process, not per request.
_vram_cache: dict[str, Optional[int]] = {}
_vram_inflight: set[str] = set()      # models currently being fetched (dedupe)
_vram_bg_tasks: set = set()           # strong refs so tasks aren't GC'd


def _vram_mb_nonblocking(model: str) -> Optional[int]:
    """Return cached VRAM (MB) for the model, scheduling a one-time background
    fetch on first sight. Never blocks the request path. Returns None until the
    background fetch completes (so the first request per model logs None)."""
    if model not in _vram_cache and model not in _vram_inflight:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None  # no running loop -> skip silently (e.g. sync import)
        _vram_inflight.add(model)
        task = loop.create_task(_populate_vram_cache(model))
        _vram_bg_tasks.add(task)
        task.add_done_callback(_vram_bg_tasks.discard)
    return _vram_cache.get(model)


async def _populate_vram_cache(model: str) -> None:
    try:
        _vram_cache[model] = await _fetch_vram_mb(model)
    finally:
        _vram_inflight.discard(model)


def _extract_answer(response: Any) -> str:
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError, KeyError) as e:
        # H-6: narrow the catch + log. A bare `except: return ""` would turn a
        # response-shape change into silent empty answers — the critic would
        # then score "" and soft-pass, shipping nothing, with no alarm.
        print(f"[extract] answer extraction failed ({type(e).__name__}); "
              f"response={repr(response)[:200]}")
        return ""


def render_handoff(
    attempts: list[Attempt],
    threshold: int,
    cloud_attempted: bool,
    best_answer: str,
) -> str:
    """Render the ledger as a user-facing assistant message."""
    lines = [
        "═══════════════════════════════════════════════════════════════════",
        "🔄  LOCAL STACK EXHAUSTED — handoff to cloud agent",
        "═══════════════════════════════════════════════════════════════════",
        "",
        "Attempted tiers (in order):",
    ]
    total_prompt = total_completion = 0
    total_time = 0.0
    best_idx = -1
    best_score = -1
    # M-1: build the attempt bullets in their own list and decorate the best
    # one in place. The previous code used `idx_line = 5 + best_idx`, a magic
    # offset that silently corrupted the marker if the header ever changed.
    attempt_lines: list[str] = []
    for i, a in enumerate(attempts):
        score_str = f"{a.critic_score}/5" if a.critic_score is not None else " — "
        if a.critic_score is not None and a.critic_score > best_score:
            best_score = a.critic_score
            best_idx = i
        attempt_lines.append(
            f"  • Tier {a.tier:<5} {a.model:<42} "
            f"| {a.prompt_tokens + a.completion_tokens:>5} tok "
            f"| {a.duration_s:>5.1f}s "
            f"| critic: {score_str}"
        )
        total_prompt += a.prompt_tokens
        total_completion += a.completion_tokens
        total_time += a.duration_s
    if 0 <= best_idx < len(attempt_lines):
        attempt_lines[best_idx] += "   ◄ best"
    lines.extend(attempt_lines)

    lines.append("")
    lines.append(
        f"Totals: {total_prompt + total_completion} tokens "
        f"({total_prompt} prompt + {total_completion} completion), "
        f"{total_time:.1f}s wall-clock, $0.00 local cost."
    )
    lines.append(f"Critic pass threshold: {threshold}/5. Best local score: "
                 f"{best_score}/5." if best_score >= 0 else
                 f"Critic pass threshold: {threshold}/5.")
    if cloud_attempted:
        lines.append("Cloud escalation was attempted and also did not pass.")
    else:
        lines.append("Cloud escalation: disabled or not configured.")
    lines.append("")
    lines.append("Recommended: rerun this prompt with your cloud-backed agent")
    lines.append("(Claude Code / ChatGPT / Codex). Best local draft below.")
    lines.append("───────────────────────────────────────────────────────────────────")
    lines.append("")
    lines.append(best_answer.rstrip() or "(no usable draft produced)")
    return "\n".join(lines)


# ─── SQLite logging ───────────────────────────────────────────────────────

# Source of truth for the decisions table schema.
#
# Adding a column: append a tuple (name, sqlite_type, in_create_table) here
# and add a corresponding positional value to the `row` tuple in
# _log_decision below. The CREATE TABLE statement, the ALTER TABLE
# migration loop, the post-migration assert, and the INSERT column list
# all derive from this constant. The `row` tuple in _log_decision is the
# only hand-written piece that still needs a manual edit -- guarded by
# test_log_decision_row_tuple_matches_decisions_columns.
#
# Order MUST match the row tuple in _log_decision. If you must reorder,
# update both in the same commit.
_DECISIONS_COLUMNS = (
    # (name, sqlite_type, in_original_create_table)
    ("ts",                   "REAL",    True),
    ("requested",            "TEXT",    True),
    ("tier",                 "TEXT",    True),
    ("model",                "TEXT",    True),
    ("tokens",               "INTEGER", True),
    ("score",                "INTEGER", True),
    ("signals",              "TEXT",    True),
    # Added via ALTER TABLE migrations (escalation ledger + flags):
    ("classifier",           "TEXT",    False),
    ("critic",               "INTEGER", False),
    ("escalated_to",         "TEXT",    False),
    ("attempts_json",        "TEXT",    False),
    ("cloud_attempted",      "INTEGER", False),
    ("handoff",              "INTEGER", False),
    ("streamed",             "INTEGER", False),
    # Capability Routing v0.2 (shadow mode) -- spec S6
    ("cap_category",         "TEXT",    False),
    ("cap_recommended_tier", "TEXT",    False),
    ("cap_reason_code",      "TEXT",    False),
    ("cap_signals",          "TEXT",    False),
    ("cap_confidence",       "REAL",    False),
    ("cap_classifier_used",  "TEXT",    False),
    ("cap_pack",             "TEXT",    False),
    ("cap_agrees_with_tier", "INTEGER", False),
)


def _init_db() -> None:
    with sqlite3.connect(DB_PATH) as c:
        # CREATE TABLE with the original columns (the ones present in the
        # initial v0.0 schema). Existing DBs are no-op'd by IF NOT EXISTS;
        # new DBs get a 7-column baseline that the ALTER TABLE loop below
        # then migrates forward to the current schema.
        create_cols = [
            f"{name} {type_}"
            for name, type_, in_create in _DECISIONS_COLUMNS
            if in_create
        ]
        c.execute(
            "CREATE TABLE IF NOT EXISTS decisions ("
            + ", ".join(create_cols)
            + ")"
        )

        # ALTER TABLE migrations for additive columns. Skip-if-present
        # via the `existing` set so re-runs are idempotent.
        existing = {row[1] for row in c.execute("PRAGMA table_info(decisions)")}
        for name, type_, in_create in _DECISIONS_COLUMNS:
            if in_create:
                continue
            if name not in existing:
                c.execute(f"ALTER TABLE decisions ADD COLUMN {name} {type_}")

        # Post-migration assert: if any expected column is still missing
        # the migration silently failed and every subsequent _log_decision
        # write will hit the H-4 fallback. Raise loudly so the operator
        # sees the problem at startup, not via a flatlined dashboard.
        present = {row[1] for row in c.execute("PRAGMA table_info(decisions)")}
        expected = {name for name, _, _ in _DECISIONS_COLUMNS}
        missing = expected - present
        if missing:
            raise RuntimeError(
                "router_hook: decisions table is missing expected columns "
                f"after migration: {sorted(missing)}. Dashboard will show "
                "no data until this is repaired. Check sqlite locks, disk "
                f"space, and the schema at {DB_PATH}."
            )


# H-4: count + preserve audit rows that fail to reach SQLite. Without this, a
# locked DB / full disk / schema drift would silently drop routing decisions —
# the dashboard would flatline and an operator would wrongly conclude "traffic
# stopped" rather than "logging broke". The fallback JSONL keeps the data.
_log_write_failures = 0

# Issue #7: one-shot warning so the operator sees "dashboard is blind"
# the first time fallback fires, but the log doesn't get spammed under
# sustained fault. Under async concurrency two writes can race past the
# check and double-print, which is acceptable -- still bounded, not
# spammy. Reset to False to re-arm (e.g. in tests).
_dashboard_blind_warned = False


def _log_decision(
    state: dict[str, Any],
    attempts: list[Attempt],
    cloud_attempted: bool,
    handoff: bool,
    streamed: bool,
) -> None:
    global _log_write_failures, _dashboard_blind_warned
    final = attempts[-1] if attempts else None
    cap = state.get("capability")
    if not isinstance(cap, dict):
        cap = {}
    # all_cols derived from _DECISIONS_COLUMNS. The `row` tuple below MUST
    # have one positional value per column in the same order -- structural
    # invariant guarded by test_log_decision_row_tuple_matches_decisions_columns.
    # If you add a column to _DECISIONS_COLUMNS, add the corresponding
    # positional value to the `row` tuple too.
    all_cols = tuple(name for name, _, _ in _DECISIONS_COLUMNS)
    row = (
        time.time(),
        state["requested"],
        state["initial_tier"],
        state["alias"],
        state["tokens"],
        state["score"],
        ",".join(state["signals"]),
        state["classifier"],
        final.critic_score if final else None,
        final.model if (final and final.tier != state["initial_tier"]) else None,
        json.dumps([asdict(a) for a in attempts]),
        1 if cloud_attempted else 0,
        1 if handoff else 0,
        1 if streamed else 0,
        cap.get("cap_category"),
        cap.get("cap_recommended_tier"),
        cap.get("cap_reason_code"),
        cap.get("cap_signals"),
        cap.get("cap_confidence"),
        cap.get("cap_classifier_used"),
        cap.get("cap_pack"),
        cap.get("cap_agrees_with_tier"),
    )
    placeholders = ",".join("?" * len(all_cols))
    sql = f"INSERT INTO decisions ({','.join(all_cols)}) VALUES ({placeholders})"
    try:
        with sqlite3.connect(DB_PATH) as c:
            c.execute(sql, row)
    except Exception as e:
        _log_write_failures += 1
        if not _dashboard_blind_warned:
            print(
                "[router_hook] WARNING: DASHBOARD WILL SHOW NO DATA "
                "while sqlite writes are failing. Audit rows are spilling "
                "to the fallback JSONL. Check schema with: "
                f'sqlite3 "{DB_PATH}" \'.schema decisions\' '
                "-- this warning prints once per process."
            )
            _dashboard_blind_warned = True
        print(f"[log] CRITICAL: sqlite write failed (#{_log_write_failures}): {e}")
        # Fallback: append the row as JSON so the audit data isn't lost. The
        # dashboard can be backfilled from this file if SQLite recovers.
        # H-4: all_cols now covers all 22 columns including cap_* -- audit trail complete.
        try:
            fallback = DB_PATH.parent / (DB_PATH.stem + "_fallback.jsonl")
            with open(fallback, "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(zip(all_cols, row))) + "\n")
            print(f"[log] row preserved to {fallback.name}")
        except Exception as e2:
            print(f"[log] CRITICAL: fallback write ALSO failed: {e2}")


def _log_fast_fail(requested: str, rule_tier: Tier, tokens: int, score: int,
                   signals: list, reason: str) -> None:
    """DEF-004: write an observable ledger row when a request is fast-failed
    because Ollama is unreachable. classifier='ollama-down-fastfail' + handoff=1
    make the fast-fail explicit in the dashboard and evidence."""
    state = {
        "requested": requested,
        "initial_tier": rule_tier,
        "alias": TIER_TO_ALIAS.get(rule_tier, "local-s"),
        "tokens": tokens,
        "score": score,
        "signals": signals,
        "classifier": "ollama-down-fastfail",
    }
    attempt = Attempt(
        tier=rule_tier, model=TIER_TO_MODEL.get(rule_tier, "unknown"),
        prompt_tokens=0, completion_tokens=0, duration_s=0.0,
        critic_score=None, preview=("FAST-FAIL: " + reason)[:200],
    )
    _log_decision(state, [attempt], cloud_attempted=False, handoff=True, streamed=False)


# ─── Advisory surfacing helpers (#18) ────────────────────────────────────

def _format_advisory_line(cap, actual_tier):
    """Pure: render the one-line per-request advisory log entry (ASCII)."""
    agree = "yes" if cap.get("cap_agrees_with_tier") else "no"
    conf = cap.get("cap_confidence")
    conf_s = ("%.2f" % conf) if isinstance(conf, (int, float)) else "n/a"
    return ("[advisory] cap=" + str(cap.get("cap_category"))
            + " suggested=" + str(cap.get("cap_recommended_tier"))
            + " actual=" + str(actual_tier)
            + " agree=" + agree
            + " conf=" + conf_s)


def _advisory_headers(cap, actual_tier):
    """Pure: build best-effort response headers carrying the capability rec (ASCII)."""
    return {
        "x-triagellm-cap-category": str(cap.get("cap_category")),
        "x-triagellm-cap-suggested-tier": str(cap.get("cap_recommended_tier")),
        "x-triagellm-cap-actual-tier": str(actual_tier),
        "x-triagellm-cap-agrees": "true" if cap.get("cap_agrees_with_tier") else "false",
    }


# ─── LiteLLM hook ─────────────────────────────────────────────────────────

class TierRouter(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.config = load_config()
        _init_db()
        ce = self.config.cloud_escalation
        cloud_status = (
            f"enabled (model={ce.model}, key_env={ce.api_key_env})"
            if ce.enabled else "disabled"
        )
        critic_place = "CPU" if self.config.critic_cpu_only else "GPU/auto"
        print(
            f"[router] config loaded: classifier={self.config.classifier_model} "
            f"critic={self.config.critic_model} ({critic_place}) "
            f"threshold={self.config.critic_pass_threshold}/5 "
            f"timeout={self.config.critic_timeout_s}s "
            f"soft_pass={'/'.join(self.config.soft_pass_tiers) or '-'} "
            f"cloud={cloud_status}"
        )

        # Pre-warm critic so the first real request doesn't pay cold-load latency.
        # Skip entirely if TRIAGELLM_SKIP_WARMUP is set (tests / health checks)
        # so we don't fire a blocking ~30s Ollama call from a bare import.
        if self.config.warmup_on_startup and not os.environ.get("TRIAGELLM_SKIP_WARMUP"):
            try:
                import asyncio
                try:
                    asyncio.get_running_loop()
                    fut = asyncio.ensure_future(self._warmup())
                    # M-4: observe the future so a failed warmup logs cleanly
                    # instead of surfacing as an "exception was never retrieved"
                    # warning at GC time (possibly mid-request).
                    def _warmup_done(f: "asyncio.Future") -> None:
                        try:
                            exc = f.exception()
                        except Exception:
                            return  # cancelled — nothing to report
                        if exc:
                            print(f"[router] warmup task error (non-fatal): {exc}")
                    fut.add_done_callback(_warmup_done)
                except RuntimeError:
                    # No running loop — execute synchronously now.
                    asyncio.run(self._warmup())
            except Exception as e:
                print(f"[router] warmup scheduling failed (non-fatal): {e}")

    async def _warmup(self) -> None:
        """Send one tiny critic call so the model is loaded before any real request."""
        try:
            t0 = time.time()
            await critic_score(
                "warmup", "warmup",
                model=self.config.critic_model,
                timeout_s=self.config.critic_timeout_s,
                cpu_only=self.config.critic_cpu_only,
            )
            print(f"[router] critic warmup OK ({time.time() - t0:.2f}s)")
        except Exception as e:
            print(f"[router] critic warmup failed (non-fatal): {e}")

    # PRE-CALL: pick tier, rewrite model -----------------------------------
    async def async_pre_call_hook(
        self,
        user_api_key_dict,
        cache,
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any] | None:
        requested = data.get("model", "")
        is_auto = (requested == "local-auto")
        pin_tier = _ALIAS_TO_TIER.get(requested)   # None unless an explicit local-{s,m,l,xl} pin
        if not is_auto and pin_tier is None:
            return data    # genuinely external model (gpt-4, ...) -- not TriageLLM's concern

        messages = data.get("messages") or []
        text = _extract_text(messages)
        # Tier for the fast-fail ledger row: classify for auto; the pin itself for a pin.
        if is_auto:
            rule_tier, score, signals = classify_rules(text)
        else:
            rule_tier, score, signals = pin_tier, 0, ["explicit-pin"]

        # DEF-004 fast-fail: preflight Ollama BEFORE the (possibly slow) LLM
        # classifier and before LiteLLM dispatches the worker call. If the
        # backend is down, fail in ~probe_timeout (target <5s) instead of ~32s,
        # keep the proxy alive, and recover automatically once Ollama returns.
        allowed, reason = await _ollama_circuit.preflight()
        if not allowed:
            print(f"[router] FAST-FAIL (Ollama preflight): {reason}")
            try:
                _log_fast_fail(requested, rule_tier, len(text) // 4, score, signals, reason)
            except Exception as e:
                print(f"[router] fast-fail ledger write skipped: {type(e).__name__}")
            msg = ("TriageLLM: local model backend (Ollama) is unreachable - "
                   "request fast-failed. Start Ollama and retry. [" + reason + "]")
            if _HTTPException is not None:
                raise _HTTPException(status_code=503, detail=msg)
            raise RuntimeError(msg)

        if pin_tier is not None:
            return data    # explicit pin escape-hatch: breaker ran, but NO classification/routing

        tier: Tier = rule_tier
        classifier_used = "rules"
        if self.config.use_llm_classifier and len(text) >= self.config.llm_classifier_min_chars:
            llm_tier = await classify_llm(
                text,
                model=self.config.classifier_model,
                timeout_s=self.config.llm_classifier_timeout_s,
            )
            if llm_tier:
                # Rules-floor safety: small classifier can only escalate.
                order = {"S": 0, "M": 1, "L": 2, "XL": 3}
                if order[llm_tier] > order[rule_tier]:
                    tier = llm_tier
                    classifier_used = "llm-up"
                else:
                    classifier_used = "llm-agree-or-down" if llm_tier == rule_tier else "rules-floor"

        alias = TIER_TO_ALIAS[tier]
        data["model"] = alias
        data["_router_state"] = {
            "requested": requested,
            "initial_tier": tier,
            "alias": alias,
            "tokens": len(text) // 4,
            "score": score,
            "signals": signals,
            "classifier": classifier_used,
            "rule_tier": rule_tier,
            "started_at": time.time(),
        }
        print(
            f"[router] auto -> {tier} ({alias}) "
            f"[{classifier_used}; rules said {rule_tier}; score={score}] {signals}"
        )

        # Capability Routing v0.2 - shadow mode (spec the design notes).
        # Records what tier the capability classifier WOULD have chosen.
        # Behavior unchanged when enabled=false (default) or on any exception.
        if self.config.capability_routing.enabled:
            try:
                import capability_router
                cap_rec = await capability_router.classify_capability(text, messages, self.config)
                cap_rec = capability_router.recommend_tier(cap_rec, tier_router_choice=tier, config=self.config)
                cap_cols = capability_router.shadow_columns(cap_rec, tier)
                data["_router_state"]["capability"] = cap_cols
                if self.config.capability_routing.mode == "advisory":
                    print(_format_advisory_line(cap_cols, tier))
            except Exception as e:
                print(f"[capability] non-fatal: {type(e).__name__}: {e}")

        return data

    # ─── Escalation helpers ────────────────────────────────────────────

    async def _critique(self, task_text: str, answer: str) -> Optional[int]:
        if not answer.strip():
            return None
        return await critic_score(
            task_text, answer,
            model=self.config.critic_model,
            timeout_s=self.config.critic_timeout_s,
            cpu_only=self.config.critic_cpu_only,
        )

    def _critique_outcome(self, tier: str, score: Optional[int]) -> str:
        """H-7: single source of truth for what a critic score MEANS, shared by
        the non-streaming and streaming paths so their semantics can't drift.

          "pass"      -> score >= threshold: ship as-is
          "soft_pass" -> score is None AND tier in soft_pass_tiers: ship anyway
                         (critic couldn't run; don't cascade uselessly)
          "escalate"  -> a real low score, OR None on a non-soft tier: the
                         answer isn't trusted. Non-streaming tries a bigger
                         tier; streaming (which can't re-route) emits a handoff.
        """
        threshold = self.config.critic_pass_threshold
        if score is not None and score >= threshold:
            return "pass"
        if score is None and tier in self.config.soft_pass_tiers:
            return "soft_pass"
        return "escalate"

    async def _call_tier(self, messages: list[dict[str, Any]], tier: Tier) -> tuple[Any, Attempt]:
        """Make one upstream call to the given tier and return (response, Attempt)."""
        import litellm
        # DEF-004: don't spend the full connect timeout on an escalation step if
        # Ollama is already known down — fail fast; the orchestrator records the
        # failed attempt in the ledger (C-5) rather than hanging the chain.
        allowed, reason = await _ollama_circuit.preflight()
        if not allowed:
            raise RuntimeError(f"Ollama unreachable (escalation fast-fail): {reason}")
        model = TIER_TO_MODEL[tier]
        t0 = time.time()
        try:
            resp = await litellm.acompletion(
                model=model,
                messages=messages,
                api_base=OLLAMA_BASE,
            )
        except Exception as e:
            if _is_model_not_found(e):
                print("[router] " + _model_not_found_hint(model))
            raise
        duration = time.time() - t0
        _maybe_warn_cold_load(model, duration)
        prompt_t, completion_t = _extract_usage(resp)
        answer = _extract_answer(resp)
        # Issue #29: vram_mb is captured fire-and-forget so it never blocks the
        # response path (this return value IS the user's answer). The wrapper
        # schedules a one-time background fetch and returns the cached value
        # (None on first sight). Default-off users never touch the cache.
        vram = (_vram_mb_nonblocking(model)
                if self.config.capability_routing.enabled else None)
        return resp, Attempt(
            tier=tier, model=model,
            prompt_tokens=prompt_t, completion_tokens=completion_t,
            duration_s=duration,
            critic_score=None,
            preview=answer[:200],
            was_warm=_mark_and_check_warm(model),
            cost_usd=_safe_cost_usd(resp),
            vram_mb=vram,
        )

    async def _call_cloud(self, messages: list[dict[str, Any]]) -> Optional[tuple[Any, Attempt]]:
        """Returns:
          - None                       -> cloud genuinely NOT attempted (disabled / no key)
          - (None, error_attempt)      -> cloud WAS attempted but the call errored (H-5)
          - (response, attempt)        -> cloud succeeded
        The (None, attempt) case lets the orchestrator record the failed CLOUD
        attempt in the ledger and say "attempted and failed" instead of the
        misleading "disabled or not configured".
        """
        ce = self.config.cloud_escalation
        if not ce.enabled:
            return None
        api_key = os.environ.get(ce.api_key_env)
        if not api_key:
            print(f"[router] cloud escalation enabled but {ce.api_key_env} is not set; skipping")
            return None
        import litellm
        t0 = time.time()
        try:
            resp = await litellm.acompletion(
                model=ce.model,
                messages=messages,
                api_key=api_key,
                timeout=ce.timeout_s,
            )
        except Exception as e:
            # H-5 + M-5: record the failed attempt (redacted) so the ledger is
            # honest, instead of returning None (which reads as "never tried").
            print(f"[router] cloud escalation FAILED ({type(e).__name__}): {_redact(e)}")
            return None, Attempt(
                tier="CLOUD", model=ce.model,
                prompt_tokens=0, completion_tokens=0,
                duration_s=time.time() - t0,
                critic_score=None,
                preview=f"(cloud call failed: {type(e).__name__}: {_redact(e)[:160]})",
            )
        duration = time.time() - t0
        prompt_t, completion_t = _extract_usage(resp)
        answer = _extract_answer(resp)
        return resp, Attempt(
            tier="CLOUD", model=ce.model,
            prompt_tokens=prompt_t, completion_tokens=completion_t,
            duration_s=duration,
            critic_score=None,
            preview=answer[:200],
            was_warm=_mark_and_check_warm(ce.model),
            cost_usd=_safe_cost_usd(resp),
            # vram_mb intentionally left default here -- Task 2 adds the gated fetch.
        )

    async def _orchestrate(
        self,
        data: dict[str, Any],
        first_response: Any,
        first_attempt: Attempt,
    ) -> tuple[Any, list[Attempt], bool, bool]:
        """
        Walk the chain from first_attempt's tier upward until critic passes or XL exhausted.
        Then optionally one cloud step. Returns (final_response, ledger, cloud_attempted, handoff).
        """
        attempts: list[Attempt] = [first_attempt]
        # C-1: track each (attempt, response) pair so the handoff draft picks
        # text from the actually-highest-scoring response, not whichever
        # response happens to be `current_resp` at the end of the loop.
        # Failed escalations (C-5) have no response object and are intentionally
        # not added here.
        attempt_resps: list[tuple[Attempt, Any]] = [(first_attempt, first_response)]
        task_text = _extract_text(data.get("messages") or [])
        messages = data.get("messages") or []
        current_resp = first_response
        threshold = self.config.critic_pass_threshold
        cloud_attempted = False

        # Tier S is never critiqued — short-circuit.
        if first_attempt.tier == "S":
            return current_resp, attempts, False, False

        # H-1: hard safety cap on the escalation loop. In production the tier
        # pointer always advances via NEXT_TIER (S→M→L→XL→stop), so this is
        # never hit. But a future bug — or a test mock — that fails to advance
        # the tier would otherwise spin forever holding the GPU. The chain can
        # be at most len(NEXT_TIER)+1 steps; +2 gives headroom without ever
        # masking correct behavior.
        max_steps = len(NEXT_TIER) + 2
        steps = 0
        while True:
            steps += 1
            if steps > max_steps:
                print(f"[router] SAFETY: orchestration exceeded {max_steps} steps "
                      f"(tier pointer not advancing?); breaking to avoid a spin loop")
                break
            answer = _extract_answer(current_resp)
            score = await self._critique(task_text, answer)
            attempts[-1].critic_score = score
            print(f"[critic] tier={attempts[-1].tier} score={score}/5"
                  if score is not None else f"[critic] tier={attempts[-1].tier} score=N/A")

            # H-7: shared decision helper (same one the streaming path uses).
            outcome = self._critique_outcome(attempts[-1].tier, score)
            if outcome == "soft_pass":
                print(f"[router] soft-pass on tier {attempts[-1].tier} (critic failed); shipping current answer")
                return current_resp, attempts, False, False
            if outcome == "pass":
                return current_resp, attempts, False, False
            # outcome == "escalate": fall through to the next tier.
            nxt = NEXT_TIER.get(attempts[-1].tier)  # type: ignore[arg-type]
            if nxt is None:
                break  # XL exhausted
            print(f"[router] escalating {attempts[-1].tier} -> {nxt}")
            try:
                current_resp, new_attempt = await self._call_tier(messages, nxt)
            except Exception as e:
                # C-5: record the failed escalation in the ledger so the
                # handoff message + dashboard show what actually happened.
                # Without this, debugging "why didn't L get tried?" requires
                # grepping stdout — which is destroyed on proxy restart.
                print(f"[router] escalation to {nxt} failed: {e}")
                attempts.append(Attempt(
                    tier=nxt,
                    model=TIER_TO_MODEL.get(nxt, "unknown"),
                    prompt_tokens=0,
                    completion_tokens=0,
                    duration_s=0.0,
                    critic_score=None,
                    preview=f"(escalation failed: {type(e).__name__}: {str(e)[:160]})",
                ))
                # Failed-attempt has no response object — do NOT append to
                # attempt_resps (would crash _extract_answer if it became best).
                break
            attempts.append(new_attempt)
            attempt_resps.append((new_attempt, current_resp))

        # XL (or last tier reached) failed. Try cloud once.
        cloud_result = await self._call_cloud(messages)
        if cloud_result is not None:
            cloud_attempted = True
            cloud_resp, cloud_attempt = cloud_result
            if cloud_resp is None:
                # H-5: cloud was attempted but the call errored. Record the
                # failed attempt in the ledger (no critique — there's no answer
                # to score). cloud_attempted=True makes render_handoff say
                # "attempted and failed" rather than "disabled".
                attempts.append(cloud_attempt)
            else:
                cloud_score = await self._critique(task_text, _extract_answer(cloud_resp))
                cloud_attempt.critic_score = cloud_score
                attempts.append(cloud_attempt)
                attempt_resps.append((cloud_attempt, cloud_resp))
                print(f"[critic] tier=CLOUD score={cloud_score}/5"
                      if cloud_score is not None else "[critic] tier=CLOUD score=N/A")
                if cloud_score is not None and cloud_score >= threshold:
                    return cloud_resp, attempts, True, False

        # Build handoff: pick the actually-highest-scoring attempt's answer
        # as the draft. C-1: previously this always used current_resp (the
        # last response, usually XL), which could disagree with render_handoff's
        # "◄ best" marker and show the user a worse draft than what was
        # actually generated.
        scored = [(a, r) for a, r in attempt_resps if a.critic_score is not None]
        if scored:
            best_attempt, best_resp = max(scored, key=lambda p: p[0].critic_score or -1)
            best_answer = _extract_answer(best_resp)
        else:
            best_answer = _extract_answer(current_resp)
        message = render_handoff(attempts, threshold, cloud_attempted, best_answer)
        # C-4: previously this was bare try/except: pass — a future LiteLLM
        # response-shape change would silently drop the handoff message and
        # the user would receive the un-augmented bad answer with no signal
        # that escalation was even attempted.
        try:
            current_resp.choices[0].message.content = message
        except Exception as e:
            print(f"[router] CRITICAL: failed to inject handoff into response: "
                  f"{type(e).__name__}: {e}. Response shape may have changed; "
                  f"audit trail (attempts={[a.tier for a in attempts]}) is in SQLite.")
        return current_resp, attempts, cloud_attempted, True

    # POST-CALL (non-streaming) ─────────────────────────────────────────
    async def async_post_call_success_hook(
        self,
        data: dict[str, Any],
        user_api_key_dict,
        response,
    ):
        state = data.get("_router_state")
        if not state:
            return response  # explicit pin

        # The first call already happened via the proxy. Build the first Attempt
        # from what we know. Duration is end-to-end since pre-call.
        # Issue #29: this first_attempt IS the logged attempt for non-escalated
        # requests (every tier-S request + anything that passes on first try), so
        # the cost fields MUST be populated here, not only on escalation/cloud.
        first_model = TIER_TO_MODEL[state["initial_tier"]]
        first_attempt = Attempt(
            tier=state["initial_tier"],
            model=first_model,
            prompt_tokens=_extract_usage(response)[0],
            completion_tokens=_extract_usage(response)[1],
            duration_s=time.time() - state["started_at"],
            critic_score=None,
            preview=_extract_answer(response)[:200],
            was_warm=_mark_and_check_warm(first_model),
            cost_usd=_safe_cost_usd(response),
            vram_mb=(_vram_mb_nonblocking(first_model)
                     if self.config.capability_routing.enabled else None),
        )

        _maybe_warn_cold_load(first_model, first_attempt.duration_s)

        final_resp, attempts, cloud_attempted, handoff = await self._orchestrate(
            data, response, first_attempt
        )
        _log_decision(state, attempts, cloud_attempted, handoff, streamed=False)
        self._attach_advisory_headers(final_resp, state)
        return final_resp

    def _attach_advisory_headers(self, response, state):
        """Best-effort: attach x-triagellm-cap-* headers to a non-streaming
        response when advisory mode is active. Fully fault-isolated -- a
        header-API change or a response without _hidden_params can never break
        the request; the headers are simply dropped."""
        try:
            if self.config.capability_routing.mode != "advisory":
                return
            cap = state.get("capability")
            if not isinstance(cap, dict):
                return
            new_headers = _advisory_headers(cap, state.get("initial_tier"))
            hp = getattr(response, "_hidden_params", None)
            if hp is None:
                hp = {}
                response._hidden_params = hp
            existing = hp.get("additional_headers") or {}
            existing.update(new_headers)
            hp["additional_headers"] = existing
        except Exception as e:   # noqa: BLE001 -- advisory must never break a response
            print(f"[advisory] header attach non-fatal: {type(e).__name__}: {e}")

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: Any,
        traceback_str: Optional[str] = None,
    ):
        """Observability: on a model-not-pulled failure, log an actionable
        'ollama pull X' hint. NEVER raises and NEVER alters non-matching
        failures -- LiteLLM owns the error response."""
        try:
            if _is_model_not_found(original_exception):
                req_model = (request_data or {}).get("model", "the requested model")
                model = _model_name_from_error(original_exception, req_model)
                print("[router] " + _model_not_found_hint(model))
        except Exception:
            pass  # a logging hook must never mask the original error

    # POST-CALL (streaming) ─────────────────────────────────────────────
    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[Any, None]:
        state = request_data.get("_router_state")
        if not state:
            async for item in response:
                yield item
            return

        accumulated: list[str] = []
        last_chunk: Any = None
        skipped = 0   # M-3: count chunks whose content we couldn't read
        async for chunk in response:
            last_chunk = chunk
            try:
                delta = chunk.choices[0].delta.content
                if delta:
                    accumulated.append(delta)
            except (AttributeError, IndexError, TypeError, KeyError):
                # Tool-call deltas / unusual shapes don't carry text content.
                # Pass the chunk through untouched but count the miss so a
                # systematic shape change (critic scoring half the answer ->
                # spurious handoffs) is visible instead of silent.
                skipped += 1
            yield chunk

        if skipped:
            print(f"[router] (stream) {skipped} chunk(s) had no readable content; "
                  f"assembled critique text may be partial")

        # Stream finished. Critique the assembled answer.
        full = "".join(accumulated)
        task_text = _extract_text(request_data.get("messages") or [])
        tier: Tier = state["initial_tier"]
        # Issue #29: this first_attempt IS the logged attempt for non-escalated
        # streaming requests, so populate the free cost signals here too.
        model = TIER_TO_MODEL[tier]
        first_attempt = Attempt(
            tier=tier,
            model=model,
            prompt_tokens=len(task_text) // 4,
            completion_tokens=len(full) // 4,
            duration_s=time.time() - state["started_at"],
            critic_score=None,
            preview=full[:200],
            was_warm=_mark_and_check_warm(model),
            vram_mb=(_vram_mb_nonblocking(model)
                     if self.config.capability_routing.enabled else None),
            # cost_usd left None: streaming has no single costable response
            # object for litellm.completion_cost; streaming is local anyway.
        )
        attempts = [first_attempt]
        cloud_attempted = False
        handoff = False

        if tier != "S":
            score = await self._critique(task_text, full)
            first_attempt.critic_score = score
            print(f"[critic] (stream) tier={tier} score={score}/5"
                  if score is not None else f"[critic] (stream) tier={tier} score=N/A")
            # H-7: use the SAME decision helper as the non-streaming path. We
            # can't re-route mid-stream, so "escalate" becomes "emit a handoff
            # note". Previously this only fired on a real low score — a critic
            # failure (None) on tier L/XL shipped silently, diverging from the
            # non-streaming path which would escalate. Now L/XL critic-fail
            # streams a handoff; soft-pass tiers (S/M) still ship quietly.
            if self._critique_outcome(tier, score) == "escalate":
                # Build a one-shot handoff note (no mid-stream re-routing).
                note = (
                    "\n\n"
                    + render_handoff(attempts, self.config.critic_pass_threshold,
                                     cloud_attempted=False, best_answer="(streamed above)")
                )
                if last_chunk is not None:
                    try:
                        # C-2: deepcopy is required. `clone = last_chunk` was an
                        # alias — mutating its delta.content rewrote the already-
                        # yielded chunk that the client may still hold a reference
                        # to (and that the test fixture asserts on). Shallow copy
                        # is not enough since .choices[0].delta is a nested object.
                        clone = copy.deepcopy(last_chunk)
                        clone.choices[0].delta.content = note
                        clone.choices[0].finish_reason = None
                        yield clone
                        handoff = True
                    except Exception as e:
                        # C-3: don't claim we sent a handoff when the yield failed.
                        # Previously handoff=True was set before the try/except, so
                        # the SQLite row said handoff=1 while the client never saw
                        # the note — audit log diverged from reality.
                        print(f"[router] CRITICAL: stream handoff chunk failed: "
                              f"{type(e).__name__}: {e}")
                        handoff = False

        _log_decision(state, attempts, cloud_attempted, handoff, streamed=True)


tier_router_instance = TierRouter()
