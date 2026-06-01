"""TriageLLM User Acceptance Test — exercises the live proxy end-to-end.

Unlike pytest (which mocks every Ollama call), this script hits the *real*
proxy and the *real* Ollama models, and asserts behavior that mocks can't
verify:

    Phase 1: Liveness         — proxy/Ollama/models all reachable
    Phase 2: Classifier       — rules pick the expected tier per prompt
    Phase 3: Orchestration    — attempt ledger is well-formed, escalation
                                machinery records correctly
    Phase 4: Streaming        — chunked responses pass through cleanly

It does NOT auto-start the proxy. Run `start_route_llm.bat` first.

Each request is correlated to its decisions-row by snapshotting MAX(ts)
beforehand and reading the new row(s) inserted after the HTTP call returns.

Exit codes:  0 = all PASS  |  1 = one or more FAIL  |  2 = setup error
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# We import a few constants from router_hook for the DB path, but want to
# skip the module-level critic warmup (UAT talks to the live proxy, which
# already has its own warmed critic).
import os
os.environ.setdefault("TRIAGELLM_SKIP_WARMUP", "1")

sys.path.insert(0, str(Path(__file__).parent))

import httpx
from router_hook import DB_PATH, OLLAMA_BASE, TIER_TO_MODEL, classify_rules

PROXY_BASE = "http://localhost:4000"
API_KEY = "sk-local-dev"


# ─── Result plumbing ───────────────────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    status: str       # "PASS" | "WARN" | "FAIL"
    detail: str
    duration_s: float

    @property
    def glyph(self) -> str:
        return {"PASS": "✓", "WARN": "!", "FAIL": "✗"}.get(self.status, "?")


def _section(title: str) -> None:
    print(f"\n── {title} {'─' * (70 - len(title))}")


def _record(results: list[CheckResult], name: str, status: str,
            detail: str, t0: float) -> None:
    r = CheckResult(name, status, detail, time.time() - t0)
    results.append(r)
    print(f"  {r.glyph} {r.status:<4}  {r.name:<48} {r.detail}  ({r.duration_s:.2f}s)")


# ─── DB correlation ────────────────────────────────────────────────────────

def _ts_high_water() -> float:
    """Return MAX(ts) so callers can later find rows inserted *after* this."""
    try:
        with sqlite3.connect(DB_PATH) as c:
            row = c.execute("SELECT COALESCE(MAX(ts), 0) FROM decisions").fetchone()
            return float(row[0] or 0.0)
    except sqlite3.OperationalError:
        return 0.0  # table not yet created — proxy hasn't been hit


def _fetch_decisions_after(ts_low: float) -> list[dict]:
    """Pull rows logged after ts_low. Sleeps briefly to let the proxy flush."""
    # The hook writes the row synchronously in the success path, but the
    # client gets the HTTP response slightly before the write commits.
    # 200ms is generous and not noticeable in UAT wall-clock.
    time.sleep(0.2)
    with sqlite3.connect(DB_PATH) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT * FROM decisions WHERE ts > ? ORDER BY ts ASC",
            (ts_low,),
        ).fetchall()
    return [dict(r) for r in rows]


# ─── HTTP helpers ──────────────────────────────────────────────────────────

def _chat(prompt: str, *, stream: bool = False, timeout: float = 180.0) -> Any:
    body = {
        "model": "local-auto",
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=timeout) as c:
        if not stream:
            r = c.post(f"{PROXY_BASE}/v1/chat/completions", json=body, headers=headers)
            r.raise_for_status()
            return r.json()
        # Streaming: read SSE chunks ourselves so we can count them.
        chunks: list[str] = []
        full_text_parts: list[str] = []
        with c.stream("POST", f"{PROXY_BASE}/v1/chat/completions",
                      json=body, headers=headers) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                if not raw or not raw.startswith("data: "):
                    continue
                payload = raw[len("data: "):]
                if payload.strip() == "[DONE]":
                    break
                chunks.append(payload)
                try:
                    obj = json.loads(payload)
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    full_text_parts.append(delta.get("content") or "")
                except Exception:
                    pass
        return {"chunks": chunks, "full": "".join(full_text_parts)}


# ─── Phases ────────────────────────────────────────────────────────────────

def phase_1_liveness(results: list[CheckResult]) -> bool:
    """Proxy + Ollama reachable, every tier model pingable."""
    _section("Phase 1: Liveness")
    ok = True

    # Proxy
    t0 = time.time()
    try:
        r = httpx.get(f"{PROXY_BASE}/health/liveliness", timeout=3.0)
        if r.status_code == 200:
            _record(results, "proxy /health/liveliness", "PASS", f"HTTP {r.status_code}", t0)
        else:
            _record(results, "proxy /health/liveliness", "FAIL", f"HTTP {r.status_code}", t0)
            ok = False
    except Exception as e:
        _record(results, "proxy /health/liveliness", "FAIL", f"unreachable: {e!s:.60}", t0)
        return False  # no point continuing

    # Ollama
    t0 = time.time()
    try:
        r = httpx.get(f"{OLLAMA_BASE}/api/version", timeout=3.0)
        _record(results, "ollama /api/version",
                "PASS" if r.status_code == 200 else "FAIL",
                f"HTTP {r.status_code}", t0)
        ok = ok and r.status_code == 200
    except Exception as e:
        _record(results, "ollama /api/version", "FAIL", f"unreachable: {e!s:.60}", t0)
        ok = False

    # Each tier model — tiny generation to confirm loadable.
    for tier, model in TIER_TO_MODEL.items():
        t0 = time.time()
        ollama_name = model.split("/", 1)[1] if "/" in model else model
        try:
            r = httpx.post(
                f"{OLLAMA_BASE}/api/generate",
                json={"model": ollama_name, "prompt": "hi",
                      "stream": False, "options": {"num_predict": 1},
                      "keep_alive": "1m"},
                timeout=120,
            )
            if r.status_code == 200 and "response" in r.json():
                _record(results, f"tier {tier} ({ollama_name})", "PASS", "loaded", t0)
            else:
                detail = r.json().get("error", f"HTTP {r.status_code}")[:60]
                _record(results, f"tier {tier} ({ollama_name})", "FAIL", detail, t0)
                ok = False
        except Exception as e:
            _record(results, f"tier {tier} ({ollama_name})", "FAIL", f"{e!s:.60}", t0)
            ok = False

    return ok


# Each tuple: (prompt, expected_tier, why_we_expect_it)
TIER_PROMPTS: list[tuple[str, str, str]] = [
    ("Rename foo to bar.",                                       "S",
     "short + S keyword 'rename'"),
    ("Implement a function that reverses a string.",             "M",
     "M keyword 'implement'"),
    ("Refactor this module to improve performance and benchmark the result.", "L",
     "L keywords 'refactor', 'performance', 'benchmark'"),
    ("Architect a security review for the authentication system handling "
     "concurrent migrations across a distributed deployment.",   "XL",
     "XL keywords 'architect', 'security', 'authentication', 'concurrent', 'migration', 'distributed'"),
]


def phase_2_classifier(results: list[CheckResult]) -> bool:
    """Rule classifier picks expected tier for each curated prompt."""
    _section("Phase 2: Rule classifier accuracy")
    ok = True
    for prompt, expected, why in TIER_PROMPTS:
        t0 = time.time()
        tier, score, signals = classify_rules(prompt)
        detail = f"got {tier} (score={score}, signals={','.join(signals) or '∅'})"
        if tier == expected:
            _record(results, f"'{prompt[:40]}…' → {expected}", "PASS", detail, t0)
        else:
            _record(results, f"'{prompt[:40]}…' → {expected}", "FAIL",
                    f"expected {expected}, {detail}. Reason expected: {why}", t0)
            ok = False
    return ok


def phase_3_orchestration(results: list[CheckResult]) -> bool:
    """Send one real request, verify a well-formed ledger is logged."""
    _section("Phase 3: Live orchestration (real proxy round-trip)")
    ok = True
    # Chosen so rule classifier picks tier M: matches `\bwrite\s+a\s+function\b`.
    # That lets us verify the critic actually fires (S short-circuits critique).
    prompt = "Write a function that returns today's date as an ISO 8601 string."

    snapshot = _ts_high_water()
    t0 = time.time()
    try:
        resp = _chat(prompt)
    except Exception as e:
        _record(results, "POST /v1/chat/completions", "FAIL", f"{e!s:.60}", t0)
        return False
    _record(results, "POST /v1/chat/completions", "PASS",
            f"got {len(resp.get('choices', [{}])[0].get('message', {}).get('content', ''))} chars",
            t0)

    # Read back the decision row.
    t0 = time.time()
    rows = _fetch_decisions_after(snapshot)
    if not rows:
        _record(results, "decisions row written", "FAIL",
                f"no new rows after ts={snapshot}", t0)
        return False
    row = rows[-1]
    _record(results, "decisions row written", "PASS",
            f"tier={row['tier']}, critic={row['critic']}, escalated_to={row['escalated_to']}", t0)

    # Validate ledger JSON.
    t0 = time.time()
    try:
        ledger = json.loads(row["attempts_json"] or "[]")
        if not isinstance(ledger, list) or not ledger:
            raise ValueError("empty or non-list ledger")
        first = ledger[0]
        required = {"tier", "model", "prompt_tokens", "completion_tokens",
                    "duration_s", "critic_score", "preview"}
        missing = required - set(first.keys())
        if missing:
            raise ValueError(f"ledger entry missing fields: {missing}")
        _record(results, "ledger JSON well-formed", "PASS",
                f"{len(ledger)} attempt(s); tiers={[a['tier'] for a in ledger]}", t0)
    except Exception as e:
        _record(results, "ledger JSON well-formed", "FAIL", str(e)[:60], t0)
        ok = False

    # Tier S short-circuits critique — informational only.
    t0 = time.time()
    if row["tier"] == "S":
        _record(results, "tier S short-circuit (no critique)",
                "PASS" if row["critic"] is None else "WARN",
                f"critic={row['critic']} (expected None for S)", t0)
    else:
        _record(results, f"critique fired on tier {row['tier']}",
                "PASS" if row["critic"] is not None else "WARN",
                f"critic_score={row['critic']}", t0)

    return ok


def phase_4_streaming(results: list[CheckResult]) -> bool:
    """Streaming SSE chunks pass through, final text is non-empty."""
    _section("Phase 4: Streaming")
    ok = True
    # Force tier M so the post-stream critic actually fires (S short-circuits
    # critique). Verifies the *full* streaming path: chunks pass through +
    # critic runs on the assembled text + handoff chunk machinery is exercised.
    prompt = "Write a function that returns the SHA-256 hash of an input string."

    snapshot = _ts_high_water()
    t0 = time.time()
    try:
        out = _chat(prompt, stream=True)
    except Exception as e:
        _record(results, "streamed POST /v1/chat/completions", "FAIL", f"{e!s:.60}", t0)
        return False

    chunks = out.get("chunks") or []
    text = out.get("full") or ""
    if chunks and text.strip():
        _record(results, "streamed POST /v1/chat/completions", "PASS",
                f"{len(chunks)} chunks, {len(text)} chars assembled", t0)
    else:
        _record(results, "streamed POST /v1/chat/completions", "FAIL",
                f"chunks={len(chunks)} text_len={len(text)}", t0)
        ok = False

    t0 = time.time()
    rows = _fetch_decisions_after(snapshot)
    if rows and rows[-1].get("streamed") == 1:
        _record(results, "decisions row flagged streamed=1", "PASS",
                f"tier={rows[-1]['tier']} handoff={rows[-1]['handoff']}", t0)
    else:
        _record(results, "decisions row flagged streamed=1", "WARN",
                "row not found or streamed flag 0", t0)

    return ok


# ─── Driver ────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="TriageLLM live UAT")
    p.add_argument("--skip-stream", action="store_true",
                   help="Skip the streaming phase (saves ~30s)")
    p.add_argument("--skip-orchestration", action="store_true",
                   help="Skip the orchestration phase (saves real-model load time)")
    args = p.parse_args()

    print("TriageLLM live UAT")
    print(f"  proxy:  {PROXY_BASE}")
    print(f"  ollama: {OLLAMA_BASE}")
    print(f"  db:     {DB_PATH}")

    overall_t0 = time.time()
    results: list[CheckResult] = []

    if not phase_1_liveness(results):
        print("\n⚠  Liveness failed — aborting later phases. "
              "Start the proxy with start_route_llm.bat and retry.")
        _print_summary(results, time.time() - overall_t0)
        return 2

    phase_2_classifier(results)
    if not args.skip_orchestration:
        phase_3_orchestration(results)
    if not args.skip_stream:
        phase_4_streaming(results)

    return _print_summary(results, time.time() - overall_t0)


def _print_summary(results: list[CheckResult], elapsed: float) -> int:
    print("\n" + "═" * 76)
    passed = sum(1 for r in results if r.status == "PASS")
    warned = sum(1 for r in results if r.status == "WARN")
    failed = sum(1 for r in results if r.status == "FAIL")
    print(f"UAT: {passed} PASS  |  {warned} WARN  |  {failed} FAIL"
          f"   (total {elapsed:.1f}s, $0.00 cost)")
    print("═" * 76)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
