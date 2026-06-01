"""TriageLLM health check — verify the full stack is working.

Checks performed:
1. Proxy alive (HTTP 200 on /health/liveliness)
2. Ollama reachable (HTTP 200 on /api/version)
3. Each configured model alias is loadable
4. Critic responds in <expected timeout
5. CPU-pinning flag respected (informational)

Exits 0 if all PASS, 1 if any FAIL. WARN exits 0 but flags concerns.

Usage:
    .\\.venv\\Scripts\\python.exe health.py
    .\\.venv\\Scripts\\python.exe health.py --json
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent))

import contextlib

import httpx
# SMK-005: import router_hook with its import-time banner ("[router] config
# loaded ...", warmup line) redirected to stderr, so `health.py --json` emits
# ONLY valid JSON on stdout and can be piped to jq / parsed cleanly.
with contextlib.redirect_stdout(sys.stderr):
    from router_hook import (
        OLLAMA_BASE, TIER_TO_MODEL, critic_score, load_config,
    )


async def _check_url(url: str, timeout: float = 3.0) -> tuple[str, str]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(url)
        return ("PASS", f"HTTP {r.status_code}") if r.status_code == 200 else ("FAIL", f"HTTP {r.status_code}")
    except Exception as e:
        return ("FAIL", str(e)[:80])


async def _check_model_pingable(model: str) -> tuple[str, str]:
    """Verify a model can be loaded by issuing a tiny generation."""
    try:
        # Strip ollama_chat/ prefix for the direct /api/generate call.
        ollama_name = model.split("/", 1)[1] if "/" in model else model
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(
                f"{OLLAMA_BASE}/api/generate",
                json={
                    "model": ollama_name,
                    "prompt": "hi",
                    "stream": False,
                    "options": {"num_predict": 1},
                    "keep_alive": "1m",
                },
            )
        if r.status_code == 200:
            data = r.json()
            if "response" in data:
                return ("PASS", f"loaded, gen ok")
            err = data.get("error", "unknown")
            return ("FAIL", err[:80])
        return ("FAIL", f"HTTP {r.status_code}")
    except Exception as e:
        return ("FAIL", str(e)[:80])


async def _check_critic_responds() -> tuple[str, str, float]:
    cfg = load_config()
    t0 = time.time()
    try:
        score = await critic_score(
            "warmup", "warmup",
            model=cfg.critic_model,
            timeout_s=cfg.critic_timeout_s,
            cpu_only=cfg.critic_cpu_only,
        )
        dur = time.time() - t0
        if score is not None:
            status = "PASS" if dur < 5.0 else "WARN"
            return (status, f"score={score}, {dur:.2f}s", dur)
        return ("FAIL", f"no score returned ({dur:.2f}s)", dur)
    except Exception as e:
        return ("FAIL", str(e)[:80], time.time() - t0)


def _emoji(status: str) -> str:
    return {"PASS": "✓", "WARN": "!", "FAIL": "✗"}.get(status, "?")


async def main_async() -> int:
    p = argparse.ArgumentParser(description="TriageLLM health check")
    p.add_argument("--json", action="store_true", help="Emit JSON")
    p.add_argument("--skip-models", action="store_true",
                   help="Skip per-model checks (much faster)")
    args = p.parse_args()

    cfg = load_config()
    results: list[dict] = []

    # 1. Proxy
    status, detail = await _check_url("http://localhost:4000/health/liveliness")
    results.append({"check": "proxy", "status": status, "detail": detail})

    # 2. Ollama
    status, detail = await _check_url(f"{OLLAMA_BASE}/api/version")
    results.append({"check": "ollama", "status": status, "detail": detail})

    # 3. Critic
    status, detail, _dur = await _check_critic_responds()
    results.append({"check": f"critic ({cfg.critic_model}, "
                             f"{'CPU' if cfg.critic_cpu_only else 'GPU'})",
                    "status": status, "detail": detail})

    # 4. Models (optional, slow on first call)
    if not args.skip_models:
        for tier, model in TIER_TO_MODEL.items():
            status, detail = await _check_model_pingable(model)
            results.append({"check": f"tier {tier} ({model})", "status": status, "detail": detail})

    # 5. Cloud config (informational)
    ce = cfg.cloud_escalation
    import os
    if ce.enabled:
        key_set = bool(os.environ.get(ce.api_key_env))
        cs = "PASS" if key_set else "WARN"
        cd = f"{ce.model} (key {'set' if key_set else 'MISSING from env'})"
    else:
        cs, cd = "PASS", "disabled"
    results.append({"check": "cloud escalation", "status": cs, "detail": cd})

    # Output
    if args.json:
        print(json.dumps({"results": results, "all_ok": all(r["status"] != "FAIL" for r in results)}, indent=2))
    else:
        print("TriageLLM health check")
        print("─" * 60)
        for r in results:
            print(f"  {_emoji(r['status'])} {r['status']:<5} {r['check']:<45} {r['detail']}")
        print("─" * 60)
        fails = [r for r in results if r["status"] == "FAIL"]
        warns = [r for r in results if r["status"] == "WARN"]
        if fails:
            print(f"FAIL: {len(fails)} check(s) failed.")
        elif warns:
            print(f"OK with {len(warns)} warning(s).")
        else:
            print("All checks passed.")
    return 1 if any(r["status"] == "FAIL" for r in results) else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async()))
