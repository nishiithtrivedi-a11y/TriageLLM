"""TriageLLM doctor - setup pre-flight diagnostics (#15/#19/#20a).

Answers "is my setup correct?" (config vs installed models, env, cloud audit,
mode) - static and fast, runs WITHOUT the proxy. Complementary to health.py
(the live "is the running stack healthy?" check). Read-only; reuses router_hook
symbols. ASCII-only output (no stdout reconfigure). No `from __future__ import
annotations` (LiteLLM spec_from_file_location + dataclass crash).
"""
import argparse
import contextlib
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

sys.path.insert(0, str(Path(__file__).parent))
# Import router_hook with its module-level banner/warmup line redirected to
# stderr so `doctor.py --json` emits ONLY valid JSON on stdout (mirrors
# health.py SMK-005).
with contextlib.redirect_stdout(sys.stderr):
    from router_hook import (  # noqa: E402
        OLLAMA_BASE, TIER_TO_MODEL, DB_PATH, CONFIG_PATH, load_config)

_CLOUD_AUDIT_WINDOW = 1000  # "last N" decisions for the cloud-attempt count


@dataclass
class CheckResult:
    name: str
    status: str            # "PASS" | "WARN" | "FAIL"
    detail: str
    fix: Optional[str] = None


def _strip_prefix(model):
    """'ollama_chat/qwen2.5-coder:1.5b' -> 'qwen2.5-coder:1.5b'."""
    return model.split("/", 1)[-1] if model else model


def check_config(config, tier_map, config_present):
    """Pure: config file present + tier map coherent + classifier/critic named."""
    if not config_present:
        return CheckResult("config", "WARN", "config.yaml not found, using defaults",
                           "create config.yaml to customize tiers/models")
    if not tier_map:
        return CheckResult("config", "FAIL", "TIER_TO_MODEL is empty",
                           "define tier->model mappings in router_hook.py")
    if not config.classifier_model or not config.critic_model:
        return CheckResult("config", "FAIL",
                           "classifier_model or critic_model is blank",
                           "set classifier_model and critic_model in config.yaml")
    return CheckResult("config", "PASS",
                       "config OK: {0} tiers mapped, classifier={1}, critic={2}".format(
                           len(tier_map), config.classifier_model, config.critic_model))


def check_models(installed, configured):
    """Pure: every configured model present in the installed set.

    installed: set of names from /api/tags ('qwen2.5-coder:1.5b').
    configured: list of bare model names expected.
    """
    if not installed:
        return CheckResult("models", "WARN", "could not verify (no models listed)",
                           "is Ollama reachable? run: ollama list")
    missing = [m for m in configured if m not in installed]
    if missing:
        return CheckResult(
            "models", "FAIL", "missing: " + ", ".join(missing),
            "Model(s) configured but not installed. Public model: 'ollama pull "
            "<name>'. Custom Modelfile build: rebuild it.")
    return CheckResult("models", "PASS",
                       "all {0} configured models installed".format(len(configured)))


def analyze_cloud(config, env, attempts_last, attempts_ever, all_local):
    """Pure: cloud-audit verdict (#15) - the evidence-level local-first proof."""
    ce = config.cloud_escalation
    key_set = bool(env.get(ce.api_key_env))
    if not ce.enabled:
        if key_set:
            return CheckResult(
                "cloud-audit", "WARN",
                "cloud disabled but {0} is set in env (harmless, unused)".format(
                    ce.api_key_env),
                "unset {0} for zero cloud credentials present".format(ce.api_key_env))
        if not all_local:
            return CheckResult(
                "cloud-audit", "WARN",
                "cloud disabled, but a tier model is non-local",
                "ensure all TIER_TO_MODEL entries are ollama_chat/ aliases")
        return CheckResult(
            "cloud-audit", "PASS",
            "fully local: cloud disabled, {0} attempts of last {1}, {2} ever".format(
                attempts_last, _CLOUD_AUDIT_WINDOW, attempts_ever))
    if key_set:
        return CheckResult(
            "cloud-audit", "PASS",
            "cloud ON: {0}, used {1} of last {2}".format(
                ce.model, attempts_last, _CLOUD_AUDIT_WINDOW))
    return CheckResult(
        "cloud-audit", "WARN",
        "cloud enabled but {0} not set in env".format(ce.api_key_env),
        "set {0}, or set cloud_escalation.enabled=false".format(ce.api_key_env))


def derive_mode(config, env):
    """Pure: orthogonal Routing x Cloud status (#19). Informational (PASS)."""
    cap = config.capability_routing.enabled
    routing = "tier-based (capability routing: {0})".format("shadow" if cap else "off")
    ce = config.cloud_escalation
    if ce.enabled:
        key = "set" if env.get(ce.api_key_env) else "MISSING"
        cloud = "on ({0}, key {1})".format(ce.model, key)
    else:
        cloud = "off (cloud_escalation.enabled = false)"
    return CheckResult("mode", "PASS", "Routing: " + routing + " | Cloud: " + cloud)


def _fetch_version():
    """Impure: GET /api/version -> reachability CheckResult."""
    try:
        with httpx.Client(timeout=3.0) as c:
            r = c.get(OLLAMA_BASE + "/api/version")
        if r.status_code == 200:
            ver = ""
            try:
                ver = r.json().get("version", "")
            except (ValueError, TypeError):
                pass
            return CheckResult("ollama", "PASS",
                               "HTTP 200" + ((" (v" + ver + ")") if ver else ""))
        return CheckResult("ollama", "FAIL", "HTTP {0}".format(r.status_code),
                           "check Ollama at " + OLLAMA_BASE)
    except Exception as e:
        return CheckResult("ollama", "FAIL", str(e)[:80],
                           "start Ollama (ollama serve), or check OLLAMA_BASE_URL")


def _fetch_tags():
    """Impure: GET /api/tags -> set of installed model names ({} on failure)."""
    try:
        with httpx.Client(timeout=5.0) as c:
            r = c.get(OLLAMA_BASE + "/api/tags")
            r.raise_for_status()
            models = r.json().get("models", [])
        return {m.get("name") for m in models if m.get("name")}
    except Exception:
        return set()


def load_cloud_attempts(db_path):
    """Impure: (last_window, ever) cloud_attempted counts from the ledger."""
    try:
        with sqlite3.connect(db_path) as c:
            ever = c.execute(
                "SELECT COUNT(*) FROM decisions WHERE cloud_attempted = 1"
            ).fetchone()[0]
            last = c.execute(
                "SELECT COUNT(*) FROM (SELECT cloud_attempted FROM decisions "
                "ORDER BY ts DESC LIMIT ?) WHERE cloud_attempted = 1",
                (_CLOUD_AUDIT_WINDOW,)).fetchone()[0]
        return (int(last), int(ever))
    except sqlite3.OperationalError:
        return (0, 0)


def render_text(results):
    """Pure: ASCII-only aligned report with fix hints + summary."""
    lines = ["TriageLLM doctor (setup pre-flight)", "=" * 60]
    for r in results:
        lines.append("[{0}] {1:<13} {2}".format(r.status, r.name, r.detail))
        if r.fix and r.status != "PASS":
            lines.append("        -> fix: " + r.fix)
    lines.append("=" * 60)
    fails = sum(1 for r in results if r.status == "FAIL")
    warns = sum(1 for r in results if r.status == "WARN")
    if fails:
        lines.append("FAIL: {0} check(s) failed.".format(fails))
    elif warns:
        lines.append("OK with {0} warning(s).".format(warns))
    else:
        lines.append("All checks passed.")
    return "\n".join(lines)


def render_json(results):
    """Pure: machine-readable report."""
    return json.dumps({
        "results": [{"name": r.name, "status": r.status,
                     "detail": r.detail, "fix": r.fix} for r in results],
        "all_ok": all(r.status != "FAIL" for r in results),
    }, indent=2)


def _configured_models(config):
    names = {_strip_prefix(m) for m in TIER_TO_MODEL.values()}
    names.add(config.classifier_model)
    names.add(config.critic_model)
    return sorted(n for n in names if n)


def _all_local():
    return all(str(m).startswith("ollama_chat/") for m in TIER_TO_MODEL.values())


def run(args):
    """Impure orchestrator. Returns process exit code (1 if any FAIL)."""
    config = load_config()
    env = os.environ

    if args.cloud_audit:
        last, ever = load_cloud_attempts(str(DB_PATH))
        results = [analyze_cloud(config, env, last, ever, _all_local())]
    elif args.mode:
        results = [derive_mode(config, env)]
    else:
        results = []
        ver = _fetch_version()
        results.append(ver)
        results.append(check_config(config, TIER_TO_MODEL, CONFIG_PATH.exists()))
        if not args.skip_models:
            if ver.status == "FAIL":
                results.append(CheckResult("models", "WARN",
                                           "skipped (Ollama unreachable)"))
            else:
                results.append(check_models(_fetch_tags(), _configured_models(config)))
        last, ever = load_cloud_attempts(str(DB_PATH))
        results.append(analyze_cloud(config, env, last, ever, _all_local()))
        results.append(derive_mode(config, env))

    print(render_json(results) if args.json else render_text(results))
    return 1 if any(r.status == "FAIL" for r in results) else 0


def build_parser():
    p = argparse.ArgumentParser(
        description="TriageLLM doctor (setup pre-flight diagnostics)")
    p.add_argument("--json", action="store_true", help="Emit JSON")
    p.add_argument("--cloud-audit", action="store_true",
                   help="Only the cloud-audit (local-first proof)")
    p.add_argument("--mode", action="store_true",
                   help="Only the routing x cloud mode")
    p.add_argument("--skip-models", action="store_true",
                   help="Skip the /api/tags model-presence check")
    return p


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
