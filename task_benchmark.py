"""TriageLLM per-machine task benchmark (Issue #24).

Runs installed Ollama models against curated per-category prompts, scores each
output via success.score_output, measures per-model latency, and writes a
per-(model, category) recommendation map to JSON (priors for advisory mode
#18b). Calls Ollama DIRECTLY (not through the LiteLLM proxy / tier router) -
it measures individual models, not the router's tier decision.

Pure functions (filter_models, aggregate, recommend, build_priors,
render_report, needs_critic) contain all judgment and are unit-tested with no
Ollama. Only _fetch_tags, run_prompt, and run_benchmark touch the network.

All runtime strings are ASCII (cp1252-safe). No `from __future__ import
annotations` (LiteLLM spec_from_file_location + dataclass crash).
"""
import json
import os
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

sys.path.insert(0, str(Path(__file__).parent))
import success  # noqa: E402  pure stdlib scorer (#18a)
from router_hook import critic_score, load_config  # noqa: E402

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# The 8 testable capability categories. high_risk / default are routing-policy
# categories, not capability categories, so the benchmark skips them.
TESTABLE_CATEGORIES = [
    "quick_question", "explanation_or_summary", "structured_output",
    "analytical_task", "creative_generation", "modification_or_edit",
    "multi_step_or_planning", "long_context",
]

# Objective categories: success comes from a deterministic check in
# success.score_output and the critic is NOT needed for the verdict.
OBJECTIVE_CATEGORIES = {"structured_output", "modification_or_edit"}


@dataclass
class ModelInfo:
    name: str
    size: int          # on-disk bytes from /api/tags (tie-break rung 3)


@dataclass
class PromptResult:
    success: bool
    latency_s: float
    reason: str        # success.py reason code, or "generation-error"


@dataclass
class CategoryResult:
    success_rate: float
    n_prompts: int
    n_passed: int
    confidence: str            # "hard" (objective) | "soft"
    latency_p50_s: float
    latency_p95_s: Optional[float]  # None when n < 4 (meaningless on tiny n)
    latency_max_s: float


@dataclass
class Recommendation:
    model: str
    success_rate: float
    latency_p50_s: float
    confidence: str
    decided_by: str            # "success_rate" | "latency" | "size"
    warning: Optional[str]     # None | "no-model-passed"


def needs_critic(category: str) -> bool:
    """True if scoring this category requires a critic score (soft bucket).

    Objective categories decide success deterministically and skip the critic
    (faithful to success.score_output, which ignores the critic for their
    verdict). Unknown categories default to soft.
    """
    return category not in OBJECTIVE_CATEGORIES


def filter_models(tags, override, classifier_model, critic_model):
    """Pure: turn raw /api/tags entries into the benchmarkable ModelInfo list.

    Skip-list is deterministic and role-based: drop the configured
    classifier/critic models (the 0.5B tripwire) and any model whose name
    contains "embed". `override` (a list or comma-string of model names)
    bypasses the skip-list and selects exactly those that are installed.
    """
    by_name = {t["name"]: int(t.get("size", 0)) for t in tags}
    if override:
        if isinstance(override, str):
            wanted = [s.strip() for s in override.split(",") if s.strip()]
        else:
            wanted = [s.strip() for s in override if s and s.strip()]
        out = []
        for name in wanted:
            if name in by_name:
                out.append(ModelInfo(name, by_name[name]))
            else:
                print("[benchmark] requested model not installed, skipping: " + name)
        return out
    skip = {classifier_model, critic_model}
    out = []
    for name, size in by_name.items():
        if name in skip:
            continue
        if "embed" in name.lower():
            continue
        out.append(ModelInfo(name, size))
    return out


def _p95(sorted_vals):
    """p95 using the same index rule as the critic-latency benchmark.py."""
    return sorted_vals[int(0.95 * len(sorted_vals)) - 1]


def aggregate(category, prompt_results):
    """Pure: roll a list[PromptResult] into one CategoryResult.

    confidence is derived from the category (hard for objective, soft
    otherwise) - it is the SCORING METHOD, not the outcome.
    """
    n = len(prompt_results)
    n_passed = sum(1 for r in prompt_results if r.success)
    latencies = sorted(r.latency_s for r in prompt_results)
    success_rate = round(n_passed / n, 4) if n else 0.0
    p50 = round(statistics.median(latencies), 2) if latencies else 0.0
    p95 = round(_p95(latencies), 2) if n >= 4 else None
    mx = round(max(latencies), 2) if latencies else 0.0
    confidence = "hard" if category in OBJECTIVE_CATEGORIES else "soft"
    return CategoryResult(success_rate, n, n_passed, confidence, p50, p95, mx)


def recommend(results, sizes):
    """Pure: per category, pick the best model.

    results: {model_name: {category: CategoryResult}}
    sizes:   {model_name: on_disk_size_bytes}

    Tie-break ladder (capability first, then cost):
      1. highest success_rate
      2. lowest latency_p50_s
      3. smallest on-disk size
    decided_by records which rung settled it vs the runner-up.
    """
    out = {}
    for cat in TESTABLE_CATEGORIES:
        cands = [(m, results[m][cat]) for m in results if cat in results[m]]
        if not cands:
            continue
        cands.sort(key=lambda mc: (-mc[1].success_rate,
                                   mc[1].latency_p50_s,
                                   sizes.get(mc[0], 0)))
        win_model, win = cands[0]
        decided_by = "success_rate"
        if len(cands) > 1:
            _, second = cands[1]
            if win.success_rate > second.success_rate:
                decided_by = "success_rate"
            elif win.latency_p50_s < second.latency_p50_s:
                decided_by = "latency"
            else:
                decided_by = "size"
        warning = "no-model-passed" if win.success_rate == 0.0 else None
        out[cat] = Recommendation(win_model, win.success_rate,
                                  win.latency_p50_s, win.confidence,
                                  decided_by, warning)
    return out


def _category_to_dict(cr):
    return {
        "success_rate": cr.success_rate,
        "n_prompts": cr.n_prompts,
        "n_passed": cr.n_passed,
        "confidence": cr.confidence,
        "latency_p50_s": cr.latency_p50_s,
        "latency_p95_s": cr.latency_p95_s,
        "latency_max_s": cr.latency_max_s,
    }


def _rec_to_dict(rec):
    return {
        "model": rec.model,
        "success_rate": rec.success_rate,
        "latency_p50_s": rec.latency_p50_s,
        "confidence": rec.confidence,
        "decided_by": rec.decided_by,
        "warning": rec.warning,
    }


def build_priors(models, mode, prompts_per_category, results, recommendations,
                 generated_at=None):
    """Pure: assemble the benchmark_results.json structure.

    generated_at is injectable for deterministic tests; defaults to now (UTC).
    """
    if generated_at is None:
        generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "schema_version": 1,
        "generated_at": generated_at,
        "mode": mode,
        "prompts_per_category": prompts_per_category,
        "models": [m.name for m in models],
        "results": {
            model: {cat: _category_to_dict(cr) for cat, cr in cats.items()}
            for model, cats in results.items()
        },
        "recommendations": {
            cat: _rec_to_dict(rec) for cat, rec in recommendations.items()
        },
    }


def write_priors(path, obj):
    """Atomically write obj as pretty JSON (temp file + os.replace).

    A crash mid-write never leaves a half-written priors file in place.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def render_report(priors, json_mode=False):
    """Pure: render the priors as JSON (json_mode) or an ASCII text table."""
    if json_mode:
        return json.dumps(priors, indent=2)
    lines = []
    lines.append("TriageLLM task benchmark (mode={0}, P={1})".format(
        priors.get("mode"), priors.get("prompts_per_category")))
    lines.append("=" * 60)
    lines.append("Models: " + ", ".join(priors.get("models", [])))
    lines.append("Generated: " + str(priors.get("generated_at")))
    lines.append("")
    lines.append("Recommended model per category:")
    recs = priors.get("recommendations", {})
    for cat in TESTABLE_CATEGORIES:
        rec = recs.get(cat)
        if rec is None:
            continue
        note = ""
        if rec.get("warning"):
            note = "  [" + rec["warning"] + "]"
        lines.append(
            "  {cat:<24} -> {model:<24} (success {sr:.2f}, p50 {p50}s, "
            "by {by}){note}".format(
                cat=cat, model=rec["model"], sr=rec["success_rate"],
                p50=rec["latency_p50_s"], by=rec["decided_by"], note=note))
    return "\n".join(lines)


async def _fetch_tags():
    """Impure: GET /api/tags, return the raw models list (or raise)."""
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.get(OLLAMA_BASE + "/api/tags")
        r.raise_for_status()
        return r.json().get("models", [])


async def discover_models(override=None, config=None):
    """Impure: fetch installed models and apply the pure skip-list/override."""
    if config is None:
        config = load_config()
    tags = await _fetch_tags()
    return filter_models(tags, override, config.classifier_model,
                         config.critic_model)


async def run_prompt(model, text, timeout_s=120.0):
    """Impure: one /api/generate call. Returns (output_text, latency_s)."""
    t0 = time.time()
    async with httpx.AsyncClient(timeout=timeout_s) as c:
        r = await c.post(
            OLLAMA_BASE + "/api/generate",
            json={"model": model, "prompt": text, "stream": False},
        )
        r.raise_for_status()
        out = r.json().get("response") or ""
    return out, time.time() - t0


async def _warmup(model):
    """Impure: page the model into memory; not timed, not scored."""
    try:
        await run_prompt(model, "Say OK")
    except Exception as e:
        print("[benchmark] warmup failed for " + model + ": "
              + type(e).__name__ + ": " + str(e))


async def run_benchmark(prompts, mode="full", override=None, config=None,
                        output="benchmark_results.json"):
    """Impure orchestrator. Returns the priors dict and writes it to `output`.

    prompts: {category: [prompt, ...]}. mode "quick" uses P=2, else P=5.
    Calls Ollama directly (run_prompt) and the reused critic (critic_score),
    scores via success.score_output, aggregates, recommends, persists.
    """
    if config is None:
        config = load_config()
    models = await discover_models(override, config)
    if not models:
        raise SystemExit("[benchmark] no benchmarkable models found "
                         "(installed models all skipped? try --models)")
    p = 2 if mode == "quick" else 5

    results = {}
    for mi in models:
        await _warmup(mi.name)
        cat_results = {}
        for cat in TESTABLE_CATEGORIES:
            prompt_list = prompts.get(cat, [])[:p]
            if not prompt_list:
                continue
            prompt_results = []
            for text in prompt_list:
                try:
                    out, latency = await run_prompt(mi.name, text)
                except Exception as e:
                    print("[benchmark] " + mi.name + "/" + cat
                          + " generation error: " + type(e).__name__)
                    prompt_results.append(PromptResult(False, 0.0, "generation-error"))
                    continue
                critic = None
                if needs_critic(cat):
                    critic = await critic_score(
                        text, out, model=config.critic_model,
                        timeout_s=config.critic_timeout_s,
                        cpu_only=config.critic_cpu_only)
                res = success.score_output(cat, out, critic,
                                           config.critic_pass_threshold)
                prompt_results.append(PromptResult(res.success, latency, res.reason))
            if prompt_results:
                cat_results[cat] = aggregate(cat, prompt_results)
        results[mi.name] = cat_results

    sizes = {mi.name: mi.size for mi in models}
    recommendations = recommend(results, sizes)
    priors = build_priors(models, mode, p, results, recommendations)
    write_priors(output, priors)
    return priors
