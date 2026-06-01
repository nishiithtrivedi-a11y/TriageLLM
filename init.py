"""Smart-defaults config generator for TriageLLM (Issue #14).

Detects installed Ollama models via /api/tags, assigns them to S/M/L/XL
tiers by parameter count, picks a tiny model for the classifier/critic, and
renders a fully-commented config.yaml. Writes config.generated.yaml by
default; --write replaces config.yaml (backing up to config.yaml.bak).

No router_hook / SQLite change -- init only writes a config file.
"""
import argparse
import asyncio
import os
import sys

from task_benchmark import _fetch_tags

# Tier bands, in billions of parameters (parsed from /api/tags).
_CLASSIFIER_MAX = 1.5   # <= this -> eligible as the 0.5B-style classifier/critic
_S_MAX = 3.0            # S worker band ceiling
_M_MAX = 14.0          # M worker band ceiling
_L_MAX = 28.0          # L worker band ceiling; anything larger is XL
_DEFAULT_CLASSIFIER = "qwen2.5:0.5b"   # fallback when no tiny model is installed


def _parse_params(model):
    """Pure: parameter count in billions.

    Prefers /api/tags `details.parameter_size` ("7.6B" / "500M"). Ollama
    almost always provides this. Safety-net fallback if absent: estimate from
    on-disk `size` bytes (~0.7 bytes/param for a typical Q4 quant).
    """
    details = model.get("details") or {}
    raw = details.get("parameter_size")
    if raw:
        s = str(raw).strip().upper()
        try:
            if s.endswith("B"):
                return float(s[:-1])
            if s.endswith("M"):
                return float(s[:-1]) / 1000.0
            return float(s)
        except ValueError:
            pass
    size_bytes = model.get("size") or 0
    if size_bytes:
        return size_bytes / 0.7e9
    return 0.0


def _is_coder(name):
    n = name.lower()
    return "coder" in n or "code" in n


def _is_embedding(name):
    """Embedding models report tiny param sizes but cannot chat/critique, so
    they must never be assigned to a tier or the classifier. Mirrors the
    role-based skip in task_benchmark.filter_models."""
    return "embed" in name.lower()


def _pick_in_band(candidates):
    """candidates: list of (name, params). Prefer a coder-named model, then
    the largest. Returns the chosen name or None if empty."""
    if not candidates:
        return None
    coders = [c for c in candidates if _is_coder(c[0])]
    pool = coders if coders else candidates
    pool = sorted(pool, key=lambda c: (c[1], c[0]))
    return pool[-1][0]


def assign_tiers(models):
    """Pure: assign detected models to classifier + S/M/L/XL workers.

    Returns the assignment dict consumed by build_config. Raises ValueError if
    no usable worker model is present.

    Embedding models (name contains "embed") are dropped up front -- they
    report tiny param sizes and would otherwise win the "smallest tiny model"
    classifier slot, but they cannot chat or critique.

    Band fill order: bucket workers by parameter count, then run a single
    top-down overflow pass -- when a band is empty and the band directly below
    holds more than one worker, its largest worker is promoted up. This spreads
    distinct models across tiers (e.g. {16B, 27B} both in L with XL empty ->
    16B stays L, 27B moves to XL) instead of picking only the largest per band
    and silently dropping the rest. Any still-empty band then borrows from the
    nearest filled band (smaller side first) and carries a comment.
    """
    enriched = [(m["name"], _parse_params(m)) for m in models
                if not _is_embedding(m["name"])]

    tiny = sorted(
        [e for e in enriched if e[1] <= _CLASSIFIER_MAX and e[1] > 0],
        key=lambda e: (e[1], e[0]),
    )
    if tiny:
        classifier = tiny[0][0]
        classifier_is_default = False
    else:
        classifier = _DEFAULT_CLASSIFIER
        classifier_is_default = True

    workers = []
    for name, params in enriched:
        if params <= 0:
            continue
        if name == classifier and params < _CLASSIFIER_MAX:
            continue
        workers.append((name, params))

    if not workers:
        raise ValueError(
            "no usable Ollama models found; pull at least one "
            "(e.g. `ollama pull qwen2.5-coder:1.5b`)")

    bands = {
        "S": [w for w in workers if w[1] <= _S_MAX],
        "M": [w for w in workers if _S_MAX < w[1] <= _M_MAX],
        "L": [w for w in workers if _M_MAX < w[1] <= _L_MAX],
        "XL": [w for w in workers if w[1] > _L_MAX],
    }

    # Overflow pass: when a lower band holds more than one worker and the band
    # directly above is empty, promote its largest worker up so distinct models
    # spread across tiers instead of stacking. Walk top-down so a promotion can
    # cascade.
    order = ["S", "M", "L", "XL"]
    for i in range(len(order) - 1, 0, -1):
        upper, lower = order[i], order[i - 1]
        if not bands[upper] and len(bands[lower]) > 1:
            biggest = sorted(bands[lower], key=lambda c: (c[1], c[0]))[-1]
            bands[lower].remove(biggest)
            bands[upper].append(biggest)

    tiers = {}
    for tier in ("S", "M", "L", "XL"):
        tiers[tier] = _pick_in_band(bands[tier])

    borrowed = {}
    for i, tier in enumerate(order):
        if tiers[tier] is not None:
            continue
        chosen = None
        for j in range(i - 1, -1, -1):
            if tiers[order[j]] is not None:
                chosen = tiers[order[j]]
                break
        if chosen is None:
            for j in range(i + 1, len(order)):
                if tiers[order[j]] is not None:
                    chosen = tiers[order[j]]
                    break
        tiers[tier] = chosen
        borrowed[tier] = (
            "no model in this size band was installed; reusing "
            + str(chosen) + " -- override for a better fit")

    return {
        "tiers": tiers,
        "classifier": classifier,
        "classifier_is_default": classifier_is_default,
        "borrowed": borrowed,
        "model_count": len(models),
    }


def _tier_comment(assignment, tier):
    """Inline comment for a borrowed tier, or empty string."""
    note = assignment["borrowed"].get(tier)
    return ("    # " + note + "\n") if note else ""


def build_config(assignment):
    """Pure: render the full config.yaml as a commented template string.

    Deliberately NOT yaml.safe_dump -- the shipped config's value is largely
    in its comments, and safe_dump would strip them. Only model names are
    interpolated into a fixed template. Output yaml.safe_loads cleanly.
    """
    tiers = assignment["tiers"]
    classifier = assignment["classifier"]
    api_base = "http://localhost:11434"

    lines = []
    lines.append(
        "# Generated by init.py from " + str(assignment["model_count"])
        + " detected Ollama models. Edit freely; re-run `init.py --write` to")
    lines.append(
        "# regenerate (backs up the existing config to config.yaml.bak).")
    if assignment["classifier_is_default"]:
        lines.append(
            "# NOTE: no model <= 1.5B was installed; classifier/critic defaults "
            "to " + _DEFAULT_CLASSIFIER + ".")
        lines.append(
            "#       Pull one for the tripwire role: `ollama pull "
            + _DEFAULT_CLASSIFIER + "`.")
    lines.append("")
    lines.append("model_list:")

    def _entry(alias, model, tier=None):
        block = []
        if tier:
            comment = _tier_comment(assignment, tier)
            if comment:
                block.append(comment.rstrip("\n"))
        block.append("  - model_name: " + alias)
        block.append("    litellm_params:")
        block.append("      model: ollama_chat/" + model)
        block.append("      api_base: " + api_base)
        return "\n".join(block)

    lines.append(_entry("local-s", tiers["S"], "S"))
    lines.append(_entry("local-m", tiers["M"], "M"))
    lines.append(_entry("local-l", tiers["L"], "L"))
    lines.append(_entry("local-xl", tiers["XL"], "XL"))
    lines.append("  # local-auto is the virtual entry the router rewrites per-request.")
    lines.append(_entry("local-auto", tiers["M"]))
    lines.append("")
    lines.append("router_settings:")
    lines.append("  num_retries: 1")
    lines.append("  timeout: 600")
    lines.append("  fallbacks:")
    lines.append("    - local-s: [local-m]")
    lines.append("    - local-m: [local-l]")
    lines.append("    - local-l: [local-xl]")
    lines.append("")
    lines.append("litellm_settings:")
    lines.append("  drop_params: true")
    lines.append("  callbacks: router_hook.tier_router_instance")
    lines.append("")
    lines.append("general_settings:")
    lines.append("  master_key: sk-local-dev")
    lines.append("")
    lines.append("# Routing/critic knobs. Defaults are baked into RouterConfig in")
    lines.append("# router_hook.py; override here only what you want to change.")
    lines.append("route_llm:")
    lines.append("  classifier_model: " + classifier)
    lines.append("  critic_model: " + classifier)
    lines.append("  critic_pass_threshold: 4")
    lines.append("  critic_timeout_s: 30.0")
    lines.append("  soft_pass_tiers: [S, M]")
    lines.append("")
    return "\n".join(lines)


def write_config(out_path, text, do_write, force):
    """Thin I/O: write the generated config.

    do_write False -> write `out_path` as-is (the config.generated.yaml path).
    do_write True  -> write `out_path` (config.yaml), backing up any existing
                      file to `<out_path>.bak` first.
    `force` is reserved for a future non-interactive overwrite guard; the
    default flow already never clobbers config.yaml without a backup.
    """
    if do_write and os.path.exists(out_path):
        bak = out_path + ".bak"
        with open(out_path, "r", encoding="utf-8") as src:
            existing = src.read()
        with open(bak, "w", encoding="utf-8", newline="\n") as dst:
            dst.write(existing)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return out_path


def build_parser():
    p = argparse.ArgumentParser(
        description="Generate a TriageLLM config.yaml from installed Ollama models")
    p.add_argument("--write", action="store_true",
                   help="Write config.yaml directly (backs up existing to config.yaml.bak)")
    p.add_argument("--output", type=str, default=None, metavar="PATH",
                   help="Output path (default: config.generated.yaml, or config.yaml with --write)")
    p.add_argument("--ollama-url", type=str, default=None, metavar="URL",
                   help="Ollama base URL (default: $OLLAMA_BASE_URL or http://localhost:11434)")
    return p


def run(args):
    """Impure orchestrator. Returns process exit code."""
    if args.ollama_url:
        import task_benchmark
        task_benchmark.OLLAMA_BASE = args.ollama_url.rstrip("/")

    try:
        models = asyncio.run(_fetch_tags())
    except Exception as e:   # noqa: BLE001 -- surface any connection failure
        import task_benchmark
        print("[init] could not reach Ollama at " + task_benchmark.OLLAMA_BASE
              + "; is it running? (" + str(e) + ")")
        return 2

    if not models:
        print("[init] no Ollama models found; pull at least one "
              "(e.g. `ollama pull qwen2.5-coder:1.5b`)")
        return 1

    try:
        assignment = assign_tiers(models)
    except ValueError as e:
        print("[init] " + str(e))
        return 1

    text = build_config(assignment)
    out_path = args.output or ("config.yaml" if args.write else "config.generated.yaml")
    written = write_config(out_path, text, do_write=args.write, force=False)

    if args.write:
        print("[init] wrote " + written + " (previous config backed up to "
              + written + ".bak if it existed).")
    else:
        print("[init] wrote " + written + ". Review it, then re-run with "
              "`--write` to apply, or rename it to config.yaml yourself.")
    print("[init] next: run `doctor.py` to validate models-vs-config before starting the proxy.")
    return 0


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
