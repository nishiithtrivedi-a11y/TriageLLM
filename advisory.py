"""TriageLLM capability advisory mode (Issue #18b).

Offline CLI: combines benchmark_results.json priors (#24) with live-ledger
critic-pass-rate corroboration to decide which per-(category, model)
recommendations have graduated, lets the operator sign off, and writes the
endorsed map to advisory_endorsements.json (the future-routing gate).

Advisory ONLY - changes no routing behaviour. Pure graduation engine + thin
I/O + CLI. All runtime strings are ASCII (cp1252-safe). No `from __future__
import annotations` (LiteLLM spec_from_file_location + dataclass crash).
"""
import argparse
import json
import os
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from router_hook import DB_PATH, load_config  # noqa: E402

# Graduation thresholds (the "evidence threshold" - calibrated, not magic).
_MIN_BENCH_RATE = 0.7          # hard graduates here; soft floor
_SOFT_BENCH_RATE = 0.8         # success_rate that graduates a SOFT rec without live
_MIN_BENCH_SAMPLES = 3         # min benchmark prompts behind the rate
_MIN_LIVE_SAMPLES = 5          # min live attempts before live evidence counts
_LIVE_CORROBORATE_FLOOR = 0.6  # live pass-rate that confirms a soft rec
_LIVE_DISAGREE_FLOOR = 0.4     # below this, live VETOES a benchmark rec

# Objective categories: benchmark success is a deterministic check, trustworthy
# without live confirmation.
_HARD_CATEGORIES = {"structured_output", "modification_or_edit"}


@dataclass
class GraduatedRec:
    category: str
    model: str
    status: str                  # graduated|insufficient-benchmark|needs-live-evidence|live-disagreement
    graduated: bool
    bench_success_rate: float
    bench_n_prompts: int
    confidence: str              # "hard" | "soft"
    live_pass_rate: Optional[float]
    live_n: int
    reason: str


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _normalize_model(name: str) -> str:
    """Strip a provider prefix: 'ollama_chat/qwen3-coder:30b' -> 'qwen3-coder:30b'.

    The benchmark keys on the bare Ollama name; the live ledger stores the
    routed model provider-prefixed. Normalize before keying or corroboration
    silently never matches.
    """
    return (name or "").split("/", 1)[-1]


def graduate(priors, live_aggregates):
    """Pure: priors + live aggregates -> {category: GraduatedRec}.

    live_aggregates: {(category, normalized_model): (pass_count, total_count)}.
    Ordered rule: benchmark gate -> live-disagreement veto -> hard/soft.
    """
    results = priors.get("results", {})
    recs = priors.get("recommendations", {})
    out = {}
    for category, rec in recs.items():
        model = rec.get("model")
        success_rate = rec.get("success_rate", 0.0)
        confidence = rec.get("confidence", "soft")
        warning = rec.get("warning")
        n_prompts = results.get(model, {}).get(category, {}).get("n_prompts", 0)

        live = live_aggregates.get((category, model))
        if live:
            live_pass, live_total = live
            live_rate = (live_pass / live_total) if live_total else None
        else:
            live_total, live_rate = 0, None

        # Step 1: benchmark gate.
        if warning == "no-model-passed":
            status, reason, graduated = "insufficient-benchmark", "no-model-passed", False
        elif n_prompts < _MIN_BENCH_SAMPLES:
            status, reason, graduated = "insufficient-benchmark", "benchmark-samples-low", False
        elif success_rate < _MIN_BENCH_RATE:
            status, reason, graduated = "insufficient-benchmark", "benchmark-rate-low", False
        # Step 2: live-disagreement veto (applies to hard and soft alike).
        elif (live_total >= _MIN_LIVE_SAMPLES and live_rate is not None
              and live_rate < _LIVE_DISAGREE_FLOOR):
            status, reason, graduated = "live-disagreement", "live-contradicts-benchmark", False
        # Step 3: hard vs soft weighting.
        elif category in _HARD_CATEGORIES:
            status, reason, graduated = "graduated", "hard-benchmark-pass", True
        elif success_rate >= _SOFT_BENCH_RATE:
            status, reason, graduated = "graduated", "soft-benchmark-strong", True
        elif (live_total >= _MIN_LIVE_SAMPLES and live_rate is not None
              and live_rate >= _LIVE_CORROBORATE_FLOOR):
            status, reason, graduated = "graduated", "soft-live-corroborated", True
        else:
            status, reason, graduated = "needs-live-evidence", "soft-no-live-evidence", False

        out[category] = GraduatedRec(
            category, model, status, graduated, success_rate, n_prompts,
            confidence, live_rate, live_total, reason)
    return out


def build_report(graduated, endorsements, priors_generated_at):
    """Pure: assemble the advisory report structure."""
    ends = endorsements.get("endorsements", {})
    categories = {}
    for category, g in graduated.items():
        endorsed = (category in ends and ends[category].get("model") == g.model)
        categories[category] = {
            "model": g.model,
            "status": g.status,
            "graduated": g.graduated,
            "benchmark": {"success_rate": g.bench_success_rate,
                          "n_prompts": g.bench_n_prompts,
                          "confidence": g.confidence},
            "live": {"pass_rate": g.live_pass_rate, "n": g.live_n},
            "endorsed": endorsed,
            "reason": g.reason,
        }
    return {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "priors_generated_at": priors_generated_at,
        "categories": categories,
    }


def render_report(report, json_mode=False):
    """Pure: render the report as JSON (json_mode) or an ASCII text table."""
    if json_mode:
        return json.dumps(report, indent=2)
    lines = ["TriageLLM advisory (capability graduation)", "=" * 60]
    lines.append("Priors generated: " + str(report.get("priors_generated_at")))
    lines.append("")
    for category, c in report.get("categories", {}).items():
        live = c["live"]
        live_str = ("live n/a" if live["pass_rate"] is None
                    else "live {0:.2f} (n={1})".format(live["pass_rate"], live["n"]))
        mark = "*" if c["endorsed"] else " "
        lines.append(
            "{mark} {cat:<24} {status:<22} {model:<22} "
            "(bench {sr:.2f}, {live})".format(
                mark=mark, cat=category, status=c["status"], model=c["model"],
                sr=c["benchmark"]["success_rate"], live=live_str))
    lines.append("")
    lines.append("(* = endorsed. Sign off graduated rows with --sign-off <category>.)")
    return "\n".join(lines)


def apply_signoff(endorsements, graduated, category):
    """Pure: endorse a graduated rec (snapshotting evidence), or refuse.

    Returns (new_endorsements, message). ENFORCES THE GATE: only a graduated
    rec can be endorsed.
    """
    g = graduated.get(category)
    if g is None:
        return endorsements, "unknown category: " + str(category)
    if not g.graduated:
        return endorsements, (category + " is " + g.status
                              + ", not graduated - cannot endorse")
    source = "benchmark+live" if g.live_n >= _MIN_LIVE_SAMPLES else "benchmark"
    new = dict(endorsements)
    ends = dict(new.get("endorsements", {}))
    ends[category] = {
        "model": g.model,
        "success_rate": g.bench_success_rate,
        "confidence": g.confidence,
        "live_pass_rate": g.live_pass_rate,
        "endorsed_at": _utc_now(),
        "source": source,
    }
    new["endorsements"] = ends
    new["schema_version"] = 1
    new["updated_at"] = _utc_now()
    return new, "endorsed " + category + " -> " + g.model


def load_priors(path):
    """Thin: read benchmark_results.json (raises FileNotFoundError if absent)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_endorsements(path):
    """Thin: read advisory_endorsements.json; missing -> empty structure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"schema_version": 1, "endorsements": {}}


def write_endorsements(path, obj):
    """Atomically write obj as pretty JSON (temp file + os.replace)."""
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


def load_live_aggregates(db_path, pass_threshold, since_ts=None):
    """Impure: one SELECT over the decisions ledger -> live critic-pass rate.

    Returns {(category, normalized_model): (pass_count, total_count)} where
    total counts only CRITIQUED attempts (critic_score is not None), and pass
    counts those with critic_score >= pass_threshold. Rows with NULL
    cap_category (shadow mode off) contribute nothing. When since_ts is set,
    only rows with ts >= since_ts are read (the ts column is referenced ONLY in
    that case, so all-time callers need no ts column).
    """
    query = "SELECT cap_category, attempts_json FROM decisions"
    params = ()
    if since_ts is not None:
        query += " WHERE ts >= ?"
        params = (since_ts,)
    try:
        with sqlite3.connect(db_path) as c:
            rows = c.execute(query, params).fetchall()
    except sqlite3.OperationalError:
        return {}   # missing DB / no decisions table
    agg = {}
    for cap_category, attempts_json in rows:
        if not cap_category or not attempts_json:
            continue
        try:
            attempts = json.loads(attempts_json)
        except (ValueError, TypeError):
            continue
        for a in attempts:
            if not isinstance(a, dict):
                continue
            model = _normalize_model(a.get("model") or "")
            score = a.get("critic_score")
            if not model or score is None:
                continue
            key = (cap_category, model)
            entry = agg.setdefault(key, [0, 0])
            entry[1] += 1
            if score >= pass_threshold:
                entry[0] += 1
    return {k: (v[0], v[1]) for k, v in agg.items()}


def build_parser():
    p = argparse.ArgumentParser(
        description="TriageLLM capability advisory (offline graduation + sign-off)")
    p.add_argument("--sign-off", type=str, default=None, metavar="CATEGORY",
                   help="Endorse the graduated recommendation for CATEGORY")
    p.add_argument("--sign-off-all", action="store_true",
                   help="Endorse every currently-graduated category")
    p.add_argument("--json", action="store_true", help="Emit JSON report")
    p.add_argument("--priors", type=str, default="benchmark_results.json",
                   help="Benchmark priors path (default: benchmark_results.json)")
    p.add_argument("--db", type=str, default=str(DB_PATH),
                   help="Decisions DB path (default: the proxy's router_decisions.sqlite)")
    p.add_argument("--endorsements", type=str, default="advisory_endorsements.json",
                   help="Endorsements path (default: advisory_endorsements.json)")
    return p


def run(args):
    """Impure orchestrator. Returns process exit code."""
    try:
        priors = load_priors(args.priors)
    except FileNotFoundError:
        print("[advisory] no benchmark priors at " + args.priors
              + " - run benchmark.py --tasks first")
        return 1
    pass_threshold = load_config().critic_pass_threshold
    live = load_live_aggregates(args.db, pass_threshold)
    graduated = graduate(priors, live)
    endorsements = load_endorsements(args.endorsements)

    if args.sign_off_all or args.sign_off:
        targets = (list(graduated.keys()) if args.sign_off_all else [args.sign_off])
        for cat in targets:
            endorsements, msg = apply_signoff(endorsements, graduated, cat)
            print("[advisory] " + msg)
        write_endorsements(args.endorsements, endorsements)
        return 0

    report = build_report(graduated, endorsements, priors.get("generated_at"))
    print(render_report(report, json_mode=args.json))
    return 0


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
