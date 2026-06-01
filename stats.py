"""Route LLM stats dashboard.

Reads router_decisions.sqlite and prints a human-readable summary of routing,
escalation, and handoff patterns. Run from the project root:

    .\\.venv\\Scripts\\python.exe stats.py
    .\\.venv\\Scripts\\python.exe stats.py --last 100
    .\\.venv\\Scripts\\python.exe stats.py --since "2 days"
    .\\.venv\\Scripts\\python.exe stats.py --json    # for piping into jq / notebooks
"""
import argparse
import json
import sqlite3
import statistics
import sys
import time

# Force UTF-8 on stdout so our box-drawing characters render on Windows cp1252 consoles.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DB_PATH = Path(__file__).parent / "router_decisions.sqlite"

# All tiers we report on. Order matters for table output.
TIERS = ["S", "M", "L", "XL", "CLOUD"]


@dataclass
class Decision:
    ts: float
    tier: str                    # initial tier (S/M/L/XL)
    critic: int | None           # final critic score, or None
    handoff: bool
    cloud_attempted: bool
    streamed: bool
    attempts: list[dict[str, Any]]   # parsed attempts_json
    # Extended fields (optional; default None/empty for backward-compat with
    # load_decisions which only reads the 7 core columns above).
    requested: str | None = None
    model: str | None = None
    tokens: int | None = None
    score: int | None = None
    signals: str | None = None
    classifier: str | None = None
    escalated_to: str | None = None


def parse_since(s: str) -> float:
    """Parse '2 days', '3 hours', '30 minutes', '1 week' to a unix-timestamp cutoff."""
    parts = s.strip().lower().split()
    if len(parts) != 2:
        raise ValueError(f"--since expected '<n> <unit>', got {s!r}")
    n = float(parts[0])
    unit = parts[1].rstrip("s")
    seconds = {"minute": 60, "hour": 3600, "day": 86400, "week": 604800}.get(unit)
    if seconds is None:
        raise ValueError(f"--since unit must be minute/hour/day/week, got {unit!r}")
    return time.time() - n * seconds


def load_decisions(
    db_path: Path = DB_PATH,
    last: int | None = None,
    since: float | None = None,
) -> list[Decision]:
    """Read decisions ordered oldest-first (so 'latest N' really is the most recent N)."""
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as c:
        cols = {row[1] for row in c.execute("PRAGMA table_info(decisions)")}
        if not cols:
            return []
        # We're robust to older schemas — only read columns that exist.
        select = (
            "ts, tier, critic, "
            f"{'handoff' if 'handoff' in cols else '0 AS handoff'}, "
            f"{'cloud_attempted' if 'cloud_attempted' in cols else '0 AS cloud_attempted'}, "
            f"{'streamed' if 'streamed' in cols else '0 AS streamed'}, "
            f"{'attempts_json' if 'attempts_json' in cols else 'NULL AS attempts_json'}"
        )
        sql = f"SELECT {select} FROM decisions"
        params: list[Any] = []
        if since is not None:
            sql += " WHERE ts >= ?"
            params.append(since)
        sql += " ORDER BY ts DESC"
        if last is not None:
            sql += " LIMIT ?"
            params.append(last)
        rows = c.execute(sql, params).fetchall()

    decisions = []
    for ts, tier, critic, handoff, cloud, streamed, aj in reversed(rows):
        try:
            attempts = json.loads(aj) if aj else []
        except json.JSONDecodeError:
            attempts = []
        decisions.append(Decision(
            ts=float(ts),
            tier=tier or "?",
            critic=critic if critic is not None else None,
            handoff=bool(handoff),
            cloud_attempted=bool(cloud),
            streamed=bool(streamed),
            attempts=attempts,
        ))
    return decisions


# ─── Aggregations (kept pure for testability) ─────────────────────────────

def tier_distribution(decisions: Iterable[Decision]) -> dict[str, int]:
    counts = {t: 0 for t in TIERS}
    for d in decisions:
        if d.tier in counts:
            counts[d.tier] += 1
    return counts


def critic_averages(decisions: Iterable[Decision]) -> dict[str, tuple[float, int]]:
    """Avg critic score per *final* tier (excluding None scores). Returns (avg, n)."""
    by_tier: dict[str, list[int]] = {}
    for d in decisions:
        if not d.attempts:
            continue
        last = d.attempts[-1]
        score = last.get("critic_score")
        if score is None:
            continue
        by_tier.setdefault(last.get("tier", "?"), []).append(int(score))
    return {t: (sum(v) / len(v), len(v)) for t, v in by_tier.items()}


def escalation_stats(decisions: Iterable[Decision]) -> dict[str, Any]:
    """Of non-S decisions, what fraction escalated, and to what avg chain length?"""
    eligible = [d for d in decisions if d.tier != "S" and d.attempts]
    if not eligible:
        return {"eligible": 0, "escalated_pct": 0.0, "avg_chain": 0.0}
    escalated = [d for d in eligible if len(d.attempts) > 1]
    avg_chain = sum(len(d.attempts) for d in eligible) / len(eligible)
    return {
        "eligible": len(eligible),
        "escalated": len(escalated),
        "escalated_pct": 100.0 * len(escalated) / len(eligible),
        "avg_chain": avg_chain,
    }


def token_totals(decisions: Iterable[Decision]) -> dict[str, int]:
    """Total prompt + completion tokens across all attempts."""
    prompt_t = completion_t = 0
    for d in decisions:
        for a in d.attempts:
            prompt_t += int(a.get("prompt_tokens") or 0)
            completion_t += int(a.get("completion_tokens") or 0)
    return {"prompt": prompt_t, "completion": completion_t, "total": prompt_t + completion_t}


def time_totals(decisions: Iterable[Decision]) -> float:
    total = 0.0
    for d in decisions:
        for a in d.attempts:
            total += float(a.get("duration_s") or 0)
    return total


def handoff_stats(decisions: Iterable[Decision]) -> dict[str, Any]:
    decisions = list(decisions)
    n = len(decisions)
    if n == 0:
        return {"total": 0, "handoff": 0, "handoff_pct": 0.0,
                "cloud_attempted": 0, "streamed": 0}
    return {
        "total": n,
        "handoff": sum(1 for d in decisions if d.handoff),
        "handoff_pct": 100.0 * sum(1 for d in decisions if d.handoff) / n,
        "cloud_attempted": sum(1 for d in decisions if d.cloud_attempted),
        "streamed": sum(1 for d in decisions if d.streamed),
    }


# ─── Issue #17 helpers (defined before pretty-printer so render_text can use
# _DEFAULT_PASS_THRESHOLD as a default argument value) ──────────────────────

# Default critic_pass_threshold for the most-failing report (matches the
# default in config.yaml). Override via --pass-threshold. A local constant
# keeps stats.py independent of router_hook (importing load_config would
# trigger the critic warmup on every CLI run).
_DEFAULT_PASS_THRESHOLD = 4


def _normalize_model(name):
    """Strip a provider prefix: 'ollama_chat/qwen2.5-coder:1.5b' -> 'qwen2.5-coder:1.5b'.
    Inline (rather than importing advisory._normalize_model) to keep stats.py
    independent of router_hook -- advisory imports router_hook."""
    return (name or "").split("/", 1)[-1]


def load_one_decision(db_path, ts=None):
    """Return one decision row as a dict (with `attempts` parsed), or None.

    `ts is None` -> latest row in the whole DB (NOT scoped by --last/--since).
    Otherwise: sub-millisecond tolerance (`abs(ts - ?) < 0.0005`) -- which
    handles float round-trips when the user pastes ts from --json output. SQLite
    REAL bit-equality would be fragile.

    Returns None for a missing DB file too (mirrors `load_decisions`'s guard at
    line 73), so a fresh-install operator running `--explain` before any traffic
    gets the friendly miss message instead of an OperationalError traceback.
    """
    if not Path(db_path).exists():
        return None
    with sqlite3.connect(db_path) as c:
        c.row_factory = sqlite3.Row
        if ts is None:
            cur = c.execute(
                "SELECT * FROM decisions ORDER BY ts DESC LIMIT 1")
        else:
            cur = c.execute(
                "SELECT * FROM decisions WHERE abs(ts - ?) < 0.0005 LIMIT 1",
                (ts,))
        row = cur.fetchone()
    if row is None:
        return None
    d = dict(row)
    try:
        d["attempts"] = json.loads(d.get("attempts_json") or "[]")
    except (ValueError, TypeError):
        d["attempts"] = []
    return d


def _p95_idx(sorted_vals):
    """Same index rule as benchmark.py / task_benchmark.py."""
    return sorted_vals[int(0.95 * len(sorted_vals)) - 1]


def _p99_idx(sorted_vals):
    return sorted_vals[int(0.99 * len(sorted_vals)) - 1]


def per_model_latency(decisions):
    """Pure: per (tier, normalized model) latency table.

    Iterates each Decision's `attempts` list; groups `duration_s` by
    (tier, model). Returns a list of dicts:
        {tier, model, n, p50, p95 | None, p99 | None, max}
    sorted by n desc (most-used first). Min-N gates: p95 needs n>=4 (matches
    benchmark.py), p99 needs n>=5. Malformed attempts (missing keys / wrong
    type) are skipped silently.
    """
    buckets = {}
    for d in decisions:
        for a in (d.attempts or []):
            if not isinstance(a, dict):
                continue
            tier = a.get("tier")
            raw_model = a.get("model")
            dur = a.get("duration_s")
            if not tier or not raw_model or dur is None:
                continue
            try:
                dur_f = float(dur)
            except (TypeError, ValueError):
                continue
            key = (tier, _normalize_model(raw_model))
            buckets.setdefault(key, []).append(dur_f)
    out = []
    for (tier, model), durs in buckets.items():
        durs.sort()
        n = len(durs)
        out.append({
            "tier": tier,
            "model": model,
            "n": n,
            "p50": round(statistics.median(durs), 3),
            "p95": round(_p95_idx(durs), 3) if n >= 4 else None,
            "p99": round(_p99_idx(durs), 3) if n >= 5 else None,
            "max": round(max(durs), 3),
        })
    out.sort(key=lambda r: -r["n"])
    return out


def most_failing_models(decisions, pass_threshold, min_n=5, top_k=5):
    """Pure: top failing models by critiqued failure rate.

    For each attempt across `decisions`, count only CRITIQUED attempts
    (`critic_score is not None`); a "failure" is `critic_score < pass_threshold`.
    Attempts where the critic didn't run (tier S, or critic timed out / was
    soft-passed) are excluded -- they're "no opinion," not failures.
    Group by NORMALIZED model name. Gate `critiqued >= min_n`. Sort by
    failure_rate desc, return top_k.

    Returns list of dicts: {model, critiqued, failed, failure_rate}.
    """
    counts = {}   # model -> [failed, critiqued]
    for d in decisions:
        for a in (d.attempts or []):
            if not isinstance(a, dict):
                continue
            score = a.get("critic_score")
            if score is None:
                continue   # uncritiqued -> not a failure
            model = _normalize_model(a.get("model") or "")
            if not model:
                continue
            entry = counts.setdefault(model, [0, 0])
            entry[1] += 1
            if score < pass_threshold:
                entry[0] += 1
    out = []
    for model, (failed, critiqued) in counts.items():
        if critiqued < min_n or failed == 0:
            continue
        out.append({
            "model": model,
            "critiqued": critiqued,
            "failed": failed,
            "failure_rate": round(failed / critiqued, 4),
        })
    out.sort(key=lambda r: -r["failure_rate"])
    return out[:top_k]


def render_explain(row, attempts):
    """Pure: ASCII drill-down of one decision. `row` is the dict from
    load_one_decision; `attempts` is the parsed attempts list.

    Capability section is shown only when row has cap_category set. All field
    reads use .get(k, default) so older rows missing #29 cost fields render
    '-' instead of raising KeyError.
    """
    iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row.get("ts", 0.0)))
    lines = []
    lines.append("=" * 72)
    lines.append("Decision at {iso} (ts={ts})".format(
        iso=iso, ts=row.get("ts")))
    lines.append("=" * 72)
    lines.append("")
    lines.append("Routing")
    lines.append("-" * 72)
    lines.append("  requested:  " + str(row.get("requested", "-")))
    lines.append("  initial:    " + str(row.get("tier", "-")) + " / "
                 + _normalize_model(row.get("model") or "-"))
    lines.append("  classifier: " + str(row.get("classifier", "-"))
                 + "  signals=" + str(row.get("signals") or "-"))
    lines.append("")
    if row.get("cap_category"):
        lines.append("Capability")
        lines.append("-" * 72)
        lines.append("  cap_category:         " + str(row.get("cap_category")))
        lines.append("  cap_recommended_tier: "
                     + str(row.get("cap_recommended_tier", "-")))
        lines.append("  cap_agrees_with_tier: "
                     + str(row.get("cap_agrees_with_tier", "-")))
        lines.append("")
    lines.append("Verdict / flags")
    lines.append("-" * 72)
    lines.append("  critic:           " + str(row.get("critic", "-")) + "/5")
    lines.append("  escalated_to:     " + str(row.get("escalated_to") or "no"))
    lines.append("  cloud_attempted:  " + str(bool(row.get("cloud_attempted"))))
    lines.append("  handoff:          " + str(bool(row.get("handoff"))))
    lines.append("  streamed:         " + str(bool(row.get("streamed"))))
    lines.append("")
    lines.append("Attempts (" + str(len(attempts)) + ")")
    lines.append("-" * 72)
    for i, a in enumerate(attempts, 1):
        a = a if isinstance(a, dict) else {}
        p = a.get("prompt_tokens", "-")
        ct = a.get("completion_tokens", "-")
        dur = a.get("duration_s", "-")
        critic = a.get("critic_score", "-")
        was_warm = a.get("was_warm", "-")
        vram = a.get("vram_mb", "-")
        cost = a.get("cost_usd", "-")
        preview = (a.get("preview") or "")[:200]
        lines.append("  {i}. {tier:<3} | {model:<32} | tokens={p}/{ct} | "
                     "{dur}s | critic={critic} | warm={warm} | "
                     "vram={vram} | cost_usd={cost}".format(
                         i=i, tier=str(a.get("tier", "?")),
                         model=_normalize_model(a.get("model") or "?"),
                         p=p, ct=ct, dur=dur, critic=critic,
                         warm=was_warm, vram=vram, cost=cost))
        if preview:
            lines.append('     preview: "' + preview + '"')
    return "\n".join(lines)


# ─── Pretty-printer ───────────────────────────────────────────────────────

BAR_WIDTH = 30


def _bar(pct: float, width: int = BAR_WIDTH) -> str:
    filled = int(round(pct / 100.0 * width))
    return "█" * filled + "░" * (width - filled)


def render_text(decisions: list[Decision],
                pass_threshold: int = _DEFAULT_PASS_THRESHOLD) -> str:
    n = len(decisions)
    if n == 0:
        return "No decisions recorded yet. Send a few requests through the proxy first."

    out: list[str] = []
    span_start = time.strftime("%Y-%m-%d %H:%M", time.localtime(decisions[0].ts))
    span_end   = time.strftime("%Y-%m-%d %H:%M", time.localtime(decisions[-1].ts))
    out.append("═" * 72)
    out.append(f"  Route-LLM stats — {n} decision(s), {span_start}  →  {span_end}")
    out.append("═" * 72)
    out.append("")

    # 1. Tier distribution
    dist = tier_distribution(decisions)
    total_routed = sum(dist[t] for t in ("S", "M", "L", "XL"))
    out.append("Initial tier distribution")
    out.append("─" * 72)
    for t in ("S", "M", "L", "XL"):
        count = dist[t]
        pct = (100.0 * count / total_routed) if total_routed else 0.0
        out.append(f"  {t:<4} {_bar(pct)} {count:>4} ({pct:5.1f}%)")
    out.append("")

    # 2. Critic averages (by final tier)
    crit = critic_averages(decisions)
    out.append("Critic scores (averaged per final tier, excludes uncriticed tier-S)")
    out.append("─" * 72)
    if not crit:
        out.append("  (no critic scores recorded yet)")
    else:
        for t in TIERS:
            if t in crit:
                avg, k = crit[t]
                out.append(f"  {t:<5} {avg:.2f}/5   ({k} sample{'s' if k != 1 else ''})")
    out.append("")

    # 3. Escalation
    esc = escalation_stats(decisions)
    out.append("Escalation")
    out.append("─" * 72)
    if esc["eligible"] == 0:
        out.append("  (no eligible decisions — all routed to tier S or no attempts logged)")
    else:
        out.append(f"  Eligible (non-S, with ledger): {esc['eligible']}")
        out.append(f"  Escalated (chain > 1 step):    {esc['escalated']} ({esc['escalated_pct']:.1f}%)")
        out.append(f"  Average chain length:          {esc['avg_chain']:.2f}")
    out.append("")

    # 4. Handoff & cloud
    ho = handoff_stats(decisions)
    out.append("Handoff & cloud")
    out.append("─" * 72)
    out.append(f"  Handoff messages emitted:      {ho['handoff']} ({ho['handoff_pct']:.1f}%)")
    out.append(f"  Cloud escalation attempted:    {ho['cloud_attempted']}")
    out.append(f"  Streaming responses:           {ho['streamed']}")
    out.append("")

    # 5. Totals
    tok = token_totals(decisions)
    tot_s = time_totals(decisions)
    out.append("Resource totals")
    out.append("─" * 72)
    out.append(f"  Tokens (prompt + completion):  {tok['total']:,}  "
               f"({tok['prompt']:,} prompt, {tok['completion']:,} completion)")
    out.append(f"  Local compute time:            {tot_s:.1f}s  "
               f"({tot_s / 60:.1f} min)")
    out.append("")

    # 6. Per-model latency P50/P95/P99 (#17)
    lat = per_model_latency(decisions)
    out.append("Per-model latency")
    out.append("-" * 72)
    if not lat:
        out.append("  (no data in window)")
    else:
        out.append("  tier  model                              n     p50    p95    p99    max")
        for r in lat:
            p95 = "-" if r["p95"] is None else "{0:.2f}".format(r["p95"])
            p99 = "-" if r["p99"] is None else "{0:.2f}".format(r["p99"])
            out.append("  {tier:<4}  {model:<32}  {n:>4}  {p50:>5.2f}  {p95:>5}  {p99:>5}  {mx:>5.2f}".format(
                tier=r["tier"], model=r["model"][:32], n=r["n"],
                p50=r["p50"], p95=p95, p99=p99, mx=r["max"]))
    out.append("")

    # 7. Most-failing models (#17)
    fail = most_failing_models(decisions, pass_threshold=pass_threshold)
    out.append("Most-failing models (critic < " + str(pass_threshold) + ")")
    out.append("-" * 72)
    if not fail:
        out.append("  (no critiqued attempts in window)")
    else:
        for r in fail:
            pct = 100.0 * r["failure_rate"]
            out.append("  {model:<32}  failed {f}/{c} ({pct:.1f}%)".format(
                model=r["model"][:32], f=r["failed"], c=r["critiqued"], pct=pct))
    out.append("")

    # 8. Recent activity
    out.append("Recent activity (last 10)")
    out.append("─" * 72)
    out.append("  when             tier  steps  critic  flags")
    for d in decisions[-10:][::-1]:
        when = time.strftime("%m-%d %H:%M:%S", time.localtime(d.ts))
        flags = []
        if d.streamed: flags.append("stream")
        if d.cloud_attempted: flags.append("cloud")
        if d.handoff: flags.append("HANDOFF")
        flag_str = ",".join(flags) if flags else "—"
        critic_str = f"{d.critic}/5" if d.critic is not None else "—"
        out.append(f"  {when}  {d.tier:<4}  {len(d.attempts):>3}    {critic_str:<6} {flag_str}")
    out.append("")
    return "\n".join(out)


def load_capability_rows(
    db_path: Path = DB_PATH,
    last: int | None = None,
    since: float | None = None,
) -> list[dict]:
    """Return raw dicts containing tier + the 6 capability columns
    `render_capability` needs (cap_category, cap_recommended_tier,
    cap_classifier_used, cap_confidence, cap_pack, cap_agrees_with_tier).
    cap_reason_code + cap_signals are intentionally omitted -- they belong
    in a future per-decision detail view, not the aggregate dashboard.

    Returns an empty list if the DB doesn't exist or the capability columns
    haven't been migrated in yet (older schema).
    """
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as c:
        c.row_factory = sqlite3.Row
        cols = {row[1] for row in c.execute("PRAGMA table_info(decisions)")}
        if not cols or "cap_category" not in cols:
            return []
        cap_cols = [
            "ts", "tier",
            "cap_category", "cap_recommended_tier", "cap_classifier_used",
            "cap_confidence", "cap_pack", "cap_agrees_with_tier",
        ]
        sql = f"SELECT {', '.join(cap_cols)} FROM decisions"
        params: list[Any] = []
        if since is not None:
            sql += " WHERE ts >= ?"
            params.append(since)
        sql += " ORDER BY ts DESC"
        if last is not None:
            sql += " LIMIT ?"
            params.append(last)
        return [dict(row) for row in c.execute(sql, params).fetchall()]


def render_capability(rows: list[dict]) -> None:
    """Pure-SELECT view: per-category breakdown + XL-avoidance opportunity rate."""
    from collections import Counter
    if not rows:
        print("No decisions yet.")
        return
    with_cap = [r for r in rows if r.get("cap_category")]
    if not with_cap:
        print("Capability routing has no data yet. Enable capability_routing.enabled "
              "and let the proxy run for some real requests, then re-run.")
        return
    print(f"Capability rows: {len(with_cap)} of {len(rows)} total decisions")
    cats = Counter(r["cap_category"] for r in with_cap)
    print("\nCategory counts:")
    for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
        sub = [r for r in with_cap if r["cap_category"] == cat]
        agree = sum(1 for r in sub if r.get("cap_agrees_with_tier") == 1)
        print(f"  {cat:<24} {n:>4}  (agree-with-tier: {agree}/{n})")
    xl_actual = [r for r in with_cap if r.get("tier") == "XL"]
    xl_avoidable = [r for r in xl_actual if r.get("cap_recommended_tier") in {"S", "M", "L"}]
    if xl_actual:
        print(f"\nXL-avoidance opportunity: {len(xl_avoidable)}/{len(xl_actual)} "
              f"({100*len(xl_avoidable)/len(xl_actual):.0f}%) XL routings the "
              f"capability router would have avoided")
    else:
        print("\nNo XL routings observed yet.")


def render_json(decisions: list[Decision],
                pass_threshold: int = _DEFAULT_PASS_THRESHOLD) -> str:
    return json.dumps({
        "total": len(decisions),
        "tier_distribution": tier_distribution(decisions),
        "critic_averages": {
            t: {"avg": avg, "n": n}
            for t, (avg, n) in critic_averages(decisions).items()
        },
        "escalation": escalation_stats(decisions),
        "tokens": token_totals(decisions),
        "wall_seconds": time_totals(decisions),
        "handoff": handoff_stats(decisions),
        "per_model_latency": per_model_latency(decisions),
        "most_failing_models": most_failing_models(decisions, pass_threshold=pass_threshold),
        "pass_threshold": pass_threshold,
    }, indent=2)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Route-LLM routing/critic/escalation stats")
    p.add_argument("--db", type=Path, default=DB_PATH, help="Path to router_decisions.sqlite")
    p.add_argument("--last", type=int, default=None, help="Only consider the last N decisions")
    p.add_argument("--since", type=str, default=None,
                   help="Only consider decisions since '<n> <unit>' (e.g. '2 days', '3 hours')")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of pretty text")
    p.add_argument("--capability", action="store_true",
                   help="Show capability-routing shadow metrics (requires capability_routing enabled)")
    p.add_argument("--explain", type=str, nargs="?", const="LATEST", default=None,
                   metavar="TS",
                   help="Drill into a decision: ts from --json/table, or no arg for the latest")
    p.add_argument("--pass-threshold", type=int, default=_DEFAULT_PASS_THRESHOLD,
                   help="Critic score below this counts as a failure for the most-failing "
                        "report (default %(default)d = config.yaml's critic_pass_threshold)")
    args = p.parse_args(argv)

    # --explain is its own mode (per-row drill-down).
    if args.explain is not None:
        if args.explain == "LATEST":
            ts = None
        else:
            try:
                ts = float(args.explain)
            except ValueError:
                print("[stats] --explain expects a numeric ts (e.g. 1700000000.123) "
                      "or no value for the latest", file=sys.stderr)
                return 1
        row = load_one_decision(args.db, ts=ts)
        if row is None:
            label = "LATEST" if ts is None else str(ts)
            print("[stats] no decision found at ts=" + label
                  + " (copy the exact ts from --json output)", file=sys.stderr)
            return 1
        print(render_explain(row, row.get("attempts", [])))
        return 0

    since_ts = parse_since(args.since) if args.since else None

    if args.capability:
        cap_rows = load_capability_rows(args.db, last=args.last, since=since_ts)
        render_capability(cap_rows)
        return 0

    decisions = load_decisions(args.db, last=args.last, since=since_ts)
    out = (render_json(decisions, pass_threshold=args.pass_threshold)
           if args.json
           else render_text(decisions, pass_threshold=args.pass_threshold))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
