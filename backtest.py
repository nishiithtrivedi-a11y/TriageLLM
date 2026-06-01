"""TriageLLM advisory backtest (Issue #28).

Offline, read-only CLI: validates advisory_endorsements.json (#18b) against
live ledger history, flagging DRIFT from each endorsement's frozen sign-off
snapshot. Reuses advisory.load_live_aggregates (generalized with since_ts).
Never mutates endorsements; changes no routing behaviour.

All runtime strings are ASCII (cp1252-safe). No `from __future__ import
annotations` (LiteLLM spec_from_file_location + dataclass crash).
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from advisory import (  # noqa: E402
    load_endorsements, load_live_aggregates, _normalize_model,
    _MIN_LIVE_SAMPLES, _LIVE_DISAGREE_FLOOR)
from router_hook import DB_PATH, load_config  # noqa: E402

# A live pass-rate drop of at least this much below the sign-off baseline flags
# as "drifted" (a named knob, not a magic number).
_DRIFT_MARGIN = 0.15

_UNITS = {"m": 60, "h": 3600, "d": 86400}


@dataclass
class BacktestVerdict:
    category: str
    model: str
    verdict: str               # upheld|drifted|contradicted|insufficient-data
    expected_rate: float
    live_rate: Optional[float]
    live_n: int
    drift: Optional[float]
    endorsed_at: Optional[str]
    reason: str


def _parse_since(s):
    """'7d'/'24h'/'90m' -> cutoff epoch (now - delta); None if s is None.

    Raises ValueError on malformed input (so a typo is loud, never a silent
    fall back to all-time).
    """
    if s is None:
        return None
    s = s.strip().lower()
    if len(s) < 2 or s[-1] not in _UNITS or not s[:-1].isdigit():
        raise ValueError("invalid --since '" + s
                         + "' (use forms like 7d, 24h, 90m)")
    return time.time() - int(s[:-1]) * _UNITS[s[-1]]


def evaluate_endorsements(endorsements, live_aggregates):
    """Pure: {category: BacktestVerdict}.

    baseline (expected_rate) = snapshot live_pass_rate if non-null else the
    benchmark success_rate. drift = live_rate_now - expected_rate. Verdicts:
    insufficient-data (live_n < min) -> contradicted (live < disagree floor) ->
    drifted (drop >= _DRIFT_MARGIN) -> upheld.
    """
    ends = endorsements.get("endorsements", {})
    out = {}
    for category, e in ends.items():
        model = e.get("model")
        snap_live = e.get("live_pass_rate")
        expected = snap_live if snap_live is not None else e.get("success_rate", 0.0)
        endorsed_at = e.get("endorsed_at")

        live = live_aggregates.get((category, _normalize_model(model or "")))
        if live:
            live_pass, live_total = live
            live_rate = (live_pass / live_total) if live_total else None
        else:
            live_total, live_rate = 0, None

        if live_total < _MIN_LIVE_SAMPLES or live_rate is None:
            verdict, reason, drift = "insufficient-data", "live-samples-low", None
        elif live_rate < _LIVE_DISAGREE_FLOOR:
            verdict, reason, drift = ("contradicted", "below-disagree-floor",
                                      round(live_rate - expected, 4))
        elif (live_rate - expected) <= -_DRIFT_MARGIN:
            verdict, reason, drift = ("drifted", "dropped-from-baseline",
                                      round(live_rate - expected, 4))
        else:
            verdict, reason, drift = ("upheld", "live-holds",
                                      round(live_rate - expected, 4))

        out[category] = BacktestVerdict(
            category, model, verdict, round(expected, 4),
            (round(live_rate, 4) if live_rate is not None else None),
            live_total, drift, endorsed_at, reason)
    return out


def build_report(verdicts, endorsements_updated_at, window_label):
    """Pure: assemble the backtest report structure + summary counts."""
    summary = {"upheld": 0, "drifted": 0, "contradicted": 0,
               "insufficient-data": 0, "total": 0}
    categories = {}
    for category, v in verdicts.items():
        summary[v.verdict] = summary.get(v.verdict, 0) + 1
        summary["total"] += 1
        categories[category] = {
            "model": v.model, "verdict": v.verdict,
            "expected_rate": v.expected_rate, "live_rate": v.live_rate,
            "live_n": v.live_n, "drift": v.drift,
            "endorsed_at": v.endorsed_at, "reason": v.reason,
        }
    return {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "window": window_label,
        "endorsements_updated_at": endorsements_updated_at,
        "summary": summary,
        "categories": categories,
    }


def render_report(report, json_mode=False):
    """Pure: render the report as JSON (json_mode) or an ASCII text table."""
    if json_mode:
        return json.dumps(report, indent=2)
    s = report.get("summary", {})
    lines = ["TriageLLM advisory backtest", "=" * 60]
    lines.append("Window: " + str(report.get("window"))
                 + "   Endorsed: " + str(report.get("endorsements_updated_at")))
    lines.append("{up}/{tot} upheld | {dr} drifted | {co} contradicted | "
                 "{ins} insufficient-data".format(
                     up=s.get("upheld", 0), tot=s.get("total", 0),
                     dr=s.get("drifted", 0), co=s.get("contradicted", 0),
                     ins=s.get("insufficient-data", 0)))
    lines.append("")
    for category, c in report.get("categories", {}).items():
        live = ("n/a" if c["live_rate"] is None
                else "{0:.2f} (n={1})".format(c["live_rate"], c["live_n"]))
        drift = "" if c["drift"] is None else " drift {0:+.2f}".format(c["drift"])
        lines.append(
            "  {cat:<24} {verdict:<18} {model:<22} "
            "(expected {exp:.2f} -> live {live}{drift})".format(
                cat=category, verdict=c["verdict"], model=c["model"],
                exp=c["expected_rate"], live=live, drift=drift))
    return "\n".join(lines)


def build_parser():
    p = argparse.ArgumentParser(
        description="TriageLLM advisory backtest (endorsement drift vs live history)")
    p.add_argument("--since", type=str, default=None, metavar="WINDOW",
                   help="Only ledger decisions within WINDOW (e.g. 7d, 24h, 90m)")
    p.add_argument("--json", action="store_true", help="Emit JSON report")
    p.add_argument("--endorsements", type=str, default="advisory_endorsements.json",
                   help="Endorsements path (default: advisory_endorsements.json)")
    p.add_argument("--db", type=str, default=str(DB_PATH),
                   help="Decisions DB path (default: the proxy's router_decisions.sqlite)")
    return p


def run(args):
    """Impure orchestrator. Returns process exit code."""
    try:
        since_ts = _parse_since(args.since)
    except ValueError as e:
        print("[backtest] " + str(e))
        return 2
    endorsements = load_endorsements(args.endorsements)
    if not endorsements.get("endorsements"):
        print("[backtest] no endorsements to backtest - run "
              "advisory.py --sign-off first")
        return 0
    pass_threshold = load_config().critic_pass_threshold
    live = load_live_aggregates(args.db, pass_threshold, since_ts=since_ts)
    verdicts = evaluate_endorsements(endorsements, live)
    window_label = args.since if args.since else "all-time"
    report = build_report(verdicts, endorsements.get("updated_at"), window_label)
    print(render_report(report, json_mode=args.json))
    return 0


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
