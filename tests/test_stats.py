"""Aggregations + DB loading for stats.py."""
import json
import sqlite3
import time
from pathlib import Path

import pytest

import stats
from stats import (
    Decision,
    critic_averages,
    escalation_stats,
    handoff_stats,
    load_decisions,
    parse_since,
    render_json,
    render_text,
    tier_distribution,
    token_totals,
)


def _attempt(tier: str, score: int | None = None, prompt: int = 100,
             completion: int = 200, dur: float = 1.0) -> dict:
    return {
        "tier": tier,
        "model": f"ollama_chat/{tier.lower()}",
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "duration_s": dur,
        "critic_score": score,
        "preview": "",
    }


def _decision(tier: str, attempts: list[dict], **kw) -> Decision:
    return Decision(
        ts=kw.get("ts", time.time()),
        tier=tier,
        critic=attempts[-1].get("critic_score") if attempts else None,
        handoff=kw.get("handoff", False),
        cloud_attempted=kw.get("cloud_attempted", False),
        streamed=kw.get("streamed", False),
        attempts=attempts,
    )


def test_tier_distribution_counts_initial_tiers() -> None:
    ds = [
        _decision("S",  [_attempt("S")]),
        _decision("S",  [_attempt("S")]),
        _decision("L",  [_attempt("L", score=5)]),
        _decision("XL", [_attempt("XL", score=4)]),
    ]
    assert tier_distribution(ds) == {"S": 2, "M": 0, "L": 1, "XL": 1, "CLOUD": 0}


def test_critic_averages_uses_final_tier_score() -> None:
    """A request that escalated L→XL averages under XL, not under L."""
    ds = [
        _decision("L", [_attempt("L", score=2), _attempt("XL", score=4)]),
        _decision("L", [_attempt("L", score=5)]),
        _decision("L", [_attempt("L", score=3), _attempt("XL", score=3)]),
    ]
    out = critic_averages(ds)
    # Final tier is XL twice (scores 4, 3 → avg 3.5) and L once (score 5)
    assert out["XL"] == (3.5, 2)
    assert out["L"]  == (5.0, 1)


def test_critic_averages_skips_none_scores() -> None:
    ds = [_decision("S", [_attempt("S", score=None)])]
    assert critic_averages(ds) == {}


def test_escalation_stats_excludes_tier_S() -> None:
    ds = [
        _decision("S",  [_attempt("S")]),                         # excluded
        _decision("L",  [_attempt("L", 5)]),                      # eligible, no escalation
        _decision("L",  [_attempt("L", 2), _attempt("XL", 4)]),   # eligible, escalated
        _decision("XL", [_attempt("XL", 2), _attempt("CLOUD", 5)]),  # eligible, escalated
    ]
    out = escalation_stats(ds)
    assert out["eligible"] == 3
    assert out["escalated"] == 2
    assert out["escalated_pct"] == pytest.approx(66.666, rel=1e-3)
    assert out["avg_chain"] == pytest.approx((1 + 2 + 2) / 3)


def test_escalation_stats_empty() -> None:
    assert escalation_stats([]) == {
        "eligible": 0, "escalated_pct": 0.0, "avg_chain": 0.0,
    }


def test_token_and_time_totals_sum_across_attempts() -> None:
    ds = [
        _decision("L", [
            _attempt("L", 2, prompt=50, completion=100, dur=1.5),
            _attempt("XL", 3, prompt=50, completion=150, dur=4.5),
        ]),
    ]
    assert token_totals(ds) == {"prompt": 100, "completion": 250, "total": 350}
    from stats import time_totals
    assert time_totals(ds) == pytest.approx(6.0)


def test_handoff_stats_counts_flags() -> None:
    ds = [
        _decision("L", [_attempt("L", 5)]),                                  # normal
        _decision("XL", [_attempt("XL", 2)], handoff=True),                  # handoff
        _decision("XL", [_attempt("XL", 2), _attempt("CLOUD", 5)],
                  cloud_attempted=True),                                     # cloud succeeded
        _decision("L", [_attempt("L", 5)], streamed=True),                   # stream
    ]
    out = handoff_stats(ds)
    assert out["total"] == 4
    assert out["handoff"] == 1
    assert out["handoff_pct"] == 25.0
    assert out["cloud_attempted"] == 1
    assert out["streamed"] == 1


def test_handoff_stats_empty() -> None:
    out = handoff_stats([])
    assert out["total"] == 0
    assert out["handoff"] == 0
    assert out["handoff_pct"] == 0.0


def test_render_text_handles_empty() -> None:
    msg = render_text([])
    assert "No decisions recorded" in msg


def test_render_text_includes_all_sections() -> None:
    ds = [
        _decision("L", [_attempt("L", 2), _attempt("XL", 4)], handoff=False),
    ]
    out = render_text(ds)
    for section in [
        "Initial tier distribution",
        "Critic scores",
        "Escalation",
        "Handoff & cloud",
        "Resource totals",
        "Recent activity",
    ]:
        assert section in out


def test_render_json_is_valid_and_has_expected_keys() -> None:
    ds = [_decision("L", [_attempt("L", 5)])]
    obj = json.loads(render_json(ds))
    assert obj["total"] == 1
    assert "tier_distribution" in obj
    assert "critic_averages" in obj
    assert "escalation" in obj
    assert "tokens" in obj
    assert "handoff" in obj


def test_parse_since_units() -> None:
    now = time.time()
    assert parse_since("2 days")    == pytest.approx(now - 2 * 86400,  abs=2)
    assert parse_since("3 hours")   == pytest.approx(now - 3 * 3600,   abs=2)
    assert parse_since("30 minutes")== pytest.approx(now - 30 * 60,    abs=2)
    assert parse_since("1 week")    == pytest.approx(now - 604800,     abs=2)


def test_parse_since_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        parse_since("yesterday")
    with pytest.raises(ValueError):
        parse_since("5 fortnights")


def test_load_decisions_empty_when_db_missing(tmp_path: Path) -> None:
    assert load_decisions(tmp_path / "nope.sqlite") == []


def test_load_decisions_reads_recent_first_and_orders_oldest_first(tmp_path: Path) -> None:
    db = tmp_path / "router_decisions.sqlite"
    with sqlite3.connect(db) as c:
        c.execute("""
            CREATE TABLE decisions (
                ts REAL, requested TEXT, tier TEXT, model TEXT,
                tokens INTEGER, score INTEGER, signals TEXT,
                classifier TEXT, critic INTEGER, escalated_to TEXT,
                attempts_json TEXT, cloud_attempted INTEGER,
                handoff INTEGER, streamed INTEGER
            )
        """)
        rows = []
        for i, tier in enumerate(["S", "M", "L", "XL"]):
            attempts = [_attempt(tier, score=5 if tier != "S" else None)]
            rows.append((1000.0 + i, "local-auto", tier, "ollama/x",
                         100, 0, "", "rules", 5, None,
                         json.dumps(attempts), 0, 0, 0))
        c.executemany(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )

    decisions = load_decisions(db)
    assert len(decisions) == 4
    # Ordered oldest-first after the reversed() in load_decisions
    assert [d.tier for d in decisions] == ["S", "M", "L", "XL"]


def test_load_decisions_respects_last_limit(tmp_path: Path) -> None:
    db = tmp_path / "r.sqlite"
    with sqlite3.connect(db) as c:
        c.execute("""CREATE TABLE decisions (ts REAL, tier TEXT, critic INTEGER,
                     handoff INTEGER, cloud_attempted INTEGER, streamed INTEGER,
                     attempts_json TEXT)""")
        for i in range(5):
            c.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?)",
                      (1000 + i, "L", 5, 0, 0, 0, "[]"))
    # Only the 2 newest (ts 1003, 1004) come back
    decisions = load_decisions(db, last=2)
    assert len(decisions) == 2
    assert decisions[0].ts == 1003.0   # oldest of the 2 newest, first after reversal
    assert decisions[1].ts == 1004.0


def test_load_decisions_handles_legacy_schema_without_new_columns(tmp_path: Path) -> None:
    """Pre-ledger DBs lacked handoff/cloud/streamed/attempts_json columns."""
    db = tmp_path / "old.sqlite"
    with sqlite3.connect(db) as c:
        c.execute("""CREATE TABLE decisions (ts REAL, tier TEXT, critic INTEGER)""")
        c.execute("INSERT INTO decisions VALUES (?, ?, ?)", (1000.0, "M", 4))
    decisions = load_decisions(db)
    assert len(decisions) == 1
    assert decisions[0].tier == "M"
    assert decisions[0].critic == 4
    # Missing columns default to falsy
    assert decisions[0].handoff is False
    assert decisions[0].attempts == []


def test_cli_runs_against_real_db() -> None:
    """Smoke test: the script's main() actually runs against the repo DB."""
    # Use --json so output is small + parseable
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = stats.main(["--json", "--last", "5"])
    assert rc == 0
    parsed = json.loads(buf.getvalue())
    assert "total" in parsed
