"""Issue #17: stats.py observability — explain, per-model latency, most-failing."""
import json as _json
import sqlite3
import time

import stats


def test_default_pass_threshold_is_four():
    # Matches config.yaml's critic_pass_threshold default; --pass-threshold overrides.
    assert stats._DEFAULT_PASS_THRESHOLD == 4


def test_normalize_model_strips_provider_prefix():
    assert stats._normalize_model("ollama_chat/qwen2.5-coder:1.5b") == "qwen2.5-coder:1.5b"
    assert stats._normalize_model("qwen2.5-coder:1.5b") == "qwen2.5-coder:1.5b"
    assert stats._normalize_model(None) == ""
    assert stats._normalize_model("") == ""


def _make_decisions_db(db_path, rows):
    """Build a minimal `decisions` table with the columns load_one_decision needs."""
    with sqlite3.connect(db_path) as c:
        c.execute(
            "CREATE TABLE decisions ("
            "ts REAL, requested TEXT, tier TEXT, model TEXT, tokens INTEGER, "
            "score INTEGER, signals TEXT, classifier TEXT, critic INTEGER, "
            "escalated_to TEXT, attempts_json TEXT, cloud_attempted INTEGER, "
            "handoff INTEGER, streamed INTEGER, cap_category TEXT)")
        for r in rows:
            c.execute(
                "INSERT INTO decisions VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (r.get("ts"), r.get("requested", "local-auto"), r.get("tier", "M"),
                 r.get("model", "ollama_chat/m-model"), r.get("tokens", 100),
                 r.get("score", 50), r.get("signals", ""), r.get("classifier", "rules"),
                 r.get("critic"), r.get("escalated_to"),
                 r.get("attempts_json", "[]"), r.get("cloud_attempted", 0),
                 r.get("handoff", 0), r.get("streamed", 0), r.get("cap_category")))


def test_load_one_decision_latest_when_ts_none(tmp_path):
    db = tmp_path / "r.sqlite"
    _make_decisions_db(db, [{"ts": 1000.0}, {"ts": 2000.0}, {"ts": 1500.0}])
    row = stats.load_one_decision(db, ts=None)
    assert row is not None
    assert row["ts"] == 2000.0   # latest by ts


def test_load_one_decision_exact_ts(tmp_path):
    db = tmp_path / "r.sqlite"
    _make_decisions_db(db, [{"ts": 1000.0}, {"ts": 2000.123}])
    row = stats.load_one_decision(db, ts=2000.123)
    assert row is not None
    assert row["ts"] == 2000.123


def test_load_one_decision_within_tolerance(tmp_path):
    # 0.0005s sub-millisecond tolerance handles float round-trips when the user
    # pastes ts from --json output. (The original plan said 1ms but the test
    # boundary at 0.0006 forces the implementation to be tighter than that.)
    db = tmp_path / "r.sqlite"
    _make_decisions_db(db, [{"ts": 2000.1234}])
    assert stats.load_one_decision(db, ts=2000.1235) is not None  # within 0.0005
    assert stats.load_one_decision(db, ts=2000.1240) is None      # outside 0.0005


def test_load_one_decision_missing(tmp_path):
    db = tmp_path / "r.sqlite"
    _make_decisions_db(db, [{"ts": 1000.0}])
    assert stats.load_one_decision(db, ts=9999.0) is None


def test_load_one_decision_missing_db_file(tmp_path):
    # Fresh-install case: the DB file doesn't exist yet. Must return None (so
    # main prints the friendly miss message + exits 1) instead of raising
    # sqlite3.OperationalError. Mirrors load_decisions' guard at line 73.
    assert stats.load_one_decision(tmp_path / "nope.sqlite", ts=None) is None
    assert stats.load_one_decision(tmp_path / "nope.sqlite", ts=1000.0) is None


def test_load_one_decision_parses_attempts_json(tmp_path):
    db = tmp_path / "r.sqlite"
    attempts = [{"tier": "S", "model": "ollama_chat/x", "duration_s": 1.0}]
    _make_decisions_db(db, [{"ts": 1000.0, "attempts_json": _json.dumps(attempts)}])
    row = stats.load_one_decision(db, ts=1000.0)
    assert row["attempts"] == attempts


from stats import Decision  # noqa: E402


def _dec(ts=1000.0, attempts=()):
    return Decision(
        ts=ts, requested="local-auto", tier="M", model="ollama_chat/m-model",
        tokens=100, score=50, signals="", classifier="rules", critic=None,
        escalated_to=None, attempts=list(attempts), cloud_attempted=False,
        handoff=False, streamed=False,
    )


def test_per_model_latency_full_percentiles_when_n_large():
    # n=5 attempts on (M, deepseek): durations 1,2,3,4,10 -> p50=3, max=10.
    # p95 = sorted[int(0.95*5)-1] = sorted[3] = 4. p99 = sorted[int(0.99*5)-1] = sorted[3] = 4.
    atts = [{"tier": "M", "model": "ollama_chat/deepseek-coder-v2:16b",
             "duration_s": d} for d in (1.0, 2.0, 3.0, 4.0, 10.0)]
    rows = stats.per_model_latency([_dec(attempts=atts)])
    assert len(rows) == 1
    r = rows[0]
    assert r["tier"] == "M"
    assert r["model"] == "deepseek-coder-v2:16b"  # normalized
    assert r["n"] == 5
    assert r["p50"] == 3.0
    assert r["p95"] == 4.0
    assert r["p99"] == 4.0
    assert r["max"] == 10.0


def test_per_model_latency_gates_p95_p99_when_small():
    # n=2 -> p95 None (n<4), p99 None (n<5). p50 + max still computed.
    atts = [{"tier": "S", "model": "ollama_chat/x", "duration_s": d} for d in (1.0, 2.0)]
    rows = stats.per_model_latency([_dec(attempts=atts)])
    assert rows[0]["n"] == 2
    assert rows[0]["p50"] == 1.5
    assert rows[0]["p95"] is None
    assert rows[0]["p99"] is None
    assert rows[0]["max"] == 2.0


def test_per_model_latency_sorts_by_n_desc():
    # (M, big) has 5 attempts; (S, small) has 2 -> big should come first.
    atts = ([{"tier": "M", "model": "big", "duration_s": 1.0}] * 5
            + [{"tier": "S", "model": "small", "duration_s": 0.5}] * 2)
    rows = stats.per_model_latency([_dec(attempts=atts)])
    assert [r["model"] for r in rows] == ["big", "small"]


def test_per_model_latency_empty_input():
    assert stats.per_model_latency([]) == []
    assert stats.per_model_latency([_dec(attempts=[])]) == []


def test_per_model_latency_skips_malformed_attempts():
    # Missing keys / wrong types must be skipped, not crash.
    atts = [
        {"tier": "M", "model": "good", "duration_s": 2.0},
        {"tier": "M", "model": "good"},           # no duration -> skip
        "not-a-dict",                              # wrong type -> skip
        {"tier": "M", "duration_s": 3.0},         # no model -> skip
    ]
    rows = stats.per_model_latency([_dec(attempts=atts)])
    assert len(rows) == 1
    assert rows[0]["n"] == 1


def test_most_failing_critiqued_only_and_failure_rate():
    # Model A: 8 critiqued, 5 failed (scores 1,2,3,2,3 < 4) -> rate 0.625.
    # Model B: 6 critiqued, 1 failed (only score 2 < 4)     -> rate 0.167.
    # Sort by failure rate desc -> A first.
    a_atts = ([{"model": "ollama_chat/A", "critic_score": s}
               for s in (1, 2, 3, 2, 3, 4, 5, 4)])
    b_atts = ([{"model": "ollama_chat/B", "critic_score": s}
               for s in (5, 4, 5, 5, 5, 2)])
    rows = stats.most_failing_models([_dec(attempts=a_atts + b_atts)],
                                     pass_threshold=4)
    assert rows[0]["model"] == "A"
    assert rows[0]["critiqued"] == 8
    assert rows[0]["failed"] == 5   # scores < 4: 1,2,3,2,3 -> 5
    assert rows[1]["model"] == "B"


def test_most_failing_excludes_uncritiqued_attempts():
    # critic_score None -> "no opinion", not a failure.
    atts = [
        {"model": "X", "critic_score": 1},
        {"model": "X", "critic_score": 2},
        {"model": "X", "critic_score": 1},
        {"model": "X", "critic_score": 1},
        {"model": "X", "critic_score": 1},
        {"model": "X", "critic_score": None},
        {"model": "X", "critic_score": None},
    ]
    rows = stats.most_failing_models([_dec(attempts=atts)], pass_threshold=4)
    assert rows[0]["critiqued"] == 5   # not 7
    assert rows[0]["failed"] == 5


def test_most_failing_min_n_gate_excludes_thin_models():
    # Model with critiqued < 5 -> excluded (avoids "100% (1 of 1)" confidently-wrong).
    atts = [{"model": "thin", "critic_score": 1},
            {"model": "thin", "critic_score": 1}]
    rows = stats.most_failing_models([_dec(attempts=atts)], pass_threshold=4)
    assert rows == []


def test_most_failing_normalizes_model_name():
    # 'ollama_chat/X' and bare 'X' aggregate under same model.
    atts = ([{"model": "ollama_chat/X", "critic_score": 1}] * 3
            + [{"model": "X", "critic_score": 1}] * 2)
    rows = stats.most_failing_models([_dec(attempts=atts)], pass_threshold=4)
    assert len(rows) == 1
    assert rows[0]["model"] == "X"
    assert rows[0]["critiqued"] == 5


def test_most_failing_top_k_cap():
    # 7 models with high failure rate -> only top 5 returned.
    atts = []
    for i in range(7):
        atts += [{"model": f"m{i}", "critic_score": 1}] * 5   # all fail, n=5
    rows = stats.most_failing_models([_dec(attempts=atts)], pass_threshold=4,
                                     top_k=5)
    assert len(rows) == 5


def test_most_failing_empty_input():
    assert stats.most_failing_models([], pass_threshold=4) == []
    assert stats.most_failing_models([_dec(attempts=[])], pass_threshold=4) == []


def _row():
    return {
        "ts": 1700000000.123, "requested": "local-auto", "tier": "M",
        "model": "ollama_chat/deepseek-coder-v2:16b", "tokens": 250,
        "score": 80, "signals": "code,large", "classifier": "llm-up",
        "critic": 4, "escalated_to": None, "cloud_attempted": 0,
        "handoff": 0, "streamed": 0, "cap_category": "modification_or_edit",
        "cap_recommended_tier": "L", "cap_agrees_with_tier": 0,
    }


def test_render_explain_contains_routing_and_capability():
    attempts = [{"tier": "M", "model": "ollama_chat/deepseek-coder-v2:16b",
                 "prompt_tokens": 100, "completion_tokens": 150, "duration_s": 4.2,
                 "critic_score": 4, "was_warm": True, "vram_mb": 8000,
                 "cost_usd": 0.0, "preview": "Here is your answer."}]
    text = stats.render_explain(_row(), attempts)
    assert "Decision at" in text
    assert "local-auto" in text
    assert "deepseek-coder-v2:16b" in text
    assert "modification_or_edit" in text    # capability section present
    assert "llm-up" in text                  # classifier
    assert "4.2" in text or "4.20" in text   # duration in attempts trail
    assert "Here is your answer." in text    # preview in attempts trail


def test_render_explain_skips_capability_when_absent():
    row = _row()
    row["cap_category"] = None
    row["cap_recommended_tier"] = None
    text = stats.render_explain(row, attempts=[])
    assert "modification_or_edit" not in text
    # Should not crash and should not show a capability section heading.
    assert "Capability" not in text


def test_render_explain_defensive_on_missing_attempt_fields():
    # Older rows may lack was_warm / vram_mb / cost_usd.
    attempts = [{"tier": "S", "model": "ollama_chat/x", "duration_s": 1.0,
                 "critic_score": 5, "preview": "ok"}]
    text = stats.render_explain(_row(), attempts)
    # No KeyError; missing fields render as '-'.
    assert "ok" in text
    assert "-" in text   # at least one '-' for the absent was_warm/vram/cost


def test_render_explain_is_ascii():
    attempts = [{"tier": "M", "model": "x", "duration_s": 1.0,
                 "critic_score": 4, "preview": "p"}]
    stats.render_explain(_row(), attempts).encode("ascii")  # no non-ASCII


def test_render_text_includes_new_sections():
    atts = [{"tier": "M", "model": "ollama_chat/m", "duration_s": 2.0,
             "critic_score": 1}] * 6
    text = stats.render_text([_dec(attempts=atts)])
    assert "Per-model latency" in text
    assert "Most-failing models" in text


def test_render_json_includes_new_aggregates():
    atts = [{"tier": "M", "model": "ollama_chat/m", "duration_s": 2.0,
             "critic_score": 1}] * 6
    payload = _json.loads(stats.render_json([_dec(attempts=atts)]))
    assert "per_model_latency" in payload
    assert "most_failing_models" in payload
    assert payload["pass_threshold"] == stats._DEFAULT_PASS_THRESHOLD


def test_main_explain_missing_ts_exits_1(tmp_path, capsys):
    db = tmp_path / "r.sqlite"
    _make_decisions_db(db, [{"ts": 1000.0}])
    rc = stats.main(["--db", str(db), "--explain", "9999"])
    assert rc == 1
    assert "no decision found" in capsys.readouterr().err


def test_main_explain_latest_prints_drilldown(tmp_path, capsys):
    db = tmp_path / "r.sqlite"
    attempts = [{"tier": "M", "model": "ollama_chat/m", "duration_s": 2.0,
                 "critic_score": 4, "preview": "ok"}]
    _make_decisions_db(db, [{"ts": 1000.0, "tier": "M",
                             "attempts_json": _json.dumps(attempts)}])
    rc = stats.main(["--db", str(db), "--explain"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Decision at" in out
    assert "ok" in out


def test_main_explain_specific_ts(tmp_path, capsys):
    db = tmp_path / "r.sqlite"
    attempts = [{"tier": "S", "model": "x", "duration_s": 1.0,
                 "critic_score": 5, "preview": "hi"}]
    _make_decisions_db(db, [
        {"ts": 1000.0, "attempts_json": _json.dumps(attempts)},
        {"ts": 2000.0, "attempts_json": "[]"},
    ])
    rc = stats.main(["--db", str(db), "--explain", "1000.0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1000.0" in out  # ts in header
    assert "hi" in out


def test_main_pass_threshold_override(tmp_path):
    # With threshold=2, a score of 3 is no longer a failure -> 0/n failed.
    atts = [{"model": "M", "critic_score": s} for s in (3, 3, 3, 3, 3)]
    rows = stats.most_failing_models([_dec(attempts=atts)], pass_threshold=2)
    assert rows == []   # 0 failures at threshold 2
    rows4 = stats.most_failing_models([_dec(attempts=atts)], pass_threshold=4)
    assert rows4 and rows4[0]["failed"] == 5  # threshold 4 -> all 3<4 failures


def test_most_failing_excludes_zero_failure_models():
    # Explicit regression for the `failed == 0` gate added during review.
    # A model with 50 critiqued attempts and 0 failures must NOT appear in a
    # "most failing" list (semantically: it's the most reliable, not failing).
    atts = [{"model": "perfect", "critic_score": 5}] * 50
    rows = stats.most_failing_models([_dec(attempts=atts)], pass_threshold=4)
    assert rows == []


def test_main_explain_malformed_ts_exits_1(tmp_path, capsys):
    # The spec mandates a friendly error for a non-numeric --explain value.
    # The implementation handles it (ValueError -> exit 1); pin it with a test.
    db = tmp_path / "r.sqlite"
    _make_decisions_db(db, [{"ts": 1000.0}])
    rc = stats.main(["--db", str(db), "--explain", "banana"])
    assert rc == 1
    assert "expects a numeric ts" in capsys.readouterr().err
