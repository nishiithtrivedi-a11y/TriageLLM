"""Tests for capability columns in the decisions ledger.

Spec: the design notes S6.
"""
import sqlite3

import pytest

import router_hook
from router_hook import Attempt


def test_init_db_creates_capability_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(router_hook, "DB_PATH", tmp_path / "rt.sqlite")
    router_hook._init_db()
    with sqlite3.connect(tmp_path / "rt.sqlite") as c:
        cols = {row[1] for row in c.execute("PRAGMA table_info(decisions)")}
    expected = {
        "cap_category", "cap_recommended_tier", "cap_reason_code",
        "cap_signals", "cap_confidence", "cap_classifier_used",
        "cap_pack", "cap_agrees_with_tier",
    }
    missing = expected - cols
    assert not missing, f"missing capability columns: {missing}"


def test_init_db_is_idempotent(tmp_path, monkeypatch):
    """Running migration twice must not error and must not duplicate columns."""
    monkeypatch.setattr(router_hook, "DB_PATH", tmp_path / "rt.sqlite")
    router_hook._init_db()
    router_hook._init_db()
    with sqlite3.connect(tmp_path / "rt.sqlite") as c:
        cols = [row[1] for row in c.execute("PRAGMA table_info(decisions)")]
    assert len(cols) == len(set(cols))


def test_init_db_raises_if_migration_did_not_create_cap_columns(tmp_path, monkeypatch):
    """If _init_db can create the base table but the cap_* ALTER TABLE
    statements never landed (sqlite quirk, locked DB, partial restore),
    we want a loud RuntimeError at proxy startup, not a silent NULL
    dashboard later."""
    db_path = tmp_path / "broken.db"
    monkeypatch.setattr(router_hook, "DB_PATH", db_path)

    # Pre-create the table WITHOUT any cap_* columns and arrange for the
    # migration loop to be a no-op (simulate a quirk where ALTER TABLE
    # silently fails). We achieve this by stubbing sqlite3.connect to a
    # connection whose execute() ignores ALTER TABLE statements.
    import sqlite3
    real_connect = sqlite3.connect

    class _BrokenConn:
        def __init__(self, inner):
            self._inner = inner
        def __enter__(self):
            self._inner.__enter__()
            return self
        def __exit__(self, *a):
            return self._inner.__exit__(*a)
        def execute(self, sql, *args, **kwargs):
            if sql.strip().upper().startswith("ALTER TABLE"):
                # Silently swallow -- simulate the bug we're guarding against
                class _Cursor:
                    def __iter__(self): return iter([])
                    def fetchall(self): return []
                return _Cursor()
            return self._inner.execute(sql, *args, **kwargs)

    def _fake_connect(path, *a, **kw):
        return _BrokenConn(real_connect(path, *a, **kw))

    monkeypatch.setattr(sqlite3, "connect", _fake_connect)

    with pytest.raises(RuntimeError) as exc:
        router_hook._init_db()
    msg = str(exc.value)
    assert "decisions" in msg
    assert "cap_" in msg  # at least one of the missing column names is mentioned


def test_log_decision_prints_dashboard_warning_once_across_multiple_fallbacks(
    tmp_path, monkeypatch, capsys
):
    """When _log_decision falls back to JSONL because the SQLite INSERT
    failed, we want exactly ONE prominent 'dashboard will show no data'
    hint printed per process, regardless of how many subsequent writes
    also fail. Spamming the warning per failed row would drown the log."""
    db_path = tmp_path / "broken.db"
    monkeypatch.setattr(router_hook, "DB_PATH", db_path)

    # Set up a decisions table that is MISSING cap_* columns so every
    # INSERT raises OperationalError.
    import sqlite3
    with sqlite3.connect(db_path) as c:
        c.execute("""
            CREATE TABLE decisions (
                ts REAL, requested TEXT, tier TEXT, model TEXT,
                tokens INTEGER, score INTEGER, signals TEXT,
                classifier TEXT, critic INTEGER, escalated_to TEXT,
                attempts_json TEXT, cloud_attempted INTEGER,
                handoff INTEGER, streamed INTEGER
            )
        """)

    # Reset module-level one-shot flag so the test isn't poisoned by prior
    # state from other tests in the same pytest process.
    router_hook._dashboard_blind_warned = False
    router_hook._log_write_failures = 0

    state = {
        "requested": "local-s", "initial_tier": "S", "alias": "local-s",
        "tokens": 0, "score": 0, "signals": [], "classifier": "rules",
    }
    a = router_hook.Attempt(tier="S", model="x", prompt_tokens=0,
                            completion_tokens=0, duration_s=0.0,
                            critic_score=None, preview="")

    # Fire three failed writes back-to-back.
    router_hook._log_decision(state, [a], cloud_attempted=False,
                              handoff=False, streamed=False)
    router_hook._log_decision(state, [a], cloud_attempted=False,
                              handoff=False, streamed=False)
    router_hook._log_decision(state, [a], cloud_attempted=False,
                              handoff=False, streamed=False)

    out = capsys.readouterr().out
    # The dashboard-blind warning must appear EXACTLY once.
    assert out.count("DASHBOARD WILL SHOW NO DATA") == 1, (
        f"warning count = {out.count('DASHBOARD WILL SHOW NO DATA')}, "
        f"output:\n{out}"
    )
    # The per-row [log] CRITICAL line still fires for each failure (3x).
    assert out.count("[log] CRITICAL: sqlite write failed") == 3


def test_log_decision_does_not_print_dashboard_warning_on_healthy_db(
    tmp_path, monkeypatch, capsys
):
    """No false alarms when the DB is healthy."""
    db_path = tmp_path / "good.db"
    monkeypatch.setattr(router_hook, "DB_PATH", db_path)
    router_hook._dashboard_blind_warned = False
    router_hook._log_write_failures = 0
    router_hook._init_db()  # full migration -- INSERT will succeed

    state = {
        "requested": "local-s", "initial_tier": "S", "alias": "local-s",
        "tokens": 0, "score": 0, "signals": [], "classifier": "rules",
    }
    a = router_hook.Attempt(tier="S", model="x", prompt_tokens=0,
                            completion_tokens=0, duration_s=0.0,
                            critic_score=None, preview="")
    router_hook._log_decision(state, [a], cloud_attempted=False,
                              handoff=False, streamed=False)

    out = capsys.readouterr().out
    assert "DASHBOARD WILL SHOW NO DATA" not in out
    assert "[log] CRITICAL" not in out


def test_log_decision_writes_capability_columns_when_state_has_them(tmp_path, monkeypatch):
    monkeypatch.setattr(router_hook, "DB_PATH", tmp_path / "rt.sqlite")
    router_hook._init_db()
    state = {
        "requested": "local-auto", "initial_tier": "M", "alias": "local-m",
        "tokens": 5, "score": 1, "signals": ["m_kw=1"], "classifier": "rules",
        "capability": {
            "cap_category": "structured_output",
            "cap_recommended_tier": "M",
            "cap_reason_code": "structured_output:rule-match",
            "cap_signals": "json,schema",
            "cap_confidence": 0.8,
            "cap_classifier_used": "rules",
            "cap_pack": "coder",
            "cap_agrees_with_tier": 1,
        },
    }
    attempt = Attempt(tier="M", model="ollama_chat/m", prompt_tokens=5,
                      completion_tokens=10, duration_s=1.0, critic_score=4,
                      preview="ok")
    router_hook._log_decision(state, [attempt], cloud_attempted=False,
                              handoff=False, streamed=False)
    with sqlite3.connect(tmp_path / "rt.sqlite") as c:
        c.row_factory = sqlite3.Row
        row = c.execute("SELECT * FROM decisions").fetchone()
    assert row["cap_category"] == "structured_output"
    assert row["cap_recommended_tier"] == "M"
    assert row["cap_signals"] == "json,schema"
    assert row["cap_confidence"] == 0.8
    assert row["cap_agrees_with_tier"] == 1
    assert row["cap_reason_code"] == "structured_output:rule-match"
    assert row["cap_classifier_used"] == "rules"
    assert row["cap_pack"] == "coder"


def test_log_decision_row_tuple_matches_decisions_columns(tmp_path, monkeypatch):
    """Structural guard for Issue #9: the positional row tuple in
    _log_decision must have one value per column in _DECISIONS_COLUMNS,
    in the same order. If someone adds a column to _DECISIONS_COLUMNS
    but forgets to add a value to the row tuple (or vice versa), the
    SQL placeholder count won't match and SQLite raises OperationalError
    on the INSERT. This test catches that at unit-test time."""
    db_path = tmp_path / "drift_guard.db"
    monkeypatch.setattr(router_hook, "DB_PATH", db_path)
    router_hook._dashboard_blind_warned = False
    router_hook._log_write_failures = 0
    router_hook._init_db()

    # Sanity: column count matches the source-of-truth tuple length.
    import sqlite3
    with sqlite3.connect(db_path) as c:
        col_count = len(list(c.execute("PRAGMA table_info(decisions)")))
    assert col_count == len(router_hook._DECISIONS_COLUMNS), (
        f"decisions table has {col_count} columns but "
        f"_DECISIONS_COLUMNS has {len(router_hook._DECISIONS_COLUMNS)}"
    )

    # Now exercise the full write path. If the row tuple's length doesn't
    # match _DECISIONS_COLUMNS, SQLite raises OperationalError mentioning
    # "X values for Y columns".
    state = {
        "requested": "local-auto",
        "initial_tier": "S",
        "alias": "local-s",
        "tokens": 5,
        "score": 0,
        "signals": ["test"],
        "classifier": "rules",
        "capability": {
            "cap_category": "default",
            "cap_recommended_tier": "S",
            "cap_reason_code": "default:no-rule-fired",
            "cap_signals": "",
            "cap_confidence": 0.5,
            "cap_classifier_used": "rules",
            "cap_pack": "coder",
            "cap_agrees_with_tier": 1,
        },
    }
    attempt = router_hook.Attempt(
        tier="S", model="x", prompt_tokens=5, completion_tokens=0,
        duration_s=0.0, critic_score=None, preview=""
    )
    router_hook._log_decision(
        state, [attempt], cloud_attempted=False, handoff=False, streamed=False
    )

    # Read back the row and confirm the column-to-value mapping is correct
    # (i.e. row tuple order matches _DECISIONS_COLUMNS order). If the order
    # drifted, e.g. cap_category got swapped with cap_recommended_tier in
    # the row tuple, this assertion would still pass for the count but the
    # values would land in the wrong columns -- catch that by asserting
    # specific column values.
    with sqlite3.connect(db_path) as c:
        c.row_factory = sqlite3.Row
        rows = list(c.execute("SELECT * FROM decisions"))
    assert len(rows) == 1
    r = rows[0]
    assert r["requested"] == "local-auto"
    assert r["tier"] == "S"
    assert r["cap_category"] == "default"
    assert r["cap_recommended_tier"] == "S"
    assert r["cap_reason_code"] == "default:no-rule-fired"
    assert r["cap_agrees_with_tier"] == 1


def test_log_decision_writes_null_when_capability_absent(tmp_path, monkeypatch):
    """When capability_routing is disabled, all 8 columns must be NULL."""
    monkeypatch.setattr(router_hook, "DB_PATH", tmp_path / "rt.sqlite")
    router_hook._init_db()
    state = {
        "requested": "local-auto", "initial_tier": "S", "alias": "local-s",
        "tokens": 1, "score": 0, "signals": [], "classifier": "rules",
        # no "capability" key
    }
    attempt = Attempt(tier="S", model="ollama_chat/s", prompt_tokens=1,
                      completion_tokens=2, duration_s=0.1, critic_score=None,
                      preview="ok")
    router_hook._log_decision(state, [attempt], cloud_attempted=False,
                              handoff=False, streamed=False)
    with sqlite3.connect(tmp_path / "rt.sqlite") as c:
        c.row_factory = sqlite3.Row
        row = c.execute("SELECT * FROM decisions").fetchone()
    for col in ("cap_category", "cap_recommended_tier", "cap_reason_code",
                "cap_signals", "cap_confidence", "cap_classifier_used",
                "cap_pack", "cap_agrees_with_tier"):
        assert row[col] is None, f"{col} should be NULL when capability absent"
