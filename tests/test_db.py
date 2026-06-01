"""SQLite schema bootstrap + migration."""
import sqlite3
from pathlib import Path

import router_hook


def test_init_db_creates_schema_with_all_columns(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "router_decisions.sqlite"
    monkeypatch.setattr(router_hook, "DB_PATH", db)

    router_hook._init_db()

    with sqlite3.connect(db) as c:
        cols = {row[1] for row in c.execute("PRAGMA table_info(decisions)")}

    assert cols >= {
        "ts", "requested", "tier", "model", "tokens", "score", "signals",
        "classifier", "critic", "escalated_to",
    }


def test_init_db_migrates_older_schema(tmp_path: Path, monkeypatch) -> None:
    """Simulate a pre-classifier-column DB and confirm migration adds the columns."""
    db = tmp_path / "router_decisions.sqlite"
    monkeypatch.setattr(router_hook, "DB_PATH", db)

    with sqlite3.connect(db) as c:
        c.execute(
            "CREATE TABLE decisions (ts REAL, requested TEXT, tier TEXT, "
            "model TEXT, tokens INTEGER, score INTEGER, signals TEXT)"
        )

    router_hook._init_db()

    with sqlite3.connect(db) as c:
        cols = {row[1] for row in c.execute("PRAGMA table_info(decisions)")}

    assert "classifier" in cols
    assert "critic" in cols
    assert "escalated_to" in cols


def test_init_db_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "router_decisions.sqlite"
    monkeypatch.setattr(router_hook, "DB_PATH", db)
    router_hook._init_db()
    router_hook._init_db()  # must not raise
