"""Unit tests for backtest (Issue #28). Pure + sqlite-via-tmp, no Ollama."""
import time

import backtest as bt


def test_drift_margin_constant():
    assert bt._DRIFT_MARGIN == 0.15


def test_parse_since_none():
    assert bt._parse_since(None) is None


def test_parse_since_forms():
    now = time.time()
    assert abs(bt._parse_since("1h") - (now - 3600)) < 5
    assert abs(bt._parse_since("2d") - (now - 2 * 86400)) < 5
    assert abs(bt._parse_since("90m") - (now - 90 * 60)) < 5


def test_parse_since_malformed_raises():
    import pytest
    for bad in ["banana", "7x", "", "d", "-3d"]:
        with pytest.raises(ValueError):
            bt._parse_since(bad)


def test_verdict_dataclass_shape():
    v = bt.BacktestVerdict("quick_question", "m1", "upheld", 0.8, 0.82, 10,
                           0.02, "2026-05-28T00:00:00Z", "live-holds")
    assert v.verdict == "upheld"
    assert v.live_n == 10


def _ends(category="quick_question", model="m1", success_rate=0.8,
          live_pass_rate=None):
    return {"endorsements": {category: {
        "model": model, "success_rate": success_rate, "confidence": "soft",
        "live_pass_rate": live_pass_rate, "endorsed_at": "2026-05-28T00:00:00Z",
        "source": "benchmark"}}}


def test_evaluate_upheld():
    # baseline = success_rate 0.8 (snapshot live null); live 0.82 -> upheld.
    v = bt.evaluate_endorsements(_ends(), {("quick_question", "m1"): (41, 50)})
    g = v["quick_question"]
    assert g.verdict == "upheld"
    assert g.expected_rate == 0.8
    assert g.reason == "live-holds"


def test_evaluate_insufficient_data():
    # live_n below _MIN_LIVE_SAMPLES (5).
    v = bt.evaluate_endorsements(_ends(), {("quick_question", "m1"): (2, 3)})
    assert v["quick_question"].verdict == "insufficient-data"
    assert v["quick_question"].drift is None


def test_evaluate_model_absent_is_insufficient():
    v = bt.evaluate_endorsements(_ends(), {})   # no live for (cat, model)
    assert v["quick_question"].verdict == "insufficient-data"
    assert v["quick_question"].live_n == 0


def test_evaluate_drifted():
    # baseline 0.8; live 0.6 -> drift -0.20 <= -0.15, above disagree floor 0.4.
    v = bt.evaluate_endorsements(_ends(), {("quick_question", "m1"): (6, 10)})
    g = v["quick_question"]
    assert g.verdict == "drifted"
    assert g.reason == "dropped-from-baseline"
    assert g.drift == -0.2


def test_evaluate_contradicted():
    # live 0.3 < disagree floor 0.4 -> contradicted (severe), regardless of drift.
    v = bt.evaluate_endorsements(_ends(), {("quick_question", "m1"): (3, 10)})
    assert v["quick_question"].verdict == "contradicted"
    assert v["quick_question"].reason == "below-disagree-floor"


def test_evaluate_baseline_uses_snapshot_live_when_present():
    # snapshot live_pass_rate 0.9 is the baseline (not success_rate); live 0.7
    # -> drift -0.20 -> drifted.
    v = bt.evaluate_endorsements(
        _ends(success_rate=0.8, live_pass_rate=0.9),
        {("quick_question", "m1"): (7, 10)})
    g = v["quick_question"]
    assert g.expected_rate == 0.9
    assert g.verdict == "drifted"


import json as _json  # noqa: E402


def _verdicts():
    return bt.evaluate_endorsements(
        {"endorsements": {
            "quick_question": {"model": "m1", "success_rate": 0.8,
                               "confidence": "soft", "live_pass_rate": None,
                               "endorsed_at": "2026-05-28T00:00:00Z",
                               "source": "benchmark"},
            "structured_output": {"model": "m1", "success_rate": 0.9,
                                  "confidence": "hard", "live_pass_rate": None,
                                  "endorsed_at": "2026-05-28T00:00:00Z",
                                  "source": "benchmark"}}},
        {("quick_question", "m1"): (9, 10),     # upheld
         ("structured_output", "m1"): (3, 10)})  # contradicted


def test_build_report_summary_counts():
    report = bt.build_report(_verdicts(), "2026-05-28T00:00:00Z", "7d")
    s = report["summary"]
    assert s["total"] == 2
    assert s["upheld"] == 1
    assert s["contradicted"] == 1
    assert s["drifted"] == 0
    assert s["insufficient-data"] == 0
    assert report["window"] == "7d"
    assert report["schema_version"] == 1


def test_render_report_text_has_headline_and_rows():
    report = bt.build_report(_verdicts(), "2026-05-28T00:00:00Z", "all-time")
    text = bt.render_report(report, json_mode=False)
    assert "upheld" in text
    assert "contradicted" in text
    assert "quick_question" in text
    assert "1/2 upheld" in text   # headline summary


def test_render_report_json_round_trips():
    report = bt.build_report(_verdicts(), "2026-05-28T00:00:00Z", "7d")
    assert _json.loads(bt.render_report(report, json_mode=True)) == report


def test_build_parser_defaults():
    args = bt.build_parser().parse_args([])
    assert args.since is None
    assert args.json is False
    assert args.endorsements == "advisory_endorsements.json"


def test_build_parser_flags():
    args = bt.build_parser().parse_args(
        ["--since", "7d", "--json", "--endorsements", "e.json", "--db", "d.sqlite"])
    assert args.since == "7d"
    assert args.json is True
    assert args.endorsements == "e.json"
    assert args.db == "d.sqlite"


def test_run_empty_endorsements_exits_0(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(bt, "load_live_aggregates", lambda db, t, since_ts=None: {})
    args = bt.build_parser().parse_args(["--endorsements", str(tmp_path / "none.json")])
    assert bt.run(args) == 0
    assert "no endorsements to backtest" in capsys.readouterr().out


def test_run_malformed_since_exits_nonzero(tmp_path, capsys):
    args = bt.build_parser().parse_args(["--since", "banana"])
    rc = bt.run(args)
    assert rc != 0
    assert "invalid --since" in capsys.readouterr().out


def test_run_report_path(tmp_path, capsys, monkeypatch):
    ends = tmp_path / "e.json"
    ends.write_text(_json.dumps({"schema_version": 1, "updated_at": "t",
        "endorsements": {"quick_question": {"model": "m1", "success_rate": 0.8,
        "confidence": "soft", "live_pass_rate": None,
        "endorsed_at": "2026-05-28T00:00:00Z", "source": "benchmark"}}}),
        encoding="utf-8")
    monkeypatch.setattr(bt, "load_live_aggregates",
                        lambda db, t, since_ts=None: {("quick_question", "m1"): (9, 10)})
    args = bt.build_parser().parse_args(["--endorsements", str(ends)])
    assert bt.run(args) == 0
    assert "upheld" in capsys.readouterr().out
