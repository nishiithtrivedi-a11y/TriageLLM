"""Issue #18b: capability advisory mode (advisory.py). Pure + tmp-sqlite, no Ollama."""
import json
import sqlite3

import advisory as adv


def test_constants():
    assert adv._MIN_BENCH_RATE == 0.7
    assert adv._SOFT_BENCH_RATE == 0.8
    assert adv._MIN_BENCH_SAMPLES == 3
    assert adv._MIN_LIVE_SAMPLES == 5
    assert adv._LIVE_CORROBORATE_FLOOR == 0.6
    assert adv._LIVE_DISAGREE_FLOOR == 0.4
    assert adv._HARD_CATEGORIES == {"structured_output", "modification_or_edit"}


def test_normalize_model():
    assert adv._normalize_model("ollama_chat/qwen3-coder:30b") == "qwen3-coder:30b"
    assert adv._normalize_model("anthropic/claude-3-5-sonnet") == "claude-3-5-sonnet"
    assert adv._normalize_model("qwen3-coder:30b") == "qwen3-coder:30b"
    assert adv._normalize_model("") == ""


def test_graduated_rec_shape():
    g = adv.GraduatedRec("structured_output", "m", "graduated", True,
                         0.8, 5, "hard", 0.78, 12, "hard-benchmark-pass")
    assert g.category == "structured_output"
    assert g.graduated is True
    assert g.live_pass_rate == 0.78


def _priors(category, model, success_rate, n_prompts, confidence, warning=None):
    """Minimal benchmark_results.json with one (model, category) recommendation."""
    return {
        "generated_at": "2026-05-28T00:00:00Z",
        "results": {model: {category: {
            "success_rate": success_rate, "n_prompts": n_prompts,
            "n_passed": int(success_rate * n_prompts), "confidence": confidence,
            "latency_p50_s": 1.0, "latency_p95_s": None, "latency_max_s": 1.0}}},
        "recommendations": {category: {
            "model": model, "success_rate": success_rate, "latency_p50_s": 1.0,
            "confidence": confidence, "decided_by": "success_rate", "warning": warning}},
    }


def test_graduate_hard_on_benchmark_alone():
    p = _priors("structured_output", "m", 0.75, 5, "hard")
    g = adv.graduate(p, {})  # no live data needed for hard
    rec = g["structured_output"]
    assert rec.status == "graduated" and rec.graduated is True
    assert rec.reason == "hard-benchmark-pass"


def test_graduate_soft_strong_benchmark_alone():
    p = _priors("creative_generation", "m", 0.85, 5, "soft")  # >= 0.8
    g = adv.graduate(p, {})
    rec = g["creative_generation"]
    assert rec.status == "graduated" and rec.reason == "soft-benchmark-strong"


def test_graduate_soft_middle_band_no_live_needs_evidence():
    p = _priors("creative_generation", "m", 0.75, 5, "soft")  # 0.7..0.8, no live
    g = adv.graduate(p, {})
    rec = g["creative_generation"]
    assert rec.status == "needs-live-evidence" and rec.graduated is False


def test_graduate_soft_middle_band_live_corroborated():
    p = _priors("analytical_task", "m", 0.75, 5, "soft")
    live = {("analytical_task", "m"): (5, 6)}  # 0.83 >= 0.6, n=6 >= 5
    g = adv.graduate(p, live)
    rec = g["analytical_task"]
    assert rec.status == "graduated" and rec.reason == "soft-live-corroborated"
    assert rec.live_pass_rate is not None


def test_graduate_live_disagreement_vetoes_even_hard():
    p = _priors("structured_output", "m", 0.9, 5, "hard")
    live = {("structured_output", "m"): (1, 8)}  # 0.125 < 0.4, n=8 >= 5
    g = adv.graduate(p, live)
    rec = g["structured_output"]
    assert rec.status == "live-disagreement" and rec.graduated is False


def test_graduate_insufficient_benchmark_low_rate():
    p = _priors("analytical_task", "m", 0.5, 5, "soft")  # < 0.7
    g = adv.graduate(p, {})
    assert g["analytical_task"].status == "insufficient-benchmark"
    assert g["analytical_task"].reason == "benchmark-rate-low"


def test_graduate_insufficient_benchmark_low_samples():
    p = _priors("analytical_task", "m", 0.9, 2, "soft")  # n < 3
    g = adv.graduate(p, {})
    assert g["analytical_task"].reason == "benchmark-samples-low"


def test_graduate_no_model_passed_warning():
    p = _priors("modification_or_edit", "m", 0.0, 5, "hard", warning="no-model-passed")
    g = adv.graduate(p, {})
    assert g["modification_or_edit"].status == "insufficient-benchmark"
    assert g["modification_or_edit"].reason == "no-model-passed"


def test_build_report_shape_and_endorsed_flag():
    p = _priors("structured_output", "m", 0.8, 5, "hard")
    g = adv.graduate(p, {})
    endorsements = {"endorsements": {"structured_output": {"model": "m"}}}
    report = adv.build_report(g, endorsements, p["generated_at"])
    assert report["schema_version"] == 1
    assert report["priors_generated_at"] == "2026-05-28T00:00:00Z"
    c = report["categories"]["structured_output"]
    assert c["model"] == "m" and c["status"] == "graduated"
    assert c["benchmark"]["success_rate"] == 0.8 and c["benchmark"]["confidence"] == "hard"
    assert c["live"]["pass_rate"] is None and c["live"]["n"] == 0
    assert c["endorsed"] is True   # endorsement matches current model
    json.dumps(report)             # serializable


def test_build_report_endorsed_false_when_model_differs():
    p = _priors("structured_output", "new-model", 0.8, 5, "hard")
    g = adv.graduate(p, {})
    endorsements = {"endorsements": {"structured_output": {"model": "old-model"}}}
    report = adv.build_report(g, endorsements, p["generated_at"])
    assert report["categories"]["structured_output"]["endorsed"] is False


def test_render_report_text_and_json():
    p = _priors("analytical_task", "m", 0.75, 5, "soft")
    g = adv.graduate(p, {})
    report = adv.build_report(g, {"endorsements": {}}, p["generated_at"])
    text = adv.render_report(report, json_mode=False)
    assert "analytical_task" in text and "needs-live-evidence" in text
    out = adv.render_report(report, json_mode=True)
    assert json.loads(out) == report


def _graduated_one(category, model, status, graduated, live_n=0, live_rate=None):
    return {category: adv.GraduatedRec(category, model, status, graduated,
                                       0.8, 5, "hard", live_rate, live_n, "r")}


def test_apply_signoff_endorses_graduated():
    g = _graduated_one("structured_output", "m", "graduated", True)
    new, msg = adv.apply_signoff({"endorsements": {}}, g, "structured_output")
    assert "endorsed structured_output -> m" in msg
    e = new["endorsements"]["structured_output"]
    assert e["model"] == "m" and e["source"] == "benchmark"  # no live evidence
    assert "endorsed_at" in e and new["schema_version"] == 1


def test_apply_signoff_source_benchmark_plus_live():
    g = _graduated_one("analytical_task", "m", "graduated", True, live_n=6, live_rate=0.83)
    new, msg = adv.apply_signoff({"endorsements": {}}, g, "analytical_task")
    assert new["endorsements"]["analytical_task"]["source"] == "benchmark+live"


def test_apply_signoff_refuses_non_graduated():
    g = _graduated_one("creative_generation", "m", "needs-live-evidence", False)
    new, msg = adv.apply_signoff({"endorsements": {}}, g, "creative_generation")
    assert "not graduated" in msg and "needs-live-evidence" in msg
    assert new["endorsements"] == {}   # unchanged


def test_apply_signoff_unknown_category():
    new, msg = adv.apply_signoff({"endorsements": {}}, {}, "nope")
    assert "unknown category" in msg


import pytest  # noqa: E402


def test_load_priors_reads_json(tmp_path):
    p = tmp_path / "bench.json"
    p.write_text(json.dumps({"schema_version": 1, "recommendations": {}}), encoding="utf-8")
    assert adv.load_priors(str(p))["schema_version"] == 1


def test_load_priors_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        adv.load_priors(str(tmp_path / "nope.json"))


def test_load_endorsements_missing_is_empty(tmp_path):
    e = adv.load_endorsements(str(tmp_path / "none.json"))
    assert e == {"schema_version": 1, "endorsements": {}}


def test_write_endorsements_atomic_round_trip(tmp_path):
    path = tmp_path / "out" / "advisory_endorsements.json"
    path.parent.mkdir(parents=True)
    obj = {"schema_version": 1, "endorsements": {"x": {"model": "m"}}}
    adv.write_endorsements(str(path), obj)
    assert json.loads(path.read_text(encoding="utf-8")) == obj
    adv.write_endorsements(str(path), {"schema_version": 1, "endorsements": {}})
    assert adv.load_endorsements(str(path))["endorsements"] == {}
    assert list(path.parent.glob("*.tmp")) == []


def test_load_live_aggregates_normalizes_and_counts(tmp_path):
    db = tmp_path / "d.sqlite"
    attempts = [
        {"model": "ollama_chat/qwen3-coder:30b", "critic_score": 5},   # pass
        {"model": "ollama_chat/qwen3-coder:30b", "critic_score": 2},   # fail
        {"model": "ollama_chat/qwen3-coder:30b", "critic_score": None},  # excluded (no signal)
    ]
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE decisions (cap_category TEXT, attempts_json TEXT)")
        c.execute("INSERT INTO decisions VALUES (?, ?)",
                  ("analytical_task", json.dumps(attempts)))
        c.execute("INSERT INTO decisions VALUES (?, ?)", (None, "[]"))  # shadow off -> skipped
    agg = adv.load_live_aggregates(str(db), pass_threshold=4)
    # normalized key, 1 pass of 2 critiqued (None excluded)
    assert agg[("analytical_task", "qwen3-coder:30b")] == (1, 2)


def test_load_live_aggregates_missing_db_is_empty(tmp_path):
    assert adv.load_live_aggregates(str(tmp_path / "nope.sqlite"), 4) == {}


def test_build_parser_flags_and_defaults():
    parser = adv.build_parser()
    a = parser.parse_args([])
    assert a.sign_off is None and a.sign_off_all is False and a.json is False
    assert a.priors == "benchmark_results.json"
    assert a.endorsements == "advisory_endorsements.json"
    a2 = parser.parse_args(["--sign-off", "structured_output", "--json"])
    assert a2.sign_off == "structured_output" and a2.json is True
    a3 = parser.parse_args(["--sign-off-all"])
    assert a3.sign_off_all is True


def test_run_report_path(tmp_path, capsys, monkeypatch):
    priors = _priors("structured_output", "m", 0.8, 5, "hard")
    pp = tmp_path / "bench.json"
    pp.write_text(json.dumps(priors), encoding="utf-8")
    # no live DB -> empty aggregates; avoid importing real config/db
    monkeypatch.setattr(adv, "load_live_aggregates", lambda db, t: {})
    monkeypatch.setattr(adv, "load_config", lambda: type("C", (), {"critic_pass_threshold": 4})())
    args = adv.build_parser().parse_args(
        ["--priors", str(pp), "--endorsements", str(tmp_path / "e.json")])
    rc = adv.run(args)
    assert rc == 0
    assert "structured_output" in capsys.readouterr().out


def test_run_sign_off_writes_endorsement(tmp_path, monkeypatch):
    priors = _priors("structured_output", "m", 0.8, 5, "hard")
    pp = tmp_path / "bench.json"
    pp.write_text(json.dumps(priors), encoding="utf-8")
    ep = tmp_path / "e.json"
    monkeypatch.setattr(adv, "load_live_aggregates", lambda db, t: {})
    monkeypatch.setattr(adv, "load_config", lambda: type("C", (), {"critic_pass_threshold": 4})())
    args = adv.build_parser().parse_args(
        ["--priors", str(pp), "--endorsements", str(ep), "--sign-off", "structured_output"])
    rc = adv.run(args)
    assert rc == 0
    saved = json.loads(ep.read_text(encoding="utf-8"))
    assert saved["endorsements"]["structured_output"]["model"] == "m"


def test_run_missing_priors_returns_1(tmp_path, capsys):
    args = adv.build_parser().parse_args(["--priors", str(tmp_path / "nope.json")])
    rc = adv.run(args)
    assert rc == 1
    assert "benchmark" in capsys.readouterr().out.lower()


def test_load_live_aggregates_since_ts_filters(tmp_path):
    db = tmp_path / "rt.sqlite"
    import json as _json
    old = _json.dumps([{"model": "ollama_chat/m1", "critic_score": 5}])
    new = _json.dumps([{"model": "ollama_chat/m1", "critic_score": 5}])
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE decisions (ts REAL, cap_category TEXT, attempts_json TEXT)")
        c.execute("INSERT INTO decisions VALUES (1000.0, 'quick_question', ?)", (old,))
        c.execute("INSERT INTO decisions VALUES (9000.0, 'quick_question', ?)", (new,))
    # since_ts excludes the ts=1000 row, keeps ts=9000.
    agg = adv.load_live_aggregates(str(db), pass_threshold=4, since_ts=5000.0)
    assert agg == {("quick_question", "m1"): (1, 1)}


def test_load_live_aggregates_since_ts_none_is_all_time(tmp_path):
    # Regression guard: since_ts=None must behave exactly as before (and must
    # NOT require a ts column to exist).
    db = tmp_path / "rt.sqlite"
    import json as _json
    a = _json.dumps([{"model": "ollama_chat/m1", "critic_score": 5},
                     {"model": "ollama_chat/m1", "critic_score": 2}])
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE decisions (cap_category TEXT, attempts_json TEXT)")
        c.execute("INSERT INTO decisions VALUES ('quick_question', ?)", (a,))
    agg = adv.load_live_aggregates(str(db), pass_threshold=4, since_ts=None)
    assert agg == {("quick_question", "m1"): (1, 2)}
