"""Unit tests for doctor (setup pre-flight). Pure + sync, no Ollama/proxy."""
from types import SimpleNamespace

import doctor as doc


def _cfg(classifier="qwen2.5:0.5b", critic="qwen2.5:0.5b",
         cloud_enabled=False, api_key_env="ANTHROPIC_API_KEY",
         cloud_model="anthropic/claude-sonnet-4-6", cap_enabled=False):
    return SimpleNamespace(
        classifier_model=classifier, critic_model=critic,
        cloud_escalation=SimpleNamespace(enabled=cloud_enabled,
                                         api_key_env=api_key_env, model=cloud_model),
        capability_routing=SimpleNamespace(enabled=cap_enabled))


def test_check_config_coherent_passes():
    r = doc.check_config(_cfg(), {"S": "ollama_chat/m1", "M": "ollama_chat/m2"}, True)
    assert r.status == "PASS"
    assert "2 tiers" in r.detail


def test_check_config_absent_warns():
    r = doc.check_config(_cfg(), {"S": "ollama_chat/m1"}, False)
    assert r.status == "WARN"
    assert "not found" in r.detail


def test_check_config_empty_map_fails():
    r = doc.check_config(_cfg(), {}, True)
    assert r.status == "FAIL"


def test_check_config_blank_critic_fails():
    r = doc.check_config(_cfg(critic=""), {"S": "ollama_chat/m1"}, True)
    assert r.status == "FAIL"


def test_check_result_shape():
    r = doc.CheckResult("x", "PASS", "ok", fix=None)
    assert r.status == "PASS" and r.fix is None


def test_check_models_all_present_passes():
    r = doc.check_models({"m1", "m2", "c"}, ["m1", "m2", "c"])
    assert r.status == "PASS"


def test_check_models_missing_fails_with_fix():
    r = doc.check_models({"m1"}, ["m1", "m2", "c"])
    assert r.status == "FAIL"
    assert "m2" in r.detail and "c" in r.detail
    assert "ollama pull" in r.fix.lower()


def test_check_models_empty_installed_warns():
    r = doc.check_models(set(), ["m1"])
    assert r.status == "WARN"
    assert "could not verify" in r.detail


def test_analyze_cloud_local_first_passes():
    r = doc.analyze_cloud(_cfg(cloud_enabled=False), {}, 0, 0, all_local=True)
    assert r.status == "PASS"
    assert "fully local" in r.detail


def test_analyze_cloud_disabled_but_key_set_warns():
    r = doc.analyze_cloud(_cfg(cloud_enabled=False), {"ANTHROPIC_API_KEY": "sk-x"},
                          0, 0, all_local=True)
    assert r.status == "WARN"
    assert "ANTHROPIC_API_KEY" in r.detail


def test_analyze_cloud_enabled_with_key_passes():
    r = doc.analyze_cloud(_cfg(cloud_enabled=True), {"ANTHROPIC_API_KEY": "sk-x"},
                          3, 9, all_local=False)
    assert r.status == "PASS"
    assert "cloud ON" in r.detail


def test_analyze_cloud_enabled_no_key_warns():
    r = doc.analyze_cloud(_cfg(cloud_enabled=True), {}, 0, 0, all_local=False)
    assert r.status == "WARN"
    assert "not set" in r.detail


def test_analyze_cloud_disabled_non_local_warns():
    # disabled + no key, but a tier model is non-local -> WARN (the 5th branch).
    r = doc.analyze_cloud(_cfg(cloud_enabled=False), {}, 0, 0, all_local=False)
    assert r.status == "WARN"
    assert "non-local" in r.detail


def test_derive_mode_tier_shadow_cloud_off():
    r = doc.derive_mode(_cfg(cap_enabled=True, cloud_enabled=False), {})
    assert r.status == "PASS"
    assert "capability routing: shadow" in r.detail
    assert "Cloud: off" in r.detail


def test_derive_mode_cap_off_cloud_on_key_set():
    r = doc.derive_mode(_cfg(cap_enabled=False, cloud_enabled=True),
                        {"ANTHROPIC_API_KEY": "sk-x"})
    assert "capability routing: off" in r.detail
    assert "Cloud: on" in r.detail and "key set" in r.detail


def test_derive_mode_cloud_on_key_missing():
    r = doc.derive_mode(_cfg(cap_enabled=False, cloud_enabled=True), {})
    assert "MISSING" in r.detail


import sqlite3  # noqa: E402
from unittest.mock import patch, MagicMock  # noqa: E402


def _resp(payload, status=200):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    return r


def test_fetch_version_pass():
    with patch("doctor.httpx.Client") as ClientCls:
        client = ClientCls.return_value.__enter__.return_value
        client.get = MagicMock(return_value=_resp({"version": "0.5.0"}))
        r = doc._fetch_version()
    assert r.status == "PASS"
    assert "0.5.0" in r.detail


def test_fetch_version_connect_error_fails():
    import httpx
    with patch("doctor.httpx.Client") as ClientCls:
        client = ClientCls.return_value.__enter__.return_value
        client.get = MagicMock(side_effect=httpx.ConnectError("down"))
        r = doc._fetch_version()
    assert r.status == "FAIL"
    assert "ollama serve" in (r.fix or "")


def test_fetch_tags_returns_name_set():
    with patch("doctor.httpx.Client") as ClientCls:
        client = ClientCls.return_value.__enter__.return_value
        client.get = MagicMock(return_value=_resp(
            {"models": [{"name": "m1"}, {"name": "m2"}]}))
        names = doc._fetch_tags()
    assert names == {"m1", "m2"}


def test_fetch_tags_error_returns_empty_set():
    import httpx
    with patch("doctor.httpx.Client") as ClientCls:
        client = ClientCls.return_value.__enter__.return_value
        client.get = MagicMock(side_effect=httpx.ConnectError("down"))
        assert doc._fetch_tags() == set()


def test_load_cloud_attempts_counts(tmp_path):
    db = tmp_path / "rt.sqlite"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE decisions (ts REAL, cloud_attempted INTEGER)")
        c.execute("INSERT INTO decisions VALUES (1.0, 1)")
        c.execute("INSERT INTO decisions VALUES (2.0, 0)")
        c.execute("INSERT INTO decisions VALUES (3.0, 1)")
    assert doc.load_cloud_attempts(str(db)) == (2, 2)


def test_load_cloud_attempts_missing_db_is_zero(tmp_path):
    assert doc.load_cloud_attempts(str(tmp_path / "nope.sqlite")) == (0, 0)


import json as _json  # noqa: E402


def _results():
    return [
        doc.CheckResult("ollama", "PASS", "HTTP 200 (v0.5.0)"),
        doc.CheckResult("models", "FAIL", "missing: m2", fix="run: ollama pull m2"),
        doc.CheckResult("mode", "PASS", "Routing: tier-based | Cloud: off"),
    ]


def test_render_text_shows_markers_and_fix():
    text = doc.render_text(_results())
    assert "[PASS]" in text and "[FAIL]" in text
    assert "fix: run: ollama pull m2" in text
    assert "FAIL: 1 check(s) failed." in text
    # ASCII-only output
    text.encode("ascii")


def test_render_text_all_pass_summary():
    text = doc.render_text([doc.CheckResult("a", "PASS", "ok")])
    assert "All checks passed." in text


def test_render_json_shape_and_all_ok():
    out = doc.render_json(_results())
    parsed = _json.loads(out)
    assert parsed["all_ok"] is False     # a FAIL present
    assert len(parsed["results"]) == 3
    assert parsed["results"][1]["fix"] == "run: ollama pull m2"


def test_build_parser_defaults():
    args = doc.build_parser().parse_args([])
    assert args.json is False and args.cloud_audit is False
    assert args.mode is False and args.skip_models is False


def test_build_parser_flags():
    args = doc.build_parser().parse_args(
        ["--json", "--cloud-audit", "--mode", "--skip-models"])
    assert args.json and args.cloud_audit and args.mode and args.skip_models


def test_run_default_runs_all_checks(monkeypatch, capsys):
    monkeypatch.setattr(doc, "_fetch_version",
                        lambda: doc.CheckResult("ollama", "PASS", "HTTP 200"))
    monkeypatch.setattr(doc, "_fetch_tags", lambda: {"m1"})
    monkeypatch.setattr(doc, "load_cloud_attempts", lambda db: (0, 0))
    rc = doc.run(doc.build_parser().parse_args([]))
    out = capsys.readouterr().out
    # 5 checks: ollama, config, models, cloud-audit, mode
    for name in ("ollama", "config", "models", "cloud-audit", "Routing"):
        assert name in out
    assert rc in (0, 1)


def test_run_ollama_down_skips_models(monkeypatch, capsys):
    monkeypatch.setattr(doc, "_fetch_version",
                        lambda: doc.CheckResult("ollama", "FAIL", "down",
                                                fix="start Ollama"))
    monkeypatch.setattr(doc, "load_cloud_attempts", lambda db: (0, 0))
    # _fetch_tags must NOT be called when ollama is down; make it raise if it is.
    monkeypatch.setattr(doc, "_fetch_tags",
                        lambda: (_ for _ in ()).throw(AssertionError("called")))
    rc = doc.run(doc.build_parser().parse_args([]))
    out = capsys.readouterr().out
    assert "skipped (Ollama unreachable)" in out
    assert rc == 1     # the ollama FAIL


def test_run_cloud_audit_only(monkeypatch, capsys):
    monkeypatch.setattr(doc, "load_cloud_attempts", lambda db: (0, 0))
    rc = doc.run(doc.build_parser().parse_args(["--cloud-audit"]))
    out = capsys.readouterr().out
    assert "cloud-audit" in out
    assert "ollama" not in out and "Routing" not in out


def test_run_mode_only(capsys):
    doc.run(doc.build_parser().parse_args(["--mode"]))
    out = capsys.readouterr().out
    assert "Routing:" in out and "cloud-audit" not in out


def test_run_skip_models(monkeypatch, capsys):
    monkeypatch.setattr(doc, "_fetch_version",
                        lambda: doc.CheckResult("ollama", "PASS", "HTTP 200"))
    monkeypatch.setattr(doc, "load_cloud_attempts", lambda db: (0, 0))
    monkeypatch.setattr(doc, "_fetch_tags",
                        lambda: (_ for _ in ()).throw(AssertionError("called")))
    doc.run(doc.build_parser().parse_args(["--skip-models"]))
    out = capsys.readouterr().out
    assert "models" not in out
