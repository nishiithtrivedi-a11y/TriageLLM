"""Unit tests for task_benchmark (Issue #24). Pure + mocked-I/O, no Ollama."""
import json

import task_benchmark as tb


def test_testable_categories_are_the_eight():
    assert tb.TESTABLE_CATEGORIES == [
        "quick_question", "explanation_or_summary", "structured_output",
        "analytical_task", "creative_generation", "modification_or_edit",
        "multi_step_or_planning", "long_context",
    ]
    # high_risk / default are routing-policy, not capability -> excluded.
    assert "high_risk" not in tb.TESTABLE_CATEGORIES
    assert "default" not in tb.TESTABLE_CATEGORIES


def test_objective_categories():
    assert tb.OBJECTIVE_CATEGORIES == {"structured_output", "modification_or_edit"}


def test_needs_critic_true_for_soft_false_for_objective():
    assert tb.needs_critic("quick_question") is True
    assert tb.needs_critic("long_context") is True
    assert tb.needs_critic("structured_output") is False
    assert tb.needs_critic("modification_or_edit") is False
    # Unknown category defaults to soft (matches success.score_output fallback).
    assert tb.needs_critic("something_new") is True


def _tags():
    # Mimics Ollama /api/tags "models" entries.
    return [
        {"name": "qwen3-coder:30b", "size": 18_000_000_000},
        {"name": "deepseek-coder-v2:16b", "size": 9_000_000_000},
        {"name": "qwen2.5:0.5b", "size": 400_000_000},          # critic/classifier
        {"name": "nomic-embed-text:latest", "size": 270_000_000},  # embedding
    ]


def test_filter_models_skips_critic_classifier_and_embed():
    out = tb.filter_models(_tags(), None, "qwen2.5:0.5b", "qwen2.5:0.5b")
    names = [m.name for m in out]
    assert names == ["qwen3-coder:30b", "deepseek-coder-v2:16b"]
    # size is captured for the tie-break.
    assert out[0].size == 18_000_000_000


def test_filter_models_override_picks_named_only():
    out = tb.filter_models(_tags(), ["deepseek-coder-v2:16b"], "qwen2.5:0.5b", "qwen2.5:0.5b")
    assert [m.name for m in out] == ["deepseek-coder-v2:16b"]
    assert out[0].size == 9_000_000_000


def test_filter_models_override_skips_uninstalled():
    out = tb.filter_models(_tags(), ["not-installed:1b", "qwen3-coder:30b"],
                           "qwen2.5:0.5b", "qwen2.5:0.5b")
    assert [m.name for m in out] == ["qwen3-coder:30b"]


def test_filter_models_override_accepts_comma_string():
    out = tb.filter_models(_tags(), "qwen3-coder:30b,deepseek-coder-v2:16b",
                           "qwen2.5:0.5b", "qwen2.5:0.5b")
    assert [m.name for m in out] == ["qwen3-coder:30b", "deepseek-coder-v2:16b"]


def _pr(success, latency, reason="x"):
    return tb.PromptResult(success, latency, reason)


def test_aggregate_success_rate_and_latency():
    results = [_pr(True, 2.0), _pr(True, 4.0), _pr(False, 6.0), _pr(True, 8.0)]
    cr = tb.aggregate("quick_question", results)
    assert cr.n_prompts == 4
    assert cr.n_passed == 3
    assert cr.success_rate == 0.75
    assert cr.confidence == "soft"
    assert cr.latency_p50_s == 5.0      # median of [2,4,6,8]
    assert cr.latency_max_s == 8.0
    assert cr.latency_p95_s is not None  # n >= 4


def test_aggregate_p95_none_when_small_sample():
    cr = tb.aggregate("quick_question", [_pr(True, 2.0), _pr(True, 4.0)])
    assert cr.latency_p95_s is None      # n < 4
    assert cr.success_rate == 1.0


def test_aggregate_objective_category_is_hard_confidence():
    cr = tb.aggregate("structured_output", [_pr(True, 1.0), _pr(False, 2.0)])
    assert cr.confidence == "hard"
    assert cr.success_rate == 0.5


def test_aggregate_all_fail_is_zero_rate():
    cr = tb.aggregate("analytical_task", [_pr(False, 1.0), _pr(False, 2.0)])
    assert cr.success_rate == 0.0
    assert cr.n_passed == 0


def _cr(success_rate, p50, confidence="soft"):
    return tb.CategoryResult(success_rate, 5, int(success_rate * 5),
                             confidence, p50, None, p50)


def test_recommend_rung1_picks_highest_success_rate():
    results = {
        "big": {"quick_question": _cr(0.8, 9.0)},
        "small": {"quick_question": _cr(0.4, 2.0)},
    }
    sizes = {"big": 18_000, "small": 400}
    recs = tb.recommend(results, sizes)
    assert recs["quick_question"].model == "big"
    assert recs["quick_question"].decided_by == "success_rate"
    assert recs["quick_question"].warning is None


def test_recommend_rung2_breaks_tie_on_latency():
    results = {
        "slow": {"quick_question": _cr(0.8, 9.0)},
        "fast": {"quick_question": _cr(0.8, 3.0)},
    }
    sizes = {"slow": 400, "fast": 18_000}   # size must NOT decide here
    recs = tb.recommend(results, sizes)
    assert recs["quick_question"].model == "fast"
    assert recs["quick_question"].decided_by == "latency"


def test_recommend_rung3_breaks_tie_on_size():
    results = {
        "big": {"quick_question": _cr(0.8, 5.0)},
        "small": {"quick_question": _cr(0.8, 5.0)},
    }
    sizes = {"big": 18_000, "small": 400}
    recs = tb.recommend(results, sizes)
    assert recs["quick_question"].model == "small"
    assert recs["quick_question"].decided_by == "size"


def test_recommend_no_model_passed_warns():
    results = {
        "a": {"analytical_task": _cr(0.0, 5.0)},
        "b": {"analytical_task": _cr(0.0, 3.0)},
    }
    sizes = {"a": 400, "b": 18_000}
    recs = tb.recommend(results, sizes)
    assert recs["analytical_task"].warning == "no-model-passed"
    assert recs["analytical_task"].model == "b"   # still names the fastest


def test_recommend_carries_confidence():
    results = {"a": {"structured_output": _cr(0.6, 5.0, confidence="hard")}}
    recs = tb.recommend(results, {"a": 400})
    assert recs["structured_output"].confidence == "hard"


def test_build_priors_shape_and_serialization():
    models = [tb.ModelInfo("big", 18_000), tb.ModelInfo("small", 400)]
    results = {
        "big": {"structured_output": _cr(0.8, 6.0, confidence="hard")},
        "small": {"structured_output": _cr(0.4, 2.0, confidence="hard")},
    }
    recs = tb.recommend(results, {"big": 18_000, "small": 400})
    priors = tb.build_priors(models, "full", 5, results, recs,
                             generated_at="2026-05-28T00:00:00Z")
    assert priors["schema_version"] == 1
    assert priors["generated_at"] == "2026-05-28T00:00:00Z"
    assert priors["mode"] == "full"
    assert priors["prompts_per_category"] == 5
    assert priors["models"] == ["big", "small"]
    # results serialized as plain dicts
    so = priors["results"]["big"]["structured_output"]
    assert so["success_rate"] == 0.8
    assert so["confidence"] == "hard"
    assert "latency_p50_s" in so and "latency_p95_s" in so
    # recommendations serialized
    rec = priors["recommendations"]["structured_output"]
    assert rec["model"] == "big"
    assert rec["decided_by"] == "success_rate"
    assert rec["warning"] is None
    # whole thing is JSON-serializable
    json.dumps(priors)


def test_write_priors_atomic_round_trip(tmp_path):
    path = tmp_path / "out" / "benchmark_results.json"
    path.parent.mkdir(parents=True)
    obj = {"schema_version": 1, "models": ["a"]}
    tb.write_priors(str(path), obj)
    assert json.loads(path.read_text(encoding="utf-8")) == obj
    # overwrite works and leaves no .tmp files behind
    tb.write_priors(str(path), {"schema_version": 1, "models": ["b"]})
    assert json.loads(path.read_text(encoding="utf-8"))["models"] == ["b"]
    leftover = list(path.parent.glob("*.tmp"))
    assert leftover == []


def _sample_priors():
    models = [tb.ModelInfo("big", 18_000), tb.ModelInfo("small", 400)]
    results = {
        "big": {"quick_question": _cr(0.8, 3.0)},
        "small": {"quick_question": _cr(0.4, 2.0)},
    }
    recs = tb.recommend(results, {"big": 18_000, "small": 400})
    return tb.build_priors(models, "full", 5, results, recs,
                           generated_at="2026-05-28T00:00:00Z")


def test_render_report_text_mentions_winner_and_basis():
    text = tb.render_report(_sample_priors(), json_mode=False)
    assert "quick_question" in text
    assert "big" in text
    assert "success_rate" in text       # decided_by shown
    assert "mode=full" in text


def test_render_report_json_round_trips():
    priors = _sample_priors()
    out = tb.render_report(priors, json_mode=True)
    assert json.loads(out) == priors


def test_render_report_shows_no_model_passed_warning():
    models = [tb.ModelInfo("a", 400)]
    results = {"a": {"analytical_task": _cr(0.0, 5.0)}}
    recs = tb.recommend(results, {"a": 400})
    priors = tb.build_priors(models, "quick", 2, results, recs,
                             generated_at="2026-05-28T00:00:00Z")
    text = tb.render_report(priors, json_mode=False)
    assert "no-model-passed" in text


from unittest.mock import patch, AsyncMock, MagicMock  # noqa: E402


def _resp(payload):
    r = MagicMock()
    r.json.return_value = payload
    return r


async def test_fetch_tags_returns_models_list():
    with patch("task_benchmark.httpx.AsyncClient") as ClientCls:
        client = ClientCls.return_value.__aenter__.return_value
        client.get = AsyncMock(return_value=_resp({"models": [{"name": "m", "size": 1}]}))
        tags = await tb._fetch_tags()
    assert tags == [{"name": "m", "size": 1}]


async def test_discover_models_wires_fetch_and_filter():
    tags = [
        {"name": "qwen3-coder:30b", "size": 18_000_000_000},
        {"name": "qwen2.5:0.5b", "size": 400_000_000},
    ]

    class _Cfg:
        classifier_model = "qwen2.5:0.5b"
        critic_model = "qwen2.5:0.5b"

    with patch("task_benchmark._fetch_tags", AsyncMock(return_value=tags)):
        out = await tb.discover_models(override=None, config=_Cfg())
    assert [m.name for m in out] == ["qwen3-coder:30b"]


async def test_run_prompt_returns_text_and_latency():
    with patch("task_benchmark.httpx.AsyncClient") as ClientCls:
        client = ClientCls.return_value.__aenter__.return_value
        client.post = AsyncMock(return_value=_resp({"response": "hello world"}))
        text, latency = await tb.run_prompt("m", "hi")
    assert text == "hello world"
    assert isinstance(latency, float) and latency >= 0.0


async def test_run_prompt_raises_on_connect_error():
    import httpx
    with patch("task_benchmark.httpx.AsyncClient") as ClientCls:
        client = ClientCls.return_value.__aenter__.return_value
        client.post = AsyncMock(side_effect=httpx.ConnectError("down"))
        try:
            await tb.run_prompt("m", "hi")
            raised = False
        except httpx.ConnectError:
            raised = True
    assert raised


def _mini_prompts():
    # 1 prompt per testable category keeps the orchestration test fast.
    return {cat: ["prompt for " + cat] for cat in tb.TESTABLE_CATEGORIES}


class _Cfg2:
    classifier_model = "qwen2.5:0.5b"
    critic_model = "qwen2.5:0.5b"
    critic_timeout_s = 30.0
    critic_cpu_only = True
    critic_pass_threshold = 4


async def test_run_benchmark_produces_wellformed_priors(tmp_path):
    out_path = tmp_path / "bench.json"
    models = [tb.ModelInfo("m1", 1000)]

    # structured_output answered with valid JSON -> objective hard pass.
    # other categories answered with a long, clean sentence -> floor passes,
    # critic decides.
    async def fake_run_prompt(model, text, timeout_s=120.0):
        if "structured_output" in text:
            return ('{"a": 1}', 1.0)
        return ("This is a sufficiently long and clean answer sentence.", 2.0)

    critic = AsyncMock(return_value=5)   # soft categories pass

    with patch("task_benchmark.discover_models", AsyncMock(return_value=models)), \
         patch("task_benchmark._warmup", AsyncMock(return_value=None)), \
         patch("task_benchmark.run_prompt", side_effect=fake_run_prompt), \
         patch("task_benchmark.critic_score", critic):
        priors = await tb.run_benchmark(_mini_prompts(), mode="quick",
                                        config=_Cfg2(), output=str(out_path))

    assert priors["mode"] == "quick"
    assert priors["models"] == ["m1"]
    # structured_output passed objectively (valid JSON)
    assert priors["results"]["m1"]["structured_output"]["success_rate"] == 1.0
    assert priors["results"]["m1"]["structured_output"]["confidence"] == "hard"
    # a soft category passed via the critic
    assert priors["results"]["m1"]["quick_question"]["success_rate"] == 1.0
    # file written
    assert out_path.exists()


async def test_run_benchmark_skips_critic_for_objective_categories(tmp_path):
    models = [tb.ModelInfo("m1", 1000)]
    critic = AsyncMock(return_value=5)

    async def fake_run_prompt(model, text, timeout_s=120.0):
        return ('{"a": 1}', 1.0)   # valid JSON for everything

    # Only the two objective categories.
    prompts = {"structured_output": ["p"], "modification_or_edit": ["p"]}

    with patch("task_benchmark.discover_models", AsyncMock(return_value=models)), \
         patch("task_benchmark._warmup", AsyncMock(return_value=None)), \
         patch("task_benchmark.run_prompt", side_effect=fake_run_prompt), \
         patch("task_benchmark.critic_score", critic):
        await tb.run_benchmark(prompts, mode="quick", config=_Cfg2(),
                               output=str(tmp_path / "bench_obj.json"))
    critic.assert_not_called()   # objective categories never call the critic


async def test_run_benchmark_generation_error_counts_as_failure(tmp_path):
    models = [tb.ModelInfo("m1", 1000)]

    async def boom(model, text, timeout_s=120.0):
        import httpx
        raise httpx.ConnectError("down")

    with patch("task_benchmark.discover_models", AsyncMock(return_value=models)), \
         patch("task_benchmark._warmup", AsyncMock(return_value=None)), \
         patch("task_benchmark.run_prompt", side_effect=boom), \
         patch("task_benchmark.critic_score", AsyncMock(return_value=5)):
        priors = await tb.run_benchmark({"quick_question": ["p"]}, mode="quick",
                                        config=_Cfg2(), output=str(tmp_path / "b.json"))
    assert priors["results"]["m1"]["quick_question"]["success_rate"] == 0.0
    assert priors["results"]["m1"]["quick_question"]["n_passed"] == 0


async def test_run_benchmark_no_models_raises(tmp_path):
    with patch("task_benchmark.discover_models", AsyncMock(return_value=[])):
        try:
            await tb.run_benchmark(_mini_prompts(), mode="quick", config=_Cfg2(),
                                   output=str(tmp_path / "b.json"))
            raised = False
        except SystemExit:
            raised = True
    assert raised


import benchmark_prompts as bp  # noqa: E402


def test_prompts_cover_all_testable_categories_with_five_each():
    for cat in tb.TESTABLE_CATEGORIES:
        assert cat in bp.PROMPTS, "missing prompts for " + cat
        assert len(bp.PROMPTS[cat]) >= 5, "need >=5 prompts for " + cat
    # no stray non-testable categories
    assert set(bp.PROMPTS.keys()) == set(tb.TESTABLE_CATEGORIES)


def test_quick_subset_first_two_nonempty():
    for cat in tb.TESTABLE_CATEGORIES:
        first_two = bp.PROMPTS[cat][:2]
        assert len(first_two) == 2
        assert all(isinstance(s, str) and s.strip() for s in first_two)


def test_structured_output_prompts_request_json():
    # success = is_valid_json, so each prompt MUST ask for JSON.
    for prompt in bp.PROMPTS["structured_output"]:
        assert "json" in prompt.lower()


def test_modification_prompts_request_python_code():
    # success = is_valid_code_edit (diff or parseable Python). Each prompt must
    # steer the model to emit Python (a "def" / "function") or a diff.
    for prompt in bp.PROMPTS["modification_or_edit"]:
        low = prompt.lower()
        assert ("def " in prompt) or ("function" in low) or ("diff" in low)


def test_all_prompts_are_ascii():
    for cat, plist in bp.PROMPTS.items():
        for prompt in plist:
            prompt.encode("ascii")   # raises if a non-ASCII char slipped in


import benchmark as bench_cli  # noqa: E402


def test_build_parser_supports_task_flags():
    parser = bench_cli.build_parser()
    args = parser.parse_args(["--tasks", "--quick", "--models", "a,b",
                              "--output", "out.json", "--json"])
    assert args.tasks is True
    assert args.quick is True
    assert args.models == "a,b"
    assert args.output == "out.json"
    assert args.json is True


def test_build_parser_defaults_to_critic_latency_mode():
    parser = bench_cli.build_parser()
    args = parser.parse_args([])
    assert args.tasks is False
    assert args.quick is False
    assert args.models is None
    assert args.output == "benchmark_results.json"
    assert args.runs == 10   # existing critic-latency knob preserved
