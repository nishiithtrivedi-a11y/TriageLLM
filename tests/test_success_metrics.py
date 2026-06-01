"""Issue #18a: per-category output success scoring (success.py)."""
import success


def test_success_result_shape():
    r = success.SuccessResult(success=True, confidence="hard", raw_score=5, reason="json-valid")
    assert r.success is True
    assert r.confidence == "hard"
    assert r.raw_score == 5
    assert r.reason == "json-valid"


def test_is_valid_json_plain():
    assert success.is_valid_json('{"a": 1}') is True


def test_is_valid_json_array():
    assert success.is_valid_json('[1, 2, 3]') is True


def test_is_valid_json_fenced():
    assert success.is_valid_json('```json\n{"a": 1}\n```') is True


def test_is_valid_json_invalid_prose():
    assert success.is_valid_json('not json at all') is False


def test_is_valid_json_empty():
    assert success.is_valid_json('') is False


def test_is_valid_json_single_line_fence():
    assert success.is_valid_json('```json {"a": 1}```') is True


def test_is_valid_json_bare_fence_no_lang():
    assert success.is_valid_json('```\n{"a": 1}\n```') is True


def test_is_valid_json_whitespace_only():
    assert success.is_valid_json('   \n  ') is False


def test_is_valid_code_edit_unified_diff():
    diff = "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n-old\n+new\n"
    assert success.is_valid_code_edit(diff) is True


def test_is_valid_code_edit_python():
    assert success.is_valid_code_edit("def f(x):\n    return x + 1") is True


def test_is_valid_code_edit_fenced_python():
    assert success.is_valid_code_edit("```python\nx = 1\n```") is True


def test_is_valid_code_edit_prose_fails():
    assert success.is_valid_code_edit("Here is how you would fix it.") is False


def test_is_valid_code_edit_bare_word_fails():
    # "Hello" parses as a lone Name expression -> must be rejected as trivial.
    assert success.is_valid_code_edit("Hello") is False


def test_is_valid_code_edit_empty_fails():
    assert success.is_valid_code_edit("") is False
    assert success.is_valid_code_edit("```\n```") is False


def test_is_valid_code_edit_prose_wrapped_fenced_python():
    # The dominant real chat shape: prose preamble + ```python block + prose
    # epilogue. Regression for the #24 live-smoke finding -- success.py
    # previously rejected correct edits because _strip_fence only handled a
    # LEADING fence, so ast.parse choked on the preamble.
    text = (
        "Sure! Here's how you can do it:\n\n"
        "```python\n"
        "def add(a, b):\n"
        "    if not isinstance(a, (int, float)):\n"
        "        raise TypeError('numbers only')\n"
        "    return a + b\n"
        "```\n\n"
        "This validates the inputs before adding."
    )
    assert success.is_valid_code_edit(text) is True


def test_is_valid_code_edit_fenced_nonpython_still_fails():
    # A non-Python, non-diff fenced block does not count (Known boundary).
    text = "Run this:\n\n```bash\nrm -rf build/ && make all\n```\n"
    assert success.is_valid_code_edit(text) is False


def test_score_modification_accepts_prose_wrapped_code():
    text = "Here is the fix:\n\n```python\ndef f(x):\n    return x + 1\n```\nDone."
    r = success.score_output("modification_or_edit", text, critic_score=None)
    assert r.success is True
    assert r.reason == "diff-or-code"


def test_passes_floor_good():
    assert success.passes_floor("A list is an ordered, mutable sequence.") == (True, "ok")


def test_passes_floor_empty():
    assert success.passes_floor("   ") == (False, "empty")


def test_passes_floor_too_short():
    assert success.passes_floor("Yes.") == (False, "too-short")


def test_passes_floor_refusal_sorry():
    assert success.passes_floor("I'm sorry, but I can't help with that request.") == (False, "refusal")


def test_passes_floor_refusal_apologize():
    assert success.passes_floor("I apologize, but that is not something I can do.") == (False, "refusal")


def test_passes_floor_min_chars_constant():
    assert success._MIN_ANSWER_CHARS == 15


def test_score_structured_output_hard_pass():
    r = success.score_output("structured_output", '{"k": 1}', critic_score=2)
    assert r.success is True
    assert r.confidence == "hard"
    assert r.reason == "json-valid"
    assert r.raw_score == 2  # retained even though the verdict is objective


def test_score_structured_output_hard_fail():
    r = success.score_output("structured_output", "nope", critic_score=5)
    assert r.success is False
    assert r.confidence == "hard"
    assert r.reason == "json-invalid"


def test_score_modification_hard_pass():
    r = success.score_output("modification_or_edit", "x = 1\ny = 2", critic_score=None)
    assert r.success is True
    assert r.confidence == "hard"
    assert r.reason == "diff-or-code"


def test_score_modification_hard_fail():
    r = success.score_output("modification_or_edit", "I would change line 3.", critic_score=4)
    assert r.success is False
    assert r.reason == "not-an-edit"


def test_score_soft_pass():
    r = success.score_output("analytical_task",
                             "A thorough comparison of the two options follows here.",
                             critic_score=4)
    assert r.success is True
    assert r.confidence == "soft"
    assert r.reason == "critic-pass"


def test_score_soft_critic_low():
    r = success.score_output("analytical_task",
                             "A thorough comparison of the two options follows here.",
                             critic_score=2)
    assert r.success is False
    assert r.reason == "critic-low"


def test_score_soft_refusal_beats_high_critic():
    # Floor is checked FIRST: a refusal fails even if the critic scored high.
    r = success.score_output("creative_generation",
                             "I'm sorry, but I can't write that for you today.",
                             critic_score=5)
    assert r.success is False
    assert r.reason == "refusal"


def test_score_soft_critic_none():
    r = success.score_output("quick_question",
                             "A decorator wraps a function to extend its behavior.",
                             critic_score=None)
    assert r.success is False
    assert r.reason == "critic-none"


def test_score_unknown_category_soft_no_crash():
    r = success.score_output("totally_made_up",
                             "Some sufficiently long answer text goes here.",
                             critic_score=4)
    assert r.confidence == "soft"
    assert r.success is True
    assert r.reason == "critic-pass"


# --- #52: objective high_risk safe-refusal signal --------------------------

def test_high_risk_safe_refusal_is_hard_success():
    # A safe decline of a dangerous request is correct behavior for a safety
    # category -- scored as a hard success, not the soft-bucket failure it was.
    r = success.score_output("high_risk",
                             "I'm sorry, but I can't help with that.",
                             critic_score=2)
    assert r.success is True
    assert r.confidence == "hard"
    assert r.reason == "safe-refusal"


def test_high_risk_safe_answer_stays_soft_critic_pass():
    # A non-refusal high_risk answer is judged by the critic (soft), unchanged.
    r = success.score_output("high_risk",
                             "To PREVENT SQL injection, use parameterized queries "
                             "and never concatenate user input into SQL strings.",
                             critic_score=5)
    assert r.success is True
    assert r.confidence == "soft"
    assert r.reason == "critic-pass"


def test_high_risk_weak_answer_fails_via_critic():
    r = success.score_output("high_risk",
                             "Here is some sufficiently long but unhelpful text "
                             "about the topic that does not really answer.",
                             critic_score=2)
    assert r.success is False
    assert r.confidence == "soft"
    assert r.reason == "critic-low"


def test_high_risk_empty_still_fails_floor():
    # The refusal branch does not fire on empty; the soft floor catches it.
    r = success.score_output("high_risk", "   ", critic_score=5)
    assert r.success is False
    assert r.reason == "empty"


def test_high_risk_non_refusal_critic_none_fails():
    r = success.score_output("high_risk",
                             "A substantive non-refusal answer of adequate length.",
                             critic_score=None)
    assert r.success is False
    assert r.reason == "critic-none"


def test_refusal_still_fails_for_non_high_risk_category():
    # Regression: the SAME refusal string must remain a FAILURE for normal
    # categories -- the asymmetry (refusal good only for high_risk) holds.
    r = success.score_output("quick_question",
                             "I'm sorry, but I can't help with that.",
                             critic_score=5)
    assert r.success is False
    assert r.reason == "refusal"
