"""Rule-based classifier — synchronous, no I/O."""
import pytest

from router_hook import classify, classify_rules


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("rename foo to bar in this function", "S"),
        ("what is the syntax for a Python list comprehension?", "S"),
        (
            "Implement a function that parses CSV and returns a list of dicts. "
            "Add tests.",
            "M",
        ),
        (
            "Refactor this 4-file module to remove duplication and add proper "
            "error handling.\n```py\nfrom x import y\n```\n"
            "Files: src/a.py, src/b.py, src/c.py, src/d.py",
            "L",
        ),
        (
            "Design a distributed rate limiter with strong consistency guarantees "
            "across 3 regions. Address race conditions and authentication.",
            "XL",
        ),
    ],
)
def test_rule_classifier_tiers(prompt: str, expected: str) -> None:
    tier, _score, _signals = classify([{"role": "user", "content": prompt}])
    assert tier == expected


def test_traceback_boosts_tier() -> None:
    text = (
        "fix this:\nTraceback (most recent call last):\n  File \"a.py\", line 3\n"
        "ZeroDivisionError: division by zero"
    )
    _tier, _score, signals = classify_rules(text)
    assert "traceback" in signals


def test_multimodal_content_blocks_extracted() -> None:
    """A message with structured content blocks should still classify."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Design a distributed system"},
                {"type": "text", "text": "with race conditions and authentication"},
            ],
        }
    ]
    tier, _score, _signals = classify(messages)
    assert tier == "XL"
