"""DEF-001 regression: `health.py --json` stdout must be valid JSON only
(no router import banner / warmup line leaking in). Hermetic: the network
probes are mocked, so this never touches real Ollama."""
import contextlib
import io
import json
import sys

import pytest

import health


@pytest.mark.asyncio
async def test_health_json_stdout_is_clean_json(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["health.py", "--json", "--skip-models"])
    from unittest.mock import AsyncMock
    with patch_all():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            await health.main_async()
    out = buf.getvalue()
    data = json.loads(out)                 # raises if banner contaminated stdout
    assert "results" in data
    assert isinstance(data["results"], list)


def patch_all():
    """Patch the two network probes so the JSON branch runs without Ollama."""
    from unittest.mock import AsyncMock, patch
    return _MultiPatch(
        patch.object(health, "_check_url", AsyncMock(return_value=("PASS", "HTTP 200"))),
        patch.object(health, "_check_critic_responds", AsyncMock(return_value=("PASS", "score=3, 0.50s", 0.5))),
    )


class _MultiPatch:
    def __init__(self, *ctxs):
        self.ctxs = ctxs

    def __enter__(self):
        for c in self.ctxs:
            c.__enter__()
        return self

    def __exit__(self, *a):
        for c in reversed(self.ctxs):
            c.__exit__(*a)
        return False
