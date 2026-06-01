import os
import sys
from pathlib import Path

import pytest

# H-8 (safe slice): guarantee tests are hermetic w.r.t. the critic warmup
# BEFORE router_hook is ever imported. Setting this at collection time means a
# bare `TierRouter()` in any fixture can never fire a real ~30s Ollama call,
# regardless of whether the developer remembered to export it in their shell.
# (The deeper "move the singleton off module import" refactor is tracked in
# Issue #1 as H-8; it's deferred because config.yaml's callback contract
# references router_hook.tier_router_instance at module level.)
os.environ.setdefault("TRIAGELLM_SKIP_WARMUP", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def isolated_router_db(tmp_path, monkeypatch):
    """Every test gets its own throwaway SQLite file — never write to the prod DB."""
    import router_hook
    monkeypatch.setattr(router_hook, "DB_PATH", tmp_path / "router_decisions.sqlite")
