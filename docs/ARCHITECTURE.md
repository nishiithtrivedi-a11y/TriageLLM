# TriageLLM architecture

The system is a single Python module (`router_hook.py`) registered as a
LiteLLM `CustomLogger` callback, plus four CLI scripts (`stats.py`,
`benchmark.py`, `health.py`, `doctor.py`) and a SQLite database. Everything runs locally;
no external services required (except optionally a cloud LLM for last-resort
escalation).

---

## Component map

![TriageLLM request flow: client to LiteLLM proxy to TierRouter to Ollama, with optional cloud escalation and a SQLite ledger](assets/architecture.svg)

---

## Hook lifecycle

When LiteLLM receives a request:

| Step | Method | Purpose |
|---|---|---|
| 1 | `async_pre_call_hook` | Classify the prompt, rewrite `data["model"]` from `local-auto` to the chosen tier alias. Stash state in `data["_router_state"]` for the post-call step. |
| 2 | *(LiteLLM dispatches upstream)* | Calls Ollama for the chosen worker model. |
| 3a | `async_post_call_success_hook` | Non-streaming responses: critique, walk escalation chain, optionally call cloud, optionally rewrite response to handoff message, write SQLite row. |
| 3b | `async_post_call_streaming_iterator_hook` | Streaming responses: pass chunks through, accumulate full text, critique at end, optionally append one extra chunk with handoff note. |

Both paths log via `_log_decision()` (identical ledger/dashboard shape) and
share the **same** critique decision via `_critique_outcome()` — so
`pass` / `soft_pass` / `escalate` mean the same thing in both. The streaming
path can't re-route mid-stream, so an `escalate` outcome becomes "append a
handoff note" rather than "try a bigger tier". (Before v0.1.2 these two paths
had drifted; H-7 converged them.)

---

## Classification

Two classifiers run in series:

1. **Rule-based** (`classify_rules`) — always runs. Scores the prompt using
   token count, code-block count, file-path count, traceback presence, plus
   tier-specific keyword regexes. Maps the score to one of S/M/L/XL.
2. **LLM-based** (`classify_llm`) — runs only if the prompt is longer than
   `llm_classifier_min_chars` (default 250). Asks a 0.5B Qwen model to output
   a single letter S/M/L/XL.

The LLM classifier may only **escalate** the tier the rules chose. This is
deliberate — see [JOURNEY.md](JOURNEY.md) ("A few smaller things I learned"). Both classifiers run
CPU-pinned to avoid the eviction cascade.

---

## Critic

The critic (`critic_score`) sends the user's prompt + the worker's answer to
a 0.5B model and asks for a digit 1-5. It's **CPU-pinned by default**
(`num_gpu: 0`) and held in RAM with `keep_alive: -1` for the entire proxy
session.

Tier S is **never critiqued** — the latency cost (even 0.6 s) exceeds the
benefit for prompts that take 2 s to answer in the first place.

---

## Escalation chain

`_orchestrate()` walks `initial_tier → next → ... → XL → CLOUD → handoff`,
recording every step in a per-request ledger. Stops as soon as the critic
returns a score ≥ `critic_pass_threshold`, or when the chain is exhausted.

### Tier-aware soft-pass (Pattern 3)

If the critic returns `None` (timeout / network error / Ollama hiccup) AND
the current tier is in `soft_pass_tiers` (default: S and M), the orchestrator
**ships the answer instead of escalating**. This breaks the eviction-cascade
loop documented in JOURNEY.md.

Higher tiers (L, XL) by default still escalate on `None`, because critique
matters more there and the resource cost of one extra step is acceptable.

### Ollama fast-fail circuit breaker (DEF-004)

A lightweight `OllamaCircuitBreaker` (CLOSED/OPEN/HALF_OPEN) runs a preflight
`GET /api/tags` in `async_pre_call_hook` **before** the LLM classifier and
before LiteLLM dispatches the worker call. If Ollama is unreachable the request
fails in ~2s (HTTP 503) instead of ~32s, the proxy stays alive, and an
observable ledger row is written (`classifier="ollama-down-fastfail"`). While
the circuit is OPEN, requests fail instantly during a cooldown (no probe); after
cooldown a HALF_OPEN probe re-closes the circuit automatically once Ollama
returns — no proxy restart. Env-tunable (`TRIAGELLM_OLLAMA_FAST_FAIL_*`).
Measured: 32s → 2.31s first fail, 0.01s subsequent, 2.09s auto-recovery
(`tests/evidence/latest_fast_fail_summary.md`).
The preflight runs for `local-auto` and explicit `local-{s,m,l,xl}` pins (#3);
genuinely-external model names skip it.

Runtime model-not-found and cold-load conditions log actionable hints (#20b); the failure path is observability-only and never alters routing.

### Cloud escalation

If `cloud_escalation.enabled` is true and the configured `api_key_env` is
set, the orchestrator makes one cloud LLM call as the final step before
giving up. The cloud model is called via `litellm.acompletion()` so any
LiteLLM-supported provider works (e.g. `anthropic/claude-sonnet-4-6`,
`openai/gpt-5`).

If the cloud call also fails the critic (or `enabled` is false), the response
body is **replaced** with a structured handoff ledger showing every attempt.

---

## SQLite ledger

Every request writes one row to `router_decisions.sqlite`:

| Column | Purpose |
|---|---|
| `ts` | unix timestamp |
| `requested` | original model string (usually `local-auto`) |
| `tier` | initial tier chosen by classifier |
| `model` | initial model alias |
| `tokens`, `score`, `signals` | classifier outputs |
| `classifier` | which classifier path ran (`rules`, `llm-up`, `rules-floor`) |
| `critic` | final critic score, or NULL |
| `escalated_to` | model name if escalation occurred |
| `attempts_json` | full per-attempt ledger (tier, model, tokens, duration, critic_score) |
| `cloud_attempted`, `handoff`, `streamed` | boolean flags |

### Ledger schema reference

Full per-column reference. The 22 columns are grouped by purpose to match how
they arrived over the project's history. Source of truth is
`router_hook.py:_DECISIONS_COLUMNS`; if the table below ever drifts, the code
wins.

#### Core routing (8)

| Column | Type | Meaning | Example |
|---|---|---|---|
| `ts` | REAL | Unix epoch seconds when the decision was logged | `1700000000.123` |
| `requested` | TEXT | The model string the client asked for | `local-auto` |
| `tier` | TEXT | Initial tier the classifier chose | `M` |
| `model` | TEXT | Initial model alias (full LiteLLM-style string) | `ollama_chat/deepseek-coder-v2:16b` |
| `tokens` | INTEGER | Rough prompt-token count (`len(text)//4`) | `250` |
| `score` | INTEGER | Numeric score the rule classifier produced | `80` |
| `signals` | TEXT | Comma-separated rule-label list | `code,large` |
| `classifier` | TEXT | Which path ran: `rules`, `llm-up`, `rules-floor`, `ollama-down-fastfail` | `llm-up` |

#### Critic + escalation (6)

| Column | Type | Meaning | Example |
|---|---|---|---|
| `critic` | INTEGER | Final critic score 1--5, or NULL if not critiqued / critic failed | `4` |
| `escalated_to` | TEXT | Model name if escalation occurred, NULL otherwise | `ollama_chat/qwen3-coder:30b` |
| `attempts_json` | TEXT | JSON array of per-attempt records (see "Per-attempt fields" below) | `[{...}, {...}]` |
| `cloud_attempted` | INTEGER 0/1 | Did the cloud step run? | `0` |
| `handoff` | INTEGER 0/1 | Was the response REPLACED by a structured handoff message? | `0` |
| `streamed` | INTEGER 0/1 | Was the response streamed (so escalation became "append handoff chunk")? | `1` |

#### Capability shadow (8) -- populated only when `TRIAGELLM_CAPABILITY_ROUTING_ENABLED=1`

| Column | Type | Meaning | Example |
|---|---|---|---|
| `cap_category` | TEXT | One of 10 categories or NULL | `modification_or_edit` |
| `cap_recommended_tier` | TEXT | What capability routing WOULD have chosen | `L` |
| `cap_reason_code` | TEXT | Stable reason code | `modification_or_edit:rule-match` |
| `cap_signals` | TEXT | Comma-separated signal labels | `code-edit` |
| `cap_confidence` | REAL | 0.0..1.0 | `0.7` |
| `cap_classifier_used` | TEXT | `rules` / `rules+llm` / `high-risk-precedence` / `default` | `rules` |
| `cap_pack` | TEXT | Active pack(s) | `coder` |
| `cap_agrees_with_tier` | INTEGER 0/1 | Did the cap recommendation match the actual tier? | `0` |

#### Per-attempt fields (inside `attempts_json`, 10)

The `Attempt` dataclass in `router_hook.py` is the source of truth.

| Field | Type | Meaning | Example |
|---|---|---|---|
| `tier` | str | Per-attempt tier | `S` |
| `model` | str | Full model string used for this attempt | `ollama_chat/qwen2.5-coder:1.5b` |
| `prompt_tokens` | int | Real prompt tokens from the response | `250` |
| `completion_tokens` | int | Real completion tokens | `80` |
| `duration_s` | float | Model-call wall time (seconds) | `2.34` |
| `critic_score` | int or null | 1--5, or null if not critiqued | `4` |
| `preview` | str | First 200 chars of the answer | `Here is your function...` |
| `was_warm` | bool | Was the model already loaded at call time? | `true` |
| `vram_mb` | int or null | Model VRAM footprint from `/api/ps` (gated; null when shadow mode off) | `8000` |
| `cost_usd` | float or null | Cloud spend via LiteLLM; null for local | `0.0023` |

Each attempt object inside `attempts_json` also carries three Issue #29
cost-tracking fields (added in v0.3):

- `was_warm` (bool) — whether the model was already loaded this process at call
  time. Process-set heuristic; does NOT account for Ollama evicting a model
  after `keep_alive` expires, so a "warm" model could have been re-loaded.
- `vram_mb` (int|null) — VRAM footprint in MB from Ollama `/api/ps`. Only
  populated when capability routing is enabled, and captured fire-and-forget so
  it never adds latency to the response: the first request for a given model
  logs null, subsequent requests get the cached value.
- `cost_usd` (float|null) — cloud spend via LiteLLM's cost calculator; null for
  local models (no pricing).

These fields live inside the JSON blob, so there is no schema change. Old
`attempts_json` rows predating v0.3 simply lack these keys (readers default
them).

Schema is migrated forward in `_init_db()` via `ALTER TABLE … ADD COLUMN` —
old databases continue to read cleanly.

If a SQLite write ever fails (locked DB, full disk, schema drift), the row is
preserved to `router_decisions_fallback.jsonl` and a failure counter is
incremented, so audit data is never silently lost (H-4).

---

## Configuration

All tunable settings live in the `route_llm:` block of `config.yaml`. Loaded
once at hook init via `load_config()`. Defaults are in the `RouterConfig`
dataclass; YAML overrides only what you want to change.

API keys are **never stored in YAML** — always referenced by env var name
(`api_key_env: ANTHROPIC_API_KEY`).

---

## Security posture

- **Loopback-only binding.** `start_proxy.ps1` launches the proxy with
  `--host 127.0.0.1`. LiteLLM's default is `--host 0.0.0.0`, which would
  expose the proxy (and the documented `sk-local-dev` key) to anyone on the
  same LAN/Wi-Fi. Loopback-only is enforced so "local-only" actually means
  local-only.
- **No secrets on disk or in logs.** API keys are read from named env vars at
  request time; nothing sensitive is written to `config.yaml` or the SQLite
  ledger (which stores only the first 200 chars of *model output*). Any string
  that looks like a key (`sk-…`) is redacted before logging (M-5), in case a
  provider error echoes the auth header.
- **No remote Ollama by default.** A non-localhost `OLLAMA_BASE_URL` is ignored
  unless `ALLOW_REMOTE_OLLAMA=1` is explicitly set — preventing a redirect of
  all prompts to an attacker host (M-6).
- **Critic injection defanged.** The critic neutralizes any `SCORE:` token a
  worker's answer tries to inject to force a passing grade (M-7).
- **Audit-hardened orchestrator.** A multi-agent code review (see the v0.1.0
  release notes and GitHub issue #1) drove the v0.1.0 critical fixes and the
  v0.1.2 backlog hardening so the audit trail can't silently diverge from
  reality: failed escalations (local and cloud) are recorded in the ledger,
  the handoff draft uses the actually-best response, streaming handoff
  failures are logged rather than swallowed, critic/classifier failures are
  logged by distinct cause (timeout vs connect vs parse), and a hard loop cap
  guards the escalation chain. Regression tests live in
  `tests/test_critical_fixes.py` and `tests/test_backlog_fixes.py`.

---

## CLI tools

| Script | What it does |
|---|---|
| `start_proxy.ps1` | Launches the LiteLLM proxy bound to `127.0.0.1` (loopback only), with UTF-8 console + a TriageLLM credits banner. Wrapped by `start_route_llm.bat`. |
| `local_mode.ps1` | Per-project local-routing launcher: manages a remembered-folders registry, ensures Ollama + proxy are up, and spawns a terminal with redirect env vars set. Wrapped by `local_mode.bat`. |
| `stats.py` | Reads the SQLite ledger and prints a dashboard. Wrapped by `dashboard.bat`. Per-decision drill-down (--explain), per-model P50/P95/P99 latency, and most-failing-model aggregation are layered on top of the existing tier/escalation/handoff summary (#17). |
| `health.py` | Probes proxy + Ollama + each model + critic. Wrapped by `health.bat`. |
| `doctor.py` | Setup pre-flight: checks config, installed models, cloud-audit, and routing mode. Runs offline (no proxy needed). Wrapped by `doctor.bat`. |
| `init.py` | Smart-defaults config generator (#14): detects installed Ollama models via `/api/tags`, assigns them to S/M/L/XL by parameter count, and renders a fully-commented `config.yaml`. Writes `config.generated.yaml` by default; `--write` replaces `config.yaml` (backing up to `config.yaml.bak`). |
| `benchmark.py` | Times the critic on CPU vs GPU; `--tasks` mode runs the per-machine task benchmark. |
| `task_benchmark.py` | Per-machine task benchmark logic: filter/aggregate/recommend/build_priors/render_report + Ollama I/O (#24). |
| `benchmark_prompts.py` | Curated benchmark prompts: 5 prompts x 8 testable capability categories (#24). |
| `advisory.py` | Capability graduation engine + offline advisory CLI (#18b). |
| `backtest.py` | Advisory endorsement drift validation -- compares signed-off endorsements against live ledger history (#28). |
| `uat.py` | Live User Acceptance Test -- four phases against the running proxy. Wrapped by `uat.bat`. |
| `start_route_llm.bat`, `stop_route_llm.bat` | One-click control for non-technical users. |

### Local Mode (client-side, not part of the proxy)

`local_mode.ps1` is a pure client-side convenience — it does **not** change the
proxy or the routing logic. It opens a terminal with the standard redirect
env vars (`OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL=local-auto`,
etc.) pre-set, so any env-respecting CLI tool launched from that window talks
to the proxy. Closing the window discards the env vars — that's the whole
"start/stop" model. Remembered folders live in a gitignored
`local_projects.json`.

---

## Capability Routing v0.2 (shadow mode)

A sidecar module (`capability_router.py`) classifies each request into one of
10 universal intent categories (`quick_question`, `explanation_or_summary`,
`structured_output`, `analytical_task`, `creative_generation`,
`modification_or_edit`, `multi_step_or_planning`, `high_risk`, `long_context`,
`default`) using rule-based keyword scoring with an optional LLM tie-breaker
for ambiguous long prompts. The result is recorded in 8 additive SQLite
columns (`cap_category`, `cap_recommended_tier`, `cap_reason_code`,
`cap_signals`, `cap_confidence`, `cap_classifier_used`, `cap_pack`,
`cap_agrees_with_tier`) without changing actual routing. The feature is
**disabled by default** (`TRIAGELLM_CAPABILITY_ROUTING_ENABLED` must be set
to `1`) so it is a pure opt-in shadow observer. Evidence is viewable via
`stats.py --capability`.

---

## Tests

Two layers:

- **Pytest, 442 unit tests, all mocked at the HTTP layer** (no Ollama required
  to run the suite). Runs in ~7 seconds. The test fixture in
  `tests/conftest.py` redirects `DB_PATH` to a tmp file for every test so
  production data is never touched, and sets `TRIAGELLM_SKIP_WARMUP` at
  collection time so the suite is hermetic with no shell setup.
- **`uat.py`, live end-to-end UAT** against the running proxy + real Ollama
  models. ~2-3 minutes per run. Verifies real classifier behavior, real
  round-trip latency, real ledger writes, real streaming.

Test files map to architectural concerns:

| File | Tests |
|---|---|
| `test_rules.py` | Rule-based classifier output |
| `test_llm_calls.py` | Mocked classifier + critic HTTP layer |
| `test_cpu_pin.py` | `num_gpu: 0` and `keep_alive: -1` get sent correctly |
| `test_softpass.py` | Pattern 3 tier-aware behavior |
| `test_orchestrate.py` | Multi-step escalation chain + cloud step |
| `test_streaming.py` | Streaming hook accumulates text + appends handoff |
| `test_handoff.py` | Handoff message rendering |
| `test_config.py` | YAML loading + defaults |
| `test_db.py` | Schema bootstrap + migration |
| `test_stats.py` | Dashboard aggregations |
| `test_critical_fixes.py` | The 5 critical-bug regressions from the multi-agent audit (C-1..C-5) |
| `test_backlog_fixes.py` | Issue #1 High/Medium hardening regressions (H-*/M-*): loop cap, log fallback, typed failures, redaction, SSRF guard, streaming convergence, config-derived models |
| `test_fast_fail.py` | DEF-004 circuit breaker: state machine, fast-fail without probe, cooldown/half-open recovery, explicit cp1252-safe reasons, pre-call hook raises + logs |
| `test_health_json_clean.py` | DEF-001 regression: `health.py --json` stdout stays valid JSON (hermetic) |
| `test_capability.py` | Capability Routing v0.2: 10-category classification, pack system, LLM tie-breaker, high-risk precedence, exception isolation |
| `test_capability_ledger.py` | Capability Routing v0.2: 8-column schema migration (idempotent), NULL-when-disabled behaviour |
| `test_task_benchmark.py` | Per-machine task benchmark (#24): pure unit tests (filter/aggregate/recommend/build_priors/render_report) + mocked Ollama I/O + orchestrator |
| `test_advisory.py` | Capability advisory mode (#18b): graduation engine, build/render report, sign-off gate, I/O helpers, CLI (pure + tmp-sqlite, no Ollama) |
| `test_backtest.py` | Advisory backtest (#28): _parse_since, BacktestVerdict, evaluate_endorsements drift verdicts, build/render report, CLI (pure + tmp-sqlite, no Ollama) |
| `test_doctor.py` | Setup doctor (#15/#19/#20a): CheckResult, check_config, check_models, analyze_cloud, derive_mode, impure fetchers, render_text/json, CLI (pure + sync-httpx-mock + tmp-sqlite, no Ollama) |
| `test_init.py` | Smart-defaults init (#14): _parse_params (parameter_size + byte fallback), assign_tiers (bands, coder preference, overflow spread, empty-band collapse, embedding skip), build_config (yaml-clean templated render), write_config (default + --write/.bak), CLI run/main (pure + monkeypatched fetch + tmp_path, no Ollama) |
| `test_failure_messages.py` | Runtime failure messages (#20b): model-not-found detection helpers, cold-load hint, async_post_call_failure_hook log-only behavior, _call_tier 404 catch re-raise |
| `test_stats_observability.py` | Stats observability (#17): load_one_decision, per_model_latency, most_failing_models, render_explain, CLI --explain + --pass-threshold integration |

---

## Why a Python module, not a service

The whole project is one `.py` file plus a config. It hot-reloads when you
edit it (LiteLLM re-imports on restart), it can be debugged with `pdb`, it
has zero deployment surface beyond `pip install -r requirements.txt`. There
is no Docker, no separate process manager, no IPC. Resist any pressure to
add those before you genuinely need them.

---

## A note on the author

This project is maintained by **Nishith Trivedi** — an SAP Supply Chain
analyst, not a professional software/ML engineer. TriageLLM was built
side-by-side with AI coding assistants to solve a real personal problem
(see [JOURNEY.md](JOURNEY.md)). Architectural decisions documented here
came from experimentation, not from formal training. If you're an
experienced engineer reading this and you see something that should be
done differently, please open an issue or reach out on
[LinkedIn](https://www.linkedin.com/in/nishith-t-5220a5b4). Feedback,
critique, and collaboration are all welcome.
