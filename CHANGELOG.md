# Changelog

All notable changes to TriageLLM are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Note on naming:** stage descriptors (shadow / advisory / active) live in the
> notes prose, never in the version tag. `v0.2.0` ships capability routing in
> *shadow* mode; advisory mode is a v0.3 feature, not a `v0.2.0-shadow` suffix.

## [Unreleased]

Shipped since v0.3.0, toward the "Public Release Prep" milestone:

- **Smart-defaults `init`** — `init.py` detects installed Ollama models via
  `/api/tags` and generates a tier-mapped `config.yaml` (writes
  `config.generated.yaml` by default; `--write` applies it).
- **Live advisory surfacing** — `TRIAGELLM_CAPABILITY_MODE=advisory` surfaces the
  capability recommendation per request: a `[advisory]` proxy-log line plus
  best-effort `x-triagellm-cap-*` response headers, with no routing change.
- **Objective `high_risk` success signal** — a safe refusal on a high-risk prompt is
  now scored as a (hard) success rather than a failure, for the offline success metric.

**Known limitations (deferred):**

- **PyPI packaging** — no `pip install triagellm` yet; clone-and-run only.
- **Critic ensemble / second-opinion** — the single 0.5B critic remains a tripwire;
  an ensemble is deferred until there is enough real critic-vs-validator data to
  measure where it actually helps.
- **Active capability routing** — capability recommendations are advisory only;
  they never override the tier the router picks.

## [0.3.0] - 2026-05-30

"Routing Intelligence." The per-machine evidence loop that turns TriageLLM from
"just another router" into one that *knows* which of your local models is best
for which task — plus the operational tooling, observability, and security
hygiene around it. Everything new is **offline / opt-in / advisory**: no routing
decision changes without explicit operator sign-off, and active routing remains
v0.4+.

### Added
- **Per-machine task benchmark — `benchmark.py --tasks`** (the v0.3 signature
  feature). Runs each installed Ollama model against curated per-category prompts,
  scores each output, measures P50/P95 latency, and writes a per-(model, category)
  recommendation map to `benchmark_results.json` (priors). `--quick` / full,
  `--models`, `--json`. Capability-first -> latency -> size tie-break. (#24)
- **Output scoring primitive — `success.py`** — pure `score_output(category, text,
  critic_score)` judging one model output per category: objective hard checks
  (`is_valid_json` for structured output, `is_valid_code_edit` for edits) vs a
  soft critic-threshold + deterministic sanity floor, each tagged `hard`/`soft`
  confidence. The scoring core the benchmark + advisory share. (#18a)
- **Capability advisory mode — `advisory.py`** — combines the benchmark priors
  with live-ledger critic-pass-rate corroboration into a per-category graduation
  status (`graduated` / `insufficient-benchmark` / `needs-live-evidence` /
  `live-disagreement`); hard categories graduate on the benchmark alone, soft
  need a stronger benchmark or live corroboration. `--sign-off` writes the
  evidence-snapshotted endorsed map to `advisory_endorsements.json` — the
  human-approved gate a future active-routing phase will consume. (#18a/#18b)
- **Advisory backtest — `backtest.py`** — validates the signed-off endorsements
  against live ledger history over a `--since` window, flagging endorsements that
  have **drifted** below their frozen sign-off snapshot (`upheld` / `drifted` /
  `contradicted` / `insufficient-data`). Read-only. (#28)
- **Setup pre-flight — `doctor.py`** (+ `doctor.bat`) — answers "is my setup
  correct?" without the proxy running: Ollama reachability, config-vs-installed
  models (with an `ollama pull` fix hint), config sanity, an evidence-level cloud
  audit (the local-first proof, from the `cloud_attempted` ledger column), and the
  orthogonal Routing x Cloud mode. PASS/WARN/FAIL, graceful independent
  degradation. (#15, #19, #20a)
- **Per-attempt cost tracking** — every `attempts_json` record now carries
  `was_warm` (process-set heuristic), `vram_mb` (gated `/api/ps` fetch,
  fire-and-forget cache), and `cost_usd` (cloud spend via LiteLLM; null for
  local). (#29)
- **Runtime failure hints** — a new `async_post_call_failure_hook` logs an
  actionable `ollama pull X` hint on a first-attempt model-not-found 404 (using
  the real model tag from the Ollama error, not the routing alias);
  `_call_tier` does the same on escalation (re-raises, ledger path intact); a
  cold-load hint logs when a model takes a long time to load. Observability-only —
  never alters routing. (#20b)
- **`stats.py` observability** — `--explain [ts]` drills into one decision (full
  per-attempt trail incl. the #29 cost fields), per-model **P50/P95/P99** latency
  and a **most-failing-models** report in the default dashboard, `--pass-threshold`
  override. (#17)
- **CI + security hygiene** — GitHub Actions matrix (ubuntu + windows, Python
  3.12, mocked suite, no Ollama), `.env.example`, `SECURITY.md`, Dependabot
  (pip + actions). (#25)
- **Docs** — "What TriageLLM is NOT", a full ledger schema reference, a
  troubleshooting playbook (#23); a client compatibility matrix (#16); dynamic
  CI-status + release badges and a hand-authored SVG architecture diagram
  (#26).

### Changed
- **`is_valid_code_edit` accepts prose-wrapped fenced code** — a live benchmark
  smoke found that real instruction-tuned models wrap edits in a prose preamble +
  a ```` ```python ```` block; the validator now extracts a fenced block from
  anywhere (not just a leading fence) before `ast.parse`, so correct edits stop
  scoring as `not-an-edit`. (#24 smoke)
- **`advisory.load_live_aggregates` generalized** with an additive `since_ts`
  time window (backward-compatible; default behaviour unchanged), reused by the
  backtest. (#28)
- Test count grew **188 -> 389** (mocked unit suite, ~6 s, no Ollama required).

### Known limitations
- **Advisory only.** The benchmark, advisory graduation, and backtest are
  operator-run CLIs that *recommend*; nothing changes the tier the existing
  router picks. The endorsed map is a human-approved input for a future
  active-routing phase (v0.4+).
- **#18 live per-request surfacing is deferred** — surfacing advisory
  recommendations live (proxy logs / response headers) needs separate design (the
  post-call hook returns the body, not HTTP headers). The umbrella issue stays
  open for it; the offline advisory core shipped.
- The **wrong-API-key 401 hint is documented, not built** — LiteLLM's auth layer
  rejects a bad key before any routing hook runs, so the hook can't customize that
  401. (#20b)
- The critic remains a 0.5B **tripwire** (catches refusals/gibberish), not a
  fine-grained quality judge — which is why soft-category evidence is weighted
  below objective hard checks throughout.
- Capability routing is still **default-OFF shadow mode**; the advisory loop reads
  its shadow data when enabled but does not require it.

### Test count: 389

## [0.2.0] - 2026-05-28

Capability Routing v0.2 (shadow mode), a full high-risk-pattern precision audit,
and SQLite schema hardening. Capability routing is **disabled by default** and,
when enabled, **records only** — it does not change any routing decision.

### Added
- **Capability Routing v0.2 (shadow mode)** — `capability_router.py` sidecar that
  classifies each request into one of 10 universal intent categories and records
  the tier a capability-aware router *would* pick, in 8 additive SQLite columns.
  Disabled by default (`TRIAGELLM_CAPABILITY_ROUTING_ENABLED`); zero behaviour
  change when off. (PR #5)
- **`security-control-bypass` high-risk pattern** — fires on a risky action verb
  near a security-control compound noun ("disable security checks"), without a
  bare "security" keyword rule. (PR #10)
- **Migration observability** — `_init_db` post-migration assert raises at startup
  if any expected column is missing; `_log_decision` emits a one-shot
  "DASHBOARD WILL SHOW NO DATA" warning the first time a write spills to the H-4
  JSONL fallback. (PR #8, closes #7)
- **`_DECISIONS_COLUMNS`** — single source-of-truth for the `decisions` table
  schema; CREATE TABLE, migrations, post-migration assert, and INSERT column list
  all derive from it. (PR #13, closes #9)
- **`stats.py --capability`** — shadow-data view (per-category counts,
  agree-with-tier rate, XL-avoidance opportunity).
- **`tests/run_capability_shadow.py`** — live capability shadow harness.
- New env vars: `TRIAGELLM_CAPABILITY_ROUTING_ENABLED`, `TRIAGELLM_CAPABILITY_PACKS`.

### Changed
- **High-risk patterns tightened to require verb-or-context** (the "topical
  mention" precision audit). Four bare-keyword patterns became context-aware and
  were renamed: `credential-mention` → `credential-action`, `auth-mention` →
  `auth-system`, `financial-logic` → `financial-system`, `injection-mention` →
  `injection-exploit`. Educational prompts ("what is CSRF protection", "explain
  MFA") no longer inflate the high-risk count; real operations ("rotate the API
  keys in production", "exploit the SQL injection") still fire. (PRs #10/#12,
  closes #6, #11)
- Test count grew 86 → **188** (mocked unit suite, ~5 s, no Ollama required).

### Known limitations
- Capability routing is **shadow-only** in v0.2 — it observes and records but
  never changes the tier the existing router chose. Advisory mode (recommendations
  surfaced to the client) is v0.3; active routing is v0.4+.
- The critic is a 0.5B model used as a **tripwire** (catches refusals/gibberish),
  not a fine-grained quality judge.
- High-risk reason codes were renamed; historical ledger rows keep the old codes
  (the `decisions` ledger is append-only). Dashboards key off the `high-risk:`
  prefix so both old and new codes aggregate correctly.

### Test count: 188

## [0.1.2] - 2026-05-23

### Fixed
- **Issue #1 backlog — 15 audit follow-ups** (High/Medium hardening). Highlights:
  H-1 escalation loop cap, H-4 log fallback, H-7 converged streaming vs
  non-streaming critique semantics (`_critique_outcome`), M-2 escalation models
  derived from `config.yaml`, M-5 prompt redaction option, M-6 SSRF guard on
  non-localhost `OLLAMA_BASE_URL` (opt-in via `ALLOW_REMOTE_OLLAMA`).

### Changed
- Test count raised to 86; `conftest.py` sets `TRIAGELLM_SKIP_WARMUP` at
  collection time for a hermetic suite.

## [0.1.1] - 2026-05-21

### Added
- **Per-project Local Mode** (`local_mode.bat` / `local_mode.ps1`) — spawn a shell
  with redirect env vars so every AI CLI in that window routes to local Ollama via
  the proxy; auto-starts Ollama + the proxy; add/remove project folders via a
  browse dialog. Client-side only; closing the window restores normal cloud AI.
- TriageLLM credits banner at proxy startup + in the launcher.

### Fixed
- Local Mode proxy path quoting (the `D:\Route LLM` space bug).

## [0.1.0] - 2026-05-19

Initial release. LiteLLM proxy + a tier-routing `CustomLogger` callback in front
of Ollama.

### Added
- Rule-based classifier + optional 0.5B LLM classifier (LLM may only *upgrade* the
  rule tier).
- CPU-pinned 0.5B critic (`num_gpu: 0`, `keep_alive: -1`) — the cascade fix for
  low-VRAM machines.
- Multi-step escalation (`initial_tier → … → XL → CLOUD`) with a per-request
  attempt ledger.
- Tier-aware soft-pass (S/M ship on critic timeout; L/XL escalate).
- Optional cloud-API escalation (off by default; API keys via env-var name only).
- Structured handoff message renderer; streaming critic with a one-chunk handoff
  append.
- SQLite decision ledger + `stats.py` dashboard, `health.py`, `benchmark.py`.
- Proxy bound to `127.0.0.1` only (loopback) for safety.
- Apache 2.0 license.
