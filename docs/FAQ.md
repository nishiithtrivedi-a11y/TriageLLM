# TriageLLM FAQ

Questions I've actually been asked, plus questions I think prospective
users will reasonably wonder. Grouped by topic.

---

## Usage

### Q. What is TriageLLM in one sentence?
A LiteLLM-based local proxy that classifies each coding prompt by complexity
and routes it to the smallest Ollama model that can answer it, escalating to
bigger local models (and optionally a cloud API) only when needed.

### Q. Who is this for?
Developers who already use Ollama locally, have a consumer GPU with 8-24 GB
VRAM, and want to stop spending cloud API tokens on routine code edits.

### Q. What does my client need to do to use TriageLLM?
Point any OpenAI-compatible (or Anthropic-compatible) client at
`http://localhost:4000`, use API key `sk-local-dev` (or whatever you set in
`config.yaml`), and ask for the model `local-auto`. The proxy classifies and
routes the rest.

### Q. Will it work with cloud-AI agents like Claude Code, Codex CLI, ChatGPT desktop?
**OpenAI-API agents** (Codex CLI variants, aider, anything respecting
`OPENAI_API_BASE`) — yes, out of the box. Point them at
`http://localhost:4000/v1` with key `sk-local-dev` and model `local-auto`.

**Anthropic-API agents** (Anthropic's Claude Code CLI, Anthropic SDK apps,
etc.) — partially. They respect `ANTHROPIC_BASE_URL` and
`ANTHROPIC_AUTH_TOKEN` env vars to redirect to a non-Anthropic endpoint,
and you'd need to add a model alias in `config.yaml` matching the model
name the client asks for (e.g. `claude-sonnet-4-6`). See the README's
"Anthropic-API agents" section.

**The honest caveat** for *any* cloud agent: cloud agents' full feature
sets (skills, MCP connectors, agentic tool loops, vision, structured
outputs) depend on frontier-model capabilities that local 1.5B-35B models
can't replicate. Plain-text chat works; advanced features will be
degraded. The cost-saving pattern is: keep using cloud agents on their
native clouds for heavy work, use TriageLLM via Continue.dev or aider for
routine work.

### Q. Will it work with ChatGPT / Claude desktop apps directly?
No — the consumer chat apps (ChatGPT desktop, Claude desktop) talk only to
their owners' clouds and have no setting to redirect them. The
*subscriptions* that come with those apps (ChatGPT Plus, Claude Max) are
also tied to the apps, not the API. To use a cloud LLM with TriageLLM you
need an API account separately (which TriageLLM only calls as a last
resort, if you turn `cloud_escalation.enabled` on).

### Q. Will it work with Continue.dev?
Yes, cleanly. Add a model entry in `~/.continue/config.json` with
`apiBase: "http://localhost:4000/v1"`, `apiKey: "sk-local-dev"`, and
`model: "local-auto"`. See the README for the exact config.

### Q. Will it work with aider?
Yes:
```powershell
$env:OPENAI_API_BASE = "http://localhost:4000/v1"
$env:OPENAI_API_KEY  = "sk-local-dev"
aider --model openai/local-auto
```

### Q. Something isn't working -- what do I run first?
`doctor.py` -- it checks your setup (models installed, config coherence,
cloud-audit, routing mode) without needing the proxy up, and tells you the
fix for anything it finds:

```powershell
.\.venv\Scripts\python.exe doctor.py
```

Or double-click `doctor.bat`. Once doctor passes, use `health.bat` to verify
the running stack.

### Q. How do I start / stop it?
Double-click `start_route_llm.bat` to start, `stop_route_llm.bat` to stop.
Run `dashboard.bat` anytime to see usage stats. `health.bat` to check
everything is wired up correctly. Create desktop shortcuts of any of these.

### Q. Can I make just ONE project use local models, and keep cloud AI everywhere else?
Yes — that's **Local Mode**. Double-click `local_mode.bat`, press `[A]` to
remember a project folder, then pick its number. A new terminal opens where
every AI CLI tool (Claude Code, Codex CLI, aider, etc.) routes to local
Ollama via the proxy. Close that window and you're back to normal cloud AI —
nothing permanent changes. The launcher auto-starts Ollama + the proxy if
they aren't already running. It does **not** work with desktop apps (Claude
Desktop / ChatGPT Desktop) — those can't be redirected. See the README's
"Local Mode" section for the full walkthrough.

### Q. Is the proxy exposed to my network?
No. As of v0.1.0 the proxy binds to `127.0.0.1` (loopback) only, so it's
reachable only from your own machine — not from other devices on your
Wi-Fi/LAN. (LiteLLM's default would have bound to all interfaces; TriageLLM
overrides that for safety.)

### Q. What does the dashboard show me?
Tier distribution (what % of prompts went S/M/L/XL), critic scores per tier,
escalation rate, handoff count, total tokens consumed, per-model latency
P50/P95/P99, most-failing models, and recent decisions table. Run
`dashboard.bat --since "2 days"` to scope to recent activity.

### Q. How do I see what happened on a specific request?
Run `stats.py --explain` (drills into the latest decision) or
`stats.py --explain <ts>` where ts is copied from `stats.py --json` output
(or the recent-activity table). You get the routing path, classifier signals,
capability classification, critic verdict, and the full per-attempt trail
(model, tokens, duration, critic, was_warm, vram, cost, preview).

### Q. The dashboard shows average latency -- how do I see the tail?
The default `stats.py` report now includes a **Per-model latency** section
with P50 / P95 / P99 / max per (tier, model). P95 requires n >= 4 and P99
requires n >= 5 (shown as `-` below those thresholds so you do not get
fictitious numbers from tiny samples).

### Q. How do I find the weakest local model?
The default `stats.py` report includes a **Most-failing models** section that
groups attempts by model, counts only critiqued attempts, and ranks by
failure rate (`critic_score < pass_threshold`). The minimum-sample gate is
5 critiqued attempts so a single bad call cannot dominate. Override the
threshold with `--pass-threshold N`.

---

## Hardware & optimization

### Q. The project is optimized for what hardware?
My laptop: **Ryzen 7 260 (with NPU), Nvidia RTX 5070 12 GB VRAM, 32 GB DDR5
RAM, models on a 1 TB SSD**. The default `config.yaml` is tuned for that
GPU's 12 GB VRAM budget — specifically, the CPU-pinned critic exists because
of the eviction problem on this exact VRAM size.

### Q. I have a smaller GPU (8 GB VRAM). What do I change?
Drop tier-L and tier-XL models from `model_list` in `config.yaml`. Remove
their entries from `router_settings.fallbacks` too. Stick to tier S and M
(1.5B and ~7B models). Keep `critic_cpu_only: true` — CPU critic is even
more critical on small GPUs.

### Q. I have a 24 GB+ GPU (RTX 3090, 4090, 5090). What changes?
You can flip `critic_cpu_only: false` if you want — the critic can fit
alongside even tier-XL on a 24 GB card. You'll save a few hundred
milliseconds of CPU time per critique. The eviction cascade won't happen
because there's enough room for both.

### Q. I have an Apple Silicon Mac (M-series).
Should work — Ollama is fully supported on Apple Silicon, and the Mac's
unified memory architecture means VRAM and RAM are the same pool, so the
eviction cascade is much less of a problem. The CPU-pinned critic is still
fine; on Apple Silicon, "CPU" inference goes through the same memory.

### Q. Can I run this without a GPU at all?
Yes, in principle, but the worker models (especially L and XL) will be
extremely slow on CPU. Realistically use S and M only, or look at smaller
quantizations.

### Q. What models do I need to pull?
Minimum: `qwen2.5:0.5b` (the classifier and critic). Then any of the worker
tiers you actually want to use (`qwen2.5-coder:1.5b` for S,
`deepseek-coder-v2:16b` for M, etc.). See README for the full list.

### Q. Can I use different models than the defaults?
Yes. Edit `config.yaml` — `model_list` for the tier aliases, and `TIER_TO_MODEL`
in `router_hook.py` for the escalation paths. Make sure both files agree.

### Q. Why is the first request after startup slow?
The first time a model is used, Ollama loads it from disk into RAM/VRAM.
This can take 10-60 seconds depending on model size. After that, models stay
"warm" thanks to `keep_alive`. The critic specifically is pre-warmed at
proxy startup via the `warmup_on_startup` setting.

---

## Performance & numbers

### Q. What's the critic latency?
On my Ryzen 7 260 + RTX 5070, with a 0.5B critic model:
- **CPU-pinned: median 0.58 s, p95 0.63 s, max 0.63 s** (n=5)
- **GPU-pinned: median 0.58 s, p95 0.58 s, max 0.59 s** (n=5)

Effectively identical. CPU placement gives us eviction immunity for free.
Run `benchmark.py` on your machine to verify.

### Q. How fast are the worker tiers?
Depends heavily on whether the model is warm or cold. Approximate ranges on
my hardware:
- Tier S (1.5B): ~2-4 seconds for typical short responses
- Tier M (16B): 5-10 seconds warm, 30-60 seconds cold-load
- Tier L (30B): 15-30 seconds warm (spills to system RAM)
- Tier XL (35B): 30-90 seconds warm (heavy spillage)

### Q. How much money will I actually save?
Depends on your workload. If you're currently spending $20-50/month on
Claude/GPT API for things like autocomplete, regex, "rename this," and
small refactors — TriageLLM should reduce that toward $0 because those
prompts route to free local tiers. Hard problems still cost cloud rates if
you enable cloud escalation, but they're a small fraction of total calls.

### Q. What's the test suite coverage?
**442 mocked unit tests** — all HTTP calls replaced with mocks, so the suite
runs without Ollama or the proxy. Covers the classifier, critic, escalation
chain, soft-pass logic, streaming, SQLite schema, dashboard aggregations,
config loading, CPU-pin behavior, the 5 critical-bug regressions from the
multi-agent audit (`tests/test_critical_fixes.py`), the High/Medium
hardening fixes (`tests/test_backlog_fixes.py`), the fast-fail circuit
breaker (`tests/test_fast_fail.py`), Capability Routing v0.2
(`tests/test_capability.py`, `tests/test_capability_ledger.py`), the
per-machine task benchmark (`tests/test_task_benchmark.py`), the
capability advisory CLI (`tests/test_advisory.py`), the advisory
backtest (`tests/test_backtest.py`), the setup doctor
(`tests/test_doctor.py`), and stats observability
(`tests/test_stats_observability.py`). The suite is hermetic —
`conftest.py` sets `TRIAGELLM_SKIP_WARMUP` automatically, so it runs in
~5 seconds with no shell setup:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

### Q. Is there a way to test against the *real* proxy end-to-end?
Yes — `uat.bat` runs a User Acceptance Test against the live proxy with
real Ollama. Four phases: liveness probe, rule classifier accuracy on
curated prompts, real round-trip through `/v1/chat/completions` with
ledger inspection, streaming + post-stream critic. Takes about 2-3 minutes:

```powershell
# In one window:
.\start_route_llm.bat       # leave running

# In another window:
.\uat.bat
```

Run this before releases or after touching `config.yaml` / `router_hook.py`.
The mocked pytest suite catches wiring regressions; the UAT catches real
behavioral regressions.

### Q. What is "capability routing"?
A v0.2 opt-in feature (default OFF) that classifies each request into one of 10
universal categories (e.g. `quick_question`, `structured_output`, `high_risk`)
and records the tier capability routing **would** have chosen, without changing
actual routing. It collects evidence to graduate to advisory/active modes in
later releases.

### Q. How do I find the best local model for each task type?
Run `benchmark.py --tasks` (or `--tasks --quick` for a faster two-prompt-per-category
pass). It scores every installed Ollama model against curated prompts for each of
the 8 testable capability categories (quick_question, explanation_or_summary,
structured_output, analytical_task, creative_generation, modification_or_edit,
multi_step_or_planning, long_context) and writes a per-(model, category) recommendation
map to `benchmark_results.json`. Use this to tune which models you keep, or to
supply priors for advisory mode.

```powershell
# Quick pass (2 prompts/category, ~minutes)
.\.venv\Scripts\python.exe benchmark.py --tasks --quick

# Full pass (5 prompts/category)
.\.venv\Scripts\python.exe benchmark.py --tasks
```

### Q. How do I know which model to trust for each task?
Run `advisory.py` after `benchmark.py --tasks`. It combines the benchmark priors with
live-ledger critic-pass rates to show which per-(category, model) recommendations have
graduated (earned enough evidence to trust), and lets you sign off graduated
recommendations into `advisory_endorsements.json`:

```powershell
# See graduation status for every category
.\.venv\Scripts\python.exe advisory.py

# Endorse a specific graduated recommendation
.\.venv\Scripts\python.exe advisory.py --sign-off structured_output

# Endorse all currently graduated recommendations at once
.\.venv\Scripts\python.exe advisory.py --sign-off-all
```

Hard categories (structured_output, modification_or_edit) graduate on the benchmark
alone. Soft categories also need a strong benchmark score (>=0.8) or live-ledger
corroboration. Advisory only -- it changes no routing. The endorsed map is the
human-approved gate a future active-routing phase will consume.

### Q. How do I know if my endorsed models are still performing?
Run `backtest.py` -- it compares each signed-off endorsement in
`advisory_endorsements.json` against live ledger history and flags any
that have drifted below what the benchmark/critic promised at sign-off:

```powershell
# Check against all-time ledger history
.\.venv\Scripts\python.exe backtest.py

# Check only decisions from the last 7 days
.\.venv\Scripts\python.exe backtest.py --since 7d

# JSON output for scripting
.\.venv\Scripts\python.exe backtest.py --json
```

Per endorsed category it reports `upheld` / `drifted` / `contradicted` /
`insufficient-data`. Read-only -- it never changes your endorsements. If a
category shows `drifted` or `contradicted`, re-run `benchmark.py --tasks` +
`advisory.py` to re-evaluate and re-endorse.

---

## Pros & limitations

### Q. What's TriageLLM genuinely good at?
- Routing routine prompts to fast small models for free
- Catching escalation needs automatically via the critic
- Keeping a transparent ledger so you can analyze usage
- Working reliably on consumer GPUs (the CPU-pinned critic was the key insight)
- Being a single Python module — easy to inspect, debug, modify

### Q. What's it genuinely bad at?
- Producing frontier-cloud-quality answers (local models just aren't there yet)
- Replacing cloud-agent feature sets (skills/MCP/agentic loops/vision in
  Claude Code, Codex CLI, ChatGPT, etc.)
- Cloud-to-cloud routing (use LiteLLM directly for that)
- Running on machines with <8 GB VRAM (you'd need to drop L/XL tiers)
- Anything involving vision or audio (text-only)

### Q. Is the critic always right?
No — it's a 0.5B model judging answers from much larger models. It's a
heuristic to detect *obviously* bad answers (refusals, off-topic, gibberish).
Tier S is excluded from critique entirely. For tier S/M, the orchestrator is
configured to "soft-pass" when the critic times out, which prevents wasted
escalations. The threshold (4/5) is set conservatively.

### Q. What if the local stack truly can't answer my prompt?
You get a structured handoff message in the response body — a clear ledger
of every tier attempted, the critic scores, the best local draft, and a
recommendation to retry against your cloud agent. The current local answer
is preserved as a draft below the divider. Decision-makers (you, or any
cloud-AI agent like Claude Code, Codex CLI, ChatGPT, etc.) can choose
whether to escalate manually.

### Q. Does it phone home / send telemetry anywhere?
No. Everything runs on `localhost`. The only outbound network call is to
Ollama (also localhost). Cloud escalation only fires if you explicitly turn
it on AND set the API key env var. The proxy binds to `127.0.0.1` only, and a
non-localhost `OLLAMA_BASE_URL` is refused unless you set `ALLOW_REMOTE_OLLAMA=1`.

### Q. What environment variables does TriageLLM use?
See [`.env.example`](../.env.example) for a copy-paste-ready template with every
variable, its default, and what it does. TriageLLM doesn't auto-load `.env` —
set these in your shell or launcher. The full list:

- `OLLAMA_MODELS` — where Ollama keeps its model files (set this if your models
  aren't in the default location; restart Ollama after changing it).
- `TRIAGELLM_SKIP_WARMUP=1` — skip the startup critic warmup. Set automatically
  by `health.bat`, `uat.bat`, and the test suite; you only need it if you import
  `router_hook` directly in your own script.
- `ALLOW_REMOTE_OLLAMA=1` — allow a non-localhost `OLLAMA_BASE_URL` (off by
  default for safety).
- **Fast-fail circuit breaker (DEF-004)** — when Ollama is unreachable, a
  request fails in ~2s instead of ~32s, the proxy stays alive, and it
  auto-recovers once Ollama returns. Tunable (defaults shown):
  `TRIAGELLM_OLLAMA_FAST_FAIL_ENABLED=true`,
  `TRIAGELLM_OLLAMA_CONNECT_TIMEOUT_SECONDS=2`,
  `TRIAGELLM_OLLAMA_CIRCUIT_COOLDOWN_SECONDS=10`,
  `TRIAGELLM_OLLAMA_PROBE_TIMEOUT_SECONDS=2`.
  As of #3, the fast-fail preflight covers explicit tier pins (`local-s/m/l/xl`)
  too -- not just `local-auto` -- so a pinned request also fails in <5s when Ollama
  is down. Genuinely-external model names still pass through untouched.
- The cloud API-key var named in `config.yaml` (`api_key_env`, e.g.
  `ANTHROPIC_API_KEY`) — only read if cloud escalation is enabled.
- `TRIAGELLM_CAPABILITY_ROUTING_ENABLED=1` — enable Capability Routing v0.2
  shadow mode (default off). Records 8 extra SQLite columns per request
  showing what the capability router would have chosen; does not change
  actual routing.
- `TRIAGELLM_CAPABILITY_PACKS` — comma-separated list of capability packs to
  activate (e.g. `coder,writing`). Overrides the `packs` list in
  `config.yaml`'s `route_llm.capability_routing` block. Only relevant when
  `TRIAGELLM_CAPABILITY_ROUTING_ENABLED=1`.
- `TRIAGELLM_CAPABILITY_MODE` -- `shadow` (default, record only) or `advisory`
  (record + surface per-request; implies the classifier is enabled).

### Q. Does it work with aider / Continue.dev?
- **aider**: yes — point it at the proxy (`OPENAI_API_BASE=http://127.0.0.1:4000/v1`,
  `OPENAI_API_KEY=sk-local-dev`, `--model openai/local-auto`). Validated
  end-to-end: aider installs, connects, and routes through TriageLLM (proven by
  a decision-ledger row). Note: a small *local* model may not always complete
  aider's strict edit-diff format — that's a model-capability limit, not a
  routing issue; use a stronger local model (or cloud) for reliable edits.
- **Continue.dev**: works at the protocol level — a ready config lives at
  `tests/fixtures/continue/config.yaml`. Continue is a VS Code/JetBrains
  extension with no headless CLI, so the actual extension run is a manual step
  (documented in `tests/evidence/latest_continue_summary.md`); the
  OpenAI-compatible endpoint it uses is verified to route through TriageLLM.

---

## Licensing & contributions

### Q. What's the license?
Apache 2.0. See [LICENSE](../LICENSE).

### Q. Can I use this commercially?
Yes. Apache 2.0 allows commercial use, modification, distribution, and
private use. You must keep the copyright notice and the LICENSE file, and
you must indicate any changes you made (a `git log` or a note in
modified files is enough).

### Q. Can I fork it and rebrand it?
Yes, under Apache 2.0 terms. Keep the LICENSE and NOTICE files, and
acknowledge the upstream project somewhere reasonable (README, About
dialog, source comments, whatever fits). The NOTICE file already names me
and the third-party dependencies — please carry that forward.

### Q. I want to contribute. What should I do?
Open an issue first describing the problem or feature. If we agree on
direction, send a pull request with tests. Be honest about edge cases. Don't
ship breaking config changes without bumping a version. That's it.

### Q. What credit do I owe to upstream projects?
See [NOTICE](../NOTICE). TriageLLM depends on LiteLLM (MIT), Ollama (MIT),
httpx (BSD-3), PyYAML (MIT), pytest (MIT). Those projects do not need to be
re-distributed by you, but their licenses require you keep their copyright
notices intact if you do redistribute them.

### Q. Who built this and why?
**Nishith Trivedi.** Full transparency: I'm an SAP Supply Chain analyst by
profession, **not a software developer or ML engineer**. I built this as a
side-project, with substantial help from AI coding assistants
("vibe-coding"), to solve a real problem I had — paying too much for
cloud-LLM tokens on routine coding work when my laptop could handle most
of it for free.

The patterns documented here are real and reproducible, but a professional
infra/ML engineer will probably spot ways to improve them. If that's you,
**I genuinely want the feedback** — open an issue or PR, or reach me on
[LinkedIn](https://www.linkedin.com/in/nishith-t-5220a5b4). If the project
saved you time or money, a shout-out there would also mean a lot, and I'm
open to collaborating on related work.

The [JOURNEY.md](JOURNEY.md) document tells the full story including the
discoveries that came out of building it.

---

## Troubleshooting

### Q. The proxy window crashes on startup.
Most common causes: (1) Python error in a recent edit — run `pytest` to see
what's broken. (2) Port 4000 already in use — check `netstat -ano | findstr 4000`.
(3) Ollama not running — check the tray icon.

### Q. `dashboard.bat` shows giant `duration_s` numbers.
That was an old test-fixture bug that polluted the SQLite database with
fake rows. Fixed in the current code; if you're on an older version, pull
the latest and the cleanup script in the commit history handles it.

### Q. `ollama list` shows zero models even though I have them on disk.
Your `OLLAMA_MODELS` env var probably isn't set, or Ollama was launched
before you set it. Run:
```powershell
[Environment]::SetEnvironmentVariable('OLLAMA_MODELS', 'D:\Your\Path', 'User')
```
Then **restart Ollama** (Quit from the tray, relaunch). The daemon only
reads `OLLAMA_MODELS` at startup.

### Q. The critic is timing out on every call.
Check `dashboard.bat` — what's the average chain length? If it's >1.5,
escalations are firing too aggressively. Make sure `critic_cpu_only: true`
in `config.yaml`. Run `benchmark.py` to verify critic latency. If CPU
critic is genuinely slow on your machine (>5 seconds), bump
`critic_timeout_s` to 60.

### Q. PowerShell says "running scripts is disabled."
Run the proxy this way:
```powershell
powershell.exe -ExecutionPolicy Bypass -NoProfile -File .\start_proxy.ps1
```
Or use `start_route_llm.bat` which already handles this.

### Q. I'm getting `model not found` errors for tiers I want to use.
You haven't pulled that model yet. Run `ollama pull <model>` for the tier
listed in `config.yaml` under `model_list`.

### Q. I got a 404 / "model not found" error -- what do I do?
TriageLLM logs an actionable hint in the proxy console the moment this happens:
`Model X is not installed on this Ollama daemon. Run: ollama pull X`. Run that
(or rebuild it if it is a custom Modelfile build). Run `doctor.py` to check all
configured models up front.

### Q. The first request to a model is very slow -- is something wrong?
No -- that is a cold load (Ollama reading the model from disk into VRAM/RAM). The
proxy logs `Model X took Ns (cold load?); it stays warm via keep_alive after the
first call`. Subsequent calls are fast.

### Q. Why doesn't a wrong proxy API key give a TriageLLM-specific hint?
The proxy's key check runs in LiteLLM's auth layer, BEFORE TriageLLM's routing
hook -- so the hook can't customize that 401. Use `sk-local-dev` (or whatever
`master_key` you set in `config.yaml`).

### Q. My request just hangs / times out at ~30s.
Likely Ollama isn't running, or `OLLAMA_BASE_URL` points somewhere unreachable.
Run `doctor.py` -- it diagnoses reachability + everything else in one pass. For
a one-liner sanity check: `curl http://localhost:11434/api/version`.

### Q. `stats.py` shows `no critiqued attempts` but I have traffic.
All your traffic went tier-S (which is critique-excluded by design -- see
ARCHITECTURE "Critic"). Confirm with `stats.py --explain` on a recent decision
(`tier=S`); this is expected behavior, not a bug. To exercise critique, send a
prompt that routes to M/L/XL or use the `local-m` alias directly.

### Q. `benchmark.py --tasks` says `no benchmarkable models found`.
Either Ollama isn't running, or every installed model matched the role-based
skip-list (the 0.5B classifier/critic + any `embed`-named models). Use
`ollama list` to inspect what's installed; pass `--models a,b,c` to force-
include something specific.

### Q. `advisory.py` shows everything as `needs-live-evidence`.
Soft categories need either a stronger benchmark score OR live ledger
corroboration. Re-run `benchmark.py --tasks` at full P=5 (P=2 doesn't reach
`n_prompts >= 3`), OR enable shadow-mode capability routing
(`TRIAGELLM_CAPABILITY_ROUTING_ENABLED=1`) so live traffic gets categorized and
builds corroboration over time.

### Q. What does advisory mode do?
Advisory mode (`TRIAGELLM_CAPABILITY_MODE=advisory`) makes the proxy *show* the
capability classifier's recommendation for each request -- a `[advisory]` log line
and `x-triagellm-cap-*` response headers (non-streaming) -- **without changing
routing**. It is the observe-only step between shadow recording and (future) active
capability routing. `advisory` implies the classifier is enabled. Streaming
responses get the log line but no headers (chunks already flushed). Default is
`shadow` (record only).

### Q. `backtest.py` shows `insufficient-data` for everything.
Either the `--since` window has no overlap with endorsed-model usage, OR live
routing never used the endorsed model for that category (the endorsed model
has to actually be exercised live to show up). Try a wider window
(`--since 30d`); confirm traffic was categorized via `stats.py --capability`.

### Q. I got `OperationalError: database is locked`.
Another process (the proxy itself, or a long-running `stats.py`) holds the
SQLite write lock. SQLite writers serialize -- brief contention while the proxy
is writing is normal, just retry. If it's persistent, restart the proxy.

---

If your question isn't here, open a GitHub issue.
