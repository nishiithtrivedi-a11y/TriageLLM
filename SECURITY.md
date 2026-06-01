# Security Policy

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report privately via GitHub's **"Report a vulnerability"** button under this
repository's **Security** tab
(https://github.com/nishiithtrivedi-a11y/TriageLLM/security/advisories/new).
That opens a private advisory only the maintainer can see.

You can expect an initial acknowledgement within a few days. This is a
solo-maintained hobby project, not a commercial product with an SLA -- response
times are best-effort.

When reporting, please include:
- What you found and where (file + line if possible)
- How to reproduce it
- The impact you think it has

## Supported versions

TriageLLM is pre-1.0; only the latest released minor line receives fixes.

| Version | Supported |
|---------|-----------|
| 0.2.x   | Yes       |
| < 0.2   | No        |

## Threat model (what TriageLLM is and isn't)

TriageLLM is a **local-first** tool. It runs on your own machine, binds the
proxy to `127.0.0.1` (loopback only -- not exposed to your LAN/Wi-Fi), talks to
a local Ollama, and stores its decision ledger in a local SQLite file. The
intended threat surface is therefore narrow. The security-relevant invariants
the project deliberately maintains:

- **Loopback binding.** The proxy is started with `--host 127.0.0.1`; it is not
  reachable from other devices on your network. (LiteLLM's default would bind to
  all interfaces; TriageLLM overrides that.)
- **No cloud calls by default.** `cloud_escalation.enabled` is `false` in the
  shipped `config.yaml`. When enabled, cloud API keys are referenced by env-var
  *name* only (`api_key_env`) and never stored in YAML; if the named var is
  unset, escalation is silently skipped.
- **SSRF guard.** A non-localhost `OLLAMA_BASE_URL` is refused unless
  `ALLOW_REMOTE_OLLAMA=1` is explicitly set.
- **No secrets in the repo.** Real keys live in your environment (see
  `.env.example`); `.env` is git-ignored.
- **Local prompt data stays local.** Your prompts and the decision ledger live
  in your own SQLite file on your own machine. This is by design for a
  local-first tool -- there is no telemetry and nothing is sent anywhere.

### Out of scope

- Multi-user / multi-tenant isolation -- TriageLLM is single-user by design.
- Exposing the proxy to a network. If you change `--host` to bind beyond
  loopback (not recommended), you take on responsibility for authentication and
  transport security yourself.
- The security of the upstream models, Ollama, LiteLLM, or any cloud provider
  you opt into.

## Good-practice reminders for operators

- Keep `cloud_escalation.enabled: false` unless you genuinely want paid
  fallback, and review which `api_key_env` it points at.
- Don't commit your `.env`. (`.gitignore` already excludes it.)
- Don't change the proxy bind address away from `127.0.0.1` unless you
  understand the exposure.
