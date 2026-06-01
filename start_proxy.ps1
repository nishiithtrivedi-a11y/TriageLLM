# Launch the LiteLLM proxy with the right env. Run from project root.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# LiteLLM's startup banner contains emoji; on Windows the default cp1252 console
# crashes with UnicodeEncodeError before the proxy can serve. Force UTF-8 (65001).
chcp 65001 > $null

# --- TriageLLM banner (pure ASCII so it renders on any console) -------------
Write-Host @"

===============================================================

        T R I A G E L L M
        Smart tier-routing for local Ollama models

===============================================================
  Stop paying cloud-API rates for the easy stuff.

  GitHub  : github.com/nishiithtrivedi-a11y/TriageLLM
  Author  : Nishith Trivedi - an SAP analyst learning to build
            with AI. Honest feedback and PRs are very welcome!
  LinkedIn: linkedin.com/in/nishith-t-5220a5b4

  Like it? A GitHub star or a LinkedIn shout-out makes my day.
===============================================================

  Starting the proxy (LiteLLM does the heavy lifting below)...

"@ -ForegroundColor Cyan

Set-Location $PSScriptRoot
# Security: bind to loopback only. LiteLLM defaults to --host 0.0.0.0 which
# exposes the proxy on every network interface (LAN / Wi-Fi). With the
# documented `sk-local-dev` API key, that's a remote-use risk and could
# trigger paid cloud escalation if it's enabled. Loopback-only by design.
& "$PSScriptRoot\.venv\Scripts\litellm.exe" --config "$PSScriptRoot\config.yaml" --host 127.0.0.1 --port 4000
