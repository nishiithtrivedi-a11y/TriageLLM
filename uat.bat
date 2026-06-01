@echo off
REM ===================================================================
REM  TriageLLM - User Acceptance Test (live proxy + real Ollama)
REM  Requires: proxy already running (run start_route_llm.bat first)
REM ===================================================================
title TriageLLM UAT
cd /d "%~dp0"

REM Flip console to UTF-8 so the ✓ ! ✗ status glyphs don't crash on cp1252.
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
set TRIAGELLM_SKIP_WARMUP=1

".venv\Scripts\python.exe" uat.py %*
echo.
echo ============================================================
echo Tip: uat.bat --skip-stream         (skip streaming phase)
echo      uat.bat --skip-orchestration  (skip live round-trip)
echo ============================================================
pause
