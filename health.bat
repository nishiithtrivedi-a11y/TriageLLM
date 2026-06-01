@echo off
REM ===================================================================
REM  TriageLLM - run health check (proxy + ollama + critic + models)
REM ===================================================================
title TriageLLM Health Check
cd /d "%~dp0"

REM Flip console to UTF-8 so the ✓ ! ✗ status glyphs don't crash on cp1252.
chcp 65001 > nul
set PYTHONIOENCODING=utf-8

REM Skip router_hook's module-level critic warmup at import time —
REM health.py runs its own critic probe and times it, so the warmup
REM is redundant *and* it would block this script for ~30s before
REM the first PASS/FAIL line ever prints.
set TRIAGELLM_SKIP_WARMUP=1

".venv\Scripts\python.exe" health.py %*
echo.
echo ============================================================
echo Tip: health.bat --skip-models  (faster, skips per-model probes)
echo      health.bat --json         (machine-readable)
echo ============================================================
pause
