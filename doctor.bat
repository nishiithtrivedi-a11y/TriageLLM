@echo off
REM ===================================================================
REM  TriageLLM - run setup doctor (config / models / cloud-audit / mode)
REM ===================================================================
title TriageLLM Doctor
cd /d "%~dp0"

REM Skip router_hook's module-level critic warmup at import time - doctor
REM does no critic call, so the warmup would just block startup for ~30s.
set TRIAGELLM_SKIP_WARMUP=1

".venv\Scripts\python.exe" doctor.py %*
echo.
echo ============================================================
echo Tip: doctor.bat --cloud-audit   (local-first proof only)
echo      doctor.bat --mode          (routing x cloud status)
echo      doctor.bat --skip-models   (fully offline)
echo      doctor.bat --json          (machine-readable)
echo ============================================================
pause
