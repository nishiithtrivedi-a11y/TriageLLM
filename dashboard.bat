@echo off
REM ===================================================================
REM  Route LLM - show usage dashboard (read-only, safe anytime)
REM ===================================================================
title Route LLM Dashboard
cd /d "%~dp0"

".venv\Scripts\python.exe" stats.py %*
echo.
echo ============================================================
echo Tip: dashboard.bat --last 50
echo      dashboard.bat --since "2 days"
echo      dashboard.bat --json
echo ============================================================
pause
