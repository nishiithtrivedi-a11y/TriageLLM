@echo off
REM ===================================================================
REM  TriageLLM - Local Mode launcher (per-project local AI routing)
REM ===================================================================
title TriageLLM Local Mode
cd /d "%~dp0"
powershell.exe -ExecutionPolicy Bypass -NoProfile -File "%~dp0local_mode.ps1"
