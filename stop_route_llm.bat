@echo off
REM ===================================================================
REM  Route LLM - stop the proxy (leaves Ollama running)
REM ===================================================================
title Route LLM Stop

echo Stopping Route LLM proxy...
taskkill /FI "WINDOWTITLE eq Route LLM Proxy*" /T /F >nul 2>&1
if errorlevel 1 (
    echo No proxy window found. Nothing to stop.
) else (
    echo [OK] Proxy stopped.
)
echo.
echo Ollama is still running in the background. To stop it:
echo right-click the llama icon in the system tray and choose "Quit Ollama".
echo.
timeout /t 5
