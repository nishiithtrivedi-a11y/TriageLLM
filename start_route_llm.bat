@echo off
REM ===================================================================
REM  Route LLM - one-click launcher
REM  Starts Ollama if needed, launches the proxy, waits for it to be
REM  healthy, prints the cheat-sheet.
REM ===================================================================

title Route LLM Launcher
cd /d "%~dp0"

echo.
echo === Route LLM ===
echo.

REM --- 1. Make sure Ollama is running --------------------------------
tasklist /FI "IMAGENAME eq ollama.exe" | find /I "ollama.exe" >nul
if errorlevel 1 (
    echo Ollama not running. Starting...
    if exist "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe" (
        start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
        timeout /t 5 /nobreak >nul
    ) else (
        echo ERROR: cannot find Ollama at %LOCALAPPDATA%\Programs\Ollama\ollama app.exe
        echo Please start Ollama manually from the Start Menu, then run this again.
        pause
        exit /b 1
    )
) else (
    echo [OK] Ollama is running.
)

REM --- 2. Launch the proxy in its own window -------------------------
echo Starting proxy on http://localhost:4000 ...
start "Route LLM Proxy" powershell.exe -ExecutionPolicy Bypass -NoProfile -NoExit -File "%~dp0start_proxy.ps1"

REM --- 3. Wait until the proxy is actually healthy -------------------
echo Waiting for proxy to become ready (usually 5-15 seconds)...
set /a tries=0
:waitloop
set /a tries+=1
if %tries% GTR 30 (
    echo.
    echo TIMEOUT: proxy did not become healthy within 60 seconds.
    echo Check the "Route LLM Proxy" window for errors.
    pause
    exit /b 1
)
timeout /t 2 /nobreak >nul
powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing http://localhost:4000/health/liveliness -TimeoutSec 2).StatusCode } catch { exit 1 }" >nul 2>&1
if errorlevel 1 goto waitloop

REM --- 4. Print the cheat-sheet --------------------------------------
echo.
echo ============================================================
echo   TriageLLM is UP at http://localhost:4000
echo.
echo   API key (Authorization header): sk-local-dev
echo   Model name to send:             local-auto
echo ============================================================
echo.
echo The proxy is running in the "Route LLM Proxy" window.
echo - To stop the proxy: close that window, or run stop_route_llm.bat
echo - To see usage stats: run dashboard.bat
echo - Per-project local routing: run local_mode.bat
echo.
echo ------------------------------------------------------------
echo   TriageLLM by Nishith Trivedi (Apache-2.0)
echo   GitHub  : github.com/nishiithtrivedi-a11y/TriageLLM
echo   LinkedIn: linkedin.com/in/nishith-t-5220a5b4
echo   A star or a shout-out is hugely appreciated!
echo ------------------------------------------------------------
echo.
echo This launcher window can be closed safely.
timeout /t 10
