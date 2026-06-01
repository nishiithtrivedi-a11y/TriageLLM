# TriageLLM Local Mode launcher.
# Pick a remembered project folder; opens a terminal where every env-respecting
# AI CLI tool routes through the local TriageLLM proxy. Close window to revert.

$ErrorActionPreference = "Stop"
$Root            = $PSScriptRoot
$RegistryPath    = Join-Path $Root "local_projects.json"
$ProxyHealthUrl  = "http://localhost:4000/health/liveliness"
$OllamaVersionUrl = "http://localhost:11434/api/version"
$ApiKey          = "sk-local-dev"

function Get-Registry {
    if (-not (Test-Path $RegistryPath)) { return @{ projects = @() } }
    try {
        $obj = Get-Content $RegistryPath -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
        if ($null -eq $obj.projects) { return @{ projects = @() } }
        # Normalize to an array even when JSON had a single object.
        return @{ projects = @($obj.projects) }
    } catch {
        Write-Host "[local-mode] registry was corrupt; resetting." -ForegroundColor Yellow
        return @{ projects = @() }
    }
}

function Save-Registry($reg) {
    $reg | ConvertTo-Json -Depth 5 | Set-Content $RegistryPath -Encoding utf8
}

function Select-FolderDialog {
    Add-Type -AssemblyName System.Windows.Forms
    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
    $dialog.Description = "Pick a project folder for Local Mode"
    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        return $dialog.SelectedPath
    }
    return $null
}

function Add-Project {
    $path = Select-FolderDialog
    if (-not $path) { Write-Host "Cancelled." -ForegroundColor Yellow; return }
    if (-not (Test-Path $path)) { Write-Host "Folder does not exist: $path" -ForegroundColor Red; return }
    $reg = Get-Registry
    if ($reg.projects | Where-Object { $_.path -eq $path }) {
        Write-Host "Already remembered: $path" -ForegroundColor Yellow
        return
    }
    $name = Split-Path $path -Leaf
    $reg.projects = @($reg.projects) + @([pscustomobject]@{ name = $name; path = $path })
    Save-Registry $reg
    Write-Host "Added: $name ($path)" -ForegroundColor Green
}

function Remove-Project {
    $reg = Get-Registry
    if ($reg.projects.Count -eq 0) { Write-Host "Nothing to remove." -ForegroundColor Yellow; return }
    $sel = Read-Host "Number to remove (1-$($reg.projects.Count))"
    if ($sel -notmatch '^\d+$') { Write-Host "Not a number." -ForegroundColor Yellow; return }
    $idx = [int]$sel - 1
    if ($idx -lt 0 -or $idx -ge $reg.projects.Count) { Write-Host "Out of range." -ForegroundColor Yellow; return }
    $removed = $reg.projects[$idx]
    $reg.projects = @($reg.projects | Where-Object { $_.path -ne $removed.path })
    Save-Registry $reg
    Write-Host "Removed: $($removed.name)" -ForegroundColor Green
}

function Test-ProxyUp {
    try {
        $r = Invoke-WebRequest -Uri $ProxyHealthUrl -TimeoutSec 3 -UseBasicParsing
        return $r.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Test-OllamaUp {
    try {
        $r = Invoke-WebRequest -Uri $OllamaVersionUrl -TimeoutSec 3 -UseBasicParsing
        return $r.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Ensure-Ollama {
    if (Test-OllamaUp) {
        Write-Host "[local-mode] Ollama is running." -ForegroundColor Green
        return $true
    }
    Write-Host "[local-mode] Ollama not running; starting it..." -ForegroundColor Yellow
    $ollamaApp = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama app.exe"
    if (-not (Test-Path $ollamaApp)) {
        Write-Host "[local-mode] Could not find Ollama at:" -ForegroundColor Red
        Write-Host "  $ollamaApp" -ForegroundColor Red
        Write-Host "Start Ollama manually from the Start Menu, then try again." -ForegroundColor Yellow
        return $false
    }
    Start-Process -FilePath $ollamaApp
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        if (Test-OllamaUp) {
            Write-Host "[local-mode] Ollama is healthy." -ForegroundColor Green
            return $true
        }
    }
    Write-Host "[local-mode] Ollama did not become healthy in 30s." -ForegroundColor Red
    return $false
}

function Ensure-Proxy {
    # Ollama must be up first - the proxy routes to it.
    if (-not (Ensure-Ollama)) { return $false }

    if (Test-ProxyUp) {
        Write-Host "[local-mode] proxy already running." -ForegroundColor Green
        return $true
    }
    Write-Host "[local-mode] proxy not running; starting it..." -ForegroundColor Yellow
    # NOTE: the project path contains a space ("D:\Route LLM"). Start-Process
    # does NOT quote -ArgumentList array elements, so the path MUST be wrapped
    # in literal quotes or the space splits it into two arguments and the
    # launch silently fails. -NoExit keeps the window open so errors are visible.
    $proxyScript = Join-Path $Root "start_proxy.ps1"
    Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-ExecutionPolicy","Bypass","-NoProfile","-NoExit","-File","`"$proxyScript`""
    )
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 1
        if (Test-ProxyUp) {
            Write-Host "[local-mode] proxy is healthy." -ForegroundColor Green
            return $true
        }
    }
    Write-Host "[local-mode] proxy did not become healthy in 60s. Check the proxy window." -ForegroundColor Red
    return $false
}

function Start-LocalModeShell($project) {
    $name = $project.name
    $path = $project.path
    if (-not (Test-Path $path)) {
        Write-Host "[local-mode] folder no longer exists: $path" -ForegroundColor Red
        Write-Host "Use [R] to remove it from the list." -ForegroundColor Yellow
        return
    }
    if (-not (Ensure-Proxy)) { return }

    $sessionScript = Join-Path $env:TEMP "triagellm_localmode_session.ps1"
    $content = @"
`$env:OPENAI_BASE_URL = 'http://localhost:4000/v1'
`$env:OPENAI_API_BASE = 'http://localhost:4000/v1'
`$env:OPENAI_API_KEY = '$ApiKey'
`$env:ANTHROPIC_BASE_URL = 'http://localhost:4000'
`$env:ANTHROPIC_AUTH_TOKEN = '$ApiKey'
`$env:ANTHROPIC_MODEL = 'local-auto'
`$env:ANTHROPIC_SMALL_FAST_MODEL = 'local-auto'
Set-Location '$path'
Write-Host ''
Write-Host '===============================================================' -ForegroundColor Green
Write-Host '  LOCAL MODE ACTIVE - $name' -ForegroundColor Green
Write-Host '  All AI tools in THIS window route to local Ollama (free).'
Write-Host '  Close this window to return to normal cloud AI.'
Write-Host ''
Write-Host '   Claude Code:  claude'
Write-Host '   Codex CLI:    codex --model local-auto'
Write-Host '   aider:        aider --model openai/local-auto'
Write-Host '===============================================================' -ForegroundColor Green
Write-Host ''
"@
    Set-Content -Path $sessionScript -Value $content -Encoding utf8
    # Quote the script path (defensive: %TEMP% usually has no space, but be safe).
    Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoExit","-NoProfile","-ExecutionPolicy","Bypass","-File","`"$sessionScript`""
    )
    Write-Host "[local-mode] opened Local Mode window for $name." -ForegroundColor Green
}

function Show-Menu($reg) {
    Write-Host ""
    Write-Host "===============================================================" -ForegroundColor Cyan
    Write-Host "  TriageLLM - Local Mode" -ForegroundColor Cyan
    Write-Host "===============================================================" -ForegroundColor Cyan
    Write-Host ""
    if ($reg.projects.Count -eq 0) {
        Write-Host "  (no remembered projects yet - use [A] to add one)"
    } else {
        Write-Host "  Remembered projects:"
        for ($i = 0; $i -lt $reg.projects.Count; $i++) {
            $p = $reg.projects[$i]
            "    [{0}] {1,-16} {2}" -f ($i + 1), $p.name, $p.path | Write-Host
        }
    }
    Write-Host ""
    Write-Host "    [A] Add a project folder"
    Write-Host "    [R] Remove a project"
    Write-Host "    [Q] Quit"
    Write-Host ""
}

# Main loop
while ($true) {
    $reg = Get-Registry
    Show-Menu $reg
    $choice = Read-Host "Pick a number or letter"
    switch -Regex ($choice) {
        '^[Qq]$' { return }
        '^[Aa]$' { Add-Project; continue }
        '^[Rr]$' { Remove-Project; continue }
        '^\d+$'  {
            $reg = Get-Registry
            $idx = [int]$choice - 1
            if ($idx -ge 0 -and $idx -lt $reg.projects.Count) {
                Start-LocalModeShell $reg.projects[$idx]
            } else {
                Write-Host "No project at number $choice." -ForegroundColor Yellow
            }
            continue
        }
        default  { Write-Host "Unrecognized choice: $choice" -ForegroundColor Yellow }
    }
}
