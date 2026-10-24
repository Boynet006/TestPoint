<#
.SYNOPSIS
    Start, stop and check the TestPoint station (Docker).

.DESCRIPTION
    start   Starts Docker Desktop if needed, waits for it, starts the TestPoint
            services and opens the station UI in the browser.
    stop    Stops the TestPoint services (Docker Desktop keeps running).
    status  Shows Docker, the GPU and the TestPoint services.
    logs    Follows the logs of the running services (Ctrl+C to leave).

    The rep-counting engine starts only with -Engine, or when .env contains
    START_ENGINE=1, because it needs a real camera address in CAMERA_URL.

    start also starts the camera service on this PC (python -m testpoint.camhost,
    127.0.0.1:8099) when the Python environment (.venv) is installed: the finish
    page uses it to start and stop the finish camera. stop stops it too.

.EXAMPLE
    .\scripts\testpoint.ps1 start
    .\scripts\testpoint.ps1 start -Engine
    .\scripts\testpoint.ps1 start -Rebuild      # after a code update: rebuild the images (needs internet)
    .\scripts\testpoint.ps1 stop
#>
param(
    [ValidateSet('start', 'stop', 'status', 'logs')]
    [string]$Action = 'start',
    [switch]$Engine,
    [switch]$NoBrowser,
    [switch]$Rebuild,
    [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Say([string]$Text, [string]$Colour = 'Gray') { Write-Host $Text -ForegroundColor $Colour }

function Find-Docker {
    $cmd = Get-Command docker -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $default = Join-Path $env:ProgramFiles 'Docker\Docker\resources\bin\docker.exe'
    if (Test-Path $default) { return $default }
    throw 'Docker is not installed. See README.md, section "Install Docker on the Windows PC".'
}
$Docker = Find-Docker

function Test-Engine {
    # Windows PowerShell 5.1 turns a native command's stderr into an error record when it is
    # redirected; with 'Stop' that aborts the script, so "engine not running" would crash the
    # very check meant to start Docker Desktop. Relax it for this one call.
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $null = & $Docker version --format '{{.Server.Version}}' 2>$null
        return ($LASTEXITCODE -eq 0)
    } finally {
        $ErrorActionPreference = $old
    }
}

$CamHostPid = Join-Path $Root 'data\camhost.pid'

function Test-CamHost {
    try { $null = Invoke-WebRequest 'http://127.0.0.1:8099/health' -UseBasicParsing -TimeoutSec 1; return $true } catch { return $false }
}

function Start-CamHost {
    # The finish camera needs this PC's cameras and GPU, which the Docker app cannot reach.
    $py = Join-Path $Root '.venv\Scripts\python.exe'
    if (-not (Test-Path $py)) { Say 'Finish camera service: not installed on this PC (no .venv). Time finishes by hand.' 'Yellow'; return }
    if (Test-CamHost) { Say 'Finish camera service: running' 'Green'; return }
    $p = Start-Process -FilePath $py -ArgumentList @('-m', 'testpoint.camhost') -WorkingDirectory $Root -WindowStyle Hidden -PassThru
    Set-Content -Path $CamHostPid -Value $p.Id
    for ($i = 0; $i -lt 20; $i++) { if (Test-CamHost) { Say 'Finish camera service: started' 'Green'; return }; Start-Sleep -Milliseconds 500 }
    Say 'Finish camera service: started, but not answering yet.' 'Yellow'
}

function Stop-CamHost {
    if (Test-Path $CamHostPid) {
        $id = Get-Content $CamHostPid
        # /T: the finish camera it started goes too
        $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        & taskkill.exe /PID $id /T /F 2>$null | Out-Null
        $ErrorActionPreference = $old
        Remove-Item $CamHostPid -ErrorAction SilentlyContinue
        Say 'Finish camera service stopped.' 'Green'
    }
}

function Start-DockerDesktop {
    if (Test-Engine) { return }
    $exe = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
    if (-not (Get-Process 'Docker Desktop' -ErrorAction SilentlyContinue)) {
        Say 'Starting Docker Desktop...'
        Start-Process $exe
    }
    Say "Waiting for the Docker engine (up to $TimeoutSeconds s)..."
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-Engine) { Say 'Docker engine is ready.' 'Green'; return }
        Start-Sleep -Seconds 3
    }
    throw "The Docker engine did not start within $TimeoutSeconds s. Open Docker Desktop to see why (a pending update or licence prompt, or WSL not running)."
}

function Test-Image([string]$Image) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'     # see Test-Engine
    try {
        $null = & $Docker image inspect $Image 2>$null
        return ($LASTEXITCODE -eq 0)
    } finally {
        $ErrorActionPreference = $old
    }
}

function Get-EnvValue([string]$Name) {
    if (-not (Test-Path '.env')) { return $null }
    $line = Get-Content '.env' | Where-Object { $_ -match "^\s*$Name\s*=" } | Select-Object -Last 1
    if ($line) { return ($line -split '=', 2)[1].Trim() }
    return $null
}

function Invoke-Compose([string[]]$ComposeArgs) {
    & $Docker compose @ComposeArgs
    if ($LASTEXITCODE -ne 0) { throw "docker compose $($ComposeArgs -join ' ') failed (exit $LASTEXITCODE)." }
}

function Show-Status {
    if (-not (Test-Engine)) { Say 'Docker engine: not running' 'Yellow'; return }
    Say "Docker engine: running ($(& $Docker version --format '{{.Server.Version}}'))" 'Green'
    Say ''
    & $Docker compose --profile gpu --profile cpu --profile prototype ps --format 'table {{.Service}}\t{{.Status}}\t{{.Ports}}'
    $port = Get-EnvValue 'APP_PORT'
    if (-not $port) { $port = '8080' }
    Say ''
    Say "Station UI: http://localhost:$port" 'Cyan'
}

switch ($Action) {
    'start' {
        if (-not (Test-Path '.env')) {
            Copy-Item '.env.example' '.env'
            Say 'Created .env from .env.example. Put the real camera address in CAMERA_URL.' 'Yellow'
        }
        Start-DockerDesktop

        $services = @('app')
        $composeArgs = @()
        $startEngine = $Engine -or ((Get-EnvValue 'START_ENGINE') -eq '1')
        if ($startEngine) {
            $composeArgs += @('--profile', 'gpu')
            $services += 'engine'
        }
        # Images do not travel with the folder: a PC that never built them (or loaded them
        # with docker load) has none. Build the missing ones once; this needs internet.
        $version = Get-EnvValue 'TESTPOINT_VERSION'
        if (-not $version) { $version = '0.1.0' }
        $images = @{ app = "testpoint/app:$version"; engine = "testpoint/engine-gpu:$version" }
        $missing = @($services | Where-Object { $Rebuild -or -not (Test-Image $images[$_]) })
        if ($missing.Count -gt 0) {
            if ($Rebuild) { Say "Rebuilding $($missing -join ', ') with the current code (needs internet)..." 'Yellow' }
            else { Say "First start on this PC: building $($missing -join ', ') (needs internet once; a few minutes, the GPU engine longer)..." 'Yellow' }
            Invoke-Compose ($composeArgs + @('build') + $missing)
        }

        Say "Starting: $($services -join ', ')"
        # --no-build: use the images on this PC (the site is offline).
        Invoke-Compose ($composeArgs + @('up', '-d', '--no-build') + $services)

        $port = Get-EnvValue 'APP_PORT'
        if (-not $port) { $port = '8080' }
        $url = "http://localhost:$port"
        $ready = $false
        for ($i = 0; $i -lt 20; $i++) {
            try { $null = Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 2; $ready = $true; break } catch { Start-Sleep -Milliseconds 500 }
        }
        if ($ready) { Say "TestPoint is running: $url" 'Green' } else { Say "Started, but $url is not answering yet." 'Yellow' }
        Start-CamHost
        if ($startEngine) { Say 'Engine output: .\scripts\testpoint.ps1 logs   (annotated video goes to the data folder)' }
        if (-not $NoBrowser) { Start-Process $url }
    }
    'stop' {
        Stop-CamHost
        if (-not (Test-Engine)) { Say 'Docker engine is not running; nothing to stop.' 'Yellow'; break }
        Invoke-Compose @('--profile', 'gpu', '--profile', 'cpu', '--profile', 'test', '--profile', 'prototype', 'down')
        Say 'TestPoint stopped.' 'Green'
    }
    'status' { Show-Status }
    'logs' {
        if (-not (Test-Engine)) { Say 'Docker engine is not running.' 'Yellow'; break }
        & $Docker compose --profile gpu logs -f --tail 100
    }
}
