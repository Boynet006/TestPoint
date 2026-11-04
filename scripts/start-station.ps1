# TestPoint on this PC (no Docker): starts the app and the camera service when they are not
# running yet, waits for the app, then opens the station screens in Chrome. Run by the desktop
# shortcut (scripts/make-shortcut.ps1); safe to run again: what already runs is left alone.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$logs = Join-Path $root "data\logs"
New-Item -ItemType Directory -Force $logs | Out-Null

function Up($url) {
    try { $null = Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 2; return $true } catch { return $false }
}
# The app's certificate is TestPoint's own: this check trusts it (only for these local requests).
Add-Type @"
using System.Net; using System.Security.Cryptography.X509Certificates;
public class TestPointTrust : ICertificatePolicy { public bool CheckValidationResult(ServicePoint s, X509Certificate c, WebRequest r, int p) { return true; } }
"@ -ErrorAction SilentlyContinue
[System.Net.ServicePointManager]::CertificatePolicy = New-Object TestPointTrust

# the app (HTTPS, so the browsers allow the webcam)
if (-not (Up "https://127.0.0.1:8000/api/settings")) {
    $env:DISABLE_SQLALCHEMY_CEXT_RUNTIME = "1"
    Start-Process $py -ArgumentList "-m testpoint.api.app --host 0.0.0.0 --port 8000 --db sqlite:///data/dev.db --https" `
        -WorkingDirectory $root -WindowStyle Hidden `
        -RedirectStandardOutput "$logs\app.log" -RedirectStandardError "$logs\app.err.log"
}
# the camera service (the IP cameras, the counting and the finish line)
if (-not (Up "http://127.0.0.1:8099/health")) {
    $ff = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($ff) { $env:TESTPOINT_FFMPEG = $ff.FullName }
    Start-Process $py -ArgumentList "-m testpoint.camhost" -WorkingDirectory $root -WindowStyle Hidden `
        -RedirectStandardOutput "$logs\camhost.log" -RedirectStandardError "$logs\camhost.err.log"
}
for ($i = 0; $i -lt 60 -and -not (Up "https://127.0.0.1:8000/api/settings"); $i++) { Start-Sleep 1 }

# The address the screens were set up on (their saved settings belong to it); this PC's own name
# when it is not on the camera network.
$ip = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object { $_.IPAddress -eq "192.168.250.149" }).IPAddress
$url = if ($ip) { "https://$($ip):8000/ui/" } else { "https://localhost:8000/ui/" }
$chrome = @("$env:ProgramFiles\Google\Chrome\Application\chrome.exe", "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
            "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($chrome) { Start-Process $chrome $url } else { Start-Process $url }
