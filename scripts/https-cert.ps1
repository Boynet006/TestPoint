<#
.SYNOPSIS
    Makes the HTTPS certificate for the station app, so other devices on the LAN can use their
    camera in the browser (browsers allow a camera only on localhost or HTTPS pages).

.DESCRIPTION
    Creates, in data\certs:
      testpoint-ca.crt / .key   TestPoint's own certificate authority (made once, kept)
      server.crt / .key         the app's certificate, signed by it, for this PC's addresses

    Each device that opens the app trusts testpoint-ca.crt once (download it from
    https://<station PC>:8000/testpoint-ca.crt, open it, Install Certificate, Local Machine,
    "Trusted Root Certification Authorities"); after that, no warning and the camera works.
    Run this again when the station PC gets a new IP address: only server.crt changes, so the
    devices keep trusting it. Then start the app with --https.

    Needs openssl (Git for Windows ships it). Extra names or addresses: -Extra 192.168.1.10,station.local

.EXAMPLE
    .\scripts\https-cert.ps1
    .\scripts\https-cert.ps1 -Extra 10.0.0.5
#>
param([string[]]$Extra = @(), [int]$Days = 825)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Dir = Join-Path $Root 'data\certs'
New-Item -ItemType Directory -Force $Dir | Out-Null

$openssl = (Get-Command openssl -ErrorAction SilentlyContinue).Source
if (-not $openssl) {
    $openssl = @("$env:ProgramFiles\Git\usr\bin\openssl.exe", "$env:ProgramFiles\Git\mingw64\bin\openssl.exe") |
        Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $openssl) { throw 'openssl not found: install Git for Windows, or put openssl on PATH.' }

function Run([string[]]$a) {
    # openssl writes progress to stderr; Windows PowerShell would turn that into an error
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { & $openssl @a 2>&1 | Out-Null } finally { $ErrorActionPreference = $old }
    if ($LASTEXITCODE -ne 0) { throw "openssl $($a[0]) failed" }
}

$caKey = Join-Path $Dir 'testpoint-ca.key'; $caCrt = Join-Path $Dir 'testpoint-ca.crt'
if (-not (Test-Path $caCrt)) {
    Run @('req', '-x509', '-newkey', 'rsa:3072', '-nodes', '-keyout', $caKey, '-out', $caCrt, '-days', '3650',
          '-subj', '/CN=TestPoint local CA', '-addext', 'basicConstraints=critical,CA:TRUE',
          '-addext', 'keyUsage=critical,keyCertSign,cRLSign')
    Write-Host 'Made the TestPoint certificate authority (testpoint-ca.crt).'
}

# Every address this PC answers on, plus its names.
$ips = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike '169.254.*' -and $_.InterfaceAlias -notlike 'vEthernet*' -and $_.InterfaceAlias -notlike '*Loopback*' } |
    Select-Object -ExpandProperty IPAddress
$names = @('localhost', $env:COMPUTERNAME) + $Extra
$san = @('IP:127.0.0.1') + ($ips | ForEach-Object { "IP:$_" }) +
       ($names | Where-Object { $_ } | ForEach-Object { if ($_ -match '^\d+\.\d+\.\d+\.\d+$') { "IP:$_" } else { "DNS:$_" } }) |
       Select-Object -Unique

$key = Join-Path $Dir 'server.key'; $csr = Join-Path $Dir 'server.csr'; $crt = Join-Path $Dir 'server.crt'
$ext = Join-Path $Dir 'server.ext'
Set-Content -Path $ext -Encoding ascii -Value @(
    'basicConstraints=CA:FALSE', 'keyUsage=critical,digitalSignature,keyEncipherment',
    'extendedKeyUsage=serverAuth', "subjectAltName=$($san -join ',')")
Run @('req', '-newkey', 'rsa:2048', '-nodes', '-keyout', $key, '-out', $csr, '-subj', "/CN=TestPoint $env:COMPUTERNAME")
Run @('x509', '-req', '-in', $csr, '-CA', $caCrt, '-CAkey', $caKey, '-CAcreateserial', '-out', $crt,
      '-days', "$Days", '-sha256', '-extfile', $ext)
Remove-Item $csr, $ext -ErrorAction SilentlyContinue

Write-Host "Certificate for: $($san -join ', ')"
Write-Host "Start the app with --https; devices open https://<address>:8000 and trust testpoint-ca.crt once."
