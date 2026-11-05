<#
.SYNOPSIS
    TestPoint Network: switch this PC between Offline mode (a fixed address on the camera
    network, no internet needed) and Online mode (automatic address from the router, DHCP), and
    see at a glance this PC's address, the internet, each camera and TestPoint.

.DESCRIPTION
    The cameras keep their own fixed addresses (best practice; they are only checked, never
    changed). TestPoint is opened by this PC's name, so a new address changes nothing for it.
    Changing the address needs administrator rights: the window asks Windows for them (UAC)
    only when a mode button is pressed. Settings are kept in data\network.json.

    Run:  powershell -ExecutionPolicy Bypass -File scripts\network.ps1
    (the "TestPoint Network" desktop icon does this; scripts\make-shortcut.ps1 makes it)
#>
param([ValidateSet("", "Offline", "Online")][string]$Apply = "", [string]$Adapter = "", [string]$Address = "",
      [string]$Snapshot = "")   # -Snapshot file.png: draw the window to a picture and quit (for the guide)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$CfgPath = Join-Path $Root "data\network.json"

# ------------------------------------------------------------------ settings
function Load-Config {
    $cfg = [ordered]@{
        adapter = ""; offline_ip = "192.168.250.240"; prefix = 24
        cameras = @(@{ name = "Start line"; ip = "192.168.250.241" }, @{ name = "Finish line"; ip = "192.168.250.242" },
                    @{ name = "Exercise station"; ip = "192.168.250.243" })
    }
    if (Test-Path $CfgPath) {
        try { (Get-Content $CfgPath -Raw | ConvertFrom-Json).PSObject.Properties | ForEach-Object { $cfg[$_.Name] = $_.Value } } catch { }
    }
    return $cfg
}
function Save-Config($cfg) {
    New-Item -ItemType Directory -Force (Split-Path $CfgPath) | Out-Null
    $cfg | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $CfgPath
}

# ------------------------------------------------------------------ checks
function Test-Port([string]$ip, [int]$port, [int]$ms = 600) {
    $c = New-Object System.Net.Sockets.TcpClient
    try { $t = $c.BeginConnect($ip, $port, $null, $null); return ($t.AsyncWaitHandle.WaitOne($ms) -and $c.Connected) }
    catch { return $false } finally { $c.Close() }
}
function Camera-Adapter($cfg) {
    # the wired adapter that reaches the cameras: the one named in the settings, else the one on
    # the cameras' network, else the first physical adapter that is connected
    $all = Get-NetAdapter -Physical -ErrorAction SilentlyContinue | Where-Object Status -eq "Up"
    if ($cfg.adapter -and ($all | Where-Object Name -eq $cfg.adapter)) { return $cfg.adapter }
    $net = ($cfg.offline_ip -split "\.")[0..2] -join "."
    $on = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object { $_.IPAddress -like "$net.*" } |
        Select-Object -First 1
    if ($on) { return $on.InterfaceAlias }
    return ($all | Sort-Object { $_.MediaType -ne "802.3" } | Select-Object -First 1).Name
}
function PC-State($alias) {
    $ip = Get-NetIPAddress -InterfaceAlias $alias -AddressFamily IPv4 -ErrorAction SilentlyContinue | Select-Object -First 1
    $dhcp = (Get-NetIPInterface -InterfaceAlias $alias -AddressFamily IPv4 -ErrorAction SilentlyContinue).Dhcp
    return @{ ip = if ($ip) { "$($ip.IPAddress)/$($ip.PrefixLength)" } else { "none" }; dhcp = "$dhcp" }
}

# ------------------------------------------------------------------ the change (administrator)
function Apply-Mode([string]$mode, [string]$alias, [string]$addr, [int]$prefix) {
    if ($mode -eq "Offline") {
        Set-NetIPInterface -InterfaceAlias $alias -AddressFamily IPv4 -Dhcp Disabled
        Get-NetIPAddress -InterfaceAlias $alias -AddressFamily IPv4 -ErrorAction SilentlyContinue | Remove-NetIPAddress -Confirm:$false
        Get-NetRoute -InterfaceAlias $alias -DestinationPrefix "0.0.0.0/0" -ErrorAction SilentlyContinue | Remove-NetRoute -Confirm:$false
        New-NetIPAddress -InterfaceAlias $alias -IPAddress $addr -PrefixLength $prefix | Out-Null   # no gateway: no internet expected
        Set-DnsClientServerAddress -InterfaceAlias $alias -ResetServerAddresses
    } else {
        Get-NetIPAddress -InterfaceAlias $alias -AddressFamily IPv4 -ErrorAction SilentlyContinue | Remove-NetIPAddress -Confirm:$false
        Get-NetRoute -InterfaceAlias $alias -DestinationPrefix "0.0.0.0/0" -ErrorAction SilentlyContinue | Remove-NetRoute -Confirm:$false
        Set-NetIPInterface -InterfaceAlias $alias -AddressFamily IPv4 -Dhcp Enabled
        Set-DnsClientServerAddress -InterfaceAlias $alias -ResetServerAddresses
        ipconfig /renew "$alias" | Out-Null
    }
}
if ($Apply) {            # started again with administrator rights by the window's button
    try { Apply-Mode $Apply $Adapter $Address (Load-Config).prefix; exit 0 }
    catch { Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show("$_", "TestPoint Network") | Out-Null; exit 1 }
}

# ------------------------------------------------------------------ the window
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()
$cfg = Load-Config
$bg = [Drawing.Color]::FromArgb(16, 22, 29); $panel = [Drawing.Color]::FromArgb(27, 37, 47)
$amber = [Drawing.Color]::FromArgb(255, 176, 0); $go = [Drawing.Color]::FromArgb(25, 210, 122)
$bad = [Drawing.Color]::FromArgb(255, 59, 59); $ink = [Drawing.Color]::FromArgb(242, 245, 247); $muted = [Drawing.Color]::FromArgb(139, 155, 171)
$fontH = New-Object Drawing.Font("Segoe UI Semibold", 15); $font = New-Object Drawing.Font("Segoe UI", 10.5)
$fontS = New-Object Drawing.Font("Segoe UI", 9)

$form = New-Object Windows.Forms.Form
$form.Text = "TestPoint Network"; $form.Size = New-Object Drawing.Size(560, 560); $form.StartPosition = "CenterScreen"
$form.BackColor = $bg; $form.ForeColor = $ink; $form.Font = $font; $form.FormBorderStyle = "FixedSingle"; $form.MaximizeBox = $false
$ico = Join-Path $Root "web\station\icon.ico"; if (Test-Path $ico) { $form.Icon = New-Object Drawing.Icon($ico) }

function Label($text, $x, $y, $w = 500, $f = $font, $c = $ink) {
    $l = New-Object Windows.Forms.Label; $l.Text = $text; $l.Location = New-Object Drawing.Point($x, $y)
    $l.Size = New-Object Drawing.Size($w, 24); $l.Font = $f; $l.ForeColor = $c; $form.Controls.Add($l); return $l
}
function Button($text, $x, $y, $w, $back, $fore) {
    $b = New-Object Windows.Forms.Button; $b.Text = $text; $b.Location = New-Object Drawing.Point($x, $y)
    $b.Size = New-Object Drawing.Size($w, 44); $b.FlatStyle = "Flat"; $b.BackColor = $back; $b.ForeColor = $fore
    $b.FlatAppearance.BorderSize = 0; $b.Font = New-Object Drawing.Font("Segoe UI Semibold", 10.5); $form.Controls.Add($b); return $b
}

Label "TESTPOINT NETWORK" 20 14 400 $fontH $amber | Out-Null
$mode = Label "" 20 46 510 (New-Object Drawing.Font("Segoe UI Semibold", 12))
Label "THIS PC" 20 84 200 $fontS $muted | Out-Null
$pcLine = Label "" 20 104 510; $netLine = Label "" 20 128 510
Label "CAMERAS (fixed addresses, only checked)" 20 166 400 $fontS $muted | Out-Null
$camLines = @(); $y = 186
foreach ($c in $cfg.cameras) { $camLines += Label "" 20 $y 510; $y += 24 }
Label "TESTPOINT" 20 ($y + 14) 200 $fontS $muted | Out-Null
$appLine = Label "" 20 ($y + 34) 510

Label "Offline address" 20 ($y + 74) 130 $fontS $muted | Out-Null
$addrBox = New-Object Windows.Forms.TextBox; $addrBox.Location = New-Object Drawing.Point(150, ($y + 70)); $addrBox.Size = New-Object Drawing.Size(150, 24)
$addrBox.Text = $cfg.offline_ip; $addrBox.BackColor = $panel; $addrBox.ForeColor = $ink; $addrBox.BorderStyle = "FixedSingle"; $form.Controls.Add($addrBox)
Label "free, on the cameras' network" 306 ($y + 74) 230 $fontS $muted | Out-Null

$bOff = Button "OFFLINE MODE`n(fixed address)" 20 ($y + 108) 160 $amber ([Drawing.Color]::FromArgb(26, 18, 0))
$bOn = Button "ONLINE MODE`n(automatic, DHCP)" 190 ($y + 108) 160 $panel $ink
$bApp = Button "START TESTPOINT" 360 ($y + 108) 160 $go ([Drawing.Color]::FromArgb(4, 32, 15))
$msg = Label "" 20 ($y + 162) 510 $fontS $muted

function Refresh {
    $alias = Camera-Adapter $cfg
    $st = PC-State $alias
    $fixed = $st.dhcp -eq "Disabled"
    $mode.Text = if ($fixed) { "OFFLINE MODE: fixed address" } else { "ONLINE MODE: automatic address (DHCP)" }
    $mode.ForeColor = if ($fixed) { $amber } else { $go }
    $pcLine.Text = "$alias   $($st.ip)   name $env:COMPUTERNAME"
    $inet = Test-Port "1.1.1.1" 443 700
    $netLine.Text = if ($inet) { "Internet: connected" } else { "Internet: not connected (TestPoint does not need it)" }
    $netLine.ForeColor = if ($inet) { $go } else { $muted }
    for ($i = 0; $i -lt $cfg.cameras.Count; $i++) {
        $c = $cfg.cameras[$i]; $ok = Test-Port $c.ip 554
        $camLines[$i].Text = "{0}  {1}   {2}" -f ($(if ($ok) { "●" } else { "○" })), $c.name, "$($c.ip)   $(if ($ok) { 'answering' } else { 'not answering: check its cable and power' })"
        $camLines[$i].ForeColor = if ($ok) { $go } else { $bad }
    }
    $up = Test-Port "127.0.0.1" 8000 400
    $appLine.Text = if ($up) { "Running:  https://$($env:COMPUTERNAME.ToLower()):8000/ui/" } else { "Not running: press Start TestPoint" }
    $appLine.ForeColor = if ($up) { $go } else { $muted }
    $script:alias = $alias
}

function Switch-To([string]$m) {
    $addr = $addrBox.Text.Trim()
    if ($m -eq "Offline") {
        if ($addr -notmatch '^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$') { $msg.Text = "The offline address is not an address (e.g. 192.168.250.240)"; $msg.ForeColor = $bad; return }
        $mine = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue).IPAddress -contains $addr
        if (-not $mine -and (Test-Connection -ComputerName $addr -Count 1 -Quiet -ErrorAction SilentlyContinue)) {
            $msg.Text = "$addr is used by another device: choose another address"; $msg.ForeColor = $bad; return
        }
        $cfg.offline_ip = $addr; $cfg.adapter = $script:alias; Save-Config $cfg
    }
    $msg.Text = "Changing the address (Windows asks for permission)..."; $msg.ForeColor = $amber; $form.Refresh()
    try {
        $p = Start-Process powershell -Verb RunAs -Wait -PassThru -WindowStyle Hidden -ArgumentList @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"", "-Apply", $m, "-Adapter", "`"$script:alias`"", "-Address", $addr)
        Start-Sleep 3
        $msg.Text = if ($p.ExitCode -eq 0) { "$m mode is on. TestPoint keeps working: it is opened by this PC's name." } else { "The change did not go through." }
        $msg.ForeColor = if ($p.ExitCode -eq 0) { $go } else { $bad }
    } catch { $msg.Text = "Not changed (permission refused)."; $msg.ForeColor = $bad }
    Refresh
}

$bOff.Add_Click({ Switch-To "Offline" })
$bOn.Add_Click({ Switch-To "Online" })
$bApp.Add_Click({
    $msg.Text = "Starting TestPoint..."; $msg.ForeColor = $amber; $form.Refresh()
    Start-Process powershell -WindowStyle Hidden -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$Root\scripts\start-station.ps1`"")
    Start-Sleep 6; $msg.Text = ""; Refresh
})
$timer = New-Object Windows.Forms.Timer; $timer.Interval = 5000; $timer.Add_Tick({ Refresh }); $timer.Start()
if ($Snapshot) {
    # drawn off screen: the controls only paint once the window has been shown
    $form.StartPosition = "Manual"; $form.Location = New-Object Drawing.Point(-3000, -3000)
    $form.Show(); Refresh; [Windows.Forms.Application]::DoEvents()
    $bmp = New-Object Drawing.Bitmap($form.Width, $form.Height)
    $form.DrawToBitmap($bmp, (New-Object Drawing.Rectangle(0, 0, $form.Width, $form.Height)))
    $bmp.Save($Snapshot); $form.Close(); exit 0
}
# Started from a shortcut set to "minimised" (so no console flashes), Windows would open this
# window minimised or hidden too: it is put back to normal and brought to the front.
$form.WindowState = "Normal"
$form.Add_Shown({ $form.WindowState = "Normal"; $form.TopMost = $true; $form.Activate(); $form.TopMost = $false; Refresh })
[void]$form.ShowDialog()
