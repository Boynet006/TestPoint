# Puts a "TestPoint" icon on this user's desktop: it runs scripts/start-station.ps1 (starts what is
# not running, opens the screens in Chrome). Run once: powershell -ExecutionPolicy Bypass -File scripts\make-shortcut.ps1
$root = Split-Path -Parent $PSScriptRoot
$lnk = Join-Path ([Environment]::GetFolderPath("Desktop")) "TestPoint.lnk"
$s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
$s.TargetPath = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$s.Arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$root\scripts\start-station.ps1`""
$s.WorkingDirectory = $root
$s.IconLocation = "$root\web\station\icon.ico,0"
$s.Description = "TestPoint fitness examination station: start and open"
$s.WindowStyle = 7          # minimised: no console window in the way
$s.Save()
Write-Output "Desktop icon: $lnk"

# TestPoint Network: offline mode (a fixed address on the camera network) or online mode (DHCP)
$net = Join-Path ([Environment]::GetFolderPath("Desktop")) "TestPoint Network.lnk"
$n = (New-Object -ComObject WScript.Shell).CreateShortcut($net)
$n.TargetPath = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$n.Arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$root\scripts\network.ps1`""
$n.WorkingDirectory = $root
$n.IconLocation = "$root\web\station\icon.ico,0"
$n.Description = "TestPoint Network: offline or online mode, cameras and app status"
$n.WindowStyle = 7
$n.Save()
Write-Output "Desktop icon: $net"
