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
