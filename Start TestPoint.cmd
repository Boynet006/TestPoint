@echo off
rem Double-click to start TestPoint (starts Docker Desktop if needed and opens the station UI).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\testpoint.ps1" start %*
set rc=%errorlevel%
if not %rc%==0 pause
if %rc%==0 timeout /t 5 >nul 2>nul
exit /b %rc%
