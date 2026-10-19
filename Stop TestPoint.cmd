@echo off
rem Double-click to stop the TestPoint services.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\testpoint.ps1" stop
set rc=%errorlevel%
if not %rc%==0 pause
if %rc%==0 timeout /t 3 >nul 2>nul
exit /b %rc%
