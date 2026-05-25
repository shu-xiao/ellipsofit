@echo off
chcp 65001 > nul
REM ============================================================
REM   ellipsofit  -  Double-click to launch Streamlit via WSL
REM ============================================================
REM   Calls run_gui_wsl.ps1 (PowerShell handles WSL + quoting
REM   more reliably than cmd nested quotes)
REM
REM   To stop:
REM     - Double-click stop_gui_wsl.bat
REM     - Or close the minimized "ellipsofit" WSL window
REM ============================================================

echo ================================================
echo   ellipsofit GUI - WSL Backend
echo ================================================
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0run_gui_wsl.ps1"

timeout /t 3 /nobreak > nul
exit
