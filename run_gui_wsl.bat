@echo off
chcp 65001 > nul
REM ============================================================
REM   ellipsofit  -  Double-click to launch Streamlit via WSL
REM ============================================================
REM   All-in-one bat: inlines PowerShell command for WSL launch
REM   (No separate .ps1 file needed)
REM
REM   To stop: double-click stop_gui_wsl.bat
REM ============================================================

set PORT=8501

echo ================================================
echo   ellipsofit GUI - WSL Backend
echo ================================================
echo   Starting WSL streamlit in background...
echo   Browser will open http://localhost:%PORT%
echo ================================================
echo.

REM Skip streamlit first-run email prompt
wsl bash -c "mkdir -p ~/.streamlit && [ -f ~/.streamlit/credentials.toml ] || printf '[general]\nemail = \"\"\n' > ~/.streamlit/credentials.toml"

REM Kill any existing streamlit
wsl bash -c "pkill -f streamlit 2>/dev/null; true"

REM Launch WSL streamlit in minimized window via PowerShell (reliable quoting)
powershell -Command "Start-Process -FilePath wsl.exe -ArgumentList 'bash','-c','cd /mnt/d/eliptometry && ~/.local/bin/streamlit run gui/app.py --server.port %PORT% --browser.gatherUsageStats false' -WindowStyle Minimized"

REM Wait for server, then open browser
echo Waiting 8 seconds for server to start...
timeout /t 8 /nobreak > nul
start http://localhost:%PORT%

echo.
echo ================================================
echo   GUI ready: http://localhost:%PORT%
echo.
echo   To stop: double-click stop_gui_wsl.bat
echo ================================================
echo.

timeout /t 3 /nobreak > nul
exit
