@echo off
chcp 65001 > nul
REM Stop the WSL Streamlit process
echo Stopping ellipsofit...
wsl bash -c "pkill -f streamlit 2>/dev/null; true"
echo Done.
timeout /t 2 /nobreak > nul
