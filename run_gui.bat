@echo off
chcp 65001 > nul
REM Windows native Python launcher (no WSL)
REM Usage: double-click this file, or run from cmd

cd /d "%~dp0"
echo ================================================
echo  ellipsofit GUI - Windows native Python
echo ================================================
echo  Starting... open browser at:
echo    http://localhost:8501
echo ================================================
echo.
python -m streamlit run gui/app.py --browser.gatherUsageStats false
pause
