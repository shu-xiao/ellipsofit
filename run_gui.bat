@echo off
REM Windows 一鍵啟動 Streamlit GUI（Windows native Python）
REM 用法：雙擊此檔，或在 cmd 執行 run_gui.bat

cd /d "%~dp0"
echo ================================================
echo  Ellipsometry Fit Tool - Streamlit GUI
echo ================================================
echo  啟動中... 完成後請開瀏覽器：
echo    http://localhost:8501
echo ================================================
echo.
python -m streamlit run gui/app.py --browser.gatherUsageStats false
pause
