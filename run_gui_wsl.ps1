# PowerShell helper called by run_gui_wsl.bat
# 啟動 WSL streamlit 在最小化視窗

param(
    [int]$Port = 8501,
    [string]$Project = '/mnt/d/eliptometry'
)

# 1. 跳過 streamlit 首次啟動的 email 詢問
wsl bash -c "mkdir -p ~/.streamlit && [ -f ~/.streamlit/credentials.toml ] || printf '[general]\nemail = `"`"\n' > ~/.streamlit/credentials.toml"

# 2. 殺掉舊的 streamlit
wsl bash -c "pkill -f streamlit 2>/dev/null; true"

# 3. 啟動 streamlit 在最小化 WSL 視窗
$cmd = "cd $Project && ~/.local/bin/streamlit run gui/app.py --server.port $Port --browser.gatherUsageStats false"
Start-Process -FilePath wsl.exe `
    -ArgumentList 'bash', '-c', $cmd `
    -WindowStyle Minimized

# 4. 等 server 就緒
Write-Host "Waiting for Streamlit server (about 8 seconds)..."
Start-Sleep -Seconds 8

# 5. 開瀏覽器
Start-Process "http://localhost:$Port"

Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host "  GUI ready: http://localhost:$Port" -ForegroundColor Green
Write-Host ""
Write-Host "  To stop: double-click stop_gui_wsl.bat"
Write-Host "  Or close the minimized WSL window"
Write-Host "================================================" -ForegroundColor Green
