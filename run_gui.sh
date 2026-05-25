#!/bin/bash
# 一鍵啟動 Streamlit GUI
# 用法：bash run_gui.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT=${PORT:-8501}

echo "================================================"
echo " Ellipsometry Fit Tool — Streamlit GUI"
echo "================================================"
echo " 啟動中... 完成後請開瀏覽器："
echo "   http://localhost:$PORT"
echo "================================================"

# 嘗試找 streamlit binary
if command -v streamlit > /dev/null; then
    STREAMLIT=streamlit
elif [ -f "$HOME/.local/bin/streamlit" ]; then
    STREAMLIT="$HOME/.local/bin/streamlit"
else
    echo "✗ 找不到 streamlit。請先執行：pip3 install --user streamlit"
    exit 1
fi

exec "$STREAMLIT" run gui/app.py --server.port "$PORT" --browser.gatherUsageStats false
