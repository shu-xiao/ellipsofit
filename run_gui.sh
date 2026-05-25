#!/bin/bash
# ============================================================
#   ellipsofit — Streamlit GUI launcher (Linux / macOS / WSL)
# ============================================================
# Usage:
#   bash run_gui.sh                # 啟動（前景）
#   bash run_gui.sh -d              # daemon 背景模式（自動開瀏覽器）
#   bash run_gui.sh -p 8888         # 指定 port
#   bash run_gui.sh -d -p 8888      # 組合
#
# Mac 使用者：可雙擊 run_gui.command（同檔不同副檔名）
# 結束：bash stop_gui.sh 或 kill -f streamlit
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 參數解析
PORT=8501
DAEMON=0
while getopts "p:d" opt; do
    case $opt in
        p) PORT=$OPTARG ;;
        d) DAEMON=1 ;;
        *) echo "用法: bash $0 [-d] [-p PORT]"; exit 1 ;;
    esac
done

# OS 偵測（顯示 + 瀏覽器開啟用）
OS="$(uname -s)"
case "$OS" in
    Linux*)   PLATFORM="Linux"   ; OPENCMD="xdg-open" ;;
    Darwin*)  PLATFORM="macOS"   ; OPENCMD="open"     ;;
    MINGW*|MSYS*|CYGWIN*) PLATFORM="Windows" ; OPENCMD="start" ;;
    *)        PLATFORM="$OS"     ; OPENCMD="" ;;
esac

echo "================================================"
echo "   ellipsofit GUI ($PLATFORM)"
echo "   http://localhost:$PORT"
echo "================================================"

# ---- 找 Python ----
if command -v python3 > /dev/null; then
    PY=python3
elif command -v python > /dev/null; then
    PY=python
else
    echo "✗ 找不到 python。請先安裝 Python 3.10+"
    exit 1
fi

# ---- 找 streamlit ----
if command -v streamlit > /dev/null; then
    STREAMLIT=streamlit
elif [ -f "$HOME/.local/bin/streamlit" ]; then
    STREAMLIT="$HOME/.local/bin/streamlit"
else
    echo "✗ 找不到 streamlit"
    echo "  請執行: $PY -m pip install --user -r requirements.txt"
    exit 1
fi

# ---- 跳過 streamlit 首次啟動的 email 詢問 ----
mkdir -p "$HOME/.streamlit"
if [ ! -f "$HOME/.streamlit/credentials.toml" ]; then
    printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"
fi

# ---- 啟動 ----
ARGS="run gui/app.py --server.port $PORT --browser.gatherUsageStats false"

if [ "$DAEMON" -eq 1 ]; then
    # 背景模式
    echo "Starting in background (PID written to .streamlit.pid)..."
    pkill -f "streamlit run.*gui/app.py" 2>/dev/null || true
    nohup "$STREAMLIT" $ARGS > /tmp/ellipsofit.log 2>&1 &
    echo $! > "$SCRIPT_DIR/.streamlit.pid"

    # 等 server 起來
    sleep 5

    # 開瀏覽器
    if [ -n "$OPENCMD" ]; then
        $OPENCMD "http://localhost:$PORT" 2>/dev/null || true
    fi

    echo ""
    echo "================================================"
    echo "  GUI 已啟動 (PID $(cat .streamlit.pid))"
    echo "  http://localhost:$PORT"
    echo "  Log: tail -f /tmp/ellipsofit.log"
    echo "  結束: bash stop_gui.sh"
    echo "================================================"
else
    # 前景模式（Ctrl-C 結束）
    echo "Starting (Ctrl-C to stop)..."
    exec "$STREAMLIT" $ARGS
fi
