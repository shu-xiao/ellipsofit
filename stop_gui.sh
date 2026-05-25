#!/bin/bash
# 停掉 Streamlit GUI
echo "Stopping ellipsofit..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="$SCRIPT_DIR/.streamlit.pid"

# 優先用 PID file，找不到再 pkill
if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill "$PID" 2>/dev/null; then
        echo "  Killed PID $PID"
    fi
    rm -f "$PIDFILE"
fi

# 保險再 pkill 殘留
pkill -f "streamlit run.*gui/app.py" 2>/dev/null && echo "  pkill cleaned up" || true
echo "Done."
