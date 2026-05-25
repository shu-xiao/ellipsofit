#!/bin/bash
# ============================================================
#   ellipsofit  —  macOS double-clickable launcher
# ============================================================
# Mac Finder 雙擊此檔即啟動（會開 Terminal 視窗）
# 啟動後自動開瀏覽器，Ctrl-C 結束
#
# 第一次用前需給執行權限：
#   chmod +x run_gui.command
# ============================================================

cd "$(dirname "$0")"

# 用 daemon 模式啟動 + 自動開瀏覽器
exec bash run_gui.sh -d

# 視窗保持開著（等 Enter 才關）
echo ""
echo "Press Enter to close this window..."
read
