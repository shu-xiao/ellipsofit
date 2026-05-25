# ellipsofit

[![Tests](https://github.com/shu-xiao/ellipsofit/actions/workflows/test.yml/badge.svg)](https://github.com/shu-xiao/ellipsofit/actions/workflows/test.yml)

擬合 ellipsometry 量測資料 (Ψ, Δ, T) 到光學模型，對應 WVASE 工作流程。

## 快速啟動

### Windows（推薦：透過 WSL 後台）
雙擊 **`run_gui_wsl.bat`**
- 在 WSL 背景跑 streamlit（最小化視窗）
- 自動開瀏覽器 http://localhost:8501
- 結束：執行 `stop_gui_wsl.bat` 或關掉最小化視窗

### Windows（用 Windows 原生 Python）
雙擊 `run_gui.bat`，瀏覽器開 http://localhost:8501

### macOS
首次需要給執行權限：
```bash
chmod +x run_gui.command run_gui.sh stop_gui.sh
```
之後 Finder **雙擊 `run_gui.command`**（會開 Terminal）— 自動啟動 + 開瀏覽器。
結束：執行 `bash stop_gui.sh` 或關掉 Terminal 視窗。

### Linux / WSL2（終端機）
```bash
bash run_gui.sh           # 前景模式，Ctrl-C 結束
bash run_gui.sh -d        # daemon 背景模式 + 自動開瀏覽器
bash run_gui.sh -p 8888   # 指定 port
bash stop_gui.sh          # 停止 daemon
```

Linux 桌面環境另可把 `ellipsofit.desktop` 拖到 `~/.local/share/applications/` 變成應用程式選單項目。

## 功能

- **GUI 介面 (Streamlit)** — 拖檔上傳、下拉選單、即時 plotly 圖
- **CLI 介面** — `python -m ellipsometry.core.fitter` 批次處理
- 14 個內建材料 (Si, SiO2, MgO, Al2O3, TiO2, Au×3, Al)
- 10 種色散模型 (Cauchy, Drude-Lorentz, GenOsc, Tauc-Lorentz, ...)
- WVASE-style MSE + Levenberg-Marquardt 擬合
- 三種 fit 模式 (both / e2_only / e1_only)
- 透明基板背面非相干處理（對應 WVASE backside correction）

## 安裝

```bash
pip3 install -r requirements.txt
python3 materials/download_materials.py  # 下載內建材料 (已下載則跳過)
```

## 目錄結構

```
eliptometry/
├── gui/app.py            ← Streamlit GUI 入口
├── ellipsometry/core/    ← 核心模組
│   ├── io.py             ← 讀 WVASE .dat
│   ├── dispersion.py     ← 色散模型
│   ├── tmm_calc.py       ← TMM 計算
│   ├── fitter.py         ← lmfit 擬合引擎
│   └── units.py          ← 單位換算
├── materials/            ← 內建材料 nk csv
├── data/                 ← 樣品資料（含 3 個 fake 範例）
├── tests/                ← 測試
├── config.example.yaml   ← 完整 config 範例
└── run_gui.sh / .bat     ← GUI 啟動腳本
```

## 工作流程

1. **啟動 GUI** → 上傳 R.dat (與 T.dat 可選) 或點「用範例」
2. **選基板** → Si / MgO / Sapphire / SiO2 / SiO2 thermal
3. **選薄膜**：
   - **built-in**: 內建材料庫直接選（適合已知材料只 fit 厚度）
   - **GenOsc**: 自己組振盪器（含 Lorentz/Gaussian/Drude/Tauc-Lorentz 任意混合）
4. **設擬合參數** → 點「🚀 開始擬合」
5. **看結果** → Ψ/Δ/T 對比圖、n,k 色散、最佳化參數表
6. **下載** → config.yaml / nk csv / 擬合曲線 csv
