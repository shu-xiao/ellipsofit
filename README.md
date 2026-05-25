# Ellipsometry Fit Tool

擬合 ellipsometry 量測資料 (Ψ, Δ, T) 到光學模型，對應 WVASE 工作流程。

## 快速啟動

### Windows
雙擊 `run_gui.bat`，瀏覽器開 http://localhost:8501

### WSL2 / Linux / macOS
```bash
bash run_gui.sh
```

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
