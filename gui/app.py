"""
Ellipsometry Fit Tool — Streamlit GUI

啟動方式：
    streamlit run gui/app.py

特色：
    - 拖曳上傳 .dat 檔
    - 下拉選單選基板、薄膜材料
    - 互動式設參數初始值與上下界
    - 即時擬合 + plotly 互動圖
    - 一鍵下載 config 與結果
"""
import os
import sys
import io
import json
import tempfile
import base64
from pathlib import Path

# 把專案根目錄加到 path（讓 ellipsometry 模組可被 import）
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ellipsometry.core.io import read_reflection, read_transmission
from ellipsometry.core.dispersion import MaterialLibrary, OSC_PARAMS, osc_param_labels
from ellipsometry.core.fitter import Fitter


# =============================================================================
# 頁面設定
# =============================================================================
st.set_page_config(
    page_title='Ellipsometry Fit Tool',
    page_icon='🔬',
    layout='wide',
)

st.title('🔬 Ellipsometry Fit Tool')
st.caption('擬合 ψ, Δ, T 量測資料到光學模型 — 對應 WVASE 工作流程')


# =============================================================================
# 暫存區（session state）初始化
# =============================================================================
def default_film(name='thin_film', material='Au'):
    """產生一個預設薄膜 dict"""
    return {
        'name': name,
        'material_source': 'built-in',
        'material_builtin': material,
        'thickness_nm': 50.0,
        'fit_thickness': True,
        'thickness_bounds': [10.0, 200.0],
        'coherent': True,
        'gen_osc': {
            'e1_offset': 1.0,
            'egap': 0.0,
            'uv_pole': {'position': 20.0, 'magnitude': 100.0, 'use': False},
            'ir_pole': {'position': 0.001, 'magnitude': 0.0, 'use': False},
            'oscillators': [
                {'type': 'lorentz', 'amp': 2.5, 'en': 2.7, 'br': 0.3,
                 'active': True, 'fit': True,
                 'bounds': {'amp': [0.0, 10.0], 'en': [0.5, 6.5], 'br': [0.05, 2.0]}},
                {'type': 'lorentz', 'amp': 1.8, 'en': 4.5, 'br': 0.5,
                 'active': True, 'fit': True,
                 'bounds': {'amp': [0.0, 10.0], 'en': [0.5, 6.5], 'br': [0.05, 2.0]}},
            ],
        },
    }


if 'films' not in st.session_state:
    st.session_state.films = [default_film()]

for key, default in [
    ('fit_result', None), ('data', None), ('T_data', None),
    ('abort_fit', False), ('fit_running', False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# =============================================================================
# 內建材料清單
# =============================================================================
AVAILABLE_MATERIALS = MaterialLibrary.list_available()


# =============================================================================
# Sidebar：設定區
# =============================================================================
# =============================================================================
# 預設 Preset 定義
# =============================================================================
PRESETS = {
    'best': {
        'label': '🌟 Best（我們推薦）',
        'description': '最現代的設定：sin/cos Δ 殘差、PCHIP 插值、可選 robust loss',
        'max_iter': 35,
        'delta_residual_mode': 'sin_cos',
        'loss': 'linear',
        'method': 'leastsq',
        'two_stage': False,
        'sigma_psi': 0.5,
        'sigma_delta': 1.0,
        'sigma_T': 0.01,
        'material_interpolation': 'pchip',
    },
    'wvase': {
        'label': '📐 WVASE 相容（接近 WVASE32 預設）',
        'description': '對應 WVASE32 行為：wrap Δ、linear loss、線性插值、LM',
        'max_iter': 35,
        'delta_residual_mode': 'wrap',
        'loss': 'linear',
        'method': 'leastsq',
        'two_stage': False,
        'sigma_psi': 0.5,
        'sigma_delta': 1.0,
        'sigma_T': 0.01,
        'material_interpolation': 'linear',
    },
    'global_search': {
        'label': '🔍 全域搜索（初始值不確定時）',
        'description': '兩段式 DE → LM。慢但穩，適合「只知道大概值」',
        'max_iter': 35,
        'delta_residual_mode': 'sin_cos',
        'loss': 'soft_l1',
        'method': 'leastsq',
        'two_stage': True,
        'sigma_psi': 0.5,
        'sigma_delta': 1.0,
        'sigma_T': 0.01,
        'material_interpolation': 'pchip',
    },
    'robust': {
        'label': '🛡️ Robust（雜訊/outlier 多時）',
        'description': 'huber loss 抵抗壞點，sin/cos Δ',
        'max_iter': 50,
        'delta_residual_mode': 'sin_cos',
        'loss': 'huber',
        'method': 'least_squares',
        'two_stage': False,
        'sigma_psi': 0.5,
        'sigma_delta': 1.0,
        'sigma_T': 0.01,
        'material_interpolation': 'pchip',
    },
}

# 初始化 preset 選擇
if 'preset_name' not in st.session_state:
    st.session_state.preset_name = 'best'
if 'preset_overrides' not in st.session_state:
    # 使用者手動覆寫過的欄位（preset 切換時不蓋掉）
    st.session_state.preset_overrides = set()


def current_preset() -> dict:
    """目前選用的 preset（套用 overrides 後）"""
    base = dict(PRESETS[st.session_state.preset_name])
    # overrides 已存在 session_state（GUI 元件自動同步），這裡只負責預設值
    return base


with st.sidebar:
    st.header('⚙️ 設定')

    # ----- 0. Preset 切換 -----
    with st.expander('🎛️ 0. 預設組合 (Preset)', expanded=True):
        new_preset = st.radio(
            '選一個 preset 一鍵套用設定：',
            options=list(PRESETS.keys()),
            format_func=lambda k: PRESETS[k]['label'],
            index=list(PRESETS.keys()).index(st.session_state.preset_name),
            help='下方欄位仍可手動覆寫。切 preset 不會蓋掉你手動改過的值。',
        )
        if new_preset != st.session_state.preset_name:
            st.session_state.preset_name = new_preset
            # 切 preset 時清除手動覆寫
            st.session_state.preset_overrides = set()
            st.rerun()
        st.caption(PRESETS[st.session_state.preset_name]['description'])

    preset = current_preset()

    # ----- 1. 上傳資料 -----
    with st.expander('📂 1. 上傳資料', expanded=True):
        st.markdown('支援 WVASE 長/寬格式')

        # 提供範例資料按鈕
        col_a, col_b = st.columns([1, 1])
        with col_a:
            use_demo = st.button('🎯 用範例', use_container_width=True)
        with col_b:
            clear_demo = st.button('🗑 清除', use_container_width=True)

        if use_demo:
            st.session_state.data = read_reflection(
                str(ROOT / 'data' / 'sample1_Au50_sapphire' / 'R_long.dat'))
            st.session_state.T_data = read_transmission(
                str(ROOT / 'data' / 'sample1_Au50_sapphire' / 'T.dat'))
            st.success('已載入：50 nm Au / sapphire 範例')

        if clear_demo:
            st.session_state.data = None
            st.session_state.T_data = None
            st.session_state.fit_result = None

        # 上傳檔案
        R_file = st.file_uploader('反射 R.dat（含 ψ, Δ 多角度）', type=['dat', 'txt', 'csv'])
        T_file = st.file_uploader('穿透 T.dat（normal incidence，可選）', type=['dat', 'txt', 'csv'])

        if R_file:
            # 寫到暫存檔讓 io 讀
            with tempfile.NamedTemporaryFile(suffix='.dat', delete=False, mode='wb') as f:
                f.write(R_file.read())
                tmp_path = f.name
            try:
                st.session_state.data = read_reflection(tmp_path)
                st.success(f'✓ 讀取 {R_file.name}')
            except Exception as e:
                st.error(f'讀取失敗：{e}')
            os.unlink(tmp_path)

        if T_file:
            with tempfile.NamedTemporaryFile(suffix='.dat', delete=False, mode='wb') as f:
                f.write(T_file.read())
                tmp_path = f.name
            try:
                st.session_state.T_data = read_transmission(tmp_path)
                st.success(f'✓ 讀取 {T_file.name}')
            except Exception as e:
                st.error(f'讀取失敗：{e}')
            os.unlink(tmp_path)

        # 顯示資料摘要
        if st.session_state.data is not None:
            d = st.session_state.data
            st.info(f'**反射:** {len(d.wavelength)} 點，AOI={d.angles}，'
                    f'{d.wavelength.min():.0f}-{d.wavelength.max():.0f} nm')
        if st.session_state.T_data is not None:
            t = st.session_state.T_data
            st.info(f'**穿透:** {len(t.wavelength)} 點，'
                    f'{t.wavelength.min():.0f}-{t.wavelength.max():.0f} nm')

    # ----- 2. 波長範圍 -----
    with st.expander('📏 2. 波長範圍 (nm)'):
        if st.session_state.data is not None:
            d = st.session_state.data
            wmin, wmax = float(d.wavelength.min()), float(d.wavelength.max())
        else:
            wmin, wmax = 300.0, 2000.0
        wl_range = st.slider(
            '擬合範圍', min_value=wmin, max_value=wmax,
            value=(wmin, wmax), step=10.0,
        )

    # ----- 3. 基板 -----
    with st.expander('🪨 3. 基板', expanded=True):
        substrate_choice = st.selectbox(
            '基板類型',
            options=['Si', 'MgO', 'Al2O3 (sapphire)', 'SiO2 (bulk)', 'SiO2 / Si (thermal oxide)'],
            help='選擇實驗用基板。SiO2/Si 是兩層結構。',
        )

        if substrate_choice == 'SiO2 / Si (thermal oxide)':
            sio2_d = st.number_input('熱氧化 SiO2 厚度 (nm)', value=285.0, min_value=1.0, max_value=10000.0)
            fit_sio2 = st.checkbox('擬合 SiO2 厚度', value=False)

        substrate_d_mm = st.number_input(
            '基板厚度 (mm)', value=1.0,
            min_value=0.001, max_value=100.0, step=0.1,
            format='%.3f',
            help='典型 Si wafer ≈ 0.5 mm，玻璃 1 mm。透明基板會走非相干疊加。',
        )
        substrate_d = substrate_d_mm * 1_000_000.0   # 轉成 nm 給內部使用

    # ----- 4. 薄膜（多層支援）-----
    with st.expander(f'🎬 4. 薄膜（共 {len(st.session_state.films)} 層）', expanded=True):
        # 加減層按鈕
        c_add, c_del = st.columns(2)
        with c_add:
            if st.button('➕ 加一層膜', use_container_width=True):
                st.session_state.films.append(
                    default_film(name=f'film_{len(st.session_state.films)+1}', material='SiO2')
                )
                st.rerun()
        with c_del:
            if st.button('➖ 刪最後層', use_container_width=True,
                         disabled=len(st.session_state.films) <= 1):
                st.session_state.films.pop()
                st.rerun()

        # 每層獨立設定（用 tab 區分多層）
        if len(st.session_state.films) > 1:
            tabs = st.tabs([f'#{i+1} {f["name"]}' for i, f in enumerate(st.session_state.films)])
        else:
            tabs = [st.container()]

        for fi, (tab, film) in enumerate(zip(tabs, st.session_state.films)):
            with tab:
                kp = f'film{fi}_'    # key prefix
                film['name'] = st.text_input('膜層名稱', value=film['name'], key=kp+'name')

                film['material_source'] = st.radio(
                    '材料來源',
                    options=['built-in', 'GenOsc (擬合模型)'],
                    index=0 if film['material_source'] == 'built-in' else 1,
                    horizontal=True, key=kp+'msrc',
                )

                if film['material_source'] == 'built-in':
                    film['material_builtin'] = st.selectbox(
                        '材料', options=AVAILABLE_MATERIALS,
                        index=AVAILABLE_MATERIALS.index(film['material_builtin'])
                              if film['material_builtin'] in AVAILABLE_MATERIALS else 0,
                        key=kp+'mbi',
                    )

                cc1, cc2 = st.columns([2, 1])
                with cc1:
                    film['thickness_nm'] = st.number_input(
                        '厚度初始值 (nm)', value=film['thickness_nm'],
                        min_value=0.1, max_value=10000.0, key=kp+'d',
                    )
                with cc2:
                    film['fit_thickness'] = st.checkbox(
                        '擬合厚度', value=film['fit_thickness'], key=kp+'fitd')
                if film['fit_thickness']:
                    c1, c2 = st.columns(2)
                    with c1:
                        film['thickness_bounds'][0] = st.number_input(
                            '下限', value=film['thickness_bounds'][0], key=kp+'dlo')
                    with c2:
                        film['thickness_bounds'][1] = st.number_input(
                            '上限', value=film['thickness_bounds'][1], key=kp+'dhi')

                # ----- GenOsc 振盪器（只在選 GenOsc 時顯示）-----
                if film['material_source'] == 'GenOsc (擬合模型)':
                    st.markdown('##### 🌀 GenOsc 振盪器')
                    go_cfg = film['gen_osc']

                    cc1, cc2, cc3, cc4 = st.columns([2, 1, 1, 2])
                    with cc1:
                        go_cfg['e1_offset'] = st.number_input(
                            'ε∞ (e1 offset)', value=go_cfg['e1_offset'], key=kp+'e1o')
                    # e1_offset bounds（DE 必需）
                    go_cfg.setdefault('e1_offset_bounds', [0.5, 10.0])
                    with cc2:
                        go_cfg['e1_offset_bounds'][0] = st.number_input(
                            'ε∞ 下限', value=float(go_cfg['e1_offset_bounds'][0]),
                            key=kp+'e1olo')
                    with cc3:
                        go_cfg['e1_offset_bounds'][1] = st.number_input(
                            'ε∞ 上限', value=float(go_cfg['e1_offset_bounds'][1]),
                            key=kp+'e1ohi')
                    with cc4:
                        go_cfg['egap'] = st.number_input(
                            'Egap (eV)', value=go_cfg['egap'], key=kp+'egap')

                    with st.expander('UV/IR Poles'):
                        c1, c2 = st.columns(2)
                        with c1:
                            go_cfg['uv_pole']['use'] = st.checkbox(
                                '啟用 UV pole', value=go_cfg['uv_pole']['use'], key=kp+'uvu')
                            if go_cfg['uv_pole']['use']:
                                go_cfg['uv_pole']['position'] = st.number_input(
                                    'UV pos (eV)', value=go_cfg['uv_pole']['position'],
                                    min_value=5.0, max_value=50.0, key=kp+'uvp')
                                go_cfg['uv_pole']['magnitude'] = st.number_input(
                                    'UV mag', value=go_cfg['uv_pole']['magnitude'], key=kp+'uvm')
                        with c2:
                            go_cfg['ir_pole']['use'] = st.checkbox(
                                '啟用 IR pole', value=go_cfg['ir_pole']['use'], key=kp+'iru')
                            if go_cfg['ir_pole']['use']:
                                go_cfg['ir_pole']['position'] = st.number_input(
                                    'IR pos (eV)', value=go_cfg['ir_pole']['position'], key=kp+'irp')
                                go_cfg['ir_pole']['magnitude'] = st.number_input(
                                    'IR mag', value=go_cfg['ir_pole']['magnitude'], key=kp+'irm')

                    # 振盪器數量
                    cc1, cc2, cc3 = st.columns([1, 1, 2])
                    with cc1:
                        if st.button('➕ 加 Osc', key=kp+'addosc'):
                            go_cfg['oscillators'].append({
                                'type': 'lorentz', 'amp': 1.0, 'en': 3.0, 'br': 0.3,
                                'active': True, 'fit': True,
                                'bounds': {'amp': [0.0, 10.0], 'en': [0.5, 6.5], 'br': [0.05, 2.0]},
                            })
                            st.rerun()
                    with cc2:
                        if st.button('➖ 刪 Osc', key=kp+'delosc',
                                     disabled=len(go_cfg['oscillators']) <= 1):
                            go_cfg['oscillators'].pop()
                            st.rerun()
                    with cc3:
                        st.caption(f'共 {len(go_cfg["oscillators"])} 個振盪器')

                    for i, osc in enumerate(go_cfg['oscillators']):
                        osc.setdefault('bounds', {'amp': [0.0, 10.0],
                                                  'en': [0.5, 6.5],
                                                  'br': [0.05, 2.0]})
                        with st.container(border=True):
                            cc1, cc2, cc3 = st.columns([2, 1, 1])
                            with cc1:
                                osc['type'] = st.selectbox(
                                    f'Osc #{i+1} 類型',
                                    options=['lorentz', 'gaussian', 'drude', 'tauc_lorentz', 'harmonic'],
                                    index=['lorentz', 'gaussian', 'drude', 'tauc_lorentz', 'harmonic'].index(osc['type']),
                                    key=f'{kp}otype_{i}',
                                )
                            with cc2:
                                osc['active'] = st.checkbox('啟用', value=osc['active'],
                                                            key=f'{kp}oact_{i}')
                            with cc3:
                                osc['fit'] = st.checkbox('🔧 擬合', value=osc['fit'],
                                                         key=f'{kp}ofit_{i}')
                            # 依振盪器類型顯示對應的參數欄
                            needed = OSC_PARAMS[osc['type']]
                            labels = osc_param_labels(osc['type'])
                            # 確保 bounds dict 有需要的 key
                            default_bounds = {
                                'amp': [0.0, 100.0] if osc['type'] == 'drude' else [0.0, 10.0],
                                'en':  [0.5, 6.5],
                                'br':  [0.01, 5.0] if osc['type'] == 'drude' else [0.05, 2.0],
                                'Eg':  [0.0, 5.0],
                            }
                            # 表頭
                            hc1, hc2, hc3 = st.columns([2, 1, 1])
                            hc1.caption('參數 (初始值)')
                            hc2.caption('min')
                            hc3.caption('max')
                            for pname in needed:
                                osc.setdefault(pname, 0.0)
                                if pname not in osc.get('bounds', {}):
                                    osc.setdefault('bounds', {})
                                    osc['bounds'][pname] = list(default_bounds[pname])

                                disp_name, unit, meaning = labels.get(pname,
                                    (pname.title(), '', ''))
                                label_str = f'{disp_name} ({unit})' if unit else disp_name
                                pc1, pc2, pc3 = st.columns([2, 1, 1])
                                with pc1:
                                    osc[pname] = st.number_input(
                                        label_str, value=float(osc[pname]),
                                        key=f'{kp}o{pname}_{i}', help=meaning,
                                    )
                                with pc2:
                                    osc['bounds'][pname][0] = st.number_input(
                                        '下限', value=float(osc['bounds'][pname][0]),
                                        key=f'{kp}o{pname}lo_{i}', label_visibility='collapsed')
                                with pc3:
                                    osc['bounds'][pname][1] = st.number_input(
                                        '上限', value=float(osc['bounds'][pname][1]),
                                        key=f'{kp}o{pname}hi_{i}', label_visibility='collapsed')

    # ----- 6. 擬合設定 -----
    with st.expander('⚙️ 6. 擬合設定（preset 已套用，可手動覆寫）', expanded=False):
        fit_target = st.radio(
            '擬合目標',
            options=['both', 'e2_only', 'e1_only'],
            help='both: 直接擬合 Ψ/Δ/T (WVASE 預設)\n'
                 'e2_only: 先反推 ε，再擬合 ε2 (WVASE 進階用)\n'
                 'e1_only: 同上但 fit ε1',
        )

        # 用 preset 預設值，使用者可改
        method_options = ['leastsq', 'least_squares', 'differential_evolution']
        fit_method = st.selectbox(
            '演算法', options=method_options,
            index=method_options.index(preset['method']),
            help='leastsq: MINPACK LM (WVASE 一致，最快)\n'
                 'least_squares: 支援 robust loss\n'
                 'differential_evolution: 全域搜索（慢）',
            key=f'_method_{st.session_state.preset_name}',
        )

        max_iter = st.number_input(
            'Max iterations',
            value=preset['max_iter'], min_value=5, max_value=1000,
            help='與 WVASE 對齊：LM 迭代次數，預設 35。\n'
                 '內部自動換算為 max_nfev = max_iter × (參數數+1)',
            key=f'_max_iter_{st.session_state.preset_name}',
        )

        st.markdown('**Δ 殘差模式**')
        delta_opts = ['sin_cos', 'wrap', 'raw']
        delta_mode = st.radio(
            '',
            options=delta_opts,
            index=delta_opts.index(preset['delta_residual_mode']),
            help='sin_cos (推薦): 把 Δ 拆 sin/cos 兩殘差，避免 ±180° 跳變\n'
                 'wrap: 標準寫法，殘差自動 mod 360（WVASE 行為）\n'
                 'raw: 不處理（debug 用）',
            horizontal=True,
            label_visibility='collapsed',
            key=f'_delta_{st.session_state.preset_name}',
        )

        st.markdown('**Robust loss (對抗 outlier)**')
        loss_opts = ['linear', 'soft_l1', 'huber', 'cauchy', 'arctan']
        loss_fn = st.selectbox(
            '',
            options=loss_opts,
            index=loss_opts.index(preset['loss']),
            help='linear: 標準 χ²（壞點主導 fit）\n'
                 'soft_l1: 平滑 L1，小殘差 ~ r²，大殘差 ~ |r|\n'
                 'huber: 顯式 outlier 門檻\n'
                 'cauchy: 對 outlier 最不敏感（壞點完全忽略）\n'
                 '\n選了非 linear 會自動切到 least_squares method',
            label_visibility='collapsed',
            key=f'_loss_{st.session_state.preset_name}',
        )

        two_stage = st.checkbox(
            '🔍 兩段式：DE 全域搜索 → LM 精調',
            value=preset['two_stage'],
            help='不確定初始值時開啟（會比較慢，每個 residual call ~2 秒，'
                 'DE 約 300 次 = 10 分鐘）。\n'
                 '對「只知道大概厚度」或「多振盪器 GenOsc」特別有用。',
            key=f'_2stage_{st.session_state.preset_name}',
        )

        st.markdown('**測量誤差 σ（WVASE-style weighting）**')
        c1, c2, c3 = st.columns(3)
        with c1:
            sigma_psi = st.number_input('σ_Ψ (deg)', value=preset['sigma_psi'],
                                        min_value=0.001,
                                        key=f'_spsi_{st.session_state.preset_name}')
        with c2:
            sigma_delta = st.number_input('σ_Δ (deg)', value=preset['sigma_delta'],
                                          min_value=0.001,
                                          key=f'_sdel_{st.session_state.preset_name}')
        with c3:
            sigma_T = st.number_input('σ_T', value=preset['sigma_T'],
                                      min_value=0.0001,
                                      key=f'_sT_{st.session_state.preset_name}')

    st.markdown('---')
    c_run, c_abort = st.columns([3, 1])
    with c_run:
        run_fit = st.button('🚀 開始擬合', type='primary', use_container_width=True,
                            disabled=(st.session_state.data is None
                                       or st.session_state.fit_running))
    with c_abort:
        if st.button('⛔ 中斷', use_container_width=True,
                     disabled=not st.session_state.fit_running):
            st.session_state.abort_fit = True
    if st.session_state.data is None:
        st.caption('👆 請先上傳資料或點「用範例」')

    # ----- 7. Save / Load Session -----
    with st.expander('💾 儲存 / 載入 Session（YAML）'):
        st.caption('把目前所有設定（layers + fit options + preset）打包成 YAML，'
                   '下次上傳即可還原。資料檔需另外上傳。')
        # Save
        if st.session_state.fit_result or st.session_state.films:
            session_dump = {
                'preset_name': st.session_state.preset_name,
                'films': st.session_state.films,
                'substrate_choice': substrate_choice,
                'substrate_thickness_nm': substrate_d,
                'sio2_thermal_thickness': sio2_d if substrate_choice == 'SiO2 / Si (thermal oxide)' else None,
                'fit_sio2': fit_sio2 if substrate_choice == 'SiO2 / Si (thermal oxide)' else False,
                'wavelength_range': list(wl_range),
                'fit_options': {
                    'target': fit_target, 'method': fit_method, 'max_iter': max_iter,
                    'delta_residual_mode': delta_mode, 'loss': loss_fn,
                    'two_stage': two_stage,
                    'sigma_psi': sigma_psi, 'sigma_delta': sigma_delta, 'sigma_T': sigma_T,
                },
            }
            st.download_button(
                '📥 下載 session (yaml)',
                data=yaml.dump(session_dump, allow_unicode=True, default_flow_style=False),
                file_name='ellipsometry_session.yaml', mime='text/yaml',
                use_container_width=True,
            )
        # Load
        sess_upload = st.file_uploader('📤 上傳 session', type=['yaml', 'yml'],
                                       key='_sess_upload')
        if sess_upload:
            try:
                loaded = yaml.safe_load(sess_upload.read())
                st.session_state.preset_name = loaded.get('preset_name', 'best')
                st.session_state.films = loaded.get('films', [default_film()])
                st.success('✓ Session 載入成功，重新整理頁面')
                st.rerun()
            except Exception as e:
                st.error(f'載入失敗：{e}')


# =============================================================================
# 主畫面：結果區
# =============================================================================

# ----- 函式：把 sidebar 設定轉成 config dict -----
def build_config():
    layers = [{'name': 'ambient', 'material': 'air', 'thickness': 'infinite'}]

    # 薄膜（多層）
    for film in st.session_state.films:
        if film['material_source'] == 'built-in':
            mat = film['material_builtin']
        else:
            go_cfg = film['gen_osc']
            # 依振盪器類型只收集需要的參數做 fit
            fit_paths = ['e1_offset']
            bounds = {'e1_offset': list(go_cfg.get('e1_offset_bounds', [0.5, 10.0]))}
            for i, osc in enumerate(go_cfg['oscillators']):
                needed_params = OSC_PARAMS[osc['type']]
                if osc.get('fit', True):
                    for pname in needed_params:
                        fit_paths.append(f'oscillators[{i}].{pname}')
                        if pname in osc.get('bounds', {}):
                            bounds[f'oscillators[{i}].{pname}'] = list(osc['bounds'][pname])

            mat = {
                'model': 'gen_osc',
                'layer': {
                    'e1_offset': go_cfg['e1_offset'],
                    'egap': go_cfg['egap'],
                    'poles': {},
                },
                # 振盪器只塞需要的欄位（避免 Drude 的 en 被當作有意義的初始值）
                'oscillators': [
                    {**{'type': o['type'], 'active': o.get('active', True)},
                     **{p: o.get(p, 0.0) for p in OSC_PARAMS[o['type']]}}
                    for o in go_cfg['oscillators']
                ],
                'params': {
                    'e1_offset': go_cfg['e1_offset'],
                    'oscillators': [
                        {p: o.get(p, 0.0) for p in OSC_PARAMS[o['type']]}
                        for o in go_cfg['oscillators']
                    ],
                },
                'fit': fit_paths,
                'bounds': bounds,
            }
            if go_cfg['uv_pole']['use']:
                mat['layer']['poles']['uv'] = {
                    'position': go_cfg['uv_pole']['position'],
                    'magnitude': go_cfg['uv_pole']['magnitude'],
                }
            if go_cfg['ir_pole']['use']:
                mat['layer']['poles']['ir'] = {
                    'position': go_cfg['ir_pole']['position'],
                    'magnitude': go_cfg['ir_pole']['magnitude'],
                }

        layer_dict = {
            'name': film['name'],
            'material': mat,
            'thickness': film['thickness_nm'],
            'coherent': film['coherent'],
            'fit_thickness': film['fit_thickness'],
            'thickness_bounds': film['thickness_bounds'],
        }
        layers.append(layer_dict)

    # 基板
    if substrate_choice == 'SiO2 / Si (thermal oxide)':
        layers.append({
            'name': 'SiO2_thermal', 'material': 'SiO2',
            'thickness': sio2_d, 'coherent': True,
            'fit_thickness': fit_sio2,
            'thickness_bounds': [50, 1000],
        })
        layers.append({
            'name': 'Si_substrate', 'material': 'Si',
            'thickness': substrate_d, 'coherent': False, 'fit_thickness': False,
        })
    else:
        sub_mat = {
            'Si': 'Si', 'MgO': 'MgO', 'Al2O3 (sapphire)': 'Al2O3',
            'SiO2 (bulk)': 'SiO2',
        }[substrate_choice]
        layers.append({
            'name': 'substrate', 'material': sub_mat,
            'thickness': substrate_d, 'coherent': False, 'fit_thickness': False,
        })

    return {
        'layers': layers,
        'fit': {
            'target': fit_target,
            'method': fit_method,
            'max_iter': max_iter,
            'delta_residual_mode': delta_mode,
            'loss': loss_fn,
            'two_stage': two_stage,
            'weighting': {
                'sigma_psi': sigma_psi,
                'sigma_delta': sigma_delta,
                'sigma_T': sigma_T,
            },
        },
    }


# ----- 跑擬合 -----
if run_fit:
    config = build_config()

    data = st.session_state.data.crop_wavelength(*wl_range)
    T_data = st.session_state.T_data
    if T_data is not None:
        T_data = T_data.crop_wavelength(*wl_range)

    st.session_state.fit_running = True
    st.session_state.abort_fit = False
    with st.spinner('擬合中…（可按右側「⛔ 中斷」停止）'):
        try:
            fitter = Fitter(config, data, T_data)
            result = fitter.fit(
                verbose=False,
                abort_check=lambda: st.session_state.get('abort_fit', False),
            )
            st.session_state.fit_result = result
            st.session_state.last_config = config
            if st.session_state.abort_fit:
                st.warning(f'⚠️ 使用者中斷。最後 MSE = {result.mse:.4e}')
            else:
                st.success(f'✓ 擬合完成，MSE = {result.mse:.4e}，{result.n_iter} evals')
        except Exception as e:
            st.error(f'擬合失敗：{e}')
            import traceback
            with st.expander('Traceback'):
                st.code(traceback.format_exc())
        finally:
            st.session_state.fit_running = False
            st.session_state.abort_fit = False


# ----- 結果顯示 -----
result = st.session_state.fit_result

if result is None:
    st.info('👈 在左側設定後按「開始擬合」')

    # 預覽：若有資料，先畫一下 raw data
    if st.session_state.data is not None:
        d = st.session_state.data
        st.subheader('📊 量測資料預覽')
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Ψ (deg)', 'Δ (deg)'))
        for j, ang in enumerate(d.angles):
            fig.add_trace(go.Scatter(x=d.wavelength, y=d.psi[:, j], name=f'AOI={ang}°',
                                     line=dict(width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=d.wavelength, y=d.delta[:, j], name=f'AOI={ang}°',
                                     showlegend=False, line=dict(width=1.5)), row=1, col=2)
        fig.update_xaxes(title_text='Wavelength (nm)')
        fig.update_layout(height=350, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        if st.session_state.T_data is not None:
            t = st.session_state.T_data
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=t.wavelength, y=t.T, mode='lines', line=dict(width=1.5)))
            fig2.update_layout(title='Transmission T', xaxis_title='Wavelength (nm)',
                               yaxis_title='T', height=300)
            st.plotly_chart(fig2, use_container_width=True)
else:
    # ---- 結果指標 ----
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        '🎯 MSE (WVASE)', f'{result.mse:.4e}',
        help='WVASE-style MSE，可直接與 WVASE 軟體輸出對比。\n'
             '公式：√(1/(2N-M)·Σ[(Ψc-Ψm)/σΨ]²+[(Δc-Δm)/σΔ]²)',
    )
    col2.metric('χ²', f'{result.chi_square:.3e}',
                help='lmfit 內部 χ² (殘差平方和)')
    col3.metric('變數參數', result.n_param)
    col4.metric('Function evals', result.n_iter,
                help='lmfit 的 nfev = 函式呼叫次數（不是 LM 迭代）')

    # WVASE MSE 解讀提示
    if result.mse < 1.0:
        st.success(f'✓ MSE = {result.mse:.4e} — 通常 WVASE MSE < 1 視為很好的擬合')
    elif result.mse < 10:
        st.info(f'MSE = {result.mse:.4e} — 中等擬合品質，可考慮加振盪器或放寬 bounds')
    else:
        st.warning(f'⚠️ MSE = {result.mse:.4e} 偏大 — 模型可能不對，或初始值離真值太遠')

    # ---- Ψ, Δ 對比圖 ----
    st.subheader('📊 Ψ, Δ 擬合結果')
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Ψ (deg)', 'Δ (deg)'))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for j, ang in enumerate(result.angles):
        c = colors[j % len(colors)]
        fig.add_trace(go.Scatter(x=result.wavelength, y=result.psi_meas[:, j],
                                 name=f'量測 AOI={ang}°', mode='markers',
                                 marker=dict(size=4, color=c)), row=1, col=1)
        fig.add_trace(go.Scatter(x=result.wavelength, y=result.psi_fit[:, j],
                                 name=f'擬合 AOI={ang}°', mode='lines',
                                 line=dict(color=c, width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=result.wavelength, y=result.delta_meas[:, j],
                                 mode='markers', marker=dict(size=4, color=c),
                                 showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=result.wavelength, y=result.delta_fit[:, j],
                                 mode='lines', line=dict(color=c, width=2),
                                 showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text='Wavelength (nm)')
    fig.update_layout(height=400, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    # ---- T 對比 ----
    if result.T_meas is not None and result.T_fit is not None:
        st.subheader('📊 穿透 T 擬合結果')
        figT = go.Figure()
        figT.add_trace(go.Scatter(x=result.wavelength, y=result.T_meas,
                                  name='量測', mode='markers', marker=dict(size=4)))
        figT.add_trace(go.Scatter(x=result.wavelength, y=result.T_fit,
                                  name='擬合', mode='lines', line=dict(width=2)))
        figT.update_layout(xaxis_title='Wavelength (nm)', yaxis_title='T',
                           height=300, hovermode='x unified')
        st.plotly_chart(figT, use_container_width=True)

    # ---- 薄膜光學常數 ----
    from ellipsometry.core.tmm_calc import pseudo_epsilon
    from ellipsometry.core.units import nm_to_eV

    # ---- X 軸單位切換 ----
    col_title, col_unit = st.columns([3, 1])
    with col_title:
        st.subheader('📊 光學常數')
    with col_unit:
        x_unit = st.radio(
            'X 軸單位', options=['nm', 'eV'],
            index=0, horizontal=True,
            label_visibility='collapsed', key='_x_unit',
        )

    # 薄膜（第二層）
    film_layer = result.layers[1]
    wl_arr = result.wavelength
    n_f, k_f = film_layer.material.n_k(wl_arr)
    eps_f = film_layer.material.epsilon(wl_arr)
    E_arr = nm_to_eV(wl_arr)

    if x_unit == 'nm':
        X_arr, x_label = wl_arr, 'Wavelength (nm)'
    else:
        X_arr, x_label = E_arr, 'Energy (eV)'

    tab_nk, tab_eps_film, tab_pseudo = st.tabs([
        '薄膜 n, k', '薄膜 ε₁, ε₂', '量測 <ε> (pseudo)',
    ])

    with tab_nk:
        fig_nk = make_subplots(rows=1, cols=2, subplot_titles=('n', 'k'))
        fig_nk.add_trace(go.Scatter(x=X_arr, y=n_f, name='n',
                                     line=dict(width=2)), row=1, col=1)
        fig_nk.add_trace(go.Scatter(x=X_arr, y=k_f, name='k',
                                     line=dict(width=2, color='#d62728')), row=1, col=2)
        fig_nk.update_xaxes(title_text=x_label)
        fig_nk.update_layout(height=380, hovermode='x unified')
        st.plotly_chart(fig_nk, use_container_width=True)

    with tab_eps_film:
        fig_eps = make_subplots(rows=1, cols=2, subplot_titles=('ε₁', 'ε₂'))
        fig_eps.add_trace(go.Scatter(x=X_arr, y=eps_f.real, name='ε₁',
                                      line=dict(width=2)), row=1, col=1)
        fig_eps.add_trace(go.Scatter(x=X_arr, y=eps_f.imag, name='ε₂',
                                      line=dict(width=2, color='#d62728')), row=1, col=2)
        fig_eps.update_xaxes(title_text=x_label)
        fig_eps.update_layout(height=380, hovermode='x unified',
                              title='薄膜真實介電函數（從擬合模型）')
        st.plotly_chart(fig_eps, use_container_width=True)

    with tab_pseudo:
        st.caption('偽介電函數 <ε> = 假設樣品是半無限均勻體反算的 ε。'
                   '對薄膜+基板樣品**不是真實薄膜 ε**，但形狀與峰位置可信，'
                   '用於決定振盪器數量。')
        fig_pe = make_subplots(rows=1, cols=2, subplot_titles=('<ε₁>', '<ε₂>'))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for j, ang in enumerate(result.angles):
            eps_p = pseudo_epsilon(result.psi_meas[:, j], result.delta_meas[:, j],
                                   ang, convention='tmm')
            c = colors[j % len(colors)]
            fig_pe.add_trace(go.Scatter(x=X_arr, y=eps_p.real,
                                         name=f'AOI={ang}°', line=dict(color=c)),
                              row=1, col=1)
            fig_pe.add_trace(go.Scatter(x=X_arr, y=eps_p.imag,
                                         name=f'AOI={ang}°', line=dict(color=c),
                                         showlegend=False), row=1, col=2)
        fig_pe.update_xaxes(title_text=x_label)
        fig_pe.update_layout(height=380, hovermode='x unified')
        st.plotly_chart(fig_pe, use_container_width=True)

    # ---- 擬合公式（依層顯示模型公式）----
    from ellipsometry.core.dispersion import (
        Pointwise, Constant, Cauchy, Sellmeier, Lorentz, Drude,
        DrudeLorentz, TaucLorentz, Gaussian, GenOsc, PolynomialNK,
    )

    def describe_layer(idx: int, layer):
        """產生單層的公式 + 參數描述（markdown）"""
        mat = layer.material
        lines = [f'**Layer {idx}: `{layer.name}`** — thickness = '
                 f'{layer.thickness_nm:.4g} nm '
                 f'({"coherent" if layer.coherent else "incoherent"})']

        if isinstance(mat, Pointwise):
            lines.append(f'- **Pointwise**: 從表插值（{mat.name}, method={mat.method}）')
        elif isinstance(mat, Constant):
            lines.append(f'- **Constant**: n + ik = {mat.n} + {mat.k}i')
        elif isinstance(mat, Cauchy):
            lines.append(f'- **Cauchy**: $n(\\lambda) = A + B/\\lambda^2 + C/\\lambda^4$')
            lines.append(f'  - A={mat.A}, B={mat.B}, C={mat.C}  (λ in μm)')
        elif isinstance(mat, Sellmeier):
            lines.append(f'- **Sellmeier**: $n^2-1 = \\sum_i B_i\\lambda^2/(\\lambda^2-C_i^2)$')
            lines.append(f'  - coefficients = {mat.coefficients}')
        elif isinstance(mat, Drude):
            lines.append(f'- **Drude**: $\\varepsilon = \\varepsilon_\\infty - \\omega_p^2/(E^2+i\\Gamma E)$')
            lines.append(f'  - ε∞={mat.eps_inf}, ωp={mat.omega_p} eV, Γ={mat.gamma} eV')
        elif isinstance(mat, DrudeLorentz):
            lines.append(f'- **Drude-Lorentz**: $\\varepsilon = \\varepsilon_\\infty + \\text{{Drude}} + \\sum_k \\text{{Lorentz}}_k$')
            lines.append(f'  - ε∞ = {mat.eps_inf}')
            if mat.drude:
                lines.append(f'  - Drude: ωp={mat.drude["omega_p"]} eV, Γ={mat.drude["gamma"]} eV')
            for i, osc in enumerate(mat.lorentz):
                lines.append(f'  - Lorentz #{i+1}: A={osc.get("A")}, '
                             f'E0={osc.get("E0")} eV, γ={osc.get("gamma")} eV')
        elif isinstance(mat, GenOsc):
            lines.append(f'- **GenOsc** (WVASE-style):')
            lines.append(f'  $\\varepsilon(E) = \\varepsilon_\\infty^{{offset}} + \\text{{poles}} + \\sum_i \\text{{osc}}_i(E)$')
            lines.append(f'  - ε∞ offset = {mat.e1_offset}')
            if mat.Egap:
                lines.append(f'  - Egap = {mat.Egap} eV')
            if mat.uv_pole:
                lines.append(f'  - UV pole: pos={mat.uv_pole.position} eV, mag={mat.uv_pole.magnitude}')
            if mat.ir_pole:
                lines.append(f'  - IR pole: pos={mat.ir_pole.position} eV, mag={mat.ir_pole.magnitude}')
            for i, osc in enumerate(mat.oscillators):
                if not osc.active:
                    continue
                if osc.type == 'drude':
                    lines.append(f'  - Osc #{i+1} **Drude**: $-\\text{{Amp}}/(E^2+i\\text{{Br}}E)$, '
                                 f'Amp={osc.amp:.4g}, Br={osc.br:.4g}')
                elif osc.type == 'lorentz':
                    lines.append(f'  - Osc #{i+1} **Lorentz**: $A\\cdot E_n^2/(E_n^2-E^2-iBrE)$, '
                                 f'A={osc.amp:.4g}, En={osc.en:.4g}, Br={osc.br:.4g}')
                elif osc.type == 'gaussian':
                    lines.append(f'  - Osc #{i+1} **Gaussian**: $A\\cdot\\exp(-((E-E_n)/\\sigma)^2)$ + KK, '
                                 f'A={osc.amp:.4g}, En={osc.en:.4g}, Br={osc.br:.4g}')
                elif osc.type == 'tauc_lorentz':
                    lines.append(f'  - Osc #{i+1} **Tauc-Lorentz**: '
                                 f'A={osc.amp:.4g}, En={osc.en:.4g}, Br={osc.br:.4g}, Eg={osc.Eg:.4g}')
                else:
                    lines.append(f'  - Osc #{i+1} {osc.type}: amp={osc.amp:.4g}, en={osc.en:.4g}, br={osc.br:.4g}')
        elif isinstance(mat, PolynomialNK):
            lines.append(f'- **PolynomialNK** (degree {len(mat.n_coeffs)-1}):')
            lines.append(f'  $n(E) = \\sum a_i E^i$, $k(E) = \\sum b_i E^i$')
            lines.append(f'  - n_coeffs = {[round(c, 4) for c in mat.n_coeffs]}')
            lines.append(f'  - k_coeffs = {[round(c, 4) for c in mat.k_coeffs]}')
        return '\n'.join(lines)

    st.subheader('📐 擬合公式')
    with st.container(border=True):
        for i, ly in enumerate(result.layers):
            st.markdown(describe_layer(i, ly))
            if i < len(result.layers) - 1:
                st.markdown('---')

    # ---- 參數表 ----
    st.subheader('📋 最佳化參數')

    def parse_param_name(pname: str) -> tuple:
        """L1__thickness → (Layer 1, thickness)
           L1__oscillators_0__amp → (Layer 1, oscillators[0].amp)
        """
        if not pname.startswith('L'):
            return ('?', pname)
        try:
            li_str, rest = pname[1:].split('__', 1)
            li = int(li_str)
            # 把 oscillators_0__amp 還原成 oscillators[0].amp
            human = rest.replace('__', '.')
            # _0 →[0] 等（簡單啟發式）
            import re
            human = re.sub(r'_(\d+)', r'[\1]', human)
            layer_name = result.layers[li].name if li < len(result.layers) else f'L{li}'
            return (layer_name, human)
        except Exception:
            return ('?', pname)

    rows = []
    for pname, val in result.params.items():
        se = result.params_stderr.get(pname)
        layer_name, human_param = parse_param_name(pname)
        # 從 lmfit 取 bounds
        try:
            lmfit_p = result._lmfit_result.params[pname]
            lo, hi = lmfit_p.min, lmfit_p.max
            vary = lmfit_p.vary
        except Exception:
            lo = hi = float('nan'); vary = True

        rel_unc = (se / abs(val) * 100) if (se and val) else None

        rows.append({
            'Layer': layer_name,
            'Parameter': human_param,
            'Value': f'{val:.5g}',
            '± Uncertainty': f'{se:.3g}' if se else '—',
            'Rel %': f'{rel_unc:.2f}%' if rel_unc is not None else '—',
            'Bound Lo': f'{lo:.4g}' if np.isfinite(lo) else '−∞',
            'Bound Hi': f'{hi:.4g}' if np.isfinite(hi) else '+∞',
            'Fit?': '✓' if vary else 'fixed',
        })
    df_params = pd.DataFrame(rows)
    st.dataframe(df_params, use_container_width=True, hide_index=True)

    # 提供下載
    st.download_button(
        '📥 下載參數表 (csv)',
        data=df_params.to_csv(index=False),
        file_name='fit_parameters.csv', mime='text/csv',
    )

    # ---- 下載結果 ----
    st.subheader('💾 下載')
    col_a, col_b, col_c = st.columns(3)

    # 1. config
    with col_a:
        cfg_yaml = yaml.dump(st.session_state.last_config, allow_unicode=True,
                             default_flow_style=False)
        st.download_button('📄 下載 config.yaml', data=cfg_yaml,
                           file_name='fit_config.yaml', mime='text/yaml')

    # 2. nk 表（用前面已算好的 n_f, k_f, eps_f, E_arr）
    with col_b:
        nk_df = pd.DataFrame({
            'wavelength_nm': result.wavelength,
            'energy_eV': E_arr,
            'n': n_f,
            'k': k_f,
            'eps1': eps_f.real,
            'eps2': eps_f.imag,
        })
        st.download_button('📊 下載 n,k,ε 表 (csv)',
                           data=nk_df.to_csv(index=False),
                           file_name=f'{film_layer.name}_nk.csv', mime='text/csv')

    # 3. 擬合曲線
    with col_c:
        fit_rows = []
        for j, a in enumerate(result.angles):
            for i, wl in enumerate(result.wavelength):
                fit_rows.append({
                    'wavelength_nm': wl, 'aoi_deg': a,
                    'psi_meas': result.psi_meas[i, j], 'psi_fit': result.psi_fit[i, j],
                    'delta_meas': result.delta_meas[i, j], 'delta_fit': result.delta_fit[i, j],
                })
        fit_df = pd.DataFrame(fit_rows)
        st.download_button('📈 下載擬合曲線 (csv)', data=fit_df.to_csv(index=False),
                           file_name='fit_curves.csv', mime='text/csv')

# 頁腳
st.markdown('---')
st.caption('Ellipsometry Fit Tool · Powered by lmfit + tmm + Streamlit · '
           f'材料庫: {len(AVAILABLE_MATERIALS)} 個')
