"""
WVASE .mat 檔解析器

支援兩種 WVASE 材料檔格式：

    1. 表格式 (tabulated)
        comment
        eV  或 nm
        NK  或 e1e2
        <x>  <col2>  <col3>
        ...

    2. GenOsc 模型
        comment
        GENOSC
        <flags 14 個數> ─→ flags[0] = 振盪器數量 N
        <UV_pos UV_mag IR_pos IR_mag e1_offset Egap>
        <fit_range_min fit_range_max fit_range_step>  (eV，僅供顯示)
        <oscillator 1>  format: 1 TYPE 0 P1 P2 P3 P4 P5 P6 P7
        <oscillator 2>
        ...

GenOsc 公式（WVASE 對應）：
    ε(E) = e1_offset
         + UV_mag / (UV_pos² − E²)         (UV pole, 對 ε1 貢獻)
         + IR_mag / (IR_pos² − E²)         (IR pole)
         + Σᵢ ε_osc_i(E)

WVASE 振盪器類型代號（已驗證）：
    2  = Lorentz       (Amp, En, Br)
    7  = Tauc-Lorentz  (Amp, En, Br, Eg)
    其他 → 警告並用 Lorentz 近似

Usage:
    from ellipsometry.core.wvase_mat import read_wvase_mat
    sapph = read_wvase_mat('data/substrate/Sapphire_e_sell.mat')
    n, k = sapph.n_k(np.array([500.0]))   # 1.766, 0
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np

from .dispersion import (
    DispersionModel, Pointwise, GenOsc, GenOscOscillator, Pole,
)
from .units import nm_to_eV, eV_to_nm


# WVASE 振盪器類型對照（已驗證部分）
_WVASE_OSC_TYPES = {
    2:  'lorentz',
    7:  'tauc_lorentz',
    # 26 = Cody-Lorentz / PSemi-M0 (5+ params), 暫用 lorentz 近似
    # 0  = Sellmeier-like? 不確定
    # 5  = ?
}


def read_wvase_mat(path: str | Path) -> DispersionModel:
    """讀 WVASE .mat 檔，自動偵測格式並回傳對應 dispersion model

    Returns
    -------
    Pointwise (若為表格式) 或 GenOsc (若為 GenOsc 模型)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'找不到 .mat 檔：{path}')

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [ln.rstrip('\r\n') for ln in f]

    if len(lines) < 3:
        raise ValueError(f'檔案太短：{path}')

    # 第 2 行決定格式
    fmt_line = lines[1].strip().upper()
    if fmt_line == 'GENOSC':
        return _parse_genosc(lines, name=path.stem)
    if fmt_line in ('EV', 'NM'):
        return _parse_tabulated(lines, name=path.stem)
    raise ValueError(f'未知 .mat 格式 (line 2 = {lines[1]!r}): {path}')


# =============================================================================
# 表格式
# =============================================================================

def _parse_tabulated(lines: list[str], name: str) -> Pointwise:
    """解析 tabulated 格式（eV/nm + NK/e1e2 + 3 col 數據）"""
    unit = lines[1].strip().lower()       # 'ev' or 'nm'
    data_type = lines[2].strip().upper()  # 'NK' or 'E1E2'

    # 4 行起為資料
    rows = []
    for ln in lines[3:]:
        ln = ln.strip()
        if not ln:
            continue
        try:
            parts = [float(x) for x in ln.replace(',', ' ').split()]
        except ValueError:
            continue
        if len(parts) >= 3:
            rows.append(parts[:3])
    if not rows:
        raise ValueError(f'{name}: 找不到資料')

    arr = np.array(rows)
    x = arr[:, 0]
    c1 = arr[:, 1]
    c2 = arr[:, 2]

    # 統一轉成 (nm, n, k)
    wl_nm = x.copy() if unit == 'nm' else eV_to_nm(x)

    if data_type == 'NK':
        n_arr, k_arr = c1, c2
    elif data_type == 'E1E2':
        eps = c1 + 1j * c2
        nc = np.sqrt(eps)
        nc = np.where(nc.imag < 0, np.conj(nc), nc)
        n_arr, k_arr = nc.real, nc.imag
    else:
        raise ValueError(f'{name}: 未知 data type {data_type!r}')

    # 反轉成升冪
    order = np.argsort(wl_nm)
    return Pointwise(
        wl_nm[order], n_arr[order], k_arr[order],
        method='pchip', name=name,
    )


# =============================================================================
# GenOsc 模型
# =============================================================================

def _parse_floats(s: str) -> list[float]:
    """解析一行內所有數字，容忍科學記號與多空白"""
    out = []
    for tok in s.replace(',', ' ').split():
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def _parse_genosc(lines: list[str], name: str) -> GenOsc:
    """解析 WVASE GenOsc 模型"""
    # 找到 GENOSC 行的索引
    gen_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().upper() == 'GENOSC':
            gen_idx = i
            break
    if gen_idx is None:
        raise ValueError(f'{name}: 找不到 GENOSC 標記')

    # Line gen_idx+1: flags (14 numbers)
    flags = _parse_floats(lines[gen_idx + 1])
    n_osc = int(flags[0]) if flags else 0

    # Line gen_idx+2: poles + offset+egap (6 numbers)
    # [UV_pos, UV_mag, IR_pos, IR_mag, e1_offset, Egap]
    poles_row = _parse_floats(lines[gen_idx + 2])
    if len(poles_row) < 6:
        poles_row += [0.0] * (6 - len(poles_row))
    uv_pos, uv_mag, ir_pos, ir_mag, e1_offset, egap = poles_row[:6]

    # Line gen_idx+3: fit range (3 numbers, 略過)
    # gen_idx+4 起：N 個 osc 行
    osc_lines = []
    for ln in lines[gen_idx + 4:]:
        s = ln.strip()
        if not s:
            continue
        vals = _parse_floats(s)
        if len(vals) >= 3:
            osc_lines.append(vals)
        if len(osc_lines) >= n_osc:
            break

    oscillators = []
    for vals in osc_lines:
        osc = _parse_osc_line(vals, name)
        if osc is not None:
            oscillators.append(osc)

    # 建 GenOsc model
    # WVASE pole 公式：ε1_pole = mag / (pos² − E²)
    # 我們 Pole class 內：ε1_contrib = mag · pos² / (pos² − E²)
    # → 兩者差一個 pos² 因子！必須做轉換：
    #   WVASE mag → 我們 mag_internal = WVASE_mag / pos²
    uv_pole = None
    ir_pole = None
    if uv_mag != 0 and uv_pos > 0:
        uv_pole = Pole(position=uv_pos, magnitude=uv_mag / (uv_pos ** 2))
    if ir_mag != 0 and ir_pos > 0:
        ir_pole = Pole(position=ir_pos, magnitude=ir_mag / (ir_pos ** 2))

    return GenOsc(
        e1_offset=e1_offset,
        Egap=egap,
        uv_pole=uv_pole,
        ir_pole=ir_pole,
        oscillators=oscillators,
    )


def _parse_osc_line(vals: list[float], name: str):
    """解析單一振盪器行：[1 TYPE 0 P1 P2 P3 P4 P5 P6 P7]

    回傳 GenOscOscillator 或 None（無法解析時）

    WVASE 振盪器標準形式：
        type=2  (Lorentz):       Amp · En² / (En² − E² − iBr·E)
        type=7  (Tauc-Lorentz):  TL 公式含 Eg

    我們 Lorentz 同公式，Tauc-Lorentz 同公式，可直接對應。
    """
    if len(vals) < 6:
        return None
    # 前三個整數：[1, TYPE, 0/other]
    osc_type_code = int(vals[1])
    osc_name = _WVASE_OSC_TYPES.get(osc_type_code)

    if osc_name is None:
        warnings.warn(f'{name}: 未支援的 WVASE 振盪器 type={osc_type_code}, '
                      f'用 lorentz 近似 (前 3 個參數)',
                      stacklevel=3)
        osc_name = 'lorentz'

    # 參數從 index 3 開始
    p = vals[3:]
    amp = p[0] if len(p) > 0 else 0.0
    en  = p[1] if len(p) > 1 else 0.0
    br  = p[2] if len(p) > 2 else 0.0
    eg  = p[3] if (len(p) > 3 and osc_name == 'tauc_lorentz') else 0.0
    return GenOscOscillator(
        type=osc_name, amp=amp, en=en, br=br, Eg=eg, active=True,
    )


# =============================================================================
# 自測：python -m ellipsometry.core.wvase_mat
# =============================================================================

if __name__ == '__main__':
    import sys
    test_files = sys.argv[1:] if len(sys.argv) > 1 else [
        'data/substrate/Sapphire_e_sell.mat',
        'data/substrate/Sapphire_o_sell.mat',
        'data/substrate/MgO_g.mat',
        'data/substrate/SiO2_jaw.mat',
        'data/substrate/Silicon.mat',
    ]

    expected = {
        'Sapphire_e_sell': ('500 nm n_e', 1.7660, 1.768),
        'Sapphire_o_sell': ('500 nm n_o', 1.7742, 1.776),
        'SiO2_jaw':        ('500 nm n', 1.4624, 1.464),
        'Silicon':         ('500 nm n', 4.30, 4.32),
    }

    for f in test_files:
        try:
            print(f'\n=== {f} ===')
            mat = read_wvase_mat(f)
            wl = np.array([300, 500, 800, 1500.0])
            n, k = mat.n_k(wl)
            for w, nn, kk in zip(wl, n, k):
                print(f'  λ={w:5.0f} nm: n={nn:.4f}  k={kk:.4f}')
            # 驗證 500nm 值
            stem = Path(f).stem
            if stem in expected:
                label, expect_val, tol = expected[stem]
                idx = list(wl).index(500.0)
                ok = abs(n[idx] - expect_val) < (tol - expect_val + 0.005)
                mark = '✓' if ok else '✗'
                print(f'  Expected {label} ≈ {expect_val:.4f}, '
                      f'got {n[idx]:.4f} {mark}')
        except Exception as e:
            print(f'  ✗ 失敗：{type(e).__name__}: {e}')
