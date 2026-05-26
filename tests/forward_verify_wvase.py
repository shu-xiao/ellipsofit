"""
Forward TMM 驗證腳本

把 WVASE 已抽出的 ε(λ) 當作「正確答案」材料，用我們的 TMM forward 算 Ψ, Δ，
與原始量測比 RMSE。

目的：
    判斷我們 fit 與 WVASE 結果差異的根源
    - RMSE 小 (~1-2°) → forward 對，問題在 fit 邏輯（初始值/bounds/iter）
    - RMSE 大 (>10°)  → forward 有 bug（基板 / Δ 慣例 / Fresnel）

使用：
    python tests/forward_verify_wvase.py [eps_file] [r_file]
    預設用 data/NbTiN/thick/ 的 100nm 樣品
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from ellipsometry.core.io import read_reflection
from ellipsometry.core.tmm_calc import Layer, calculate
from ellipsometry.core.dispersion import Pointwise, MaterialLibrary


# =============================================================================
# 從 ε 轉回 n, k（複數開根號取主分支 Im≥0）
# =============================================================================

def eps_to_nk(eps1, eps2):
    """ε = (n + ik)² → 取 n, k"""
    eps = np.asarray(eps1) + 1j * np.asarray(eps2)
    nc = np.sqrt(eps)
    # 主分支：Im(n+ik) ≥ 0
    nc = np.where(nc.imag < 0, np.conj(nc), nc)
    return nc.real, nc.imag


# =============================================================================
# 主流程
# =============================================================================

def verify(eps_file: str, r_file: str, film_thickness_nm: float,
           substrate_name: str = 'Al2O3', substrate_thickness_nm=np.inf):
    print('='*70)
    print(f'  Forward verify: {os.path.basename(eps_file)}')
    print(f'  R file:     {os.path.basename(r_file)}')
    print(f'  Film:       {film_thickness_nm} nm')
    print(f'  Substrate:  {substrate_name}, '
          f'{"semi-infinite" if not np.isfinite(substrate_thickness_nm) else f"{substrate_thickness_nm/1e6} mm"}')
    print('='*70)

    # ---- 讀 WVASE ε ----
    eps_df = pd.read_csv(eps_file, sep=r'\s+', header=None,
                         names=['nm', 'eps1', 'eps2'])
    n_arr, k_arr = eps_to_nk(eps_df.eps1.values, eps_df.eps2.values)
    print(f'\n📊 WVASE ε 摘要：')
    print(f'  {len(eps_df)} 點，{eps_df.nm.min():.0f}-{eps_df.nm.max():.0f} nm')
    for w in [350, 500, 800, 1500]:
        idx = np.argmin(np.abs(eps_df.nm.values - w))
        print(f'  λ={w}nm: ε=({eps_df.eps1.iloc[idx]:.2f},{eps_df.eps2.iloc[idx]:.2f}) '
              f'→ n+ik = {n_arr[idx]:.3f} + {k_arr[idx]:.3f}i')

    # ---- 包成 Pointwise material ----
    nbn_mat = Pointwise(eps_df.nm.values, n_arr, k_arr,
                        method='pchip', name='WVASE_extracted_NbN')

    # ---- 讀量測 R（io 自動 Δ convention 處理）----
    R = read_reflection(r_file)
    # 截取 ε 涵蓋的波長範圍
    R = R.crop_wavelength(eps_df.nm.min(), eps_df.nm.max())
    print(f'\n📊 量測 R: {R}')

    # ---- 組層 + forward ----
    layers = [
        Layer('air', MaterialLibrary.get('air'), np.inf, coherent=False),
        Layer('NbN', nbn_mat, float(film_thickness_nm), coherent=True),
        Layer('substrate', MaterialLibrary.get(substrate_name),
              substrate_thickness_nm, coherent=False),
    ]
    res = calculate(layers, R.wavelength, R.angles, compute_transmission=False)

    # ---- 比對 ----
    rmse_psi = np.sqrt(np.mean((res.psi - R.psi)**2))
    ddelta = ((res.delta - R.delta + 180) % 360) - 180
    rmse_delta = np.sqrt(np.mean(ddelta**2))

    # WVASE-style MSE（假設 σ_Ψ=0.028, σ_Δ=0.12 — 與量測一致）
    sigma_psi = R.sigma_psi.mean() if R.sigma_psi is not None else 0.5
    sigma_delta = R.sigma_delta.mean() if R.sigma_delta is not None else 1.0
    ndata = R.psi.size + R.delta.size
    mse_wvase = np.sqrt(
        (np.sum((res.psi - R.psi)**2) / sigma_psi**2
         + np.sum(ddelta**2) / sigma_delta**2)
        / max(ndata - 0, 1)   # forward，無 fitted parameters → DoF = ndata
    )

    print(f'\n🎯 結果：')
    print(f'  Ψ RMSE = {rmse_psi:.3f}°  (儀器 σ_Ψ = {sigma_psi:.4f}°)')
    print(f'  Δ RMSE = {rmse_delta:.3f}°  (儀器 σ_Δ = {sigma_delta:.4f}°)')
    print(f'  MSE (WVASE 公式) = {mse_wvase:.2f}')

    # 解讀
    if rmse_psi < 3 and rmse_delta < 5:
        print('\n  ✅ Forward TMM 與 WVASE 高度一致')
        print('     若 fit 後 MSE 仍大 → 問題在初始值/bounds/iter，不是物理')
    elif rmse_psi < 10 and rmse_delta < 15:
        print('\n  🟡 Forward 大致對但有差異 (~%.0f° on average)' % rmse_psi)
        print('     可能：基板 nk 數據不同 / Δ convention 殘留誤差 / sigma 計算法')
    else:
        print('\n  ❌ Forward 與 WVASE 差距大')
        print('     可能：Fresnel/Snell bug / 基板模型錯 / Δ convention 未對齊')

    # 細節：列幾個波長 + AOI
    print(f'\n  細節：')
    for w in [400, 800, 1500]:
        idx = np.argmin(np.abs(R.wavelength - w))
        for j, ang in enumerate(R.angles):
            print(f'    λ={w} AOI={ang:.0f}°: 我 Ψ={res.psi[idx,j]:6.2f} '
                  f'Δ={res.delta[idx,j]:7.2f}  '
                  f'vs 量 Ψ={R.psi[idx,j]:6.2f} Δ={R.delta[idx,j]:7.2f}')
    return rmse_psi, rmse_delta, mse_wvase


if __name__ == '__main__':
    # 預設：100nm NbN/sapphire
    eps_default = 'data/NbTiN/thick/nbn_sapph_100nm_12_0p1.txt'
    r_default   = 'data/NbTiN/thick/nbn_r.dat'
    eps_file = sys.argv[1] if len(sys.argv) > 1 else eps_default
    r_file   = sys.argv[2] if len(sys.argv) > 2 else r_default

    # WVASE .env 是二進位讀不到精確厚度 — 假設 100 nm（檔名暗示）
    # 後續可從 fit 微調
    verify(eps_file, r_file, film_thickness_nm=100.0,
           substrate_name='Al2O3', substrate_thickness_nm=np.inf)
