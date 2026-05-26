"""
TMM (Transfer Matrix Method) 計算模組

封裝 tmm 套件，將色散模型 (DispersionModel) 組成的層堆疊轉換為：
    - 反射：Ψ(λ, AOI), Δ(λ, AOI)
    - 穿透：T(λ) at AOI=0
    - 偽介電函數 <ε>(λ, AOI)（用於探索/快速看圖）
    - 點對點 nk 反演（給 fit e2_only 模式用）

對應 WVASE 「Substrate Backside Correction = ON」行為：
    薄膜層 coherent=True → 干涉
    基板層 coherent=False + 有限厚度 → 強度疊加（無 fringes）
    自動在最後加 air exit 處理穿透

Usage:
    layers = [
        Layer('ambient', MaterialLibrary.get('air'),   np.inf, coherent=False),
        Layer('film',    drude_lorentz_model,           50.0,   coherent=True),
        Layer('subst',   MaterialLibrary.get('Si'),     1e6,    coherent=False),
    ]
    res = calculate(layers, wl_nm, [65, 70, 75])
    print(res.psi.shape)   # (Nw, 3)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import tmm

from .dispersion import DispersionModel
from .units import nm_to_eV
from .tmm_vec import ellips_vec, coh_tmm_vec


# =============================================================================
# 資料容器
# =============================================================================

@dataclass
class Layer:
    """單層膜定義

    Attributes
    ----------
    name : str
        識別名稱（log 用）
    material : DispersionModel
        色散模型（可呼叫 .n_complex(wl)）
    thickness_nm : float
        厚度 (nm)。np.inf 表示半無限（最後一層基板用）
    coherent : bool
        True=相干（薄膜應 True），False=非相干（厚基板應 False）
    """
    name: str
    material: DispersionModel
    thickness_nm: float
    coherent: bool = True


@dataclass
class TMMResult:
    """計算結果"""
    wavelength: np.ndarray             # (Nw,) nm
    angles: list                       # [AOI 度]
    psi: np.ndarray                    # (Nw, Nangle) 度
    delta: np.ndarray                  # (Nw, Nangle) 度
    T: Optional[np.ndarray] = None     # (Nw,) 穿透率 (0-1) at AOI=0
    R_unp: Optional[np.ndarray] = None # (Nw, Nangle) 非偏振反射強度
    layers: list = field(default_factory=list)   # 紀錄用


# =============================================================================
# 主計算函式
# =============================================================================

def calculate(layers: list[Layer],
              wavelength_nm: np.ndarray,
              aoi_deg_list: list[float],
              compute_transmission: bool = True,
              compute_R: bool = False,
              nk_cache: Optional[dict] = None) -> TMMResult:
    """正向計算 Ψ, Δ, T

    Parameters
    ----------
    layers : list[Layer]
        從入射側到基板（含 ambient 和基板）
    wavelength_nm : array (Nw,)
    aoi_deg_list : list[float]
    compute_transmission : bool
        True=計算 T (AOI=0)，需要基板有有限厚度
    compute_R : bool
        True=額外回傳非偏振反射強度

    Returns
    -------
    TMMResult
    """
    if len(layers) < 2:
        raise ValueError(f'至少需要 ambient + 基板兩層，收到 {len(layers)} 層')

    wl = np.asarray(wavelength_nm, dtype=float)
    Nw = len(wl)
    Nang = len(aoi_deg_list)

    # 先預計算所有層的複數折射率 (Nw,)
    # 加速 B：用 nk_cache 避免重複計算固定材料（id(material) 為 key）
    nk_table = []
    for ly in layers:
        if nk_cache is not None:
            key = (id(ly.material), id(wl))
            if key in nk_cache:
                nk_table.append(nk_cache[key])
                continue
        nk = ly.material.n_complex(wl)
        if nk_cache is not None:
            nk_cache[key] = nk
        nk_table.append(nk)

    # 判斷基板是否為有限厚度（決定要不要加 air exit）
    substrate = layers[-1]
    substrate_finite = np.isfinite(substrate.thickness_nm)

    # 建構反射用堆疊
    # - 若基板有限：[ambient, films..., substrate, air_exit]
    # - 若基板無限：[ambient, films..., substrate(inf)]
    if substrate_finite:
        d_R = [np.inf] + [ly.thickness_nm for ly in layers[1:]] + [np.inf]
        c_R = ['i']    + [_coh_flag(ly) for ly in layers[1:]] + ['i']
        n_arrays_R = nk_table + [np.ones_like(wl, dtype=complex)]  # air exit
    else:
        d_R = [np.inf] + [ly.thickness_nm for ly in layers[1:]]
        c_R = ['i']    + [_coh_flag(ly) for ly in layers[1:]]
        n_arrays_R = nk_table

    aoi_rad = [np.deg2rad(a) for a in aoi_deg_list]

    psi = np.zeros((Nw, Nang))
    delta = np.zeros((Nw, Nang))
    R_unp = np.zeros((Nw, Nang)) if compute_R else None
    T = np.zeros(Nw) if compute_transmission else None

    # 加速 A：預判堆疊類型
    has_incoherent = any(c == 'i' for c in c_R[1:-1]) or \
                     (substrate_finite and 'i' in c_R[-2:])

    # ★ 向量化：對每個 AOI 一次處理所有波長（127× 加速 vs 純量 tmm）
    for j, theta in enumerate(aoi_rad):
        # ellips_vec 全波長同時算 → Ψ, Δ
        r_ell = ellips_vec(n_arrays_R, d_R, theta, wl)
        delta[:, j] = np.rad2deg(r_ell['Delta'])

        if has_incoherent:
            # 含 incoherent → 仍需 inc_tmm 算 Ψ 強度修正
            # 此部分尚未向量化（複雜），用 scalar loop
            for i, wlam in enumerate(wl):
                n_list = [arr[i] for arr in n_arrays_R]
                rs = tmm.inc_tmm('s', n_list, d_R, c_R, theta, wlam)
                rp = tmm.inc_tmm('p', n_list, d_R, c_R, theta, wlam)
                Rs, Rp = rs['R'], rp['R']
                psi[i, j] = np.rad2deg(np.arctan(np.sqrt(max(Rp / max(Rs, 1e-30), 0))))
                if compute_R:
                    R_unp[i, j] = 0.5 * (Rs + Rp)
        else:
            # 全相干 → Ψ 直接用向量化 ellips 結果
            psi[:, j] = np.rad2deg(r_ell['psi'])
            if compute_R:
                Rs_vec = np.abs(r_ell['r_s'])**2
                Rp_vec = np.abs(r_ell['r_p'])**2
                R_unp[:, j] = 0.5 * (Rs_vec + Rp_vec)

    # 穿透：normal incidence (仍 scalar，掃過所有波長)
    if compute_transmission and substrate_finite:
        for i, wlam in enumerate(wl):
            n_list = [arr[i] for arr in n_arrays_R]
            inc = tmm.inc_tmm('s', n_list, d_R, c_R, 0.0, wlam)
            T[i] = inc['T']

    return TMMResult(
        wavelength=wl, angles=list(aoi_deg_list),
        psi=psi, delta=delta, T=T, R_unp=R_unp,
        layers=layers,
    )


def _coh_flag(layer: Layer) -> str:
    """Layer.coherent → tmm 的 'c'/'i' 字串"""
    return 'c' if layer.coherent else 'i'


def _delta_from_coherent(n_list, d_list, theta, wlam) -> float:
    """對相干堆疊（忽略背面非相干部分）算 Δ 相位

    背面非相干時，Δ 主要由前表面決定；用相干 ellips 抽 Δ 是常用近似。
    """
    # 把所有 incoherent 層也當 coherent 處理只為了取相位資訊
    try:
        r = tmm.ellips(n_list, d_list, theta, wlam)
        return r['Delta']
    except Exception:
        return 0.0


# =============================================================================
# 偽介電函數 (pseudo-eps) — 直接從 Ψ, Δ 反推（單一介面近似）
# =============================================================================

def pseudo_epsilon(psi_deg: np.ndarray, delta_deg: np.ndarray,
                   aoi_deg: float, convention: str = 'tmm') -> np.ndarray:
    """偽介電函數 <ε>（假設樣品是半無限均勻體）

        ρ = ±tan(Ψ)·exp(iΔ)        ← 符號由 convention 決定
        <ε> = sin²θ · [1 + tan²θ · ((1-ρ)/(1+ρ))²]

    Parameters
    ----------
    convention : 'tmm' or 'standard'
        'tmm'      — ρ = -rp/rs (Steven Byrnes tmm 套件慣例，本程式內部用)
        'standard' — ρ =  rp/rs (Azzam & Bashara 教科書 / 多數 WVASE 輸出)
        差別只在 Δ 偏 180°；對應 ρ 符號正/負。

    回傳複數 <ε>
    """
    psi = np.deg2rad(psi_deg)
    delta = np.deg2rad(delta_deg)
    rho = np.tan(psi) * np.exp(1j * delta)
    if convention == 'tmm':
        rho = -rho                                   # 轉成 rp/rs 的標準慣例
    elif convention != 'standard':
        raise ValueError(f'unknown convention: {convention}')

    sin_t = np.sin(np.deg2rad(aoi_deg))
    sin_t2 = sin_t**2
    tan_t2 = (sin_t / np.cos(np.deg2rad(aoi_deg)))**2
    inner = ((1 - rho) / (1 + rho))**2
    return sin_t2 * (1 + tan_t2 * inner)


# =============================================================================
# 點對點 nk 反演 (point-by-point inversion)
# =============================================================================

def invert_nk_pointbypoint(measured_psi: np.ndarray,
                           measured_delta: np.ndarray,
                           wavelength_nm: np.ndarray,
                           aoi_deg: float,
                           fixed_layers_above: list[Layer],
                           target_thickness_nm: float,
                           substrate: Layer,
                           initial_nk: Optional[np.ndarray] = None,
                           verbose: bool = False) -> np.ndarray:
    """對每個波長 λᵢ 解 2×2 方程組找 (n, k)，使 TMM 算出 Ψ, Δ 吻合實驗

    Parameters
    ----------
    measured_psi, measured_delta : (Nw,) array
        實驗值（單一 AOI）
    wavelength_nm : (Nw,)
    aoi_deg : float
    fixed_layers_above : list[Layer]
        目標薄膜之上的層（含 ambient），nk 已知不變
    target_thickness_nm : float
        目標薄膜厚度（必須已知並固定）
    substrate : Layer
        基板（nk 已知）
    initial_nk : (Nw, 2) array, optional
        每個 λ 的初始 (n, k) 猜測。預設為 (1.5, 0.01)
    verbose : bool

    Returns
    -------
    nk : (Nw, 2) array
        反推的 n(λᵢ), k(λᵢ)
    """
    from scipy.optimize import least_squares
    from .dispersion import Constant

    Nw = len(wavelength_nm)
    if initial_nk is None:
        initial_nk = np.tile([1.5, 0.01], (Nw, 1))

    result = np.zeros((Nw, 2))

    for i, wl in enumerate(wavelength_nm):
        n0, k0 = initial_nk[i]
        psi_meas = measured_psi[i]
        delta_meas = measured_delta[i]

        def residual(params):
            n_try, k_try = params
            if k_try < 0:
                return [1e6, 1e6]
            target_layer = Layer(
                name='_target', material=Constant(n=n_try, k=k_try),
                thickness_nm=target_thickness_nm, coherent=True,
            )
            stack = list(fixed_layers_above) + [target_layer, substrate]
            res = calculate(
                stack, np.array([wl]), [aoi_deg],
                compute_transmission=False, compute_R=False,
            )
            d_psi = res.psi[0, 0] - psi_meas
            d_delta = ((res.delta[0, 0] - delta_meas + 180) % 360) - 180  # wrap
            return [d_psi, d_delta]

        try:
            sol = least_squares(residual, [n0, k0],
                                bounds=([0.01, 0], [10, 20]),
                                max_nfev=50)
            result[i] = sol.x
            if verbose and i % 50 == 0:
                print(f'  λ={wl:.0f}nm  n={sol.x[0]:.3f}  k={sol.x[1]:.3f}  '
                      f'cost={sol.cost:.2e}')
        except Exception as e:
            if verbose:
                print(f'  ✗ λ={wl:.0f}nm 失敗: {e}')
            result[i] = [n0, k0]

    return result


# =============================================================================
# 自測：python -m ellipsometry.core.tmm_calc
# =============================================================================

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from ellipsometry.core.dispersion import MaterialLibrary, DrudeLorentz
    import numpy as np

    wl = np.arange(300, 2001, 50, dtype=float)

    # 測試 1：bare Si
    print('=== Test 1: bare Si wafer ===')
    layers = [
        Layer('air', MaterialLibrary.get('air'), np.inf, coherent=False),
        Layer('Si',  MaterialLibrary.get('Si'),  1e6,    coherent=False),
    ]
    res = calculate(layers, wl, [65, 70, 75])
    print(f'  Ψ shape: {res.psi.shape}')
    print(f'  Ψ @ 500nm AOI=70°: {res.psi[wl==500, 1][0]:.3f} deg')
    print(f'  Δ @ 500nm AOI=70°: {res.delta[wl==500, 1][0]:.3f} deg')
    print(f'  T @ 1500nm: {res.T[wl==1500][0]:.4f} (應 > 0, Si 在 IR 透光)')
    print(f'  T @ 500nm:  {res.T[wl==500][0]:.4f} (應 ≈ 0, Si 吸光)')

    # 測試 2：50 nm Au / sapphire
    print('\n=== Test 2: 50 nm Au / sapphire ===')
    layers = [
        Layer('air',      MaterialLibrary.get('air'),   np.inf, coherent=False),
        Layer('Au',       MaterialLibrary.get('Au'),    50.0,   coherent=True),
        Layer('sapphire', MaterialLibrary.get('Al2O3'), 1e6,    coherent=False),
    ]
    res = calculate(layers, wl, [65, 70, 75])
    print(f'  Ψ範圍: [{res.psi.min():.2f}, {res.psi.max():.2f}] deg')
    print(f'  Δ範圍: [{res.delta.min():.2f}, {res.delta.max():.2f}] deg')
    print(f'  T 範圍: [{res.T.min():.4f}, {res.T.max():.4f}]')

    # 測試 3：與 fake data 對比
    print('\n=== Test 3: 對比 fake data (應一致或極接近) ===')
    from ellipsometry.core.io import read_reflection
    truth = read_reflection('data/sample1_Au50_sapphire/R_long.dat')
    wl_full = truth.wavelength
    layers_full = [
        Layer('air',      MaterialLibrary.get('air'),   np.inf, coherent=False),
        Layer('Au',       MaterialLibrary.get('Au'),    50.0,   coherent=True),
        Layer('sapphire', MaterialLibrary.get('Al2O3'), 1e6,    coherent=False),
    ]
    res_full = calculate(layers_full, wl_full, truth.angles)
    diff_psi = np.abs(res_full.psi - truth.psi).max()
    diff_delta = np.abs(((res_full.delta - truth.delta + 180) % 360) - 180).max()
    print(f'  Ψ max diff: {diff_psi:.4f} deg')
    print(f'  Δ max diff (wrap-aware): {diff_delta:.4f} deg')

    # 測試 4：pseudo_epsilon 的數學恆等式（只對半無限基板成立）
    print('\n=== Test 4: pseudo_epsilon @ 半無限 Si（應等於真實 ε）===')
    res2 = calculate(
        [Layer('air', MaterialLibrary.get('air'), np.inf, coherent=False),
         Layer('Si',  MaterialLibrary.get('Si'),  np.inf, coherent=False)],   # ← np.inf
        np.array([500.0]), [70.0],
    )
    eps_pseudo = pseudo_epsilon(res2.psi[0, 0], res2.delta[0, 0], 70.0)
    eps_true = MaterialLibrary.get('Si').epsilon(np.array([500.0]))[0]
    print(f'  <ε> pseudo: {eps_pseudo:.4f}')
    print(f'  ε true:    {eps_true:.4f}')
    print(f'  Δε:        {abs(eps_pseudo - eps_true):.4e}  (應 ≈ 0)')
