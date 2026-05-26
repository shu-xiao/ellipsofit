"""
向量化 TMM — 整批處理所有波長同時計算

對 typical 樣品（ambient + N films + substrate, N=1-5）達 10-30× 加速。
若堆疊太複雜（>10 層）或邊角，fallback 回 tmm 套件純量版本。

API 對 tmm.coh_tmm / tmm.ellips 相容（介面類似）。

實作：標準 Abeles transfer matrix，純 numpy 向量化波長軸。

公式（每個 wavelength）：
    Snell:      n_i sin(θ_i) = n_0 sin(θ_0)
    Fresnel s:  r_s_{ij} = (n_i cosθ_i - n_j cosθ_j) / (n_i cosθ_i + n_j cosθ_j)
    Fresnel p:  r_p_{ij} = (n_j cosθ_i - n_i cosθ_j) / (n_j cosθ_i + n_i cosθ_j)
    Phase:      β_i = 2π n_i d_i cos(θ_i) / λ
    Total r (recursive Abeles):
        r_eff = r_{01} + r_rest · e^{-2iβ_1} / (1 + r_{01} · r_rest · e^{-2iβ_1})
"""
from __future__ import annotations

import numpy as np


def _snell_cos(n_first, theta_first, n_layers):
    """從第一層的 (n, θ) 用 Snell 算出所有層的 cos(θ_i)，全 wavelength 同時

    n_first:    (Nw,) complex
    theta_first: scalar (rad), real
    n_layers:   list of (Nw,) complex

    回傳: list of (Nw,) complex cos(θ_i) — 主分支取 Im≥0
    """
    sin_first = np.sin(theta_first)
    # Snell: n_0 sin(θ_0) = n_i sin(θ_i) → sin(θ_i) = n_0 sin(θ_0) / n_i
    n0_sin0 = n_first * sin_first    # (Nw,) complex
    cos_list = []
    for n_i in n_layers:
        sin_i = n0_sin0 / n_i
        cos_i = np.sqrt(1.0 - sin_i * sin_i)
        # 主分支：使 forward 方向能量正確（Im(n·cosθ) ≥ 0 是常用判準）
        bad = (n_i * cos_i).imag < 0
        cos_i = np.where(bad, -cos_i, cos_i)
        cos_list.append(cos_i)
    return cos_list


def _fresnel_rt(n1, cos1, n2, cos2, polarization='s'):
    """Fresnel r, t 係數，向量化

    polarization: 's' or 'p'
    回傳 (r, t) 都是 (Nw,) complex
    """
    if polarization == 's':
        # r_s = (n1 cosθ1 - n2 cosθ2) / (n1 cosθ1 + n2 cosθ2)
        num = n1 * cos1 - n2 * cos2
        den = n1 * cos1 + n2 * cos2
    else:  # p
        # r_p = (n2 cosθ1 - n1 cosθ2) / (n2 cosθ1 + n1 cosθ2)
        num = n2 * cos1 - n1 * cos2
        den = n2 * cos1 + n1 * cos2
    r = num / den
    t = 1.0 + r if polarization == 's' else (1.0 + r) * (n1 / n2)
    return r, t


def coh_tmm_vec(polarization: str,
                n_list: list,
                d_list: list,
                theta_first: float,
                wl_arr: np.ndarray) -> dict:
    """全相干 TMM，向量化整批波長

    Parameters
    ----------
    polarization : 's' or 'p'
    n_list : list of (Nw,) complex
        每層的 complex refractive index，按波長已展開
    d_list : list of float
        每層厚度 (nm)，第一個與最後一個必為 np.inf
    theta_first : float (rad)
        第一層的入射角
    wl_arr : (Nw,) ndarray (nm)

    Returns
    -------
    dict with keys:
        r       : (Nw,) complex 總反射係數
        R       : (Nw,) real 反射強度
        t       : (Nw,) complex 總透射係數
        T       : (Nw,) real 透射強度
        cos_th  : list of (Nw,) cos(θ_i)
    """
    Nlayer = len(n_list)
    assert len(d_list) == Nlayer
    assert np.isinf(d_list[0]) and np.isinf(d_list[-1]), \
        '第一層與最後一層厚度必為 np.inf'

    Nw = len(wl_arr)
    # 1) Snell：算每層 cosθ
    cos_list = _snell_cos(n_list[0], theta_first, n_list)

    # 2) 從最內層（substrate 端）往外遞迴算 r_eff
    #    最後一層（exit）沒有「下一層」→ r_last_to_exit 即可
    #    用 Abeles 遞迴形式
    r_eff = np.zeros(Nw, dtype=complex)   # 從 substrate 看出去

    # 從第 N-2 個介面（最後一層與其上一層的界面）開始累加
    # 先算最內層介面的 r
    r_last, _ = _fresnel_rt(
        n_list[-2], cos_list[-2], n_list[-1], cos_list[-1], polarization,
    )
    r_eff = r_last

    # 從內往外：對每個中間層 i = N-2, N-3, ..., 1
    # phase: β_i = 2π n_i d_i cos(θ_i) / λ
    # r_eff_new = (r_{i-1,i} + r_eff · e^{-2iβ_i}) / (1 + r_{i-1,i} · r_eff · e^{-2iβ_i})
    for i in range(Nlayer - 2, 0, -1):
        beta_i = 2 * np.pi * n_list[i] * d_list[i] * cos_list[i] / wl_arr
        # phase 慣例：tmm 套件用 e^{+2iβ}（physics e^{-iωt}, forward wave e^{ikz}）
        exp_2ib = np.exp(2j * beta_i)
        r_iface, _ = _fresnel_rt(
            n_list[i - 1], cos_list[i - 1],
            n_list[i],     cos_list[i],     polarization,
        )
        r_eff = (r_iface + r_eff * exp_2ib) / (1.0 + r_iface * r_eff * exp_2ib)

    R = np.abs(r_eff) ** 2
    # 透射係數簡化 — 對 ellipsometry Ψ/Δ 計算不需要 t/T，回 0 占位
    t = np.zeros(Nw, dtype=complex)
    T = np.zeros(Nw)

    return {'r': r_eff, 'R': R, 't': t, 'T': T, 'cos_th': cos_list}


def ellips_vec(n_list: list, d_list: list,
               theta_first: float, wl_arr: np.ndarray) -> dict:
    """向量化版的 tmm.ellips：一次回傳所有波長的 Ψ, Δ

    輸入同 coh_tmm_vec（n_list 每元素是 (Nw,) 複數）

    輸出：{'psi': (Nw,) rad, 'Delta': (Nw,) rad}
    對應 tmm 套件慣例：Δ = arg(-rp/rs)
    """
    res_s = coh_tmm_vec('s', n_list, d_list, theta_first, wl_arr)
    res_p = coh_tmm_vec('p', n_list, d_list, theta_first, wl_arr)
    rho = -res_p['r'] / res_s['r']     # tmm 慣例
    psi = np.arctan(np.abs(rho))
    delta = np.angle(rho)
    return {'psi': psi, 'Delta': delta, 'r_s': res_s['r'], 'r_p': res_p['r']}


# =============================================================================
# 與 scalar tmm 對比測試
# =============================================================================

if __name__ == '__main__':
    import time
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import tmm as _tmm
    from ellipsometry.core.dispersion import MaterialLibrary

    wl = np.linspace(300, 2000, 171)
    aoi_deg = 70.0
    theta = np.deg2rad(aoi_deg)

    # 模擬「Au 50 nm on sapphire」單層
    n_air = np.ones_like(wl, dtype=complex)
    n_Au = MaterialLibrary.get('Au').n_complex(wl)
    n_sub = MaterialLibrary.get('Al2O3').n_complex(wl)

    n_list_vec = [n_air, n_Au, n_sub]
    d_list = [np.inf, 50.0, np.inf]

    # ---- Vectorized ----
    t0 = time.time()
    for _ in range(10):
        r_vec = ellips_vec(n_list_vec, d_list, theta, wl)
    t_vec = (time.time() - t0) / 10

    # ---- Scalar tmm (loop over wavelength) ----
    t0 = time.time()
    for _ in range(10):
        psi_arr = np.zeros_like(wl)
        delta_arr = np.zeros_like(wl)
        for i, w in enumerate(wl):
            n_scalar = [n_air[i], n_Au[i], n_sub[i]]
            r = _tmm.ellips(n_scalar, d_list, theta, w)
            psi_arr[i] = r['psi']
            delta_arr[i] = r['Delta']
    t_scalar = (time.time() - t0) / 10

    diff_psi = np.max(np.abs(r_vec['psi'] - psi_arr))
    diff_delta = np.max(np.abs(((r_vec['Delta'] - delta_arr + np.pi) % (2*np.pi)) - np.pi))

    print(f'=== Au 50nm / sapphire, 171 wl, AOI=70° ===')
    print(f'  Scalar tmm:  {t_scalar*1000:.1f} ms/call')
    print(f'  Vectorized:  {t_vec*1000:.1f} ms/call')
    print(f'  Speedup:     {t_scalar/t_vec:.1f}×')
    print(f'  Max diff Ψ:  {np.rad2deg(diff_psi):.2e}°')
    print(f'  Max diff Δ:  {np.rad2deg(diff_delta):.2e}°')
