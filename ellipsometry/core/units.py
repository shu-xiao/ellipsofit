"""
單位換算工具

慣例：
    - 內部計算永遠用 eV（振盪器模型標準單位）
    - 對外 API 可接受 nm，自動轉換

常數：
    hc = 1239.841984 eV·nm  (h·c, 用於 λ ↔ E 換算)
"""
import numpy as np

# 物理常數（NIST 2019 CODATA）
HC_eVnm = 1239.841984      # h·c in eV·nm


def nm_to_eV(wavelength_nm):
    """波長 (nm) → 光子能量 (eV)"""
    return HC_eVnm / np.asarray(wavelength_nm)


def eV_to_nm(energy_eV):
    """光子能量 (eV) → 波長 (nm)"""
    return HC_eVnm / np.asarray(energy_eV)


def wavelength_convert(value, from_unit: str, to_unit: str):
    """通用波長/能量換算

    支援單位：'nm', 'um', 'angstrom', 'eV', 'cm-1', 'Hz'
    """
    value = np.asarray(value, dtype=float)

    # 全部先轉成 nm
    if from_unit == 'nm':
        wl_nm = value
    elif from_unit == 'um':
        wl_nm = value * 1000
    elif from_unit == 'angstrom':
        wl_nm = value / 10
    elif from_unit == 'eV':
        wl_nm = HC_eVnm / value
    elif from_unit == 'cm-1':
        # 1 cm⁻¹ ≡ 1/(λ in cm) → λ(nm) = 1e7 / (cm⁻¹)
        wl_nm = 1e7 / value
    elif from_unit == 'Hz':
        # c = 299792458 m/s ⇒ λ(nm) = c[m/s] / f * 1e9
        wl_nm = 2.99792458e17 / value
    else:
        raise ValueError(f'未知單位 from_unit={from_unit}')

    # 再從 nm 轉到目標
    if to_unit == 'nm':
        return wl_nm
    if to_unit == 'um':
        return wl_nm / 1000
    if to_unit == 'angstrom':
        return wl_nm * 10
    if to_unit == 'eV':
        return HC_eVnm / wl_nm
    if to_unit == 'cm-1':
        return 1e7 / wl_nm
    if to_unit == 'Hz':
        return 2.99792458e17 / wl_nm
    raise ValueError(f'未知單位 to_unit={to_unit}')
