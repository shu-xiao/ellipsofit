"""
色散模型 (Dispersion Models)

實作模型清單：
    Pointwise          — 從 csv 插值（內建材料 Si, SiO2, Au... 用這個）
    Constant           — 固定 n + ik
    Cauchy             — 透明介電 n = A + B/λ² + C/λ⁴
    Sellmeier          — 寬頻透明 n² = 1 + Σ Bᵢλ²/(λ²-Cᵢ²)
    Lorentz            — 單一 Lorentz 振盪器
    Drude              — 純金屬（自由電子）
    DrudeLorentz       — Drude + N 個 Lorentz
    TaucLorentz        — 非晶半導體（含 Egap）
    Gaussian           — KK consistent Gaussian 振盪器
    GenOsc             — WVASE 風格綜合模型
                         (e1_offset + Egap + UV/IR poles + 多種振盪器混用)

所有模型都實作 epsilon(wavelength_nm) → complex ε(λ) 介面。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, CubicSpline

from .units import nm_to_eV, HC_eVnm


# =============================================================================
# 抽象基底類別
# =============================================================================

class DispersionModel(ABC):
    """色散模型的抽象基底

    子類別必須實作 epsilon(wavelength_nm)，回傳複數 ε(λ)
    """

    @abstractmethod
    def epsilon(self, wavelength_nm: np.ndarray) -> np.ndarray:
        """回傳複數介電函數 ε(λ) = ε1 + i·ε2"""

    def n_complex(self, wavelength_nm: np.ndarray) -> np.ndarray:
        """回傳複數折射率 ñ = √ε（取主分支 Im(ñ) ≥ 0）"""
        eps = self.epsilon(wavelength_nm)
        n = np.sqrt(eps)
        # 保證 Im(ñ) ≥ 0
        n = np.where(n.imag < 0, np.conj(n), n)
        return n

    def n_k(self, wavelength_nm: np.ndarray):
        """回傳 (n, k) 兩個實數陣列"""
        nc = self.n_complex(wavelength_nm)
        return nc.real, nc.imag

    def eps1_eps2(self, wavelength_nm: np.ndarray):
        """回傳 (ε1, ε2) 兩個實數陣列"""
        e = self.epsilon(wavelength_nm)
        return e.real, e.imag


# =============================================================================
# 1. Pointwise — 從 csv 插值（內建材料用）
# =============================================================================

class Pointwise(DispersionModel):
    """從 (wavelength_nm, n, k) 表格插值，支援多種插值法

    method:
        'linear' — np.interp（最快，最簡單，但有折角）
        'cubic'  — scipy CubicSpline（平滑但可能過衝）
        'pchip'  — scipy PchipInterpolator（預設，形狀保留，無過衝）

    超出範圍時用最近端值常數外推並警告（一次性）。
    """

    def __init__(self, wavelength_nm, n, k, method: str = 'pchip', name: str = ''):
        wl = np.asarray(wavelength_nm, dtype=float)
        sort = np.argsort(wl)
        self.wl = wl[sort]
        self.n = np.asarray(n, dtype=float)[sort]
        self.k = np.asarray(k, dtype=float)[sort]
        self.method = method.lower()
        self.name = name
        self._warned = False
        self._build_interpolator()

    def _build_interpolator(self):
        if self.method == 'linear':
            self._n_fn = None  # 用 np.interp
            self._k_fn = None
        elif self.method == 'cubic':
            self._n_fn = CubicSpline(self.wl, self.n, bc_type='natural', extrapolate=False)
            self._k_fn = CubicSpline(self.wl, self.k, bc_type='natural', extrapolate=False)
        elif self.method == 'pchip':
            self._n_fn = PchipInterpolator(self.wl, self.n, extrapolate=False)
            self._k_fn = PchipInterpolator(self.wl, self.k, extrapolate=False)
        else:
            raise ValueError(f'未知插值法: {self.method} (linear/cubic/pchip)')

    @classmethod
    def from_csv(cls, path: Union[str, Path], method: str = 'pchip',
                 name: str = '') -> 'Pointwise':
        df = pd.read_csv(path)
        return cls(df['wavelength_nm'], df['n'], df['k'],
                   method=method, name=name or Path(path).stem)

    def _interp(self, wl, fn, data):
        """執行插值，超範圍用最近端值外推"""
        if fn is None:  # linear
            return np.interp(wl, self.wl, data)
        out = fn(wl)
        # PCHIP / CubicSpline 超範圍會傳 NaN → 用端點值補
        if np.any(np.isnan(out)):
            below = wl < self.wl[0]
            above = wl > self.wl[-1]
            out = np.where(below, data[0], out)
            out = np.where(above, data[-1], out)
        return out

    def epsilon(self, wavelength_nm: np.ndarray) -> np.ndarray:
        wl = np.asarray(wavelength_nm, dtype=float)
        if not self._warned and (wl.min() < self.wl.min() or wl.max() > self.wl.max()):
            print(f'  ⚠️  Pointwise[{self.name}]: 請求 [{wl.min():.0f},{wl.max():.0f}] '
                  f'超出資料 [{self.wl.min():.0f},{self.wl.max():.0f}], 端點外推')
            self._warned = True
        n = self._interp(wl, self._n_fn, self.n)
        k = self._interp(wl, self._k_fn, self.k)
        nc = n + 1j * k
        return nc * nc


# =============================================================================
# 1b. PolynomialNK — n(E), k(E) 多項式（係數可參與 fit）
# =============================================================================

@dataclass
class PolynomialNK(DispersionModel):
    """n(E), k(E) 各為 E (eV) 的多項式

        n(E) = a0 + a1·E + a2·E² + ... + a_N·E^N
        k(E) = b0 + b1·E + b2·E² + ... + b_M·E^M

    係數可參與擬合（透過 fitter 設定）。

    用途：
        1. 從現有 Pointwise 材料抽出近似（fit_from_data），減少 fit 參數量
        2. 直接給高次多項式作為 nk 模型，係數全部可調

    domain_eV: 多項式在這個 eV 範圍訓練/有效
    """
    n_coeffs: list = field(default_factory=lambda: [1.5])
    k_coeffs: list = field(default_factory=lambda: [0.0])
    domain_eV: tuple = (0.5, 6.0)

    def epsilon(self, wavelength_nm: np.ndarray) -> np.ndarray:
        E = nm_to_eV(wavelength_nm)
        # 多項式由低次到高次評估
        n = np.polynomial.polynomial.polyval(E, self.n_coeffs)
        k = np.polynomial.polynomial.polyval(E, self.k_coeffs)
        # 避免 k < 0
        k = np.maximum(k, 0.0)
        nc = n + 1j * k
        return nc * nc

    @classmethod
    def fit_from_data(cls, wavelength_nm, n_data, k_data,
                      n_degree: int = 4, k_degree: Optional[int] = None,
                      target_mse: Optional[float] = None,
                      max_degree: int = 12, verbose: bool = True) -> 'PolynomialNK':
        """從現有 n, k 資料擬合多項式（WVASE-style）

        兩種模式：
          1. 指定 degree：固定階數做最小平方擬合
          2. 指定 target_mse：從 degree=1 開始增加，直到 MSE ≤ target 或達 max_degree

        Parameters
        ----------
        wavelength_nm, n_data, k_data : array
        n_degree : int
            n 的多項式階數（target_mse 給時當作起始）
        k_degree : int or None
            k 的階數，None=同 n_degree
        target_mse : float or None
            目標 MSE = √(<(n_fit-n)² + (k_fit-k)²>)，None=不自動
        max_degree : int
            自動模式的最大階數上限
        """
        if k_degree is None:
            k_degree = n_degree

        wl = np.asarray(wavelength_nm, dtype=float)
        n_d = np.asarray(n_data, dtype=float)
        k_d = np.asarray(k_data, dtype=float)
        E = nm_to_eV(wl)

        def fit_with_degree(deg_n, deg_k):
            # 用 numpy.polynomial.polynomial.polyfit (低次到高次)
            cn = np.polynomial.polynomial.polyfit(E, n_d, deg_n)
            ck = np.polynomial.polynomial.polyfit(E, k_d, deg_k)
            n_fit = np.polynomial.polynomial.polyval(E, cn)
            k_fit = np.polynomial.polynomial.polyval(E, ck)
            mse = float(np.sqrt(np.mean((n_fit - n_d)**2 + (k_fit - k_d)**2)))
            return cn.tolist(), ck.tolist(), mse

        if target_mse is None:
            # 模式 1：固定 degree
            cn, ck, mse = fit_with_degree(n_degree, k_degree)
            if verbose:
                print(f'  PolynomialNK fit: n_deg={n_degree}, k_deg={k_degree}, MSE={mse:.4e}')
        else:
            # 模式 2：自動增 degree 直到達標
            best = None
            for deg in range(max(1, n_degree), max_degree + 1):
                cn, ck, mse = fit_with_degree(deg, deg)
                if verbose:
                    print(f'  PolynomialNK try deg={deg:2d}: MSE={mse:.4e}  '
                          f'{"✓ 達標" if mse <= target_mse else ""}')
                best = (cn, ck, mse, deg)
                if mse <= target_mse:
                    break
            cn, ck, mse, deg = best
            if mse > target_mse:
                print(f'  ⚠️  未達 target MSE={target_mse:.2e}，最終 deg={deg} MSE={mse:.4e}')

        return cls(
            n_coeffs=cn, k_coeffs=ck,
            domain_eV=(float(E.min()), float(E.max())),
        )


# =============================================================================
# 2. Constant — 固定 n + ik
# =============================================================================

@dataclass
class Constant(DispersionModel):
    n: float = 1.0
    k: float = 0.0

    def epsilon(self, wavelength_nm: np.ndarray) -> np.ndarray:
        nc = self.n + 1j * self.k
        return np.full_like(np.asarray(wavelength_nm, dtype=float), nc * nc, dtype=complex)


# =============================================================================
# 3. Cauchy — 透明介電
# =============================================================================

@dataclass
class Cauchy(DispersionModel):
    """n(λ) = A + B/λ² + C/λ⁴   (λ 單位 μm)

    k = k0·exp(B_k·(1/λ - 1/λ_edge))  — extension band (預設關閉 k=0)
    """
    A: float = 1.5
    B: float = 0.01
    C: float = 0.0
    # extension band (吸收尾)
    k0: float = 0.0
    B_k: float = 0.0
    lambda_edge_nm: float = 400.0

    def epsilon(self, wavelength_nm: np.ndarray) -> np.ndarray:
        wl_um = np.asarray(wavelength_nm, dtype=float) / 1000.0
        n = self.A + self.B / wl_um**2 + self.C / wl_um**4
        if self.k0 > 0:
            lam_edge_um = self.lambda_edge_nm / 1000.0
            k = self.k0 * np.exp(self.B_k * (1 / wl_um - 1 / lam_edge_um))
        else:
            k = np.zeros_like(wl_um)
        nc = n + 1j * k
        return nc * nc


# =============================================================================
# 4. Sellmeier — 寬頻透明
# =============================================================================

@dataclass
class Sellmeier(DispersionModel):
    """n²(λ) = 1 + Σᵢ Bᵢ λ² / (λ² - Cᵢ²)   (λ 單位 μm)

    coefficients: [B1, C1, B2, C2, B3, C3, ...] 列表
    """
    coefficients: list = field(default_factory=list)

    def epsilon(self, wavelength_nm: np.ndarray) -> np.ndarray:
        wl_um = np.asarray(wavelength_nm, dtype=float) / 1000.0
        n_sq = np.ones_like(wl_um)
        for i in range(0, len(self.coefficients), 2):
            B = self.coefficients[i]
            C = self.coefficients[i + 1]
            n_sq += B * wl_um**2 / (wl_um**2 - C**2)
        n = np.sqrt(np.maximum(n_sq, 0))
        return (n + 0j)**2


# =============================================================================
# 5. Lorentz 振盪器（KK consistent）
# =============================================================================

def _lorentz_eps(E_eV: np.ndarray, A: float, E0: float, gamma: float) -> np.ndarray:
    """單一 Lorentz 振盪器的複數 ε 貢獻

    ε_L(E) = A · E0² / (E0² - E² - iγE)

    自動 KK consistent（複數一行算完）
    """
    return A * E0**2 / (E0**2 - E_eV**2 - 1j * gamma * E_eV)


@dataclass
class Lorentz(DispersionModel):
    """單一 Lorentz：ε = ε∞ + Σ A·E0²/(E0²-E²-iγE)

    oscillators: list of (A, E0, gamma) tuples 或 dict
    """
    eps_inf: float = 1.0
    oscillators: list = field(default_factory=list)

    def epsilon(self, wavelength_nm: np.ndarray) -> np.ndarray:
        E = nm_to_eV(wavelength_nm)
        eps = np.full_like(E, self.eps_inf, dtype=complex)
        for osc in self.oscillators:
            A, E0, g = _osc_unpack(osc)
            eps += _lorentz_eps(E, A, E0, g)
        return eps


# =============================================================================
# 6. Drude — 純金屬
# =============================================================================

def _drude_eps(E_eV: np.ndarray, omega_p: float, gamma: float) -> np.ndarray:
    """Drude 自由電子貢獻 (ωp, Γ) 形式

    ε_D(E) = -ωₚ² / (E² + iγE)
    """
    return -omega_p**2 / (E_eV**2 + 1j * gamma * E_eV)


# 物理常數（與 (ρ, τ) ↔ (ωp, Γ) 換算用）
_HBAR_eV_s = 6.582119569e-16   # eV·s
_EPS0      = 8.854187817e-12   # F/m


def drude_rho_tau_to_omegap_gamma(rho_ohm_cm: float, tau_fs: float) -> tuple:
    """Drude (resistivity ρ, scattering time τ) → (ωp, Γ) (WVASE 替代形式)

    公式：
        Γ  = ℏ/τ
        ωp² = 1/(ε₀·ρ·τ)             （SI 單位內部換算）
        ℏωp = √(ℏ²/(ε₀·ρ·τ))         （回到 eV）

    輸入：
        ρ  in Ω·cm  （典型金屬 1e-6 - 1e-3；半導體 0.01 - 1000）
        τ  in fs    （典型金屬 5-50；半導體 1-100）

    輸出：
        ωp (eV), Γ (eV)

    驗證（Au @ RT）：ρ≈2.2e-6, τ≈30 → ωp≈8.6 eV, Γ≈0.022 eV ✓
    """
    rho_SI = rho_ohm_cm * 0.01      # Ω·m
    tau_SI = tau_fs * 1e-15         # s
    gamma = _HBAR_eV_s / tau_SI
    omega_p_sq_rad2 = 1.0 / (_EPS0 * rho_SI * tau_SI)
    omega_p_eV = _HBAR_eV_s * np.sqrt(omega_p_sq_rad2)
    return omega_p_eV, gamma


def _drude_eps_rho_tau(E_eV: np.ndarray, rho_ohm_cm: float, tau_fs: float) -> np.ndarray:
    """Drude (ρ, τ) 形式直接算 ε（內部轉成 ωp/Γ 再算）"""
    omega_p, gamma = drude_rho_tau_to_omegap_gamma(rho_ohm_cm, tau_fs)
    return _drude_eps(E_eV, omega_p, gamma)


@dataclass
class Drude(DispersionModel):
    """ε = ε∞ - ωₚ²/(E²+iΓE)"""
    eps_inf: float = 1.0
    omega_p: float = 8.0       # eV
    gamma: float = 0.05        # eV

    def epsilon(self, wavelength_nm: np.ndarray) -> np.ndarray:
        E = nm_to_eV(wavelength_nm)
        return self.eps_inf + _drude_eps(E, self.omega_p, self.gamma)


# =============================================================================
# 7. Drude-Lorentz — Drude + N 個 Lorentz
# =============================================================================

@dataclass
class DrudeLorentz(DispersionModel):
    """ε = ε∞ + Drude + Σ Lorentz

    drude: {'omega_p': ..., 'gamma': ...}   或 None 表示不含 Drude
    lorentz: [{'A':..., 'E0':..., 'gamma':...}, ...]
    """
    eps_inf: float = 1.0
    drude: Optional[dict] = None
    lorentz: list = field(default_factory=list)

    def epsilon(self, wavelength_nm: np.ndarray) -> np.ndarray:
        E = nm_to_eV(wavelength_nm)
        eps = np.full_like(E, self.eps_inf, dtype=complex)
        if self.drude:
            eps += _drude_eps(E, self.drude['omega_p'], self.drude['gamma'])
        for osc in self.lorentz:
            A, E0, g = _osc_unpack(osc)
            eps += _lorentz_eps(E, A, E0, g)
        return eps


# =============================================================================
# 8. Tauc-Lorentz — 非晶半導體
# =============================================================================

def _tauc_lorentz_eps2(E_eV: np.ndarray, A: float, E0: float, C: float, Eg: float) -> np.ndarray:
    """Tauc-Lorentz ε2:
        ε2(E) = A·E0·C·(E-Eg)² / [E·((E²-E0²)² + C²·E²)]   for E > Eg
        ε2(E) = 0                                            for E ≤ Eg
    """
    e2 = np.zeros_like(E_eV)
    mask = E_eV > Eg
    Em = E_eV[mask]
    numer = A * E0 * C * (Em - Eg)**2
    denom = Em * ((Em**2 - E0**2)**2 + C**2 * Em**2)
    e2[mask] = numer / denom
    return e2


def _tauc_lorentz_eps1_analytic(E_eV: np.ndarray, A: float, E0: float, C: float, Eg: float) -> np.ndarray:
    """Tauc-Lorentz ε1（解析式，Jellison & Modine 1996）

    分 4 個輔助項加總。詳見 J. Appl. Phys. 69, 371 (1992) 與後續修正。
    """
    E = E_eV
    aln = (Eg**2 - E0**2) * E**2 + Eg**2 * C**2 - E0**2 * (E0**2 + 3 * Eg**2)
    aatan = (E**2 - E0**2) * (E0**2 + Eg**2) + Eg**2 * C**2
    alpha = np.sqrt(np.maximum(4 * E0**2 - C**2, 1e-12))
    gamma = np.sqrt(np.maximum(E0**2 - C**2 / 2, 1e-12))
    zeta4 = (E**2 - gamma**2)**2 + (alpha**2) * (C**2) / 4

    safe_log_arg = np.maximum(
        (E0**2 + Eg**2 + alpha * Eg) / np.maximum(E0**2 + Eg**2 - alpha * Eg, 1e-30),
        1e-30,
    )

    term1 = (A * C * aln) / (2 * np.pi * zeta4 * alpha * E0) * np.log(safe_log_arg)
    term2 = -(A * aatan) / (np.pi * zeta4 * E0) * (
        np.pi - np.arctan((2 * Eg + alpha) / C) + np.arctan((-2 * Eg + alpha) / C)
    )
    term3 = (2 * A * E0 * Eg * (E**2 - gamma**2)) / (np.pi * zeta4 * alpha) * (
        np.pi + 2 * np.arctan(2 * (gamma**2 - Eg**2) / (alpha * C))
    )
    # term4 log of safe ratio
    safe4_num = (E**2 - E0**2)**2 + (C**2 * E**2)
    safe4_den = np.maximum((Eg**2 - E0**2)**2 + (C**2 * Eg**2), 1e-30)
    term4 = -(A * E0 * C * (E**2 + Eg**2)) / (np.pi * zeta4 * E) * np.log(
        np.maximum(safe4_num / safe4_den, 1e-30)
    )
    # term5
    term5 = (2 * A * E0 * C * Eg) / (np.pi * zeta4) * np.log(
        np.maximum(np.abs(E - Eg) * (E + Eg) /
                   np.sqrt(np.maximum((E0**2 - Eg**2)**2 + Eg**2 * C**2, 1e-30)),
                   1e-30)
    )
    return term1 + term2 + term3 + term4 + term5


@dataclass
class TaucLorentz(DispersionModel):
    """Tauc-Lorentz：非晶半導體（a-Si, a-SiN）

    每個 oscillator: {'A':..., 'E0':..., 'C':..., 'Eg':...}
    (注意 Tauc-Lorentz 的廣度叫 C 不叫 gamma)
    """
    eps_inf: float = 1.0
    Eg: float = 1.0                # 全模型的 bandgap (eV)
    oscillators: list = field(default_factory=list)

    def epsilon(self, wavelength_nm: np.ndarray) -> np.ndarray:
        E = nm_to_eV(wavelength_nm)
        e1 = np.full_like(E, self.eps_inf)
        e2 = np.zeros_like(E)
        for osc in self.oscillators:
            A = osc['A']
            E0 = osc['E0']
            C = osc.get('C', osc.get('gamma', 0.5))
            Eg = osc.get('Eg', self.Eg)
            e1 += _tauc_lorentz_eps1_analytic(E, A, E0, C, Eg)
            e2 += _tauc_lorentz_eps2(E, A, E0, C, Eg)
        return e1 + 1j * e2


# =============================================================================
# 9. Gaussian 振盪器（KK consistent，用 Hilbert transform 數值算 ε1）
# =============================================================================

def _gaussian_eps2(E_eV: np.ndarray, A: float, E0: float, Br: float) -> np.ndarray:
    """Gaussian ε2 (兩個 peak: ±E0)，broadening Br = FWHM

    ε2(E) = A·[exp(-((E-E0)/σ)²) - exp(-((E+E0)/σ)²)]
    σ = Br / (2·√ln2)
    """
    sigma = Br / (2 * np.sqrt(np.log(2)))
    return A * (np.exp(-((E_eV - E0) / sigma)**2) -
                np.exp(-((E_eV + E0) / sigma)**2))


def _kk_eps1_from_eps2(E_grid: np.ndarray, eps2: np.ndarray) -> np.ndarray:
    """Kramers-Kronig 從 ε2 算 ε1（含主值積分）

    ε1(E) = 1 + (2/π) · P∫₀^∞ E'·ε2(E')/(E'²-E²) dE'

    用離散版本（梯形法 + 跳過奇異點）
    """
    e1 = np.zeros_like(E_grid)
    n = len(E_grid)
    for i, E in enumerate(E_grid):
        # 跳過奇異點 i 本身
        mask = np.arange(n) != i
        Ep = E_grid[mask]
        e2 = eps2[mask]
        integrand = Ep * e2 / (Ep**2 - E**2)
        # 梯形積分
        e1[i] = np.trapezoid(integrand, Ep) * (2 / np.pi)
    return e1


@dataclass
class Gaussian(DispersionModel):
    """Gaussian 振盪器（用數值 KK 算 ε1，效能較差但結構通用）

    oscillators: [{'A':..., 'E0':..., 'Br':...}, ...]
    """
    eps_inf: float = 1.0
    oscillators: list = field(default_factory=list)
    # KK 積分用的隱含 E grid 範圍與密度
    kk_E_min: float = 0.01
    kk_E_max: float = 30.0
    kk_n_grid: int = 2000

    def epsilon(self, wavelength_nm: np.ndarray) -> np.ndarray:
        E_query = nm_to_eV(wavelength_nm)
        # 在統一格點上算 ε2，再做 KK
        E_grid = np.linspace(self.kk_E_min, self.kk_E_max, self.kk_n_grid)
        e2_grid = np.zeros_like(E_grid)
        for osc in self.oscillators:
            A, E0, Br = _osc_unpack(osc)
            e2_grid += _gaussian_eps2(E_grid, A, E0, Br)
        e1_grid = _kk_eps1_from_eps2(E_grid, e2_grid) + self.eps_inf
        # 插值到查詢點
        e1 = np.interp(E_query, E_grid, e1_grid)
        e2 = np.interp(E_query, E_grid, e2_grid)
        return e1 + 1j * e2


# =============================================================================
# 10. GenOsc — WVASE 風格綜合模型
# =============================================================================

@dataclass
class Pole:
    """UV / IR pole — 只對 ε1 有貢獻（無吸收）

    ε1_pole(E) = Mag·E_pole² / (E_pole² - E²)
    """
    position: float       # eV
    magnitude: float

    def eps1_contrib(self, E_eV: np.ndarray) -> np.ndarray:
        return self.magnitude * self.position**2 / (self.position**2 - E_eV**2)


@dataclass
class GenOscOscillator:
    """單一振盪器（GenOsc 用）

    參數依類型而異（對應 WVASE GenOsc style strings）：
        lorentz / gaussian / harmonic:  (amp, en, br)  3 參數
        drude:        (amp, br)             2 參數，無 en（共振在 E=0）
                                            amp = ωp² (eV²), br = Γ (eV)
        drude_rt:     (rho, tau)            WVASE 替代，rho [Ω·cm], tau [fs]
        tauc_lorentz: (amp, en, br, Eg)     4 參數

    無關參數會被忽略（不影響計算）。
    """
    type: str             # 'lorentz' / 'gaussian' / 'drude' / 'drude_rt' / 'tauc_lorentz' / 'harmonic'
    amp: float = 0.0      # Amp（drude_rt 不用）
    en: float = 0.0       # En (eV)（Drude/Drude_rt 不用）
    br: float = 0.0       # Br (eV)（drude_rt 不用）
    rho: float = 0.0      # Resistivity (Ω·cm)，僅 drude_rt
    tau: float = 0.0      # Scattering time (fs)，僅 drude_rt
    active: bool = True
    Eg: float = 0.0       # 僅 Tauc-Lorentz


# 每種振盪器需要的參數（GUI 與驗證用）
OSC_PARAMS = {
    'lorentz':       ['amp', 'en', 'br'],
    'gaussian':      ['amp', 'en', 'br'],
    'harmonic':      ['amp', 'en', 'br'],
    'drude':         ['amp', 'br'],          # ⚠️ Drude (ωp²,Γ) 無 En
    'drude_rt':      ['rho', 'tau'],         # ⚠️ Drude (ρ,τ) 形式（WVASE 替代）
    'tauc_lorentz':  ['amp', 'en', 'br', 'Eg'],
}

# 參數標籤（GUI 顯示用）：(顯示名稱, 單位, 物理意義)
PARAM_LABELS_GENERIC = {
    'amp': ('Amp', '',   'Amplitude (振幅)'),
    'en':  ('En',  'eV', 'Center energy (中心能量)'),
    'br':  ('Br',  'eV', 'Broadening FWHM (展寬)'),
    'Eg':  ('Eg',  'eV', 'Tauc bandgap (能隙)'),
}

# Drude (ωp², Γ) 專用標籤
PARAM_LABELS_DRUDE = {
    'amp': ('Amp', 'eV²', 'ωp² plasma frequency squared'),
    'br':  ('Br',  'eV',  'Γ damping (Drude 阻尼)'),
}

# Drude (ρ, τ) 形式（WVASE 替代參數化）
PARAM_LABELS_DRUDE_RT = {
    'rho': ('ρ',   'Ω·cm', 'Resistivity (電阻率, Au≈2.2e-6)'),
    'tau': ('τ',   'fs',   'Scattering time (散射時間, Au≈30)'),
}


def osc_param_labels(osc_type: str) -> dict:
    """取得振盪器類型對應的參數標籤 dict"""
    if osc_type == 'drude':
        return PARAM_LABELS_DRUDE
    if osc_type == 'drude_rt':
        return PARAM_LABELS_DRUDE_RT
    return PARAM_LABELS_GENERIC


@dataclass
class GenOsc(DispersionModel):
    """WVASE-style General Oscillator Layer

    結構：ε(E) = e1_offset + Σ poles + Σ oscillators
    """
    e1_offset: float = 1.0
    Egap: float = 0.0     # 預設 bandgap (給 Tauc-Lorentz 用)
    uv_pole: Optional[Pole] = None
    ir_pole: Optional[Pole] = None
    oscillators: list = field(default_factory=list)
    # KK 數值積分設定（給 Gaussian 振盪器用）
    kk_E_min: float = 0.01
    kk_E_max: float = 30.0
    kk_n_grid: int = 2000

    def epsilon(self, wavelength_nm: np.ndarray) -> np.ndarray:
        E_query = nm_to_eV(wavelength_nm)

        # 統一 E_grid 算所有振盪器與 poles 的 ε
        E_grid = np.linspace(self.kk_E_min, self.kk_E_max, self.kk_n_grid)
        e1_grid = np.full_like(E_grid, self.e1_offset)
        e2_grid = np.zeros_like(E_grid)

        # ---- Poles (only ε1) ----
        if self.uv_pole:
            e1_grid += self.uv_pole.eps1_contrib(E_grid)
        if self.ir_pole:
            e1_grid += self.ir_pole.eps1_contrib(E_grid)

        # ---- Oscillators ----
        gaussian_present = False
        for osc in self.oscillators:
            if not osc.active:
                continue
            t = osc.type.lower()
            if t == 'lorentz':
                eps_L = _lorentz_eps(E_grid, osc.amp, osc.en, osc.br)
                e1_grid += eps_L.real
                e2_grid += eps_L.imag
            elif t == 'drude':
                # Drude 在 GenOsc 內以 (Amp, Br) = (ωₚ², Γ) 詮釋（無 En）
                eps_D = _drude_eps(E_grid, np.sqrt(max(osc.amp, 0)), osc.br)
                e1_grid += eps_D.real
                e2_grid += eps_D.imag
            elif t == 'drude_rt':
                # Drude (ρ, τ) 形式 — WVASE 替代參數化
                eps_D = _drude_eps_rho_tau(E_grid, osc.rho, osc.tau)
                e1_grid += eps_D.real
                e2_grid += eps_D.imag
            elif t == 'gaussian':
                gaussian_present = True
                e2_grid += _gaussian_eps2(E_grid, osc.amp, osc.en, osc.br)
            elif t == 'tauc_lorentz':
                Eg = osc.Eg if osc.Eg > 0 else self.Egap
                e2_grid += _tauc_lorentz_eps2(E_grid, osc.amp, osc.en, osc.br, Eg)
                e1_grid += _tauc_lorentz_eps1_analytic(E_grid, osc.amp, osc.en, osc.br, Eg)
            elif t == 'harmonic':
                # Harmonic 同 Lorentz (γ 改名)
                eps_H = _lorentz_eps(E_grid, osc.amp, osc.en, osc.br)
                e1_grid += eps_H.real
                e2_grid += eps_H.imag
            else:
                raise ValueError(f'未知振盪器類型: {osc.type}')

        # Gaussian / TaucLorentz 需要把 ε2 → ε1 補 KK（這裡簡化：只對 Gaussian 補）
        # （TaucLorentz 已用解析式）
        if gaussian_present:
            # 抽出 gaussian 的 ε2 貢獻，做 KK
            e2_gauss = np.zeros_like(E_grid)
            for osc in self.oscillators:
                if osc.active and osc.type.lower() == 'gaussian':
                    e2_gauss += _gaussian_eps2(E_grid, osc.amp, osc.en, osc.br)
            e1_grid += _kk_eps1_from_eps2(E_grid, e2_gauss)

        # 插值到查詢點
        e1 = np.interp(E_query, E_grid, e1_grid)
        e2 = np.interp(E_query, E_grid, e2_grid)
        return e1 + 1j * e2


# =============================================================================
# 振盪器參數打包輔助
# =============================================================================

def _osc_unpack(osc):
    """支援多種寫法：tuple/list (A, E0, Br) 或 dict {'A','E0','gamma'/'Br'/'br'}"""
    if isinstance(osc, (list, tuple)):
        A, E0, g = osc[:3]
    elif isinstance(osc, dict):
        A = osc.get('A', osc.get('amp'))
        E0 = osc.get('E0', osc.get('en'))
        g = osc.get('gamma', osc.get('Br', osc.get('br')))
    else:
        raise TypeError(f'unknown oscillator format: {osc}')
    return float(A), float(E0), float(g)


# =============================================================================
# 工廠函式：從 config dict 建模型
# =============================================================================

def create_dispersion(spec) -> DispersionModel:
    """從 config 的 material 欄位建模型

    支援兩種寫法：
      1) 字串：'Si' / 'Au_JC' / ...  → 從 materials/ 載入 Pointwise
      2) dict：{'model': 'cauchy', 'params': {...}}
    """
    if isinstance(spec, str):
        return MaterialLibrary.get(spec)

    if not isinstance(spec, dict):
        raise TypeError(f'material 必須是 str 或 dict，收到 {type(spec)}')

    model = spec.get('model', '').lower()
    p = spec.get('params', {})

    if model == 'pointwise':
        # 從 csv 載入並指定插值法
        source = spec.get('source') or p.get('source')
        method = p.get('method', 'pchip')
        if source:
            return MaterialLibrary.get(source, method=method)
        # 或直接提供 nk 表
        return Pointwise(
            p['wavelength_nm'], p['n'], p['k'],
            method=method, name=p.get('name', 'custom'),
        )
    if model == 'polynomial_nk':
        # 模式 A: 直接給 coeffs
        if 'n_coeffs' in p:
            return PolynomialNK(
                n_coeffs=p['n_coeffs'], k_coeffs=p.get('k_coeffs', [0.0]),
                domain_eV=tuple(p.get('domain_eV', (0.5, 6.0))),
            )
        # 模式 B: 從現有材料 fit
        if 'fit_from' in p:
            base = MaterialLibrary.get(p['fit_from'])
            assert isinstance(base, Pointwise), f'fit_from 只能是 Pointwise 材料'
            return PolynomialNK.fit_from_data(
                base.wl, base.n, base.k,
                n_degree=p.get('n_degree', 4),
                k_degree=p.get('k_degree'),
                target_mse=p.get('target_mse'),
                max_degree=p.get('max_degree', 12),
                verbose=p.get('verbose', True),
            )
        raise ValueError('polynomial_nk 需要 n_coeffs 或 fit_from')
    if model == 'constant':
        return Constant(n=p.get('n', 1.0), k=p.get('k', 0.0))
    if model == 'cauchy':
        return Cauchy(**p)
    if model == 'sellmeier':
        return Sellmeier(coefficients=p.get('coefficients', []))
    if model == 'lorentz':
        return Lorentz(eps_inf=p.get('eps_inf', 1.0),
                       oscillators=p.get('oscillators', []))
    if model == 'drude':
        return Drude(eps_inf=p.get('eps_inf', 1.0),
                     omega_p=p.get('omega_p', 8.0),
                     gamma=p.get('gamma', 0.05))
    if model == 'drude_lorentz':
        return DrudeLorentz(eps_inf=p.get('eps_inf', 1.0),
                            drude=p.get('drude'),
                            lorentz=p.get('lorentz', []))
    if model == 'tauc_lorentz':
        return TaucLorentz(eps_inf=p.get('eps_inf', 1.0),
                           Eg=p.get('Eg', 1.0),
                           oscillators=p.get('oscillators', []))
    if model == 'gaussian':
        return Gaussian(eps_inf=p.get('eps_inf', 1.0),
                        oscillators=p.get('oscillators', []))
    if model == 'gen_osc':
        # 解析 poles 與 oscillators
        layer = spec.get('layer', {})
        oscs_raw = spec.get('oscillators', [])
        poles_raw = layer.get('poles', {})

        uv = Pole(**poles_raw['uv']) if 'uv' in poles_raw else None
        ir = Pole(**poles_raw['ir']) if 'ir' in poles_raw else None

        oscs = []
        for o in oscs_raw:
            # 用 .get 避免 KeyError（每類型需要的欄位不同：Drude 無 en，
            # drude_rt 用 rho/tau 而非 amp/en/br）
            oscs.append(GenOscOscillator(
                type=o['type'],
                amp=o.get('amp', 0.0),
                en=o.get('en', 0.0),
                br=o.get('br', 0.0),
                rho=o.get('rho', 0.0),
                tau=o.get('tau', 0.0),
                active=o.get('active', True),
                Eg=o.get('Eg', 0.0),
            ))
        return GenOsc(
            e1_offset=layer.get('e1_offset', 1.0),
            Egap=layer.get('egap', 0.0),
            uv_pole=uv, ir_pole=ir,
            oscillators=oscs,
        )
    raise ValueError(f'未知色散模型: {model}')


# =============================================================================
# 內建材料庫
# =============================================================================

class MaterialLibrary:
    """內建材料庫 — 對 materials/*.csv 包裝 Pointwise 模型，含快取

    用法：
        m = MaterialLibrary.get('Si')                  # 預設 PCHIP
        m = MaterialLibrary.get('Au_JC', method='cubic')
        m = MaterialLibrary.get('air')                 # 空氣 (n=1, k=0)
    """
    _MATERIALS_DIR = Path(__file__).resolve().parents[2] / 'materials'
    _cache: dict = {}

    @classmethod
    def get(cls, name: str, method: str = 'pchip') -> DispersionModel:
        key = (name, method)
        if key in cls._cache:
            return cls._cache[key]

        if name == 'air':
            model: DispersionModel = Constant(n=1.0, k=0.0)
        else:
            real_name = 'Al2O3' if name == 'sapphire' else name
            path = cls._MATERIALS_DIR / f'{real_name}.csv'
            if not path.exists():
                raise FileNotFoundError(
                    f'材料 "{name}" 找不到。可用：{cls.list_available()}')
            model = Pointwise.from_csv(path, method=method, name=name)

        cls._cache[key] = model
        return model

    @classmethod
    def list_available(cls) -> list:
        return sorted({p.stem for p in cls._MATERIALS_DIR.glob('*.csv')})


# =============================================================================
# 自測：python -m ellipsometry.core.dispersion
# =============================================================================

if __name__ == '__main__':
    print('=== 內建材料 ===')
    print(' ', MaterialLibrary.list_available())

    # ---- 插值法比較（取 Au_JC，49 點稀疏資料，差異最明顯）----
    print('\n=== 插值法比較：Au_JC @ 555 nm（介於資料點之間）===')
    wl_test = np.array([555.0])
    for method in ['linear', 'cubic', 'pchip']:
        n, k = MaterialLibrary.get('Au_JC', method=method).n_k(wl_test)
        print(f'  {method:7s}: n={n[0]:.5f}  k={k[0]:.5f}')

    # ---- PolynomialNK 自動 degree（從 Au McPeak fit）----
    print('\n=== PolynomialNK: 從 Au 自動 fit 多項式（target MSE=0.05）===')
    au = MaterialLibrary.get('Au')
    poly = PolynomialNK.fit_from_data(
        au.wl, au.n, au.k,
        n_degree=1, target_mse=0.05, max_degree=12,
    )
    print(f'  最終 n_coeffs 階數 = {len(poly.n_coeffs) - 1}')
    print(f'  最終 k_coeffs 階數 = {len(poly.k_coeffs) - 1}')

    # ---- 基本範例 ----
    wl = np.array([300, 500, 800, 1500, 2000.0])
    print(f'\n=== Pointwise: Au @ {wl} nm (PCHIP) ===')
    n, k = MaterialLibrary.get('Au').n_k(wl)
    for w, nn, kk in zip(wl, n, k):
        print(f'  λ={w:6.0f} nm  n={nn:.3f}  k={kk:.3f}')

    print('\n=== Cauchy: A=1.5 B=0.01 C=0 @ 同上 ===')
    n, k = Cauchy(A=1.5, B=0.01, C=0).n_k(wl)
    for w, nn, kk in zip(wl, n, k):
        print(f'  λ={w:6.0f} nm  n={nn:.4f}  k={kk:.4f}')

    print('\n=== Drude-Lorentz (Au-like) @ 同上 ===')
    dl = DrudeLorentz(
        eps_inf=9.84,
        drude={'omega_p': 9.01, 'gamma': 0.072},
        lorentz=[
            {'A': 5.6, 'E0': 4.18, 'gamma': 0.83},   # interband ~ 300 nm
            {'A': 3.1, 'E0': 2.4,  'gamma': 0.43},   # ~ 520 nm
        ],
    )
    n, k = dl.n_k(wl)
    for w, nn, kk in zip(wl, n, k):
        print(f'  λ={w:6.0f} nm  n={nn:.3f}  k={kk:.3f}')

    print('\n=== GenOsc: 1 Lorentz + UV pole ===')
    go = create_dispersion({
        'model': 'gen_osc',
        'layer': {
            'e1_offset': 1.0,
            'poles': {'uv': {'position': 20.0, 'magnitude': 100.0}},
        },
        'oscillators': [
            {'type': 'lorentz', 'amp': 3.0, 'en': 4.5, 'br': 0.3},
        ],
    })
    n, k = go.n_k(wl)
    for w, nn, kk in zip(wl, n, k):
        print(f'  λ={w:6.0f} nm  n={nn:.3f}  k={kk:.3f}')
