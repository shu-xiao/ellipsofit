"""
擬合引擎 (Fit Engine)

把 config (yaml) + 量測資料 → 跑 lmfit Levenberg-Marquardt（與 WVASE 一致）→ 結果

支援三種 fit 目標（對應 WVASE GenOsc 的 Fit e1/e2/Both）：
    both     — 直接擬合 Ψ, Δ, T（透過 TMM 正向計算，預設）
    e2_only  — 先 point-by-point 反推 ε，再擬合 ε2（KK consistent 模型，使用者習慣）
    e1_only  — 同上但 fit ε1

支援 WVASE-style weighting：
    sigma_psi, sigma_delta, sigma_T, angle_weights, wavelength_weights

MSE 公式（與 WVASE 對齊）：
    MSE = √(1/(2N-M) · Σ {[(Ψc-Ψm)/σΨ]² + [(Δc-Δm)/σΔ]²})
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
import lmfit

from .io import EllipsometryData, TransmissionData, read_reflection, read_transmission
from .tmm_calc import Layer, calculate, invert_nk_pointbypoint
from .dispersion import (
    DispersionModel, MaterialLibrary, create_dispersion,
    GenOsc, DrudeLorentz, Lorentz, Cauchy, Sellmeier, Drude,
    TaucLorentz, Gaussian, PolynomialNK, Pointwise, Constant,
)


# =============================================================================
# 結果容器
# =============================================================================

@dataclass
class FitResult:
    """擬合結果"""
    success: bool
    message: str
    mse: float                       # WVASE-style MSE（可與 WVASE 直接比）
    chi_square: float                # lmfit χ²
    n_data: int
    n_param: int
    n_iter: int
    params: dict                     # {name: value}
    params_stderr: dict              # {name: stderr 或 None}

    # 重組後的層堆疊（含最佳化後的 material 物件）
    layers: list

    # 對比資料（給畫圖用）
    wavelength: np.ndarray
    angles: list
    psi_meas: np.ndarray
    delta_meas: np.ndarray
    T_meas: Optional[np.ndarray]
    psi_fit: np.ndarray
    delta_fit: np.ndarray
    T_fit: Optional[np.ndarray]

    target: str                      # 'both' / 'e2_only' / 'e1_only'

    # lmfit 原始物件（debug 用）
    _lmfit_result: Any = None


# =============================================================================
# 參數路徑展開（處理 [*] 萬用字元）
# =============================================================================

def _expand_param_paths(paths: list, params_dict: dict) -> list:
    """展開含 [*] 的參數路徑

    範例：
        'lorentz[*].A' + lorentz 有 2 個振盪器 → ['lorentz[0].A', 'lorentz[1].A']
    """
    expanded = []
    for p in paths:
        if '[*]' in p:
            # 找到 [*] 前的 key (例如 'lorentz')
            list_key = p.split('[*]')[0].split('.')[-1]
            list_obj = _resolve_path(params_dict, p.split('[*]')[0])
            if not isinstance(list_obj, list):
                raise ValueError(f'[*] 用在非 list 欄位: {p}')
            for i in range(len(list_obj)):
                expanded.append(p.replace('[*]', f'[{i}]', 1))
        else:
            expanded.append(p)
    return expanded


def _resolve_path(obj, path: str):
    """跟隨 dot/bracket 路徑取值
       'drude.omega_p'    → obj['drude']['omega_p']
       'lorentz[0].A'     → obj['lorentz'][0]['A']
    """
    cur = obj
    tokens = _tokenize_path(path)
    for tok in tokens:
        if isinstance(tok, int):
            cur = cur[tok]
        else:
            cur = cur[tok]
    return cur


def _set_path(obj, path: str, value):
    """設定路徑值（in-place）"""
    tokens = _tokenize_path(path)
    cur = obj
    for tok in tokens[:-1]:
        cur = cur[tok]
    cur[tokens[-1]] = value


def _tokenize_path(path: str) -> list:
    """'lorentz[0].A' → ['lorentz', 0, 'A']"""
    tokens = []
    for part in path.split('.'):
        # 處理 lorentz[0] 之類
        while '[' in part:
            base, rest = part.split('[', 1)
            idx_str, after = rest.split(']', 1)
            if base:
                tokens.append(base)
                base = ''
            tokens.append(int(idx_str))
            part = after
        if part:
            tokens.append(part)
    return tokens


def _param_name(layer_idx: int, path: str) -> str:
    """把 path 變成 lmfit 合法的參數名

        layer1 + 'drude.omega_p'  → 'L1__drude__omega_p'
        layer1 + 'lorentz[0].A'   → 'L1__lorentz_0__A'
    """
    s = path.replace('[', '_').replace(']', '').replace('.', '__')
    return f'L{layer_idx}__{s}'


# =============================================================================
# 從 config 建立 lmfit Parameters
# =============================================================================

def build_lmfit_parameters(config: dict) -> tuple[lmfit.Parameters, dict]:
    """掃描 config['layers']，產生 lmfit.Parameters 與 mapping

    Returns
    -------
    params : lmfit.Parameters
    mapping : dict
        { lmfit_name: (layer_idx, path_in_material_params) }
        path = None 表示這是厚度參數
    """
    params = lmfit.Parameters()
    mapping = {}

    for li, layer in enumerate(config['layers']):
        # ---- thickness ----
        if layer.get('fit_thickness', False):
            tname = f'L{li}__thickness'
            t = float(layer['thickness'])
            bounds = layer.get('thickness_bounds', [None, None])
            params.add(tname, value=t,
                       min=bounds[0] if bounds[0] is not None else -np.inf,
                       max=bounds[1] if bounds[1] is not None else np.inf,
                       vary=True)
            mapping[tname] = (li, None)   # None = thickness

        # ---- material params ----
        mat = layer.get('material')
        if not isinstance(mat, dict):
            continue        # 內建材料字串，無 fittable param

        fit_paths = mat.get('fit', [])
        if not fit_paths:
            continue

        material_params = mat.get('params', {})
        expanded = _expand_param_paths(fit_paths, material_params)

        bounds_cfg = mat.get('bounds', {})
        for path in expanded:
            value = _resolve_path(material_params, path)
            pname = _param_name(li, path)

            # 找對應 bounds（先找展開後的精確 path，再找含 [*] 的通用 path）
            b = bounds_cfg.get(path)
            if b is None:
                for bp, bv in bounds_cfg.items():
                    if '[*]' in bp:
                        if _matches_wildcard(path, bp):
                            b = bv
                            break
            b = b or [None, None]

            params.add(pname, value=float(value),
                       min=b[0] if b[0] is not None else -np.inf,
                       max=b[1] if b[1] is not None else np.inf,
                       vary=True)
            mapping[pname] = (li, path)

    return params, mapping


def _matches_wildcard(concrete: str, pattern: str) -> bool:
    """'lorentz[0].A' 是否符合 'lorentz[*].A'"""
    import re
    regex = pattern.replace('[*]', r'\[\d+\]').replace('.', r'\.')
    return re.fullmatch(regex, concrete) is not None


# =============================================================================
# 從 lmfit Parameters 還原 Layer 列表
# =============================================================================

def build_layers_from_params(config: dict, params: lmfit.Parameters,
                             mapping: dict) -> list[Layer]:
    """套用 lmfit 當前參數值到 config 上，重建 Layer 物件列表"""
    layer_dicts = copy.deepcopy(config['layers'])

    # 寫回 params 到 layer_dicts
    for pname, (li, path) in mapping.items():
        val = params[pname].value
        if path is None:
            layer_dicts[li]['thickness'] = val
        else:
            _set_path(layer_dicts[li]['material']['params'], path, val)

    # 建 Layer 物件
    layers = []
    for li, ld in enumerate(layer_dicts):
        mat_spec = ld['material']
        material = create_dispersion(mat_spec)
        thickness = ld['thickness']
        if isinstance(thickness, str) and thickness.lower() == 'infinite':
            thickness = np.inf
        layers.append(Layer(
            name=ld.get('name', f'layer{li}'),
            material=material,
            thickness_nm=float(thickness),
            coherent=ld.get('coherent', True),
        ))
    return layers


# =============================================================================
# Δ 殘差（處理 ±180° wrap）
# =============================================================================

def _delta_residual(delta_calc, delta_meas, unwrap: bool = True):
    """Δ 殘差：unwrap 後再相減，避免 179° vs -179° 看起來差 358°"""
    if unwrap:
        diff = ((delta_calc - delta_meas + 180) % 360) - 180
    else:
        diff = delta_calc - delta_meas
    return diff


# =============================================================================
# 主擬合類別
# =============================================================================

class Fitter:
    """主擬合器

    config: 完整 yaml 解析結果（dict）
    data:   EllipsometryData
    T_data: TransmissionData or None
    """

    def __init__(self, config: dict,
                 data: EllipsometryData,
                 T_data: Optional[TransmissionData] = None):
        self.config = config
        self.data = data
        self.T_data = T_data

        # 解析 fit 設定
        fit_cfg = config.get('fit', {})
        self.target = fit_cfg.get('target', 'both')
        self.method = fit_cfg.get('method', 'leastsq')
        # max_iter：使用者輸入「LM 迭代次數」，預設 35（與 WVASE 對齊）
        # 內部會視 method 換算成 max_nfev
        self.max_iter = fit_cfg.get('max_iter', 35)

        # Δ 殘差模式（新）
        # 'wrap'    — (Δc-Δm) 對 ±180° wrap，標準
        # 'sin_cos' — 拆成 sin/cos 兩個殘差，topology 正確（建議預設）
        # 'raw'     — 直接相減（debug 用）
        self.delta_residual_mode = fit_cfg.get('delta_residual_mode', 'sin_cos')
        # 舊參數相容
        if 'unwrap_delta' in fit_cfg and 'delta_residual_mode' not in fit_cfg:
            self.delta_residual_mode = 'wrap' if fit_cfg['unwrap_delta'] else 'raw'

        # Robust loss（新）
        # 'linear' / 'soft_l1' / 'huber' / 'cauchy' / 'arctan'
        # 對 outlier 漸進不敏感。只在 method='least_squares' 時生效。
        self.loss = fit_cfg.get('loss', 'linear')

        # 兩段式擬合（新）：先 DE 全域搜索，再 LM 精調
        self.two_stage = fit_cfg.get('two_stage', False)
        # DE 平行 workers：-1 用全部 CPU，1 單核（Streamlit/Notebook 安全預設）
        self.de_workers = fit_cfg.get('de_workers', 1)

        # weighting
        w = fit_cfg.get('weighting', {})
        self.sigma_psi   = w.get('sigma_psi', 1.0)
        self.sigma_delta = w.get('sigma_delta', 1.0)
        self.sigma_T     = w.get('sigma_T', 1.0)
        self.angle_weights = w.get('angle_weights', {})
        self.wl_weights = w.get('wavelength_weights', [])

        # 建 lmfit Parameters
        self.params, self.mapping = build_lmfit_parameters(config)
        self._check_valid_target()

        # 加速：固定材料的 nk 快取（不變層的 Pointwise 只算一次）
        self._nk_cache: dict = {}

    def _check_valid_target(self):
        if self.target in ('e2_only', 'e1_only'):
            # 不能 fit 厚度
            has_thickness_fit = any(
                pname.endswith('__thickness') and self.params[pname].vary
                for pname in self.params)
            if has_thickness_fit:
                raise ValueError(
                    f'target={self.target} 不能 fit 厚度（需要先用 both 鎖定厚度）')

    # ---------- 殘差函式 ----------

    def _residual_both(self, params):
        """target=both 殘差：直接 fit Ψ, Δ, T"""
        layers = build_layers_from_params(self.config, params, self.mapping)
        res = calculate(
            layers, self.data.wavelength, self.data.angles,
            compute_transmission=(self.T_data is not None),
            nk_cache=self._nk_cache,
        )

        # 計算 per-angle weights
        ang_w = np.array([self.angle_weights.get(a, 1.0) for a in self.data.angles])
        # 計算 per-wavelength weights
        wl_w = np.ones_like(self.data.wavelength)
        for wrng in self.wl_weights:
            mask = ((self.data.wavelength >= wrng['range'][0]) &
                    (self.data.wavelength <= wrng['range'][1]))
            wl_w[mask] = wrng['weight']

        # (Nw, Nangle) → flatten
        sigma_psi = self.data.sigma_psi if self.data.sigma_psi is not None else self.sigma_psi
        sigma_del = self.data.sigma_delta if self.data.sigma_delta is not None else self.sigma_delta

        r_psi = (res.psi - self.data.psi) / sigma_psi

        # Δ 殘差三種模式
        ang_factor = np.sqrt(ang_w)[np.newaxis, :]
        wl_factor  = np.sqrt(wl_w)[:, np.newaxis]
        r_psi *= ang_factor * wl_factor

        if self.delta_residual_mode == 'sin_cos':
            # 拆 sin/cos 兩個殘差 — topology 正確、無 wrap 問題
            d_c = np.deg2rad(res.delta)
            d_m = np.deg2rad(self.data.delta)
            # σ 在 rad 表示為 σ_deg·π/180，sin/cos 的雜訊比例與 Δ 等同
            sd_rad = np.deg2rad(sigma_del)
            r_sin = (np.sin(d_c) - np.sin(d_m)) / sd_rad * ang_factor * wl_factor
            r_cos = (np.cos(d_c) - np.cos(d_m)) / sd_rad * ang_factor * wl_factor
            residuals = [r_psi.ravel(), r_sin.ravel(), r_cos.ravel()]
        elif self.delta_residual_mode == 'wrap':
            r_del = (_delta_residual(res.delta, self.data.delta, True)
                     / sigma_del * ang_factor * wl_factor)
            residuals = [r_psi.ravel(), r_del.ravel()]
        else:   # raw
            r_del = ((res.delta - self.data.delta)
                     / sigma_del * ang_factor * wl_factor)
            residuals = [r_psi.ravel(), r_del.ravel()]

        if self.T_data is not None and res.T is not None:
            wl_w_T = np.ones_like(self.T_data.wavelength)
            for wrng in self.wl_weights:
                mask = ((self.T_data.wavelength >= wrng['range'][0]) &
                        (self.T_data.wavelength <= wrng['range'][1]))
                wl_w_T[mask] = wrng['weight']
            r_T = (res.T - self.T_data.T) / self.sigma_T * np.sqrt(wl_w_T)
            residuals.append(r_T)

        return np.concatenate(residuals)

    def _residual_e2(self, params, eps_extracted_per_angle):
        """target=e2_only 殘差：fit oscillator 的 ε2 到 extracted ε2"""
        layers = build_layers_from_params(self.config, params, self.mapping)
        # 找「目標層」= 第一個含 fittable material params 的層
        target_li = self._find_target_layer_index()
        target_mat: DispersionModel = layers[target_li].material

        # 計算目標層的 ε2(λ)
        eps_model = target_mat.epsilon(self.data.wavelength)
        e2_model = eps_model.imag

        residuals = []
        for ai, aoi in enumerate(self.data.angles):
            e2_ext = eps_extracted_per_angle[ai].imag
            r = (e2_model - e2_ext) / max(self.sigma_psi, 1e-6)
            residuals.append(r)
        return np.concatenate(residuals)

    def _residual_e1(self, params, eps_extracted_per_angle):
        """target=e1_only 殘差"""
        layers = build_layers_from_params(self.config, params, self.mapping)
        target_li = self._find_target_layer_index()
        target_mat: DispersionModel = layers[target_li].material

        eps_model = target_mat.epsilon(self.data.wavelength)
        e1_model = eps_model.real

        residuals = []
        for ai, aoi in enumerate(self.data.angles):
            e1_ext = eps_extracted_per_angle[ai].real
            r = (e1_model - e1_ext) / max(self.sigma_psi, 1e-6)
            residuals.append(r)
        return np.concatenate(residuals)

    def _find_target_layer_index(self) -> int:
        """找哪一層是「待擬合的目標薄膜」（第一個有 material.fit 的層）"""
        for li, layer in enumerate(self.config['layers']):
            mat = layer.get('material')
            if isinstance(mat, dict) and mat.get('fit'):
                return li
        raise ValueError('找不到目標層（沒有任何層的 material.fit 不為空）')

    # ---------- 點對點 ε 反演（e2_only / e1_only 用）----------

    def _extract_eps_per_angle(self) -> list:
        """對每個 AOI 做 point-by-point nk 反演，回傳 [ε(λ) at aoi1, ε(λ) at aoi2, ...]"""
        target_li = self._find_target_layer_index()
        layers_init = build_layers_from_params(self.config, self.params, self.mapping)

        fixed_above = layers_init[:target_li]
        substrate = layers_init[-1]
        target_d = layers_init[target_li].thickness_nm

        eps_list = []
        for ai, aoi in enumerate(self.data.angles):
            psi_aoi = self.data.psi[:, ai]
            del_aoi = self.data.delta[:, ai]
            print(f'    [point-by-point] AOI={aoi}°...')
            nk = invert_nk_pointbypoint(
                psi_aoi, del_aoi, self.data.wavelength, aoi,
                fixed_above, target_d, substrate,
                verbose=False,
            )
            n = nk[:, 0]; k = nk[:, 1]
            eps = (n + 1j * k)**2
            eps_list.append(eps)
        return eps_list

    # ---------- 主擬合驅動 ----------

    def _compute_wvase_mse(self, psi_fit, delta_fit, T_fit, nvarys):
        """獨立算 WVASE-style MSE（與優化用的殘差脫鉤，可直接和 WVASE 對）

            MSE = √(1/(2N-M) · Σ {[(Ψc-Ψm)/σΨ]² + [wrap(Δc-Δm)/σΔ]²})

        永遠用 wrap mode 算 Δ，不受 delta_residual_mode 影響。
        """
        sigma_psi = self.data.sigma_psi if self.data.sigma_psi is not None else self.sigma_psi
        sigma_del = self.data.sigma_delta if self.data.sigma_delta is not None else self.sigma_delta

        r_psi = (psi_fit - self.data.psi) / sigma_psi
        r_del = _delta_residual(delta_fit, self.data.delta, True) / sigma_del

        ss = np.sum(r_psi**2) + np.sum(r_del**2)
        n_obs = r_psi.size + r_del.size
        if T_fit is not None and self.T_data is not None:
            r_T = (T_fit - self.T_data.T) / self.sigma_T
            ss += np.sum(r_T**2)
            n_obs += r_T.size
        dof = max(n_obs - nvarys, 1)
        return float(np.sqrt(ss / dof))

    def _auto_finite_bounds(self, params: 'lmfit.Parameters',
                            verbose: bool = True) -> 'lmfit.Parameters':
        """確保所有 vary=True 參數都有 finite bounds（DE 必需）

        沒設的用 ±50% 邊界（最小 ±1.0），印警告列出哪些被自動填。
        """
        auto_filled = []
        for name, p in params.items():
            if not p.vary:
                continue
            lo, hi = p.min, p.max
            if not (np.isfinite(lo) and np.isfinite(hi)):
                v = abs(p.value)
                pad = max(v * 0.5, 1.0)
                new_lo = p.value - pad if not np.isfinite(lo) else lo
                new_hi = p.value + pad if not np.isfinite(hi) else hi
                # 厚度等不能負的常識保護
                if 'thickness' in name and new_lo < 0:
                    new_lo = max(p.value * 0.1, 0.01)
                p.min = new_lo
                p.max = new_hi
                auto_filled.append((name, new_lo, new_hi))
        if auto_filled and verbose:
            print('  ⚠️  以下參數自動補 bounds (DE 需要 finite)：')
            for n, lo, hi in auto_filled:
                print(f'      {n}: [{lo:.4g}, {hi:.4g}]')
        return params

    def _max_nfev(self, nvarys: int) -> int:
        """把 WVASE-style「LM 迭代次數」換算成 lmfit 的 max_nfev (函式評估數)

        LM 每次迭代約用 (nvarys + 1) 次函式評估（gradient 算 nvarys，function 1 次）
        DE 等全域演算法用較大上限。
        """
        if self.method in ('leastsq', 'least_squares'):
            return int(self.max_iter * (nvarys + 1))
        return int(self.max_iter * 100)   # DE 等

    def fit(self, verbose: bool = True, abort_check=None) -> FitResult:
        """跑擬合並回傳結果"""
        n_vary = sum(1 for p in self.params.values() if p.vary)
        if verbose:
            print(f'\n=== 開始擬合 ===')
            print(f'  Target:        {self.target}')
            print(f'  Δ residual:    {self.delta_residual_mode}')
            print(f'  Loss:          {self.loss}')
            print(f'  Method:        {self.method}'
                  + (' (兩段式 DE→LM)' if self.two_stage else ''))
            print(f'  Max iterations:{self.max_iter}  → max_nfev={self._max_nfev(n_vary)}')
            print(f'  變數參數:      {n_vary} 個')
            print(f'  資料點:        {self.data.psi.size * 2}'
                  + (f' + {self.T_data.T.size}' if self.T_data else ''))

        # 預備殘差函式
        if self.target == 'both':
            residual_fn = self._residual_both
        elif self.target in ('e2_only', 'e1_only'):
            if verbose:
                print(f'  先做 point-by-point ε 反演...')
            eps_extracted = self._extract_eps_per_angle()
            if self.target == 'e2_only':
                residual_fn = lambda p: self._residual_e2(p, eps_extracted)
            else:
                residual_fn = lambda p: self._residual_e1(p, eps_extracted)
        else:
            raise ValueError(f'未知 fit target: {self.target}')

        max_nfev = self._max_nfev(n_vary)

        # 兩段式：先 DE 全域搜索找 basin，再用 LM 精調
        params_init = self.params

        # DE 需要所有變數參數有有限 bound — 沒設的自動補（±50% of |value|，
        # 至少 ±1.0）。給友善錯誤訊息列出哪些缺。
        if self.two_stage or self.method == 'differential_evolution':
            params_init = self._auto_finite_bounds(params_init, verbose=verbose)

        if self.two_stage:
            if verbose:
                print(f'  Stage 1: differential_evolution 全域搜索...')
            # DE 預算：每 residual call 可能 50-200 ms（TMM 跑全波長 × 角度）
            # 用 max_nfev 控制總評估次數（lmfit 在 DE 模式下會強制這個上限）
            # 對 1-3 參數約 ~300，5+ 參數線性放大
            de_budget = 300 if n_vary <= 3 else 100 * n_vary
            de_result = lmfit.minimize(
                residual_fn, params_init,
                method='differential_evolution',
                max_nfev=de_budget,
                tol=0.01,
                seed=42,             # 可重現
                polish=False,        # 跳過 DE 自帶的 LM polish（我們自己會做）
                workers=self.de_workers,    # 多核加速 (#14)
                updating='deferred' if self.de_workers != 1 else 'immediate',
            )
            params_init = de_result.params
            if verbose:
                print(f'    DE 完成 ({de_result.nfev} evals)，切到 LM 精調...')

        # 中斷檢查 hook（lmfit 每次 iter 都呼叫；非 0 回傳值會停止）
        def _iter_cb(params, iternum, resid, *args, **kws):
            if abort_check is not None and abort_check():
                return True   # lmfit 接受 truthy 回傳值來停止
            return None

        # 主擬合（或第二階段精調）
        fit_kwargs = {'method': self.method, 'max_nfev': max_nfev,
                      'iter_cb': _iter_cb}
        if self.method == 'least_squares':
            fit_kwargs['loss'] = self.loss
        elif self.loss != 'linear':
            if verbose:
                print(f'  ⚠️  loss={self.loss} 只在 method=least_squares 時生效，'
                      f'自動切換')
            fit_kwargs['method'] = 'least_squares'
            fit_kwargs['loss'] = self.loss

        result = lmfit.minimize(residual_fn, params_init, **fit_kwargs)

        # 重建 final layers + 算 final calc
        final_layers = build_layers_from_params(self.config, result.params, self.mapping)
        final_calc = calculate(
            final_layers, self.data.wavelength, self.data.angles,
            compute_transmission=(self.T_data is not None),
        )

        # 中斷時 lmfit 屬性可能未完整填充，用 getattr 安全取
        nvarys = getattr(result, 'nvarys', n_vary)
        ndata = getattr(result, 'ndata',
                        self.data.psi.size * 2 + (self.T_data.T.size if self.T_data else 0))
        residual = getattr(result, 'residual', np.zeros(1))

        # WVASE-style MSE（永遠獨立計算，可直接與 WVASE 對比）
        mse_wvase = self._compute_wvase_mse(
            final_calc.psi, final_calc.delta, final_calc.T, nvarys)
        # Optimizer 內部 MSE（殘差直接算）
        dof = max(ndata - nvarys, 1)
        mse_opt = float(np.sqrt(np.sum(residual**2) / dof)) if len(residual) > 1 else mse_wvase

        if verbose:
            print(f'\n=== 擬合結果 ===')
            print(f'  Success: {result.success}')
            print(f'  Message: {result.message}')
            print(f'  Function evals: {result.nfev}')
            print(f'  MSE (WVASE-comparable): {mse_wvase:.4e}')
            print(f'  MSE (optimizer internal): {mse_opt:.4e}')
            print(f'\n  最佳化參數：')
            for pname, p in result.params.items():
                if p.vary:
                    se = f'± {p.stderr:.4g}' if p.stderr else ''
                    print(f'    {pname:<40} = {p.value:.5g} {se}')

        return FitResult(
            success=getattr(result, 'success', False),
            message=str(getattr(result, 'message', '')),
            mse=mse_wvase,             # 對外永遠回傳 WVASE-comparable MSE
            chi_square=float(getattr(result, 'chisqr', float('nan')) or float('nan')),
            n_data=ndata, n_param=nvarys, n_iter=getattr(result, 'nfev', 0),
            params={k: v.value for k, v in result.params.items()},
            params_stderr={k: v.stderr for k, v in result.params.items()},
            layers=final_layers,
            wavelength=self.data.wavelength,
            angles=self.data.angles,
            psi_meas=self.data.psi,
            delta_meas=self.data.delta,
            T_meas=self.T_data.T if self.T_data else None,
            psi_fit=final_calc.psi,
            delta_fit=final_calc.delta,
            T_fit=final_calc.T,
            target=self.target,
            _lmfit_result=result,
        )


# =============================================================================
# Config 載入便利函式
# =============================================================================

def load_config(path: str | Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def fit_from_config(config_path: str | Path, verbose: bool = True) -> FitResult:
    """端到端便利函式：讀 config → 讀資料 → 跑擬合"""
    cfg = load_config(config_path)

    # 讀資料
    R_cfg = cfg['data']['reflection']
    data = read_reflection(R_cfg['file'])
    # 波長範圍截取
    wr = cfg['data'].get('wavelength_range')
    if wr:
        data = data.crop_wavelength(*wr)

    T_data = None
    if 'transmission' in cfg['data']:
        T_cfg = cfg['data']['transmission']
        T_data = read_transmission(T_cfg['file'])
        if wr:
            T_data = T_data.crop_wavelength(*wr)

    fitter = Fitter(cfg, data, T_data)
    return fitter.fit(verbose=verbose)
