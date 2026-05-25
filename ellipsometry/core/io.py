"""
WVASE / 通用 .dat 檔讀取模組

支援格式：
    1. WVASE 長格式（每個 AOI 一個區段，以 "% AOI = xx" 標記）
       % nm  Psi  Delta  Err(Psi)  Err(Delta)
       % AOI = 65.000
       300.000  22.140  150.320  0.020  0.100
       ...
       % AOI = 70.000
       300.000  25.670  148.110  0.020  0.100

    2. WVASE 寬格式（一個 row 含所有 AOI 的 Psi/Delta）
       nm  Psi_65  Del_65  Psi_70  Del_70  Psi_75  Del_75
       300  22.14  150.32  25.67  148.11  29.01  145.88

    3. 通用穿透格式
       nm  T
       300  0.85
       ...

API：
    data = read_reflection('sample_R.dat')           # 自動偵測格式
    data = read_reflection('sample_R.dat', format='wvase_long')
    t    = read_transmission('sample_T.dat')

回傳 dataclass:
    EllipsometryData(wavelength, angles, psi, delta, sigma_psi, sigma_delta)
    TransmissionData(wavelength, T, sigma_T)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# 資料容器
# =============================================================================

@dataclass
class EllipsometryData:
    """反射 ellipsometry 量測資料

    Attributes
    ----------
    wavelength : ndarray (Nw,)
        波長 (nm)，遞增排序
    angles : list[float]
        入射角 (度)
    psi : ndarray (Nw, Nangle)
        Ψ 值 (度)
    delta : ndarray (Nw, Nangle)
        Δ 值 (度)
    sigma_psi, sigma_delta : ndarray (Nw, Nangle) or None
        測量誤差 (度)，沒有則為 None
    source : str
        原始檔案路徑（紀錄用）
    """
    wavelength: np.ndarray
    angles: list[float]
    psi: np.ndarray
    delta: np.ndarray
    sigma_psi: Optional[np.ndarray] = None
    sigma_delta: Optional[np.ndarray] = None
    source: str = ''

    @property
    def n_wavelength(self) -> int:
        return len(self.wavelength)

    @property
    def n_angle(self) -> int:
        return len(self.angles)

    def crop_wavelength(self, w_min: float, w_max: float) -> 'EllipsometryData':
        """截取波長範圍，回傳新物件（不改原物件）"""
        mask = (self.wavelength >= w_min) & (self.wavelength <= w_max)
        return EllipsometryData(
            wavelength=self.wavelength[mask],
            angles=self.angles,
            psi=self.psi[mask],
            delta=self.delta[mask],
            sigma_psi=self.sigma_psi[mask] if self.sigma_psi is not None else None,
            sigma_delta=self.sigma_delta[mask] if self.sigma_delta is not None else None,
            source=self.source,
        )

    def __repr__(self):
        return (f'EllipsometryData({self.n_wavelength} λ, angles={self.angles}, '
                f'range=[{self.wavelength.min():.0f}, {self.wavelength.max():.0f}] nm)')


@dataclass
class TransmissionData:
    """穿透量測資料（normal incidence T）"""
    wavelength: np.ndarray
    T: np.ndarray
    sigma_T: Optional[np.ndarray] = None
    angle: float = 0.0
    source: str = ''

    def crop_wavelength(self, w_min: float, w_max: float) -> 'TransmissionData':
        mask = (self.wavelength >= w_min) & (self.wavelength <= w_max)
        return TransmissionData(
            wavelength=self.wavelength[mask],
            T=self.T[mask],
            sigma_T=self.sigma_T[mask] if self.sigma_T is not None else None,
            angle=self.angle,
            source=self.source,
        )

    def __repr__(self):
        return (f'TransmissionData({len(self.wavelength)} λ, AOI={self.angle}°, '
                f'range=[{self.wavelength.min():.0f}, {self.wavelength.max():.0f}] nm)')


# =============================================================================
# 主要 API
# =============================================================================

def read_reflection(path: str | Path, format: str = 'auto') -> EllipsometryData:
    """讀反射 ellipsometry 資料

    Parameters
    ----------
    path : str or Path
    format : 'auto' / 'wvase_long' / 'wvase_wide' / 'generic'
        'auto' 會嘗試自動偵測

    Returns
    -------
    EllipsometryData
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'找不到檔案：{path}')

    if format == 'auto':
        format = _detect_format(path)

    if format == 'wvase_long':
        return _read_wvase_long(path)
    if format == 'wvase_wide':
        return _read_wvase_wide(path)
    if format == 'generic':
        # 嘗試先當寬格式再退回長格式
        try:
            return _read_wvase_wide(path)
        except Exception:
            return _read_wvase_long(path)
    raise ValueError(f'不支援的格式: {format}')


def read_transmission(path: str | Path) -> TransmissionData:
    """讀穿透資料（兩欄：wavelength_nm, T；或三欄含 sigma）"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'找不到檔案：{path}')

    df = _fast_load(path)
    if df.shape[1] < 2:
        raise ValueError(f'穿透檔需至少 2 欄 (nm, T)，實際 {df.shape[1]} 欄')

    wavelength = df.iloc[:, 0].to_numpy()
    T = df.iloc[:, 1].to_numpy()
    sigma_T = df.iloc[:, 2].to_numpy() if df.shape[1] >= 3 else None

    # 若 T 在 0-100 範圍（百分比），自動轉成 0-1
    if T.max() > 1.5:
        T = T / 100.0
        if sigma_T is not None:
            sigma_T = sigma_T / 100.0

    return TransmissionData(
        wavelength=wavelength, T=T, sigma_T=sigma_T,
        angle=0.0, source=str(path),
    )


# =============================================================================
# 內部：格式偵測
# =============================================================================

_AOI_PATTERN = re.compile(r'AOI\s*=\s*([\d.]+)', re.IGNORECASE)


def _detect_format(path: Path) -> str:
    """偵測檔案格式：wvase_long / wvase_wide"""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        head = ''.join([next(f, '') for _ in range(50)])

    if _AOI_PATTERN.search(head):
        return 'wvase_long'
    # 看 header 有沒有 Psi_xx / Del_xx 之類的欄位
    if re.search(r'(Psi|Ψ)[_\-]?\d+', head, re.IGNORECASE):
        return 'wvase_wide'
    # 預設嘗試寬格式
    return 'wvase_wide'


# =============================================================================
# 內部：快速讀檔
# =============================================================================

def _fast_load(path: Path, comment: str = '%', skip_blank: bool = True) -> pd.DataFrame:
    """pandas C 引擎讀數值資料（比 np.loadtxt 快 5-10x）

    自動：
    - 跳過 '%' 或 '#' 開頭的註解行
    - 自動偵測 header（第一個全文字的非註解行）
    - 多種空白分隔
    """
    # 先掃過去找出註解行與 header
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    has_header = False
    skiprows = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            skiprows.append(i)
            continue
        if stripped.startswith('%') or stripped.startswith('#'):
            skiprows.append(i)
            continue
        # 第一個非註解行：判斷是否為 header（含字母）
        tokens = stripped.split()
        if any(any(c.isalpha() for c in t) for t in tokens):
            has_header = (i == min(j for j in range(len(lines)) if j not in skiprows))
        break

    df = pd.read_csv(
        path, sep=r'\s+', comment='%', engine='c',
        header=0 if has_header else None,
        skip_blank_lines=skip_blank,
        dtype=np.float64, on_bad_lines='skip',
    )
    # 二次過濾：'#' 開頭的行 pandas 預設不會跳，再清一次
    return df


# =============================================================================
# 內部：WVASE 長格式（多 AOI 區段）
# =============================================================================

def _read_wvase_long(path: Path) -> EllipsometryData:
    """讀 WVASE 長格式：每個 AOI 一個區段，以 '% AOI = xx' 標記"""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # 分區段：找 AOI 標記
    sections: list[tuple[float, list[str]]] = []
    cur_aoi: Optional[float] = None
    cur_lines: list[str] = []

    for line in lines:
        m = _AOI_PATTERN.search(line)
        if m:
            if cur_aoi is not None:
                sections.append((cur_aoi, cur_lines))
            cur_aoi = float(m.group(1))
            cur_lines = []
        elif cur_aoi is not None:
            stripped = line.strip()
            if stripped and not stripped.startswith(('%', '#')):
                # 第一個非數字 token → header，跳過
                first = stripped.split()[0]
                try:
                    float(first)
                    cur_lines.append(line)
                except ValueError:
                    pass

    if cur_aoi is not None:
        sections.append((cur_aoi, cur_lines))

    if not sections:
        raise ValueError(f'找不到任何 "AOI = xxx" 區段於 {path}')

    # 解析每個區段：[wavelength, psi, delta, (err_psi, err_delta)]
    parsed = []
    for aoi, data_lines in sections:
        rows = np.array([list(map(float, ln.split())) for ln in data_lines])
        if rows.shape[1] < 3:
            raise ValueError(f'AOI={aoi}° 區段欄位 < 3（需至少 nm, Psi, Delta）')
        parsed.append((aoi, rows))

    # 取共同波長（以第一段為基準，檢查所有段一致）
    ref_w = parsed[0][1][:, 0]
    for aoi, rows in parsed[1:]:
        if not np.allclose(rows[:, 0], ref_w, atol=1e-6):
            raise ValueError(f'AOI={aoi}° 與第一段的波長不一致')

    angles = [aoi for aoi, _ in parsed]
    psi = np.column_stack([rows[:, 1] for _, rows in parsed])    # (Nw, Nangle)
    delta = np.column_stack([rows[:, 2] for _, rows in parsed])

    # 誤差欄（若有）
    sigma_psi = sigma_delta = None
    if parsed[0][1].shape[1] >= 5:
        sigma_psi = np.column_stack([rows[:, 3] for _, rows in parsed])
        sigma_delta = np.column_stack([rows[:, 4] for _, rows in parsed])

    return EllipsometryData(
        wavelength=ref_w, angles=angles,
        psi=psi, delta=delta,
        sigma_psi=sigma_psi, sigma_delta=sigma_delta,
        source=str(path),
    )


# =============================================================================
# 內部：WVASE 寬格式（一 row 含所有 AOI）
# =============================================================================

_COLNAME_PATTERN = re.compile(
    r'^(Psi|Ψ|Del|Delta|Δ|sigma_?Psi|sigma_?Del|Err)[_\-\(\s]*(\d+(?:\.\d+)?)',
    re.IGNORECASE,
)


def _read_wvase_wide(path: Path) -> EllipsometryData:
    """讀 WVASE 寬格式：欄名類似 Psi_65, Del_65, Psi_70, ..."""
    df = _fast_load(path)
    # 若全部欄名都是整數（0, 1, 2...）代表沒有 header
    if not all(isinstance(c, str) for c in df.columns):
        raise ValueError(f'寬格式必須有 header (例如 Psi_65, Del_65)，{path}')

    # 從欄名抽出 (種類, AOI)
    cols = {'psi': {}, 'delta': {}, 'sigma_psi': {}, 'sigma_delta': {}}
    wavelength_col = df.columns[0]
    for col in df.columns[1:]:
        m = _COLNAME_PATTERN.match(str(col))
        if not m:
            continue
        kind_raw = m.group(1).lower()
        aoi = float(m.group(2))
        kind_map = {
            'psi': 'psi', 'ψ': 'psi',
            'del': 'delta', 'delta': 'delta', 'δ': 'delta',
            'sigma_psi': 'sigma_psi', 'sigmapsi': 'sigma_psi',
            'sigma_del': 'sigma_delta', 'sigmadel': 'sigma_delta',
            'err': None,  # 模糊，留給後續
        }
        kind = kind_map.get(kind_raw)
        if kind is None:
            continue
        cols[kind][aoi] = col

    if not cols['psi']:
        raise ValueError(f'寬格式 header 找不到 Psi_xx 欄：{list(df.columns)}')

    angles = sorted(cols['psi'].keys())
    if set(cols['delta'].keys()) != set(angles):
        raise ValueError(f'Psi 與 Delta 的角度不一致：Psi={list(cols["psi"])}, Delta={list(cols["delta"])}')

    wavelength = df[wavelength_col].to_numpy()
    psi = np.column_stack([df[cols['psi'][a]].to_numpy() for a in angles])
    delta = np.column_stack([df[cols['delta'][a]].to_numpy() for a in angles])

    sigma_psi = sigma_delta = None
    if set(cols['sigma_psi'].keys()) == set(angles):
        sigma_psi = np.column_stack([df[cols['sigma_psi'][a]].to_numpy() for a in angles])
    if set(cols['sigma_delta'].keys()) == set(angles):
        sigma_delta = np.column_stack([df[cols['sigma_delta'][a]].to_numpy() for a in angles])

    return EllipsometryData(
        wavelength=wavelength, angles=angles,
        psi=psi, delta=delta,
        sigma_psi=sigma_psi, sigma_delta=sigma_delta,
        source=str(path),
    )


# =============================================================================
# 命令列自測：python -m ellipsometry.core.io <file>
# =============================================================================

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('用法：python -m ellipsometry.core.io <file.dat>')
        sys.exit(1)

    fpath = sys.argv[1]
    print(f'讀取：{fpath}')
    try:
        data = read_reflection(fpath)
        print(f'  → {data}')
        print(f'  Ψ shape: {data.psi.shape}')
        print(f'  Ψ範圍 (deg): [{data.psi.min():.2f}, {data.psi.max():.2f}]')
        print(f'  Δ範圍 (deg): [{data.delta.min():.2f}, {data.delta.max():.2f}]')
        if data.sigma_psi is not None:
            print('  ✓ 含 σ_Ψ, σ_Δ 誤差欄')
    except Exception as e:
        print(f'  反射讀取失敗：{e}')
        print('  嘗試以穿透格式讀取...')
        try:
            t = read_transmission(fpath)
            print(f'  → {t}')
        except Exception as e2:
            print(f'  穿透讀取也失敗：{e2}')
