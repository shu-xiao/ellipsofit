"""
從 refractiveindex.info 下載材料 nk 資料並存成 csv

命名規則：不同作者分開存
  - {material}.csv          主要版本（現代寬頻量測，預設用這個）
  - {material}_JC.csv       Johnson & Christy 1972（noble metals 經典，僅 Au/Ag/Cu）
  - {material}_Palik.csv    Palik Handbook 章節作者原始資料

備註：
  - Palik 是編者不是實驗者，Handbook 收錄各章作者（Hagemann, Querry, Ordal...）的量測
  - CRC Handbook 只有少數固定波長的折射率，不適合擬合，不下載
  - J&C 1972 (Phys. Rev. B 6, 4370) 只有 Cu, Ag, Au 三個 noble metals

執行：
    python materials/download_materials.py

輸出：
    materials/*.csv            各種版本
    materials/SOURCES.md       來源紀錄
"""
import os
import urllib.request
import yaml
import numpy as np


# (reference, citation, range_doc, tag)
# tag → 決定輸出檔名後綴：'primary'='', 'jc'='_JC', 'palik'='_Palik'
MATERIALS = {
    'Si': [
        ('Franta-25C', 'Franta et al. 2017 (25°C), Crystals 7, 277',           '31-309000 nm', 'primary'),
        ('Aspnes',     'Aspnes & Studna 1983, Phys. Rev. B 27, 985 (Palik)',   '207-840 nm',   'palik'),
    ],
    'SiO2': [
        ('Malitson',   'Malitson 1965, J. Opt. Soc. Am. 55, 1205 (in Palik)',  '210-6700 nm (公式)', 'primary'),
        # Malitson 同時是 primary 與 Palik 章節，不重複
    ],
    'Al2O3': [
        ('Boidin',     'Boidin et al. 2016, Thin Solid Films 615, 11',         '250-2500 nm',  'primary'),
        ('Querry',     'Querry 1985 (Palik Handbook ch.)',                     '200 nm-200 um','palik'),
    ],
    'TiO2': [
        ('Sarkar',     'Sarkar et al. 2019, ACS Appl. Opt. Mater.',            '300-2000 nm',  'primary'),
        ('Devore-o',   'DeVore 1951 (Palik Handbook ch., ordinary ray)',       '430-1530 nm',  'palik'),
    ],
    'Au': [
        ('McPeak',     'McPeak et al. 2015, ACS Photonics 2, 326 (template-stripped film)', '300-1700 nm', 'primary'),
        ('Johnson',    'Johnson & Christy 1972, Phys. Rev. B 6, 4370',         '188-1937 nm',  'jc'),
        ('Hagemann',   'Hagemann et al. 1975 (Palik Handbook ch.)',            '1.24nm-249um', 'palik'),
    ],
    'Al': [
        ('Rakic',      'Rakić 1995, Appl. Opt. 34, 4755 (Brendel-Bormann)',    '125 nm-200 um','primary'),
        ('Hagemann',   'Hagemann et al. 1975 (Palik Handbook ch.)',            '1.24nm-249um', 'palik'),
    ],
    'MgO': [
        ('Synowicki',  'Synowicki & Tiwald 2004, Thin Solid Films 455, 248',   '130 nm-33 um', 'primary'),
        ('Stephens',   'Stephens & Malitson 1952 (Palik Handbook ch.)',        '360-5400 nm (公式)', 'palik'),
    ],
}

# 別名：sapphire 即 Al2O3 (單晶剛玉)
ALIASES = {
    'sapphire': 'Al2O3',
}

TAG_SUFFIX = {'primary': '', 'jc': '_JC', 'palik': '_Palik'}
TAG_LABEL  = {'primary': '主要', 'jc': 'J&C', 'palik': 'Palik'}

BASE_URL = 'https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data/main/{mat}/nk/{ref}.yml'

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def fetch_yaml(material, reference):
    url = BASE_URL.format(mat=material, ref=reference)
    with urllib.request.urlopen(url, timeout=30) as resp:
        return yaml.safe_load(resp.read().decode('utf-8'))


def parse_nk_data(yml):
    """從 YAML 抽出 (rows, kind)
       kind: 'nk' / 'n_only' / 'formula'
    """
    for entry in yml['DATA']:
        if entry['type'] == 'tabulated nk':
            rows = [list(map(float, ln.split())) for ln in entry['data'].strip().split('\n')]
            return rows, 'nk', None
        if entry['type'] == 'tabulated n':
            rows = [list(map(float, ln.split())) + [0.0] for ln in entry['data'].strip().split('\n')]
            return rows, 'n_only', None
    for entry in yml['DATA']:
        if entry['type'].startswith('formula'):
            return None, 'formula', entry
    raise ValueError('No supported data format')


def evaluate_formula(entry, wl_um):
    """Sellmeier 公式：formula 1 / formula 2"""
    coeffs = list(map(float, entry['coefficients'].split()))
    c0 = coeffs[0]
    n_sq_minus_1 = np.full_like(wl_um, c0)
    if entry['type'] == 'formula 1':
        for i in range(1, len(coeffs), 2):
            C, D = coeffs[i], coeffs[i + 1]
            n_sq_minus_1 += C * wl_um**2 / (wl_um**2 - D**2)
    elif entry['type'] == 'formula 2':
        for i in range(1, len(coeffs), 2):
            C, D = coeffs[i], coeffs[i + 1]
            n_sq_minus_1 += C * wl_um**2 / (wl_um**2 - D)
    elif entry['type'] == 'formula 4':
        # refractiveindex.info formula 4:
        # n² = c0 + c1·λ^c2/(λ² - c3^c4) + c5·λ^c6/(λ² - c7^c8)
        #      + c9·λ^c10 + c11·λ^c12 + c13·λ^c14 + c15·λ^c16
        n_sq = np.full_like(wl_um, coeffs[0])
        # 兩個分式項
        for base in (1, 5):
            if base + 3 >= len(coeffs):
                break
            C, p_num, D, p_den = coeffs[base:base + 4]
            n_sq += C * wl_um**p_num / (wl_um**2 - D**p_den)
        # 多項式項（c9·λ^c10, c11·λ^c12, ...）
        for base in range(9, len(coeffs) - 1, 2):
            C, p = coeffs[base], coeffs[base + 1]
            n_sq += C * wl_um**p
        return np.sqrt(n_sq)
    else:
        raise NotImplementedError(f'公式 {entry["type"]} 未支援')
    return np.sqrt(1 + n_sq_minus_1)


def save_csv(out_path, wl_nm, n_arr, k_arr):
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('wavelength_nm,n,k\n')
        for w, n, k in zip(wl_nm, n_arr, k_arr):
            f.write(f'{w:.4f},{n:.6f},{k:.6f}\n')


def download_one(material, reference, tag):
    """下載單一 (material, reference) → 存 csv，回傳 (出檔路徑, 點數, range)"""
    yml = fetch_yaml(material, reference)
    rows, kind, formula = parse_nk_data(yml)

    if kind in ('nk', 'n_only'):
        arr = np.array(rows)
        wl_nm = arr[:, 0] * 1000
        n_arr = arr[:, 1]
        k_arr = arr[:, 2]
    else:  # formula
        # 若公式有 wavelength_range，限制在範圍內取樣；否則 200-2000 nm
        if 'wavelength_range' in formula:
            w0_um, w1_um = map(float, str(formula['wavelength_range']).split())
            wl_nm = np.arange(w0_um * 1000, w1_um * 1000 + 1, 1, dtype=float)
        else:
            wl_nm = np.arange(200, 2001, 1, dtype=float)
        wl_um = wl_nm / 1000
        n_arr = evaluate_formula(formula, wl_um)
        k_arr = np.zeros_like(n_arr)

    out_path = os.path.join(OUT_DIR, f'{material}{TAG_SUFFIX[tag]}.csv')
    save_csv(out_path, wl_nm, n_arr, k_arr)
    return out_path, len(wl_nm), (wl_nm[0], wl_nm[-1])


def main():
    sources = [
        '# 材料資料來源\n',
        '所有資料來自 [refractiveindex.info](https://refractiveindex.info/) GitHub database。\n',
        '命名規則（不同作者分開存）：',
        '- `{material}.csv` — 主要版本（現代寬頻量測，預設用這個）',
        '- `{material}_JC.csv` — Johnson & Christy 1972（noble metals 經典，僅 Au）',
        '- `{material}_Palik.csv` — Palik Handbook 章節作者原始資料\n',
        '> **CRC Handbook** 只有少數固定波長的折射率，不適合擬合，未收錄。',
        '> **J&C 1972** (Phys. Rev. B 6, 4370) 只有 Cu, Ag, Au 三個 noble metals。\n',
    ]

    for material, refs in MATERIALS.items():
        sources.append(f'## {material}\n')
        for reference, citation, wlrange_doc, tag in refs:
            tag_label = TAG_LABEL[tag]
            print(f'\n[{material}] {tag_label} = {reference}')
            try:
                out_path, npts, (w0, w1) = download_one(material, reference, tag)
                print(f'  ✓ {os.path.basename(out_path)}  ({npts} 點，{w0:.0f}-{w1:.0f} nm)')
                sources.append(f'### {tag_label}：{reference}')
                sources.append(f'- {citation}')
                sources.append(f'- 文件覆蓋：{wlrange_doc}')
                sources.append(f'- 實際下載：{npts} 點，{w0:.0f}–{w1:.0f} nm\n')
            except Exception as e:
                print(f'  ✗ 失敗: {e}')
                sources.append(f'### {tag_label}：{reference}\n- ✗ 下載失敗：{e}\n')

    # 加上別名說明
    sources.append('\n## 別名\n')
    for alias, target in ALIASES.items():
        sources.append(f'- **{alias}** → 等同 `{target}`（{alias} 是 {target} 的別名）')
    # 加上膜層堆疊基板說明
    sources.append('\n## 複合基板（不是單一材料，需用膜層堆疊定義）\n')
    sources.append('### SiO2/Si （熱氧化片）')
    sources.append('Si wafer 上長一層熱氧化 SiO2，常見厚度 100/200/285/300/500 nm。')
    sources.append('在 config 內定義為兩層：')
    sources.append('```yaml')
    sources.append('layers:')
    sources.append('  - { name: SiO2_thermal, material: SiO2, thickness: 285,      fit_thickness: false }')
    sources.append('  - { name: substrate,    material: Si,   thickness: infinite }')
    sources.append('```')

    with open(os.path.join(OUT_DIR, 'SOURCES.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(sources))
    print('\n完成，紀錄寫到 SOURCES.md')


if __name__ == '__main__':
    main()
