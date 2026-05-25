# 材料資料來源

所有資料來自 [refractiveindex.info](https://refractiveindex.info/) GitHub database。

命名規則（不同作者分開存）：
- `{material}.csv` — 主要版本（現代寬頻量測，預設用這個）
- `{material}_JC.csv` — Johnson & Christy 1972（noble metals 經典，僅 Au）
- `{material}_Palik.csv` — Palik Handbook 章節作者原始資料

> **CRC Handbook** 只有少數固定波長的折射率，不適合擬合，未收錄。
> **J&C 1972** (Phys. Rev. B 6, 4370) 只有 Cu, Ag, Au 三個 noble metals。

## Si

### 主要：Franta-25C
- Franta et al. 2017 (25°C), Crystals 7, 277
- 文件覆蓋：31-309000 nm
- 實際下載：4001 點，31–309963 nm

### Palik：Aspnes
- Aspnes & Studna 1983, Phys. Rev. B 27, 985 (Palik)
- 文件覆蓋：207-840 nm
- 實際下載：46 點，207–827 nm

## SiO2

### 主要：Malitson
- Malitson 1965, J. Opt. Soc. Am. 55, 1205 (in Palik)
- 文件覆蓋：210-6700 nm (公式)
- 實際下載：6491 點，210–6700 nm

## Al2O3

### 主要：Boidin
- Boidin et al. 2016, Thin Solid Films 615, 11
- 文件覆蓋：250-2500 nm
- 實際下載：424 點，300–18003 nm

### Palik：Querry
- Querry 1985 (Palik Handbook ch.)
- 文件覆蓋：200 nm-200 um
- 實際下載：550 點，210–12500 nm

## TiO2

### 主要：Sarkar
- Sarkar et al. 2019, ACS Appl. Opt. Mater.
- 文件覆蓋：300-2000 nm
- 實際下載：977 點，300–1690 nm

### Palik：Devore-o
- DeVore 1951 (Palik Handbook ch., ordinary ray)
- 文件覆蓋：430-1530 nm
- 實際下載：1101 點，430–1530 nm

## Au

### 主要：McPeak
- McPeak et al. 2015, ACS Photonics 2, 326 (template-stripped film)
- 文件覆蓋：300-1700 nm
- 實際下載：141 點，300–1700 nm

### J&C：Johnson
- Johnson & Christy 1972, Phys. Rev. B 6, 4370
- 文件覆蓋：188-1937 nm
- 實際下載：49 點，188–1937 nm

### Palik：Hagemann
- Hagemann et al. 1975 (Palik Handbook ch.)
- 文件覆蓋：1.24nm-249um
- 實際下載：149 點，0–248000 nm

## Al

### 主要：Rakic
- Rakić 1995, Appl. Opt. 34, 4755 (Brendel-Bormann)
- 文件覆蓋：125 nm-200 um
- 實際下載：206 點，0–200000 nm

### Palik：Hagemann
- Hagemann et al. 1975 (Palik Handbook ch.)
- 文件覆蓋：1.24nm-249um
- 實際下載：148 點，0–1240000 nm

## MgO

### 主要：Synowicki
- Synowicki & Tiwald 2004, Thin Solid Films 455, 248
- 文件覆蓋：130 nm-33 um
- 實際下載：400 點，130–33000 nm

### Palik：Stephens
- Stephens & Malitson 1952 (Palik Handbook ch.)
- 文件覆蓋：360-5400 nm (公式)
- 實際下載：5041 點，360–5400 nm


## 別名

- **sapphire** → 等同 `Al2O3`（sapphire 是 Al2O3 的別名）

## 複合基板（不是單一材料，需用膜層堆疊定義）

### SiO2/Si （熱氧化片）
Si wafer 上長一層熱氧化 SiO2，常見厚度 100/200/285/300/500 nm。
在 config 內定義為兩層：
```yaml
layers:
  - { name: SiO2_thermal, material: SiO2, thickness: 285,      fit_thickness: false }
  - { name: substrate,    material: Si,   thickness: infinite }
```