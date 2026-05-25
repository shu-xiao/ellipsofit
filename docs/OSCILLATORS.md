# 振盪器類型對照表

ellipsofit GenOsc 模型支援的振盪器類型與其物理意義。對應 WVASE32 GenOsc Layer 的振盪器選項。

## 公式對照

所有公式中：`E` = 光子能量 (eV)，`i` = √-1，輸出 ε = ε₁ + i·ε₂。

### Lorentz（標準振盪器）

```
ε(E) = Amp · En² / (En² − E² − i·Br·E)
```

- **物理：** 古典 Lorentz 阻尼振盪器，內帶 KK 一致性
- **參數：** Amp（振幅，無單位）/ En（中心能量 eV）/ Br（FWHM 展寬 eV）
- **適用：** 透明區的折射率調整、單峰吸收、interband 躍遷

---

### Gaussian

```
ε₂(E) = Amp · [exp(-((E-En)/σ)²) − exp(-((E+En)/σ)²)]    其中 σ = Br / (2√ln2)
ε₁(E) = Kramers-Kronig 從 ε₂ 數值反推
```

- **物理：** 高斯峰（非 Lorentz 拖尾）+ KK 確保因果性
- **參數：** Amp / En / Br（FWHM）
- **適用：** 非晶吸收峰、振動峰、雜質吸收

---

### Drude（自由電子，金屬必備）

```
ε(E) = −Amp / (E² + i·Br·E)        其中 Amp = ωp²（plasma 頻率平方）
                                          Br  = Γ（散射阻尼）
```

- **物理：** 自由電子氣體，**沒有共振能量**（峰在 E=0）
- **參數：** Amp (eV²)/ Br (eV)，**沒有 En**
- **適用：** 金屬（Au、Ag、Al、Cu）、高摻雜半導體
- **典型值：** Au ωp²≈81 eV² (ωp≈9 eV)，Γ≈0.07 eV

---

### Tauc-Lorentz（非晶半導體）

```
ε₂(E) = Amp · En · Br · (E − Eg)² / [E · ((E²−En²)² + Br²·E²)]    for E > Eg
ε₂(E) = 0                                                          for E ≤ Eg
ε₁(E) = Jellison-Modine 1996 解析式
```

- **物理：** Tauc band edge + Lorentz oscillator，KK 一致
- **參數：** Amp / En / Br / Eg（bandgap）
- **適用：** a-Si、a-SiN、a-Ge、非晶氧化物

---

### Harmonic

```
ε(E) = Amp · En² / (En² − E² − i·Br·E)
```

- **物理：** 同 Lorentz（在 WVASE 是別名）
- **參數：** Amp / En / Br
- **適用：** 等同 Lorentz

---

## 參數需求快查

| 類型 | Amp | En | Br | Eg | 共幾個 |
|--|:--:|:--:|:--:|:--:|:--:|
| `lorentz` | ✅ | ✅ | ✅ | – | 3 |
| `gaussian` | ✅ | ✅ | ✅ | – | 3 |
| `harmonic` | ✅ | ✅ | ✅ | – | 3 |
| **`drude`** | ✅ | ❌ | ✅ | – | **2** |
| `tauc_lorentz` | ✅ | ✅ | ✅ | ✅ | 4 |

---

## Layer-level 參數（所有振盪器共用）

GenOsc 整層的設定，**不屬於任何單一振盪器**：

| 參數 | 單位 | 意義 |
|--|--|--|
| `e1_offset` | 無 | 高頻介電常數 ε∞（相當於 1 + UV pole 貢獻）|
| `Egap` | eV | 全模型 bandgap（給 Tauc-Lorentz 用）|
| `uv_pole` | (eV, 無) | UV pole 位置與強度（高頻吸收外推）|
| `ir_pole` | (eV, 無) | IR pole 位置與強度（低頻吸收外推）|

```
ε_total(E) = e1_offset
           + UV_pole 貢獻 (ε₁ only)
           + IR_pole 貢獻 (ε₁ only)
           + Σ_i ε_oscillator_i(E)
```

---

## 選擇建議

| 情境 | 建議模型 |
|--|--|
| 已知是純金屬（Au, Ag, Al）| Drude + 1-2 Lorentz（interband 修正） |
| 透明介電（SiO2, Al2O3, MgO） | Cauchy 或 Sellmeier |
| 半導體 (Si, GaAs) | Tauc-Lorentz × 1-3 |
| 非晶半導體 (a-Si) | Tauc-Lorentz × 2-3 |
| 不確定材料 | Lorentz × N + Drude（依 pseudo-ε 形狀） |

## 參考資料

- WVASE32 Manual: General Oscillator Layer 章節
- Jellison & Modine, *Appl. Phys. Lett.* 69, 371 (1996) — Tauc-Lorentz 解析 KK
- Rakić et al., *Appl. Opt.* 34, 4755 (1995) — Drude-Lorentz for Au/Ag/Al/Cu
- Johnson & Christy, *Phys. Rev. B* 6, 4370 (1972) — Au/Ag/Cu 經典資料
