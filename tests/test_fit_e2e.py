"""
端到端擬合測試：用 fake data 對答案

策略：
    1. 用 fake data (50 nm Au / sapphire)
    2. 把厚度初始值故意設錯（50 → 80 nm）
    3. fit 應該收斂回 50 nm
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ellipsometry.core.fitter import Fitter, load_config
from ellipsometry.core.io import read_reflection, read_transmission


def test_fit_au_sapphire_thickness():
    """測 1: 只 fit Au 厚度（材料 nk 固定用 McPeak）"""
    print('\n' + '='*70)
    print('  Test 1: Fit Au thickness only (50 nm truth, init=80 nm)')
    print('='*70)

    data = read_reflection('data/sample1_Au50_sapphire/R_long.dat')
    T_data = read_transmission('data/sample1_Au50_sapphire/T.dat')

    cfg = {
        'layers': [
            {'name': 'air', 'material': 'air', 'thickness': 'infinite'},
            {'name': 'Au',  'material': 'Au',
             'thickness': 80,                        # 故意設錯（真值 50）
             'coherent': True,
             'fit_thickness': True,
             'thickness_bounds': [10, 200]},
            {'name': 'substrate', 'material': 'Al2O3',
             'thickness': 1_000_000, 'coherent': False, 'fit_thickness': False},
        ],
        'fit': {
            'target': 'both',
            'method': 'leastsq',
            'max_iter': 200,
            'weighting': {'sigma_psi': 0.5, 'sigma_delta': 1.0, 'sigma_T': 0.01},
            'unwrap_delta': True,
        },
    }

    fitter = Fitter(cfg, data, T_data)
    result = fitter.fit(verbose=True)

    fitted_d = result.params['L1__thickness']
    print(f'\n  ✓ 收斂厚度: {fitted_d:.2f} nm  (truth: 50.00 nm)')
    assert abs(fitted_d - 50) < 0.5, f'厚度誤差過大: {fitted_d}'
    print(f'  ✓ 通過（誤差 {abs(fitted_d-50):.3f} nm < 0.5 nm）')


def test_fit_sio2_thickness():
    """測 2: SiO2 thermal oxide 厚度"""
    print('\n' + '='*70)
    print('  Test 2: Fit SiO2 thickness (285 nm truth, init=200 nm)')
    print('='*70)

    data = read_reflection('data/sample2_SiO2_285_Si/R_long.dat')

    cfg = {
        'layers': [
            {'name': 'air', 'material': 'air', 'thickness': 'infinite'},
            {'name': 'SiO2', 'material': 'SiO2',
             'thickness': 200,                       # 故意設錯（真值 285）
             'coherent': True,
             'fit_thickness': True,
             'thickness_bounds': [50, 500]},
            {'name': 'Si', 'material': 'Si',
             'thickness': 1_000_000, 'coherent': False, 'fit_thickness': False},
        ],
        'fit': {
            'target': 'both',
            'method': 'leastsq',
            'max_iter': 500,
            'weighting': {'sigma_psi': 0.5, 'sigma_delta': 1.0},
            'unwrap_delta': True,
        },
    }

    fitter = Fitter(cfg, data, None)
    result = fitter.fit(verbose=True)

    fitted_d = result.params['L1__thickness']
    print(f'\n  ✓ 收斂厚度: {fitted_d:.2f} nm  (truth: 285.00 nm)')
    assert abs(fitted_d - 285) < 1.0, f'厚度誤差過大: {fitted_d}'
    print(f'  ✓ 通過（誤差 {abs(fitted_d-285):.3f} nm < 1.0 nm）')


if __name__ == '__main__':
    test_fit_au_sapphire_thickness()
    test_fit_sio2_thickness()
    print('\n' + '='*70)
    print('  所有測試通過 ✓')
    print('='*70)
