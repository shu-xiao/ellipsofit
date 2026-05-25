"""
進階擬合測試：sin_cos 殘差 + two-stage (DE→LM) 從爛初始值收斂
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ellipsometry.core.fitter import Fitter
from ellipsometry.core.io import read_reflection, read_transmission


def make_config(init_thickness, target=285, two_stage=False,
                delta_mode='sin_cos', loss='linear'):
    return {
        'layers': [
            {'name': 'air', 'material': 'air', 'thickness': 'infinite'},
            {'name': 'SiO2', 'material': 'SiO2',
             'thickness': init_thickness,
             'coherent': True,
             'fit_thickness': True,
             'thickness_bounds': [10, 800]},
            {'name': 'Si', 'material': 'Si',
             'thickness': 1_000_000, 'coherent': False, 'fit_thickness': False},
        ],
        'fit': {
            'target': 'both',
            'method': 'leastsq',
            'max_iter': 35,
            'delta_residual_mode': delta_mode,
            'loss': loss,
            'two_stage': two_stage,
            'weighting': {'sigma_psi': 0.5, 'sigma_delta': 1.0},
        },
    }


def run_one(name, cfg, truth):
    print('\n' + '='*72)
    print(f'  {name}')
    print('='*72)
    data = read_reflection('data/sample2_SiO2_285_Si/R_long.dat')
    fitter = Fitter(cfg, data, None)
    res = fitter.fit(verbose=True)
    err = abs(res.params['L1__thickness'] - truth)
    print(f'\n  ★ 收斂 = {res.params["L1__thickness"]:.2f} nm (truth {truth}), 誤差 {err:.3f} nm')
    return res, err


if __name__ == '__main__':
    # 測 A: sin_cos 殘差 vs wrap，給好初始值
    run_one('A1. delta_mode=sin_cos, init=200',
            make_config(200, delta_mode='sin_cos'), 285)
    run_one('A2. delta_mode=wrap, init=200',
            make_config(200, delta_mode='wrap'), 285)

    # 測 B: 兩段式 fit 從爛初始值（離 truth 很遠）
    run_one('B1. 單階段 LM, init=600 (truth 285)',
            make_config(600, two_stage=False), 285)
    run_one('B2. 兩階段 DE→LM, init=600',
            make_config(600, two_stage=True), 285)

    # 測 C: robust loss
    run_one('C1. loss=huber, init=200',
            make_config(200, loss='huber'), 285)
