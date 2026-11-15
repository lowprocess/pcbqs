"""An inductor's losses worked out after the run (circuit/windings.py, R-E10): the iGSE against Steinmetz's own
sine and its closed form for a triangle, the AC resistance's curve, and a buck whose inductor has both."""
import math
import shutil

import numpy as np
import pytest

from pcbqs.circuit import model as M
from pcbqs.circuit import run
from pcbqs.circuit import windings as W

CORE = dict(k=8.0, alpha=1.4, beta=2.6, turns=12, Ae_m2=20e-6, Ve_m3=1.5e-6)


def test_the_igse_of_a_sine_is_steinmetz():
    f, peak = 100e3, 0.08                                               # 80 mT at 100 kHz
    t = np.linspace(0, 1 / f, 20001)
    v = CORE['turns'] * CORE['Ae_m2'] * peak * 2 * math.pi * f * np.cos(2 * math.pi * f * t)
    want = CORE['Ve_m3'] * CORE['k'] * f ** CORE['alpha'] * peak ** CORE['beta']
    assert W.igse_loss(t, v, CORE) == pytest.approx(want, rel=1e-5)


def test_the_igse_of_a_triangle_is_its_closed_form():
    f, D, swing = 200e3, 0.3, 0.1                                       # a buck's flux: up for D, down for 1 - D
    T = 1 / f
    t = np.array([0, D * T, D * T, T]) + np.array([0, 0, 1e-15, 0])
    na = CORE['turns'] * CORE['Ae_m2']
    v = np.array([1, 1, -D / (1 - D), -D / (1 - D)]) * na * swing / (D * T)
    a, b = CORE['alpha'], CORE['beta']
    ki = W.igse_coefficient(CORE['k'], a, b)
    want = CORE['Ve_m3'] * ki * swing ** b * f ** a * (D ** (1 - a) + (1 - D) ** (1 - a))
    assert W.igse_loss(t, v, CORE) == pytest.approx(want, rel=1e-6)


def test_the_ac_resistance_is_its_points_with_low_and_high_frequency_laws():
    table, dcr = [[100e3, 0.05], [1e6, 0.2]], 0.03
    r = W.ac_resistance(table, dcr, [10e3, 100e3, math.sqrt(1e11), 1e6, 4e6])
    assert r[0] == pytest.approx(dcr + 0.02 * 0.01)                    # (R - DCR) as f^2 below the first point
    assert r[1] == pytest.approx(0.05) and r[3] == pytest.approx(0.2)
    assert r[2] == pytest.approx(math.sqrt(0.05 * 0.2))                 # log-log between
    assert r[4] == pytest.approx(0.4)                                   # sqrt(f) above the last


def test_core_k_si_converts_a_datasheets_units():
    k, a, b = 2.0, 1.3, 2.5                                             # mW/cm^3, kHz, mT
    si = W.core_k_si(k, a, b)
    assert si * 50e3 ** a * 0.1 ** b == pytest.approx(1e3 * k * 50 ** a * 100 ** b)


@pytest.mark.parametrize('change, says', [
    (dict(AC_R_ohm=[[1e6, 0.1], [1e5, 0.2]]), 'rising'),
    (dict(AC_R_ohm=[[1e5, 0.01]]), 'below its DCR_ohm'),
    (dict(core_loss=dict(k=1, alpha=1.5)), 'needs k, alpha'),
    (dict(core_loss=CORE, core_R_ohm=1e4), 'keep one'),
])
def test_what_cannot_be_used_is_said_when_the_model_is_resolved(change, says):
    m = M.template('04_buck')
    L1 = next(p for p in m['parts'] if p['ref'] == 'L1')
    L1.update(change)
    with pytest.raises(M.ModelError, match=says):
        M.resolve(M.validate(m))


@pytest.mark.skipif(not shutil.which('ngspice'), reason='ngspice not installed')
def test_a_bucks_inductor_adds_its_ac_and_core_losses_to_the_case_and_the_thermal_model(tmp_path):
    from pcbqs.thermal.problem import losses_of
    m = M.template('04_buck')
    L1 = next(p for p in m['parts'] if p['ref'] == 'L1')
    L1.update(AC_R_ohm=[[1.0, 0.09], [1e9, 0.09]], core_loss=CORE)     # a flat 90 mOhm: (R - DCR) x the AC's mean square
    summary = run(m, tmp_path, jobs=1, log=lambda s: None)
    assert summary['complete'] and not summary['failed'], summary['failed']
    c, = summary['cases']
    p = c['parts']['L1']
    ac2 = p['rms_A'] ** 2 - p['avg_A'] ** 2
    assert p['AC_winding_loss_W'] == pytest.approx((0.09 - 0.03) * ac2, rel=0.01)
    assert p['core_loss_W'] > 0
    assert p['added_loss_W'] == pytest.approx(p['AC_winding_loss_W'] + p['core_loss_W'])
    assert c['added_loss_W'] == pytest.approx(p['added_loss_W'])
    assert c['efficiency'] == pytest.approx(c['load_power_W'] / (c['load_power_W'] + c['loss_W']), rel=2e-3)
    assert any('worked out after the run' in n for n in c['notes'])
    heat, _ = losses_of(c, M.resolve(m))
    assert heat['L1'] == pytest.approx(p['power_W'] + p['added_loss_W'])
