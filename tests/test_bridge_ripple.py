"""power_bridge.metrics: settled bus ripple and switching-edge spikes reported separately (I1)."""
import numpy as np

from pcbqs.power_bridge import metrics


def run(f=100e3, ripple=0.2, spike=3.0):
    """Six cycles on an adaptive-looking grid: a sinusoidal bus ripple, plus one 20 ns spike per cycle that only the
    solver's own fine steps catch."""
    t = np.linspace(0, 6 / f, 6 * 4000 + 1)
    edge = np.concatenate([k / f + 0.3 / f + np.array([-10e-9, 0.0, 10e-9]) for k in range(6)])
    t = np.unique(np.concatenate([t, edge]))
    bus = 24 + ripple / 2 * np.sin(2 * np.pi * f * t)
    for k in range(6):
        bus[np.argmin(abs(t - (k / f + 0.3 / f)))] += spike
    sig = np.sin(2 * np.pi * f * t)
    result = {'time_s': t, 'bus.V': bus, 'PRIMARY.winding_I': sig, 'SECONDARY.winding_I': sig, 'supply.I': sig,
              'load.V': sig}
    result['_measures'] = dict(load_power_W=10.0, input_power_W=11.0, gate_drive_power_W=0.1, primary_rms_A=1.0,
                               secondary_rms_A=1.0, primary_peak_A=1.4, bus_ripple_V=float(np.ptp(bus[t >= 5 / f])))
    return result


def test_settled_ripple_and_spikes_are_separate():
    m = metrics(run(), 100e3, match=None)
    assert abs(m['bus_ripple_V'] - 0.2) < 0.01                 # the sampled final cycle: the sinusoid only
    assert m['bus_spike_pp_V'] > 3.0                            # the solver's steps: the spike too
