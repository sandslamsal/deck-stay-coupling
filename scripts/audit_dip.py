# -*- coding: utf-8 -*-
"""Is 'the spectrum shows ONE peak' physics, or the analysis settings?

Three curves at exact tuning, stay channel, no measurement noise and no
realisation noise: the true PSD, the ensemble-mean Welch spectrum at a
stated segment length, and that same curve after the three-bin decibel
smoothing the campaign's peak counter applies.  The dip depth of each is
printed in dB, against the manuscript's 3 dB convention.
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, HERE)
from run_damping import BRIDGE, FBAND
from simulate_records import RecordSimulator, expected_welch
from oma_fdd import _smooth_db
from scipy.signal import find_peaks

T_TUNE = 151.316e3     # the campaign's exact-tuning tension (check [A])

def dip_db(f, db):
    pk, _ = find_peaks(db)
    if len(pk) < 2:
        return 0.0, 0
    o = np.argsort(-db[pk])[:2]
    i, j = sorted(pk[o])
    return float(min(db[i], db[j]) - db[i:j+1].min()), len(pk)

print("stay auto-spectrum at exact tuning, T = %.3f kN" % (T_TUNE/1e3))
print("dip depth in dB of the smaller peak over the trough;")
print("the manuscript's merge convention is 3 dB.\n")
print("%6s %6s | %9s | %-24s | %-24s" % ("zeta%", "u", "true PSD",
      "mean Welch 8192 (raw/sm3)", "mean Welch 32768 (raw/sm3)"))
for zeta in (0.001, 0.002, 0.005, 0.010, 0.020, 0.030):
    sim = RecordSimulator(T_TUNE, zeta)
    u = sim.s_split / (2*zeta)
    sep = sim.f_hi - sim.f_lo
    lo, hi = sim.f_lo - 3*sep, sim.f_hi + 3*sep
    ff = np.linspace(lo, hi, 40001)
    dbt = 10*np.log10(sim.psd_recorded(ff, sim.dof_stay))
    d_true, _ = dip_db(ff, dbt)
    cells = []
    for nps in (8192, 32768):
        df = sim.fs/nps
        fb = np.arange(lo, hi, df)
        E = expected_welch(sim, fb, nps)
        raw = 10*np.log10(E)
        d_raw, n_raw = dip_db(fb, raw)
        sm, nsm = _smooth_db(fb, E, 3*df)
        d_sm, n_sm = dip_db(fb, sm)
        cells.append("%5.1f dB (n=%d) / %5.1f dB (n=%d)" % (d_raw, n_raw, d_sm, n_sm))
    print("%6.1f %6.2f | %6.1f dB | %-24s | %-24s"
          % (100*zeta, u, d_true, cells[0], cells[1]))
