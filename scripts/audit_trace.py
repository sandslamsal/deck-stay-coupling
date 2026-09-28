# -*- coding: utf-8 -*-
"""Does the trace rule need SSI, or only two frequencies?

The campaign credits its best result to a two-pole rule read off the
subspace poles.  The rule itself, f_s^2 = f_+^2 + f_-^2 - f_d^2, needs two
frequencies and the deck frequency; it does not care where the two
frequencies came from.  This script applies exactly the same rule to the
two tallest PICKED peaks of the same stay auto-spectrum, on the same
records, so that the gain from the identity is separated from the gain
from mode fitting.
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, HERE)
from run_damping import BRIDGE, FBAND
from simulate_records import RecordSimulator
from oma_fdd import cpsd_welch, peak_pick, errors, SMOOTH_BINS
import run_oma as R

ZETAS = (0.001, 0.002, 0.005, 0.010, 0.020, 0.030)
SEEDS = (0, 1, 2, 3, 4)
NPS = 8192

rows = []
T0 = R.tension_for_d(0.0)
tr = R.model_truth(T0)
for zeta in ZETAS:
    sim = RecordSimulator(T0, zeta)
    for dur in (600.0, 3600.0):
        for seed in SEEDS:
            rec = sim.record(duration=dur, snr_db=20.0, seed=seed)
            X = np.vstack([rec["a_stay"], rec["a_deck"]])
            f, G, n_seg, k_eff = cpsd_welch(X, sim.fs, nperseg=NPS)
            hw = SMOOTH_BINS * sim.fs / NPS
            pp = peak_pick(f, G[:, 0, 0].real, k_eff, hw, band=FBAND)
            fs_sorted = pp.get("f_sorted", np.array([]))
            # the same reading the campaign gives SSI: the two tallest
            f_tr = np.nan
            if len(fs_sorted) >= 2:
                a, b = float(fs_sorted[0]), float(fs_sorted[1])
                q = a*a + b*b - tr["f_deck"]**2
                f_tr = np.sqrt(q) if q > 0 else np.nan
            e_top = errors(pp["f_top"], T0, tr["f_iso1"])
            e_tr = errors(f_tr, T0, tr["f_iso1"])
            rows.append(dict(zeta=zeta, duration=dur, seed=seed,
                             n_peaks=pp["n"], eps_pp=e_top[2], errT_pp=e_top[1],
                             eps_pptrace=e_tr[2], errT_pptrace=e_tr[1]))
d = pd.DataFrame(rows)
d.to_csv(os.path.join(ROOT, "data", "audit_trace.csv"), index=False)

O = pd.read_csv(os.path.join(ROOT, "data", "oma.csv"))
O = O[(O.arm == "grid") & (O.snr_db == 20.0) & (np.abs(O.d) < 1e-9)]

def rms(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return np.sqrt(np.mean(x**2)) if len(x) else np.nan

print("Coupling error at exact tuning, RMS per cent, 20 dB, 5 seeds.")
print("PP TRACE is the campaign's own trace identity applied to the two")
print("tallest peaks of the SAME single-channel spectrum peak picking uses.")
print()
for dur in (600.0, 3600.0):
    print("  %.0f s" % dur)
    print("   %6s %8s %10s %10s %8s %8s" % ("zeta%", "pp", "PP TRACE", "n_pp>=2",
                                            "SSItrace", "n_SSI"))
    for zeta in ZETAS:
        s = d[(d.zeta == zeta) & (d.duration == dur)]
        o = O[(np.abs(O.zeta - zeta) < 1e-9) & (O.duration == dur)]
        n2 = int((s.n_peaks >= 2).sum())
        nssi = int(np.isfinite(o.eps_cov2_tr_pct).sum())
        print("   %6.1f %8.2f %10.2f %10d %8.2f %8d"
              % (100*zeta, rms(s.eps_pp), rms(s.eps_pptrace), n2,
                 rms(o.eps_cov2_tr_pct), nssi))
