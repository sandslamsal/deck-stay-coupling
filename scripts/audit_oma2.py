# -*- coding: utf-8 -*-
"""Audit part 2: where the separation claim lives, and how tight it is."""
import os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = pd.read_csv(os.path.join(ROOT, "data", "oma.csv"))
pd.set_option("display.width", 220)
G = D[(D.arm == "grid") & (D.snr_db == 20.0)]

print("[A] both-branch rate: whole traverse vs AT THE CROSSING (|d|<=0.004)")
print("  %5s | %-22s | %-22s" % ("zeta%", "600 s  trav / cross", "3600 s trav / cross"))
for z, sub in G.groupby("zeta"):
    row = []
    for dur in (600.0, 3600.0):
        s = sub[sub.duration == dur]
        c = s[np.abs(s.d) <= 0.004]
        row.append("%3.0f%% / %3.0f%%  (n=%d)" % (100*s.both_cov2.mean(),
                                                  100*c.both_cov2.mean(), len(c)))
    print("  %5.1f | %-22s | %-22s" % (100*z, row[0], row[1]))

print("\n[B] at exact tuning only (d = 0, n = 5): both-branch rate")
print("  %5s %8s %8s %10s %10s" % ("zeta%", "600 s", "3600 s", "split/true600", "split/true3600"))
for z, sub in G.groupby("zeta"):
    t = sub[np.abs(sub.d) < 1e-9]
    o = []
    for dur in (600.0, 3600.0):
        s = t[t.duration == dur]
        o += [100*s.both_cov2.mean(),
              np.nanmean(s.split_cov2 / s.Delta)]
    print("  %5.1f %7.0f%% %7.0f%% %10.3f %10.3f" % (100*z, o[0], o[2], o[1], o[3]))

print("\n[C] how many poles does SSI-COV return in band? (3600 s, 20 dB)")
s = G[G.duration == 3600.0]
print(pd.crosstab(100*s.zeta, s.n_cov2))

print("\n[D] re-grade 'both branches' with a STRICTER match tolerance")
print("   uses the two strongest poles f_cov2_p1/p2 (exact when n_cov2 == 2)")
for dur in (600.0, 3600.0):
    s = G[G.duration == dur]
    print("  %.0f s" % dur)
    print("    %5s " % "zeta%" + " ".join("tol=%.2f%%" % (100*t) for t in (0.01,0.005,0.003,0.002)))
    for z, sub in s.groupby("zeta"):
        cells = []
        for tol in (0.01, 0.005, 0.003, 0.002):
            p1, p2 = sub.f_cov2_p1.to_numpy(), sub.f_cov2_p2.to_numpy()
            lo, hi = sub.f_lo.to_numpy(), sub.f_hi.to_numpy()
            ok = np.zeros(len(sub), bool)
            for k in range(len(sub)):
                c = np.array([v for v in (p1[k], p2[k]) if np.isfinite(v)])
                if len(c) < 2:
                    continue
                ilo = int(np.argmin(np.abs(c - lo[k])))
                ihi = int(np.argmin(np.abs(c - hi[k])))
                ok[k] = (ilo != ihi
                         and abs(c[ilo]-lo[k])/lo[k] < tol
                         and abs(c[ihi]-hi[k])/hi[k] < tol)
            cells.append(100*ok.mean())
        print("    %5.1f " % (100*z) + " ".join("%7.0f%%" % c for c in cells))
