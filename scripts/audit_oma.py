# -*- coding: utf-8 -*-
"""Independent audit of the OMA campaign.  Reads data/oma.csv only."""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = pd.read_csv(os.path.join(ROOT, "data", "oma.csv"))
pd.set_option("display.width", 220)

def main_grid(dur, snr=20.0):
    return D[(D.arm == "grid") & (D.duration == dur) & (D.snr_db == snr)]

print("rows", len(D), "arms", D.arm.unique().tolist())
print("durations", sorted(D.duration.unique()), "snr", sorted(D.snr_db.unique()))

# ---- 1. whole-traverse separation, the 'SSI pair found' column ----------
print("\n[1] SSI-COV(stay+deck) both-branch rate over the WHOLE traverse")
for dur in (600.0, 3600.0):
    g = main_grid(dur)
    r = g.groupby("zeta")["both_cov2"].mean() * 100
    print("  %5.0f s: " % dur, " ".join("%.0f" % v for v in r))

# ---- 2. trace rule: how many records does it actually score? ------------
print("\n[2] trace rule availability and the effect of dropping failures")
for dur in (600.0, 3600.0):
    g = main_grid(dur)
    tune = g[np.abs(g.d) < 1e-6]
    print("  %.0f s, at exact tuning (5 records per zeta)" % dur)
    print("    %6s %6s %8s %8s %8s %8s" % ("zeta%","n_ok","RMS_ok","RMS_all*","pp_RMS","both%"))
    for z, sub in tune.groupby("zeta"):
        e = sub["eps_cov2_tr_pct"].to_numpy()
        ok = np.isfinite(e)
        rms_ok = np.sqrt(np.nanmean(e[ok]**2)) if ok.any() else np.nan
        # * substitute the amplitude reading when the trace rule returns nothing,
        #   which is what an engineer with one pole must do
        sub_amp = sub["eps_cov2_pct"].to_numpy()
        e_all = np.where(ok, e, sub_amp)
        rms_all = np.sqrt(np.nanmean(e_all**2))
        pp = np.sqrt(np.nanmean(sub["eps_pp_pct"].to_numpy()**2))
        print("    %6.1f %6d %8.2f %8.2f %8.2f %8.0f"
              % (100*z, ok.sum(), rms_ok, rms_all, pp, 100*sub["both_cov2"].mean()))

# ---- 3. paired comparison, whole traverse, and its scatter -------------
print("\n[3] paired trace-vs-pp over the whole traverse (errT, per cent)")
for dur in (600.0, 3600.0):
    g = main_grid(dur)
    a = np.abs(g["errT_cov2_tr_pct"]); b = np.abs(g["errT_pp_pct"])
    m = np.isfinite(a) & np.isfinite(b)
    print("  %5.0f s: n_paired %4d / %4d (%.0f%%), win %.0f%%, mean gain %+.2f"
          % (dur, m.sum(), len(g), 100*m.mean(), 100*(a[m] < b[m]).mean(),
             (a[m]-b[m]).mean()))
    # what happens if the unpaired records are scored as the amplitude read
    a2 = np.where(np.isfinite(g["errT_cov2_tr_pct"]), g["errT_cov2_tr_pct"],
                  g["errT_cov2_pct"])
    a2 = np.abs(a2); mm = np.isfinite(a2) & np.isfinite(b)
    print("           fallback-scored: n %4d, win %.0f%%, mean gain %+.2f"
          % (mm.sum(), 100*(a2[mm] < b[mm]).mean(), (a2[mm]-b[mm]).mean()))
