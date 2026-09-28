# -*- coding: utf-8 -*-
"""Miss and false-alarm rates of the screen |d| >= (s^2 - tol^2) / (2 tol)
under input uncertainty, on the graded designs of the campaign.

Per draw, the anchorage ordinate phi_a has a normal error of coefficient of
variation cov_phi (s' = s |1 + e|) and the deck frequency a normal error of
standard deviation sig_f (d' = d - (1 - d) e_f). Margins scanned: a factor
gamma on s and a subtraction delta from |d|. The miss rate is over the truly
unsafe designs and the false-alarm rate over the truly safe ones.
Reads data/campaign.csv; writes data/screen_uncertainty.csv.
Run:  python3 scripts/run_screen_uncertainty.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

COV_PHI = (0.0, 0.05, 0.10, 0.20, 0.30)
SIG_F = (0.0, 0.01, 0.02, 0.03)
TOLS = (0.02, 0.05, 0.10)
GAMMAS = (1.0, 1.1, 1.2, 1.3, 1.5, 2.0)
DELTAS = (0.0, 0.01, 0.02, 0.03, 0.05)
NDRAW = 4000
SEED = 20260927


def main():
    c = pd.read_csv(os.path.join(DATA, "campaign.csv"))
    g = c[(c.mac > 0.5) & (c.xi > 150)].reset_index(drop=True)
    s = g.s.to_numpy()
    d = g.d.to_numpy()
    truth = np.abs(g.eps_coupling.to_numpy())      # coupled-model error, the reference
    n = len(g)
    rng = np.random.default_rng(SEED)
    rows = []
    print(f"{n} graded designs; truly unsafe at tol 2/5/10 %: "
          + "/".join(str(int((truth > t).sum())) for t in TOLS))

    for cov in COV_PHI:
        for sf in SIG_F:
            e_phi = rng.normal(0.0, cov, size=(NDRAW, n)) if cov else np.zeros((NDRAW, n))
            e_f = rng.normal(0.0, sf, size=(NDRAW, n)) if sf else np.zeros((NDRAW, n))
            s_p = s[None, :] * np.abs(1.0 + e_phi)
            d_p = d[None, :] - (1.0 - d[None, :]) * e_f
            for tol in TOLS:
                unsafe = truth > tol
                safe = ~unsafe
                for gam in GAMMAS:
                    for dl in DELTAS:
                        sg = gam * s_p
                        passes = (np.abs(d_p) - dl) >= (sg ** 2 - tol ** 2) / (2.0 * tol)
                        # per-design probabilities, then rates over the class
                        p_pass = passes.mean(axis=0)
                        miss = p_pass[unsafe].mean() if unsafe.any() else np.nan
                        alarm = (1.0 - p_pass[safe]).mean() if safe.any() else np.nan
                        rows.append(dict(cov_phi=cov, sig_f=sf, tol=tol, gamma=gam, delta_d=dl,
                                         n_unsafe=int(unsafe.sum()), n_safe=int(safe.sum()),
                                         miss_rate=miss, false_alarm_rate=alarm,
                                         flagged_fraction=1.0 - p_pass.mean()))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA, "screen_uncertainty.csv"), index=False)

    # summary tables at tol = 2 %, no margin
    pd.set_option("display.width", 200)
    base = out[(out.tol == 0.02) & (out.gamma == 1.0) & (out.delta_d == 0.0)]
    print("\ntol = 2 %, no margin: miss rate among truly unsafe designs (rows cov_phi, cols sig_f)")
    print(base.pivot(index="cov_phi", columns="sig_f", values="miss_rate").round(3).to_string())
    print("\ntol = 2 %, no margin: false-alarm rate among truly safe designs")
    print(base.pivot(index="cov_phi", columns="sig_f", values="false_alarm_rate").round(3).to_string())

    print("\nsmallest margin holding the miss rate <= 1 % at tol = 2 %, and its false-alarm rate:")
    print("  (gamma on s alone | delta on |d| alone | cheapest pair by false alarms)")
    for cov in COV_PHI:
        for sf in SIG_F:
            sub = out[(out.tol == 0.02) & (out.cov_phi == cov) & (out.sig_f == sf) & (out.miss_rate <= 0.01)]
            g_only = sub[sub.delta_d == 0.0].sort_values("gamma").head(1)
            d_only = sub[sub.gamma == 1.0].sort_values("delta_d").head(1)
            best = sub.sort_values("false_alarm_rate").head(1)
            fmt = lambda r, k: (f"{k}={r[k].iloc[0]:g} fa={r.false_alarm_rate.iloc[0]:.3f}" if len(r) else "none")
            print(f"  cov {cov:.2f} sig_f {sf:.2f}:  {fmt(g_only,'gamma'):28s} | {fmt(d_only,'delta_d'):28s} | "
                  + (f"gamma={best.gamma.iloc[0]:g} delta={best.delta_d.iloc[0]:g} fa={best.false_alarm_rate.iloc[0]:.3f}" if len(best) else "none"))
    print("\nwrote", os.path.join(DATA, "screen_uncertainty.csv"))


if __name__ == "__main__":
    main()
