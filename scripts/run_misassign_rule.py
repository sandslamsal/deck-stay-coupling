# -*- coding: utf-8 -*-
"""Revision 1, R1.6: a single-sensor rule against branch mis-assignment.

Section 6 reports that inside a 2.18 kN window around the crossing of the
worked bridge the taller peak is the deck-rooted one, so that a field pick
of the fundamental takes the wrong branch and doubles the bias. The rule
tested here needs nothing but the same stay-mounted spectrum.

    1. Read orders two to five and fit them (multi-mode, Section 6). The
       coupling falls as 1/n, so this tension is nearly unbiased, and it
       implies an isolated fundamental f1_implied.
    2. Compare the picked fundamental with f1_implied. The deviation IS the
       displacement a crossing has imposed, whichever branch was picked,
       and its sign says which. If no peak sits within half the tolerance
       of f1_implied (a tension tolerance tol is a frequency tolerance
       tol/2), the fundamental is at a crossing and no single pick is
       safe: the tension is taken from step 1.

So the rule does not assign the branch. It makes the assignment
unnecessary, and it flags a crossing from the record alone. A first
version of this script tested "f1_implied lies between the two peaks",
which is true almost everywhere because repulsion always places the
isolated frequency between the stay branch and the deck line; that test
discriminates nothing and was dropped.

The worked bridge is traversed through its crossing. At each tension the
coupled modes are solved, the stay-mounted spectrum's peaks are screened
by the harmonic comb exactly as scripts/run_identify2.py does, orders two
to five are fitted, and the rule is scored against the true error of the
fundamental-only reading on the stay branch and on the taller peak.

Writes data/misassign_rule.csv.

Run:  python3 scripts/run_misassign_rule.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from cablefe import (CableDeck, invert_multimode, pick_peaks,  # noqa: E402
                     screened_pick, tensioned_beam_freq)

DATA = os.path.join(ROOT, "data")

BRIDGE = dict(Ld=80.0, EId=2.0e9, md=1000.0,
              Lc=25.0, EIc=1.2e4, mc=5.5, EA=1.4e8,
              theta=np.deg2rad(35.0), nd=40, nc=40)
BAND = (2.6, 4.2)                     # the crossing sits here
WINDOW = (146.3e3, 157.0e3)           # the mis-assignment window of Section 6
TOL = 0.02                            # tension tolerance the rule is scored at


def main():
    Ts = np.linspace(135e3, 170e3, 141)
    rows = []
    for T in Ts:
        cd = CableDeck(T=T, **BRIDGE)
        f, Phi = cd.modes(60)
        es = cd.energy_split(Phi)
        p = cd.cable_sensor_dof(2.0)
        fpk, apk = pick_peaks(f, Phi, p, nmax=40, rel_floor=0.02)
        fc, nc = screened_pick(fpk, apk, nmax=5, tol=0.04)
        keep = nc >= 2                            # orders two to five only
        if keep.sum() < 3:
            continue
        T25, _ = invert_multimode(fc[keep], cd.Lc, cd.mc, n=nc[keep])
        f1_implied = tensioned_beam_freq(1, cd.Lc, T25, cd.EIc, cd.mc)
        f1_iso = tensioned_beam_freq(1, cd.Lc, T, cd.EIc, cd.mc)
        inb = np.where((f > BAND[0]) & (f < BAND[1]))[0]
        assert len(inb) == 2, (T, f[inb])
        f_lo, f_hi = f[inb[0]], f[inb[1]]
        stay_upper = es[inb[1]] > es[inb[0]]
        taller_upper = np.abs(Phi[p, inb[1]]) > np.abs(Phi[p, inb[0]])
        f_stay = f_hi if stay_upper else f_lo
        f_tall = f_hi if taller_upper else f_lo
        eps_stay = (f_stay / f1_iso) ** 2 - 1.0     # branch law, right branch
        eps_tall = (f_tall / f1_iso) ** 2 - 1.0     # what a field pick returns
        dev_lo = abs(f_lo / f1_implied - 1.0)
        dev_hi = abs(f_hi / f1_implied - 1.0)
        dev_near = min(dev_lo, dev_hi)
        flag = dev_near > TOL / 2.0
        rows.append(dict(T=T, f_lo=f_lo, f_hi=f_hi, f1_iso=f1_iso,
                         T25=T25, err_T25_pct=100 * (T25 / T - 1),
                         f1_implied=f1_implied,
                         dev_implied_pct=100 * (f1_implied / f1_iso - 1),
                         eps_stay_pct=100 * eps_stay, eps_tall_pct=100 * eps_tall,
                         dev_nearest_pct=100 * dev_near, flag=int(flag),
                         unsafe_stay=int(abs(eps_stay) > TOL),
                         unsafe_tall=int(abs(eps_tall) > TOL),
                         in_window=int(WINDOW[0] <= T <= WINDOW[1]),
                         stay_upper=int(stay_upper), taller_upper=int(taller_upper),
                         misassigned=int(stay_upper != taller_upper)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA, "misassign_rule.csv"), index=False)

    w = out[out.in_window == 1]
    m = out[out.misassigned == 1]
    print(f"{len(out)} tensions, {len(w)} inside the {WINDOW[0]/1e3:.1f}-{WINDOW[1]/1e3:.1f} kN window,"
          f" {len(m)} where the taller peak is the wrong branch")
    print(f"orders 2-5 fit: tension error median {out.err_T25_pct.abs().median():.3f} %, "
          f"worst {out.err_T25_pct.abs().max():.3f} % (at T = {out.loc[out.err_T25_pct.abs().idxmax(),'T']/1e3:.1f} kN);"
          f" inside the window worst {w.err_T25_pct.abs().max():.3f} %")
    print(f"implied fundamental vs exact isolated: median {out.dev_implied_pct.abs().median():.3f} %, worst {out.dev_implied_pct.abs().max():.3f} %")
    print(f"fundamental-only reading, taller peak: worst error {out.eps_tall_pct.abs().max():.2f} %;"
          f" right branch: worst {out.eps_stay_pct.abs().max():.2f} %")
    agree_tall = (out.flag == out.unsafe_tall).mean()
    agree_stay = (out.flag == out.unsafe_stay).mean()
    print(f"rule (no peak within {100*TOL/2:.0f} % of the implied fundamental) against |error| > {100*TOL:.0f} %:"
          f" agreement {100*agree_tall:.0f} % on the taller-peak reading, {100*agree_stay:.0f} % on the right branch")
    fired = out[out.flag == 1]
    if len(fired):
        print(f"rule fires from {fired['T'].min()/1e3:.1f} to {fired['T'].max()/1e3:.1f} kN;"
              f" inside the window {100*w.flag.mean():.0f} % of states, on mis-assigned states {100*m.flag.mean():.0f} %")
    print("wrote", os.path.join(DATA, "misassign_rule.csv"))


if __name__ == "__main__":
    main()
