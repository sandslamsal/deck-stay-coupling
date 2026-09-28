# -*- coding: utf-8 -*-
"""Revision 1, R1.8: resolvability of the veering pair when the two branches
carry unequal residues and unequal damping.

Appendix B derives the dip-existence threshold u > sqrt(sqrt5 - 2) and the
3 dB threshold u > 1.14 for two resonances of EQUAL residue and EQUAL
damping, and Section 5 quotes the population fractions that follow. This
script drops both equalities. In units of the mean damping zeta_bar and
with the pair centred at x = 0,

    H(x) = 1 / (x + u + i a)  +  rho / (x - u + i b),
    a = zeta_1 / zeta_bar = 2 / (1 + kappa),   b = kappa a,   kappa = zeta_2 / zeta_1,
    u = s / (2 zeta_bar),

and the smallest u at which |H|^2 shows two maxima (dip exists) and two
maxima with a 3 dB dip (a peak picker acts) is found by bisection for a
grid of residue ratios rho and damping ratios kappa. The population
fractions of the graded campaign that are unresolvable at zeta_bar = 0.2,
0.5, 1 and 2 per cent follow from each threshold.

Nothing here assumes the direction of the answer. Unequal residues are
expected to merge the pair earlier; unequal damping is not obvious, since
the sharper of the two resonances can stand as a separate maximum on the
flank of the broader one.

Writes data/resolve_sweep.csv.

Run:  python3 scripts/run_resolve_sweep.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

RHOS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
KAPPAS = (0.5, 0.75, 1.0, 1.5, 2.0)
ZETAS = (0.002, 0.005, 0.010, 0.020)
U_DIP_EQUAL = np.sqrt(np.sqrt(5.0) - 2.0)


def hmag2(x, u, rho, kappa):
    a = 2.0 / (1.0 + kappa)
    b = kappa * a
    return np.abs(1.0 / (x + u + 1j * a) + rho / (x - u + 1j * b)) ** 2


def census(u, rho, kappa):
    """Number of maxima of |H|^2 and the depth (dB) of the dip between the
    two outermost maxima, measured below the lower of the two."""
    x = np.linspace(-(u + 10.0), u + 10.0, 200001)
    db = 10.0 * np.log10(hmag2(x, u, rho, kappa))
    pk, _ = find_peaks(db, prominence=1e-7)
    if len(pk) < 2:
        return len(pk), 0.0
    lo, hi = pk[0], pk[-1]
    dip = db[lo:hi + 1].min()
    return len(pk), min(db[lo], db[hi]) - dip


def bisect(pred, lo, hi, it=36):
    assert not pred(lo) and pred(hi), "predicate not bracketed"
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if pred(mid) else (mid, hi)
    return 0.5 * (lo + hi)


def thresholds(rho, kappa):
    u_dip = bisect(lambda u: census(u, rho, kappa)[0] >= 2, 0.02, 12.0)
    u_3db = bisect(lambda u: census(u, rho, kappa)[1] >= 3.0, u_dip, 20.0)
    return u_dip, u_3db


def main():
    c = pd.read_csv(os.path.join(DATA, "campaign.csv"))
    g = c[(c.mac > 0.5) & (c.xi > 150)]
    s = g.s.to_numpy()

    # the equal case must reproduce Appendix B before anything else is trusted
    u_dip, u_3db = thresholds(1.0, 1.0)
    assert abs(u_dip - U_DIP_EQUAL) < 2e-3, (u_dip, U_DIP_EQUAL)
    assert abs(u_3db - 1.1398) < 3e-3, u_3db
    print(f"equal residues and damping: u_dip = {u_dip:.4f} (closed form "
          f"{U_DIP_EQUAL:.4f}), u_3db = {u_3db:.4f} (Appendix B 1.1398)")

    rows = []
    for rho in RHOS:
        for kappa in KAPPAS:
            ud, u3 = thresholds(rho, kappa)
            row = dict(rho=rho, kappa=kappa, u_dip=ud, u_3db=u3,
                       s_over_zeta_dip=2 * ud, s_over_zeta_3db=2 * u3)
            for z in ZETAS:
                row[f"frac_unres_dip_{100*z:g}pct"] = float(np.mean(s <= 2 * ud * z))
                row[f"frac_unres_3db_{100*z:g}pct"] = float(np.mean(s <= 2 * u3 * z))
            rows.append(row)
            print(f"  rho {rho:5.2f}  kappa {kappa:4.2f}:  u_dip {ud:6.3f}  u_3db {u3:6.3f}"
                  f"   unresolvable at 1 %: dip {row['frac_unres_dip_1pct']:.3f}"
                  f"  3 dB {row['frac_unres_3db_1pct']:.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA, "resolve_sweep.csv"), index=False)

    print("\nrange over the grid at zeta_bar = 1 %:")
    for col, lab in (("frac_unres_dip_1pct", "dip exists"), ("frac_unres_3db_1pct", "3 dB")):
        print(f"  {lab:10s}: {out[col].min():.3f} to {out[col].max():.3f}"
              f"  (equal case {out[(out.rho == 1) & (out.kappa == 1)][col].iloc[0]:.3f})")
    lower = (out.u_dip < U_DIP_EQUAL - 1e-3).sum()
    print(f"  cases where the dip survives to SMALLER u than the equal case: {lower} of {len(out)}")
    print("wrote", os.path.join(DATA, "resolve_sweep.csv"))


if __name__ == "__main__":
    main()
