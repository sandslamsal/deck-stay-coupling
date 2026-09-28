# -*- coding: utf-8 -*-
"""Tension error caused by unequal branch damping at exact tuning.

At exact tuning the tension is read from the amplitude balance of the two
peaks, which assumes the two branches are equally damped. This script
computes the tension error that a damping ratio between the branches
introduces. Writes data/dampingasym.csv.

Run:  python3 scripts/run_dampingasym.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

DATA = os.path.join(ROOT, "data")
S_WORKED = 0.0234                      # the example bridge's split


def apparent_error(r, s):
    """Apparent detuning and tension error for branch-damping ratio r and split s.

    Peak height scales as residue/zeta, so at exact tuning the observed
    height ratio is r. Read as mixing, cos(2 alpha) = (r-1)/(r+1) =
    d/sqrt(d^2+s^2) gives the apparent detuning d_app. The error is the
    difference of eps = sqrt(d^2+s^2) - |d| at d = 0 and at d_app.
    Returns (d_app, error).
    """
    rho = (r - 1.0) / (r + 1.0)
    d_app = rho * s / np.sqrt(1.0 - rho ** 2)
    eps_true = s                                     # d = 0
    eps_app = np.sqrt(d_app ** 2 + s ** 2) - abs(d_app)
    return d_app, eps_true - eps_app


def main():
    print("Tension error from unequal branch damping at exact tuning,")
    print(f"on the example bridge (s = {100*S_WORKED:.2f} %).\n")
    print(f"{'zeta_hi/zeta_lo':>16}{'apparent d':>13}{'tension error':>16}"
          f"{'as a share of s':>18}")
    rows = []
    for r in (1.02, 1.05, 1.10, 1.25, 1.50, 2.00, 3.00):
        d_app, err = apparent_error(r, S_WORKED)
        print(f"{r:>16.2f}{d_app:>13.5f}{100*err:>15.3f}%{err/S_WORKED:>17.3f}")
        rows.append(dict(ratio=r, d_apparent=d_app, err=err,
                         share_of_s=err / S_WORKED, s=S_WORKED))

    print("\n  the same for several splits, at a 25 per cent damping inequality")
    print(f"{'s (%)':>8}{'tension error (%)':>20}")
    for s in (0.0032, 0.0100, 0.0234, 0.0500):
        _, err = apparent_error(1.25, s)
        print(f"{100*s:>8.2f}{100*err:>19.3f}")
        rows.append(dict(ratio=1.25, d_apparent=np.nan, err=err,
                         share_of_s=err / s, s=s))

    # as r grows, d_app grows and eps_app tends to zero, so the error tends to s
    print("\n  limit for a large damping ratio:")
    for r in (10.0, 100.0, 1e4):
        _, err = apparent_error(r, S_WORKED)
        print(f"    r = {r:>8.0f}:  {100*err:.4f} %  "
              f"({err/S_WORKED:.4f} of s)")
    print("  the error approaches the split from below and does not")
    print("  exceed it.")

    os.makedirs(DATA, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "dampingasym.csv"), index=False)
    print(f"\n  wrote {DATA}/dampingasym.csv")


if __name__ == "__main__":
    main()
