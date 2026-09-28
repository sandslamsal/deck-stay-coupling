# -*- coding: utf-8 -*-
"""What a damping inequality does to the amplitude channel.

At exact tuning the frequencies carry no first-order information about the
tension and the amplitude balance of the two peaks carries all of it. That
channel is only as good as the assumption behind it, that the two branches
are equally damped: the height of a resonance goes as its residue divided by
its damping, so unequal damping shifts the balance for a reason that has
nothing to do with the mixing, and an analyst reading the balance attributes
the shift to a detuning that is not there.

This script prices that confusion in tension, in the study's own convention.

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
S_WORKED = 0.0234                      # the worked bridge's split


def apparent_error(r, s):
    """Tension error mimicked by a branch-damping ratio r at exact tuning.

    Peak height goes as residue/zeta, so at exact tuning, where the true
    residues are equal, the observed height ratio is r. Read as a mixing
    imbalance it implies cos(2 alpha) = (r-1)/(r+1), hence an apparent
    detuning d from cos(2 alpha) = d/sqrt(d^2+s^2). The tension the analyst
    then infers differs from the truth by the difference of the two branch
    laws, eps = sqrt(d^2+s^2) - |d|, evaluated at the apparent and the true
    detuning.
    """
    rho = (r - 1.0) / (r + 1.0)
    d_app = rho * s / np.sqrt(1.0 - rho ** 2)
    eps_true = s                                     # d = 0
    eps_app = np.sqrt(d_app ** 2 + s ** 2) - abs(d_app)
    return d_app, eps_true - eps_app


def main():
    print("Tension error mimicked by unequal branch damping, at exact tuning,")
    print(f"on the worked bridge (s = {100*S_WORKED:.2f} %).\n")
    print(f"{'zeta_hi/zeta_lo':>16}{'apparent d':>13}{'tension error':>16}"
          f"{'as a share of s':>18}")
    rows = []
    for r in (1.02, 1.05, 1.10, 1.25, 1.50, 2.00, 3.00):
        d_app, err = apparent_error(r, S_WORKED)
        print(f"{r:>16.2f}{d_app:>13.5f}{100*err:>15.3f}%{err/S_WORKED:>17.3f}")
        rows.append(dict(ratio=r, d_apparent=d_app, err=err,
                         share_of_s=err / S_WORKED, s=S_WORKED))

    print("\n  the same, across splits, at a 25 per cent inequality")
    print(f"{'s (%)':>8}{'tension error (%)':>20}")
    for s in (0.0032, 0.0100, 0.0234, 0.0500):
        _, err = apparent_error(1.25, s)
        print(f"{100*s:>8.2f}{100*err:>19.3f}")
        rows.append(dict(ratio=1.25, d_apparent=np.nan, err=err,
                         share_of_s=err / s, s=s))

    # the limit: as the ratio diverges the apparent detuning diverges with it
    # and eps_app tends to zero, so the mimicked error approaches s and never
    # exceeds it. The channel degrades, it does not invert.
    print("\n  limit as the damping ratio diverges:")
    for r in (10.0, 100.0, 1e4):
        _, err = apparent_error(r, S_WORKED)
        print(f"    r = {r:>8.0f}:  {100*err:.4f} %  "
              f"({err/S_WORKED:.4f} of s)")
    print("  the mimicked error approaches the split from below and never")
    print("  exceeds it, so the channel degrades gracefully.")

    os.makedirs(DATA, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "dampingasym.csv"), index=False)
    print(f"\n  wrote {DATA}/dampingasym.csv")


if __name__ == "__main__":
    main()
