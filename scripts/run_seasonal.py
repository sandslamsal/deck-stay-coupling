# -*- coding: utf-8 -*-
"""Revision 1, R1.9: the seasonal projections as scenario intervals, and
the sample size the ranking test needs.

Section 10 passes a seasonal detuning drift through the bias law for three
stays (far from, near, and at a crossing) at one drift amplitude, 0.01.
The amplitude is inferred from sensitivities measured on different bridges
and spans 0.005 to 0.025, so the figures for the far and near stays are
scenario values, not measurements. This script tabulates the peak-to-peak
apparent tension swing over that amplitude band, and confirms that the
at-crossing swing is the full veering gap 2s whatever the amplitude.

It also gives the number of stays a Spearman rank test between seasonal
amplitude and |d| needs, N = ((z_{1-a/2} + z_{1-b}) / artanh(rho_s))^2 + 3
(Fisher z), at 80 and 90 per cent power and a 5 per cent level.

Writes data/seasonal.csv and data/seasonal_power.csv.

Run:  python3 scripts/run_seasonal.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

S = 0.0234                                   # worked bridge split (fig_temperature.py)
D0 = {"far": 0.10, "near": 0.03, "at": 0.00}
AMPS = (0.005, 0.01, 0.025)
AMP_GRID = np.linspace(0.0005, 0.20, 400)


def eps_signed(d, s):
    mag = np.sqrt(d * d + s * s) - np.abs(d)
    return np.where(d >= 0, mag, -mag)


def swing(d0, amp, s=S):
    t = np.linspace(0.0, 1.0, 4001)
    e = eps_signed(d0 + amp * np.sin(2 * np.pi * t), s)
    return 100.0 * (e.max() - e.min())


def main():
    rows = []
    for k, d0 in D0.items():
        for a in AMPS:
            rows.append(dict(stay=k, d0=d0, amplitude=a, swing_pct=swing(d0, a)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA, "seasonal.csv"), index=False)
    print("peak-to-peak apparent tension swing (%), worked bridge s = 2.34 %:")
    print(out.pivot(index="stay", columns="amplitude", values="swing_pct").round(3).to_string())
    at = np.array([swing(0.0, a) for a in AMP_GRID])
    print(f"\nat-crossing swing over amplitude 0.0005-0.20: {at.min():.3f} to {at.max():.3f} %"
          f"  (2s = {200*S:.3f} %)")

    z_a = norm.ppf(0.975)
    prow = []
    for rho in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        for power in (0.80, 0.90):
            z_b = norm.ppf(power)
            N = ((z_a + z_b) / np.arctanh(rho)) ** 2 + 3
            prow.append(dict(rho_s=rho, power=power, N=int(np.ceil(N))))
    p = pd.DataFrame(prow)
    p.to_csv(os.path.join(DATA, "seasonal_power.csv"), index=False)
    print("\nstays needed for a Spearman test at the 5 % level:")
    print(p.pivot(index="rho_s", columns="power", values="N").to_string())
    print("wrote seasonal.csv and seasonal_power.csv")


if __name__ == "__main__":
    main()
