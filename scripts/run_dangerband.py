# -*- coding: utf-8 -*-
"""Peak-picking tension bias against damping, from a two-mode pencil.

The worst picked-peak error over detuning is computed against u = s/(2 zeta)
at the median split, and the seasonal swing at the example bridge's split.
The two peaks merge below u = sqrt(sqrt(5) - 2); the worst error is
sqrt(5)/2 times s, at u = 2/sqrt(3).

Writes data/dangerband.csv.  Run:  python3 scripts/run_dangerband.py
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

U_SPLIT = np.sqrt(np.sqrt(5.0) - 2.0)          # 0.48587, u below which the two peaks merge
S_MEDIAN = 0.0032                              # median fractional split s
S_WORKED = 0.0234                              # split s of the example bridge


def pencil(d, s):
    """Two-mode pencil in omega^2, isolated stay at 1, deck at (1-d)^2.

    At exact tuning the eigenvalues are 1 +/- s, so the branch frequencies are
    1 +/- s/2 and the fractional split is s.
    """
    K = np.array([[1.0, s], [s, (1.0 - d) ** 2]])
    w, V = np.linalg.eigh(K)
    return np.sqrt(np.abs(w)), V


def _mag(x, f, V, zeta):
    H = np.zeros(np.shape(x), dtype=complex)
    for r in range(2):
        H = H + V[0, r] ** 2 / (f[r] ** 2 - x ** 2 + 2j * zeta * f[r] * x)
    return np.abs(H)


def picked_error(d, s, zeta, nf=6001, span=8.0):
    """Tension error from the maximum of |H| at the stay, T going as f^2.

    Coarse scan, then golden-section refinement on the best grid cell.
    """
    f, V = pencil(d, s)
    mid, sep = 0.5 * (f[0] + f[1]), f[1] - f[0]
    half = max(sep, 4.0 * zeta) * span
    x = np.linspace(mid - half, mid + half, nf)
    i = int(np.argmax(_mag(x, f, V, zeta)))
    lo = x[max(i - 1, 0)]
    hi = x[min(i + 1, nf - 1)]
    g = 0.5 * (np.sqrt(5.0) - 1.0)
    a, b = lo, hi
    c, e = b - g * (b - a), a + g * (b - a)
    fc, fe = _mag(c, f, V, zeta), _mag(e, f, V, zeta)
    for _ in range(80):
        if fc > fe:
            b, e, fe = e, c, fc
            c = b - g * (b - a); fc = _mag(c, f, V, zeta)
        else:
            a, c, fc = c, e, fe
            e = a + g * (b - a); fe = _mag(e, f, V, zeta)
    return (0.5 * (a + b)) ** 2 - 1.0


def worst_over_detuning(s, zeta, dmax=0.03, nd=241):
    return max(abs(picked_error(d, s, zeta)) for d in np.linspace(-dmax, dmax, nd))


def main():
    rows = []
    print("Peak-picking bias against damping, at the median split "
          f"s = {100*S_MEDIAN:.2f} %\n")
    print(f"{'u = s/2z':>9} {'zeta %':>8} {'resolvable':>11} "
          f"{'worst |eps| %':>14} {'ratio to s':>11}")
    for u in (4.0, 2.0, 1.155, 1.2, 0.8, U_SPLIT, 0.35, 0.15, 0.10):
        z = S_MEDIAN / (2 * u)
        w = worst_over_detuning(S_MEDIAN, z)
        res = "yes" if u > U_SPLIT else "no"
        print(f"{u:>9.3f} {100*z:>8.3f} {res:>11} {100*w:>13.3f}% "
              f"{w/S_MEDIAN:>10.2f}x")
        rows.append(dict(s=S_MEDIAN, u=u, zeta=z, resolvable=u > U_SPLIT,
                         worst_eps=w, ratio=w / S_MEDIAN))

    us = np.linspace(0.15, 6.0, 60)
    ws = np.array([worst_over_detuning(S_MEDIAN, S_MEDIAN / (2 * u)) for u in us])
    i = int(np.argmax(ws))
    print(f"\n  maximum {100*ws[i]:.4f} % at u = {us[i]:.3f}, "
          f"zeta = {100*S_MEDIAN/(2*us[i]):.3f} %, "
          f"ratio {ws[i]/S_MEDIAN:.3f} (closed form sqrt(5)/2 = "
          f"{np.sqrt(5)/2:.3f} at u = 2/sqrt(3) = {2/np.sqrt(3):.3f})")

    print("\n  bare stay without dampers, zeta = 0.1 to 0.3 %")
    for z in (0.001, 0.002, 0.003):
        u = S_MEDIAN / (2 * z)
        w = worst_over_detuning(S_MEDIAN, z)
        print(f"    zeta = {100*z:.1f} %  ->  u = {u:.2f},  worst "
              f"{100*w:.3f} % = {w/S_MEDIAN:.2f} x s")

    print("\n  seasonal swing on the example bridge, at damping ratios typical "
          "of monitored bridges")
    print(f"    (split s = {100*S_WORKED:.2f} %; the nominal swing "
          f"is 2s = {200*S_WORKED:.2f} %, which assumes a resolved pair)")
    for z in (0.001, 0.002, 0.005, 0.01, 0.02, 0.03):
        u = S_WORKED / (2 * z)
        # swing: peak-to-peak of the signed picked error over the detuning sweep
        e = np.array([picked_error(d, S_WORKED, z)
                      for d in np.linspace(-0.04, 0.04, 321)])
        ptp = e.max() - e.min()
        print(f"    zeta = {100*z:>4.1f} %  u = {u:>5.2f}  swing "
              f"{100*ptp:>6.3f} %  ({ptp/(2*S_WORKED):.2f} of 2s)")
        rows.append(dict(s=S_WORKED, u=u, zeta=z, resolvable=u > U_SPLIT,
                         worst_eps=ptp / 2, ratio=ptp / (2 * S_WORKED)))

    os.makedirs(DATA, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "dangerband.csv"), index=False)
    print(f"\n  wrote {DATA}/dangerband.csv")


if __name__ == "__main__":
    main()
