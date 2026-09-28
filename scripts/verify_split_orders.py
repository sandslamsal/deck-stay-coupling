"""Closed-form veering width against the exact coupled roots, orders 1 to 6.

A taut stay cable pinned at the pylon and tied through c = cos(theta) to a
deck mode of modal mass M and frequency w_d satisfies (Appendix A)

    sin(kL) (M w^2 - M w_d^2) - c^2 T k cos(kL) = 0,   k = w sqrt(m/T).

At exact tuning (w_d equal to the isolated stay frequency w_n) the two roots
next to w_n give the split, compared here with s = (2c/n pi) sqrt(mu_eff),
mu_eff = M_s/M, M_s = m L / 2, for n = 1 to 6, two tie factors and mass
ratios spanning the design set (median 4e-4, maximum 0.062).

Writes data/verify_split_orders.csv.
Run:  python3 scripts/verify_split_orders.py
"""
import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

L, T, m = 100.0, 2.0e6, 50.0          # stay length (m), tension (N), mass (kg/m)
MS = 0.5 * m * L                       # stay modal mass (kg)
MU = [1e-5, 1e-4, 4e-4, 1e-3, 3e-3, 1e-2, 3e-2, 6.2e-2]
CS = [1.0, 0.5]


def exact_split(n, mu, c):
    """Relative split of the two roots next to the tuned stay frequency."""
    wn = n * np.pi / L * np.sqrt(T / m)
    M = MS / mu

    def f(w):
        k = w * np.sqrt(m / T)
        return np.sin(k * L) * (M * w**2 - M * wn**2) - c**2 * T * k * np.cos(k * L)

    s_guess = 2 * c / (n * np.pi) * np.sqrt(mu)
    lo = brentq(f, wn * (1 - 3 * s_guess), wn * (1 - 1e-6 * s_guess))
    hi = brentq(f, wn * (1 + 1e-6 * s_guess), wn * (1 + 3 * s_guess))
    return (hi - lo) / wn


def main():
    rows = []
    for n in range(1, 7):
        for c in CS:
            for mu in MU:
                s_ex = exact_split(n, mu, c)
                s_cf = 2 * c / (n * np.pi) * np.sqrt(mu)
                rows.append(dict(n=n, c=c, mu_eff=mu, s_exact=s_ex, s_closed=s_cf,
                                 dev_pct=100 * (s_cf / s_ex - 1)))
    d = pd.DataFrame(rows)
    os.makedirs(DATA, exist_ok=True)
    d.to_csv(os.path.join(DATA, "verify_split_orders.csv"), index=False)
    print("closed-form width against the exact roots, n = 1 to 6, c = 1 and 0.5")
    for lim in (1e-3, 1e-2, 3e-2, 6.2e-2):
        g = d[d.mu_eff <= lim + 1e-12]
        print("  mu_eff <= %-7g  worst |deviation| %.3f %%" % (lim, g.dev_pct.abs().max()))
    print("  design-set median mu_eff 4e-4: worst %.4f %%"
          % d[d.mu_eff == 4e-4].dev_pct.abs().max())


if __name__ == "__main__":
    main()
