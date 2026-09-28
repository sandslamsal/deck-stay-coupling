# -*- coding: utf-8 -*-
"""Tension error below the xi = 150 gate and across Irvine's lambda^2.

Reads data/campaign.csv. Inverts the coupled branch frequency with the exact
tensioned-beam form T = 4 m L^2 f^2 / n^2 - n^2 pi^2 EI / L^2, compares the
residual with eps = sqrt(d^2 + s^2) - |d|, checks that bending and coupling
errors superpose, and bins the graded designs by lambda^2. Writes
data/xi_combined.csv and data/xi_combined_summary.csv.

Run:  python3 scripts/run_xi_combined.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
DATA = os.path.join(ROOT, "data")

from cablefe import irvine_lambda2  # noqa: E402


def r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return 1.0 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    c = pd.read_csv(os.path.join(DATA, "campaign.csv"))
    n, L, m, EI, T = c.n_stay, c.Lc, c.mc, c.EIc, c["T"]
    # Irvine's parameter in the chord form
    c["lam2"] = irvine_lambda2(c.Lc, c["T"], c.EA, c.mc, c.theta)

    # exact pinned-pinned tensioned-beam inversion of the coupled branch
    # frequency, and of the isolated control frequency as a check
    c["eps_tb"] = (4 * m * L**2 * c.f_coupled**2 / n**2
                   - n**2 * np.pi**2 * EI / L**2) / T - 1
    c["eps_tb_control"] = (4 * m * L**2 * c.f_s**2 / n**2
                           - n**2 * np.pi**2 * EI / L**2) / T - 1
    assert c.eps_tb_control.abs().max() < 1e-9, \
        "the exact inversion should zero the isolated control"
    # superposition: total string error against bending bias + signed law
    c["eps_superposed"] = c.eps_control + np.sign(c.eps_coupling) * c.eps_pred
    c["res_superposition"] = c.eps - c.eps_superposed
    c["res_tb"] = c.eps_tb.abs() - c.eps_pred
    c["sign_agrees"] = np.sign(c.eps_coupling) == np.sign(c.d)
    c["graded"] = (c.mac > 0.5) & (c.xi > 150)
    c.to_csv(os.path.join(DATA, "xi_combined.csv"), index=False)

    rows = []

    def add(label, sub, y):
        res = np.abs(np.abs(sub[y]) - sub.eps_pred)
        mat = sub[sub.eps_pred > 0.0005]          # where the error is material
        rows.append(dict(
            subset=label, n=len(sub), r2=r2(np.abs(sub[y]), sub.eps_pred),
            median_res_pp=100 * np.median(res), p90_res_pp=100 * np.percentile(res, 90),
            worst_res_pp=100 * res.max(),
            sign_agree_pct=100 * (np.sign(sub[y]) == np.sign(sub.d))[sub.eps_pred > 0.0005].mean()
            if len(mat) else np.nan,
            n_material=len(mat)))

    g = c[c.graded]
    sub = c[c.xi <= 150]
    add("graded 421, string reading, coupling part", g, "eps_coupling")
    add("graded 421, tensioned-beam reading", g, "eps_tb")
    add("all 699, string reading, total", c, "eps")
    add("all 699, tensioned-beam reading", c, "eps_tb")
    add("sub-gate 278, string reading, total", sub, "eps")
    add("sub-gate 278, tensioned-beam reading", sub, "eps_tb")
    for lo, hi in [(0, 50), (50, 100), (100, 150)]:
        add(f"sub-gate xi in ({lo},{hi}], tensioned-beam reading",
            sub[(sub.xi > lo) & (sub.xi <= hi)], "eps_tb")
    for lim in [0.1, 0.5, 1.0, 4.0]:
        add(f"graded lam2 <= {lim}, coupling part", g[g.lam2 <= lim], "eps_coupling")
    add("graded lam2 > 1, coupling part", g[g.lam2 > 1], "eps_coupling")

    summ = pd.DataFrame(rows)
    # superposition on the sub-gate designs
    sp = sub.res_superposition.abs()
    extra = pd.DataFrame([dict(
        subset="sub-gate 278, superposition eps_bend + eps_coupling", n=len(sub),
        r2=np.nan, median_res_pp=100 * sp.median(), p90_res_pp=100 * sp.quantile(0.9),
        worst_res_pp=100 * sp.max(), sign_agree_pct=np.nan, n_material=np.nan)])
    summ = pd.concat([summ, extra], ignore_index=True)
    summ.to_csv(os.path.join(DATA, "xi_combined_summary.csv"), index=False)

    pd.set_option("display.width", 160)
    print(summ.round(4).to_string(index=False))
    print()
    q = g.lam2.quantile([0, .25, .5, .75, .9, 1]).values
    print("graded lambda^2 quantiles (0,25,50,75,90,100 %):", np.round(q, 3))
    print("graded designs with lambda^2 > 0.1: %d, > 1: %d, > 4: %d"
          % ((g.lam2 > 0.1).sum(), (g.lam2 > 1).sum(), (g.lam2 > 4).sum()))
    print("sub-gate bending bias of the string formula: median %.2f %%, worst %.1f %%"
          % (100 * sub.eps_control.abs().median(), 100 * sub.eps_control.abs().max()))
    print("sub-gate xi quantiles:", np.round(sub.xi.quantile([0, .5, 1]).values, 1))


if __name__ == "__main__":
    main()
