# -*- coding: utf-8 -*-
"""Designs re-solved with the stay on its sagged profile; two-route law test.

Solves every design of data/campaign.csv in the sagged model (src/cablefe2d.py)
at its own tension and gravity, removes the bias of the isolated sagged stay,
and compares the straight-chord law with the extended law of width s |1 - r_n|,
where r_n = (2 / (n^2 pi^2)) sin(theta) (EA / T) (m g L / T) for odd stay
orders n and r_n = 0 for even n. Writes data/campaign_sag.csv.
Run:  python3 scripts/run_campaign_sag.py      (about three minutes)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from cablefe2d import CableDeck2D, G  # noqa: E402

DATA = os.path.join(ROOT, "data")


def r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return 1.0 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def isolated_shapes(cd, nmodes):
    """Frequencies and transverse shapes (normal to the chord) of the sagged
    stay alone, both ends held, from the same elements."""
    from scipy.linalg import eigh
    from cablefe2d import polyline_chain
    xy, N, P, A = cd.stay_nodes()
    Kc, Mc = polyline_chain(xy, cd.EA, cd.EIc, cd.mc, N)
    ntot = Kc.shape[0]
    fixed = {0, 1, ntot - 3, ntot - 2}
    keep = [i for i in range(ntot) if i not in fixed]
    w2, V = eigh(Kc[np.ix_(keep, keep)], Mc[np.ix_(keep, keep)])
    f = np.sqrt(np.maximum(w2, 0.0)) / (2 * np.pi)
    full = np.zeros((ntot, nmodes))
    full[keep, :] = V[:, :nmodes]
    c, s = np.cos(cd.theta), np.sin(cd.theta)
    u = full[0::3, :]
    w = full[1::3, :]
    return f[:nmodes], s * u + c * w


def mac(a, b):
    return float((a @ b) ** 2 / ((a @ a) * (b @ b) + 1e-300))


def solve(row):
    n = int(row.n_stay)
    cd = CableDeck2D(Ld=row.Ld, EId=row.EId, md=row.md, Lc=row.Lc, EIc=row.EIc, mc=row.mc,
                     T=row["T"], EA=row.EA, theta=row.theta, x_anchor=row.xfrac * row.Ld,
                     nd=40, nc=40, g=G, tension_variation=True)
    f, Phi = cd.modes(80)
    f_iso_all, shapes = isolated_shapes(cd, n + 2)
    f_iso = f_iso_all[n - 1]
    ref = shapes[:, n - 1]
    V = cd.stay_transverse(Phi)                     # (nc+1) x modes
    win = np.where((f > 0.7 * f_iso) & (f < 1.35 * f_iso))[0]
    if len(win) == 0:
        return None
    macs = np.array([mac(V[:, j], ref) for j in win])
    j = win[int(np.argmax(macs))]
    f_branch = f[j]
    T_est = 4 * row.mc * row.Lc ** 2 * f_branch ** 2 / n ** 2
    T_ctrl = 4 * row.mc * row.Lc ** 2 * f_iso ** 2 / n ** 2
    eps = T_est / row["T"] - 1
    eps_ctrl = T_ctrl / row["T"] - 1
    return dict(f_iso_sag=f_iso, f_branch_sag=f_branch, share=float(macs.max()),
                eps_sag=eps, eps_control_sag=eps_ctrl, eps_coupling_sag=eps - eps_ctrl)


def main():
    c = pd.read_csv(os.path.join(DATA, "campaign.csv"))
    t0 = time.time()
    out = []
    for i, row in c.iterrows():
        r = solve(row)
        if r is None:
            r = dict(f_iso_sag=np.nan, f_branch_sag=np.nan, share=np.nan, eps_sag=np.nan,
                     eps_control_sag=np.nan, eps_coupling_sag=np.nan)
        out.append(r)
        if i % 100 == 0:
            print(f"  {i}/{len(c)}  {time.time()-t0:.0f} s")
    o = pd.DataFrame(out)
    c = pd.concat([c.reset_index(drop=True), o], axis=1)
    q = c.mc * G * c.Lc / c["T"]
    c["r_n"] = np.where(c.n_stay % 2 == 1,
                        2 / np.pi ** 2 * np.sin(c.theta) * (c.EA / c["T"]) * q / c.n_stay ** 2, 0.0)
    c["s_ext"] = c.s * np.abs(1 - c.r_n)
    c["d_sag"] = (c.f_iso_sag - c.f_deck) / c.f_iso_sag
    c["eps_pred_ext"] = np.sqrt(c.d_sag ** 2 + c.s_ext ** 2) - np.abs(c.d_sag)
    c["eps_pred_straight_dsag"] = np.sqrt(c.d_sag ** 2 + c.s ** 2) - np.abs(c.d_sag)
    c.to_csv(os.path.join(DATA, "campaign_sag.csv"), index=False)

    g = c[(c.mac > 0.5) & (c.xi > 150) & (c.share > 0.5) & c.eps_coupling_sag.notna()]
    y = g.eps_coupling_sag.abs()
    print(f"\ngraded and resolved in the sagged model: {len(g)} of 421")
    print(f"  straight-chord law, on the sagged errors:               R2 = {r2(y, g.eps_pred_straight_dsag):.4f}")
    print(f"  extended law, s |1 - r_n|:                              R2 = {r2(y, g.eps_pred_ext):.4f}")
    for lab, sub in (("even orders (no elastic route)", g[g.n_stay % 2 == 0]),
                     ("odd orders", g[g.n_stay % 2 == 1]),
                     ("odd, r_n < 0.3", g[(g.n_stay % 2 == 1) & (g.r_n < 0.3)]),
                     ("odd, 0.3 <= r_n < 3", g[(g.n_stay % 2 == 1) & (g.r_n >= 0.3) & (g.r_n < 3)]),
                     ("odd, r_n >= 3", g[(g.n_stay % 2 == 1) & (g.r_n >= 3)])):
        yy = sub.eps_coupling_sag.abs()
        if len(sub) > 2:
            print(f"  {lab:32s} n = {len(sub):3d}: straight R2 = {r2(yy, sub.eps_pred_straight_dsag):7.4f},"
                  f" extended R2 = {r2(yy, sub.eps_pred_ext):7.4f}, worst |eps| = {100*yy.max():.2f} %,"
                  f" median |res_ext| = {100*np.median(np.abs(yy - sub.eps_pred_ext)):.3f} pp")
    ratio = (g.eps_coupling_sag.abs() / g.eps_coupling.abs().clip(lower=1e-6))
    print(f"  sagged/straight coupling error, graded: median {ratio.median():.2f}, 10th {ratio.quantile(.1):.2f}, 90th {ratio.quantile(.9):.2f}")
    print(f"  worst sagged coupling error: {100*y.max():.2f} % (straight-chord model worst 5.09 %)")
    # screening at 2 % tolerance with each law, against the sagged errors
    for lab, s_use in (("straight s", g.s), ("s |1 - r_n|", g.s_ext)):
        passes = np.abs(g.d_sag) >= (s_use ** 2 - 0.02 ** 2) / (2 * 0.02)
        unsafe = y > 0.02
        print(f"  screen at 2 % with {lab:12s}: unsafe {int(unsafe.sum())}, missed {int((passes & unsafe).sum())},"
              f" flagged safe {int((~passes & ~unsafe).sum())}")
    print("wrote", os.path.join(DATA, "campaign_sag.csv"))


if __name__ == "__main__":
    main()
