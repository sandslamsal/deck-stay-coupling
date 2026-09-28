# -*- coding: utf-8 -*-
"""Two-sided branch mis-assignment across the stay-1 / deck crossing.

For each tension, takes both branches of the avoided crossing near stay mode 1
(the two coupled modes with the largest stay-energy fraction within WINDOW of
f_iso), applies the order-1 taut-string inversion to each, and reports both
errors and the tension window in which both branches hold more than AMBIG of
the stay energy. Writes data/twosided.csv.
Run: python3 scripts/run_twosided.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from cablefe import (CableDeck, chain, invert_string, mu_effective,  # noqa: E402
                     tensioned_beam_freq, veering_split)
from run_identify2 import BRIDGE, TENSIONS  # noqa: E402

DATA = os.path.join(ROOT, "data")

WINDOW = 0.30            # search half-width around f_iso, relative
AMBIG = 0.20             # stay-energy fraction defining ambiguity
EXTRA = np.linspace(145e3, 160e3, 20)   # tensions spanning the crossing


def branch_pair(T, bridge=BRIDGE, window=WINDOW):
    """Both branches of the avoided crossing near stay mode 1.

    Returns ``(f_lower, f_upper, frac_lower, frac_upper, f_iso)``: the two
    coupled modes with the largest stay-energy fraction within
    ``window * f_iso`` of the isolated stay frequency, ordered by frequency.
    """
    cd = CableDeck(T=T, **bridge)
    f, Phi = cd.modes(60)
    frac = cd.energy_split(Phi)
    f_iso = tensioned_beam_freq(1, bridge["Lc"], T, bridge["EIc"],
                                bridge["mc"])
    w = window
    sel = np.where(np.abs(f - f_iso) <= w * f_iso)[0]
    while len(sel) < 2:                     # never triggers on this bridge
        w *= 1.5
        sel = np.where(np.abs(f - f_iso) <= w * f_iso)[0]
    top2 = sel[np.argsort(frac[sel])[-2:]]
    lo, up = sorted(top2, key=lambda j: f[j])
    return float(f[lo]), float(f[up]), float(frac[lo]), float(frac[up]), \
        float(f_iso)


def err_pct(f1, T_true, bridge=BRIDGE):
    """Taut-string order-1 inversion error, percent of the true tension."""
    T_est = float(invert_string(np.array([f1]), bridge["Lc"], bridge["mc"],
                                np.array([1]))[0])
    return 100.0 * (T_est - T_true) / T_true


def deck_mode_near(f_target, bridge=BRIDGE):
    """Deck-alone frequency nearest ``f_target`` and its mass-normalized
    amplitude at the anchorage.  ``k_ax`` does not depend on T."""
    cd = CableDeck(T=150e3, **bridge)
    Kd, Md = chain(bridge["Ld"], bridge["nd"], bridge["EId"],
                   bridge["md"], 0.0)
    Kd = Kd.copy()
    Kd[2 * cd.ia, 2 * cd.ia] += cd.k_ax
    keep = [i for i in range(Kd.shape[0]) if i not in (0, 2 * cd.nd)]
    w2, V = eigh(Kd[np.ix_(keep, keep)], Md[np.ix_(keep, keep)])
    fd = np.sqrt(np.maximum(w2, 0.0)) / (2.0 * np.pi)
    j = int(np.argmin(np.abs(fd - f_target)))
    phi_a = abs(V[keep.index(2 * cd.ia), j])    # eigh mass-normalizes
    return float(fd[j]), float(phi_a)


def ambiguity(T):
    """Stay-energy fraction of the weaker branch: the ambiguity margin."""
    _, _, cl, cu, _ = branch_pair(T)
    return min(cl, cu)


def bisect_edge(T_in, T_out, tol=1.0):
    """Tension where the ambiguity margin crosses ``AMBIG``, between a
    point inside the zone and a point outside it."""
    if not (ambiguity(T_in) > AMBIG >= ambiguity(T_out)):
        raise ValueError("bracket does not straddle the threshold")
    while abs(T_out - T_in) > tol:
        Tm = 0.5 * (T_in + T_out)
        if ambiguity(Tm) > AMBIG:
            T_in = Tm
        else:
            T_out = Tm
    return 0.5 * (T_in + T_out)


def main():
    tensions = np.unique(np.concatenate([TENSIONS, EXTRA]))
    rows = []
    for T in tensions:
        T = float(T)
        fl, fu, cl, cu, _ = branch_pair(T)
        rows.append(dict(T_true=T, f_upper=fu, f_lower=fl,
                         stayfrac_upper=cu, stayfrac_lower=cl,
                         err_upper_pct=err_pct(fu, T),
                         err_lower_pct=err_pct(fl, T)))
    d = pd.DataFrame(rows)[["T_true", "f_upper", "f_lower",
                            "stayfrac_upper", "stayfrac_lower",
                            "err_upper_pct", "err_lower_pct"]]
    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, "twosided.csv")
    d.to_csv(out, index=False)
    print(f"wrote {out}  ({len(d)} tensions)")

    # ---- stay-dominated branch vs the other one --------------------------
    dom_up = d.stayfrac_upper.values >= d.stayfrac_lower.values
    err_dom = np.where(dom_up, d.err_upper_pct.values, d.err_lower_pct.values)
    err_oth = np.where(dom_up, d.err_lower_pct.values, d.err_upper_pct.values)
    margin = np.minimum(d.stayfrac_upper.values, d.stayfrac_lower.values)

    i_dom = int(np.argmax(np.abs(err_dom)))
    i_oth = int(np.argmax(np.abs(err_oth)))
    print("\nworst on the stay-dominated branch : "
          f"{err_dom[i_dom]:+7.3f} %  at T = {d.T_true[i_dom]/1e3:7.2f} kN")
    print("worst when the OTHER branch is used: "
          f"{err_oth[i_oth]:+7.3f} %  at T = {d.T_true[i_oth]/1e3:7.2f} kN")

    amb = margin > AMBIG
    if amb.any():
        ia = int(np.argmax(np.where(amb, np.abs(err_oth), -np.inf)))
        print(f"worst OTHER-branch error inside the >{AMBIG:.0%} "
              f"ambiguous zone     : {err_oth[ia]:+7.3f} %  "
              f"at T = {d.T_true[ia]/1e3:7.2f} kN")

    # ---- ambiguous window, refined off-grid -----------------------------
    grid = np.linspace(140e3, 168e3, 141)
    g_marg = np.array([ambiguity(float(t)) for t in grid])
    inside = np.where(g_marg > AMBIG)[0]
    if len(inside):
        lo_in, hi_in = grid[inside[0]], grid[inside[-1]]
        T_lo = bisect_edge(lo_in, grid[inside[0] - 1]) \
            if inside[0] > 0 else grid[0]
        T_hi = bisect_edge(hi_in, grid[inside[-1] + 1]) \
            if inside[-1] < len(grid) - 1 else grid[-1]

        fd, phi_a = deck_mode_near(3.32)
        b = BRIDGE

        def detune(T):
            fi = tensioned_beam_freq(1, b["Lc"], T, b["EIc"], b["mc"])
            return (fi - fd) / fd

        k = np.pi / b["Lc"]
        om = 2.0 * np.pi * fd
        T_x = (b["mc"] * om ** 2 - b["EIc"] * k ** 4) / k ** 2

        print(f"\nambiguous zone (both branches > {AMBIG:.0%} stay energy):")
        print(f"    tension   {T_lo/1e3:8.2f} .. {T_hi/1e3:8.2f} kN"
              f"   (width {(T_hi-T_lo)/1e3:.2f} kN, "
              f"{100*(T_hi-T_lo)/T_x:.2f} % of T_x = {T_x/1e3:.2f} kN)")
        print(f"    detuning  {detune(T_lo):+8.4f} .. {detune(T_hi):+8.4f}"
              f"   (width {detune(T_hi)-detune(T_lo):.4f})")

        # ---- closed-form cross-check ------------------------------------
        M_s = b["mc"] * b["Lc"] / 2.0
        mu = mu_effective(M_s, phi_a)
        s_pred = veering_split(mu, b["theta"], 1)
        fls = np.array([branch_pair(float(t))[:2] for t in grid])
        gaps = (fls[:, 1] - fls[:, 0]) / (0.5 * (fls[:, 1] + fls[:, 0]))
        print(f"\ncross-check: min gap (f+-f-)/f over grid = "
              f"{gaps.min():.5f}  vs  closed-form s = {s_pred:.5f}"
              f"   (mu_eff = {mu:.3e}, f_deck = {fd:.4f} Hz, "
              f"phi_a = {phi_a:.4e})")
    else:
        print("\nno tension on the refinement grid keeps both branches "
              f"above {AMBIG:.0%} stay energy")

    with pd.option_context("display.width", 120,
                           "display.float_format", lambda v: f"{v:10.5f}"):
        print("\n", d.assign(T_kN=d.T_true / 1e3)
              [["T_kN", "f_lower", "f_upper", "stayfrac_lower",
                "stayfrac_upper", "err_lower_pct", "err_upper_pct"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
