# -*- coding: utf-8 -*-
"""Coupled sag and veering check on the two-dimensional model of src/cablefe2d.py.

A. Example bridge with the sag alone varied: gravity is scaled so that the chord
   lambda^2 at the crossing runs from 0 to 4, at stay orders 1 and 2, with
   uniform and varying tension along the chord.
B. Graded parametric-study designs with chord lambda^2 nearest 0.5 to 8, traversed at
   their own gravity with the sag drawn and with it suppressed.
Reads data/campaign.csv; writes data/sag.csv and data/sag_worked.csv.
Run:  python3 scripts/run_sag.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from cablefe import irvine_lambda2  # noqa: E402
from cablefe2d import CableDeck2D, G, split_one_ended  # noqa: E402

DATA = os.path.join(ROOT, "data")
BRIDGE = dict(Ld=80.0, EId=2.0e9, md=1000.0,
              Lc=25.0, EIc=1.2e4, mc=5.5, EA=1.4e8,
              theta=np.deg2rad(35.0), nd=40, nc=60)
LAM2_TARGETS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
REAL_TARGETS = (0.5, 1.0, 2.0, 4.0, 8.0)


def g_for_lambda2(target, Lc, T, EA, mc, theta):
    """Gravity that gives the chord lambda^2 = target at tension T."""
    if target <= 0:
        return 0.0
    g = np.sqrt(target / irvine_lambda2(Lc, T, EA, mc, theta, g=1.0))
    for _ in range(4):
        g *= np.sqrt(target / irvine_lambda2(Lc, T, EA, mc, theta, g=g))
    return g


def traverse(make, Ts, n, f_guess):
    """Minimum relative separation of the two lines nearest the stay order
    n along a tension traverse; returns (s, T_at, f_lo, f_hi, model)."""
    best = (np.inf, None, None, None, None)
    for T in Ts:
        cd = make(T)
        f, Phi = cd.modes(40)
        fi = cd.stay_alone(n + 1)[n - 1]
        lo, hi = 0.80 * fi, 1.25 * fi
        inb = np.where((f > lo) & (f < hi))[0]
        if len(inb) < 2:
            continue
        ff = f[inb]
        # the pair: the two lines closest to each other in the window
        k = int(np.argmin(np.diff(ff)))
        gap = (ff[k + 1] - ff[k]) / (0.5 * (ff[k] + ff[k + 1]))
        if gap < best[0]:
            best = (gap, T, ff[k], ff[k + 1], cd)
    return best


def host_ordinate(cd, f_target):
    """Frequency and anchorage ordinate of the host mode nearest f_target."""
    fh, pa, pp = cd.host_alone(16)
    j = int(np.argmin(np.abs(fh - f_target)))
    return fh[j], pa[j]


def case_row(label, cd, n, s_fe, T_at, f_lo, f_hi, lam2, g, tv):
    """Result row: finite element and predicted widths, isolated frequencies
    with and without sag, and the tension errors of the stay branch."""
    M_s = 0.5 * cd.mc * cd.Lc
    f_iso = cd.stay_alone(n + 1)[n - 1]                  # sagged, both ends held
    f_iso_straight = CableDeck2D(T=T_at, g=0.0, **{k: getattr(cd, k) for k in
                                 ("Ld", "EId", "md", "Lc", "EIc", "mc", "EA", "theta", "nd", "nc")},
                                 x_anchor=cd.x_anchor).stay_alone(n + 1)[n - 1]
    f0 = 0.5 * (f_lo + f_hi)
    fh, phi_a = host_ordinate(cd, f0)
    s_pred = split_one_ended(M_s, phi_a, cd.theta, n)
    # tension error of the stay branch (larger stay energy share), read
    # through the sag-corrected isolated frequency
    f, Phi = cd.modes(40)
    es = cd.energy_split(Phi)
    j_lo, j_hi = int(np.argmin(np.abs(f - f_lo))), int(np.argmin(np.abs(f - f_hi)))
    f_stay = f_hi if es[j_hi] > es[j_lo] else f_lo
    eps_corrected = (f_stay / f_iso) ** 2 - 1.0
    eps_string = (f_stay / f_iso_straight) ** 2 - 1.0     # no sag correction at all
    return dict(case=label, n=n, lam2=lam2, g=g, tension_variation=int(tv),
                T_at=T_at, f_lo=f_lo, f_hi=f_hi, f_host=fh, phi_a=phi_a,
                f_iso_sag=f_iso, f_iso_straight=f_iso_straight,
                sag_lift_pct=100 * (f_iso / f_iso_straight - 1),
                s_fe=s_fe, s_pred=s_pred, ratio=s_fe / s_pred,
                eps_corrected=eps_corrected, eps_string=eps_string)


def part_a(rows):
    print("A. example bridge, sag varied through g, all else fixed")
    T_cross = 151.6e3
    for n, Ts in ((1, np.linspace(120e3, 190e3, 141)), (2, np.linspace(92e3, 125e3, 133))):
        for lam2_t in LAM2_TARGETS:
            g = g_for_lambda2(lam2_t, BRIDGE["Lc"], T_cross, BRIDGE["EA"], BRIDGE["mc"], BRIDGE["theta"])
            for tv in ((False, True) if lam2_t > 0 else (False,)):
                make = lambda T, g=g, tv=tv: CableDeck2D(T=T, g=g, tension_variation=tv, **BRIDGE)
                s_fe, T_at, f_lo, f_hi, cd = traverse(make, Ts, n, None)
                lam2 = irvine_lambda2(BRIDGE["Lc"], T_at, BRIDGE["EA"], BRIDGE["mc"], BRIDGE["theta"], g=g) if g else 0.0
                r = case_row("worked", cd, n, s_fe, T_at, f_lo, f_hi, lam2, g, tv)
                rows.append(r)
                print(f"   n = {n}  lambda^2 = {lam2:5.2f}  {'varying T' if tv else 'uniform T'}:"
                      f" crossing at {T_at/1e3:6.1f} kN, sag lift of f_iso {r['sag_lift_pct']:+6.2f} %,"
                      f" s_FE {100*s_fe:.3f} % vs closed form {100*r['s_pred']:.3f} % (ratio {r['ratio']:.3f});"
                      f" eps corrected {100*r['eps_corrected']:+.2f} %, uncorrected {100*r['eps_string']:+.2f} %")
        # fundamental at 0.5, 1 and 2 times the bridge's own gravity (r_1 near one)
        if n == 1:
            for mult in (0.5, 1.0, 2.0):
                g = mult * G
                make = lambda T, g=g: CableDeck2D(T=T, g=g, tension_variation=True, **BRIDGE)
                s_fe, T_at, f_lo, f_hi, cd = traverse(make, Ts, n, None)
                lam2 = irvine_lambda2(BRIDGE["Lc"], T_at, BRIDGE["EA"], BRIDGE["mc"], BRIDGE["theta"], g=g)
                r = case_row("worked_gmult", cd, n, s_fe, T_at, f_lo, f_hi, lam2, g, True)
                rows.append(r)
                print(f"   n = 1  {mult:g} g  lambda^2 = {lam2:.4f}: crossing at {T_at/1e3:6.1f} kN,"
                      f" s_FE {100*s_fe:.3f} % vs closed form {100*r['s_pred']:.3f} % (ratio {r['ratio']:.3f})")


def part_b(rows):
    print("B. parametric-study designs nearest the target lambda^2, own gravity")
    c = pd.read_csv(os.path.join(DATA, "campaign.csv"))
    g = c[(c.mac > 0.5) & (c.xi > 150)].copy()
    g["lam2c"] = irvine_lambda2(g.Lc, g["T"], g.EA, g.mc, g.theta)
    g1 = g[g.n_stay == 1]
    for target in REAL_TARGETS:
        r = g1.iloc[int(np.argmin(np.abs(np.log(g1.lam2c / target))))]
        base = dict(Ld=r.Ld, EId=r.EId, md=r.md, Lc=r.Lc, EIc=r.EIc, mc=r.mc, EA=r.EA,
                    theta=r.theta, x_anchor=r.xfrac * r.Ld, nd=40, nc=60)
        Ts = np.linspace(0.7 * r["T"], 1.4 * r["T"], 141)
        out = {}
        for lab, gg in (("straight", 0.0), ("sagged", G)):
            make = lambda T, gg=gg: CableDeck2D(T=T, g=gg, tension_variation=True, **base)
            s_fe, T_at, f_lo, f_hi, cd = traverse(make, Ts, 1, None)
            if cd is None:
                print(f"   design lambda^2 {r.lam2c:.2f}: no pair found ({lab})")
                break
            lam2 = irvine_lambda2(r.Lc, T_at, r.EA, r.mc, r.theta) if gg else 0.0
            row = case_row(f"design_{target:g}_{lab}", cd, 1, s_fe, T_at, f_lo, f_hi, lam2, gg, True)
            row.update(Ld=r.Ld, Lc=r.Lc, theta_deg=np.degrees(r.theta), T_design=r["T"])
            rows.append(row)
            out[lab] = row
        if len(out) == 2:
            a, b = out["straight"], out["sagged"]
            print(f"   Ld {r.Ld:5.0f} m, Lc {r.Lc:5.1f} m, theta {np.degrees(r.theta):4.1f} deg, lambda^2 {b['lam2']:.2f}:"
                  f" s straight {100*a['s_fe']:.3f} % -> sagged {100*b['s_fe']:.3f} % (x{b['s_fe']/a['s_fe']:.3f});"
                  f" closed form {100*b['s_pred']:.3f} %; sag lift {b['sag_lift_pct']:+.2f} %;"
                  f" eps corrected {100*b['eps_corrected']:+.2f} %, uncorrected {100*b['eps_string']:+.2f} %")


def part_0():
    """Chord lambda^2 and normal sag ratio of the example bridge at its own
    gravity, at three tensions including the crossing. Writes data/sag_worked.csv."""
    print("0. example bridge at its own gravity")
    rows = []
    for T in (105e3, 151.6e3, 215e3):
        lam2 = irvine_lambda2(BRIDGE["Lc"], T, BRIDGE["EA"], BRIDGE["mc"], BRIDGE["theta"])
        sag = BRIDGE["mc"] * G * np.cos(BRIDGE["theta"]) * BRIDGE["Lc"] / (8.0 * T)   # normal sag / chord
        rows.append(dict(T=T, lam2_chord=lam2, sag_over_chord=sag))
        print(f"   T = {T/1e3:5.1f} kN: lambda^2 = {lam2:.4f}, normal sag / chord = {sag:.2e}")
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "sag_worked.csv"), index=False)


def main():
    rows = []
    part_0()
    part_a(rows)
    part_b(rows)
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "sag.csv"), index=False)
    print("wrote", os.path.join(DATA, "sag.csv"))


if __name__ == "__main__":
    main()
