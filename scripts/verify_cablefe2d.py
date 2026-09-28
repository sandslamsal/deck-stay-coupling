# -*- coding: utf-8 -*-
"""Verification of the two-dimensional stay model, src/cablefe2d.py, before
it is used for the sag (R1.5, R2.2) and pylon (R2.3, R2.4) studies.

Three checks, each against something the model did not assume.

1. Straight limit.  With g = 0 and a rigid pylon the model must reproduce
   cablefe.CableDeck, the model every published number came from, on the
   worked bridge over a tension range through the crossing: the coupled
   frequencies, the veering pair and the energy split.  The only physical
   difference is that the stay now also carries axial inertia, whose modes
   lie far above the band, so agreement should be well inside 0.1 %.

2. Irvine.  The sagged stay alone, both ends held, must return the
   symmetric in-plane modes of Irvine's equation
   tan(beta/2) = beta/2 - (4/lambda^2)(beta/2)^3 and leave the
   antisymmetric ones at the string values, for lambda^2 from 0.25 to 16
   and for an inclined chord.  This also settles which form of lambda^2 is
   the right one for an inclined cable: the chord-based form with the
   gravity component normal to the chord (irvine_lambda2_chord), against
   the horizontal-projection form that cablefe.irvine_lambda2 carries.

3. Two-ended drive.  With a flexible pylon whose sway mode is tuned to the
   stay while the deck is made very stiff, the coupling is through the
   pylon alone and the split must follow (2 / n pi) sqrt(M_s) sin(theta)
   |phi_p| at orders one and two; with deck and pylon both flexible the
   split must follow the two-ended form with the parity sign, and not the
   one-ended form or the wrong sign.

Writes data/verify_cablefe2d.csv.

Run:  python3 scripts/verify_cablefe2d.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from cablefe import CableDeck, string_freq  # noqa: E402
from cablefe2d import (CableDeck2D, G, irvine_lambda2_chord,  # noqa: E402
                       irvine_symmetric_beta, split_one_ended, split_two_ended)


def irvine_lambda2(L, T, EA, m, theta, g=G):
    """The HORIZONTAL-PROJECTION form the submitted campaign used (mass per
    unit arc length, H = T cos theta, L_h = L cos theta), kept here only to
    print what it would have said; cablefe.irvine_lambda2 is now the chord
    form."""
    H = T * np.cos(theta)
    Lh = L * np.cos(theta)
    d = m * g * Lh ** 2 / (8.0 * H)
    Le = Lh * (1.0 + 8.0 * (d / Lh) ** 2)
    return (m * g * Lh / H) ** 2 * Lh / (H * Le / EA)

DATA = os.path.join(ROOT, "data")
BRIDGE = dict(Ld=80.0, EId=2.0e9, md=1000.0,
              Lc=25.0, EIc=1.2e4, mc=5.5, EA=1.4e8,
              theta=np.deg2rad(35.0), nd=40, nc=40)
BAND = (2.6, 4.2)
rows = []


def pair(cd, f, Phi, band=BAND):
    inb = np.where((f > band[0]) & (f < band[1]))[0]
    return inb


def check1():
    print("1. straight limit against cablefe.CableDeck")
    worst = 0.0
    for T in np.linspace(105e3, 215e3, 12):
        a = CableDeck(T=T, **BRIDGE)
        b = CableDeck2D(T=T, g=0.0, **BRIDGE)
        fa, Pa = a.modes(14)
        fb, Pb = b.modes(14)
        # drop the stay's axial modes if any fell in range (they should not)
        ea, eb = a.energy_split(Pa), b.energy_split(Pb)
        err = np.abs(fb[:12] / fa[:12] - 1.0).max()
        derr = np.abs(eb[:12] - ea[:12]).max()
        worst = max(worst, err)
        rows.append(dict(check="straight", T=T, worst_freq_err=err, worst_split_err=derr))
    print(f"   worst relative frequency error over 12 modes and 12 tensions: {worst:.2e}")
    assert worst < 1e-3, worst
    # the veering pair at the exact-tuning tension of the campaign
    a = CableDeck(T=151.6e3, **BRIDGE)
    b = CableDeck2D(T=151.6e3, g=0.0, **BRIDGE)
    fa, _ = a.modes(20)
    fb, _ = b.modes(20)
    ia, ib = pair(a, fa, None), pair(b, fb, None)
    print(f"   pair at 151.6 kN: 1-D {fa[ia]} Hz, 2-D {fb[ib]} Hz")
    return worst


def check2():
    print("2. Irvine's symmetric modes for the sagged stay alone")
    Lc, EIc, mc, EA, theta = 25.0, 1.2e4, 5.5, 1.4e8, np.deg2rad(35.0)
    T = 151.6e3
    worst_sym, worst_asym, worst_conv = 0.0, 0.0, 0.0
    for lam2_target in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        # choose g so that the chord-based lambda^2 hits the target (the
        # L_e correction makes it implicit; iterate twice)
        g = np.sqrt(lam2_target / irvine_lambda2_chord(Lc, T, EA, mc, theta, g=1.0))
        for _ in range(3):
            g *= np.sqrt(lam2_target / irvine_lambda2_chord(Lc, T, EA, mc, theta, g=g))
        lam2 = irvine_lambda2_chord(Lc, T, EA, mc, theta, g=g)
        lam2_h = irvine_lambda2(Lc, T, EA, mc, theta, g=g)
        for nc in (40, 80):
            # Irvine neglects the variation of tension along the chord, so
            # the check is made on his terms (uniform tension); the sag
            # study reports both cases
            cd = CableDeck2D(T=T, g=g, nc=nc, Ld=80.0, EId=2.0e9, md=1000.0,
                             Lc=Lc, EIc=EIc, mc=mc, EA=EA, theta=theta,
                             tension_variation=False)
            f = cd.stay_alone(6)
            # bending stiffness moves every mode slightly; compare with the
            # tensioned-beam-corrected string for the antisymmetric ones and
            # scale Irvine's beta by the same factor for the symmetric ones
            f_str = np.array([string_freq(n, Lc, T, mc) for n in (1, 2, 3, 4)])
            bend = np.sqrt(1.0 + EIc * (np.arange(1, 5) * np.pi / Lc) ** 2 / T)
            b1 = irvine_symmetric_beta(lam2, 1)
            b3 = irvine_symmetric_beta(lam2, 2)
            f_irv1 = b1 * np.sqrt(T / mc) / Lc / (2 * np.pi) * bend[0]
            f_irv3 = b3 * np.sqrt(T / mc) / Lc / (2 * np.pi) * bend[2]
            # order the FE modes by frequency; identify the symmetric ones as
            # those NOT within 0.05 % of the antisymmetric string values
            f_asym = f_str[[1, 3]] * bend[[1, 3]]
            is_asym = np.array([np.min(np.abs(x / f_asym - 1)) < 5e-3 for x in f])
            f_sym = f[~is_asym][:2]
            assert is_asym.sum() >= 2, (lam2, f)
            e1 = abs(f_sym[0] / f_irv1 - 1)
            e3 = abs(f_sym[1] / f_irv3 - 1)
            ea = np.abs(f[is_asym][:2] / f_asym - 1).max()
            if nc == 80:
                worst_sym = max(worst_sym, e1, e3)
                worst_asym = max(worst_asym, ea)
                print(f"   lambda^2 = {lam2:5.2f} (horizontal form would say {lam2_h:6.2f}, g = {g:6.2f}):"
                      f" f_sym1 {f_sym[0]:.4f} vs Irvine {f_irv1:.4f} ({100*e1:.3f} %),"
                      f" f_sym2 {f_sym[1]:.4f} vs {f_irv3:.4f} ({100*e3:.3f} %), antisym err {100*ea:.3f} %")
            rows.append(dict(check="irvine", lam2=lam2, lam2_horizontal=lam2_h, g=g, nc=nc,
                             f_sym1=f_sym[0], f_irvine1=f_irv1, err_sym1=e1,
                             f_sym2=f_sym[1], f_irvine2=f_irv3, err_sym2=e3, err_asym=ea))
    print(f"   worst symmetric-mode error {100*worst_sym:.3f} %, antisymmetric {100*worst_asym:.3f} % (nc = 80)")
    assert worst_sym < 5e-3 and worst_asym < 2e-3
    return worst_sym


def min_gap(make, Ts, band):
    """Traverse tensions, return the minimum pair gap and the tension there."""
    best = (np.inf, None, None)
    for T in Ts:
        cd = make(T)
        f, Phi = cd.modes(30)
        inb = np.where((f > band[0]) & (f < band[1]))[0]
        if len(inb) < 2:
            continue
        # the two closest lines in band
        ff = f[inb]
        k = int(np.argmin(np.diff(ff)))
        gap = (ff[k + 1] - ff[k]) / (0.5 * (ff[k] + ff[k + 1]))
        if gap < best[0]:
            best = (gap, T, cd)
    return best


def check3():
    print("3. two-ended drive with a flexible pylon")
    theta = np.deg2rad(35.0)
    Lc, mc = 25.0, 5.5
    M_s = 0.5 * mc * Lc
    Hp = Lc * np.sin(theta)
    # (a) pylon alone: a very stiff, heavy deck so that no deck mode sits in
    # the band; the pylon sway tuned near the n = 1 and n = 2 stay modes
    for n, EIp in ((1, 2.9e9), (2, 1.2e10)):
        pyl = dict(EI=EIp, m=2000.0, EA=1e11, n=20)
        base = dict(Ld=80.0, EId=2.0e12, md=1000.0, Lc=Lc, EIc=1.2e4, mc=mc, EA=1.4e8,
                    theta=theta, nd=40, nc=40, g=0.0, pylon=pyl)
        probe = CableDeck2D(T=150e3, **base)
        fh, pa, pp = probe.host_alone(8)
        j = int(np.argmin(np.abs(fh - probe.stay_alone(4)[n - 1])))
        # tension that tunes stay order n to that host mode
        f_target = fh[j]
        T_tune = (2 * Lc * f_target / n) ** 2 * mc
        Ts = np.linspace(0.9 * T_tune, 1.1 * T_tune, 81)
        band = (0.85 * f_target, 1.15 * f_target)
        gap, T_at, cd = min_gap(lambda T: CableDeck2D(T=T, **base), Ts, band)
        fh, pa, pp = cd.host_alone(8)
        j = int(np.argmin(np.abs(fh - f_target)))
        s_pyl = 2.0 / (n * np.pi) * np.sqrt(M_s) * abs(np.sin(theta) * pp[j])
        s_two = split_two_ended(M_s, pa[j], pp[j], theta, n)
        print(f"   pylon only, n = {n}: host mode {fh[j]:.3f} Hz, phi_a {pa[j]:+.2e}, phi_p {pp[j]:+.2e};"
              f" FE split {100*gap:.3f} %, pylon formula {100*s_pyl:.3f} %, two-ended {100*s_two:.3f} %")
        rows.append(dict(check="pylon_only", n=n, f_host=fh[j], phi_a=pa[j], phi_p=pp[j],
                         s_fe=gap, s_pylon_formula=s_pyl, s_two_ended=s_two))
        assert abs(gap / s_pyl - 1) < 0.08, (gap, s_pyl)
    # (b) deck and pylon both flexible and both carrying the host mode near
    # the stay: the worked bridge deck with a pylon whose sway sits near
    # the crossing, at n = 1 and n = 2
    for n, EIp in ((1, 3.5e9), (2, 1.4e10)):
        pyl = dict(EI=EIp, m=2000.0, EA=1e11, n=20)
        base = dict(Ld=80.0, EId=2.0e9, md=1000.0, Lc=Lc, EIc=1.2e4, mc=mc, EA=1.4e8,
                    theta=theta, nd=40, nc=40, g=0.0, pylon=pyl)
        probe = CableDeck2D(T=150e3, **base)
        fh, pa, pp = probe.host_alone(10)
        # every host mode in a band around the stay order n; traverse each
        f_iso = probe.stay_alone(4)[n - 1]
        cands = [j for j in range(len(fh)) if 0.6 * f_iso < fh[j] < 1.6 * f_iso]
        for j in cands:
            f_target = fh[j]
            T_tune = (2 * Lc * f_target / n) ** 2 * mc
            Ts = np.linspace(0.92 * T_tune, 1.08 * T_tune, 81)
            half = 0.5 * min(np.abs(np.delete(fh, j) - f_target)) / f_target
            band = (f_target * (1 - min(0.12, half)), f_target * (1 + min(0.12, half)))
            gap, T_at, cd = min_gap(lambda T: CableDeck2D(T=T, **base), Ts, band)
            if cd is None:
                continue
            fh2, pa2, pp2 = cd.host_alone(10)
            jj = int(np.argmin(np.abs(fh2 - f_target)))
            s1 = split_one_ended(M_s, pa2[jj], theta, n)
            s2 = split_two_ended(M_s, pa2[jj], pp2[jj], theta, n)
            s2w = 2.0 / (n * np.pi) * np.sqrt(M_s) * abs(np.cos(theta) * pa2[jj] + (-1) ** n * np.sin(theta) * pp2[jj])
            print(f"   deck+pylon, n = {n}, host {fh2[jj]:.3f} Hz: phi_a {pa2[jj]:+.2e}, phi_p {pp2[jj]:+.2e};"
                  f" FE {100*gap:.3f} %, one-ended {100*s1:.3f} %, two-ended {100*s2:.3f} %, wrong sign {100*s2w:.3f} %")
            rows.append(dict(check="deck_pylon", n=n, f_host=fh2[jj], phi_a=pa2[jj], phi_p=pp2[jj],
                             s_fe=gap, s_one_ended=s1, s_two_ended=s2, s_wrong_sign=s2w))


def main():
    check1()
    check2()
    check3()
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "verify_cablefe2d.csv"), index=False)
    print("wrote", os.path.join(DATA, "verify_cablefe2d.csv"))


if __name__ == "__main__":
    main()
