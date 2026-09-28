# -*- coding: utf-8 -*-
"""Veering width with a flexible pylon: finite element against the one- and
two-ended predictions (cablefe2d.split_one_ended, split_two_ended).

A. Example bridge, pylon of the stay's height, bending stiffness scanned from
   rigid to a sway frequency below the crossing, stay orders 1 and 2.
B. Longest-deck campaign design with a fundamental stay crossing, concrete
   pylon of the stay's height, bending stiffness scanned.
Reads data/campaign.csv; writes data/pylon.csv.
Run:  python3 scripts/run_pylon.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from cablefe2d import CableDeck2D, split_one_ended, split_two_ended  # noqa: E402

DATA = os.path.join(ROOT, "data")
BRIDGE = dict(Ld=80.0, EId=2.0e9, md=1000.0,
              Lc=25.0, EIc=1.2e4, mc=5.5, EA=1.4e8,
              theta=np.deg2rad(35.0), nd=40, nc=40)


def cantilever_f1(EI, m, H):
    """First bending frequency (Hz) of a uniform cantilever of height H."""
    return 1.875104 ** 2 / (2 * np.pi * H ** 2) * np.sqrt(EI / m)


def traverse(make, Ts, n):
    """Scan tension for the narrowest relative gap near stay order n.

    Returns (gap, T, f_lo, f_hi, model) at the narrowest gap.
    """
    best = (np.inf, None, None, None, None)
    for T in Ts:
        cd = make(T)
        f, Phi = cd.modes(40)
        fi = cd.stay_alone(n + 1)[n - 1]
        inb = np.where((f > 0.8 * fi) & (f < 1.25 * fi))[0]
        if len(inb) < 2:
            continue
        ff = f[inb]
        k = int(np.argmin(np.diff(ff)))
        gap = (ff[k + 1] - ff[k]) / (0.5 * (ff[k] + ff[k + 1]))
        if gap < best[0]:
            best = (gap, T, ff[k], ff[k + 1], cd)
    return best


def score(label, cd, n, s_fe, T_at, f_lo, f_hi, extra):
    """Result row: host mode nearest the pair, its deck and pylon ordinates,
    and the finite element, one-ended and two-ended widths."""
    M_s = 0.5 * cd.mc * cd.Lc
    fh, pa, pp = cd.host_alone(16)
    f0 = 0.5 * (f_lo + f_hi)
    j = int(np.argmin(np.abs(fh - f0)))
    s1 = split_one_ended(M_s, pa[j], cd.theta, n)
    s2 = split_two_ended(M_s, pa[j], pp[j], cd.theta, n)
    c, s = np.cos(cd.theta), np.sin(cd.theta)
    row = dict(case=label, n=n, T_at=T_at, f_lo=f_lo, f_hi=f_hi, f_host=fh[j],
               phi_a=pa[j], phi_p=pp[j], term_deck=c * pa[j], term_pylon=s * pp[j],
               pylon_over_deck=(s * pp[j]) / (c * pa[j]) if pa[j] else np.nan,
               s_fe=s_fe, s_one=s1, s_two=s2, ratio_fe_one=s_fe / s1 if s1 else np.nan,
               ratio_fe_two=s_fe / s2 if s2 else np.nan, mu_factor=(s2 / s1) ** 2 if s1 else np.nan)
    row.update(extra)
    return row


def part_a(rows):
    print("A. example bridge with a pylon of the stay's height, stiffness scanned")
    Hp = BRIDGE["Lc"] * np.sin(BRIDGE["theta"])
    m_p = 2000.0
    for n, Ts in ((1, np.linspace(120e3, 190e3, 141)), (2, np.linspace(92e3, 125e3, 133))):
        # rigid reference
        make = lambda T: CableDeck2D(T=T, g=0.0, **BRIDGE)
        s_fe, T_at, f_lo, f_hi, cd = traverse(make, Ts, n)
        r = score("worked_rigid", cd, n, s_fe, T_at, f_lo, f_hi, dict(EI_p=np.inf, f_pylon=np.inf, Hp=Hp))
        rows.append(r)
        print(f"   n = {n} rigid pylon: s_FE {100*s_fe:.3f} %, one-ended {100*r['s_one']:.3f} %")
        for EIp in (1e11, 3e10, 1e10, 5e9, 3e9, 2e9, 1e9, 5e8):
            pyl = dict(EI=EIp, m=m_p, EA=1e11, n=20)
            fp = cantilever_f1(EIp, m_p, Hp)
            make = lambda T, pyl=pyl: CableDeck2D(T=T, g=0.0, pylon=pyl, **BRIDGE)
            s_fe, T_at, f_lo, f_hi, cd = traverse(make, Ts, n)
            if cd is None:
                continue
            r = score("worked_pylon", cd, n, s_fe, T_at, f_lo, f_hi, dict(EI_p=EIp, f_pylon=fp, Hp=Hp))
            rows.append(r)
            print(f"   n = {n} EI_p {EIp:.0e} (sway {fp:5.2f} Hz): host {r['f_host']:.3f} Hz,"
                  f" pylon/deck term {r['pylon_over_deck']:+.3f}; s_FE {100*s_fe:.3f} %,"
                  f" one-ended {100*r['s_one']:.3f} %, two-ended {100*r['s_two']:.3f} %"
                  f" (FE/two {r['ratio_fe_two']:.3f}); mu factor {r['mu_factor']:.2f}")


def part_b(rows):
    print("B. representative long-span design from the design set")
    c = pd.read_csv(os.path.join(DATA, "campaign.csv"))
    g = c[(c.mac > 0.5) & (c.xi > 150) & (c.n_stay == 1)]
    r = g.sort_values("Ld").iloc[-1]
    base = dict(Ld=r.Ld, EId=r.EId, md=r.md, Lc=r.Lc, EIc=r.EIc, mc=r.mc, EA=r.EA,
                theta=r.theta, x_anchor=r.xfrac * r.Ld, nd=40, nc=40)
    Hp = r.Lc * np.sin(r.theta)
    m_p = 15000.0
    Ts = np.linspace(0.7 * r["T"], 1.4 * r["T"], 141)
    make = lambda T: CableDeck2D(T=T, g=0.0, **base)
    s_fe, T_at, f_lo, f_hi, cd = traverse(make, Ts, 1)
    rr = score("longspan_rigid", cd, 1, s_fe, T_at, f_lo, f_hi,
               dict(EI_p=np.inf, f_pylon=np.inf, Hp=Hp, Ld=r.Ld, Lc=r.Lc, theta_deg=np.degrees(r.theta)))
    rows.append(rr)
    print(f"   Ld {r.Ld:.0f} m, Lc {r.Lc:.0f} m, theta {np.degrees(r.theta):.0f} deg, pylon height {Hp:.0f} m;"
          f" rigid: s_FE {100*s_fe:.3f} %, host {rr['f_host']:.3f} Hz")
    for EIp in (2e13, 5e12, 2e12, 1e12, 5e11, 2e11):
        pyl = dict(EI=EIp, m=m_p, EA=1e12, n=20)
        fp = cantilever_f1(EIp, m_p, Hp)
        make = lambda T, pyl=pyl: CableDeck2D(T=T, g=0.0, pylon=pyl, **base)
        s_fe, T_at, f_lo, f_hi, cd = traverse(make, Ts, 1)
        if cd is None:
            continue
        rr = score("longspan_pylon", cd, 1, s_fe, T_at, f_lo, f_hi,
                   dict(EI_p=EIp, f_pylon=fp, Hp=Hp, Ld=r.Ld, Lc=r.Lc, theta_deg=np.degrees(r.theta)))
        rows.append(rr)
        print(f"   EI_p {EIp:.0e} (sway {fp:5.2f} Hz): host {rr['f_host']:.3f} Hz, pylon/deck term {rr['pylon_over_deck']:+.3f};"
              f" s_FE {100*s_fe:.3f} %, one-ended {100*rr['s_one']:.3f} %, two-ended {100*rr['s_two']:.3f} %"
              f" (FE/two {rr['ratio_fe_two']:.3f}); mu factor {rr['mu_factor']:.2f}")


def main():
    rows = []
    part_a(rows)
    part_b(rows)
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "pylon.csv"), index=False)
    print("wrote", os.path.join(DATA, "pylon.csv"))


if __name__ == "__main__":
    main()
