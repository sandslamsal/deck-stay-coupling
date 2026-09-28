# -*- coding: utf-8 -*-
"""Revision 1, R1.8: three more finite element points for the resolvability
check.

scripts/run_damping.py drives the worked bridge at its crossing with
Rayleigh damping calibrated on both hybrid modes and finds, by bisection,
the damping at which the two peaks of the stay-mounted driving-point
receptance merge (3 dB prominence) and at which the dip vanishes; the
manuscript reports the merge earlier than the equal-residue two-Lorentzian
model predicts, 0.81 against 1.03 per cent, and calls the population
fractions lower bounds on that one point. This script repeats the check on
three graded campaign designs chosen for their split: near the population
median (0.32 %), the ninetieth percentile (1.5 %) and the ninety-ninth
(4.5 %), each at its own crossing, found by traversing the tension.

The sensor sits a short way up the chord from the anchorage, 2 m or five
per cent of the chord, whichever is larger. Everything else follows
run_damping.py: exact modal-sum receptance, 3 dB prominence, bisection.

Writes data/damping_points.csv.

Run:  python3 scripts/run_damping_points.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from cablefe import CableDeck, tensioned_beam_freq  # noqa: E402

DATA = os.path.join(ROOT, "data")
PROM_DB = 3.0
R_DIP = np.sqrt(np.sqrt(5.0) - 2.0)
R_3DB = 1.1398
TARGET_S = (0.0032, 0.015, 0.045)


class Driven:
    def __init__(self, row, T):
        self.row = row
        self.cd = CableDeck(Ld=row.Ld, EId=row.EId, md=row.md, Lc=row.Lc, EIc=row.EIc,
                            mc=row.mc, T=T, EA=row.EA, theta=row.theta,
                            x_anchor=row.xfrac * row.Ld, nd=40, nc=40)
        w2, V = eigh(self.cd.K, self.cd.M)
        self.w = np.sqrt(np.maximum(w2, 0.0))
        Phi = self.cd.Lmat @ V
        self.p = self.cd.cable_sensor_dof(max(2.0, 0.05 * row.Lc))
        self.phip = Phi[self.p, :]
        self.f = self.w / (2 * np.pi)
        self.es = self.cd.energy_split(Phi)

    def pair(self, n):
        fi = tensioned_beam_freq(n, self.row.Lc, self.cd.T, self.row.EIc, self.row.mc)
        inb = np.where((self.f > 0.85 * fi) & (self.f < 1.18 * fi))[0]
        if len(inb) < 2:
            return None
        ff = self.f[inb]
        k = int(np.argmin(np.diff(ff)))
        return inb[k], inb[k + 1]

    def rayleigh(self, zeta, i, j):
        wa, wb = self.w[i], self.w[j]
        return 2 * zeta * wa * wb / (wa + wb), 2 * zeta / (wa + wb)

    def frf(self, zeta, i, j, fgrid):
        alpha, beta = self.rayleigh(zeta, i, j)
        om = 2 * np.pi * fgrid
        den = (self.w ** 2)[None, :] - (om ** 2)[:, None] \
            + 1j * om[:, None] * (alpha + beta * self.w ** 2)[None, :]
        return (self.phip ** 2 / den).sum(axis=1)


def census(fgrid, H):
    db = 20 * np.log10(np.abs(H))
    pk3, _ = find_peaks(db, prominence=PROM_DB)
    pk0, _ = find_peaks(db, prominence=1e-6)
    return len(pk3), len(pk0)


def bisect(pred, lo, hi, it=18):
    assert not pred(lo) and pred(hi), "not bracketed"
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if pred(mid) else (mid, hi)
    return 0.5 * (lo + hi)


def main():
    c = pd.read_csv(os.path.join(DATA, "campaign.csv"))
    g = c[(c.mac > 0.5) & (c.xi > 150) & (c.n_stay == 1)].copy()
    rows = []
    for st in TARGET_S:
        row = g.iloc[int(np.argmin(np.abs(np.log(g.s / st))))]
        n = int(row.n_stay)
        # traverse to the crossing
        Ts = np.linspace(0.85 * row["T"], 1.18 * row["T"], 67)
        best = (np.inf, None)
        for T in Ts:
            d = Driven(row, T)
            pr = d.pair(n)
            if pr is None:
                continue
            i, j = pr
            gap = (d.f[j] - d.f[i]) / (0.5 * (d.f[i] + d.f[j]))
            if gap < best[0]:
                best = (gap, T)
        s_fe, T_at = best
        d = Driven(row, T_at)
        i, j = d.pair(n)
        f0 = 0.5 * (d.f[i] + d.f[j])
        half = max(6 * s_fe, 0.03)
        fgrid = np.linspace(f0 * (1 - half), f0 * (1 + half), 40001)
        amp_ratio = (d.phip[j] ** 2) / (d.phip[i] ** 2)

        def merged(z):
            return census(fgrid, d.frf(z, i, j, fgrid))[0] < 2

        def dipgone(z):
            return census(fgrid, d.frf(z, i, j, fgrid))[1] < 2

        z_lo = s_fe / (2 * R_3DB) * 0.15
        z_hi = s_fe / (2 * R_DIP) * 4.0
        z3 = bisect(merged, z_lo, z_hi)
        zd = bisect(dipgone, z3 * 0.999, z_hi * 3.0)
        z3_pred, zd_pred = s_fe / (2 * R_3DB), s_fe / (2 * R_DIP)
        r = dict(target_s=st, s_fe=s_fe, T_at=T_at, f_lo=d.f[i], f_hi=d.f[j], f0=f0,
                 Ld=row.Ld, Lc=row.Lc, theta_deg=np.degrees(row.theta), n=n,
                 residue_ratio_at_sensor=amp_ratio,
                 zeta_merge_3db_pct=100 * z3, zeta_merge_3db_pred_pct=100 * z3_pred,
                 zeta_dipvanish_pct=100 * zd, zeta_dipvanish_pred_pct=100 * zd_pred,
                 ratio_3db=z3 / z3_pred, ratio_dip=zd / zd_pred)
        rows.append(r)
        print(f"s = {100*s_fe:.3f} % (Ld {row.Ld:.0f} m, Lc {row.Lc:.1f} m, theta {np.degrees(row.theta):.0f} deg):"
              f" residue ratio at sensor {amp_ratio:.2f};"
              f" 3 dB merge at zeta {100*z3:.3f} % vs predicted {100*z3_pred:.3f} % (x{z3/z3_pred:.2f});"
              f" dip vanishes at {100*zd:.3f} % vs {100*zd_pred:.3f} % (x{zd/zd_pred:.2f})")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA, "damping_points.csv"), index=False)
    print("wrote", os.path.join(DATA, "damping_points.csv"))


if __name__ == "__main__":
    main()
