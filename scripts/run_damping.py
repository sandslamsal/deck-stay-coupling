# -*- coding: utf-8 -*-
"""Damping and peak resolvability at a deck-stay crossing.

Part 1 applies the two-Lorentzian criterion (a dip between the peaks exists for
s > 0.9717 zeta) to the graded designs of data/campaign.csv. Part 2 computes
the driving-point receptance at the stay sensor of the example bridge at
T = 151.6 kN with Rayleigh damping and finds the damping ratio at which the
two peaks merge. Writes data/damping.csv.
Run:  python3 scripts/run_damping.py
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

from cablefe import CableDeck  # noqa: E402

DATA = os.path.join(ROOT, "data")

# example bridge of scripts/run_identify2.py; check_bridge_copy() asserts it
BRIDGE = dict(Ld=80.0, EId=2.0e9, md=1000.0,
              Lc=25.0, EIc=1.2e4, mc=5.5, EA=1.4e8,
              theta=np.deg2rad(35.0), nd=40, nc=40)
T_TUNE = 151.6e3                    # N, exact-tuning tension of that bridge
ZETAS = (0.002, 0.005, 0.010, 0.020)
FBAND = (2.5, 4.2)
NF = 17001                          # df = 1e-4 Hz; the narrowest peak here
                                    # (zeta 0.2 %) is 133 points wide
PROM_DB = 3.0
R_DIP = np.sqrt(np.sqrt(5.0) - 2.0)     # dip exists for r > R_DIP


def check_bridge_copy():
    """The BRIDGE dict above must match scripts/run_identify2.py."""
    src = open(os.path.join(HERE, "run_identify2.py")).read().replace(" ", "")
    for tok in ("Ld=80.0", "EId=2.0e9", "md=1000.0", "Lc=25.0",
                "EIc=1.2e4", "mc=5.5", "EA=1.4e8",
                "theta=np.deg2rad(35.0)", "nd=40", "nc=40"):
        assert tok in src, f"BRIDGE drift: {tok} not in run_identify2.py"


# --- two-Lorentzian pair model --------------------------------------------
# u = (omega - omega_0) / (zeta omega_0), r = s / (2 zeta)

def pair_mag(u, r):
    """|h(u)| for the equal-residue in-phase pair at resolution number r."""
    return np.abs(1.0 / (1.0 + 1j * (u - r)) + 1.0 / (1.0 + 1j * (u + r)))


def dip_exists(r, e=1e-3):
    """Is the midpoint a local minimum of |h|?"""
    return pair_mag(0.0, r) < pair_mag(e, r)


def dip_depth_db(r):
    """Lower-peak-over-midpoint depth in dB (0 when there is no dip)."""
    if not dip_exists(r):
        return 0.0
    u = np.linspace(0.0, r + 6.0, 20001)
    pk = pair_mag(u, r).max()
    return 20.0 * np.log10(pk / pair_mag(0.0, r))


def bisect(pred, lo, hi, it=40):
    """Smallest x in (lo, hi) with pred(x) True; pred monotone in x."""
    assert not pred(lo) and pred(hi)
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if pred(mid) else (mid, hi)
    return 0.5 * (lo + hi)


def model_thresholds():
    r_dip_num = bisect(dip_exists, 0.05, 2.0)
    assert abs(r_dip_num - R_DIP) < 1e-4, \
        f"dip threshold mismatch: numeric {r_dip_num:.6f} vs closed form {R_DIP:.6f}"
    r_3db = bisect(lambda r: dip_depth_db(r) > PROM_DB, r_dip_num, 6.0)
    return r_dip_num, r_3db


# --- driven bridge ---------------------------------------------------------

class DrivenBridge:
    """The example bridge with its eigensolution computed once and cached."""

    def __init__(self, T):
        self.cd = CableDeck(T=T, **BRIDGE)
        w2, V = eigh(self.cd.K, self.cd.M)
        self.w = np.sqrt(np.maximum(w2, 0.0))       # all reduced modes
        Phi = self.cd.Lmat @ V                       # mass-normalized
        self.p = self.cd.cable_sensor_dof(2.0)
        self.phip = Phi[self.p, :]
        f = self.w / (2.0 * np.pi)
        inb = f[(f > FBAND[0]) & (f < FBAND[1])]
        assert len(inb) == 2, f"expected the pair alone in band, got {inb}"
        self.f_lo, self.f_hi = float(inb[0]), float(inb[1])

    def rayleigh(self, zeta):
        """Rayleigh alpha, beta giving damping ratio zeta on both pair modes."""
        wa, wb = 2 * np.pi * self.f_lo, 2 * np.pi * self.f_hi
        return (2.0 * zeta * wa * wb / (wa + wb),
                2.0 * zeta / (wa + wb))

    def frf(self, zeta, fgrid):
        """Driving-point receptance at the stay sensor, exact modal sum."""
        alpha, beta = self.rayleigh(zeta)
        om = 2.0 * np.pi * fgrid
        den = (self.w ** 2)[None, :] - (om ** 2)[:, None] \
            + 1j * om[:, None] * (alpha + beta * self.w ** 2)[None, :]
        return (self.phip ** 2 / den).sum(axis=1)

    def verify_frf(self, zeta, fgrid, nspot=5, seed=0):
        """Spot-check the modal sum against direct complex solves.

        Errors are normalized by the band peak; a pointwise relative error is
        ill-conditioned at the antiresonance notch.
        """
        H = self.frf(zeta, fgrid)
        scale = np.abs(H).max()
        alpha, beta = self.rayleigh(zeta)
        C = alpha * self.cd.M + beta * self.cd.K
        Fred = self.cd.Lmat.T[:, self.p].copy()
        rng = np.random.default_rng(seed)
        worst = 0.0
        for k in rng.integers(0, len(fgrid), nspot):
            om = 2.0 * np.pi * fgrid[k]
            A = self.cd.K + 1j * om * C - om ** 2 * self.cd.M
            Hd = (self.cd.Lmat @ np.linalg.solve(A, Fred))[self.p]
            worst = max(worst, abs(Hd - H[k]) / scale)
        assert worst < 1e-8, f"modal sum vs direct solve: {worst:.2e}"
        return worst


def peak_census(fgrid, H):
    """Peaks above the 3 dB prominence floor, plus the raw local maxima."""
    db = 20.0 * np.log10(np.abs(H))
    pk3, _ = find_peaks(db, prominence=PROM_DB)
    pk0, _ = find_peaks(db, prominence=1e-6)
    out = dict(n_peaks_3db=len(pk3), n_maxima=len(pk0),
               f_pk1=np.nan, f_pk2=np.nan, sep_hz=np.nan, dip_db=np.nan)
    fp = fgrid[pk3]
    if len(fp) >= 1:
        out["f_pk1"] = fp[0]
    if len(fp) >= 2:
        out["f_pk2"] = fp[1]
        out["sep_hz"] = fp[1] - fp[0]
    if len(pk0) >= 2:                 # dip depth below the lower maximum
        lo, hi = pk0[0], pk0[-1]
        dip = db[lo:hi + 1].min()
        out["dip_db"] = min(db[lo], db[hi]) - dip
    return out


# ---------------------------------------------------------------------------

def main():
    check_bridge_copy()
    r_dip_num, r_3db = model_thresholds()
    print("resolvability thresholds (r = s / 2 zeta):")
    print(f"  dip exists      r > {r_dip_num:.4f}  ->  "
          f"s > {2*r_dip_num:.4f} zeta   (closed form 2 sqrt(sqrt5-2))")
    print(f"  dip >= 3 dB     r > {r_3db:.4f}  ->  s > {2*r_3db:.4f} zeta")
    print("  half-power rule                    s > 2 zeta")

    rows = [dict(record="model_threshold", r_dip=r_dip_num, r_3db=r_3db,
                 s_over_zeta_dip=2 * r_dip_num, s_over_zeta_3db=2 * r_3db)]

    # ---- part 1: the campaign under the criterion ------------------------
    d = pd.read_csv(os.path.join(DATA, "campaign.csv"))
    g = d[(d.mac > 0.5) & (d.xi > 150)]
    s = g.s.to_numpy()
    print(f"\ndesign set: {len(s)} graded designs "
          f"(median s = {np.median(s)*100:.2f} %)")
    print("  zeta   unresolvable fraction:   dip     3 dB    s<2zeta")
    for z in ZETAS:
        fr = dict(record="campaign", zeta_pct=100 * z,
                  s_star_dip=2 * r_dip_num * z, s_star_3db=2 * r_3db * z,
                  frac_unres_dip=float(np.mean(s <= 2 * r_dip_num * z)),
                  frac_unres_3db=float(np.mean(s <= 2 * r_3db * z)),
                  frac_unres_halfpower=float(np.mean(s <= 2 * z)),
                  n_graded=len(s))
        rows.append(fr)
        print(f"  {100*z:4.1f} %                       "
              f"{fr['frac_unres_dip']:.3f}   {fr['frac_unres_3db']:.3f}"
              f"   {fr['frac_unres_halfpower']:.3f}")

    # ---- part 2: the driven example bridge --------------------------------
    br = DrivenBridge(T_TUNE)
    f0 = 0.5 * (br.f_lo + br.f_hi)
    s_meas = (br.f_hi - br.f_lo) / f0
    print(f"\nexample bridge at T = {T_TUNE/1e3:.1f} kN:")
    print(f"  hybrid pair {br.f_lo:.4f} / {br.f_hi:.4f} Hz, "
          f"f0 = {f0:.4f} Hz, s = {100*s_meas:.3f} %")

    # check that 151.6 kN is near the closest approach of the pair
    Ts = np.linspace(145e3, 158e3, 27)
    gaps = []
    for t in Ts:
        b = DrivenBridge(t)
        gaps.append(b.f_hi - b.f_lo)
    tmin = Ts[int(np.argmin(gaps))]
    print(f"  minimum pair gap over 145-158 kN at T = {tmin/1e3:.1f} kN "
          f"(gap {min(gaps):.4f} Hz); analysis at 151.6 kN")

    fgrid = np.linspace(FBAND[0], FBAND[1], NF)
    worst = br.verify_frf(ZETAS[0], fgrid)
    print(f"  modal sum vs direct solve, worst spot error "
          f"{worst:.1e} of band peak")

    print(f"\n  FRF at the stay sensor DOF, {FBAND[0]}-{FBAND[1]} Hz:")
    for z in ZETAS:
        c = peak_census(fgrid, br.frf(z, fgrid))
        merged = c["n_peaks_3db"] < 2
        rows.append(dict(record="frf", zeta_pct=100 * z, merged=int(merged),
                         **c))
        pk = (f"{c['f_pk1']:.4f} Hz" if merged
              else f"{c['f_pk1']:.4f}+{c['f_pk2']:.4f} Hz")
        print(f"    zeta {100*z:4.1f} %:  {c['n_peaks_3db']} peak(s) "
              f"[{pk}], dip {c['dip_db']:.2f} dB "
              f"-> {'merged' if merged else 'resolvable'}")

    # merge zeta by bisection, against the two-Lorentzian predictions
    def merged_at(z):
        return peak_census(fgrid, br.frf(z, fgrid))["n_peaks_3db"] < 2

    def dip_gone_at(z):
        return peak_census(fgrid, br.frf(z, fgrid))["n_maxima"] < 2

    z3 = bisect(merged_at, ZETAS[0], 0.06, it=20)
    zd = bisect(dip_gone_at, z3, 0.08, it=20)
    z3_pred = s_meas / (2 * r_3db)
    zd_pred = s_meas / (2 * r_dip_num)
    rows.append(dict(record="bridge_summary", T=T_TUNE, f_lo=br.f_lo,
                     f_hi=br.f_hi, f0=f0, s_bridge=s_meas,
                     zeta_merge_3db_pct=100 * z3,
                     zeta_dipvanish_pct=100 * zd,
                     zeta_merge_3db_pred_pct=100 * z3_pred,
                     zeta_dipvanish_pred_pct=100 * zd_pred))
    print(f"\n  merge (3 dB) at zeta = {100*z3:.2f} %   "
          f"(two-Lorentzian prediction {100*z3_pred:.2f} %)")
    print(f"  dip vanishes at zeta = {100*zd:.2f} %   "
          f"(prediction {100*zd_pred:.2f} %)")

    out = pd.DataFrame(rows)
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "damping.csv")
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
