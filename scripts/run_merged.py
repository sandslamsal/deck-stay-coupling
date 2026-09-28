# -*- coding: utf-8 -*-
"""Tension error read from a merged peak, when a veering pair does not resolve.

Part A: the maximum x* of |H|^2 for H = 1/(x + u + i) + rho/(x - u + i), with
x = (w - w0)/(zeta w0) and u = s/(2 zeta), from the roots of its stationarity
quintic; at rho = 1, x*^2 = u sqrt(u^2 + 4) - 1. Part B: the driven worked
bridge of scripts/run_damping.py over T = 140-165 kN and each zeta, with the
picked peak compared with the branch errors and the Part A law.
Writes data/merged.csv.

Run:  python3 scripts/run_merged.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from cablefe import (CableDeck, chain, invert_string,  # noqa: E402
                     mu_effective, tensioned_beam_freq, veering_split)
from run_damping import (BRIDGE, DrivenBridge, FBAND, NF,  # noqa: E402
                         T_TUNE, ZETAS, peak_census)

DATA = os.path.join(ROOT, "data")


# --- Part A: unequal-residue two-Lorentzian model ---

def hmag2(x, u, rho):
    """|H|^2 for H = 1/(x+u+i) + rho/(x-u+i), vectorized in x."""
    x = np.asarray(x, dtype=float)
    S, D = 1.0 + rho, rho - 1.0
    num = (S * x + u * D) ** 2 + S ** 2
    den = (x ** 2 - 1.0 - u ** 2) ** 2 + 4.0 * x ** 2
    return num / den


def hmag2_direct(x, u, rho):
    """|H|^2 from the definition, to check hmag2."""
    x = np.asarray(x, dtype=float)
    return np.abs(1.0 / (x + u + 1j) + rho / (x - u + 1j)) ** 2


def stationary_poly(u, rho):
    """Coefficients (highest first) of N' Dn - N Dn', the quintic in x."""
    S, D = 1.0 + rho, rho - 1.0
    lin = np.array([S, u * D])                    # S x + u D
    N = np.polyadd(np.polymul(lin, lin), np.array([S ** 2]))
    Nd = np.polymul(np.array([2.0 * S]), lin)     # dN/dx
    q = np.array([1.0, 0.0, -(1.0 + u ** 2)])     # x^2 - 1 - u^2
    Dn = np.polyadd(np.polymul(q, q), np.array([4.0, 0.0, 0.0]))
    Dnd = np.array([4.0, 0.0, 4.0 * (1.0 - u ** 2), 0.0])   # dDn/dx
    return np.polysub(np.polymul(Nd, Dn), np.polymul(N, Dnd))


def _xstar_core(u, rho):
    """argmax of |H|^2 from the exact stationarity polynomial (rho <= 1)."""
    r = np.roots(stationary_poly(u, rho))
    xs = np.real(r[np.abs(np.imag(r)) < 1e-9 * (1.0 + np.abs(np.real(r)))])
    if not len(xs):
        xs = np.real(r)
    v = hmag2(xs, u, rho)
    return float(xs[int(np.argmax(v))])


def xstar(u, rho):
    """argmax of |H|^2 for any rho >= 0.

    For rho > 1 the exact symmetry |H|^2(x; rho) = rho^2 |H|^2(-x; 1/rho)
    is used, which keeps the polynomial well conditioned when one residue
    dominates.
    """
    if rho > 1.0:
        return -_xstar_core(u, 1.0 / rho)
    return _xstar_core(u, rho)


def xstar_scan(u, rho, n=400001):
    """argmax by dense scan plus Brent refinement, to check xstar."""
    span = u + 12.0
    g = np.linspace(-span, span, n)
    v = hmag2(g, u, rho)
    i = int(np.argmax(v))
    i = min(max(i, 1), n - 2)
    res = minimize_scalar(lambda z: -hmag2(z, u, rho),
                          bracket=(g[i - 1], g[i], g[i + 1]), method="brent")
    return float(res.x) if -res.fun >= v[i] else float(g[i])


def xstar_equal_closed(u):
    """Closed form at rho = 1: x*^2 = u sqrt(u^2+4) - 1 when positive.

    At rho = 1 the quintic factors as x [x^4 + 2 x^2 - (u^4 + 4u^2 - 1)] = 0,
    so the side maxima exist when u^4 + 4u^2 - 1 > 0, the dip condition.
    """
    y = u * np.sqrt(u ** 2 + 4.0) - 1.0
    return np.sqrt(y) if y > 0.0 else 0.0


def rho_from_split(d, Delta):
    """Residue ratio (upper/lower) from the observed split Delta and detuning d.

    Same as :func:`rho_2dof`, but with the measured Delta = sqrt(d^2 + s^2).
    """
    return (Delta + d) / (Delta - d)


def rho_2dof(d, s):
    """Residue ratio (upper/lower) of a 2-DOF veering pair at a stay sensor.

    rho = (Delta + d)/(Delta - d), with Delta = sqrt(d^2 + s^2).
    """
    Delta = np.hypot(d, s)
    return (Delta + d) / (Delta - d)


def eps_merged_law(d, s, zeta):
    """Merged-peak tension error of the 2-DOF pair, relative (not percent).

    eps = (f_pick / f_iso)^2 - 1 ~= 2 zeta x* - d, with x* the argmax of the
    unequal-residue model at u = Delta/(2 zeta) and rho = rho_2dof(d, s).
    """
    Delta = np.hypot(d, s)
    u = Delta / (2.0 * zeta)
    return 2.0 * zeta * xstar(u, rho_2dof(d, s)) - d


def _sgn(d):
    """Sign of d, taken as +1 at d = 0 (np.sign would give a zero branch error)."""
    return 1.0 if d >= 0 else -1.0


def eps_branch_law(d, s):
    """Stay-dominated branch error, signed: sgn(d) (sqrt(d^2+s^2) - |d|)."""
    return _sgn(d) * (np.hypot(d, s) - abs(d))


def eps_wrong_law(d, s):
    """Wrong-branch error, signed."""
    return -_sgn(d) * (np.hypot(d, s) + abs(d))


def k_factor(u, rho):
    """x*/u: the fraction of the half separation the merged peak reaches."""
    return xstar(u, rho) / u


def u_collapse(rho, lo=1e-6, hi=50.0, it=200):
    """u at which |x*|/u is half way between its u -> 0 limit |k0| and 1.

    k0 = (rho-1)/(rho+1). Bisection; |k| is monotone in u below its maximum.
    """
    k0 = abs((rho - 1.0) / (rho + 1.0))
    tgt = 0.5 * (k0 + 1.0)
    if abs(k_factor(lo, rho)) > tgt:
        return np.nan
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if abs(k_factor(mid, rho)) > tgt:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def part_a(quick=False):
    print("=" * 74)
    print("PART A  unequal-residue merged peak")
    print("=" * 74)

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(400):
        u = float(rng.uniform(0.01, 8.0))
        rho = float(rng.uniform(0.0, 6.0))
        x = rng.uniform(-10, 10, 50)
        a, b = hmag2(x, u, rho), hmag2_direct(x, u, rho)
        worst = max(worst, float(np.max(np.abs(a - b) / np.abs(b))))
    print(f"combined |H|^2 vs definition, worst relative error {worst:.2e}")
    assert worst < 1e-10

    # equal-residue form at rho = 1
    x = np.linspace(-8, 8, 1001)
    app = 4.0 * (x ** 2 + 1.0) / ((x ** 2 - 1.0 - 2.5 ** 2) ** 2 + 4 * x ** 2)
    assert np.allclose(hmag2(x, 2.5, 1.0), app, rtol=1e-12)
    print("reduces to the equal-residue form at rho = 1          ok")

    # roots vs scan; at rho = 1 the two maxima are equal, so x* is compared
    # up to sign there, and the attained maximum is compared in every case
    worst_x = worst_v = 0.0
    cases = [(u, r) for u in (0.05, 0.2, 0.5, 0.9717 / 2, 1.0, 1.14, 2.0,
                              4.0, 10.0)
             for r in (0.0, 0.1, 0.3, 0.6, 1.0, 1.8, 4.0, 20.0)]
    for u, r in cases:
        a, b = xstar(u, r), xstar_scan(u, r)
        dx = abs(abs(a) - abs(b)) if r == 1.0 else abs(a - b)
        worst_x = max(worst_x, dx)
        va, vb = hmag2(a, u, r), hmag2(b, u, r)
        worst_v = max(worst_v, abs(va - vb) / max(va, vb))
    print(f"quintic roots vs dense scan, worst |dx*| {worst_x:.2e}, "
          f"worst relative |H|^2 gap {worst_v:.2e}")
    assert worst_x < 1e-5 and worst_v < 1e-12

    # closed form at rho = 1 and the dip threshold
    u_dip = np.sqrt(np.sqrt(5.0) - 2.0)
    worst = 0.0
    for u in np.linspace(0.02, 12.0, 200):
        worst = max(worst, abs(abs(xstar(u, 1.0)) - xstar_equal_closed(u)))
    print(f"rho = 1 closed form x*^2 = u sqrt(u^2+4) - 1, "
          f"worst error {worst:.2e}")
    assert worst < 1e-7
    print(f"side maxima appear at u = {u_dip:.6f} "
          f"(closed-form dip threshold {u_dip:.6f})   ok")

    # limits; the approach to the weighted mean is second order in u
    print("\nlimits")
    for rho in (0.0, 0.25, 0.5, 2.0, 4.0):
        k0 = (rho - 1.0) / (rho + 1.0)
        e1 = abs(xstar(1e-2, rho) / 1e-2 - k0)
        e2 = abs(xstar(1e-3, rho) / 1e-3 - k0)
        got0 = xstar(1e-3, rho) / 1e-3
        got_inf = xstar(1e3, rho) / 1e3
        want_inf = 1.0 if rho > 1 else (-1.0 if rho < 1 else 0.0)
        order = np.log10(e1 / e2) if e2 > 0 else np.inf
        print(f"  rho = {rho:5.2f}   u->0: x*/u = {got0:+.8f} "
              f"(weighted mean {k0:+.8f}, order {order:.2f})    "
              f"u->inf: x*/u = {got_inf:+.6f} (branch {want_inf:+.1f})")
        assert e2 < 1e-5
        assert abs(got_inf - want_inf) < 2e-3

    # k(u, rho) map
    print("\n  x*/u, the fraction of the half split the merged peak reaches")
    us = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    rhos = [0.0, 0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0, 10.0]
    print("      u \\ rho " + "".join(f"{r:7.2f}" for r in rhos))
    for u in us:
        print(f"      {u:6.2f}   " +
              "".join(f"{xstar(u, r)/u:+7.3f}" for r in rhos))

    # the sum rule, and the closed 2-DOF law
    print("\nsum rule: residue-weighted mean of the branch positions")
    s = 0.02338
    for d in (-0.06, -0.02, -0.005, 0.0, 0.005, 0.02, 0.06):
        De = np.hypot(d, s)
        r = rho_2dof(d, s)
        xbar_rel = (1.0 * (-De / 2) + r * (De / 2)) / (1.0 + r)   # in f/f0
        print(f"  d = {d:+.4f}  rho = {r:9.4f}   weighted mean "
              f"{xbar_rel:+.6f}  vs  d/2 = {d/2:+.6f}")
        assert abs(xbar_rel - d / 2) < 1e-12
    print("  in the u -> 0 limit the peak sits at f_iso and the coupling "
          "bias is zero;\n  u -> 0 needs damping of the order of the "
          "split, and on the way\n  the peak first overshoots the branch "
          "(next block)")

    print("\nmerged-peak law, s = 2.338 % (example bridge), errors in percent")
    print("     d       branch    wrong  |  " +
          "  ".join(f"zeta={100*z:.1f}%" for z in ZETAS))
    for d in (-0.08, -0.04, -0.02, -0.01, -0.005, -0.002, 0.0,
              0.002, 0.005, 0.01, 0.02, 0.04, 0.08):
        row = [100 * eps_merged_law(d, s, z) for z in ZETAS]
        print(f"  {d:+.4f}  {100*eps_branch_law(d, s):+8.3f} "
              f"{100*eps_wrong_law(d, s):+8.3f}  |  " +
              "  ".join(f"{v:+8.3f}" for v in row))

    # peak repulsion: the merged maximum overshoots the pole
    print("\npeak repulsion (equal residues): the maximum of the composite "
          "sits outside\n  the pole, because the in-phase tail of the other "
          "resonance is real and\n  positive on the far side, so x*/u > 1.")
    ug = np.geomspace(0.3, 60.0, 4000)
    kg = np.array([k_factor(float(v), 1.0) for v in ug])
    i = int(np.argmax(kg))
    print(f"  max overshoot x*/u = {kg[i]:.4f} at u = {ug[i]:.4f}")
    u_one = ug[np.argmax(kg > 1.0)]
    print(f"  x*/u crosses 1 at u = {u_one:.4f}; "
          f"x*/u = 0 at u = {np.sqrt(np.sqrt(5)-2):.4f} (the dip threshold)")
    print(f"  at exact tuning the observed bias vanishes only "
          f"when\n  zeta > s / (2 x {np.sqrt(np.sqrt(5)-2):.4f}) = "
          f"{1/(2*np.sqrt(np.sqrt(5)-2)):.3f} s, i.e. "
          f"{100*2.338e-2/(2*np.sqrt(np.sqrt(5)-2)):.2f} % damping on the "
          f"example bridge")
    print("\n  halfway threshold: u at which |x*|/u reaches the "
          "midpoint\n  between its u->0 limit |k0| and 1")
    for r in (1.0, 0.9, 0.75, 0.5, 0.3, 0.1, 0.03):
        uc = u_collapse(r)
        k0 = abs((r - 1.0) / (r + 1.0))
        tail = (f"u_c = {uc:.4f}  (zeta = {1/(2*uc):5.2f} s)"
                if np.isfinite(uc) else "u_c = n/a")
        print(f"    rho = {r:5.2f}  |k0| = {k0:.3f}   {tail}")

    # worst merged error over detuning, per zeta
    print("\n  worst |merged| over detuning (analytical, s = 2.338 %)")
    # d = 0 is excluded: rho = 1 there, so the sign of the error is undefined
    dg = np.concatenate([-np.geomspace(3e-4, 0.3, 600)[::-1],
                         np.geomspace(3e-4, 0.3, 600)])
    rows = []
    for z in ZETAS:
        e = np.array([eps_merged_law(float(d), s, z) for d in dg])
        eb = np.array([eps_branch_law(float(d), s) for d in dg])
        i = int(np.argmax(np.abs(e)))
        ratio = np.abs(e) / np.abs(eb)
        j = int(np.argmax(ratio))
        flips = int(np.sum(e * eb < 0))
        rows.append(dict(record="law_worst", zeta_pct=100 * z, s=s,
                         worst_eps_pct=100 * e[i], d_worst=float(dg[i]),
                         branch_at_worst_pct=100 * eb[i],
                         max_ratio=float(ratio.max()),
                         d_max_ratio=float(dg[j]),
                         min_ratio=float(ratio.min()),
                         d_min_ratio=float(dg[int(np.argmin(ratio))]),
                         n_sign_flip=flips, n_grid=len(dg)))
        print(f"    zeta {100*z:4.1f} %:  worst |eps| {abs(100*e[i]):6.3f} % at "
              f"d = {dg[i]:+.4f} (branch {abs(100*eb[i]):6.3f} %)"
              f"   merged/branch in [{ratio.min():.3f}, {ratio.max():.3f}]"
              f"   sign flips {flips}")
    return rows


# --- Part B: driven finite element bridge ---

def band_pair(br):
    """(i_lo, i_hi) indices of the two in-band modes of a DrivenBridge."""
    f = br.w / (2.0 * np.pi)
    idx = np.where((f > FBAND[0]) & (f < FBAND[1]))[0]
    assert len(idx) == 2, f"expected 2 modes in band, got {f[idx]}"
    return int(idx[0]), int(idx[1])


def residue_ratio(br):
    """Residue ratio rho at the sensor, as Part A defines it.

    The receptance residue is phi^2 / w^2, not phi^2; the w^-2 factor decides
    which of two nearly equal peaks is taller. Returns (rho, phi ratio squared).
    """
    i, j = band_pair(br)
    amp = float((br.phip[j] / br.phip[i]) ** 2)
    return amp * float((br.w[i] / br.w[j]) ** 2), amp


def frf_subset(br, zeta, fgrid, idx):
    """Receptance from a chosen subset of modes only, same damping model."""
    alpha, beta = br.rayleigh(zeta)
    om = 2.0 * np.pi * np.atleast_1d(fgrid)
    w, ph = br.w[idx], br.phip[idx]
    den = (w ** 2)[None, :] - (om ** 2)[:, None] \
        + 1j * om[:, None] * (alpha + beta * w ** 2)[None, :]
    return (ph ** 2 / den).sum(axis=1)


def argmax_f(hfun, fgrid):
    """Global maximum of |h| over the grid, refined off it by Brent."""
    a = np.abs(hfun(fgrid))
    i = int(np.argmax(a))
    i = min(max(i, 1), len(fgrid) - 2)
    fun = lambda ff: -abs(hfun(np.array([ff]))[0])          # noqa: E731
    res = minimize_scalar(fun, bracket=(fgrid[i - 1], fgrid[i], fgrid[i + 1]),
                          method="brent")
    f_ref = float(res.x)
    if not (fgrid[0] < f_ref < fgrid[-1]) or -res.fun < a[i]:
        f_ref = float(fgrid[i])
    return f_ref


def flat_width(fgrid, a, drop_db=0.1):
    """Width of the contiguous plateau within ``drop_db`` of the maximum.

    Only the run containing the maximum is measured.
    """
    db = 20.0 * np.log10(a / a.max())
    i = int(np.argmax(a))
    lo = i
    while lo > 0 and db[lo - 1] > -drop_db:
        lo -= 1
    hi = i
    while hi < len(a) - 1 and db[hi + 1] > -drop_db:
        hi += 1
    return float(fgrid[hi] - fgrid[lo])


def background(br, zeta, f_at, idx_pair):
    """Receptance at f_at of every mode outside the pair (residual flexibility)."""
    others = [k for k in range(len(br.w)) if k not in idx_pair]
    return complex(frf_subset(br, zeta, np.array([f_at]), others)[0])


def stay_fracs(cd):
    """Stay kinetic-energy fraction of the two in-band modes, by frequency."""
    f, Phi = cd.modes(60)
    frac = cd.energy_split(Phi)
    idx = np.where((f > FBAND[0]) & (f < FBAND[1]))[0]
    return f[idx], frac[idx]


def bisect_T(fun, lo, hi, it=30):
    """Tension where fun changes sign, fun(lo) < 0 < fun(hi)."""
    flo, fhi = fun(lo), fun(hi)
    if flo * fhi > 0:
        return np.nan
    if flo > 0:
        lo, hi = hi, lo
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if fun(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def crossovers(fgrid, zetas):
    """Crossover tensions of the veering pair.

    T_gap    closest approach of the two frequency loci (the veering center)
    T_energy where the stay-dominated branch switches from lower to upper
    T_res    where the two peaks have equal height at the sensor
    T_pick   where the picked peak jumps from the lower to the upper branch
    """
    def gap(T):
        b = DrivenBridge(float(T))
        return b.f_hi - b.f_lo

    Ts = np.linspace(145e3, 158e3, 53)
    g = np.array([gap(t) for t in Ts])
    i = int(np.argmin(g))
    lo, hi = Ts[max(i - 1, 0)], Ts[min(i + 1, len(Ts) - 1)]
    r = minimize_scalar(gap, bracket=(lo, Ts[i], hi), method="brent")
    T_gap = float(r.x)

    def energy_sign(T):
        _, fr = stay_fracs(DrivenBridge(float(T)).cd)
        return fr[1] - fr[0]

    def res_sign(T):
        return np.log(residue_ratio(DrivenBridge(float(T)))[0])

    out = dict(T_gap=T_gap, gap_min=float(r.fun),
               T_energy=bisect_T(energy_sign, 145e3, 160e3),
               T_res=bisect_T(res_sign, 145e3, 160e3))
    for z in zetas:
        def pick_sign(T, z=z):
            b = DrivenBridge(float(T))
            f0 = 0.5 * (b.f_lo + b.f_hi)
            return argmax_f(lambda ff: b.frf(z, ff), fgrid) - f0
        out[f"T_pick_{100*z:g}"] = bisect_T(pick_sign, 145e3, 160e3, it=22)
    return out


def part_b(quick=False):
    print("\n" + "=" * 74)
    print("PART B  driven finite element bridge")
    print("=" * 74)

    b = BRIDGE
    L, m, EI = b["Lc"], b["mc"], b["EIc"]
    Ts = np.arange(140e3, 165e3 + 1.0, 1e3 if quick else 0.5e3)
    fgrid = np.linspace(FBAND[0], FBAND[1], NF)

    rows, checked = [], False
    for T in Ts:
        T = float(T)
        br = DrivenBridge(T)
        i_lo, i_hi = band_pair(br)
        f_lo, f_hi = br.f_lo, br.f_hi
        f0 = 0.5 * (f_lo + f_hi)
        s = (f_hi - f_lo) / f0
        rho, rho_amp = residue_ratio(br)
        f_iso = float(tensioned_beam_freq(1, L, T, EI, m))
        d_meas = 2.0 * (f_iso - f0) / f0            # f_iso = f0 (1 + d/2)

        ff, fr = stay_fracs(br.cd)
        stay_up = bool(fr[1] >= fr[0])
        f_branch = f_hi if stay_up else f_lo
        f_other = f_lo if stay_up else f_hi

        if not checked:
            print(f"  modal-sum vs direct-solve check at T = {T/1e3:.1f} kN, "
                  f"zeta = {100*ZETAS[0]:.1f} %: "
                  f"{br.verify_frf(ZETAS[0], fgrid):.1e} of band peak")
            checked = True

        def e_ts(f, T=T):                                    # noqa: E306
            Te = float(invert_string(np.array([f]), L, m, np.array([1]))[0])
            return 100.0 * (Te - T) / T

        def e_cp(f, f_iso=f_iso):                            # noqa: E306
            return 100.0 * ((f / f_iso) ** 2 - 1.0)

        for z in ZETAS:
            Hfull = br.frf(z, fgrid)
            f_pick = argmax_f(lambda ff: br.frf(z, ff), fgrid)
            f_pair = argmax_f(lambda ff: frf_subset(br, z, ff,
                                                    [i_lo, i_hi]), fgrid)
            # accelerance peak, as picked in an ambient survey
            f_acc = argmax_f(lambda ff: ff ** 2 * br.frf(z, ff), fgrid)
            cen = peak_census(fgrid, Hfull)
            resolved = cen["n_peaks_3db"] >= 2

            # the residual background, and the first-order peak shift it
            # produces: dx = -Re(B)/A with A the resonant amplitude of the
            # mode being picked
            jn = i_hi if abs(f_pair - f_hi) < abs(f_pair - f_lo) else i_lo
            A_res = br.phip[jn] ** 2 / (2.0 * z * br.w[jn] ** 2)
            B = background(br, z, f_pair, (i_lo, i_hi))
            x_bg_pred = -B.real / A_res
            w01 = flat_width(fgrid, np.abs(Hfull), 0.1)
            w05 = flat_width(fgrid, np.abs(Hfull), 0.5)

            u = s / (2.0 * z)
            x_law = xstar(u, rho)
            f_law = f0 * (1.0 + z * x_law)
            rows.append(dict(
                T_true=T, zeta=z, f_pick=f_pick,
                eps_merged_pct=e_ts(f_pick),
                eps_branch_pct=e_ts(f_branch),
                rho=rho, resolved_bool=bool(resolved),
                eps_wrong_pct=e_ts(f_other),
                epsc_merged_pct=e_cp(f_pick),
                epsc_branch_pct=e_cp(f_branch),
                epsc_wrong_pct=e_cp(f_other),
                f_pair=f_pair, epsc_pair_pct=e_cp(f_pair),
                f_acc=f_acc, eps_acc_pct=e_ts(f_acc),
                bg_ratio=abs(B.real) / A_res, x_bg_pred=x_bg_pred,
                flat01_pct=200.0 * w01 / f0, flat05_pct=200.0 * w05 / f0,
                f_law=f_law, epsc_law_pct=e_cp(f_law),
                rho_amp=rho_amp, rho_2dof=rho_from_split(d_meas, s),
                epsc_mean_pct=100.0 * (s * (rho - 1.0) / (rho + 1.0)
                                       - d_meas),
                f_lo=f_lo, f_hi=f_hi, f0=f0, f_iso=f_iso,
                s_pct=100 * s, u=u, d_meas=d_meas,
                x_frf=(f_pick - f0) / (z * f0),
                x_pair=(f_pair - f0) / (z * f0),
                x_law=x_law,
                stay_upper=stay_up, pick_upper=bool(f_pick > f0),
                n_peaks_3db=int(cen["n_peaks_3db"]),
                dip_db=float(cen["dip_db"])))
        print(f"  T = {T/1e3:7.2f} kN  pair {f_lo:.4f}/{f_hi:.4f} Hz  "
              f"s = {100*s:5.3f} %  rho = {rho:8.4f}  "
              f"(2-DOF {rho_from_split(d_meas, s):8.4f})")

    return pd.DataFrame(rows), fgrid


def report(d, fgrid):
    print("\n" + "=" * 74)
    print("REPORT")
    print("=" * 74)
    g = d.drop_duplicates("T_true")

    print("\n1. residue ratio of the model at the sensor")
    print(f"  rho spans {g.rho.min():.4f} .. {g.rho.max():.4f} over "
          f"{g.T_true.min()/1e3:.0f}-{g.T_true.max()/1e3:.0f} kN")
    rel = np.abs(np.log(g.rho / g.rho_2dof))
    print(f"  vs the 2-DOF law (Delta+d)/(Delta-d): worst |log ratio| "
          f"{rel.max():.3f} (factor {np.exp(rel.max()):.2f}) at "
          f"T = {g.T_true.values[int(np.argmax(rel.values))]/1e3:.1f} kN, "
          f"median {np.median(rel):.3f}")
    print("  the sensor is 2 m above the anchorage, where the tie drives "
          "the stay\n  directly, so the deck-rooted branch appears more "
          "strongly than its\n  stay energy fraction implies; the residue "
          "ratio differs from the energy ratio.")

    print("  a fully merged peak is unbiased only with the exact 2-DOF "
          "residues;\n  with the residues of the model, the u -> 0 limit "
          "of the merged error\n  is not zero:")
    print(f"    residue-weighted mean of the pair, as a tension error: "
          f"{g.epsc_mean_pct.min():+.3f} .. {g.epsc_mean_pct.max():+.3f} % "
          f"(branch {g.epsc_branch_pct.min():+.3f} .. "
          f"{g.epsc_branch_pct.max():+.3f} %)")

    print("\n2. the merged peak against the analytical law")
    print("   x_law  two Lorentzians only (Part A, at the model's own u, rho)")
    print("   x_pair exact two-mode receptance (no narrow-band linearization)")
    print("   x_frf  full receptance, all modes")
    for z in sorted(d.zeta.unique()):
        s = d[d.zeta == z]
        same = np.sign(s.x_law) == np.sign(s.x_frf)
        q = s[same]
        print(f"  zeta {100*z:4.1f} %:  same branch {int(same.sum())}/{len(s)}"
              f"   |x_pair-x_law| {np.abs(q.x_pair-q.x_law).max():.4f}"
              f"   |x_frf-x_pair| {np.abs(q.x_frf-q.x_pair).max():.4f}"
              f"   eps gap: law {np.abs(q.epsc_merged_pct-q.epsc_law_pct).max():.3f} %"
              f"   pair {np.abs(q.epsc_merged_pct-q.epsc_pair_pct).max():.3f} %")

    print("\n2b. departure of the full receptance from the two-Lorentzian "
          "model:\n    residual flexibility of the modes outside the "
          "band")
    for z in sorted(d.zeta.unique()):
        q = d[d.zeta == z]
        obs = (q.x_frf - q.x_pair).values
        pred = q.x_bg_pred.values
        same = np.sign(q.x_law) == np.sign(q.x_frf)
        r = np.corrcoef(obs[same.values], pred[same.values])[0, 1] \
            if same.sum() > 2 else np.nan
        print(f"  zeta {100*z:4.1f} %:  |Re B| / A_res = "
              f"{q.bg_ratio.min():.3f}..{q.bg_ratio.max():.3f}"
              f"   observed shift {obs[same.values].min():+.3f}.."
              f"{obs[same.values].max():+.3f}"
              f"   first-order prediction {pred[same.values].min():+.3f}.."
              f"{pred[same.values].max():+.3f}   corr {r:+.3f}")
    print("  the background is quasi-static, so it does not scale with zeta "
          "while the\n  resonance does: |B|/A grows linearly with zeta and "
          "the peak shift with zeta^2.")

    print("\n2c. how well determined the picked frequency is")
    for z in sorted(d.zeta.unique()):
        q = d[d.zeta == z]
        print(f"  zeta {100*z:4.1f} %:  plateau within 0.1 dB of the maximum "
              f"spans {q.flat01_pct.min():.2f}..{q.flat01_pct.max():.2f} % "
              f"in tension; within 0.5 dB "
              f"{q.flat05_pct.min():.2f}..{q.flat05_pct.max():.2f} %")
    print("  a plateau wider than the bias means the reported tension is "
          "set by noise\n  as much as by the crossing.")

    print("\n2d. receptance vs accelerance pick (ambient surveys pick from "
          "the accelerance)")
    for z in sorted(d.zeta.unique()):
        q = d[d.zeta == z]
        dif = (q.eps_acc_pct - q.eps_merged_pct).values
        print(f"  zeta {100*z:4.1f} %:  worst |difference| "
              f"{np.abs(dif).max():.3f} pp   worst accelerance error "
              f"{q.eps_acc_pct.values[int(np.argmax(np.abs(q.eps_acc_pct.values)))]:+.3f} %")

    print("\n3. bracketing.  hull = the interval spanned by the two branch "
          "errors and 0")
    tol = 1e-3
    for z in sorted(d.zeta.unique()):
        s = d[d.zeta == z]
        em = s.epsc_merged_pct.values
        eb, ew = s.epsc_branch_pct.values, s.epsc_wrong_pct.values
        lo = np.minimum(np.minimum(eb, ew), 0.0)
        hi = np.maximum(np.maximum(eb, ew), 0.0)
        out_hull = (em < lo - tol) | (em > hi + tol)
        excess = np.maximum(lo - em, em - hi)
        over_branch = np.abs(em) > np.abs(eb) + tol
        flip = em * eb < 0
        j = int(np.argmax(excess))
        print(f"  zeta {100*z:4.1f} %:  outside [0, branch] "
              f"{int((over_branch | flip).sum()):3d}/{len(s)}"
              f"   (sign flips {int(flip.sum())})"
              f"   outside the hull {int(out_hull.sum()):3d}/{len(s)}"
              f"   worst excess {excess[j]:+.3f} pp at "
              f"T = {s.T_true.values[j]/1e3:.1f} kN "
              f"(merged {em[j]:+.3f}, branch {eb[j]:+.3f}, "
              f"wrong {ew[j]:+.3f})")

    print("\n4. worst merged-peak error over the grid "
          "(taut-string order 1, vs true T)")
    for z in sorted(d.zeta.unique()):
        s = d[d.zeta == z]
        j = int(np.argmax(np.abs(s.eps_merged_pct.values)))
        jb = int(np.argmax(np.abs(s.eps_branch_pct.values)))
        jw = int(np.argmax(np.abs(s.eps_wrong_pct.values)))
        print(f"  zeta {100*z:4.1f} %:  merged "
              f"{s.eps_merged_pct.values[j]:+7.3f} % at "
              f"{s.T_true.values[j]/1e3:6.1f} kN"
              f"  | branch {s.eps_branch_pct.values[jb]:+7.3f} % at "
              f"{s.T_true.values[jb]/1e3:6.1f} kN"
              f"  | wrong {s.eps_wrong_pct.values[jw]:+8.3f} % at "
              f"{s.T_true.values[jw]/1e3:6.1f} kN"
              f"  | rms merged {np.sqrt((s.eps_merged_pct**2).mean()):.3f} %"
              f"  resolved {int(s.resolved_bool.sum())}/{len(s)}")

    print("\n5. dependence on zeta at fixed tension (merged, percent)")
    zs = sorted(d.zeta.unique())
    print("      T kN   branch    wrong  |  " +
          "  ".join(f"z={100*z:.1f}%" for z in zs) + "   resolved")
    for T in sorted(g.T_true.values)[::4]:
        s = d[d.T_true == T].set_index("zeta")
        r0 = s.iloc[0]
        res = "".join("R" if s.loc[z].resolved_bool else "m" for z in zs)
        print(f"    {T/1e3:6.1f}  {r0.eps_branch_pct:+7.3f} "
              f"{r0.eps_wrong_pct:+8.3f}  |  " +
              "  ".join(f"{s.loc[z].eps_merged_pct:+7.3f}" for z in zs) +
              f"     {res}")

    print("\n6. crossover tensions (the branch read by the analyst switches "
          "after\n   the frequency loci cross)")
    cx = crossovers(fgrid, sorted(d.zeta.unique()))
    print(f"  closest approach of the loci   T_gap    = "
          f"{cx['T_gap']/1e3:7.3f} kN  (min gap {cx['gap_min']:.5f} Hz)")
    print(f"  stay energy switches branch    T_energy = "
          f"{cx['T_energy']/1e3:7.3f} kN")
    print(f"  peak heights equal at sensor   T_res    = "
          f"{cx['T_res']/1e3:7.3f} kN")
    for k, v in cx.items():
        if k.startswith("T_pick"):
            print(f"  picked peak jumps, zeta {k.split('_')[-1]:>4} %  = "
                  f"{v/1e3:7.3f} kN")
    w = cx["T_res"] - cx["T_energy"]
    print(f"  over {w/1e3:.2f} kN ({100*w/cx['T_gap']:.2f} % of the "
          f"crossing tension) the taller peak\n  is the deck-rooted "
          f"branch, not the stay-dominated branch")

    print("\n7. cross-checks")
    p = os.path.join(DATA, "twosided.csv")
    if os.path.exists(p):
        t = pd.read_csv(p)
        dom_up = t.stayfrac_upper.values >= t.stayfrac_lower.values
        t = t.assign(err_dom=np.where(dom_up, t.err_upper_pct,
                                      t.err_lower_pct),
                     err_oth=np.where(dom_up, t.err_lower_pct,
                                      t.err_upper_pct))
        j = g[["T_true", "eps_branch_pct", "eps_wrong_pct"]].merge(
            t[["T_true", "err_dom", "err_oth"]], on="T_true")
        print(f"  vs data/twosided.csv on {len(j)} shared tensions: "
              f"branch worst |diff| "
              f"{np.abs(j.eps_branch_pct - j.err_dom).max():.2e} %, "
              f"wrong {np.abs(j.eps_wrong_pct - j.err_oth).max():.2e} %")
    dc = pd.read_csv(os.path.join(DATA, "damping.csv"))
    bs = dc[dc.record == "bridge_summary"]
    if len(bs):
        sb = float(bs.s_bridge.iloc[0])
        near = g.iloc[int(np.argmin(np.abs(g.T_true.values - T_TUNE)))]
        print(f"  vs data/damping.csv at the crossing: s = {100*sb:.3f} % "
              f"there, {near.s_pct:.3f} % here at "
              f"T = {near.T_true/1e3:.1f} kN")
    return cx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    law_rows = part_a(quick=args.quick)
    d, fgrid = part_b(quick=args.quick)
    report(d, fgrid)

    cols = ["T_true", "zeta", "f_pick", "eps_merged_pct", "eps_branch_pct",
            "rho", "resolved_bool"]
    d = d[cols + [c for c in d.columns if c not in cols]]
    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, "merged.csv")
    d.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(d)} rows)")

    print("\nanalytical worst-case table:\n",
          pd.DataFrame(law_rows).to_string(index=False))


if __name__ == "__main__":
    main()
