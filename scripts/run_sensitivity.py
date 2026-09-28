# -*- coding: utf-8 -*-
"""Sensitivity of the shape-fit tension method, one factor at a time.

Runs src/shapefit.py on the example bridge of run_identify2.py at the mode-1
crossing tension T = 151.6 kN, about a baseline of 9 sensors, exact parameters
and no noise. Factors: sensor count, chord length (geometry scale error),
bending stiffness EI_c, mass per length m_c, and shape noise. Writes
data/sensitivity.csv in long format (factor, level, metric, value_pct).
Run:  python3 scripts/run_sensitivity.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from cablefe import CableDeck          # noqa: E402
from shapefit import fit_shape         # noqa: E402

DATA = os.path.join(ROOT, "data")

# the example bridge of run_identify2.py, at the mode-1 crossing tension
BRIDGE = dict(Ld=80.0, EId=2.0e9, md=1000.0,
              Lc=25.0, EIc=1.2e4, mc=5.5, EA=1.4e8,
              theta=np.deg2rad(35.0), nd=40, nc=40)
T_TRUE = 151.6e3

NSENS_BASE = 9
NSENS_LEVELS = (3, 5, 7, 9, 15)
CHORD_LEVELS = (-2.0, -1.0, +1.0, +2.0)      # percent
EIC_LEVELS = (-20.0, +20.0)                  # percent
MC_LEVELS = (-2.0, +2.0)                     # percent
SIGMA_S_LEVELS = (0.005, 0.01, 0.02)         # fraction of peak
NREP = 100


def observe_mode1(T_true, bridge):
    """Mode-1 stay-branch frequency (Hz), chord coordinates (m) and shape,
    matched by MAC to sin(pi x / L) as in run_identify2.observe()."""
    cd = CableDeck(T=T_true, **bridge)
    f, Phi = cd.modes(60)
    cdofs = np.array(cd.cable_dofs())
    xs = np.linspace(0.0, 1.0, len(cdofs))
    tgt = np.sin(np.pi * xs)
    tt = float(tgt @ tgt)
    best, bm = 0, -1.0
    for j in range(Phi.shape[1]):
        a = Phi[cdofs, j]
        den = float(a @ a) * tt
        mm = (float(a @ tgt) ** 2 / den) if den > 0 else 0.0
        if mm > bm:
            bm, best = mm, j
    return f[best], xs * bridge["Lc"], Phi[cdofs, best]


def sensor_idx(npoints, nsens):
    """The run_identify2.py sensor layout: equally spaced indices including
    both ends, the anchorage (last index) among them."""
    return np.linspace(0, npoints - 1, nsens).astype(int)


def try_fit(xs, vs, omega, L, m, EI):
    """fit_shape, returning NaN when the fit is underdetermined or fails."""
    ncols = 4 if np.isfinite(EI) and EI > 0 else 2
    if len(xs) <= ncols:
        return np.nan, "underdetermined (%d sensors, %d basis columns)" % (
            len(xs), ncols)
    try:
        T, info = fit_shape(xs, vs, omega, L, m, EI)
    except ValueError as exc:
        return np.nan, str(exc)
    return T, info


def main():
    L, m, EI = BRIDGE["Lc"], BRIDGE["mc"], BRIDGE["EIc"]
    f1, x_m, shape1 = observe_mode1(T_TRUE, BRIDGE)
    om = 2.0 * np.pi * f1
    print("mode-1 stay branch at T = %.1f kN:  f1 = %.4f Hz" % (
        T_TRUE / 1e3, f1))

    idx9 = sensor_idx(len(x_m), NSENS_BASE)
    xs9, vs9 = x_m[idx9], shape1[idx9]

    e = lambda T: 100.0 * (T - T_TRUE) / T_TRUE    # noqa: E731
    rows = []

    def add(factor, level, metric, value):
        rows.append(dict(factor=factor, level=level, metric=metric,
                         value_pct=value))

    # ---- baseline --------------------------------------------------------
    T0, _ = try_fit(xs9, vs9, om, L, m, EI)
    add("baseline", 0.0, "err_pct", e(T0))
    print("baseline (9 sensors, exact parameters): err = %+.4f %%" % e(T0))

    # ---- (a) sensor count ------------------------------------------------
    print("\n(a) sensor count")
    for ns in NSENS_LEVELS:
        ii = sensor_idx(len(x_m), ns)
        Tf, info = try_fit(x_m[ii], shape1[ii], om, L, m, EI)
        err = e(Tf) if np.isfinite(Tf) else np.nan
        add("nsens", float(ns), "err_pct", err)
        note = "" if np.isfinite(Tf) else "   [%s]" % info
        print("    n = %2d   err = %+10.4f %%%s" % (
            ns, err if np.isfinite(err) else float("nan"), note))

    # ---- (b) chord length error -----------------------------------------
    # geometry scale error: sensor coordinates and L both scaled by (1+e)
    print("\n(b) chord length error (geometry scale), expected err ~ 2e")
    for lv in CHORD_LEVELS:
        s = 1.0 + lv / 100.0
        Tf, _ = try_fit(xs9 * s, vs9, om, L * s, m, EI)
        exp = 100.0 * (s ** 2 - 1.0)
        add("chord_pct", lv, "err_pct", e(Tf))
        add("chord_pct", lv, "expected_err_pct", exp)
        print("    %+5.1f %%   err = %+8.4f %%   expected %+8.4f %%" % (
            lv, e(Tf), exp))

    # ---- (c) EI_c error --------------------------------------------------
    print("\n(c) EI_c error, expected err ~ -k^2 dEI / T")
    k2 = (np.pi / L) ** 2
    for lv in EIC_LEVELS:
        EIw = EI * (1.0 + lv / 100.0)
        Tf, _ = try_fit(xs9, vs9, om, L, m, EIw)
        exp = -100.0 * k2 * (EIw - EI) / T_TRUE
        add("EIc_pct", lv, "err_pct", e(Tf))
        add("EIc_pct", lv, "expected_err_pct", exp)
        print("    %+5.1f %%   err = %+8.4f %%   expected %+8.4f %%" % (
            lv, e(Tf), exp))

    # ---- (d) m_c error ---------------------------------------------------
    # T = (m omega^2 - EI k^4)/k^2 at the fitted k: linear in m, so the
    # expected line is 1:1 (times the small 1 + EI k^2/T bending correction)
    print("\n(d) m_c error, expected 1:1 (dispersion T ~ m)")
    for lv in MC_LEVELS:
        mw = m * (1.0 + lv / 100.0)
        Tf, _ = try_fit(xs9, vs9, om, L, mw, EI)
        exp = lv * (1.0 + EI * k2 / T_TRUE)
        add("mc_pct", lv, "err_pct", e(Tf))
        add("mc_pct", lv, "expected_err_pct", exp)
        print("    %+5.1f %%   err = %+8.4f %%   expected 1:1 -> %+8.4f %%"
              "   ratio err/level = %.4f" % (lv, e(Tf), exp, e(Tf) / lv))

    # ---- (e) shape noise -------------------------------------------------
    print("\n(e) shape noise at %d sensors, %d reps" % (NSENS_BASE, NREP))
    peak = np.max(np.abs(vs9))
    for sig in SIGMA_S_LEVELS:
        errs = []
        for rep in range(NREP):
            rng = np.random.default_rng(7000 + 1000 * rep + int(sig * 1e4))
            vn = vs9 + sig * peak * rng.standard_normal(len(vs9))
            Tf, _ = try_fit(xs9, vn, om, L, m, EI)
            if np.isfinite(Tf):
                errs.append(e(Tf))
        errs = np.array(errs)
        bias = float(errs.mean())
        rmse = float(np.sqrt((errs ** 2).mean()))
        add("sigma_s", sig, "bias_pct", bias)
        add("sigma_s", sig, "rmse_pct", rmse)
        print("    sigma_s = %.3f   bias = %+8.4f %%   rmse = %8.4f %%"
              "   (n = %d)" % (sig, bias, rmse, len(errs)))

    d = pd.DataFrame(rows)
    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, "sensitivity.csv")
    d.to_csv(out, index=False)
    print("\nwrote %s  (%d rows)" % (out, len(d)))


if __name__ == "__main__":
    main()
