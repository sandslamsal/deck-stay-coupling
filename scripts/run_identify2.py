# -*- coding: utf-8 -*-
"""Identification, second campaign: the repaired estimators, with noise.

The first identification study (run_identify.py) found the coupled PINN
worse than the incumbent and explained it as a joint-identifiability
failure.  The explanation was wrong.  The interior of the stay satisfies the
tensioned-beam equation with the true tension whatever the deck does, so a
measured mode SHAPE determines the wavenumber, the dispersion relation then
determines the tension, and the end impedance falls out as a by-product.
What failed was the soft-penalty Robin parameterisation, whose exponential
impedance could not even represent the negative sign Z takes just above a
deck frequency.

This campaign tests the repaired estimators the way the first one tested
the broken ones, plus measurement noise, which the noise-free comparison of
run_identify.py ignored and which is where a shape-based method could
plausibly lose to a frequency-based one.

Estimators:
    string_n1    incumbent taut string, stay mode 1 frequency
    multi_iso    isolated multi-mode least squares, orders 1..5 frequencies
    shapefit     tensioned-beam general solution fitted to the mode-1 shape,
                 nothing assumed at the anchorage (src/shapefit.py)
    pinn_free    the repaired network, same premise (only in the noise-free
                 and one noisy condition; it is 3000x slower than shapefit
                 and exists to show the machinery is immaterial)

Noise model, applied consistently:
    frequencies  f -> f (1 + sigma_f xi),  sigma_f = 0.002  (0.2 %)
    shape        v -> v + sigma_s max|v| xi  per sensor
    every estimator sees the same noisy frequency; shapefit and pinn_free
    additionally see the noisy shape.  Mode mis-assignment is NOT simulated
    here; frequencies are branch-matched by MAC as in run_identify.py.

Slices 0..8: the nine tensions of run_identify.py, either side of the stay
mode 1 crossing.  Slice 9: a heavier stay (m_c = 20 kg/m at the same
stress), effective mass ratio 3.6x larger, at its exact crossing tension.

Writes data/identify2_slice_<i>.csv (long format).

Run:  python3 scripts/run_identify2.py --slice I [--smoke]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from cablefe import CableDeck, invert_string, tensioned_beam_freq  # noqa: E402
from identify import pinn_identify_free  # noqa: E402
from shapefit import fit_shape  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

DATA = os.path.join(ROOT, "data")

BRIDGE = dict(Ld=80.0, EId=2.0e9, md=1000.0,
              Lc=25.0, EIc=1.2e4, mc=5.5, EA=1.4e8,
              theta=np.deg2rad(35.0), nd=40, nc=40)
TENSIONS = np.linspace(132e3, 176e3, 9)

# heavier stay at the same 27.6 kN/cm^2 stress: area x3.64, EA linear in A,
# EI quadratic in A
STRESS_SCALE = 20.0 / 5.5
BRIDGE_S = dict(BRIDGE, mc=20.0, EA=BRIDGE["EA"] * STRESS_SCALE,
                EIc=BRIDGE["EIc"] * STRESS_SCALE ** 2)

NMAX = 5
NSENS = 9
SIGMA_F = 0.002
SIGMA_S_LEVELS = (0.005, 0.01, 0.02)
NREP = 100
NREP_PINN = 3
PINN_SIGMA = 0.01


def observe(T_true, bridge):
    """Branch-matched stay frequencies for orders 1..NMAX + mode-1 shape."""
    cd = CableDeck(T=T_true, **bridge)
    f, Phi = cd.modes(60)
    cdofs = np.array(cd.cable_dofs())
    xs = np.linspace(0.0, 1.0, len(cdofs))
    obs, shape1 = [], None
    for n in range(1, NMAX + 1):
        tgt = np.sin(n * np.pi * xs)
        tt = float(tgt @ tgt)
        best, bm = 0, -1.0
        for j in range(Phi.shape[1]):
            a = Phi[cdofs, j]
            den = float(a @ a) * tt
            mm = (float(a @ tgt) ** 2 / den) if den > 0 else 0.0
            if mm > bm:
                bm, best = mm, j
        obs.append(f[best])
        if n == 1:
            shape1 = Phi[cdofs, best]
    return np.array(obs), xs * bridge["Lc"], shape1


def fit_isolated_multi(f_obs, orders, L, m, EI):
    """Isolated tensioned beam, multi-mode least squares (the strong
    incumbent from run_identify.py)."""
    def resid(p):
        T = np.exp(p[0])
        kn = orders * np.pi / L
        return np.sqrt(orders ** 2 * (T + EI * kn ** 2)
                       / (4 * m * L ** 2)) - f_obs
    T0 = 4 * m * L ** 2 * f_obs[-1] ** 2 / orders[-1] ** 2
    return float(np.exp(least_squares(resid, [np.log(T0)],
                                      method="lm").x[0]))


def stress_case_tension():
    """Tension putting the heavy stay's mode 1 on the nearest deck mode."""
    cd = CableDeck(T=500e3, **BRIDGE_S)     # k_ax does not depend on T
    fd = cd.deck_alone(10)
    # target the same deck mode the light stay crossed (~3.32 Hz)
    fdk = fd[np.argmin(np.abs(fd - 3.32))]
    k = np.pi / BRIDGE_S["Lc"]
    m, EI = BRIDGE_S["mc"], BRIDGE_S["EIc"]
    om = 2 * np.pi * fdk
    return (m * om ** 2 - EI * k ** 4) / k ** 2


def run_slice(i, smoke=False):
    if i < 9:
        bridge, T_true, case = BRIDGE, float(TENSIONS[i]), f"T{i}"
    else:
        bridge, T_true, case = BRIDGE_S, float(stress_case_tension()), "heavy"

    L, m, EI = bridge["Lc"], bridge["mc"], bridge["EIc"]
    orders = np.arange(1, NMAX + 1)
    obs, x_m, shape1 = observe(T_true, bridge)
    f1 = obs[0]
    idx = np.linspace(0, len(x_m) - 1, NSENS).astype(int)
    xs, vs = x_m[idx], shape1[idx]

    e = lambda v: 100.0 * (v - T_true) / T_true    # noqa: E731
    rows = []

    def add(est, sig_s, rep, err, **kw):
        rows.append(dict(slice=i, case=case, T_true=T_true,
                         estimator=est, sigma_s=sig_s, rep=rep,
                         err_pct=err, **kw))

    # ---- noise-free ------------------------------------------------------
    add("string_n1", 0.0, -1,
        e(float(invert_string(np.array([f1]), L, m, np.array([1]))[0])))
    add("multi_iso", 0.0, -1, e(fit_isolated_multi(obs, orders, L, m, EI)))
    T_sf, info = fit_shape(xs, vs, 2 * np.pi * f1, L, m, EI)
    add("shapefit", 0.0, -1, e(T_sf), Z=info["Z"])
    nseeds = 1 if smoke else 3
    for sd in range(nseeds):
        T_pf, ip = pinn_identify_free(xs, vs, 2 * np.pi * f1, L, m, EI,
                                      epochs=6000, lbfgs=600, seed=sd)
        add("pinn_free", 0.0, -(sd + 1), e(T_pf), Z=ip["Z"])

    # ---- noisy -----------------------------------------------------------
    nrep = 10 if smoke else NREP
    for sig_s in SIGMA_S_LEVELS:
        for rep in range(nrep):
            rng = np.random.default_rng(100000 * i + 1000 * rep
                                        + int(sig_s * 1e4))
            fn = obs * (1.0 + SIGMA_F * rng.standard_normal(len(obs)))
            vn = vs + sig_s * np.max(np.abs(vs)) * \
                rng.standard_normal(len(vs))
            add("string_n1", sig_s, rep,
                e(float(invert_string(np.array([fn[0]]), L, m,
                                      np.array([1]))[0])))
            add("multi_iso", sig_s, rep,
                e(fit_isolated_multi(fn, orders, L, m, EI)))
            T_sf, _ = fit_shape(xs, vn, 2 * np.pi * fn[0], L, m, EI)
            add("shapefit", sig_s, rep, e(T_sf))

    npr = 1 if smoke else NREP_PINN
    for rep in range(npr):
        rng = np.random.default_rng(900000 * (i + 1) + rep)
        fn1 = f1 * (1.0 + SIGMA_F * rng.standard_normal())
        vn = vs + PINN_SIGMA * np.max(np.abs(vs)) * \
            rng.standard_normal(len(vs))
        T_pf, _ = pinn_identify_free(xs, vn, 2 * np.pi * fn1, L, m, EI,
                                     epochs=6000, lbfgs=600, seed=rep)
        add("pinn_free", PINN_SIGMA, rep, e(T_pf))

    d = pd.DataFrame(rows)
    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, f"identify2_slice_{i}.csv")
    d.to_csv(out, index=False)

    nf = d[(d.rep < 0)]
    print(f"slice {i} ({case}, T={T_true/1e3:.1f} kN) -> {out}")
    for _, r in nf.iterrows():
        print(f"    noise-free {r.estimator:10s} {r.err_pct:+8.4f} %")
    for sig_s in SIGMA_S_LEVELS:
        sub = d[(d.sigma_s == sig_s) & (d.rep >= 0)]
        if not len(sub):
            continue
        line = "  ".join(
            f"{est}:{np.sqrt((g.err_pct**2).mean()):.3f}"
            for est, g in sub.groupby("estimator"))
        print(f"    rmse@{sig_s:.3f}  {line}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run_slice(args.slice, smoke=args.smoke)
