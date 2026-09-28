# -*- coding: utf-8 -*-
"""Main campaign over the (detuning, effective mass ratio) plane.

The pilot established the mechanism and produced a closed form for the
veering width at exact tuning.  This campaign asks the design question: over
the whole plane, and over the property ranges of real cable-stayed bridges
from footbridges to long spans, how large is the tension error of the
isolated-cable inversion, and where is it tolerable.

THE PREDICTED LAW.  Two modes approaching each other behave as a classical
avoided crossing.  With the uncoupled stay frequency ``f_s``, the deck
frequency it approaches ``f_d``, relative detuning ``d = (f_s - f_d)/f_s``
and split at exact tuning ``s = (2/pi) cos(theta) sqrt(mu_eff)``, the
coupled pair is

    f_pm = (f_s + f_d)/2  pm  (1/2) sqrt(D^2 + S^2)

so the stay-dominated branch is displaced from ``f_s`` by
``(1/2)[sqrt(D^2+S^2) - |D|]``.  The incumbent formula takes tension as
proportional to the square of frequency, so to leading order the TENSION
ERROR is

    eps = sqrt(d^2 + s^2) - |d|                                        (*)

which gives ``eps = s`` at exact tuning and decays as ``s^2/(2|d|)`` far from
it.  Inverting (*) for a tolerance ``tol`` gives the screening criterion

    |d| >= (s^2 - tol^2) / (2 tol)                                    (**)

Everything above is a prediction.  The campaign's job is to test it against
the finite element model over a wide sample and report where it fails.

Sampling constructs crossings rather than waiting for them: for each design
a deck mode with real motion at the anchorage is chosen, a stay mode order
and a target detuning are drawn, and the tension that realises that detuning
is solved for.  Anchorages are never placed at a node of the targeted mode,
which is the trap the pilot fell into.

Writes data/campaign.csv.

Run:  python3 scripts/run_campaign.py [nsamples] [nworkers]
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from cablefe import (CableDeck, chain, invert_string,  # noqa: E402
                     mu_effective, tensioned_beam_freq, veering_split,
                     xi_param, irvine_lambda2)

DATA = os.path.join(ROOT, "data")

E_STEEL = 195e9
RHO_STEEL = 7850.0
ND = NC = 40


def deck_alone_modes(Ld, EId, md, nd, ia, k_ax, nmodes=12):
    """Deck with the stay's axial spring, mass-normalised, amplitude at anchor."""
    Kd, Md = chain(Ld, nd, EId, md, 0.0)
    Kd = Kd.copy()
    Kd[2 * ia, 2 * ia] += k_ax
    keep = [i for i in range(Kd.shape[0]) if i not in (0, 2 * nd)]
    w2, V = eigh(Kd[np.ix_(keep, keep)], Md[np.ix_(keep, keep)])
    f = np.sqrt(np.maximum(w2, 0.0)) / (2.0 * np.pi)
    pos = {d: j for j, d in enumerate(keep)}
    phi_a = V[pos[2 * ia], :]
    return f[:nmodes], phi_a[:nmodes]


def one_case(seed):
    """One design.

    Sampling note.  With ``m_c = rho A`` and ``T = sigma A`` the stay's
    frequency is ``f_n = (n/2L) sqrt(sigma/rho)``, independent of area: it is
    set by STRESS, not tension.  So the tension cannot be solved for to hit a
    target detuning at fixed stress, and an earlier version of this function
    tried to and diverged.  The stress is solved for instead, and the area is
    then a free variable that sets ``mu_eff`` without moving the frequency,
    which is exactly the independent driver the campaign needs.

    Geometry is made consistent by construction: the anchorage position
    follows from the stay length and inclination rather than being drawn and
    then rejected.
    """
    rng = np.random.default_rng(seed)

    # ---- design draw, spanning footbridge to long span ------------------
    Ld = float(np.exp(rng.uniform(np.log(40.0), np.log(600.0))))
    md = float(np.exp(rng.uniform(np.log(800.0), np.log(35000.0))))
    fd1 = float(np.exp(rng.uniform(np.log(0.20), np.log(3.5))))   # deck f1
    EId = md * (fd1 * 2 * np.pi * Ld ** 2 / np.pi ** 2) ** 2

    Lc = float(np.exp(rng.uniform(np.log(12.0), np.log(280.0))))
    theta = float(rng.uniform(np.deg2rad(20.0), np.deg2rad(60.0)))
    A = float(np.exp(rng.uniform(np.log(5e-4), np.log(2.0e-2))))
    n_stay = int(rng.integers(1, 6))
    d_target = float(rng.uniform(-0.15, 0.15))

    # anchorage follows from the stay geometry, pylon at the left support
    xfrac = Lc * np.cos(theta) / Ld
    if not (0.10 <= xfrac <= 0.48):
        return None
    ia = int(round(xfrac * ND))
    if ia < 2 or ia > ND - 2:
        return None
    xfrac = ia / ND

    mc = RHO_STEEL * A
    EA = E_STEEL * A
    k_ax = EA / Lc * np.sin(theta) ** 2

    fd, phia = deck_alone_modes(Ld, EId, md, ND, ia, k_ax)

    # only modes with real motion at the anchorage can couple at all
    ok = np.abs(phia) > 0.15 * np.abs(phia).max()
    if not ok.any():
        return None
    j = int(rng.choice(np.where(ok)[0]))
    f_deck, phi_a = float(fd[j]), float(phia[j])

    # stress that puts stay mode n at the target detuning from that mode
    f_target = f_deck / (1.0 - d_target)
    sigma = RHO_STEEL * (2.0 * Lc * f_target / n_stay) ** 2
    if not (120e6 <= sigma <= 800e6):
        return None

    T = sigma * A
    d_eq = np.sqrt(4.0 * A / np.pi)
    EIc = 0.2 * E_STEEL * np.pi * d_eq ** 4 / 64.0

    # ---- coupled solution ----------------------------------------------
    try:
        cd = CableDeck(Ld=Ld, EId=EId, md=md, Lc=Lc, EIc=EIc, mc=mc, T=T,
                       EA=EA, theta=theta, x_anchor=xfrac * Ld,
                       nd=ND, nc=NC)
        f, Phi = cd.modes(70)
    except Exception:
        return None

    f_s = tensioned_beam_freq(n_stay, Lc, T, EIc, mc)

    # the stay-dominated branch, matched by MAC against the isolated shape
    cdofs = np.array(cd.cable_dofs())
    x = np.linspace(0.0, 1.0, len(cdofs))
    target = np.sin(n_stay * np.pi * x)
    best, bestm = None, -1.0
    tt = float(target @ target)
    for jj in range(Phi.shape[1]):
        a = Phi[cdofs, jj]
        den = float(a @ a) * tt
        m = (float(a @ target) ** 2 / den) if den > 0 else 0.0
        if m > bestm:
            bestm, best = m, jj
    f_coupled = float(f[best])

    # ---- what the incumbent inversion returns ---------------------------
    T_hat = float(invert_string(np.array([f_coupled]), Lc, mc,
                                np.array([n_stay]))[0])
    eps_measured = (T_hat - T) / T
    T_ctrl = float(invert_string(np.array([f_s]), Lc, mc,
                                 np.array([n_stay]))[0])
    eps_control = (T_ctrl - T) / T

    # ---- the predicted law ----------------------------------------------
    M_stay = mc * Lc / 2.0
    mu_eff = mu_effective(M_stay, phi_a)
    s = veering_split(mu_eff, theta, n_stay)
    d_actual = (f_s - f_deck) / f_s
    eps_pred = np.sqrt(d_actual ** 2 + s ** 2) - abs(d_actual)
    eps_coupling = eps_measured - eps_control   # the EI bias removed

    return dict(
        Ld=Ld, md=md, fd1=fd1, EId=EId, Lc=Lc, theta=theta, sigma=sigma,
        A=A, xfrac=xfrac, n_stay=n_stay, T=T, mc=mc, EA=EA, EIc=EIc,
        f_deck=f_deck, phi_a=phi_a, f_s=f_s, f_coupled=f_coupled,
        mac=bestm, M_stay=M_stay, mu_eff=mu_eff, s=s,
        d=d_actual, eps=eps_measured, eps_pred=eps_pred,
        eps_coupling=eps_coupling,
        eps_control=eps_control,
        xi=xi_param(Lc, T, EIc),
        lam2=irvine_lambda2(Lc, T, EA, mc, theta))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    print(f"  drawing {n} designs on {w} workers ...")
    out = Parallel(n_jobs=w, verbose=1)(
        delayed(one_case)(s) for s in range(n))
    rows = [r for r in out if r is not None]
    d = pd.DataFrame(rows)
    os.makedirs(DATA, exist_ok=True)
    d.to_csv(os.path.join(DATA, "campaign.csv"), index=False)
    print(f"  {len(d)} of {n} designs usable, written to "
          f"{DATA}/campaign.csv")

    # only trust cases where the stay branch was cleanly identified
    g = d[(d.mac > 0.5) & (d.xi > 150)].copy()
    print(f"  {len(g)} with a cleanly matched stay branch (MAC > 0.5) and\n  xi > 150, above which the bending-stiffness bias of the formula is\n  below 1.1 % and does not mask the coupling")

    print()
    print("=" * 74)
    print("  DOES THE PREDICTED LAW HOLD?")
    print("=" * 74)
    print("    eps = sqrt(d^2 + s^2) - |d|,   s = (2/pi) cos(theta) "
          "sqrt(mu_eff)")
    print()
    res = g.eps_coupling.abs() - g.eps_pred
    ss_res = float((res ** 2).sum())
    ss_tot = float(((g.eps_coupling.abs() - g.eps_coupling.abs().mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    print(f"    R^2 of the closed form against the model: {r2:.4f}")
    print(f"    median |residual|: {np.median(np.abs(res)) * 100:.3f} pp")
    print(f"    p90    |residual|: {np.percentile(np.abs(res), 90) * 100:.3f} pp")
    print(f"    control, no coupling, worst |eps|: "
          f"{g.eps_control.abs().max() * 100:.3f} %")

    print()
    print("=" * 74)
    print("  HOW BIG IS THE ERROR, AND WHERE")
    print("=" * 74)
    for lo, hi, lab in ((0.00, 0.02, "|d| <= 0.02  (near a crossing)"),
                        (0.02, 0.05, "0.02 < |d| <= 0.05"),
                        (0.05, 0.10, "0.05 < |d| <= 0.10"),
                        (0.10, 1.00, "|d| > 0.10")):
        sub = g[(g.d.abs() > lo) & (g.d.abs() <= hi)]
        if not len(sub):
            continue
        print(f"    {lab:34s} n={len(sub):4d}  median "
              f"{sub.eps.abs().median() * 100:6.2f} %  p90 "
              f"{sub.eps.abs().quantile(0.90) * 100:6.2f} %  worst "
              f"{sub.eps.abs().max() * 100:7.2f} %")

    print()
    print("=" * 74)
    print("  THE SCREENING CRITERION")
    print("=" * 74)
    print("    the isolated-cable inversion is within tol provided")
    print("        |d| >= (s^2 - tol^2) / (2 tol)")
    for tol in (0.02, 0.05, 0.10):
        pred_ok = g.d.abs() >= (g.s ** 2 - tol ** 2) / (2 * tol)
        actual_ok = g.eps.abs() <= tol
        tp = int((pred_ok & actual_ok).sum())
        fp = int((pred_ok & ~actual_ok).sum())
        fn = int((~pred_ok & actual_ok).sum())
        print(f"    tol {tol*100:4.0f} %: criterion passes {int(pred_ok.sum()):4d} "
              f"designs, of which {fp:3d} actually exceed tol "
              f"({100*fp/max(int(pred_ok.sum()),1):.2f} % unconservative); "
              f"{fn:4d} passed designs it needlessly rejects")
    print()


if __name__ == "__main__":
    main()
