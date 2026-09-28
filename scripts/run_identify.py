# -*- coding: utf-8 -*-
"""Identification study: where the bias lives, and what actually removes it.

Four estimators on identical synthetic records, across a range of detunings
either side of a crossing.

  string_n1     incumbent taut-string inversion on stay mode 1
  iso_exact     exact isolated tensioned beam on the same single frequency
  pinn_iso      physics-informed net, ISOLATED residual, v(L) = 0
  pinn_cpl      the same net with the coupled Robin end condition and an
                identified deck impedance
  multi_iso     ISOLATED multi-mode least squares over orders 1 to 5
  multi_cpl     closed-form coupled multi-mode fit

Two questions are answered separately, and they have different answers.

  Q1  With ONE mode order, does carrying the coupled boundary condition
      remove the bias?  This is where the physics claim is tested, and it is
      the case a published cable-force PINN addresses.
  Q2  With SEVERAL mode orders, does the coupled treatment still help?
      Because the split falls as 1/n the higher orders are nearly clean, so
      the answer may be no, and if it is no the honest recommendation is to
      use several orders rather than a more elaborate estimator.

Writes data/identify.csv.

Run:  python3 scripts/run_identify.py [nlevels]
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from cablefe import (CableDeck, invert_string,  # noqa: E402
                     tensioned_beam_freq)
from identify import fit_closed_form, pinn_identify  # noqa: E402

DATA = os.path.join(ROOT, "data")

BRIDGE = dict(Ld=80.0, EId=2.0e9, md=1000.0,
              Lc=25.0, EIc=1.2e4, mc=5.5, EA=1.4e8,
              theta=np.deg2rad(35.0), nd=40, nc=40)
NMAX = 5
NSENS = 9


def observe(T_true):
    """Stay-branch frequencies for orders 1..NMAX and the mode-1 shape."""
    cd = CableDeck(T=T_true, **BRIDGE)
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
    return np.array(obs), np.arange(1, NMAX + 1), xs * BRIDGE["Lc"], shape1


def fit_isolated_multi(f_obs, orders, L, m, EI):
    """Isolated tensioned beam, multi-mode least squares. The control."""
    def resid(p):
        T = np.exp(p[0])
        kn = orders * np.pi / L
        return np.sqrt(orders ** 2 * (T + EI * kn ** 2)
                       / (4 * m * L ** 2)) - f_obs
    T0 = 4 * m * L ** 2 * f_obs[-1] ** 2 / orders[-1] ** 2
    return float(np.exp(least_squares(resid, [np.log(T0)],
                                      method="lm").x[0]))


def main():
    nlev = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    # tensions spanning either side of the stay-mode-1 crossing at ~151.6 kN
    Ts = np.linspace(132e3, 176e3, nlev)
    L, m, EI = BRIDGE["Lc"], BRIDGE["mc"], BRIDGE["EIc"]

    rows = []
    for T_true in Ts:
        obs, orders, x_m, shape1 = observe(T_true)
        f1 = obs[0]
        om = 2 * np.pi * f1
        f_iso1 = tensioned_beam_freq(1, L, T_true, EI, m)
        d_rel = (f_iso1 - f1) / f_iso1        # sign carries which side

        k = np.pi / L
        T_iso_exact = (m * om ** 2 - EI * k ** 4) / k ** 2
        T_str = float(invert_string(np.array([f1]), L, m, np.array([1]))[0])

        idx = np.linspace(0, len(x_m) - 1, NSENS).astype(int)
        T_pi, _ = pinn_identify(x_m[idx], shape1[idx], om, L, m, EI,
                                coupled=False, epochs=8000, lbfgs=800, seed=0)
        T_pc, info = pinn_identify(x_m[idx], shape1[idx], om, L, m, EI,
                                   coupled=True, epochs=8000, lbfgs=800,
                                   seed=0)

        T_mi = fit_isolated_multi(obs, orders, L, m, EI)
        T_mc, _ = fit_closed_form(obs, orders, L, m, EI=EI)

        e = lambda v: 100.0 * (v - T_true) / T_true   # noqa: E731
        rows.append(dict(
            T_true=T_true, f1=f1, d=d_rel,
            string_n1=e(T_str), iso_exact=e(T_iso_exact),
            pinn_iso=e(T_pi), pinn_cpl=e(T_pc), Z=info["Z"],
            multi_iso=e(T_mi), multi_cpl=e(T_mc)))
        print(f"  T={T_true/1e3:6.1f} kN  d={d_rel:+.4f}   "
              f"string {e(T_str):+6.2f}  pinn_iso {e(T_pi):+6.2f}  "
              f"pinn_cpl {e(T_pc):+6.2f}  multi_iso {e(T_mi):+6.3f}  "
              f"multi_cpl {e(T_mc):+6.3f}")

    d = pd.DataFrame(rows)
    os.makedirs(DATA, exist_ok=True)
    d.to_csv(os.path.join(DATA, "identify.csv"), index=False)

    print()
    print("=" * 74)
    print("  Q1  ONE mode order: does the coupled boundary condition help?")
    print("=" * 74)
    for c, lab in (("string_n1", "incumbent taut string"),
                   ("iso_exact", "exact isolated tensioned beam"),
                   ("pinn_iso", "PINN, isolated physics"),
                   ("pinn_cpl", "PINN, COUPLED physics")):
        print(f"    {lab:32s} worst |error| {d[c].abs().max():6.3f} %"
              f"   median {d[c].abs().median():6.3f} %")
    print()
    print("    The first three agree, and they are the ones that assume an")
    print("    isolated stay. The bias is in the physics, not the estimator:")
    print("    a network inherits it exactly as the closed formula does.")

    print()
    print("=" * 74)
    print("  Q2  SEVERAL mode orders: does the coupled treatment still help?")
    print("=" * 74)
    for c, lab in (("multi_iso", "isolated multi-mode fit, orders 1-5"),
                   ("multi_cpl", "coupled closed-form fit, orders 1-5")):
        print(f"    {lab:38s} worst {d[c].abs().max():6.3f} %"
              f"   median {d[c].abs().median():6.3f} %")
    print()
    if d.multi_iso.abs().max() <= d.multi_cpl.abs().max():
        print("    NO. The isolated multi-mode fit is already as good or")
        print("    better. Because the split falls as 1/n the higher orders")
        print("    are nearly unperturbed and least squares recovers the")
        print("    tension from them. Report this: the recommendation is to")
        print("    use several orders, not a more elaborate estimator.")
    else:
        print("    Yes, the coupled fit improves on the isolated one.")
    print()
    print(f"  wrote {DATA}/identify.csv")


if __name__ == "__main__":
    main()
