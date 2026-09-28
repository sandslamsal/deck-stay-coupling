# -*- coding: utf-8 -*-
"""Revision 1, R1.5: the sag lift of the fundamental on the four
instrumented Ponte del Mare stays, recomputed.

Table 5 of the submitted manuscript carried a "sag lift" column computed
in scripts/validate_pontedelmare2.py as lambda^2 / (4 pi^2) with the
horizontal-projection lambda^2. Two things were wrong with that. The
parameter of an inclined cable referred to its chord is smaller by
cos^3(theta) (see cablefe.irvine_lambda2, verified in
scripts/verify_cablefe2d.py), and lambda^2 / (4 pi^2) is not the frequency
lift of the first symmetric mode: for small lambda^2 that lift is
4 lambda^2 / pi^4, and at the values the long stays carry it must be
taken from the root of Irvine's equation. Both are done here, over the
same inclination bands as before, with E = 165 GPa and rho = 8289 kg/m^3
from the thesis (Table 5.3), and the tension at the construction-control
pull.

Writes data/pontedelmare_sag.csv.

Run:  python3 scripts/run_pontedelmare_sag.py
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
from cablefe2d import irvine_symmetric_beta  # noqa: E402

DATA = os.path.join(ROOT, "data")
G = 9.80665
E_STAY, RHO_STAY = 165e9, 8289.0
# Values transcribed from Kumar (2011), PhD thesis, University of Trento (Ponte del Mare footbridge). They are not redistributed with
# this code: they live in data/external/run_pontedelmare_sag_data.py (see README).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "external"))
try:
    from run_pontedelmare_sag_data import (  # noqa: E402
        STAYS)
except ImportError as exc:
    raise SystemExit("scripts/run_pontedelmare_sag.py needs values transcribed from "
                     "Kumar (2011), PhD thesis, University of Trento (Ponte del Mare footbridge), which are not redistributed here. "
                     "See README, 'Third-party data'.") from exc
OFFSET_TENSION_PCT = 5.6      # the offset all four multi-order residuals share (Table 5)


def lam2_horizontal(L, T, m, th, EA):
    H = T * np.cos(th)
    Lh = L * np.cos(th)
    d = m * G * Lh ** 2 / (8 * H)
    Le = Lh * (1 + 8 * (d / Lh) ** 2)
    return (m * G * Lh / H) ** 2 * Lh / (H * Le / EA)


def main():
    rows = []
    print("stay   anomaly   old lift (lam2_h/4pi^2)   chord lam2      exact Irvine lift   residual")
    for k, s in STAYS.items():
        L, m, T = s["L"], s["m"], s["T_meas"] * 1e3
        EA = E_STAY * m / RHO_STAY
        anom = 100 * (s["f1"] / s["f_string"] - 1)
        old = [100 * lam2_horizontal(L, T, m, np.deg2rad(t), EA) / (4 * np.pi ** 2) for t in s["th"]]
        lam = [irvine_lambda2(L, T, EA, m, np.deg2rad(t)) for t in s["th"]]
        new = [100 * (irvine_symmetric_beta(l, 1) / np.pi - 1) for l in lam]
        net = anom - 0.5 * OFFSET_TENSION_PCT     # frequency share of the shared tension offset
        rows.append(dict(stay=k, anomaly_pct=anom, lift_old_lo=min(old), lift_old_hi=max(old),
                         lam2_chord_lo=min(lam), lam2_chord_hi=max(lam),
                         lift_exact_lo=min(new), lift_exact_hi=max(new),
                         anomaly_net_of_offset_pct=net,
                         residual_lo=net - max(new), residual_hi=net - min(new)))
        print(f"{k:5s} {anom:+7.1f} %   {min(old):5.1f}-{max(old):4.1f} %          "
              f"{min(lam):5.2f}-{max(lam):4.2f}     {min(new):5.1f}-{max(new):4.1f} %      "
              f"net {net:+5.1f} % -> {net - max(new):+5.1f}..{net - min(new):+5.1f} %")
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "pontedelmare_sag.csv"), index=False)
    print("small-lambda^2 check: 4 lambda^2 / pi^4 at lambda^2 = 0.1 gives %.2f %%, exact root %.2f %%"
          % (100 * 4 * 0.1 / np.pi ** 4, 100 * (irvine_symmetric_beta(0.1, 1) / np.pi - 1)))
    print("wrote", os.path.join(DATA, "pontedelmare_sag.csv"))


if __name__ == "__main__":
    main()
