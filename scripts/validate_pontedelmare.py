# -*- coding: utf-8 -*-
"""Screening criterion against the measured Ponte del Mare footbridge, Pescara.

Data: Kumar (2011), PhD thesis, University of Trento, Tables 4.4 and 5.6.
Computes the harmonicity f_n/n of each instrumented stay, the first-mode
anomaly against the tension from modes 2 to 5, and the detuning of every stay
mode against the nearest identified deck mode. Reads
data/external/validate_pontedelmare_data.py; writes data/pontedelmare.csv.

Run:  python3 scripts/validate_pontedelmare.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

DATA = os.path.join(ROOT, "data")

# --- measured data, transcribed from Kumar (2011) ---

# Table 4.4: instrumented stays, peak-picked transverse frequencies
# Values transcribed from Kumar (2011) are not redistributed with this code:
# they live in data/external/validate_pontedelmare_data.py (see README).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "external"))
try:
    from validate_pontedelmare_data import (  # noqa: E402
        STAYS)
except ImportError as exc:
    raise SystemExit("scripts/validate_pontedelmare.py needs values transcribed from "
                     "Kumar (2011), PhD thesis, University of Trento (Ponte del Mare footbridge), which are not redistributed here. "
                     "See README, 'Third-party data'.") from exc

# Table 5.6: identified global (deck) modal frequencies, EMA column
DECK = np.array([0.747, 1.065, 1.126, 1.243, 1.394, 1.510,
                 1.716, 1.791, 2.306, 2.364, 2.512, 2.862])

G = 9.80665
RHO = 7850.0
E_STEEL = 195e9


def harmonicity(f):
    """f_n/n, whose constancy is the signature of an isolated taut stay."""
    n = np.arange(1, len(f) + 1)
    return np.asarray(f) / n


def tension_from_high_modes(f, L, m, orders=(2, 3, 4, 5)):
    """Tension from modes 2 to 5, which sag and coupling perturb least.

    Returns (T, c), with c the least-squares slope of f_n = c n in Hz.
    """
    idx = [o - 1 for o in orders]
    n = np.array(orders, dtype=float)
    fn = np.array([f[i] for i in idx])
    c = np.sum(n * fn) / np.sum(n * n)        # least squares on f_n = c n
    return 4.0 * m * L ** 2 * c ** 2, c


def irvine_lambda2(L, T, m, theta, EA):
    """Irvine parameter lambda^2 and sag ratio d/L of an inclined stay."""
    H = T * np.cos(theta)
    Lh = L * np.cos(theta)
    d = m * G * Lh ** 2 / (8.0 * H)
    Le = Lh * (1.0 + 8.0 * (d / Lh) ** 2)
    return (m * G * Lh / H) ** 2 * Lh / (H * Le / EA), d / L


def main():
    rows = []
    print("=" * 78)
    print("  PONTE DEL MARE, PESCARA: measured stays against the criterion")
    print("=" * 78)
    print("\n1. Harmonicity of each stay's own series, f_n/n [Hz]")
    print("   a constant series means an isolated taut stay\n")
    print("   %-5s %8s %8s %8s %8s %8s   %s" %
          ("stay", "n=1", "2", "3", "4", "5", "spread"))
    for name, (L, m, f) in STAYS.items():
        h = harmonicity(f)
        spread = 100.0 * (h.max() - h.min()) / h.mean()
        print("   %-5s %8.3f %8.3f %8.3f %8.3f %8.3f   %5.1f %%"
              % (name, *h, spread))

    print("\n2. First mode against the tension implied by modes 2 to 5\n")
    print("   %-5s %10s %10s %10s %10s %8s" %
          ("stay", "T [kN]", "f1 pred", "f1 meas", "anomaly", "lambda^2"))
    for name, (L, m, f) in STAYS.items():
        T, c = tension_from_high_modes(f, L, m)
        f1_pred = c                                  # n = 1 on the fitted line
        anom = 100.0 * (f[0] - f1_pred) / f1_pred
        # representative 30 deg inclination; the mast is tilted and the decks curved
        A = m / RHO
        lam2, sag = irvine_lambda2(L, T, m, np.deg2rad(30.0), E_STEEL * A)
        print("   %-5s %10.1f %10.3f %10.3f %+9.1f %% %8.2f"
              % (name, T / 1e3, f1_pred, f[0], anom, lam2))
        rows.append(dict(stay=name, L=L, m=m, T_kN=T / 1e3,
                         f1_pred=f1_pred, f1_meas=f[0], anomaly_pct=anom,
                         lambda2=lam2, sag_ratio=sag))

    print("\n3. Detuning of every stay mode against every identified deck mode")
    print("   d = (f_stay - f_deck)/f_stay, nearest deck mode only\n")
    print("   %-5s %4s %9s %10s %9s   %s" %
          ("stay", "n", "f_n [Hz]", "deck [Hz]", "d", "within 5 %?"))
    near = []
    for name, (L, m, f) in STAYS.items():
        for n, fn in enumerate(f, start=1):
            j = int(np.argmin(np.abs(DECK - fn)))
            d = (fn - DECK[j]) / fn
            flag = "YES" if abs(d) <= 0.05 else ""
            if flag:
                near.append((name, n, fn, DECK[j], d))
            print("   %-5s %4d %9.3f %10.3f %+9.4f   %s"
                  % (name, n, fn, DECK[j], d, flag))
    print()
    if near:
        print("   stay modes sitting within 5 %% of an identified deck mode:")
        for name, n, fn, fd, d in near:
            print("     %s mode %d at %.3f Hz, deck mode at %.3f Hz, d = %+.4f"
                  % (name, n, fn, fd, d))

    d = pd.DataFrame(rows)
    os.makedirs(DATA, exist_ok=True)
    d.to_csv(os.path.join(DATA, "pontedelmare.csv"), index=False)
    print(f"\n   wrote {DATA}/pontedelmare.csv")


if __name__ == "__main__":
    main()
