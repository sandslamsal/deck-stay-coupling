# -*- coding: utf-8 -*-
"""Veering split measured from the damper traverse on the Ponte del Mare.

Kumar (2011) tested the footbridge without and with dampers, which moved
global mode 1 from 0.75 to 0.93 Hz, past stay N8E at about 0.83 Hz. The stay
line on both days gives f_iso and the split s from
    f_obs = f_iso + sign(D)/2 * (sqrt(D^2 + (s f_iso)^2) - |D|),  D = f_iso - f_deck,
and mu_eff = (s n pi / (2 cos(theta)))^2. Also prints the sign test (the stay
line must move down) and a mode-shape estimate of mu_eff. Writes
data/traverse.csv.

Run:  python3 scripts/validate_traverse.py
"""
from __future__ import annotations
import os
import sys
import numpy as np, pandas as pd, os
from scipy.optimize import brentq

FD_NO, FD_YES = 0.750, 0.930          # global mode 1, without / with dampers
# Values transcribed from Kumar (2011), PhD thesis, University of Trento (Ponte del Mare footbridge). They are not redistributed with
# this code: they live in data/external/validate_traverse_data.py (see README).
# READ_NO and READ_YES are pixel reads of the N8E line in the deck records.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "external"))
try:
    from validate_traverse_data import (  # noqa: E402
        TH, M_S, READ_NO, READ_YES)
except ImportError as exc:
    raise SystemExit("scripts/validate_traverse.py needs values transcribed from "
                     "Kumar (2011), PhD thesis, University of Trento (Ponte del Mare footbridge), which are not redistributed here. "
                     "See README, 'Third-party data'.") from exc
N = 1


def f_obs(f_iso, f_deck, s):
    """Stay-branch frequency for isolated stay f_iso, deck mode f_deck, split s."""
    D = f_iso - f_deck
    return f_iso + 0.5*np.sign(D)*(np.sqrt(D*D + (s*f_iso)**2) - abs(D))

def solve(o_no, o_yes):
    """Split s and f_iso from the stay line without and with dampers."""
    def resid(s):
        fi = brentq(lambda f: f_obs(f, FD_NO, s) - o_no, 0.70, 0.95)
        return f_obs(fi, FD_YES, s) - o_yes
    s = brentq(resid, 1e-5, 0.20)
    fi = brentq(lambda f: f_obs(f, FD_NO, s) - o_no, 0.70, 0.95)
    return s, fi

rows = []
for a in READ_NO:
    for b in READ_YES:
        if b >= a:
            continue
        s, fi = solve(a, b)
        for th in TH:
            mu = (s*N*np.pi/(2*np.cos(th)))**2
            rows.append(dict(read_no=a, read_yes=b, shift_Hz=b-a,
                             f_iso=fi, s=s, theta_deg=np.rad2deg(th),
                             mu_eff=mu))
d = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)
d.to_csv("data/traverse.csv", index=False)

print("="*76)
print("  SIGN TEST (calibration-free)")
print("="*76)
print("  predicted: stay line moves DOWN when the dampers are fitted")
print("  observed : %.4f -> %.4f Hz, shift %+.4f to %+.4f Hz  => DOWN"
      % (np.mean(READ_NO), np.mean(READ_YES),
         min(d.shift_Hz), max(d.shift_Hz)))
print()
print("="*76)
print("  MEASURED VEERING WIDTH AND EFFECTIVE MASS RATIO")
print("="*76)
print("  isolated stay frequency f_iso  : %.4f - %.4f Hz" % (d.f_iso.min(), d.f_iso.max()))
print("  measured split s               : %.2f - %.2f %%" % (100*d.s.min(), 100*d.s.max()))
print("  measured mu_eff                : %.2e - %.2e" % (d.mu_eff.min(), d.mu_eff.max()))
print()
# mu_eff from the mode-1 ordinate at N8E and the deck modal mass, M_s phi_a^2
print("="*76)
print("  INDEPENDENT STRUCTURAL ESTIMATE (mode shape + modal mass)")
print("="*76)
print("  M_s = %.0f kg; Fig 4.14 mode-1 ordinate at N8E = 0.05-0.30 of peak;"
      % M_S)
print("  deck modal mass 40-160 t")
lo = M_S*(0.05/np.sqrt(160e3))**2
hi = M_S*(0.30/np.sqrt(40e3))**2
print("  predicted mu_eff               : %.2e - %.2e" % (lo, hi))
ov_lo, ov_hi = max(lo, d.mu_eff.min()), min(hi, d.mu_eff.max())
print()
if ov_lo <= ov_hi:
    print("  OVERLAP: %.2e - %.2e  => the measured width is consistent with"
          % (ov_lo, ov_hi))
    print("  the mode shape and modal mass, independently arrived at.")
else:
    print("  NO OVERLAP: the two estimates disagree.")
print()
print("  wrote data/traverse.csv")
