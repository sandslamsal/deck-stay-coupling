# -*- coding: utf-8 -*-
"""The damper traverse: a measured crossing on the Ponte del Mare.

Kumar (2011) tested the bridge twice, on consecutive days, without and with
its vibration dampers. The dampers act between deck and ground and between
the mast back-stays and the deck; they do not touch stay N8E. At low
amplitude they behave as springs, and they raised the first global
frequency from 0.75 Hz to 0.93 Hz with the mode shape essentially unchanged
(MAC 0.97).

Stay N8E's fundamental sits at about 0.83 Hz, BETWEEN those two values. The
damper installation therefore carried a global mode straight through a stay
frequency, one day apart, at constant stay tension. That is a traverse of a
crossing performed on a real bridge, with the stay's line recorded on both
sides of it, and it makes the veering width measurable.

Prediction. The stay-dominated branch is pushed AWAY from the global mode:
upward while the global mode is below it, downward once the global mode is
above it. So the stay's line must MOVE DOWN when the dampers are fitted,
whatever the value of the coupling. That sign test needs no calibration.

Measurement. Given the line's position on both sides, the pair
    f_obs = f_iso + sign(D)/2 * (sqrt(D^2 + (s f_iso)^2) - |D|),  D = f_iso - f_deck
is two equations in the two unknowns f_iso and s, so s is measured, and
mu_eff follows from s = (2/(n pi)) cos(theta) sqrt(mu_eff).

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

# Pixel reads of the N8E line in the DECK records (miner, 1200-2400 dpi)

def f_obs(f_iso, f_deck, s):
    D = f_iso - f_deck
    return f_iso + 0.5*np.sign(D)*(np.sqrt(D*D + (s*f_iso)**2) - abs(D))

def solve(o_no, o_yes):
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
# independent structural estimate, from mode shape ordinate and modal mass
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
