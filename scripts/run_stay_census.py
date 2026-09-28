# -*- coding: utf-8 -*-
"""Stays of Ponte del Mare inside the veering zone of an identified global mode.

Reads data/pontedelmare_all.csv (Kumar (2011), Table 5.5) and the identified
global modes from data/external/. For each stay order in the identified band,
computes the detuning d = (f_n - f_g) / f_n to the nearest mode and counts the
stays with |d| < s and those failing the screening criterion at 2 %, for three
veering widths s. Writes data/stay_census.csv and data/stay_census_summary.csv.
Run:  python3 scripts/run_stay_census.py
"""
from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

# Values transcribed from Kumar (2011), PhD thesis, University of Trento
# (Ponte del Mare footbridge). They are not redistributed with this code:
# they live in data/external/run_stay_census_data.py (see README).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "external"))
try:
    from run_stay_census_data import (  # noqa: E402
        DECK)
except ImportError as exc:
    raise SystemExit("scripts/run_stay_census.py needs values transcribed from "
                     "Kumar (2011), PhD thesis, University of Trento (Ponte del Mare footbridge), which are not redistributed here. "
                     "See README, 'Third-party data'.") from exc
BAND = (0.70, 2.90)          # Hz, band covered by the identified modes
S_MEDIAN, S_P90 = 0.0032, 0.015   # campaign median and 90th-percentile widths
M_DECK_MIN = 40e3            # kg, lightest deck modal mass the record admits
TOL = 0.02                   # tolerance of the screening criterion


def crit_fail(d, s, tol=TOL):
    return np.abs(d) < (s ** 2 - tol ** 2) / (2.0 * tol)


def main():
    t = pd.read_csv(os.path.join(DATA, "pontedelmare_all.csv"))
    f_chk = np.sqrt(t.T_meas_kN * 1e3 / t.m_kgpm) / (2.0 * t.L_m)
    worst = np.abs(f_chk - t.f1_Hz).max()
    assert worst < 0.01, f"transcription check failed: {worst:.3f} Hz"
    print(f"{len(t)} stays; taut-string check of the transcription: worst {worst*1e3:.1f} mHz")

    rows = []
    for _, r in t.iterrows():
        M_s = 0.5 * r.m_kgpm * r.L_m
        for n in range(1, 6):
            fn = n * r.f1_Hz
            if not (BAND[0] <= fn <= BAND[1]):
                continue
            j = int(np.argmin(np.abs(DECK - fn)))
            d = (fn - DECK[j]) / fn
            # width ceiling from mass normalization, with cos(theta) = 1
            s_max = (2.0 / (n * np.pi)) * np.sqrt(M_s / M_DECK_MIN)
            rows.append(dict(stay=r.stay, n=n, f_n=fn, f_global=DECK[j], d=d, abs_d=abs(d),
                             M_s=M_s, s_max=s_max,
                             in_median=int(abs(d) < S_MEDIAN), in_p90=int(abs(d) < S_P90),
                             in_ceiling=int(abs(d) < s_max),
                             fail_median=int(crit_fail(d, S_MEDIAN)),
                             fail_p90=int(crit_fail(d, S_P90)),
                             fail_ceiling=int(crit_fail(d, s_max))))
    c = pd.DataFrame(rows)
    c.to_csv(os.path.join(DATA, "stay_census.csv"), index=False)

    fund = c[c.n == 1]
    anyo = c.groupby("stay").max(numeric_only=True)
    summ = []
    for lab, sub in (("fundamental", fund), ("any order in band", anyo)):
        summ.append(dict(scope=lab, n_stays=len(sub),
                         inside_width_median=int(sub.in_median.sum()),
                         inside_width_p90=int(sub.in_p90.sum()),
                         inside_width_ceiling=int(sub.in_ceiling.sum()),
                         fail_crit2_median=int(sub.fail_median.sum()),
                         fail_crit2_p90=int(sub.fail_p90.sum()),
                         fail_crit2_ceiling=int(sub.fail_ceiling.sum())))
    s = pd.DataFrame(summ)
    s.to_csv(os.path.join(DATA, "stay_census_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(s.to_string(index=False))
    dens = len(DECK) / (DECK.max() - DECK.min())
    print(f"\nidentified modal density {dens:.1f} modes/Hz over {DECK.min():.2f}-{DECK.max():.2f} Hz;"
          f" expected fraction of stays inside +-s of some mode at s = 0.32 %: "
          f"{2*S_MEDIAN*dens*np.median(fund.f_n):.3f}, at 1.5 %: {2*S_P90*dens*np.median(fund.f_n):.3f}")
    print(f"stays with a fundamental in the identified band: {len(fund)} of {len(t)}")
    print("\nranking by |d| of the fundamental (the seasonal test's prediction), closest first:")
    rk = fund.sort_values("abs_d")[["stay", "f_n", "f_global", "d", "s_max"]]
    print(rk.head(10).round(4).to_string(index=False))
    print("wrote stay_census.csv and stay_census_summary.csv")


if __name__ == "__main__":
    main()
