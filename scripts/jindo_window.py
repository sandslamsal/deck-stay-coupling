# -*- coding: utf-8 -*-
"""Jindo: predicted branch spread truncated to the measured frequency window.

For each width variant the bordered pencil gives branch frequencies and bright
fractions b_k. The spread sqrt(sum_k b_k (lambda_k - lambda_0)^2) / lambda_0,
with lambda = f^2 and b_k renormalized over the kept branches, is formed over
all branches (it reproduces sqrt(sum_i s_i^2)) and over the branches inside
the identified window of Caetano (2001). Widths are read from data/jindo.csv.
Writes data/jindo_window.csv.  Run:  python3 scripts/jindo_window.py
"""

from __future__ import annotations

import sys
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

# Values transcribed from Caetano (2001), PhD thesis, University of Porto,
# Chapter 7 (Jindo 1:150 model). They are not redistributed with this code:
# they live in data/external/jindo_window_data.py (see README).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "external"))
try:
    from jindo_window_data import (  # noqa: E402
        F0, WINDOW)
except ImportError as exc:
    raise SystemExit("scripts/jindo_window.py needs values transcribed from "
                     "Caetano (2001), PhD thesis, University of Porto, Chapter 7 (Jindo 1:150 model), which are not redistributed here. "
                     "See README, 'Third-party data'.") from exc
WINDOWS = [(8.63, 11.18), (8.50, 11.30), (8.30, 11.60), (8.00, 12.00)]
NCOPY = 4                      # four copies of every stay in the model
MEASURED = {                   # validate_jindo step 4, damping-inferred weights
    "low zeta": 9.41,
    "mid zeta": 12.45,
    "high zeta": 14.98,
}
VARIANTS = [                   # csv key, label
    ("s_study_pct", "tie route only, rigid pylon"),
    ("s_generalised_pct", "tie route, with tower-top drive"),
    ("s_two_route_pct", "two routes, rigid pylon"),
    ("s_two_route_tower_pct", "two routes, with tower-top drive"),
]


def load():
    """Per-stay frequency and width variants from data/jindo.csv."""
    stays = {}
    with open(os.path.join(DATA, "jindo.csv"), encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            item = row["item"]
            if not item.startswith("stay "):
                continue
            no = int(item.split()[1])
            s = stays.setdefault(no, {})
            if row["block"] == "stay_geometry" and row["quantity"] in ("f1_irvine", "f1_fem"):
                s[row["quantity"]] = float(row["value"])
            if row["block"] == "ordinates" and row["quantity"].endswith("_pct"):
                s[row["quantity"]] = float(row["value"]) / 100.0
    return [stays[k] for k in sorted(stays)]


def bordered_branches(f0, f_stays, s_list):
    """N+1 branches of one structure mode coupled to N stays (as in
    validate_jindo.bordered_branches). Returns (frequencies ascending, bright
    fraction of each)."""
    n = len(f_stays)
    A = np.zeros((n + 1, n + 1))
    A[0, 0] = f0 ** 2
    for k, (fi, si) in enumerate(zip(f_stays, s_list)):
        A[k + 1, k + 1] = fi ** 2
        A[0, k + 1] = A[k + 1, 0] = si * f0 ** 2
    w, V = np.linalg.eigh(A)
    f = np.sqrt(np.maximum(w, 0.0))
    o = np.argsort(f)
    return f[o], (V[0, :] ** 2)[o]


def spread(f, b, f0=F0, window=None):
    """sqrt(weighted variance of lambda about lambda_0)/lambda_0, in percent.

    `b` is renormalized over the branches inside `window` (all if None).
    Returns (spread %, bright weight kept, number of branches kept)."""
    lam, lam0 = f ** 2, f0 ** 2
    keep = np.ones_like(f, dtype=bool) if window is None else (f >= window[0]) & (f <= window[1])
    w = b[keep]
    if w.sum() <= 0:
        return float("nan"), 0.0, int(keep.sum())
    wn = w / w.sum()
    var = float(wn @ (lam[keep] - lam0) ** 2)
    return 100 * np.sqrt(var) / lam0, float(w.sum()), int(keep.sum())


def main():
    stays = load()
    rows = []

    def rec(*a):
        rows.append(a)

    print("=" * 78)
    print("JINDO: THE PREDICTION TRUNCATED TO THE MEASURED WINDOW")
    print("=" * 78)
    print("Decoupled reference %.2f Hz, physically built." % F0)
    print("Identified branch window %.2f to %.2f Hz (nine modes)." % WINDOW)
    print("Measured spread about that reference, by the weighting of the")
    print("bright pair: %s." % ", ".join("%s %.2f %%" % (k, v) for k, v in MEASURED.items()))
    print()
    print("Stay frequencies against the window (Irvine column):")
    print("  %-5s %9s %9s  %s" % ("stay", "f1 (Hz)", "d (%)", "own frequency in the window?"))
    for i, s in enumerate(stays, 1):
        f = s["f1_irvine"]
        inside = WINDOW[0] <= f <= WINDOW[1]
        print("  %-5d %9.2f %9.1f  %s" % (i, f, 100 * (f - F0) / F0, "yes" if inside else "no"))
        rec("stay_window", "stay %d" % i, "f1_irvine", round(f, 4), "Hz",
            "inside window" if inside else "outside window")
    print()

    for key, label in VARIANTS:
        print("-" * 78)
        print(label)
        for tag, fkey in (("Irvine (taut string)", "f1_irvine"), ("FEM (thesis)", "f1_fem")):
            fs = [s[fkey] for s in stays]
            # the in-phase combination of the NCOPY copies couples as sqrt(NCOPY) s
            ss = [np.sqrt(NCOPY) * s[key] for s in stays]
            f, b = bordered_branches(F0, fs, ss)
            full, wfull, nfull = spread(f, b)
            trunc, wtr, ntr = spread(f, b, window=WINDOW)
            rule = 100 * np.sqrt(NCOPY * sum(s[key] ** 2 for s in stays))
            print("  %-22s sum rule %6.2f %%   full list %6.2f %% (%d branches)"
                  % (tag, rule, full, nfull))
            print("  %-22s truncated to the window %6.2f %%  (%d branches, "
                  "%.0f %% of the bright weight)" % ("", trunc, ntr, 100 * wtr))
            rec("window", "%s / %s" % (label, tag), "sum_rule_pct", round(rule, 3), "%")
            rec("window", "%s / %s" % (label, tag), "full_list_pct", round(full, 3), "%")
            rec("window", "%s / %s" % (label, tag), "truncated_pct", round(trunc, 3), "%")
            rec("window", "%s / %s" % (label, tag), "bright_weight_kept", round(wtr, 4), "-")
            rec("window", "%s / %s" % (label, tag), "branches_kept", ntr, "-")
        print()

    # --- sensitivity to the window edges ---
    # the two-route pencil has a bright branch at 8.60 Hz, just below the lower edge
    print("=" * 78)
    print("SENSITIVITY TO THE WINDOW EDGES")
    print("=" * 78)
    print("The strict window is the thesis's identified range.  The wider ones")
    print("test whether the comparison turns on a branch sitting just outside.")
    print("  %-30s %s" % ("variant", "".join("%12s" % ("%.2f-%.2f" % w) for w in WINDOWS)))
    band = {}
    for key, label in VARIANTS:
        vals = []
        for w in WINDOWS:
            fs = [s["f1_irvine"] for s in stays]
            ss = [np.sqrt(NCOPY) * s[key] for s in stays]
            f, b = bordered_branches(F0, fs, ss)
            sp, _, _ = spread(f, b, window=w)
            vals.append(sp)
            rec("window_sensitivity", label, "%.2f-%.2f Hz" % w, round(sp, 3), "%")
        band[key] = (min(vals), max(vals))
        print("  %-30s %s" % (label, "".join("%12.2f" % v for v in vals)))
    tie = [band["s_study_pct"], band["s_generalised_pct"]]
    two = [band["s_two_route_pct"], band["s_two_route_tower_pct"]]
    tie_lo, tie_hi = min(t[0] for t in tie), max(t[1] for t in tie)
    two_lo, two_hi = min(t[0] for t in two), max(t[1] for t in two)
    print()
    print("  over every window tested:")
    print("     tie route alone   %.1f to %.1f %%" % (tie_lo, tie_hi))
    print("     two routes        %.1f to %.1f %%" % (two_lo, two_hi))
    print("     measured          %.1f to %.1f %%" % (MEASURED["low zeta"], MEASURED["high zeta"]))
    for tag, lo, hi in (("tie_route_alone", tie_lo, tie_hi), ("two_routes", two_lo, two_hi)):
        rec("window_band", tag, "low_pct", round(lo, 3), "%")
        rec("window_band", tag, "high_pct", round(hi, 3), "%")
    print()

    print("=" * 78)
    print("READING")
    print("=" * 78)
    print("The full-list figure reproduces sqrt(sum s^2), which is the check")
    print("that the pencil and the sum rule agree.  The truncated figure is the")
    print("one to set beside the measurement, because the measurement sees only")
    print("the branches inside the window.  A variant whose truncated spread")
    print("falls inside %.2f to %.2f %% is consistent with what was identified."
          % (MEASURED["low zeta"], MEASURED["high zeta"]))

    out = os.path.join(DATA, "jindo_window.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["block", "item", "quantity", "value", "unit", "note"])
        for r in rows:
            w.writerow(list(r) + [""] * (6 - len(r)))
    print()
    print("wrote %s (%d rows)" % (out, len(rows)))


if __name__ == "__main__":
    main()
