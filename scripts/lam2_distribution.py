# -*- coding: utf-8 -*-
"""Irvine's parameter across the graded population, in the corrected form.

WHY THIS EXISTS
---------------
`cablefe.irvine_lambda2` was corrected during this revision. It had evaluated
the parameter on the horizontal projection of the chord, which overstates an
inclined cable's value by 1/cos^3(theta); it now evaluates it on the chord
itself. The correction is recorded in that function's docstring.

`data/campaign.csv` and `data/campaign_sag.csv` were written before the
correction, so their `lam2` column is the horizontal-projection value while
every lambda^2 quoted in the manuscript is the chord value. Nothing else
depends on that column - the error law is computed from the finite element
frequencies, not from lambda^2 - so the column is diagnostic only, and
re-running the whole 421-design campaign to refresh one derived column would
risk perturbing figures the manuscript already quotes.

This script therefore recomputes the column in place from the stored geometry
(L_c, T, EA, m_c, theta), which is all the parameter depends on, so the
deposited data agrees with the deposited code. It then writes the
distribution the manuscript quotes.

usage:  python3 lam2_distribution.py          (patch the columns and report)
        python3 lam2_distribution.py --dry    (report only)

Output: data/lam2_distribution.csv, and the patched `lam2` columns.
"""

from __future__ import annotations

import csv
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, os.path.join(ROOT, "src"))

from cablefe import irvine_lambda2          # noqa: E402  (after sys.path)

GATE = 150.0            # xi above which a design is graded, as in run_campaign.py
DRY = "--dry" in sys.argv


def chord_lam2(df):
    return [irvine_lambda2(r.Lc, r.T, r.EA, r.mc, r.theta) for r in df.itertuples()]


def r2(sub):
    y, f = sub.eps_coupling.abs(), sub.eps_pred.abs()
    return 1.0 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()


rows = []
for name in ("campaign.csv", "campaign_sag.csv"):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        print("  %s not found, skipped" % name)
        continue
    d = pd.read_csv(path)
    old = d["lam2"].copy()
    new = chord_lam2(d)
    d["lam2"] = new
    changed = (abs(old - d["lam2"]) > 1e-9).sum()
    print("%-20s %d rows, %d values updated (max ratio old/new %.3f)"
          % (name, len(d), changed, (old / d["lam2"]).max()))
    if not DRY:
        d.to_csv(path, index=False)

# the distribution the manuscript quotes, from the sag campaign (which carries
# the same designs plus the sagged solution)
d = pd.read_csv(os.path.join(DATA, "campaign_sag.csv"))
if DRY:
    d["lam2"] = chord_lam2(d)
g = d[d.xi > GATE]
l = g["lam2"]
lo, hi = g[l <= 1], g[l > 1]
print()
print("graded population (xi > %g): %d designs" % (GATE, len(g)))
print("  median lambda^2            %.4f" % l.median())
print("  lambda^2 > 0.1             %d" % (l > 0.1).sum())
print("  lambda^2 > 1               %d" % (l > 1).sum())
print("  R^2 of the error law, all  %.4f" % r2(g))
print("     on the %3d with lambda^2 <= 1   %.4f" % (len(lo), r2(lo)))
print("     on the %3d with lambda^2 > 1    %.4f" % (len(hi), r2(hi)))
print("  (the straight-chord model carries no sag, so the second figure shows")
print("   only that the model has no sag physics, not that sag does not matter)")

rows = [
    ("lam2_distribution", "graded population", "n_designs", len(g), "-", "xi > %g" % GATE),
    ("lam2_distribution", "graded population", "median_lam2", round(float(l.median()), 5), "-", "chord form"),
    ("lam2_distribution", "graded population", "n_lam2_gt_0p1", int((l > 0.1).sum()), "-", ""),
    ("lam2_distribution", "graded population", "n_lam2_gt_1", int((l > 1).sum()), "-", ""),
    ("lam2_distribution", "error law", "R2_all", round(float(r2(g)), 4), "-", ""),
    ("lam2_distribution", "error law", "R2_lam2_le_1", round(float(r2(lo)), 4), "-", "n = %d" % len(lo)),
    ("lam2_distribution", "error law", "R2_lam2_gt_1", round(float(r2(hi)), 4), "-", "n = %d" % len(hi)),
]
out = os.path.join(DATA, "lam2_distribution.csv")
if not DRY:
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["block", "item", "quantity", "value", "unit", "note"])
        w.writerows(rows)
    print("\nwrote %s" % out)
