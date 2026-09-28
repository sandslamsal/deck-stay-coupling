# -*- coding: utf-8 -*-
"""What the crossing costs depends on how the record is read.

Panel (a): the tension error at exact tuning against damping, for three ways
of reading the same record. Peak picking reads one maximum. The trace rule
reads BOTH peaks of the same single-channel spectrum and uses
f_s^2 = f_+^2 + f_-^2 - f_d^2. Subspace identification fits modes, and is
scored with the one-pole fallback on the records where it returns a single
pole, since an engineer must still report a tension there.

Panel (b): the fraction of the traverse on which the spectrum shows two
peaks, against the tolerance within which a peak must match a branch to
count. A separation rate quoted at one tolerance says as much about the
tolerance as about the method.

Run:  python3 scripts/fig_oma.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import figstyle as F  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "revision 1", "sources", "figures")
DUR = 600.0                     # an ordinary ambient survey


def rms(g, c):
    v = g[c].astype(float).to_numpy()
    v = v[np.isfinite(v)]
    return np.sqrt(np.mean(v ** 2)) if len(v) else np.nan


def main():
    F.apply()
    d = pd.read_csv(os.path.join(DATA, "oma.csv"))
    at = d[(d.d.abs() < 0.004) & (d.snr_db == 20) & (d.duration == DUR)]
    zs = np.array(sorted(at.zeta.unique()))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(F.FIG_W, 2.75))

    series = (("errT_pp_pct",       "model",     "one peak picked"),
              ("errT_pp_tr_pct",    "law",       "two peaks, trace rule"),
              ("errT_cov2_trf_pct", "incumbent", "subspace fit, trace rule"))
    curves = {}
    for col, key, lab in series:
        y = [rms(at[at.zeta == z], col) for z in zs]
        curves[col] = dict(zip(np.round(100 * zs, 6), y))
        a1.plot(100 * zs, y, label=lab, **F.style(key, lw=1.9, ms=6.0, label=False))
    a1.set_xscale("log")
    a1.set_xticks([0.1, 0.2, 0.5, 1.0, 2.0, 3.0])
    a1.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "3"])
    a1.set_xticks([], minor=True)
    a1.set_xlim(0.085, 3.6)
    a1.set_xlabel(r"damping  $\zeta$  (%)")
    a1.set_ylabel(r"error at tuning  $\varepsilon$  (%)")
    F.clean(a1)
    F.headroom(a1, top=0.34)
    a1.legend(loc="upper left", fontsize=9.0, labelspacing=0.24, frameon=False)
    F.panel(a1, "a", "error by reading method")
    # zeta = 0.5 %: above it the spectrum shows one peak and the trace rule
    # cannot be applied. A short dashed line from the axis up to the
    # subspace-fit curve marks it; the caption says what it means.
    y0 = a1.get_ylim()[0]
    a1.plot([0.5, 0.5], [y0, curves["errT_cov2_trf_pct"][0.5]], ls=(0, (3, 2)),
            color="0.40", lw=1.3, zorder=0.5)
    a1.set_ylim(bottom=y0)

    tols = [(1.0, "both_pp_tol10"), (0.5, "both_pp_tol5"),
            (0.3, "both_pp_tol3"), (0.2, "both_pp_tol2")]
    g0 = d[(d.snr_db == 20) & (d.duration == DUR)]
    for z, key in ((0.001, "n1"), (0.002, "n2"), (0.005, "n3")):
        g = g0[np.isclose(g0.zeta, z)]
        y = [100 * g[c].mean() for _, c in tols]
        a2.plot([t for t, _ in tols], y,
                label=rf"$\zeta = {100*z:g}\,\%$",
                **F.style(key, lw=1.9, ms=6.0, label=False))
    a2.set_xlabel("peak-match tolerance  (%)")
    a2.set_ylabel("range with two peaks  (%)")
    a2.set_xlim(1.08, 0.12)
    a2.set_ylim(0, 100)
    F.clean(a2)
    a2.legend(loc="lower left", fontsize=9.0, labelspacing=0.24, frameon=False)
    F.panel(a2, "b", "dependence on the match tolerance")

    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.175, top=0.845,
                        wspace=0.32)
    os.makedirs(OUT, exist_ok=True)
    probs = F.audit(fig)
    if probs:
        print(f"  {len(probs)} audit problems above")
    png = os.path.join(OUT, "fig_oma.png")
    fig.savefig(png)
    fig.savefig(png.replace(".png", ".pdf"))
    print(f"  wrote {png}")
    for z in zs:
        g = at[at.zeta == z]
        print(f"  zeta {100*z:>4.1f} %: pick {rms(g,'errT_pp_pct'):5.2f}  "
              f"trace {rms(g,'errT_pp_tr_pct'):5.2f} (n={int((g.n_pp_trace>=2).sum())})  "
              f"ssi+fallback {rms(g,'errT_cov2_trf_pct'):5.2f}")


if __name__ == "__main__":
    main()
