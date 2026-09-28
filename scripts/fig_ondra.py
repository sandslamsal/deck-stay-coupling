# -*- coding: utf-8 -*-
"""Closed form against the beam-tendon rig of Ondra and Titurus (2019).

Panel (a): measured and predicted veering width times mode order, n*s, for
three veering events (a flat line is the 1/n scaling). Panel (b): tension
error against relative detuning for the n = 1 and n = 2 events.
Reads data/external/fig_ondra_data.py and data/ondra.csv; writes fig_ondra.png
and fig_ondra.pdf to the figures directory OUT.

Run:  python3 scripts/fig_ondra.py
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
from matplotlib.lines import Line2D  # noqa: E402

DATA = os.path.join(ROOT, "data")
OUT = os.environ.get("FIGURE_DIR", os.path.join(ROOT, "figures"))

# The three measured veering events are transcribed from Ondra and Titurus
# (2019). They are not redistributed with this code: they live in
# data/external/fig_ondra_data.py (see README).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "external"))
try:
    from fig_ondra_data import (  # noqa: E402
        EVENTS)
except ImportError as exc:
    raise SystemExit("scripts/fig_ondra.py needs values transcribed from "
                     "Ondra and Titurus (2019) (beam-tendon laboratory rig), which are not redistributed here. "
                     "See README, 'Third-party data'.") from exc


def main():
    F.apply()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(F.FIG_W, 2.75))

    # ---- (a) the closed form, and the 1/n scaling ----------------------
    n = np.array([e[0] for e in EVENTS], dtype=float)
    sm = np.array([e[1] for e in EVENTS])
    sp = np.array([e[2] for e in EVENTS])
    sf = np.array([e[3] for e in EVENTS])
    a1.plot(n, 100 * n * sm, label="measured",
            **F.style("model", lw=1.9, ms=6.5, label=False))
    a1.plot(n, 100 * n * sp, label="closed form",
            **F.style("law", lw=1.9, ms=6.0, label=False))
    ok = ~np.isnan(sf)
    a1.plot(n[ok], 100 * (n * sf)[ok], label="coupled model, same rig",
            **F.style("incumbent", lw=1.9, ms=6.0, label=False))
    a1.set_xticks([1, 2, 3])
    a1.set_xlabel("stay mode order  $n$")
    a1.set_ylabel(r"width times order  $n\,s$   (%)")
    a1.set_ylim(0, 22)
    F.clean(a1)
    a1.legend(loc="lower left", fontsize=9.0, labelspacing=0.24, frameon=False)
    F.panel(a1, "a", "the width, and the $1/n$ scaling")
    # a flat n s line is the 1/n scaling
    F.note(a1, 0.52, 0.80, r"$s \propto 1/n$", fontsize=10.0, color="0.30")

    # ---- (b) the tension error against detuning --------------------------
    d = pd.read_csv(os.path.join(DATA, "ondra.csv"))
    for panel, key, lab in (("(a)", "n1", "$n=1$ event"),
                            ("(b)", "n2", "$n=2$ event")):
        g = d[d.panel == panel].sort_values("d_meas")
        lo, hi = g.eps_meas_lo_pct.values, g.eps_meas_hi_pct.values
        a2.fill_between(g.d_meas, lo, hi, color=F.color(key), alpha=0.22, lw=0)
        a2.plot(g.d_meas, g.eps_pred_pct, label=lab,
                **F.style(key, marker="none", lw=1.9, label=False))
    a2.axhline(0, color=F.GRAY, lw=0.9, zorder=0)
    a2.axvline(0, color=F.GRAY, lw=0.9, ls=(0, (2, 2)), zorder=0)
    a2.set_xlabel("relative detuning  $d$")
    a2.set_ylabel(r"tension error  $\varepsilon$  (%)")
    F.clean(a2)
    F.headroom(a2, top=0.30)
    a2.legend(loc="lower right", fontsize=9.0, labelspacing=0.24, frameon=False)
    F.panel(a2, "b", "the tension error it induces")

    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.175, top=0.845,
                        wspace=0.30)
    os.makedirs(OUT, exist_ok=True)
    probs = F.audit(fig)
    if probs:
        print(f"  {len(probs)} audit problems above")
    png = os.path.join(OUT, "fig_ondra.png")
    fig.savefig(png)
    fig.savefig(png.replace(".png", ".pdf"))
    print(f"  wrote {png}")
    print(f"  n*s measured: {', '.join(f'{100*x:.2f}' for x in n*sm)} %"
          f"   spread {100*(n*sm).ptp()/(n*sm).mean():.1f} % about the mean")
    for k, sm_, sp_ in zip(n, sm, sp):
        print(f"  n={int(k)}: measured {100*sm_:.2f} %, law {100*sp_:.2f} %,"
              f"  {100*(sp_/sm_-1):+.1f} %")


if __name__ == "__main__":
    main()
