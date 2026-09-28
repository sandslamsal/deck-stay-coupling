# -*- coding: utf-8 -*-
"""Figure: the closed form, the collapse onto it, and the screening criterion.

Three panels at the printed width of the cas-sc single column.

  (a)  the tension error of the isolated-cable inversion against relative
       detuning, with the closed form drawn for three values of the split.
       Shows the shape of the law: a peak of height s at an exact crossing,
       decaying as s^2/(2|d|).
  (b)  measured against predicted over the campaign, coloured and marked by
       stay mode order.  The collapse is the claim, and separating the
       orders is what shows the 1/n factor is carrying its weight.
  (c)  the screening criterion, the detuning required to hold the error
       within a tolerance, against mu_eff, one family per tolerance.  This is
       the panel an engineer uses.

Run:  python3 scripts/fig_collapse.py
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
from cablefe import veering_split  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(ROOT, "revision 1", "sources", "figures")


def main():
    F.apply()
    d = pd.read_csv(os.path.join(ROOT, "data", "campaign.csv"))
    g = d[(d.mac > 0.5) & (d.xi > 150)].copy()
    g["eps_c"] = (g.eps - g.eps_control).abs()

    fig, axes = plt.subplots(1, 3, figsize=(F.FIG_W, 2.45))
    ax1, ax2, ax3 = axes

    # ---- (a) the shape of the law ---------------------------------------
    dd = np.linspace(-0.20, 0.20, 601)
    for key, sval in (("n1", 0.08), ("n3", 0.04), ("n5", 0.02)):
        eps = np.sqrt(dd ** 2 + sval ** 2) - np.abs(dd)
        ax1.plot(dd, 100 * eps, label=f"$s = {sval*100:.0f}$ %",
                 **F.style(key, marker='none', lw=1.8, label=False))
    ax1.set_xlabel(r"relative detuning $d$")
    ax1.set_ylabel(r"tension error $\varepsilon$  (%)")
    F.clean(ax1)
    ax1.legend(loc="upper right", fontsize=F.FS_LEGEND)
    F.panel(ax1, "a", "the closed form")

    # ---- (b) the collapse ------------------------------------------------
    for i, key in enumerate(F.ORDERS, start=1):
        sub = g[g.n_stay == i]
        if not len(sub):
            continue
        ax2.plot(100 * sub.eps_pred, 100 * sub.eps_c,
                 **F.style(key, ls='none', ms=3.0, mew=0.6, alpha=0.85))
    lim = max(100 * g.eps_pred.max(), 100 * g.eps_c.max()) * 1.05
    ax2.plot([0, lim], [0, lim], color=F.GRAY, lw=1.0, ls=(0, (5, 2)),
             zorder=0, label="1:1")
    ax2.set_xlim(0, lim)
    ax2.set_ylim(0, lim * 1.28)      # headroom so the legend clears the data
    ax2.set_xlabel(r"closed form  $\varepsilon$  (%)")
    ax2.set_ylabel(r"finite element  $\varepsilon$  (%)")
    F.clean(ax2)
    ax2.legend(loc="upper left", fontsize=F.FS_SMALL, ncol=1,
               labelspacing=0.22, borderaxespad=0.1)
    F.panel(ax2, "b", "the parametric study")

    # ---- (c) the criterion ----------------------------------------------
    mu = np.logspace(-4, -1, 500)
    for key, tol in (("tol2", 0.02), ("tol5", 0.05), ("tol10", 0.10)):
        sv = veering_split(mu, np.deg2rad(35.0), 1)
        req = np.maximum((sv ** 2 - tol ** 2) / (2 * tol), 0.0)
        ax3.plot(mu, 100 * req, label=f"tolerance {tol*100:.0f} %",
                 **F.style(key, marker='none', lw=1.8, label=False))
    ax3.set_ylim(0, 30)
    ax3.set_xlim(1e-4, 1e-1)
    ax3.set_xscale("log")
    # mathtext exponents render at 0.7 of the declared size, so the decade
    # labels need a larger tick size to clear the 6.5 pt floor.
    ax3.tick_params(axis='x', labelsize=9.6)
    ax3.set_xlabel(r"effective mass ratio $\mu_{\mathrm{eff}}$")
    ax3.set_ylabel(r"detuning needed  $|d|$  (%)")
    F.clean(ax3)
    ax3.legend(loc="upper left", fontsize=F.FS_SMALL, labelspacing=0.22,
               borderaxespad=0.1)
    # The mode order the family is drawn for goes in the caption, not on the
    # axes: every placement here clipped the 2 per cent curve, and the house
    # rule is that no label sits over data.
    F.panel(ax3, "c", "screening criterion")

    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.19, top=0.86,
                        wspace=0.36)
    os.makedirs(OUT, exist_ok=True)
    probs = F.audit(fig)
    if probs:
        print(f"  {len(probs)} audit problems above")
    png = os.path.join(OUT, "fig_collapse.png")
    fig.savefig(png)
    fig.savefig(png.replace(".png", ".pdf"))
    print(f"  wrote {png}")
    print(f"  n = {len(g)} graded designs, stay orders "
          f"{sorted(g.n_stay.unique())}")


if __name__ == "__main__":
    main()
