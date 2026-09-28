# -*- coding: utf-8 -*-
"""Figure: the mechanism, on one bridge, as the stay tension is varied.

Everything in this study rests on one claim about the physics: near a
crossing there is no isolated stay mode to measure.  This figure shows that
happening.

  (a)  frequency loci against stay tension.  The uncoupled stay and deck
       frequencies cross; the coupled ones do not.  They approach, repel and
       exchange character, which is what frequency loci veering means.
  (b)  the fraction of modal kinetic energy carried by the stay, for the two
       branches.  Away from the crossing one branch is the stay and the other
       is the deck.  At the crossing both are half and half, so the question
       "which peak is the stay mode" has no answer there.
  (c)  the consequence: the tension error the isolated-cable inversion
       incurs, computed on this bridge, with the closed form drawn through
       it.  The first figure gives the law in the abstract; this is one real
       instance of it.

Reads the CSVs written by scripts/run_veering.py, which is separated out
because the computation costs minutes and the figure needs iterating.

Run:  python3 scripts/fig_veering.py
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

OUT = os.path.join(ROOT, "revision 1", "sources", "figures")
DATA = os.path.join(ROOT, "data")


def main():
    F.apply()
    loci = pd.read_csv(os.path.join(DATA, "veering_loci.csv"))
    br = pd.read_csv(os.path.join(DATA, "veering_branches.csv"))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(F.FIG_W, 2.5))

    # ---- (a) loci --------------------------------------------------------
    # Filled against open marks, not colour alone, so the two branches stay
    # separable in a greyscale print.
    stay = loci[loci.stay_frac > 0.5]
    deck = loci[loci.stay_frac <= 0.5]
    ax1.plot(stay['T'] / 1e3, stay.f, ls='none', marker='o', ms=2.4,
             color=F.BLACK, mew=0, zorder=3)
    ax1.plot(deck['T'] / 1e3, deck.f, ls='none', marker='o', ms=2.8,
             mfc='white', mec=F.SKY, mew=0.7, zorder=3)
    ax1.plot(br['T'] / 1e3, br.f_stay_unc, color=F.GRAY, ls=(0, (5, 2)),
             lw=1.1, zorder=1)
    ax1.plot(br['T'] / 1e3, br.f_deck_unc, color=F.GRAY, ls=(0, (1.4, 1.3)),
             lw=1.1, zorder=1)
    ax1.set_xlabel(r"stay tension  $T$  (kN)")
    ax1.set_ylabel(r"frequency  $f$  (Hz)")
    F.clean(ax1)
    ax1.legend(handles=[
        Line2D([], [], color=F.BLACK, ls='none', marker='o', ms=3.0, mew=0,
               label="stay-dominated"),
        Line2D([], [], color=F.SKY, ls='none', marker='o', ms=3.2,
               mfc='white', mew=0.7, label="deck-dominated"),
        Line2D([], [], color=F.GRAY, ls=(0, (5, 2)), lw=1.1,
               label="uncoupled stay cable"),
        Line2D([], [], color=F.GRAY, ls=(0, (1.4, 1.3)), lw=1.1,
               label="uncoupled deck")],
        loc="lower right", fontsize=F.FS_SMALL, labelspacing=0.22,
        borderaxespad=0.15)
    F.panel(ax1, "a", "veering of the loci")

    # ---- (b) character exchange -----------------------------------------
    ax2.plot(br.d, br.lower, label="lower branch",
             **F.style('n1', marker='none', lw=1.8, label=False))
    ax2.plot(br.d, br.upper, label="upper branch",
             **F.style('n3', marker='none', lw=1.8, label=False))
    ax2.axhline(0.5, color=F.GRAY, lw=0.8, ls=(0, (2, 2)), zorder=0)
    ax2.set_xlabel(r"relative detuning $d$")
    ax2.set_ylabel(r"energy share  $E_{\mathrm{s}}/E$")
    ax2.set_ylim(-0.04, 1.22)
    F.clean(ax2)
    # the panels are narrow; automatic ticks collide at the lower-left
    # corner, so the detuning axes carry explicit round values
    ax2.set_xticks([-0.2, -0.1, 0.0, 0.1])
    ax2.legend(loc="upper right", fontsize=F.FS_SMALL, labelspacing=0.22,
               borderaxespad=0.15)
    F.panel(ax2, "b", "exchange of character")

    # ---- (c) the consequence --------------------------------------------
    ax3.plot(br.d, 100 * br.eps.abs(), label="coupled model",
             **F.style('model', ls='none', ms=2.6, mew=0, label=False))
    dd = np.linspace(br.d.min(), br.d.max(), 400)
    s = float(br.s_pred.iloc[0])
    ax3.plot(dd, 100 * (np.sqrt(dd ** 2 + s ** 2) - np.abs(dd)),
             label="closed form",
             **F.style('law', marker='none', lw=1.8, label=False))
    ax3.set_xlabel(r"relative detuning $d$")
    ax3.set_ylabel(r"tension error  (%)")
    F.headroom(ax3, top=0.30)          # keep the legend clear of the peak
    F.clean(ax3)
    ax3.set_xticks([-0.2, -0.1, 0.0, 0.1])
    ax3.legend(loc="upper right", fontsize=F.FS_SMALL, labelspacing=0.22,
               borderaxespad=0.15)
    F.panel(ax3, "c", "the resulting error")

    fig.subplots_adjust(left=0.072, right=0.995, bottom=0.185, top=0.86,
                        wspace=0.36)
    os.makedirs(OUT, exist_ok=True)
    probs = F.audit(fig)
    if probs:
        print(f"  {len(probs)} audit problems above")
    png = os.path.join(OUT, "fig_veering.png")
    fig.savefig(png)
    fig.savefig(png.replace(".png", ".pdf"))
    print(f"  wrote {png}")
    print(f"  split predicted {100*s:.2f} %, peak computed "
          f"{100*br.eps.abs().max():.2f} %")


if __name__ == "__main__":
    main()
