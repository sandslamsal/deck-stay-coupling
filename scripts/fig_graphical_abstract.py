# -*- coding: utf-8 -*-
"""Graphical abstract: the problem, the criterion, and the repair.

The journal asks for at least 1328 x 531 px. The panel is built at the
journal's proportions and rendered at a resolution that clears that with
room to spare.

Three beats, left to right, which is the whole argument as the manuscript
now orders it:

  1. a stay frequency meets a deck frequency and the loci veer, so the peak
     on a stay-mounted record is not a stay mode;
  2. the screen that flags an exposed stay from its schedule alone;
  3. the error the reading incurs, which depends on how the record is read:
     one picked peak is the worst of the three ways of reading it.

Run:  python3 scripts/fig_graphical_abstract.py
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
from matplotlib.lines import Line2D  # noqa: E402

DATA = os.path.join(ROOT, "data")


def main():
    F.apply()
    plt.rcParams.update({"font.size": 11.0, "axes.labelsize": 11.0,
                         "xtick.labelsize": 10.0, "ytick.labelsize": 10.0,
                         "legend.fontsize": 9.5})

    loci = pd.read_csv(os.path.join(DATA, "veering_loci.csv"))
    br = pd.read_csv(os.path.join(DATA, "veering_branches.csv"))
    idf = pd.read_csv(os.path.join(DATA, "identify2.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.35))
    a1, a2, a3 = axes

    # ---- 1. the loci veer ------------------------------------------------
    stay = loci[loci.stay_frac > 0.5]
    deck = loci[loci.stay_frac <= 0.5]
    a1.plot(stay["T"] / 1e3, stay.f, ls="none", marker="o", ms=3.0,
            color=F.BLACK, mew=0)
    a1.plot(deck["T"] / 1e3, deck.f, ls="none", marker="o", ms=3.4,
            mfc="white", mec=F.SKY, mew=0.9)
    a1.plot(br["T"] / 1e3, br.f_stay_unc, color=F.GRAY, ls=(0, (5, 2)), lw=1.3)
    a1.plot(br["T"] / 1e3, br.f_deck_unc, color=F.GRAY, ls=(0, (1.5, 1.4)),
            lw=1.3)
    a1.set_xlabel("stay tension  (kN)")
    a1.set_ylabel("frequency  (Hz)")
    a1.set_title("a stay cable mode meets a deck mode", fontsize=11.5,
                 fontweight="bold", pad=8)
    a1.legend(handles=[
        Line2D([], [], color=F.BLACK, ls="none", marker="o", ms=3.6, mew=0,
               label="stay-dominated"),
        Line2D([], [], color=F.SKY, ls="none", marker="o", ms=3.8,
               mfc="white", mew=0.9, label="deck-dominated"),
        Line2D([], [], color=F.GRAY, ls=(0, (5, 2)), lw=1.3,
               label="uncoupled")],
        loc="lower right", frameon=False, labelspacing=0.25)
    F.clean(a1)
    F.note(a1, 0.04, 0.90, "the loci veer:\nneither peak is\nthe stay cable mode",
           fontsize=9.5, color="0.25")

    # ---- 2. the criterion ------------------------------------------------
    mu = np.logspace(-4, -1, 400)
    for key, tol, lab in (("tol2", 0.02, "2 %"), ("tol5", 0.05, "5 %"),
                          ("tol10", 0.10, "10 %")):
        sv = veering_split(mu, np.deg2rad(35.0), 1)
        req = np.maximum((sv ** 2 - tol ** 2) / (2 * tol), 0.0)
        a2.plot(mu, 100 * req, label=f"tolerance {lab}",
                **F.style(key, marker="none", lw=2.2, label=False))
    a2.set_xscale("log")
    a2.set_xlim(1e-4, 1e-1)
    a2.set_ylim(0, 30)
    a2.tick_params(axis="x", labelsize=11.0)
    a2.set_xlabel(r"effective mass ratio  $\mu_{\mathrm{eff}}$")
    a2.set_ylabel(r"detuning required,  $|d|$  (%)")
    a2.set_title("screen it before instrumenting", fontsize=11.5,
                 fontweight="bold", pad=8)
    a2.legend(loc="upper left", frameon=False, labelspacing=0.25)
    F.clean(a2)
    F.note(a2, 0.05, 0.53,
           r"$\mu_{\mathrm{eff}} = M_s\,\phi_a^2$" "\n"
           r"$\varepsilon=\sqrt{d^2+s^2}-|d|$",
           fontsize=13.0, color="0.25")

    # ---- 3. the reading decides ------------------------------------------
    oma = pd.read_csv(os.path.join(DATA, "oma.csv"))
    at = oma[(oma.d.abs() < 0.004) & (oma.snr_db == 20) & (oma.duration == 600.0)]
    zs = np.array(sorted(at.zeta.unique()))

    def rms(g, c):
        v = g[c].astype(float).to_numpy()
        v = v[np.isfinite(v)]
        return np.sqrt(np.mean(v ** 2)) if len(v) else np.nan

    for col, key, lab in (("errT_pp_pct", "incumbent", "one peak picked"),
                          ("errT_cov2_trf_pct", "model", "subspace fit"),
                          ("errT_pp_tr_pct", "law", "two peaks, trace rule")):
        y = [rms(at[at.zeta == z], col) for z in zs]
        a3.plot(100 * zs, y, label=lab,
                **F.style(key, lw=2.2, ms=6.5, label=False))
    a3.set_xscale("log")
    a3.set_xticks([0.1, 0.2, 0.5, 1.0, 2.0, 3.0])
    a3.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "3"])
    a3.set_xticks([], minor=True)
    a3.set_xlim(0.085, 3.6)
    a3.set_xlabel(r"modal damping ratio  $\zeta$  (%)")
    a3.set_ylabel("tension error  (%)")
    a3.set_title("the reading decides the error", fontsize=11.5,
                 fontweight="bold", pad=8)
    F.clean(a3)
    F.headroom(a3, top=0.34)
    a3.legend(loc="upper left", frameon=False, labelspacing=0.25)
    F.note(a3, 0.30, 0.30, "combining two peaks\nremoves most of the bias",
           fontsize=9.5, color="0.28")

    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.135, top=0.90,
                        wspace=0.30)
    out = os.path.join(ROOT, "Graphical_Abstract.png")
    fig.savefig(out, dpi=600)
    fig.savefig(out.replace('.png', '.pdf'))
    from PIL import Image
    w, h = Image.open(out).size
    print(f"  wrote {out}")
    print(f"  {w} x {h} px  (journal minimum 1328 x 531)"
          f"  {'OK' if w >= 1328 and h >= 531 else 'TOO SMALL'}")
    probs = F.audit(fig)
    if probs:
        print(f"  {len(probs)} audit problems above")


if __name__ == "__main__":
    main()
