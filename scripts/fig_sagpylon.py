# -*- coding: utf-8 -*-
"""Figure: the two idealisations the reviewers questioned, computed.

Panel (a): the second route. The veering width of the two-dimensional
model, drawn along the stay's sagged profile, relative to the straight-chord
law of Eq. (5), against the ratio r of the elastic route to the transverse
tie. On the worked bridge the sag is raised alone (gravity scaled,
everything else held) at the fundamental; the line is |1 - r|. Reads
data/sag.csv (scripts/run_sag.py) and the direct gravity checks.

Panel (b): the campaign re-solved with the sag drawn at each design's own
gravity: the coupling error of the incumbent reading against the extended
law, marked by the size of r. Reads data/campaign_sag.csv
(scripts/run_campaign_sag.py).

Panel (c): pylon. The width with a pylon of the stay's height whose sway
frequency is scanned across the host mode the stay meets, relative to the
rigid-pylon law: the finite element against the two-ended reduction, on the
worked bridge and on the longest-deck campaign design. Reads
data/pylon.csv (scripts/run_pylon.py).

Run:  python3 scripts/fig_sagpylon.py
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
OUT = os.path.join(ROOT, "revision 1", "sources", "figures")
G = 9.80665


def r_of(theta, EA, T, mc, g, Lc, n=1):
    return 2 / (n ** 2 * np.pi ** 2) * np.sin(theta) * (EA / T) * (mc * g * Lc / T)


def main():
    F.apply()
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(F.FIG_W, 2.55))

    # ---- (a) the second route on the worked bridge ------------------------
    s = pd.read_csv(os.path.join(DATA, "sag.csv"))
    w = s[((s.case == "worked") & (s.n == 1) & (s.tension_variation == 0))
          | (s.case == "worked_gmult")].copy()
    theta, EA, mc, Lc = np.deg2rad(35.0), 1.4e8, 5.5, 25.0
    w["r"] = r_of(theta, EA, w.T_at, mc, w.g, Lc)
    w = w.sort_values("r")
    rr = np.linspace(0, 8.5, 400)
    a1.plot(rr, np.abs(1 - rr), color=F.GREEN, ls="--", lw=1.8, label=r"$|1 - r_1|$")
    a1.plot(w.r, w.ratio, ls="none", **{k: v for k, v in F.style("n1", label=False).items()
                                         if k in ("color", "marker")}, ms=5.5,
            label="example bridge")
    a1.set_xlabel(r"second-route ratio  $r_1$")
    a1.set_ylabel(r"width ratio  $s / s_{\mathrm{straight}}$")
    a1.set_xlim(-0.2, 8.6)
    F.clean(a1)
    F.headroom(a1, top=0.30)
    a1.legend(loc="upper left", fontsize=9.3, labelspacing=0.22, frameon=False)
    F.panel(a1, "a", "the second route")

    # ---- (b) the sagged campaign against the extended law -----------------
    c = pd.read_csv(os.path.join(DATA, "campaign_sag.csv"))
    g = c[(c.mac > 0.5) & (c.xi > 150) & (c.share > 0.5) & c.eps_coupling_sag.notna()].copy()
    g["y"] = 100 * g.eps_coupling_sag.abs()
    g["x"] = 100 * g.eps_pred_ext
    bins = ((r"even order, $r_n = 0$", g[g.n_stay % 2 == 0], "n2"),
            (r"odd, $r_n < 0.3$", g[(g.n_stay % 2 == 1) & (g.r_n < 0.3)], "n1"),
            (r"odd, $0.3 \leq r_n < 3$", g[(g.n_stay % 2 == 1) & (g.r_n >= 0.3) & (g.r_n < 3)], "n3"),
            (r"odd, $r_n \geq 3$", g[(g.n_stay % 2 == 1) & (g.r_n >= 3)], "n5"))
    lim = 6.0
    a2.plot([0, lim], [0, lim], color=F.GRAY, lw=0.9, zorder=0)
    for lab, sub, key in bins:
        a2.plot(sub.x.clip(upper=lim), sub.y.clip(upper=lim), ls="none",
                **{k: v for k, v in F.style(key, label=False).items() if k in ("color", "marker")},
                ms=4.0, mfc="none", mew=0.9, label=lab)
    a2.set_xlim(0, lim)
    a2.set_ylim(0, lim)
    a2.set_xlabel(r"law with sag  $\varepsilon$  (%)")
    a2.set_ylabel(r"sagged model  $\varepsilon$  (%)")
    F.clean(a2)
    a2.legend(loc="upper left", fontsize=9.3, labelspacing=0.2, frameon=False, handletextpad=0.3)
    F.panel(a2, "b", "the sagged design set")

    # ---- (c) pylon --------------------------------------------------------
    p = pd.read_csv(os.path.join(DATA, "pylon.csv"))
    sets = (("worked_pylon", 1, "n1", "example bridge"),
            ("longspan_pylon", 1, "n3", "560 m design"))
    for case, n, key, lab in sets:
        q = p[(p.case == case) & (p.n == n)].copy()
        q["x"] = q.f_pylon / q.f_host
        q = q.sort_values("x")
        a3.plot(q.x, q.s_two / q.s_one, **F.style(key, marker="none", lw=1.8, label=False))
        a3.plot(q.x, q.s_fe / q.s_one, ls="none",
                **{k: v for k, v in F.style(key, label=False).items() if k in ("color", "marker")},
                ms=5.0, label=lab)
    a3.axhline(1.0, color=F.GRAY, lw=0.9, zorder=0)
    a3.axvline(1.0, color=F.GRAY, lw=0.8, ls=(0, (2, 2)), zorder=0)
    a3.set_xscale("log")
    a3.set_xticks([0.5, 1, 2, 5])
    a3.set_xticklabels(["0.5", "1", "2", "5"])
    a3.minorticks_off()
    a3.set_xlabel(r"sway frequency ratio  $f_p / f_h$")
    a3.set_ylabel(r"width ratio  $s / s_{\mathrm{rigid}}$")
    F.clean(a3)
    F.headroom(a3, top=0.30)
    handles = [Line2D([], [], color="0.3", ls="-", lw=1.8, label="two-ended reduction"),
               Line2D([], [], color="0.3", ls="none", marker="o", ms=5.0, label="finite element")]
    handles += [Line2D([], [], color=F.color(k), ls="none", marker=F.ENTITY[k]["marker"], ms=5.0, label=l)
                for _, _, k, l in sets]
    a3.legend(handles=handles, loc="upper right", fontsize=7.4, labelspacing=0.2, frameon=False)
    F.panel(a3, "c", "pylon flexibility")

    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.19, top=0.845, wspace=0.42)
    os.makedirs(OUT, exist_ok=True)
    probs = F.audit(fig)
    if probs:
        print(f"  {len(probs)} audit problems above")
    png = os.path.join(OUT, "fig_sagpylon.png")
    fig.savefig(png)
    fig.savefig(png.replace(".png", ".pdf"))
    print(f"  wrote {png}")
    print("  worked bridge points (r, ratio):", ", ".join(f"({a:.2f}, {b:.2f})" for a, b in zip(w.r, w.ratio)))


if __name__ == "__main__":
    main()
