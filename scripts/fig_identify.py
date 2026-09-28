# -*- coding: utf-8 -*-
"""Figure: stay tension identification through the frequency crossing.

(a) noise-free tension error against stay tension, four methods.
(b) tension RMSE under shape measurement noise.

Reads data/identify2.csv written by scripts/run_identify2.py and writes
fig_identify.png and fig_identify.pdf to the figures directory OUT.
Run:  python3 scripts/fig_identify.py
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

OUT = os.environ.get("FIGURE_DIR", os.path.join(ROOT, "figures"))

EST = {
    "string_n1": dict(key="incumbent", label="taut string, order 1"),
    "multi_iso": dict(key="n3", label=r"multi-mode fit, $V(L_c)=0$"),
    "shapefit": dict(key="law", label="shape fit"),
    "pinn_free": dict(key="model", label="network, free end"),
}


def main():
    F.apply()
    d = pd.read_csv(os.path.join(ROOT, "data", "identify2.csv"))
    main9 = d[d.slice < 9]
    heavy = d[d.slice == 9]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(F.FIG_W, 2.6))

    # ---- (a) noise-free error through the crossing -----------------------
    nf = main9[main9.rep < 0]
    for est, spec in EST.items():
        g = nf[nf.estimator == est].groupby("T_true").err_pct.median()  # median over initializations
        ax1.plot(g.index / 1e3, g.values, label=spec["label"],
                 **F.style(spec["key"], lw=1.7, ms=4.0, label=False))
    hn = heavy[(heavy.rep < 0) & (heavy.estimator == "string_n1")]
    ax1.axhline(0, color=F.GRAY, lw=0.7, zorder=0)
    ax1.set_xlabel(r"stay tension  $T$  (kN)")
    ax1.set_ylabel(r"tension error  $\varepsilon$  (%)")
    F.clean(ax1)
    F.headroom(ax1, top=0.62)
    ax1.legend(loc="upper right", fontsize=9.6, labelspacing=0.22,
               borderaxespad=0.15)
    F.panel(ax1, "a", "noise-free records")

    # ---- (b) RMSE under noise -------------------------------------------
    noisy = main9[main9.rep >= 0]
    sig_levels = sorted(noisy.sigma_s.unique())
    width = 0.24
    xpos = np.arange(len(sig_levels))
    for j, est in enumerate(("multi_iso", "shapefit")):
        rm = [np.sqrt((noisy[(noisy.estimator == est)
                             & (noisy.sigma_s == s)].err_pct ** 2).mean())
              for s in sig_levels]
        spec = EST[est]
        ax2.bar(xpos + (j - 0.5) * width, rm, width * 0.92,
                color=F.color(spec["key"]), label=spec["label"],
                edgecolor='black', linewidth=0.5)
    # network estimate at its single tested noise level
    pn = main9[(main9.estimator == "pinn_free") & (main9.rep >= 0)]
    if len(pn):
        s0 = pn.sigma_s.iloc[0]
        if s0 in sig_levels:
            xj = sig_levels.index(s0)
            ax2.plot([xj + 1.1 * width],
                     [np.sqrt((pn.err_pct ** 2).mean())],
                     marker='o', ms=5, color=F.color("model"), ls='none',
                     label="network, free end")
    ax2.set_xticks(xpos)
    ax2.set_xticklabels([f"{100*s:.1f}" for s in sig_levels])
    ax2.set_xlabel("shape noise, % of peak amplitude")
    ax2.set_ylabel("tension RMSE  (%)")
    F.clean(ax2)
    ax2.legend(loc="upper left", fontsize=9.6, labelspacing=0.22,
               borderaxespad=0.15)
    F.panel(ax2, "b", "under measurement noise")

    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.185, top=0.85,
                        wspace=0.30)
    os.makedirs(OUT, exist_ok=True)
    probs = F.audit(fig)
    if probs:
        print(f"  {len(probs)} audit problems above")
    png = os.path.join(OUT, "fig_identify.png")
    fig.savefig(png)
    fig.savefig(png.replace(".png", ".pdf"))
    print(f"  wrote {png}")
    if len(hn):
        print(f"  heavy-stay taut-string noise-free error: "
              f"{hn.err_pct.iloc[0]:+.2f} %")


if __name__ == "__main__":
    main()
