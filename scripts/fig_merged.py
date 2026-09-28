# -*- coding: utf-8 -*-
"""Figure of the merged-peak law: the reading when the pair is not resolved.

Panel (a): k = |x*|/u, the fraction of the half split reached by the maximum
of H(x) = A1/(x + u + i) + A2/(x - u + i), with u = s/2 zeta and rho = A2/A1.
Panel (b): branch-law and merged-peak tension errors on the example bridge,
from data/merged.csv. Writes fig_merged.png and fig_merged.pdf to OUT.
Run: python3 scripts/fig_merged.py
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

U_SPLIT = np.sqrt(np.sqrt(5.0) - 2.0)          # 0.4859, the same root as the dip


def peak_offset(u, rho, ngrid=200001, span=3.0):
    """|x*| at the maximum of |H|^2, by dense scan.

    The scan avoids choosing among the up to five roots of the quintic for x*.
    The sign is dropped; the peak moves toward the larger-residue branch.
    """
    x = np.linspace(-span * max(u, 1.0), span * max(u, 1.0), ngrid)
    S, D = 1.0 + rho, rho - 1.0
    h2 = ((S * x + u * D) ** 2 + S ** 2) / ((x ** 2 - 1 - u ** 2) ** 2 + 4 * x ** 2)
    return abs(x[int(np.argmax(h2))])


def main():
    F.apply()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(F.FIG_W, 2.75))

    # ---- (a) how far the merged peak travels, k = x*/u -------------------
    u = np.logspace(-1, 1.0, 260)
    # |k| is invariant under rho -> 1/rho, so only rho <= 1 is drawn
    for rho, key in ((1.0, "n1"), (0.5, "n2"), (0.25, "n3"), (0.05, "n4")):
        k = np.array([peak_offset(ui, rho) / ui for ui in u])
        a1.plot(u, k, label=rf"$\rho = {rho:g}$",
                **F.style(key, marker="none", lw=1.9, label=False))
    a1.axhline(1.0, color=F.GRAY, lw=0.9, ls=(0, (2, 2)), zorder=0)
    a1.axvline(U_SPLIT, color=F.GRAY, lw=0.9, ls=(0, (1, 2)), zorder=0)
    a1.set_xscale("log")
    # explicit ticks; the default log formatter prints small exponent labels
    a1.set_xticks([0.1, 0.3, 1.0, 3.0, 10.0])
    a1.set_xticklabels(["0.1", "0.3", "1", "3", "10"])
    a1.set_xticks([], minor=True)
    a1.set_xlim(0.1, 10.0)
    a1.set_xlabel(r"$u = s/2\zeta$")
    a1.set_ylabel(r"$k = |x_*|/u$")
    a1.set_ylim(-0.05, 1.30)
    F.clean(a1)
    a1.legend(loc="lower right", fontsize=9.0, labelspacing=0.22, frameon=False)
    F.panel(a1, "a", "displacement of the merged peak")

    # ---- (b) what it costs on the example bridge --------------------------
    m = pd.read_csv(os.path.join(DATA, "merged.csv"))
    T = m.T_true.unique()
    b = m[m.zeta == m.zeta.min()]
    a2.plot(b.T_true / 1e3, b.eps_branch_pct, label="branch, resolved",
            **F.style("law", marker="none", lw=1.9, label=False))
    for z, key in ((0.005, "n3"), (0.01, "n2"), (0.02, "incumbent")):
        g = m[np.isclose(m.zeta, z)]
        a2.plot(g.T_true / 1e3, g.eps_merged_pct,
                label=rf"merged, $\zeta = {100*z:g}\,\%$",
                **F.style(key, marker="none", lw=1.9, label=False))
    a2.axhline(0, color=F.GRAY, lw=0.9, zorder=0)
    a2.set_xlabel(r"stay tension  $T$  (kN)")
    a2.set_ylabel(r"tension error  $\varepsilon$  (%)")
    F.clean(a2)
    F.headroom(a2, top=0.42)
    a2.legend(loc="upper right", fontsize=9.0, labelspacing=0.22, frameon=False)
    F.panel(a2, "b", "the resulting tension error")

    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.175, top=0.845,
                        wspace=0.30)
    os.makedirs(OUT, exist_ok=True)
    probs = F.audit(fig)
    if probs:
        print(f"  {len(probs)} audit problems above")
    png = os.path.join(OUT, "fig_merged.png")
    fig.savefig(png)
    fig.savefig(png.replace(".png", ".pdf"))
    print(f"  wrote {png}")

    # Both constants are exact at rho = 1, from x*^2 = u sqrt(u^2+4) - 1:
    # k > 1 iff 2u^2 > 1, and dk/du = 0 at u = 2/sqrt(3) where k = sqrt(5)/2.
    u_one, u_max = 1 / np.sqrt(2.0), 2 / np.sqrt(3.0)
    print(f"  k crosses 1 at u = 1/sqrt(2) = {u_one:.6f}"
          f"   (scan gives k = {peak_offset(u_one, 1.0)/u_one:.6f})")
    print(f"  k maximal   at u = 2/sqrt(3) = {u_max:.6f},"
          f"  k = sqrt(5)/2 = {np.sqrt(5)/2:.6f}"
          f"   (scan gives {peak_offset(u_max, 1.0)/u_max:.6f})")
    print(f"  peaks split at u = {U_SPLIT:.4f}, the same root as the dip")
    for z in sorted(m.zeta.unique()):
        g = m[np.isclose(m.zeta, z)]
        gm = g[~g.resolved_bool]
        print(f"  zeta = {100*z:.1f} %: worst merged {g.eps_merged_pct.abs().max():.3f} %"
              f"   (unresolved rows only: "
              f"{gm.eps_merged_pct.abs().max() if len(gm) else float('nan'):.3f} %)")
    print(f"  worst branch {m.eps_branch_pct.abs().max():.3f} %,"
          f"  worst wrong branch {m.eps_wrong_pct.abs().max():.3f} %")


if __name__ == "__main__":
    main()
