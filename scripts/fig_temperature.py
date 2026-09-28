# -*- coding: utf-8 -*-
"""Figure: apparent tension trend from a seasonal drift of the detuning.

Panel (a): detuning d over two years for stays far from, near, and at a
crossing, all with the same drift amplitude. Panel (b): the apparent tension
change each one logs at zero true tension change, eps = sqrt(d^2 + s^2) - |d|,
signed by the branch displacement. Writes fig_temperature.pdf and .png to OUT.

Run:  python3 scripts/fig_temperature.py
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import figstyle as F
import matplotlib.pyplot as plt

OUT = os.environ.get("FIGURE_DIR", os.path.join(ROOT, "figures"))

S = 0.0234           # predicted split at the crossing of the example bridge
# Seasonal detuning drift amplitude: a deck-minus-stay frequency sensitivity
# of 0.03-0.08 %/degC over a 15-25 degC half-range gives 0.005-0.025.
DRIFT = 0.01
DRIFT_BAND = (0.005, 0.025)
D0 = {"far": 0.10, "near": 0.03, "at": 0.00}
STYLE = {"far": "n5", "near": "n3", "at": "n1"}
LABEL = {"far": r"$d_0 = 0.10$", "near": r"$d_0 = 0.03$",
         "at": r"$d_0 = 0$ (at the crossing)"}

def eps_signed(d, s):
    # the stay-dominated branch moves away from the deck mode, so the
    # apparent tension error carries the sign of the detuning
    mag = np.sqrt(d * d + s * s) - np.abs(d)
    return np.where(d >= 0, mag, -mag)

def main():
    F.apply()
    t = np.linspace(0, 2, 600)                      # years
    drift = DRIFT * np.sin(2 * np.pi * t)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(F.FIG_W, 2.5))
    for k in ("far", "near", "at"):
        d = D0[k] + drift
        a1.plot(t, d, label=LABEL[k],
                **F.style(STYLE[k], marker='none', lw=1.8, label=False))
        a2.plot(t, 100 * eps_signed(d, S), label=LABEL[k],
                **F.style(STYLE[k], marker='none', lw=1.8, label=False))
    a1.axhline(0, color=F.GRAY, lw=0.8, ls=(0, (2, 2)), zorder=0)
    a1.set_xlabel("time  (years)")
    a1.set_ylabel(r"detuning $d$")
    F.clean(a1)
    F.headroom(a1, top=0.75)
    a1.legend(loc="upper right", fontsize=9.5, labelspacing=0.22)
    F.panel(a1, "a", "seasonal migration of the detuning")

    a2.set_xlabel("time  (years)")
    a2.set_ylabel(r"tension change  $\Delta T/T$  (%)")
    F.clean(a2)
    F.headroom(a2, top=0.80)
    a2.legend(loc="upper right", fontsize=9.5, labelspacing=0.22)
    F.panel(a2, "b", "the apparent tension logged")

    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.19, top=0.85,
                        wspace=0.30)
    os.makedirs(OUT, exist_ok=True)
    probs = F.audit(fig)
    if probs: print(f"  {len(probs)} audit problems above")
    png = os.path.join(OUT, "fig_temperature.png")
    fig.savefig(png); fig.savefig(png.replace(".png", ".pdf"))
    print(f"  wrote {png}")
    ptp = lambda e: 100 * (e.max() - e.min())
    for k in ("at", "near", "far"):
        e = eps_signed(D0[k] + drift, S)
        print(f"  {k:5s} stay (d0 = {D0[k]:.2f}): {ptp(e):6.3f} % peak to peak"
              f"  with zero true tension change")

    # The at-crossing swing is 2s for any amplitude that carries d through
    # zero; check it over a range of amplitudes.
    print(f"\n  invariance of the at-crossing swing to the drift amplitude"
          f"  (2s = {200*S:.3f} %)")
    for A in (0.0005, 0.005, 0.01, 0.025, 0.05, 0.20):
        e = eps_signed(A * np.sin(2 * np.pi * t), S)
        band = "  <- band" if DRIFT_BAND[0] <= A <= DRIFT_BAND[1] else ""
        print(f"    A = {A:6.4f}:  {ptp(e):6.3f} %{band}")

    # Same stay, tracking one continuous spectral ridge instead of the
    # stay-dominated branch: no sign reversal.
    cont = np.sqrt((0.0 + drift) ** 2 + S ** 2) - np.abs(0.0 + drift)
    print(f"\n  same stay, one continuous ridge tracked instead:"
          f" {100*(cont.max()-cont.min()):.3f} % peak to peak")

if __name__ == "__main__":
    main()
