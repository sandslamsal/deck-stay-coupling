# -*- coding: utf-8 -*-
"""Compute the veering loci for one bridge as the stay tension is varied.

Writes data/veering_loci.csv (one row per mode per tension) and
data/veering_branches.csv (one row per tension), read by fig_veering.py.

Run: python3 scripts/run_veering.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import pandas as pd  # noqa: E402
from cablefe import (CableDeck, chain, invert_string,  # noqa: E402
                     mu_effective, tensioned_beam_freq, veering_split)
from scipy.linalg import eigh  # noqa: E402

DATA = os.path.join(ROOT, "data")

BRIDGE = dict(Ld=80.0, EId=2.0e9, md=1000.0,
              Lc=25.0, EIc=1.2e4, mc=5.5, EA=1.4e8,
              theta=np.deg2rad(35.0), nd=40, nc=40)
N_STAY = 1
BAND = (2.6, 4.2)          # the frequency window the crossing sits in


def deck_ref(cd):
    """Deck-alone frequencies and mass-normalized anchorage amplitudes."""
    Kd, Md = chain(cd.Ld, cd.nd, cd.EId, cd.md, 0.0)
    Kd = Kd.copy()
    Kd[2 * cd.ia, 2 * cd.ia] += cd.k_ax
    keep = [i for i in range(Kd.shape[0]) if i not in (0, 2 * cd.nd)]
    w2, V = eigh(Kd[np.ix_(keep, keep)], Md[np.ix_(keep, keep)])
    f = np.sqrt(np.maximum(w2, 0.0)) / (2.0 * np.pi)
    pos = {d: j for j, d in enumerate(keep)}
    return f, V[pos[2 * cd.ia], :]


def main():
    Ts = np.linspace(105e3, 215e3, 220)
    loci_rows, br_rows = [], []
    mu_best = None

    for T in Ts:
        cd = CableDeck(T=T, **BRIDGE)
        f, Phi = cd.modes(40)
        es = cd.energy_split(Phi)

        sel = (f > BAND[0]) & (f < BAND[1])
        for fv, ev in zip(f[sel], es[sel]):
            loci_rows.append(dict(T=T, f=fv, stay_frac=ev))

        fs = tensioned_beam_freq(N_STAY, BRIDGE["Lc"], T,
                                 BRIDGE["EIc"], BRIDGE["mc"])
        fd_all, phia = deck_ref(cd)
        k = int(np.argmin(np.abs(fd_all - fs)))
        if mu_best is None or abs(fs - fd_all[k]) < mu_best[0]:
            mu_best = (abs(fs - fd_all[k]),
                       mu_effective(BRIDGE["mc"] * BRIDGE["Lc"] / 2.0,
                                    phia[k]))

        # stay mode chosen by MAC against the isolated sine shape
        cdofs = np.array(cd.cable_dofs())
        x = np.linspace(0.0, 1.0, len(cdofs))
        tgt = np.sin(N_STAY * np.pi * x)
        tt = float(tgt @ tgt)
        best, bm = 0, -1.0
        for j in range(Phi.shape[1]):
            a = Phi[cdofs, j]
            den = float(a @ a) * tt
            m = (float(a @ tgt) ** 2 / den) if den > 0 else 0.0
            if m > bm:
                bm, best = m, j
        T_hat = float(invert_string(np.array([f[best]]), BRIDGE["Lc"],
                                    BRIDGE["mc"], np.array([N_STAY]))[0])

        # the two branches nearest half-and-half character, for the
        # character-exchange panel
        fv, ev = f[sel], es[sel]
        o = np.argsort(fv)
        fv, ev = fv[o], ev[o]
        c = np.sort(np.argsort(np.abs(ev - 0.5))[:2])
        br_rows.append(dict(
            T=T, f_stay_unc=fs, f_deck_unc=fd_all[k],
            d=(fs - fd_all[k]) / fs, eps=(T_hat - T) / T,
            lower=ev[c[0]] if len(c) > 1 else np.nan,
            upper=ev[c[1]] if len(c) > 1 else np.nan))

    os.makedirs(DATA, exist_ok=True)
    pd.DataFrame(loci_rows).to_csv(
        os.path.join(DATA, "veering_loci.csv"), index=False)
    br = pd.DataFrame(br_rows)
    br["mu_eff"] = mu_best[1]
    br["s_pred"] = veering_split(mu_best[1], BRIDGE["theta"], N_STAY)
    br.to_csv(os.path.join(DATA, "veering_branches.csv"), index=False)
    print(f"  wrote {DATA}/veering_loci.csv and veering_branches.csv")
    print(f"  mu_eff at the crossing = {mu_best[1]:.4e}, "
          f"predicted split = {100*br.s_pred.iloc[0]:.2f} %, "
          f"peak computed error = {100*br.eps.abs().max():.2f} %")


if __name__ == "__main__":
    main()
