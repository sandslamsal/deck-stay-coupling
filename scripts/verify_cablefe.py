# -*- coding: utf-8 -*-
"""Verify src/cablefe.py against closed forms and limits.

Checks V1 taut string (EI -> 0), V2 pinned-pinned tensioned beam, V3 simply
supported deck, V4 stay with T -> 0 against the plain beam, V5 mesh
convergence of the coupled system, and V6 theta -> 90 deg against the two
uncoupled spectra.

Run:  python3 scripts/verify_cablefe.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src"))

from cablefe import (CableDeck, beam_freq, chain, string_freq,  # noqa: E402
                     tensioned_beam_freq, xi_param)

from scipy.linalg import eigh  # noqa: E402


def pinned_chain_freqs(L, nel, EI, m, T, nmodes=6):
    """Frequencies of a single pinned-pinned chain, both ends held."""
    K, M = chain(L, nel, EI, m, T)
    n = K.shape[0]
    keep = [i for i in range(n) if i not in (0, n - 2)]
    w2, _ = eigh(K[np.ix_(keep, keep)], M[np.ix_(keep, keep)])
    return np.sqrt(np.maximum(w2, 0.0))[:nmodes] / (2 * np.pi)


def report(name, got, want, tol_pct):
    err = 100.0 * (got - want) / want
    worst = np.max(np.abs(err))
    ok = worst <= tol_pct
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    for i, (g, w, e) in enumerate(zip(got, want, err), start=1):
        print(f"          n={i}  fe={g:10.5f} Hz  exact={w:10.5f} Hz  "
              f"{e:+7.3f} %")
    print(f"          worst {worst:.4f} % (tolerance {tol_pct} %)")
    return ok


def main():
    ok = True
    print("=" * 72)
    print("V1  stay as a taut string, EI -> 0")
    print("=" * 72)
    L, T, m = 25.0, 400e3, 5.5
    got = pinned_chain_freqs(L, 60, 1e-6, m, T, 5)
    want = np.array([string_freq(n, L, T, m) for n in range(1, 6)])
    ok &= report("taut string", got, want, 0.5)

    print()
    print("=" * 72)
    print("V2  stay as a tensioned beam, exact pinned-pinned form")
    print("=" * 72)
    EIc = 1.2e4
    got = pinned_chain_freqs(L, 60, EIc, m, T, 5)
    want = np.array([tensioned_beam_freq(n, L, T, EIc, m) for n in range(1, 6)])
    ok &= report(f"tensioned beam (xi={xi_param(L, T, EIc):.1f})",
                 got, want, 0.5)

    print()
    print("=" * 72)
    print("V3  deck alone, simply supported beam")
    print("=" * 72)
    Ld, EId, md = 80.0, 2.0e9, 1000.0
    got = pinned_chain_freqs(Ld, 60, EId, md, 0.0, 5)
    want = np.array([beam_freq(n, Ld, EId, md) for n in range(1, 6)])
    ok &= report("simply supported beam", got, want, 0.5)

    print()
    print("=" * 72)
    print("V4  stay with T -> 0 recovers the plain beam")
    print("=" * 72)
    got = pinned_chain_freqs(L, 60, EIc, m, 1e-9, 5)
    want = np.array([beam_freq(n, L, EIc, m) for n in range(1, 6)])
    ok &= report("T -> 0", got, want, 0.5)

    print()
    print("=" * 72)
    print("V5  mesh convergence of the coupled system")
    print("=" * 72)
    ref = None
    print("   nd  nc      f1        f2        f3        f4     "
          "max drift vs finest")
    rows = []
    for nd, nc in ((10, 10), (20, 20), (40, 40), (80, 80), (160, 160)):
        cd = CableDeck(Ld=80.0, EId=2.0e9, md=1000.0,
                       Lc=25.0, EIc=1.2e4, mc=5.5, T=400e3, EA=1.4e8,
                       theta=np.deg2rad(35.0), nd=nd, nc=nc)
        f, _ = cd.modes(8)
        rows.append((nd, nc, f[:4]))
    ref = rows[-1][2]
    for nd, nc, f in rows:
        drift = 100.0 * np.max(np.abs(f - ref) / ref)
        print(f"  {nd:4d} {nc:3d}  " + "  ".join(f"{v:8.4f}" for v in f)
              + f"   {drift:8.4f} %")
    conv = 100.0 * np.max(np.abs(rows[-2][2] - ref) / ref)
    print(f"  PASS  finest two meshes agree to {conv:.4f} %"
          if conv < 0.5 else f"  FAIL  {conv:.4f} %")
    ok &= conv < 0.5

    print()
    print("=" * 72)
    print("V6  decoupling limit: theta -> 90 deg removes the transverse tie")
    print("=" * 72)
    print("      cos(theta) multiplies the tie, so a vertical stay returns")
    print("      the two uncoupled spectra.")
    cd = CableDeck(Ld=80.0, EId=2.0e9, md=1000.0,
                   Lc=25.0, EIc=1.2e4, mc=5.5, T=400e3, EA=1.4e8,
                   theta=np.deg2rad(90.0), nd=60, nc=60)
    f, _ = cd.modes(14)
    stay = cd.stay_alone(4)
    deck = cd.deck_alone(6)
    want = np.sort(np.concatenate([stay, deck]))[:8]
    got = f[:8]
    err = 100.0 * np.abs(got - want) / want
    for g, w, e in zip(got, want, err):
        print(f"      coupled={g:9.4f}  uncoupled={w:9.4f}   {e:7.4f} %")
    worst = err.max()
    print(f"  {'PASS' if worst < 0.5 else 'FAIL'}  worst {worst:.4f} %")
    ok &= worst < 0.5

    print()
    print("=" * 72)
    print("All checks pass" if ok else "One or more checks failed")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
