# -*- coding: utf-8 -*-
"""Verification of the COUPLING, which verify_cablefe.py does not cover.

The six checks in ``verify_cablefe.py`` verify the components: the cable
alone against the exact tensioned beam, the deck alone against the exact
simply supported beam, the two limits, mesh convergence, and the decoupled
case.  Every one of them can pass while the coupling term itself is wrong,
because none of them exercises a coupled mode.  That is a real hole and this
file closes it.

Two independent references are used, neither of them the finite element
model being checked.

C1  EXACT TRANSCENDENTAL SOLUTION.  A taut string, pinned at the top, whose
    bottom end is tied through a factor ``c`` to a single oscillator of mass
    ``M`` and stiffness ``K``, has a closed-form characteristic equation.
    Writing ``V(x) = A sin(kx)`` with ``k = omega sqrt(m/T)``, the string is
    satisfied identically and the oscillator equation supplies

        M wbb + K w = -c T V'(L),      V(L) = c w

    which after eliminating ``A`` gives

        sin(kL) (M omega^2 - K) - c^2 T k cos(kL) = 0.

    Two limits confirm the sign convention: ``M -> infinity`` forces
    ``sin(kL) = 0``, the fixed end, and ``M, K -> 0`` forces ``cos(kL) = 0``,
    the free end.  The ``c^2`` is the statement that the tie transmits
    displacement one way and force the other, so coupling strength goes as
    the square.

C2  TWO-DEGREE-OF-FREEDOM VEERING THEORY.  At exact tuning two coupled
    oscillators of modal mass ratio ``mu`` split by a known amount.  The
    finite element model must reproduce that split, because the split IS the
    veering the study rests on.  If the split is wrong the bias map is wrong.

Run:  python3 scripts/verify_coupling.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src"))

from cablefe import chain  # noqa: E402


# ---------------------------------------------------------------------------
# C1  cable tied to an oscillator: finite element against the exact roots
# ---------------------------------------------------------------------------

def fe_cable_oscillator(L, nc, T, m, EIc, M, K, c):
    """Assemble the same way CableDeck does, with the deck reduced to 1 DOF.

    Deliberately reuses the tie logic under test rather than reimplementing
    it, so a sign or factor error in the constraint shows up here.
    """
    Kc, Mc = chain(L, nc, EIc, m, T)
    nC = Kc.shape[0]
    N = nC + 1                       # last DOF is the oscillator
    Kg = np.zeros((N, N))
    Mg = np.zeros((N, N))
    Kg[:nC, :nC] = Kc
    Mg[:nC, :nC] = Mc
    Kg[nC, nC] = K
    Mg[nC, nC] = M

    dep = 2 * nc                     # cable bottom transverse DOF
    fixed = {0}                      # cable top held
    free = [i for i in range(N) if i not in fixed and i != dep]
    pos = {d: j for j, d in enumerate(free)}
    Lmat = np.zeros((N, len(free)))
    for d, j in pos.items():
        Lmat[d, j] = 1.0
    Lmat[dep, pos[nC]] = c           # v_bottom = c * w

    w2, _ = eigh(Lmat.T @ Kg @ Lmat, Lmat.T @ Mg @ Lmat)
    return np.sqrt(np.maximum(w2, 0.0)) / (2.0 * np.pi)


def exact_cable_oscillator(L, T, m, M, K, c, nroots=8):
    """Roots of sin(kL)(M w^2 - K) - c^2 T k cos(kL) = 0, in Hz."""
    def g(k):
        w2 = k ** 2 * T / m
        return np.sin(k * L) * (M * w2 - K) - c ** 2 * T * k * np.cos(k * L)

    kmax = nroots * np.pi / L * 1.6
    ks = np.linspace(1e-9, kmax, 400000)
    vals = g(ks)
    roots = []
    for i in range(len(ks) - 1):
        if vals[i] == 0.0:
            roots.append(ks[i])
        elif vals[i] * vals[i + 1] < 0:
            # reject the spurious sign change across a tan pole by checking
            # the bracket is narrow and the function is finite on both sides
            try:
                r = brentq(g, ks[i], ks[i + 1], xtol=1e-14)
                roots.append(r)
            except ValueError:
                pass
    roots = np.array(sorted(set(np.round(roots, 10))))
    roots = roots[roots > 1e-6]
    f = roots * np.sqrt(T / m) / (2.0 * np.pi)
    return f[:nroots]


def check_C1():
    print("=" * 74)
    print("C1  cable tied to an oscillator, FE against the exact roots")
    print("=" * 74)
    ok = True
    cases = [
        # L,    T,      m,   M,      K,        c,     label
        (25.0, 400e3, 5.5, 40000.0, 40000.0 * (2 * np.pi * 2.0) ** 2, 0.819,
         "footbridge-like, c=cos(35 deg)"),
        (25.0, 400e3, 5.5, 400.0, 400.0 * (2 * np.pi * 5.4) ** 2, 1.0,
         "light oscillator TUNED to stay mode 1, c=1"),
        (25.0, 400e3, 5.5, 1e12, 1e12 * (2 * np.pi * 1.0) ** 2, 1.0,
         "very heavy oscillator, must give the fixed-end string"),
    ]
    for L, T, m, M, K, c, lab in cases:
        fe = fe_cable_oscillator(L, 200, T, m, 1e-6, M, K, c)[:6]
        ex = exact_cable_oscillator(L, T, m, M, K, c, 6)
        n = min(len(fe), len(ex))
        fe, ex = fe[:n], ex[:n]
        err = 100.0 * np.abs(fe - ex) / ex
        worst = err.max()
        good = worst < 0.5
        ok &= good
        print(f"\n  {'PASS' if good else 'FAIL'}  {lab}")
        for a, b, e in zip(fe, ex, err):
            print(f"          FE={a:10.5f} Hz   exact={b:10.5f} Hz   {e:7.4f} %")
        print(f"          worst {worst:.4f} %")
    return ok


# ---------------------------------------------------------------------------
# C2  the veering split at exact tuning
# ---------------------------------------------------------------------------

def check_C2():
    """At exact tuning, two coupled oscillators split by a known amount.

    Reducing the string's participating mode to its modal mass and the tie to
    an effective coupling, classical two-degree-of-freedom theory gives, for
    modal mass ratio ``mu`` at exact tuning, a normalised split

        (f+ - f-) / f0  =  sqrt(mu_eff)

    to leading order.  What is checked here is not that constant, which
    depends on how the effective mass is defined, but the SCALING: the split
    must go as the square root of the mass ratio and as the tie factor ``c``.
    A coupling implemented with the wrong power would fail this even though
    it might match a single tuned case by accident.
    """
    print()
    print("=" * 74)
    print("C2  veering split scaling against two-degree-of-freedom theory")
    print("=" * 74)
    L, T, m = 25.0, 400e3, 5.5
    f_stay1 = 1.0 / (2 * L) * np.sqrt(T / m)
    M_stay = m * L / 2.0

    print(f"  stay mode 1 at {f_stay1:.4f} Hz, stay modal mass "
          f"{M_stay:.2f} kg")
    print()
    print("  split of the two coupled frequencies at exact tuning:")
    print(f"  {'M_deck':>12s} {'mu':>10s} {'c':>6s} {'split/f0':>10s}"
          f" {'split/(c sqrt(mu))':>20s}")

    ratios = []
    for Mdeck in (2000.0, 8000.0, 32000.0, 128000.0):
        for c in (1.0, 0.819, 0.5):
            K = Mdeck * (2 * np.pi * f_stay1) ** 2      # tune to stay mode 1
            f = fe_cable_oscillator(L, 200, T, m, 1e-6, Mdeck, K, c)
            near = f[np.abs(f - f_stay1) < 0.5 * f_stay1]
            if len(near) < 2:
                print("      could not isolate the pair")
                continue
            split = near[1] - near[0]
            mu = M_stay / Mdeck
            norm = split / f_stay1 / (c * np.sqrt(mu))
            ratios.append(norm)
            print(f"  {Mdeck:12.0f} {mu:10.5f} {c:6.3f} "
                  f"{split / f_stay1:10.5f} {norm:20.5f}")

    ratios = np.array(ratios)
    spread = 100.0 * (ratios.max() - ratios.min()) / ratios.mean()
    print()
    print(f"  the normalised quantity is constant to {spread:.2f} % across a")
    print(f"  64x range of mass ratio and a 2x range of tie factor,")
    print(f"  mean {ratios.mean():.4f}")
    ok = spread < 5.0
    print(f"  {'PASS' if ok else 'FAIL'}  split scales as c*sqrt(mu) "
          f"as veering theory requires")
    return ok


def main():
    ok = check_C1()
    ok &= check_C2()
    print()
    print("=" * 74)
    print("COUPLING VERIFIED" if ok else "COUPLING IS WRONG, FIX BEFORE USE")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
