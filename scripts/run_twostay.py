# -*- coding: utf-8 -*-
"""Two stays on one deck: veering width against the single-stay law
s = (2/(n pi)) cos(theta) sqrt(mu_eff), with mu_eff from the deck-alone mode
carrying both axial springs.

Stay A (at 0.30 L_d) sweeps 120 to 200 kN with an identical stay B (at 0.55 L_d)
(a) detuned at 400 kN and (b) tuned to the deck mode A crosses. With both stays
tuned, the outer split of the three coupled modes is compared with
sqrt(s_A^2 + s_B^2). The assembly is checked against CableDeck first.
Writes data/twostay.csv.  Run:  python3 scripts/run_twostay.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from cablefe import (CableDeck, chain, mu_effective,  # noqa: E402
                     tensioned_beam_freq, veering_split)

DATA = os.path.join(ROOT, "data")

DECK = dict(Ld=80.0, EId=2.0e9, md=1000.0, nd=40)
STAY = dict(Lc=25.0, EIc=1.2e4, mc=5.5, EA=1.4e8,
            theta=np.deg2rad(35.0), nc=40)
XFRAC_A, XFRAC_B = 0.30, 0.55
T_B_DETUNED = 400e3
T_SWEEP = np.linspace(120e3, 200e3, 161)
BAND = (2.4, 6.5)               # Hz window written to the CSV
WINDOW = 0.15                   # relative window used to isolate hybrids
N_STAY = 1


def stay(xfrac, T, EA=STAY["EA"], theta=STAY["theta"]):
    """Stay parameters for TwoStayDeck: anchorage at xfrac L_d, tension T (N)."""
    return dict(Lc=STAY["Lc"], EIc=STAY["EIc"], mc=STAY["mc"],
                nc=STAY["nc"], EA=EA, theta=theta, T=T, xfrac=xfrac)


class TwoStayDeck:
    """One deck with several stays, assembled as in ``CableDeck._assemble``.

    Each stay adds its axial spring ``EA/Lc sin^2(theta)`` at its anchorage
    node and its transverse tie ``v_bottom = cos(theta) w_deck(x_anchor)``
    through one dependent DOF in the constraint matrix.
    """

    def __init__(self, Ld, EId, md, nd, stays):
        self.Ld, self.EId, self.md, self.nd = Ld, EId, md, nd
        self.stays = [dict(s) for s in stays]
        self._assemble()

    def _assemble(self):
        Kd, Md = chain(self.Ld, self.nd, self.EId, self.md, 0.0)
        blocks = [(Kd, Md)]
        offs = [0]
        for s in self.stays:
            Kc, Mc = chain(s["Lc"], s["nc"], s["EIc"], s["mc"], s["T"])
            offs.append(offs[-1] + blocks[-1][0].shape[0])
            blocks.append((Kc, Mc))
        N = offs[-1] + blocks[-1][0].shape[0]
        K = np.zeros((N, N))
        M = np.zeros((N, N))
        for (Kb, Mb), off in zip(blocks, offs):
            nb = Kb.shape[0]
            K[off:off + nb, off:off + nb] = Kb
            M[off:off + nb, off:off + nb] = Mb

        # anchorages snapped to deck nodes, axial springs added there
        for s in self.stays:
            ia = int(round(s["xfrac"] * self.nd))
            s["ia"] = ia
            s["x_anchor"] = ia * self.Ld / self.nd
            s["k_ax"] = s["EA"] / s["Lc"] * np.sin(s["theta"]) ** 2
            K[2 * ia, 2 * ia] += s["k_ax"]

        # constraints: deck simply supported, stay tops held at the pylon,
        # stay bottoms slaved to the deck through cos(theta)
        fixed = {0, 2 * self.nd}
        deps = {}
        for s, off in zip(self.stays, offs[1:]):
            s["off"] = off
            fixed.add(off)
            deps[off + 2 * s["nc"]] = (2 * s["ia"], np.cos(s["theta"]))

        free = [i for i in range(N)
                if i not in fixed and i not in deps]
        pos = {d: j for j, d in enumerate(free)}
        Lmat = np.zeros((N, len(free)))
        for d, j in pos.items():
            Lmat[d, j] = 1.0
        for dep, (indep, coeff) in deps.items():
            Lmat[dep, pos[indep]] = coeff

        self.N, self.offs = N, offs
        self.Lmat = Lmat
        self.K = Lmat.T @ K @ Lmat
        self.M = Lmat.T @ M @ Lmat
        self._Mfull = M                     # block-diagonal, for energies

    def modes(self, nmodes=40):
        """Mass-normalized modes, full-DOF shapes, f in Hz ascending."""
        w2, V = eigh(self.K, self.M)
        w2 = np.maximum(w2, 0.0)
        f = np.sqrt(w2) / (2.0 * np.pi)
        k = min(nmodes, len(f))
        return f[:k], (self.Lmat @ V[:, :k])

    def energy_fractions(self, Phi):
        """Kinetic-energy fractions per component, rows (deck, stay1, ...)."""
        out = []
        bounds = list(self.offs) + [self.N]
        for a, b in zip(bounds[:-1], bounds[1:]):
            P = Phi[a:b, :]
            out.append(np.einsum("ij,ij->j", P, self._Mfull[a:b, a:b] @ P))
        out = np.array(out)
        tot = out.sum(axis=0)
        return out / np.where(tot > 0, tot, 1.0)

    def deck_alone(self, nmodes=12):
        """Deck alone with every stay's axial spring, mass-normalized.

        Returns ``(f, phi)``, ``phi[i, k]`` the mode-k amplitude at the
        anchorage of stay ``i``.
        """
        Kd, Md = chain(self.Ld, self.nd, self.EId, self.md, 0.0)
        Kd = Kd.copy()
        for s in self.stays:
            Kd[2 * s["ia"], 2 * s["ia"]] += s["k_ax"]
        keep = [i for i in range(Kd.shape[0])
                if i not in (0, 2 * self.nd)]
        w2, V = eigh(Kd[np.ix_(keep, keep)], Md[np.ix_(keep, keep)])
        f = np.sqrt(np.maximum(w2, 0.0)) / (2.0 * np.pi)
        pos = {d: j for j, d in enumerate(keep)}
        phi = np.array([V[pos[2 * s["ia"]], :] for s in self.stays])
        return f[:nmodes], phi[:, :nmodes]


# --- verification ---

def verify_assembly():
    """Check the two-stay assembly against CableDeck; True if all checks pass."""
    print("=" * 74)
    print("V   two-stay assembly against the verified single-stay model")
    print("=" * 74)
    ok = True

    # V1: stay B decoupled (theta_B = 90 deg zeroes the tie, EA_B = 0 removes
    # the axial spring); with stay B's own modes filtered out by energy, the
    # spectrum must match CableDeck for stay A alone
    T_A = 150e3
    ref = CableDeck(Ld=DECK["Ld"], EId=DECK["EId"], md=DECK["md"],
                    Lc=STAY["Lc"], EIc=STAY["EIc"], mc=STAY["mc"],
                    T=T_A, EA=STAY["EA"], theta=STAY["theta"],
                    x_anchor=XFRAC_A * DECK["Ld"],
                    nd=DECK["nd"], nc=STAY["nc"])
    f_ref, _ = ref.modes(16)
    two = TwoStayDeck(stays=[stay(XFRAC_A, T_A),
                             stay(XFRAC_B, 397e3, EA=0.0,
                                  theta=np.pi / 2.0)], **DECK)
    f2, Phi2 = two.modes(80)
    e2 = two.energy_fractions(Phi2)
    fk = f2[e2[2] < 0.5][:len(f_ref)]
    dev = float(np.max(np.abs(fk - f_ref) / f_ref))
    good = dev < 1e-3
    ok &= good
    print(f"  V1 decoupled stay B: worst deviation from CableDeck over "
          f"{len(f_ref)} modes = {100 * dev:.2e} %  "
          f"{'PASS' if good else 'FAIL'} (< 0.1 %)")

    # V2: stay order must not matter
    a = TwoStayDeck(stays=[stay(XFRAC_A, 150e3),
                           stay(XFRAC_B, 400e3)], **DECK)
    b = TwoStayDeck(stays=[stay(XFRAC_B, 400e3),
                           stay(XFRAC_A, 150e3)], **DECK)
    fa, _ = a.modes(40)
    fb, _ = b.modes(40)
    drec = float(np.max(np.abs(fa - fb) / fa))
    good = drec < 1e-9
    ok &= good
    print(f"  V2 stay-order reciprocity: worst deviation = {drec:.2e}  "
          f"{'PASS' if good else 'FAIL'}")

    # V3: component energy fractions must partition unity
    f3, Phi3 = a.modes(40)
    esum = a.energy_fractions(Phi3).sum(axis=0)
    dsum = float(np.max(np.abs(esum - 1.0)))
    good = dsum < 1e-9
    ok &= good
    print(f"  V3 energy fractions sum to one within {dsum:.2e}  "
          f"{'PASS' if good else 'FAIL'}")
    return ok


# --- hybrid extraction ---

def hybrid_pair(f, eA, f_dk):
    """Frequencies of the two modes near the deck mode with the most stay-A energy."""
    cand = np.where((np.abs(f / f_dk - 1.0) < WINDOW) & (eA > 0.02))[0]
    if len(cand) < 2:
        return None
    top = cand[np.argsort(eA[cand])[-2:]]
    return np.sort(f[top])


def hybrid_triplet(f, eA, eB, f_dk):
    """Indices of the three modes near the deck mode with the most stay energy."""
    cand = np.where((np.abs(f / f_dk - 1.0) < WINDOW)
                    & (eA + eB > 0.02))[0]
    if len(cand) < 3:
        return None
    top = cand[np.argsort((eA + eB)[cand])[-3:]]
    return np.sort(top)


def solve(T_A, T_B):
    """Frequencies and energy fractions (deck, A, B) for stay tensions T_A, T_B."""
    two = TwoStayDeck(stays=[stay(XFRAC_A, T_A),
                             stay(XFRAC_B, T_B)], **DECK)
    f, Phi = two.modes(60)
    e = two.energy_fractions(Phi)
    return f, e


# --- main study ---

def main():
    if not verify_assembly():
        print("ASSEMBLY CHECK FAILED")
        return 1

    Lc, EIc, mc = STAY["Lc"], STAY["EIc"], STAY["mc"]
    theta = STAY["theta"]

    # --- deck-alone reference with both axial springs ---
    two0 = TwoStayDeck(stays=[stay(XFRAC_A, 150e3),
                              stay(XFRAC_B, T_B_DETUNED)], **DECK)
    fd, phi = two0.deck_alone(10)
    f1_lo = tensioned_beam_freq(N_STAY, Lc, T_SWEEP[0], EIc, mc)
    f1_hi = tensioned_beam_freq(N_STAY, Lc, T_SWEEP[-1], EIc, mc)

    print()
    print("=" * 74)
    print("D   deck-alone modes (both axial springs), anchorage amplitudes")
    print("=" * 74)
    print(f"  stay A mode 1 spans {f1_lo:.3f} to {f1_hi:.3f} Hz")
    for k in range(len(fd)):
        mark = "  <-- within stay A range" if f1_lo < fd[k] < f1_hi else ""
        print(f"  mode {k + 1}: {fd[k]:8.4f} Hz   phi(x_A)={phi[0, k]:+.5f}"
              f"   phi(x_B)={phi[1, k]:+.5f}{mark}")
    cand = np.where((fd > f1_lo) & (fd < f1_hi))[0]
    j = int(cand[np.argmax(np.abs(phi[0, cand]))])
    f_dk, phiA, phiB = float(fd[j]), float(phi[0, j]), float(phi[1, j])

    # --- predictions ---
    M_stay = mc * Lc / 2.0
    mu_A = mu_effective(M_stay, phiA)
    mu_B = mu_effective(M_stay, phiB)
    s_A = veering_split(mu_A, theta, N_STAY)
    s_B = veering_split(mu_B, theta, N_STAY)
    s_AB = float(np.hypot(s_A, s_B))

    k1 = np.pi / Lc
    om = 2.0 * np.pi * f_dk
    T_star = (mc * om ** 2 - EIc * k1 ** 4) / k1 ** 2

    print()
    print(f"  crossed deck mode: mode {j + 1} at f_dk = {f_dk:.4f} Hz")
    print(f"  M_stay = {M_stay:.2f} kg")
    print(f"  mu_eff_A = {mu_A:.4e}   s_A = {100 * s_A:.4f} %")
    print(f"  mu_eff_B = {mu_B:.4e}   s_B = {100 * s_B:.4f} %")
    print(f"  sqrt(s_A^2 + s_B^2) = {100 * s_AB:.4f} %")
    print(f"  exact-tuning tension T* = {T_star / 1e3:.2f} kN")

    rows = []

    def record(case, T_A, T_B, f, e):
        sel = (f >= BAND[0]) & (f <= BAND[1])
        fsA = tensioned_beam_freq(N_STAY, Lc, T_A, EIc, mc)
        fsB = tensioned_beam_freq(N_STAY, Lc, T_B, EIc, mc)
        for jj in np.where(sel)[0]:
            rows.append(dict(case=case, T_A=T_A, T_B=T_B, f=float(f[jj]),
                             e_deck=float(e[0, jj]), e_A=float(e[1, jj]),
                             e_B=float(e[2, jj]), f_sA=fsA, f_sB=fsB,
                             f_deck=f_dk))

    # --- case (a): stay B detuned at 400 kN ---
    print()
    print("=" * 74)
    print("A   stay A tension varied, stay B detuned at 400 kN")
    print("=" * 74)
    for T_A in T_SWEEP:
        f, e = solve(T_A, T_B_DETUNED)
        record("detuned", float(T_A), T_B_DETUNED, f, e)

    # measured split: gap at T*, and the minimum gap over a fine scan
    # (stay B's tie shifts the crossing slightly off T*)
    f, e = solve(T_star, T_B_DETUNED)
    pr = hybrid_pair(f, e[1], f_dk)
    s_at_star = float((pr[1] - pr[0]) / f_dk)
    Tf = np.linspace(T_star - 10e3, T_star + 10e3, 401)
    best = None
    for T_A in Tf:
        f, e = solve(float(T_A), T_B_DETUNED)
        pr = hybrid_pair(f, e[1], f_dk)
        if pr is None:
            continue
        gap = pr[1] - pr[0]
        if best is None or gap < best[0]:
            best = (float(gap), float(T_A), 0.5 * float(pr[0] + pr[1]))
    gap_min, T_min, f_mid = best
    s_meas = gap_min / f_mid
    if not (Tf[0] + 1.0 < T_min < Tf[-1] - 1.0):
        print("  WARNING: minimum gap sits at the edge of the fine scan")
    print(f"  pair gap at T*                : {100 * s_at_star:.4f} %")
    print(f"  minimum gap, fine scan        : {100 * s_meas:.4f} % "
          f"at T_A = {T_min / 1e3:.2f} kN (f_mid = {f_mid:.4f} Hz)")
    print(f"  predicted s_A                 : {100 * s_A:.4f} %")
    print(f"  measured / predicted          : {s_meas / s_A:.4f}  "
          f"({100 * (s_meas / s_A - 1):+.2f} %)")

    # --- case (b): stay B tuned to the crossed deck mode ---
    print()
    print("=" * 74)
    print("B   stay A tension varied, stay B tuned to the same deck mode "
          f"(T_B = {T_star / 1e3:.2f} kN)")
    print("=" * 74)
    for T_A in T_SWEEP:
        f, e = solve(T_A, T_star)
        record("tuned", float(T_A), float(T_star), f, e)

    # --- both stays at exact tuning: three coupled modes ---
    f, e = solve(T_star, T_star)
    record("double", float(T_star), float(T_star), f, e)
    tri = hybrid_triplet(f, e[1], e[2], f_dk)
    ft = f[tri]
    print(f"  three coupled modes at T_A = T_B = T*:")
    for kk, jj in enumerate(tri):
        print(f"    f{kk + 1} = {f[jj]:.4f} Hz   e_deck={e[0, jj]:.3f}  "
              f"e_A={e[1, jj]:.3f}  e_B={e[2, jj]:.3f}")
    outer = float((ft[2] - ft[0]) / f_dk)
    mid_off = float((ft[1] - f_dk) / f_dk)
    print(f"  outer split (f3 - f1)/f_dk    : {100 * outer:.4f} %")
    print(f"    vs s_A                      : {outer / s_A:.4f}")
    print(f"    vs s_B                      : {outer / s_B:.4f}")
    print(f"    vs sqrt(s_A^2 + s_B^2)      : {outer / s_AB:.4f}")
    print(f"  middle mode offset from f_dk  : {100 * mid_off:+.4f} %")
    dark = tri[1]
    if e[1, dark] > 0 and e[2, dark] > 0:
        print(f"  middle-mode e_A/e_B           : "
              f"{e[1, dark] / e[2, dark]:.4f}   "
              f"(dark-state prediction s_B^2/s_A^2 = "
              f"{(s_B / s_A) ** 2:.4f})")

    # joint fine scan, both tensions moved together
    best3 = None
    for T in np.linspace(T_star - 10e3, T_star + 10e3, 401):
        f, e = solve(float(T), float(T))
        tri = hybrid_triplet(f, e[1], e[2], f_dk)
        if tri is None:
            continue
        ft = f[tri]
        og = float(ft[2] - ft[0])
        if best3 is None or og < best3[0]:
            best3 = (og, float(T), ft.copy())
    og_min, T3_min, ft_min = best3
    s_outer_min = og_min / f_dk
    print(f"  minimum outer split, joint scan: {100 * s_outer_min:.4f} % "
          f"at T = {T3_min / 1e3:.2f} kN "
          f"(f = {ft_min[0]:.4f}, {ft_min[1]:.4f}, {ft_min[2]:.4f} Hz)")
    print(f"    vs sqrt(s_A^2 + s_B^2)      : {s_outer_min / s_AB:.4f}")

    # --- write ---
    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, "twostay.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print()
    print(f"  wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
