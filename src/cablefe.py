# -*- coding: utf-8 -*-
"""Coupled cable-deck finite element model for a cable-stayed footbridge.

A simply supported Euler-Bernoulli deck and one tensioned-beam stay, inclined
at theta, held at a rigid pylon and anchored to the deck. The stay acts on the
deck through an axial spring ``k_ax = (EA/L_c) sin^2(theta)`` and a transverse
tie ``v_stay(bottom) = cos(theta) w_deck(anchorage)``. Planar motion, straight
chord (no sag), no damping. Also provides closed-form frequencies, the
isolated-cable tension inversions and the veering-split law.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


# --- elements ---

def beam_element(l, EI, m, T=0.0):
    """Planar Euler-Bernoulli element with axial tension.

    DOF order ``[v1, th1, v2, th2]``.  Returns ``(k, mm)``.  The geometric
    stiffness carries the sign convention that tension stiffens, so ``T`` is
    positive in tension.
    """
    l = float(l)
    k = EI / l ** 3 * np.array([
        [12.0,   6 * l,  -12.0,   6 * l],
        [6 * l, 4 * l * l, -6 * l, 2 * l * l],
        [-12.0, -6 * l,   12.0,  -6 * l],
        [6 * l, 2 * l * l, -6 * l, 4 * l * l]])

    if T:
        k = k + T / (30.0 * l) * np.array([
            [36.0,   3 * l,  -36.0,   3 * l],
            [3 * l, 4 * l * l, -3 * l,  -l * l],
            [-36.0, -3 * l,   36.0,  -3 * l],
            [3 * l,  -l * l, -3 * l, 4 * l * l]])

    mm = m * l / 420.0 * np.array([
        [156.0,  22 * l,   54.0, -13 * l],
        [22 * l, 4 * l * l, 13 * l, -3 * l * l],
        [54.0,  13 * l,  156.0, -22 * l],
        [-13 * l, -3 * l * l, -22 * l, 4 * l * l]])

    return k, mm


def chain(L, nel, EI, m, T=0.0):
    """Assemble a uniform chain of ``nel`` elements over length ``L``.

    Returns ``(K, M)`` of size ``2(nel+1)``, DOFs ordered node by node as
    ``[v0, th0, v1, th1, ...]``.
    """
    ndof = 2 * (nel + 1)
    K = np.zeros((ndof, ndof))
    M = np.zeros((ndof, ndof))
    ke, me = beam_element(L / nel, EI, m, T)
    for e in range(nel):
        idx = np.array([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3])
        K[np.ix_(idx, idx)] += ke
        M[np.ix_(idx, idx)] += me
    return K, M


# --- closed forms ---

def string_freq(n, L, T, m):
    """Taut-string frequency with pinned ends, ``f_n = (n / 2L) sqrt(T/m)``."""
    return n / (2.0 * L) * np.sqrt(T / m)


def tensioned_beam_freq(n, L, T, EI, m):
    """Pinned-pinned tensioned beam, exact.

    ``f_n^2 = n^2 [T + EI (n pi / L)^2] / (4 m L^2)``.  Reduces to
    :func:`string_freq` as ``EI -> 0``.
    """
    kn = n * np.pi / L
    return np.sqrt(n ** 2 * (T + EI * kn ** 2) / (4.0 * m * L ** 2))


def beam_freq(n, L, EI, m):
    """Simply supported Euler-Bernoulli beam, exact."""
    return (n * np.pi / L) ** 2 / (2.0 * np.pi) * np.sqrt(EI / m)


def irvine_lambda2(L, T, EA, m, theta, g=9.80665):
    """Irvine sag parameter of an inclined stay, referred to the chord.

    ``lambda^2 = (m g cos(theta) L / T)^2 * L / (T L_e / EA)`` with the
    normal sag ``d_n = m g cos(theta) L^2 / (8 T)`` and
    ``L_e = L (1 + 8 (d_n / L)^2)``. This is Irvine's horizontal-cable result with ``g -> g cos(theta)`` and the
    chord tension in place of the horizontal one; scripts/verify_cablefe2d.py
    checks it against a 2-D finite element model. Small values mean the
    straight-chord idealization is safe.
    """
    dn = m * g * np.cos(theta) * L ** 2 / (8.0 * T)
    Le = L * (1.0 + 8.0 * (dn / L) ** 2)
    return (m * g * np.cos(theta) * L / T) ** 2 * L / (T * Le / EA)


def sag_ratio(L, T, m, theta, g=9.80665):
    """Sag as a fraction of the chord length."""
    H = T * np.cos(theta)
    Lh = L * np.cos(theta)
    return (m * g * Lh ** 2 / (8.0 * H)) / L


def xi_param(L, T, EI):
    """Bending stiffness parameter ``xi = L sqrt(T/EI)``.

    Large ``xi`` means the stay behaves as a string; small ``xi`` means
    bending stiffness dominates and the string formula is already biased
    before any coupling is considered.
    """
    return L * np.sqrt(T / EI)


# --- coupled system ---

class CableDeck:
    """One deck with one stay, assembled and reduced by the tie constraint.

    Parameters are SI throughout: N, m, kg, s.
    """

    def __init__(self, Ld, EId, md, Lc, EIc, mc, T, EA, theta,
                 x_anchor=None, nd=40, nc=40, pinned_stay=True):
        self.Ld, self.EId, self.md = Ld, EId, md
        self.Lc, self.EIc, self.mc = Lc, EIc, mc
        self.T, self.EA, self.theta = T, EA, theta
        self.nd, self.nc = nd, nc
        self.x_anchor = 0.5 * Ld if x_anchor is None else x_anchor
        self.pinned_stay = pinned_stay

        # anchorage must land on a deck node
        self.ia = int(round(self.x_anchor / Ld * nd))
        self.x_anchor = self.ia * Ld / nd

        self._assemble()

    # -- assembly ---------------------------------------------------------

    def _assemble(self):
        Kd, Md = chain(self.Ld, self.nd, self.EId, self.md, 0.0)
        Kc, Mc = chain(self.Lc, self.nc, self.EIc, self.mc, self.T)

        nD, nC = Kd.shape[0], Kc.shape[0]
        N = nD + nC
        K = np.zeros((N, N))
        M = np.zeros((N, N))
        K[:nD, :nD] = Kd
        M[:nD, :nD] = Md
        K[nD:, nD:] = Kc
        M[nD:, nD:] = Mc

        # stay axial restraint at the anchorage
        k_ax = self.EA / self.Lc * np.sin(self.theta) ** 2
        K[2 * self.ia, 2 * self.ia] += k_ax
        self.k_ax = k_ax

        self.nD, self.nC, self.N = nD, nC, N

        # ---- constraints -------------------------------------------------
        # deck simply supported; stay top held at the pylon; stay bottom tied
        # to the deck through the inclination.
        fixed = {0, 2 * self.nd, nD + 0}
        if not self.pinned_stay:
            fixed |= {nD + 1, nD + 2 * self.nc + 1}

        dep = nD + 2 * self.nc          # stay bottom transverse DOF
        indep_of_dep = 2 * self.ia      # deck vertical DOF at the anchorage
        coeff = np.cos(self.theta)

        free = [i for i in range(N) if i not in fixed and i != dep]
        pos = {d: j for j, d in enumerate(free)}

        Lmat = np.zeros((N, len(free)))
        for d, j in pos.items():
            Lmat[d, j] = 1.0
        Lmat[dep, pos[indep_of_dep]] = coeff

        self.Lmat, self.free, self.pos = Lmat, free, pos
        self.K = Lmat.T @ K @ Lmat
        self.M = Lmat.T @ M @ Lmat

    # -- solution ---------------------------------------------------------

    def modes(self, nmodes=30):
        """Mass-normalized modes of the coupled system.

        Returns ``(f, Phi)`` with ``f`` in Hz ascending and ``Phi`` the full
        DOF mode shapes, columns matching ``f``.
        """
        w2, V = eigh(self.K, self.M)
        w2 = np.maximum(w2, 0.0)
        f = np.sqrt(w2) / (2.0 * np.pi)
        k = min(nmodes, len(f))
        return f[:k], (self.Lmat @ V[:, :k])

    # -- interrogation ----------------------------------------------------

    def cable_dofs(self):
        """Transverse DOF indices along the stay, top to bottom."""
        return [self.nD + 2 * i for i in range(self.nc + 1)]

    def deck_dofs(self):
        return [2 * i for i in range(self.nd + 1)]

    def cable_sensor_dof(self, s_from_anchor):
        """DOF of a stay sensor ``s_from_anchor`` meters up the chord."""
        i = int(round((1.0 - s_from_anchor / self.Lc) * self.nc))
        i = max(0, min(self.nc, i))
        return self.nD + 2 * i

    def participation(self, Phi, dof):
        """Absolute mass-normalized modal amplitude at one DOF."""
        return np.abs(Phi[dof, :])

    def energy_split(self, Phi):
        """Fraction of modal kinetic energy in the stay, per mode.

        The generalized mass ratio of Liu, Lin and Wang (2012): near 1 a stay
        mode, near 0 a deck mode, in between a hybrid.
        """
        Md_full, Mc_full = self._mass_blocks()
        ed = np.einsum('ij,ij->j', Phi, Md_full @ Phi)
        ec = np.einsum('ij,ij->j', Phi, Mc_full @ Phi)
        tot = ec + ed
        return np.where(tot > 0, ec / np.where(tot > 0, tot, 1.0), 0.0)

    def _mass_blocks(self):
        """Deck and stay mass matrices padded to the full system, cached."""
        if getattr(self, "_mblocks", None) is None:
            _, Md = chain(self.Ld, self.nd, self.EId, self.md, 0.0)
            _, Mc = chain(self.Lc, self.nc, self.EIc, self.mc, self.T)
            Md_full = np.zeros((self.N, self.N))
            Mc_full = np.zeros((self.N, self.N))
            Md_full[:self.nD, :self.nD] = Md
            Mc_full[self.nD:, self.nD:] = Mc
            self._mblocks = (Md_full, Mc_full)
        return self._mblocks

    # -- uncoupled references ---------------------------------------------

    def stay_alone(self, nmodes=10):
        """Frequencies of the stay with both ends held, the isolated ideal."""
        return np.array([tensioned_beam_freq(n, self.Lc, self.T, self.EIc,
                                             self.mc)
                         for n in range(1, nmodes + 1)])

    def deck_alone(self, nmodes=10):
        """Deck frequencies with the stay present only as its axial spring."""
        Kd, Md = chain(self.Ld, self.nd, self.EId, self.md, 0.0)
        Kd = Kd.copy()
        Kd[2 * self.ia, 2 * self.ia] += self.k_ax
        keep = [i for i in range(Kd.shape[0]) if i not in (0, 2 * self.nd)]
        w2, _ = eigh(Kd[np.ix_(keep, keep)], Md[np.ix_(keep, keep)])
        f = np.sqrt(np.maximum(w2, 0.0)) / (2.0 * np.pi)
        return f[:nmodes]


# --- the isolated-cable tension inversions ---

def pick_peaks(f, Phi, sensor_dof, nmax=8, rel_floor=0.05):
    """Peaks a stay-mounted accelerometer would show, in frequency order.

    Modes are kept when their amplitude at the sensor is at least
    ``rel_floor`` of the largest. No knowledge of which modes are stay modes
    is used.
    """
    amp = np.abs(Phi[sensor_dof, :])
    if amp.max() <= 0:
        return np.array([]), np.array([])
    keep = amp >= rel_floor * amp.max()
    fk, ak = f[keep], amp[keep]
    order = np.argsort(fk)
    return fk[order][:nmax], ak[order][:nmax]


def invert_string(f_peaks, L, m, n=None):
    """Single-mode taut-string inversion, one estimate per picked peak."""
    f_peaks = np.asarray(f_peaks, dtype=float)
    if n is None:
        n = np.arange(1, len(f_peaks) + 1)
    n = np.asarray(n, dtype=float)
    return 4.0 * m * L ** 2 * f_peaks ** 2 / n ** 2


def invert_multimode(f_peaks, L, m, n=None):
    """Multi-mode regression for T and EI together.

    For a pinned-pinned tensioned beam
    ``f_n^2 / n^2 = T/(4 m L^2) + EI pi^2 n^2 / (4 m L^4)``,
    linear in ``n^2``: the intercept gives T and the slope gives EI.
    Returns ``(T, EI)``, or NaN for fewer than three peaks.
    """
    f_peaks = np.asarray(f_peaks, dtype=float)
    if n is None:
        n = np.arange(1, len(f_peaks) + 1, dtype=float)
    n = np.asarray(n, dtype=float)
    if len(n) < 3:
        return np.nan, np.nan
    y = f_peaks ** 2 / n ** 2
    x = n ** 2
    A = np.vstack([np.ones_like(x), x]).T
    c, slope = np.linalg.lstsq(A, y, rcond=None)[0]
    T = 4.0 * m * L ** 2 * c
    EI = slope * 4.0 * m * L ** 4 / np.pi ** 2
    return T, EI


def screened_pick(f_all, amp, nmax=5, tol=0.04):
    """Harmonic-comb screening of spectral peaks.

    Each peak is tried as the fundamental, its harmonics within ``tol`` are
    matched against the peak list, and the candidate explaining the most
    peaks wins, so a deck mode off the comb is rejected. Returns
    ``(freqs, orders)`` of the surviving peaks.
    """
    f_all = np.asarray(f_all, dtype=float)
    order = np.argsort(f_all)
    fs = f_all[order]
    best = (-1, None)
    for f0 in fs:
        if f0 <= 0:
            continue
        got_f, got_n = [], []
        for n in range(1, nmax + 1):
            target = n * f0
            j = int(np.argmin(np.abs(fs - target)))
            if abs(fs[j] - target) / target <= tol:
                got_f.append(fs[j])
                got_n.append(n)
        # strict > keeps the lower fundamental on ties (fs is ascending)
        score = len(got_f)
        if score > best[0]:
            best = (score, (np.array(got_f), np.array(got_n)))
    if best[1] is None:
        return np.array([]), np.array([])
    return best[1]


# --- the governing group and the veering split ---

def mu_effective(M_stay, phi_anchor):
    """Effective modal mass ratio governing deck-stay veering.

        mu_eff = M_stay * phi_a^2

    with ``phi_a`` the mass-normalized deck mode amplitude at the anchorage.
    The plain ratio of stay to deck modal mass does not govern: a deck mode
    with a node at the anchorage does not couple to the stay. This is the
    generalized mass ratio of Liu, Lin and Wang (2012).
    """
    return M_stay * phi_anchor ** 2


def veering_split(mu_eff, theta, n=1):
    """Normalized frequency split of the hybrid pair at exact tuning.

        (f+ - f-) / f0 = (2 / (n pi)) cos(theta) sqrt(mu_eff)

    for stay mode order ``n``, from expanding the exact characteristic
    equation about ``kL = n pi``. The absolute split does not depend on
    ``n``, so the normalized split falls as ``1/n``; scripts/verify_coupling.py
    checks ``n = 1`` to ``5``. To leading order this is also the tension error
    of the isolated-cable inversion at exact tuning.
    """
    return 2.0 / (n * np.pi) * np.cos(theta) * np.sqrt(mu_eff)


def tension_error(d, mu_eff, theta, n=1):
    """Tension error of the isolated-cable inversion at relative detuning d.

        eps = sqrt(d^2 + s^2) - |d|,   s = veering_split(mu_eff, theta, n)

    Equals ``s`` at exact tuning and decays as ``s^2 / (2|d|)`` away from it.
    """
    s = veering_split(mu_eff, theta, n)
    return np.sqrt(d ** 2 + s ** 2) - np.abs(d)


def detuning_required(tol, mu_eff, theta, n=1):
    """Screening criterion: detuning needed to hold the error within ``tol``.

        |d| >= (s^2 - tol^2) / (2 tol)

    Negative or zero means the tolerance is met at any detuning, including an
    exact crossing.
    """
    s = veering_split(mu_eff, theta, n)
    return (s ** 2 - tol ** 2) / (2.0 * tol)
