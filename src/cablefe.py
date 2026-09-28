# -*- coding: utf-8 -*-
"""Coupled cable-deck finite element model for a cable-stayed footbridge.

The point of this model is to possess something the models used in cable
force identification do not: stay cables that carry distributed mass and
therefore have transverse modes of their own, attached to a deck that has
modes of its own, so that the two can interact.

Two components, both planar Euler-Bernoulli chains:

    deck    simply supported beam, EI_d, m_d, span L_d
    stay    tensioned beam, EI_c, m_c, chord length L_c, tension T,
            inclined at theta to the horizontal, top end held at a rigid
            pylon, bottom end anchored to the deck

The stay enters the deck problem twice, and both paths matter:

    axially     k_ax = (EA/L_c) sin^2(theta) at the anchorage, the vertical
                restraint that makes the deck cable-stayed rather than a
                plain beam
    transversely  v_cable(bottom) = cos(theta) * w_deck(anchorage), the tie
                that lets a deck mode drive the stay and a stay mode push
                back on the deck

The second path is the one under examination. It is what produces frequency
loci veering when a stay frequency approaches a deck frequency, and it is
absent from every model behind the incumbent tension formulas, which treat
the stay as an isolated element with idealised ends.

Assumptions, all stated because the pilot rests on them:

* planar motion only, so out-of-plane stay modes and deck torsion are not
  represented;
* straight chord, so sag is neglected.  Defensible for the short steep stays
  of a footbridge, where the Irvine parameter is small, and checked in
  ``scripts/verify_cablefe.py`` rather than assumed;
* rigid pylon;
* the stay tension is a parameter, not a result of a form-finding step;
* no damping.  The pilot is about frequencies and mode shapes.

The element matrices are standard: Euler-Bernoulli bending, the consistent
geometric stiffness for axial tension, and the consistent mass matrix.  With
``EI -> 0`` the stay reduces to the taut string and with ``T -> 0`` to a
beam, and both limits are checked.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


# ---------------------------------------------------------------------------
# elements
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# closed forms, used for verification and for placing the campaign
# ---------------------------------------------------------------------------

def string_freq(n, L, T, m):
    """Taut string, pinned ends.  The formula the incumbent method inverts."""
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

    ``lambda^2 = (m g cos(theta) L / T)^2 * L / (T L_e / EA)``: the gravity
    component normal to the chord, the chord tension ``T``, the chord length
    ``L``, the normal sag ``d_n = m g cos(theta) L^2 / (8 T)`` and
    ``L_e = L (1 + 8 (d_n / L)^2)``.  Irvine's treatment of the inclined
    cable: the horizontal-cable theory holds with ``g -> g cos(theta)`` and
    the chord tension in place of the horizontal one.  Small values mean the
    straight-chord idealisation is safe.

    This function has been corrected twice, and both corrections are
    recorded here rather than silently made.  The first version used ``T``
    where the horizontal-projection form uses ``H`` and carried a factor
    that cancelled; nothing called it.  The second version (submitted
    manuscript, campaign column ``lam2``) used the horizontal-projection form
    ``(m g L_h / H)^2 L_h / (H L_e / EA)`` with ``H = T cos(theta)``,
    ``L_h = L cos(theta)`` and ``m`` per unit ARC length, which for an
    inclined cable overstates the chord-based parameter by ``1 / cos^3
    (theta)`` (1.82 at 35 degrees).  The chord form is the one the
    two-dimensional finite element of ``cablefe2d.py`` verifies against
    Irvine's symmetric-mode equation, to 0.06 % over lambda^2 = 0.25 to 8
    (``scripts/verify_cablefe2d.py``), so it is the form kept.  Every
    lambda^2 quoted in the revised manuscript is recomputed with it.
    """
    dn = m * g * np.cos(theta) * L ** 2 / (8.0 * T)
    Le = L * (1.0 + 8.0 * (dn / L) ** 2)
    return (m * g * np.cos(theta) * L / T) ** 2 * L / (T * Le / EA)


def sag_ratio(L, T, m, theta, g=9.80665):
    """Sag as a fraction of the chord, the direct statement of the same thing."""
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


# ---------------------------------------------------------------------------
# coupled system
# ---------------------------------------------------------------------------

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
        """Mass-normalised modes of the coupled system.

        Returns ``(f, Phi)`` with ``f`` in Hz ascending and ``Phi`` the FULL
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
        """DOF of a stay-mounted accelerometer ``s_from_anchor`` up the chord.

        Field practice puts the accelerometer a short distance above the
        lower anchorage, where it is reachable, so that is what is modelled.
        """
        i = int(round((1.0 - s_from_anchor / self.Lc) * self.nc))
        i = max(0, min(self.nc, i))
        return self.nD + 2 * i

    def participation(self, Phi, dof):
        """Absolute mass-normalised modal amplitude at one DOF."""
        return np.abs(Phi[dof, :])

    def energy_split(self, Phi):
        """Fraction of modal kinetic energy in the stay, per mode.

        The generalized mass ratio Liu, Lin and Wang use to grade how coupled
        a mode is.  Near 1 the mode is a stay mode, near 0 a deck mode, and
        intermediate values are the hybrids that veering produces.

        Written as two matrix products rather than a loop over modes.  The
        loop form spent 2.7 s on 40 modes of a 164 degree of freedom system,
        because each ``p @ M @ p`` is a BLAS call too small to cover its own
        threading overhead.  Cast as one ``M @ Phi`` per component it costs
        0.013 s, which matters because the figures call this at every step of
        a tension range.
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


# ---------------------------------------------------------------------------
# the incumbent inversion, applied exactly as a field engineer would
# ---------------------------------------------------------------------------

def pick_peaks(f, Phi, sensor_dof, nmax=8, rel_floor=0.05):
    """Peaks a stay-mounted accelerometer would show, in frequency order.

    Modes are kept when their amplitude at the sensor is at least
    ``rel_floor`` of the largest, which is the practical statement that a
    peak has to rise out of the spectrum to be picked. No knowledge of which
    modes are "really" stay modes is used, because the engineer has none.
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
    linear in ``n^2``.  The intercept gives T and the slope gives EI.  This
    is the variant field practice prefers because it returns the bending
    stiffness and does not rely on identifying a single mode order, and it
    is the variant with the most to lose when one picked peak is not a stay
    mode at all.
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
    """Harmonic-comb screening, which is what careful practice actually does.

    An engineer does not number spectral peaks 1, 2, 3 blindly.  The stay
    modes of a near-taut cable are close to a harmonic series, so the peaks
    are searched for the comb that best explains them: each candidate peak is
    tried as the fundamental, its harmonics are matched against the peak
    list, and the candidate explaining the most peaks wins.  A deck mode
    sitting next to a stay mode is then rejected, because it does not fall on
    the comb.

    This sits between the oracle, which never mis-assigns, and blind
    numbering, which always does.  It is the realistic case, and unlike blind
    numbering its answer does not hinge on where the amplitude floor is put.

    Returns ``(freqs, orders)`` of the surviving peaks.
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
        # score on how much of the comb is explained, tie-broken by
        # preferring the lower fundamental so a harmonic is not mistaken
        # for the fundamental itself
        score = len(got_f)
        if score > best[0]:
            best = (score, (np.array(got_f), np.array(got_n)))
    if best[1] is None:
        return np.array([]), np.array([])
    return best[1]


# ---------------------------------------------------------------------------
# the governing group and the closed form for the veering width
# ---------------------------------------------------------------------------

def mu_effective(M_stay, phi_anchor):
    """Effective modal mass ratio governing deck-stay veering.

    The plain ratio of stay modal mass to deck modal mass is NOT the group
    that governs, and using it is a trap: a deck mode with a node at the
    anchorage cannot couple to the stay at all, however light the deck is.
    What governs is the deck mode's participation AT THE ANCHORAGE,

        mu_eff = M_stay * phi_a^2

    with ``phi_a`` the mass-normalised deck mode amplitude there.  This is the
    generalized mass ratio Liu, Lin and Wang use to grade deck-stay coupling,
    arrived at here independently and by correcting a wrong first guess.
    """
    return M_stay * phi_anchor ** 2


def veering_split(mu_eff, theta, n=1):
    """Normalised frequency split of the hybrid pair at exact tuning.

        (f+ - f-) / f0 = (2 / (n pi)) cos(theta) sqrt(mu_eff)

    for stay mode order ``n``.  Derived by expanding the exact characteristic
    equation about ``kL = n pi``: with ``sin(kL) -> (-1)^n delta`` and
    ``cos(kL) -> (-1)^n`` the half split solves
    ``x^2 = c^2 T k / (2 n pi M)``, and ``k = n pi / L`` makes the tension and
    the mode order cancel to leave ``x = c sqrt(T / (2 L M))``, independent of
    ``n`` in absolute terms and therefore falling as ``1/n`` once normalised
    by ``f_n``.

    The ``1/n`` matters and was missing from the first version of this
    function.  Every check of that version was run at ``n = 1``, where the
    factor is unity, so all of them passed while the law was wrong for every
    higher mode order.  The campaign caught it, and
    ``scripts/verify_coupling.py`` now exercises ``n = 1`` to ``5``
    specifically so it cannot recur.

    Because the incumbent inversion takes tension as proportional to the
    square of frequency, this split is also, to leading order, the tension
    error the isolated-cable formula incurs at an exact crossing.
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
