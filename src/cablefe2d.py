# -*- coding: utf-8 -*-
"""Planar cable-deck model with the stay drawn in two dimensions: sag and a
flexible pylon.  Revision 1 extension of ``cablefe.py``.

``cablefe.CableDeck`` carries the stay as a straight chain of transverse
Euler-Bernoulli elements, enters the deck axially through a spring
k_ax = (EA / L_c) sin^2(theta) and transversely through the tie
v(L_c) = cos(theta) w_d, and holds the top of the stay at a rigid pylon.  Two
reviewers asked what happens when the stay sags (R1.5, R2.2) and when the
pylon sways (R2.3, R2.4).  Both need the stay to have a shape and an axial
degree of freedom, so here the stay is a chain of planar FRAME elements
(axial + bending, three degrees of freedom per node, global coordinates)
laid along its static profile:

* profile: the parabola of a cable under gravity, drawn as a vertical
  offset below the chord, mid-sag d_v = m g L_c^2 / (8 T), so that the sag
  normal to the chord is d_v cos(theta) = m g cos(theta) L_c^2 / (8 T).  The
  gravity that draws it is a PARAMETER ``g`` so that the sag can be varied
  with everything else held (frequencies, axial spring, mass ratio);
* axial force along the stay: T at mid-length, varying by m g sin(theta)
  along the chord as the upper part carries the weight below it;
* the two routes into the deck are no longer imposed: the anchorage node
  moves with the deck vertically and the element geometry resolves that
  motion along and across the chord by itself.  With g = 0 the model must
  reproduce ``CableDeck`` and it is checked to (``scripts/verify_cablefe2d.py``);
* pylon (optional): a vertical cantilever of frame elements from deck level
  to the stay's top, fixed at its base, carrying the stay's top node.  Its
  tip sways and drives the stay from the second end.  The compression
  T sin(theta) it carries enters its geometric stiffness.

Retained idealisations: planar motion; deck as a simply supported beam with
vertical and rotational degrees of freedom only, so the anchorage does not
move horizontally; no damping; tension a parameter.

Verification targets: ``CableDeck`` in the straight limit; Irvine's
symmetric in-plane modes of the sagged stay alone; the two-ended reduction
of the coupling with a swaying pylon.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq

from cablefe import beam_element, chain

G = 9.80665


# ---------------------------------------------------------------------------
# elements
# ---------------------------------------------------------------------------

def frame_element(l, EA, EI, m, N=0.0):
    """Planar frame element in LOCAL coordinates, DOF ``[u1, v1, th1, u2, v2, th2]``.

    Axial (linear) + Euler-Bernoulli bending (cubic) + consistent geometric
    stiffness of the axial force ``N`` (positive in tension) on the
    transverse DOFs, and the consistent mass of both.  Returns ``(k, mm)``.
    """
    k = np.zeros((6, 6))
    mm = np.zeros((6, 6))
    ka = EA / l
    for i, j, v in ((0, 0, ka), (0, 3, -ka), (3, 0, -ka), (3, 3, ka)):
        k[i, j] += v
    ma = m * l / 6.0
    for i, j, v in ((0, 0, 2 * ma), (0, 3, ma), (3, 0, ma), (3, 3, 2 * ma)):
        mm[i, j] += v
    kb, mb = beam_element(l, EI, m, N)
    idx = np.array([1, 2, 4, 5])
    k[np.ix_(idx, idx)] += kb
    mm[np.ix_(idx, idx)] += mb
    return k, mm


def rotation(c, s):
    """Global -> local transform for one frame element with direction (c, s)."""
    R = np.zeros((6, 6))
    for b in (0, 3):
        R[b, b], R[b, b + 1] = c, s
        R[b + 1, b], R[b + 1, b + 1] = -s, c
        R[b + 2, b + 2] = 1.0
    return R


def polyline_chain(xy, EA, EI, m, N):
    """Assemble frame elements along the nodes ``xy`` (n+1 rows) with per-
    element axial force ``N`` (length n).  Global DOFs ``[u, w, th]`` per node."""
    n = len(xy) - 1
    ndof = 3 * (n + 1)
    K = np.zeros((ndof, ndof))
    M = np.zeros((ndof, ndof))
    for e in range(n):
        dx, dz = xy[e + 1] - xy[e]
        l = float(np.hypot(dx, dz))
        c, s = dx / l, dz / l
        k, mm = frame_element(l, EA, EI, m, N[e])
        R = rotation(c, s)
        kg, mg = R.T @ k @ R, R.T @ mm @ R
        idx = np.arange(3 * e, 3 * e + 6)
        K[np.ix_(idx, idx)] += kg
        M[np.ix_(idx, idx)] += mg
    return K, M


# ---------------------------------------------------------------------------
# closed forms for verification
# ---------------------------------------------------------------------------

def irvine_lambda2_chord(L, T, EA, m, theta, g=G):
    """Irvine's parameter for an inclined cable, referred to the chord.

    ``lambda^2 = (m g cos(theta) L / T)^2 * L / (T L_e / EA)`` with the
    gravity component normal to the chord, the chord tension ``T`` and the
    chord length ``L``; ``L_e = L (1 + 8 (d_n/L)^2)`` with the normal sag
    ``d_n = m g cos(theta) L^2 / (8 T)``.  Irvine's treatment of the inclined
    cable: the horizontal-cable theory holds with ``g -> g cos(theta)`` and
    the horizontal tension replaced by the chord tension.  This is the form
    the finite element of this module is checked against.
    """
    dn = m * g * np.cos(theta) * L ** 2 / (8.0 * T)
    Le = L * (1.0 + 8.0 * (dn / L) ** 2)
    return (m * g * np.cos(theta) * L / T) ** 2 * L / (T * Le / EA)


def irvine_symmetric_beta(lam2, k=1):
    """k-th root ``beta`` of Irvine's symmetric-mode equation
    ``tan(beta/2) = beta/2 - (4/lambda^2) (beta/2)^3``; the mode frequency is
    ``omega = beta sqrt(T/m) / L``.  ``lam2 -> 0`` returns ``(2k-1) pi``."""
    if lam2 <= 0:
        return (2 * k - 1) * np.pi

    def f(b):
        h = 0.5 * b
        return np.tan(h) - h + 4.0 / lam2 * h ** 3

    # the k-th root lies between the (2k-1)-th and (2k+1)-th odd multiples
    # of pi, where tan(beta/2) runs from -inf to +inf
    lo, hi = (2 * k - 1) * np.pi + 1e-9, (2 * k + 1) * np.pi - 1e-9
    return brentq(f, lo, hi, xtol=1e-12, maxiter=500)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

class CableDeck2D:
    """One deck, one stay drawn in the plane, optional pylon.

    Parameters as ``cablefe.CableDeck`` plus ``g`` (gravity drawing the sag;
    0 for a straight chord) and ``pylon = dict(EI, m, EA, n)`` for a flexible
    pylon of the stay's own height (``None`` for rigid).
    """

    def __init__(self, Ld, EId, md, Lc, EIc, mc, T, EA, theta,
                 x_anchor=None, nd=40, nc=40, g=0.0, pylon=None,
                 tension_variation=True):
        self.Ld, self.EId, self.md = Ld, EId, md
        self.Lc, self.EIc, self.mc = Lc, EIc, mc
        self.T, self.EA, self.theta, self.g = T, EA, theta, g
        # the weight component along the chord makes the tension vary along
        # the stay; Irvine's theory neglects it, so the switch lets the
        # verification meet Irvine on his own terms and the sag study state
        # which case it computed
        self.tension_variation = tension_variation
        self.nd, self.nc = nd, nc
        self.x_anchor = 0.5 * Ld if x_anchor is None else x_anchor
        self.ia = int(round(self.x_anchor / Ld * nd))
        self.x_anchor = self.ia * Ld / nd
        self.pylon = pylon
        self._assemble()

    # -- geometry ---------------------------------------------------------

    def stay_nodes(self):
        """Node coordinates from the pylon top P to the anchorage A, and the
        per-element axial force."""
        c, s = np.cos(self.theta), np.sin(self.theta)
        A = np.array([self.x_anchor, 0.0])
        P = A + self.Lc * np.array([-c, s])
        xi = np.linspace(0.0, 1.0, self.nc + 1)
        dv = self.mc * self.g * self.Lc ** 2 / (8.0 * self.T)      # vertical mid-sag
        xy = P[None, :] + xi[:, None] * (A - P)[None, :]
        xy[:, 1] -= 4.0 * dv * xi * (1.0 - xi)
        # axial force: T at mid-length, the upper part carrying the weight
        # below it along the chord
        sm = 0.5 * (xi[:-1] + xi[1:]) * self.Lc
        if self.tension_variation:
            N = self.T + self.mc * self.g * s * (0.5 * self.Lc - sm)
        else:
            N = self.T * np.ones(self.nc)
        return xy, N, P, A

    def pylon_nodes(self, P):
        npy = self.pylon["n"]
        B = np.array([P[0], 0.0])
        t = np.linspace(0.0, 1.0, npy + 1)
        return B[None, :] + t[:, None] * (P - B)[None, :]

    # -- assembly ---------------------------------------------------------

    def _assemble(self):
        Kd, Md = chain(self.Ld, self.nd, self.EId, self.md, 0.0)
        xy, N, P, A = self.stay_nodes()
        Kc, Mc = polyline_chain(xy, self.EA, self.EIc, self.mc, N)
        nD, nC = Kd.shape[0], Kc.shape[0]
        blocks = [(Kd, Md), (Kc, Mc)]
        if self.pylon is not None:
            pxy = self.pylon_nodes(P)
            Np = -self.T * np.sin(self.theta) * np.ones(self.pylon["n"])   # compression
            Kp, Mp = polyline_chain(pxy, self.pylon["EA"], self.pylon["EI"],
                                    self.pylon["m"], Np)
            blocks.append((Kp, Mp))
        sizes = [b[0].shape[0] for b in blocks]
        offs = np.cumsum([0] + sizes)
        Ntot = offs[-1]
        K = np.zeros((Ntot, Ntot))
        M = np.zeros((Ntot, Ntot))
        for (k, m), o, n in zip(blocks, offs[:-1], sizes):
            K[o:o + n, o:o + n] = k
            M[o:o + n, o:o + n] = m
        self.offs, self.sizes, self.N = offs, sizes, Ntot
        self.nD, self.nC = nD, nC

        # ---- constraints -------------------------------------------------
        oC = offs[1]
        top = oC + 0                      # stay node 0 (pylon end): u, w, th
        bot = oC + 3 * self.nc            # stay node nc (anchorage)
        fixed = {0, 2 * self.nd}          # deck simply supported
        slaves = {}                       # slave dof -> [(master, coeff)]
        fixed |= {bot + 0}                # anchorage does not move horizontally
        slaves[bot + 1] = [(2 * self.ia, 1.0)]     # w_bottom = w_deck(anchorage)
        if self.pylon is None:
            fixed |= {top + 0, top + 1}
        else:
            oP = offs[2]
            fixed |= {oP + 0, oP + 1, oP + 2}     # pylon base fixed
            tip = oP + 3 * self.pylon["n"]
            slaves[top + 0] = [(tip + 0, 1.0)]
            slaves[top + 1] = [(tip + 1, 1.0)]
            self.tip = tip
        free = [i for i in range(Ntot) if i not in fixed and i not in slaves]
        pos = {d: j for j, d in enumerate(free)}
        Lmat = np.zeros((Ntot, len(free)))
        for d, j in pos.items():
            Lmat[d, j] = 1.0
        for sl, lst in slaves.items():
            for mst, cf in lst:
                Lmat[sl, pos[mst]] += cf
        self.Lmat, self.free, self.pos = Lmat, free, pos
        self.K = Lmat.T @ K @ Lmat
        self.M = Lmat.T @ M @ Lmat
        self._Kfull, self._Mfull = K, M
        self.top, self.bot = top, bot

    # -- solution ---------------------------------------------------------

    def modes(self, nmodes=30):
        w2, V = eigh(self.K, self.M)
        w2 = np.maximum(w2, 0.0)
        f = np.sqrt(w2) / (2.0 * np.pi)
        k = min(nmodes, len(f))
        return f[:k], (self.Lmat @ V[:, :k])

    def energy_split(self, Phi):
        """Fraction of modal kinetic energy in the stay (host = deck + pylon)."""
        oC, nC = self.offs[1], self.sizes[1]
        Mc = np.zeros_like(self._Mfull)
        Mc[oC:oC + nC, oC:oC + nC] = self._Mfull[oC:oC + nC, oC:oC + nC]
        ec = np.einsum('ij,ij->j', Phi, Mc @ Phi)
        et = np.einsum('ij,ij->j', Phi, self._Mfull @ Phi)
        return np.where(et > 0, ec / np.where(et > 0, et, 1.0), 0.0)

    def deck_dof(self, i):
        return 2 * i

    def stay_transverse(self, Phi):
        """Mode ordinates normal to the chord along the stay (top to bottom)."""
        oC = self.offs[1]
        c, s = np.cos(self.theta), np.sin(self.theta)
        n_vec = np.array([s, c])          # normal to the chord, common to both ends
        u = Phi[oC + 0:oC + 3 * (self.nc + 1):3, :]
        w = Phi[oC + 1:oC + 3 * (self.nc + 1):3, :]
        return n_vec[0] * u + n_vec[1] * w

    # -- references -------------------------------------------------------

    def stay_alone(self, nmodes=8):
        """Frequencies of the (sagged) stay with both ends held: Irvine's
        isolated cable, from the same elements."""
        xy, N, P, A = self.stay_nodes()
        Kc, Mc = polyline_chain(xy, self.EA, self.EIc, self.mc, N)
        n = Kc.shape[0]
        fixed = {0, 1, n - 3, n - 2}
        keep = [i for i in range(n) if i not in fixed]
        w2, V = eigh(Kc[np.ix_(keep, keep)], Mc[np.ix_(keep, keep)])
        f = np.sqrt(np.maximum(w2, 0.0)) / (2.0 * np.pi)
        return f[:nmodes]

    def host_alone(self, nmodes=12):
        """Deck (+ pylon) with the stay present only as a massless bar of
        stiffness EA / L_c between its two ends: the host of the reduction.

        Returns ``(f, phi_a, phi_p)``: frequencies, mass-normalised vertical
        ordinate at the anchorage, and horizontal ordinate at the pylon top
        (zero for a rigid pylon)."""
        Kd, Md = chain(self.Ld, self.nd, self.EId, self.md, 0.0)
        c, s = np.cos(self.theta), np.sin(self.theta)
        kt = self.EA / self.Lc
        if self.pylon is None:
            K, M = Kd.copy(), Md.copy()
            K[2 * self.ia, 2 * self.ia] += kt * s ** 2
            fixed = {0, 2 * self.nd}
            keep = [i for i in range(K.shape[0]) if i not in fixed]
            w2, V = eigh(K[np.ix_(keep, keep)], M[np.ix_(keep, keep)])
            f = np.sqrt(np.maximum(w2, 0.0)) / (2.0 * np.pi)
            pos = {d: j for j, d in enumerate(keep)}
            return f[:nmodes], V[pos[2 * self.ia], :nmodes], np.zeros(nmodes)
        xy, N, P, A = self.stay_nodes()
        pxy = self.pylon_nodes(P)
        Np = -self.T * s * np.ones(self.pylon["n"])
        Kp, Mp = polyline_chain(pxy, self.pylon["EA"], self.pylon["EI"],
                                self.pylon["m"], Np)
        nD, nP = Kd.shape[0], Kp.shape[0]
        K = np.zeros((nD + nP, nD + nP))
        M = np.zeros_like(K)
        K[:nD, :nD], M[:nD, :nD] = Kd, Md
        K[nD:, nD:], M[nD:, nD:] = Kp, Mp
        tip = nD + 3 * self.pylon["n"]
        # bar from P to A: elongation = (u_A - u_P) . e, e = (c, -s), with
        # u_A = (0, w_d) and u_P = (u_p, w_p)
        dofs = [tip + 0, tip + 1, 2 * self.ia]
        b = np.array([-c, s, -s])
        K[np.ix_(dofs, dofs)] += kt * np.outer(b, b)
        fixed = {0, 2 * self.nd, nD + 0, nD + 1, nD + 2}
        keep = [i for i in range(K.shape[0]) if i not in fixed]
        w2, V = eigh(K[np.ix_(keep, keep)], M[np.ix_(keep, keep)])
        f = np.sqrt(np.maximum(w2, 0.0)) / (2.0 * np.pi)
        pos = {d: j for j, d in enumerate(keep)}
        return (f[:nmodes], V[pos[2 * self.ia], :nmodes], V[pos[tip + 0], :nmodes])


# ---------------------------------------------------------------------------
# the reduction, one-ended and two-ended
# ---------------------------------------------------------------------------

def split_one_ended(M_s, phi_a, theta, n=1):
    """Eq. (5): s = (2 / n pi) cos(theta) sqrt(M_s) |phi_a|."""
    return 2.0 / (n * np.pi) * np.sqrt(M_s) * np.abs(np.cos(theta) * phi_a)


def split_two_ended(M_s, phi_a, phi_p, theta, n=1):
    """The stay driven at both ends: the deck ordinate through cos(theta) and
    the pylon-top ordinate through sin(theta), combined with the sign the
    stay mode's parity sets, since its end slopes are equal for odd orders
    and opposite for even ones:

        s = (2 / n pi) sqrt(M_s) |cos(theta) phi_a - (-1)^n sin(theta) phi_p|.

    ``phi_p`` is the host mode's horizontal ordinate at the pylon top, taken
    positive toward the anchorage, and ``phi_a`` its vertical ordinate at
    the anchorage, positive upward, both mass-normalised in the host.
    """
    c, s = np.cos(theta), np.sin(theta)
    return 2.0 / (n * np.pi) * np.sqrt(M_s) * np.abs(c * phi_a - (-1) ** n * s * phi_p)
