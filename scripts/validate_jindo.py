# -*- coding: utf-8 -*-
"""Jindo 1:150 physical model: a measured test of the veering-width law.

WHAT THIS SCRIPT IS FOR
-----------------------
Caetano's doctoral thesis (FEUP, 2001, Chapter 7) reports the only dataset
found in which the decoupled reference of the isolated-deck idealisation was
PHYSICALLY BUILT.  The Bristol 1:150 model of the Jindo cable-stayed bridge was
rebuilt for the ISMES shaking table with the distributed stay mass removed and
lumped at the anchorages (thesis Sec. 7.5.3.1), which eliminates cable dynamics
by construction.  Its first vertical anti-symmetric mode measured 9.02 Hz as a
single peak (Table 7.19).  The ORIGINAL distributed-mass model, measured at
Bristol, gave NINE modes of the same deck/tower configuration spread over
8.63-11.18 Hz (Tables 7.17 and 7.18).

The width between the coupled branches is therefore a MEASURED veering width
about a MEASURED decoupled reference.  The study's law

    mu_eff = M_s phi_a^2,   phi_a = mass-normalised structure ordinate at the
                                    anchorage (phi^T M phi = 1)
    s      = (2/(n pi)) cos(theta) sqrt(mu_eff)          [width at exact tuning]
    N stays tuned at once: bright pair splits by sqrt(sum_i s_i^2), N-1 dark

can be tested against it, PROVIDED the structure modal mass and the anchorage
ordinate are computed rather than assumed.  Computing them is the job here.

WHAT CANNOT BE TESTED HERE, AND WHY
-----------------------------------
There is no independent tension.  Each wire was tuned with a magnetic pickup to
its taut-string design frequency, and the thesis's "(Irvine)" column is
reproduced exactly by f_n = (n/2L) sqrt(T/m) from the tabulated (L, m, T).  The
tabulated tensions are therefore the tuning targets, not measurements.  The
TENSION-ERROR LAW IS NOT TESTED in this script.  Only the amplitude law (the
width) and the resolvability condition are.

MODEL
-----
A planar (vertical) frame model of the deck, the two towers and the piers, with
the stays as pretensioned bars.  Two configurations:

  "decoupled"  each stay's total mass M_s = m L split half to the deck
               anchorage and half to the tower anchorage, the stay itself
               massless.  This is the physically built ISMES configuration and
               the numerical OECS idealisation.
  "bare"       no stay mass anywhere.  The study's own zero-coupling reference.
  "quasistatic" M_s cos^2(theta)/3 at the deck anchorage only, which is the
               mass a taut string actually presents to a slowly moving support.

Every calibration is listed in CALIBRATIONS below.  Nothing was tuned to fit.

Outputs: printed report, plus data/jindo.csv in long format.
"""

from __future__ import annotations

import os
import sys
import csv
import numpy as np
from scipy.linalg import eigh, solve

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from cablefe import beam_element, mu_effective, veering_split, string_freq  # noqa: E402

G = 9.80665

ROWS = []   # long-format CSV accumulator


def rec(block, item, quantity, value, unit="", note=""):
    ROWS.append(dict(block=block, item=item, quantity=quantity,
                     value=value, unit=unit, note=note))


# =========================================================================
# 1.  THE TRANSCRIBED THESIS DATA.  Every number here is from the thesis.
# =========================================================================

# ---- Table 7.6, stay schedule (model, one stay per tower per cable plane) --
# x_anchor is the deck station in mm from the LEFT abutment for the LEFT tower.
# L, m, T are Table 7.6 verbatim.  dia is read off Figure 7.2.
# f1_irv and f1_fem are Table 7.6's "(Irvine)" and "FEM" first frequencies.
# Values transcribed from Caetano (2001), PhD thesis, University of Porto, Chapter 7 (Jindo 1:150 model). They are not redistributed with
# this code: they live in data/external/validate_jindo_data.py (see README).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "external"))
try:
    from validate_jindo_data import (  # noqa: E402
        STAYS, DECK_BOXES_HALF, TOWER_MASSES, CALIBRATIONS, DECOUPLED_MEASURED, OECS_VERTICAL, T717_VERT_ASM, T717_TRANSV_ASM, T717_VERT_SYM, MEAS_VERT_ASM_TABLE, MEAS_VERT_ASM_SHAKER, MECS_VERT_ASM)
except ImportError as exc:
    raise SystemExit("scripts/validate_jindo.py needs values transcribed from "
                     "Caetano (2001), PhD thesis, University of Porto, Chapter 7 (Jindo 1:150 model), which are not redistributed here. "
                     "See README, 'Third-party data'.") from exc

# ---- Table 7.5 + Figure 7.2, deck added-mass boxes, LEFT half (mm -> kg) ----

# ---- Table 7.5 + Figure 7.2, tower added masses, heights above deck (mm) ----

# ---- geometry, Figure 7.2 -------------------------------------------------
L_TOT = 3222.0e-3          # total model length, m
X_TOWER_L = 467.0e-3
X_TOWER_R = L_TOT - X_TOWER_L
X_MID = L_TOT / 2.0
Z_TOWER_TOP = 460.0e-3
Z_PIER_BASE = -170.0e-3

# ---- materials and sections, thesis p. 7.12 -------------------------------
E_AL = 72.0e9              # aluminium alloy, girder and tower legs
E_WIRE = 210.0e9           # piano wire
RHO_AL = 2700.0

B_DECK, H_DECK = 25.4e-3, 7.92e-3           # width x depth, deck
A_DECK = B_DECK * H_DECK
I_DECK_V = B_DECK * H_DECK ** 3 / 12.0      # vertical bending
I_DECK_L = H_DECK * B_DECK ** 3 / 12.0      # lateral bending (reported only)
M_DECK = RHO_AL * A_DECK

B_LEG, H_LEG = 9.2e-3, 8.4e-3               # tower leg, 9.2 longitudinal
A_LEG = B_LEG * H_LEG
I_LEG_LONG = H_LEG * B_LEG ** 3 / 12.0      # bending in the vertical plane
I_LEG_TRAN = B_LEG * H_LEG ** 3 / 12.0
N_LEG = 2                                   # legs per A-tower

EA_TOWER = N_LEG * E_AL * A_LEG
EI_TOWER = N_LEG * E_AL * I_LEG_LONG
M_TOWER = N_LEG * RHO_AL * A_LEG

EA_DECK = E_AL * A_DECK
EI_DECK = E_AL * I_DECK_V

N_PLANES = 2                                # upstream and downstream stays



# =========================================================================
# 2.  DERIVED STAY GEOMETRY
# =========================================================================

def axial_scale_check():
    """Independent check that the wire diameters and EA are right.

    The thesis says the stay axial stiffness was correctly scaled, so
    EA_model should equal EA_prototype / (sE sL^2) = EA_prototype / 69075.
    Prototype areas and E come from Table 7.2; model areas from the Figure 7.2
    wire diameters.  Nothing in this function feeds the model; it only tests
    the reconstruction.
    """
    E_P = 157.0e9              # Table 7.2 header, "157Mpa" as printed
    SF = 3.07 * 150.0 ** 2
    proto = {1: 6 * 4990.0, 2: 2990.0, 3: 2130.0, 4: 2130.0, 5: 2130.0,
             6: 2990.0, 7: 2990.0, 8: 3810.0, 9: 3810.0, 10: 3810.0,
             11: 3810.0, 12: 4990.0}       # model stay i <- prototype rope i-1
    out = []
    for s in stay_geometry():
        EAp = E_P * proto[s['no']] * 1e-6 / SF
        out.append((s['no'], s['dia'] * 1e3, s['EA'], EAp,
                    100 * (s['EA'] - EAp) / EAp))
    return out


def stay_geometry():
    """Per-stay chord geometry, all in SI.  Returns a list of dicts."""
    out = []
    for (no, xa, L, m, T, dia, f1i, f1f) in STAYS:
        xa_m = xa * 1e-3
        L_m = L * 1e-3
        dx = abs(xa_m - X_TOWER_L)
        h2 = L_m ** 2 - dx ** 2
        h = np.sqrt(h2)
        theta = np.arctan2(h, dx)
        A = np.pi * (dia * 1e-3) ** 2 / 4.0
        out.append(dict(no=no, x=xa_m, L=L_m, m=m, T=T, dia=dia * 1e-3,
                        A=A, EA=E_WIRE * A, dx=dx, h=h, theta=theta,
                        cos=np.cos(theta), Ms=m * L_m,
                        f1_irv=f1i, f1_fem=f1f,
                        f1_check=string_freq(1, L_m, T, m)))
    return out


# =========================================================================
# 3.  PLANAR FRAME / BAR FINITE ELEMENT MODEL
# =========================================================================

def frame_elem(l, EA, EI, m, N=0.0):
    """6-DOF planar frame element, local DOF [u1,w1,th1,u2,w2,th2].

    Bending and its geometric stiffness come from cablefe.beam_element, so the
    arithmetic is the study's own.  N is positive in tension.
    """
    kb, mb = beam_element(l, EI, m, N)
    k = np.zeros((6, 6))
    mm = np.zeros((6, 6))
    ax = EA / l * np.array([[1.0, -1.0], [-1.0, 1.0]])
    ma = m * l / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
    k[np.ix_([0, 3], [0, 3])] += ax
    mm[np.ix_([0, 3], [0, 3])] += ma
    idx = [1, 2, 4, 5]
    k[np.ix_(idx, idx)] += kb
    mm[np.ix_(idx, idx)] += mb
    return k, mm


def rot6(c, s):
    t = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    T = np.zeros((6, 6))
    T[:3, :3] = t
    T[3:, 3:] = t
    return T


def rot4(c, s):
    t = np.array([[c, s], [-s, c]])
    T = np.zeros((4, 4))
    T[:2, :2] = t
    T[2:, 2:] = t
    return T


class Frame2D:
    """Minimal planar frame + pretensioned-bar assembler, 3 DOF per node."""

    def __init__(self):
        self.xy = []        # (x, z)
        self.frames = []    # dict(i, j, EA, EI, m)
        self.bars = []      # dict(i, j, EA, T)
        self.lumped = {}    # node -> mass
        self.fixed = set()
        self.ties = []      # (dep_dof, master_dof, coeff)

    # -- topology ---------------------------------------------------------
    def node(self, x, z, tol=5e-4):
        for k, (xx, zz) in enumerate(self.xy):
            if abs(xx - x) < tol and abs(zz - z) < tol:
                return k
        self.xy.append((x, z))
        return len(self.xy) - 1

    def add_frame(self, i, j, EA, EI, m):
        self.frames.append(dict(i=i, j=j, EA=EA, EI=EI, m=m))

    def add_bar(self, i, j, EA, T):
        self.bars.append(dict(i=i, j=j, EA=EA, T=T))

    def add_mass(self, i, mass):
        self.lumped[i] = self.lumped.get(i, 0.0) + mass

    def fix(self, i, dofs=(0, 1, 2)):
        for d in dofs:
            self.fixed.add(3 * i + d)

    def tie(self, dep_node, master_node, dof, coeff=1.0):
        self.ties.append((3 * dep_node + dof, 3 * master_node + dof, coeff))

    # -- geometry helpers -------------------------------------------------
    def _dir(self, i, j):
        xi, zi = self.xy[i]
        xj, zj = self.xy[j]
        dx, dz = xj - xi, zj - zi
        l = np.hypot(dx, dz)
        return l, dx / l, dz / l

    # -- assembly ---------------------------------------------------------
    def assemble(self, Nf=None, Nb=None):
        nd = 3 * len(self.xy)
        K = np.zeros((nd, nd))
        M = np.zeros((nd, nd))
        for e, fr in enumerate(self.frames):
            l, c, s = self._dir(fr['i'], fr['j'])
            N = 0.0 if Nf is None else Nf[e]
            k, mm = frame_elem(l, fr['EA'], fr['EI'], fr['m'], N)
            T = rot6(c, s)
            k = T.T @ k @ T
            mm = T.T @ mm @ T
            g = np.array([3 * fr['i'], 3 * fr['i'] + 1, 3 * fr['i'] + 2,
                          3 * fr['j'], 3 * fr['j'] + 1, 3 * fr['j'] + 2])
            K[np.ix_(g, g)] += k
            M[np.ix_(g, g)] += mm
        for e, br in enumerate(self.bars):
            l, c, s = self._dir(br['i'], br['j'])
            N = br['T'] if Nb is None else Nb[e]
            ka, kt = br['EA'] / l, N / l
            blk = np.array([[ka, 0.0], [0.0, kt]])
            k = np.zeros((4, 4))
            k[np.ix_([0, 1], [0, 1])] = blk
            k[np.ix_([2, 3], [2, 3])] = blk
            k[np.ix_([0, 1], [2, 3])] = -blk
            k[np.ix_([2, 3], [0, 1])] = -blk
            T = rot4(c, s)
            k = T.T @ k @ T
            g = np.array([3 * br['i'], 3 * br['i'] + 1,
                          3 * br['j'], 3 * br['j'] + 1])
            K[np.ix_(g, g)] += k
        for n, mass in self.lumped.items():
            M[3 * n, 3 * n] += mass
            M[3 * n + 1, 3 * n + 1] += mass
        return K, M

    # -- constraint reduction ---------------------------------------------
    def reduction(self):
        nd = 3 * len(self.xy)
        dep = {d for d, _, _ in self.ties}
        free = [d for d in range(nd) if d not in self.fixed and d not in dep]
        pos = {d: j for j, d in enumerate(free)}
        L = np.zeros((nd, len(free)))
        for d, j in pos.items():
            L[d, j] = 1.0
        for d, mst, c in self.ties:
            if mst in pos:
                L[d, pos[mst]] = c
        return L

    # -- loads and internal forces ----------------------------------------
    def gravity(self):
        f = np.zeros(3 * len(self.xy))
        for n, mass in self.lumped.items():
            f[3 * n + 1] -= mass * G
        for fr in self.frames:
            l, _, _ = self._dir(fr['i'], fr['j'])
            w = fr['m'] * l * G / 2.0
            f[3 * fr['i'] + 1] -= w
            f[3 * fr['j'] + 1] -= w
        return f

    def pretension_load(self):
        f = np.zeros(3 * len(self.xy))
        for br in self.bars:
            l, c, s = self._dir(br['i'], br['j'])
            f[3 * br['i']] += br['T'] * c
            f[3 * br['i'] + 1] += br['T'] * s
            f[3 * br['j']] -= br['T'] * c
            f[3 * br['j'] + 1] -= br['T'] * s
        return f

    def member_forces(self, u):
        Nf = []
        for fr in self.frames:
            l, c, s = self._dir(fr['i'], fr['j'])
            ai = c * u[3 * fr['i']] + s * u[3 * fr['i'] + 1]
            aj = c * u[3 * fr['j']] + s * u[3 * fr['j'] + 1]
            Nf.append(fr['EA'] / l * (aj - ai))
        Nb = []
        for br in self.bars:
            l, c, s = self._dir(br['i'], br['j'])
            ai = c * u[3 * br['i']] + s * u[3 * br['i'] + 1]
            aj = c * u[3 * br['j']] + s * u[3 * br['j'] + 1]
            Nb.append(br['T'] + br['EA'] / l * (aj - ai))
        return np.array(Nf), np.array(Nb)


# =========================================================================
# 4.  BUILD THE JINDO MODEL IN THE VERTICAL PLANE
# =========================================================================

def deck_stations(max_seg=0.040):
    """Symmetric list of deck x stations, in m, subdivided to <= max_seg."""
    key = [0.0, 127.0, 282.0, 374.5, 467.0, 559.5, 652.0, 763.0, 874.0,
           985.0, 1096.0, 1207.0, 1318.0, 1429.0, 1540.0, 1611.0]
    key = [k * 1e-3 for k in key]
    half = []
    for a, b in zip(key[:-1], key[1:]):
        n = max(1, int(np.ceil((b - a) / max_seg)))
        for q in range(n):
            half.append(a + (b - a) * q / n)
    half.append(key[-1])
    full = half + [L_TOT - x for x in half[-2::-1]]
    return sorted(set(round(x, 9) for x in full))


def build(cable_mass="decoupled", geom_stiff=True, rigid_pier=False,
          max_seg=0.040, tower_top_mass=True):
    """Assemble the planar model.

    cable_mass : "decoupled"   M_s/2 at each anchorage  (the ISMES build)
                 "bare"        no stay mass at all      (study's d=0 reference)
                 "quasistatic" M_s cos^2(theta)/3 at the deck end only
    """
    g = stay_geometry()
    fm = Frame2D()

    # ---- deck spine -----------------------------------------------------
    xs = deck_stations(max_seg)
    dn = {}
    for x in xs:
        dn[round(x, 9)] = fm.node(x, 0.0)
    for a, b in zip(xs[:-1], xs[1:]):
        fm.add_frame(dn[round(a, 9)], dn[round(b, 9)],
                     EA_DECK, EI_DECK, M_DECK)

    # ---- deck added masses (both halves) --------------------------------
    for (xmm, mass) in DECK_BOXES_HALF:
        for x in (xmm * 1e-3, L_TOT - xmm * 1e-3):
            fm.add_mass(dn[round(x, 9)], mass)

    # ---- towers ---------------------------------------------------------
    tower_nodes = {}
    for xt in (X_TOWER_L, X_TOWER_R):
        zlist = [Z_PIER_BASE, 0.0, Z_TOWER_TOP]
        zlist += [z * 1e-3 for z, _ in TOWER_MASSES]
        zlist += [s['h'] for s in g]
        zlist = sorted(set(round(z, 6) for z in zlist))
        # merge nodes closer than 0.5 mm
        merged = [zlist[0]]
        for z in zlist[1:]:
            if z - merged[-1] > 5e-4:
                merged.append(z)
        zn = {}
        for z in merged:
            zn[z] = fm.node(xt, z)
        for a, b in zip(merged[:-1], merged[1:]):
            fm.add_frame(zn[a], zn[b], EA_TOWER, EI_TOWER, M_TOWER)
        for (zmm, mass) in TOWER_MASSES:
            z = min(merged, key=lambda q: abs(q - zmm * 1e-3))
            fm.add_mass(zn[z], mass if tower_top_mass else 0.0)
        tower_nodes[xt] = (zn, merged)
        # pier base clamped
        fm.fix(zn[merged[0]], (0, 1, 2))
        if rigid_pier:
            fm.fix(zn[0.0], (0, 1, 2))

    # ---- deck supports --------------------------------------------------
    # abutments: vertical support only (rocker), longitudinal and rotation free
    fm.fix(dn[round(0.0, 9)], (1,))
    fm.fix(dn[round(L_TOT, 9)], (1,))
    # deck-tower junctions: equal vertical at both, equal longitudinal at the
    # right (North) tower only.  Thesis p. 7.23 (iv) and Figure 7.6 symbols.
    znL, mgL = tower_nodes[X_TOWER_L]
    znR, mgR = tower_nodes[X_TOWER_R]
    fm.tie(dn[round(X_TOWER_L, 9)], znL[0.0], 1)
    fm.tie(dn[round(X_TOWER_R, 9)], znR[0.0], 1)
    fm.tie(dn[round(X_TOWER_R, 9)], znR[0.0], 0)

    # ---- stays ----------------------------------------------------------
    stay_index = []      # (bar_index, stay dict, tower_x, deck_node, tower_node)
    for xt, zn, sign in ((X_TOWER_L, znL, +1.0), (X_TOWER_R, znR, -1.0)):
        for s in g:
            xa = xt + sign * (s['x'] - X_TOWER_L)
            i = dn[round(xa, 9)]
            zt = min(zn.keys(), key=lambda q: abs(q - s['h']))
            j = zn[zt]
            for _ in range(N_PLANES):
                fm.add_bar(i, j, s['EA'], s['T'])
                stay_index.append(dict(bar=len(fm.bars) - 1, s=s, xt=xt,
                                       deck=i, top=j, xa=xa))
            if cable_mass == "decoupled":
                fm.add_mass(i, N_PLANES * s['Ms'] / 2.0)
                fm.add_mass(j, N_PLANES * s['Ms'] / 2.0)
            elif cable_mass == "quasistatic":
                fm.add_mass(i, N_PLANES * s['Ms'] * s['cos'] ** 2 / 3.0)
            elif cable_mass == "bare":
                pass
            else:
                raise ValueError(cable_mass)

    fm.deck_nodes = dn
    fm.deck_xs = xs
    fm.tower_nodes = tower_nodes
    fm.stay_index = stay_index
    fm.geom_stiff = geom_stiff
    return fm


def solve_model(fm, nmodes=40):
    """Prestress static step, then the eigenproblem.  Returns a result dict."""
    L = fm.reduction()
    f = fm.gravity() + fm.pretension_load()
    Nf = Nb = None
    u = np.zeros(3 * len(fm.xy))
    for _ in range(4 if fm.geom_stiff else 1):
        K, M = fm.assemble(Nf, Nb)
        Kr = L.T @ K @ L
        ur = solve(Kr, L.T @ f, assume_a='sym')
        u = L @ ur
        if not fm.geom_stiff:
            Nf = np.zeros(len(fm.frames))
            Nb = np.array([b['T'] for b in fm.bars])
            break
        Nf, Nb = fm.member_forces(u)
    K, M = fm.assemble(Nf, Nb)
    Kr, Mr = L.T @ K @ L, L.T @ M @ L
    w2, V = eigh(Kr, Mr)
    w2 = np.maximum(w2, 0.0)
    fr = np.sqrt(w2) / (2.0 * np.pi)
    k = min(nmodes, len(fr))
    Phi = L @ V[:, :k]
    return dict(fm=fm, f=fr[:k], Phi=Phi, u=u, Nf=Nf, Nb=Nb, L=L, K=K, M=M)


# =========================================================================
# 5.  MODE CLASSIFICATION
# =========================================================================

def deck_shape(res, mode):
    fm = res['fm']
    xs = np.array(fm.deck_xs)
    w = np.array([res['Phi'][3 * fm.deck_nodes[round(x, 9)] + 1, mode]
                  for x in xs])
    return xs, w


def classify(res, mode):
    """Return (kind, sym, nzero, deck_ke_fraction) for one mode."""
    fm = res['fm']
    xs, w = deck_shape(res, mode)
    wm = w[::-1]                                    # mirrored (xs is symmetric)
    denom = float(w @ w)
    sym = float(w @ wm) / denom if denom > 0 else 0.0
    # zero crossings inside the main span
    inmain = (xs > X_TOWER_L + 1e-9) & (xs < X_TOWER_R - 1e-9)
    ww = w[inmain]
    nz = int(np.sum(np.sign(ww[:-1]) * np.sign(ww[1:]) < 0))
    # kinetic energy fraction on the deck
    M = res['M']
    p = res['Phi'][:, mode]
    dofs_deck = []
    for x in fm.deck_xs:
        n = fm.deck_nodes[round(x, 9)]
        dofs_deck += [3 * n, 3 * n + 1, 3 * n + 2]
    mask = np.zeros(M.shape[0], dtype=bool)
    mask[dofs_deck] = True
    Md = M.copy()
    Md[~mask, :] = 0.0
    Md[:, ~mask] = 0.0
    ke_deck = float(p @ Md @ p)
    ke_tot = float(p @ M @ p)
    kind = 'SYM' if sym > 0.3 else ('ASM' if sym < -0.3 else 'mixed')
    return kind, sym, nz, (ke_deck / ke_tot if ke_tot > 0 else 0.0)


def label_vertical(res, nshow=20):
    """Identify the vertical deck modes and give them thesis-style labels."""
    out = []
    nsym = nasm = 0
    for m in range(len(res['f'])):
        kind, sym, nz, kef = classify(res, m)
        if kef < 0.30:
            out.append((m, res['f'][m], 'tower/local', sym, nz, kef))
            continue
        if kind == 'SYM':
            nsym += 1
            lab = '%d%s vert. SYM' % (nsym, 'st' if nsym == 1 else
                                      ('nd' if nsym == 2 else
                                       ('rd' if nsym == 3 else 'th')))
        elif kind == 'ASM':
            nasm += 1
            lab = '%d%s vert. ASM' % (nasm, 'st' if nasm == 1 else
                                      ('nd' if nasm == 2 else
                                       ('rd' if nasm == 3 else 'th')))
        else:
            lab = 'mixed'
        out.append((m, res['f'][m], lab, sym, nz, kef))
        if len(out) >= nshow:
            break
    return out


# =========================================================================
# 6.  MEASURED DATA, THESIS TABLES 7.17 / 7.18 / 7.19
# =========================================================================

# Table 7.19: the MODIFIED (decoupled) model on the ISMES table.
#   type : (identified f, zeta_lo %, zeta_hi %, OECS calculated f)

# Table 7.9, 3-D OECS calculated, vertical family only

# Table 7.17, ORIGINAL distributed-mass model.  (f, zeta_lo, zeta_hi) per
# excitation technique; None where that technique did not identify the mode.
# Rows are the thesis's own row order, which pairs the two columns.

# Table 7.18, shaking-table identified groups
MEAS_VERT_SYM_SHAKER = [6.25, 6.69, 7.12]
MEAS_VERT_SYM_TABLE = [6.27, 6.73]

# Table 7.10, 3-D MECS: cable/beam maximum-displacement ratio in Z for the
# modes the thesis assigns to the 1st vertical ASM family.  Small = structural.

# The study's resolvability thresholds
K_DIP = 0.9717      # a dip survives while s > K_DIP * zeta
K_3DB = 2.280       # the dip is at least 3 dB while s > K_3DB * zeta


# =========================================================================
# 7.  THE LAW
# =========================================================================

def anchor_data(res, mode, geom):
    """Per stay copy: ordinates, mu_eff and s at the anchorage of one mode."""
    fm = res['fm']
    Phi = res['Phi']
    out = []
    for si in fm.stay_index:
        s = si['s']
        i, j = si['deck'], si['top']
        xi, zi = fm.xy[i]
        xj, zj = fm.xy[j]
        L = np.hypot(xj - xi, zj - zi)
        d = np.array([xj - xi, zj - zi]) / L          # chord unit, deck -> top
        t = np.array([-d[1], d[0]])                   # transverse unit
        pd = np.array([Phi[3 * i, mode], Phi[3 * i + 1, mode]])
        pt = np.array([Phi[3 * j, mode], Phi[3 * j + 1, mode]])
        phi_a = pd[1]                                 # vertical deck ordinate
        mu = mu_effective(s['Ms'], phi_a)
        s_study = veering_split(mu, s['theta'], n=1)
        drive = float(pd @ t + pt @ t)                # n = 1 generalised drive
        s_gen = 2.0 / np.pi * np.sqrt(s['Ms']) * abs(drive)
        # Revision 1: the second route.  A sagged wire is also driven by the
        # along-chord pull at the deck anchorage through its dynamic tension,
        # in the ratio r_1 = (2/pi^2) sin(theta) (EA/T)(m g L/T) to the tie
        # and with the opposite sign, so the deck-side coupling carries the
        # factor (1 - r_1); the tower-top drive is left as it is.
        r1 = 2.0 / np.pi ** 2 * np.sin(s['theta']) * (s['EA'] / s['T']) \
            * (s['m'] * G * s['L'] / s['T'])
        s_two = s_study * abs(1.0 - r1)
        drive_two = float((pd @ t) * (1.0 - r1) + pt @ t)
        s_gen_two = 2.0 / np.pi * np.sqrt(s['Ms']) * abs(drive_two)
        out.append(dict(no=s['no'], xt=si['xt'], xa=si['xa'], Ms=s['Ms'],
                        theta=s['theta'], cos=s['cos'], phi_a=phi_a,
                        mu_eff=mu, s_study=s_study, s_gen=s_gen,
                        r1=r1, s_two=s_two, s_gen_two=s_gen_two,
                        drive_deck=float(pd @ t), drive_top=float(pt @ t),
                        Mmodal=(1.0 / phi_a ** 2 if abs(phi_a) > 1e-12
                                else np.inf),
                        f1_irv=s['f1_irv'], f1_fem=s['f1_fem']))
    return out


def bordered_branches(f0, f_stays, s_list):
    """N+1 branches of one structure mode coupled to N stays.

    A = diag(f0^2, f1^2, ... fN^2) with A[0,i] = s_i f0^2.  At exact tuning
    with one stay the roots are f0 sqrt(1 +- s), i.e. a normalised split s,
    which is exactly cablefe.veering_split.  With all stays tuned the split is
    sqrt(sum s_i^2), which is the bordered-pencil rule the study states.
    Returns (frequencies ascending, bright fraction of each branch).
    """
    n = len(f_stays)
    A = np.zeros((n + 1, n + 1))
    A[0, 0] = f0 ** 2
    for k, (fi, si) in enumerate(zip(f_stays, s_list)):
        A[k + 1, k + 1] = fi ** 2
        A[0, k + 1] = A[k + 1, 0] = si * f0 ** 2
    w, V = np.linalg.eigh(A)
    f = np.sqrt(np.maximum(w, 0.0))
    bright = V[0, :] ** 2
    o = np.argsort(f)
    return f[o], bright[o]


# =========================================================================
# 8.  REPORT
# =========================================================================

def hr(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def find_mode(res, label):
    for (m, f, lab, sym, nz, kef) in label_vertical(res, 16):
        if lab == label:
            return m, f
    return None, None


def step1_verify(variants):
    hr("STEP 1.  THE DECOUPLED MODEL, BUILT AND VERIFIED")
    print("Configuration built: stay mass M_s = m L removed from the wire and")
    print("split M_s/2 to the deck anchorage, M_s/2 to the tower anchorage.")
    print("That is the ISMES build (thesis Sec. 7.5.3.1) and the OECS idealisation.")
    print()
    print("Calibrations made (none of them tuned to a measured frequency):")
    for c in CALIBRATIONS:
        print("  - " + c)
    print()
    labels = ['1st vert. SYM', '1st vert. ASM', '2nd vert. SYM', '2nd vert. ASM',
              '3rd vert. SYM', '3rd vert. ASM', '4th vert. SYM', '4th vert. ASM']
    meas = {'1st vert. SYM': 6.00, '1st vert. ASM': 9.02,
            '2nd vert. SYM': 13.70, '2nd vert. ASM': 18.30}
    oecs = dict((k.replace('vert.', 'vert.'), v) for k, v in OECS_VERTICAL)
    print("%-22s %s" % ("", "".join("%13s" % l[:13] for l in labels)))
    for name, res in variants:
        d = {}
        for (m, f, lab, sym, nz, kef) in label_vertical(res, 16):
            d.setdefault(lab, f)
        print("%-22s %s" % (name, "".join("%13.3f" % d.get(l, np.nan)
                                          for l in labels)))
        for l in labels:
            if l in d:
                rec("verify", name, l, round(d[l], 4), "Hz")
    print("%-22s %s" % ("MEASURED Tab 7.19",
                        "".join("%13s" % (("%.2f" % meas[l]) if l in meas else "-")
                                for l in labels)))
    print("%-22s %s" % ("OECS calc Tab 7.9",
                        "".join("%13s" % (("%.2f" % oecs[l]) if l in oecs else "-")
                                for l in labels)))
    base = dict()
    for (m, f, lab, sym, nz, kef) in label_vertical(variants[0][1], 16):
        base.setdefault(lab, f)
    print()
    print("Errors of the built model against the two references:")
    print("%-16s %10s %10s %10s %10s %10s" %
          ("mode", "FE (Hz)", "meas (Hz)", "err %", "OECS (Hz)", "err %"))
    errs_m, errs_o = [], []
    for l in labels:
        fe = base.get(l, np.nan)
        mv = meas.get(l)
        ov = oecs.get(l)
        em = 100 * (fe - mv) / mv if mv else np.nan
        eo = 100 * (fe - ov) / ov if ov else np.nan
        if mv:
            errs_m.append(em)
        if ov:
            errs_o.append(eo)
        print("%-16s %10.3f %10s %10s %10s %10s" %
              (l, fe, ("%.2f" % mv) if mv else "-",
               ("%+.1f" % em) if mv else "-",
               ("%.2f" % ov) if ov else "-",
               ("%+.1f" % eo) if ov else "-"))
        rec("verify_error", l, "fe_hz", round(fe, 4), "Hz")
        if mv:
            rec("verify_error", l, "measured_hz", mv, "Hz")
            rec("verify_error", l, "error_vs_measured_pct", round(em, 3), "%")
        if ov:
            rec("verify_error", l, "oecs_hz", ov, "Hz")
            rec("verify_error", l, "error_vs_oecs_pct", round(eo, 3), "%")
    print()
    print("mean error vs measured  %+.2f %%   (n=%d, rms %.2f %%)"
          % (np.mean(errs_m), len(errs_m), np.sqrt(np.mean(np.square(errs_m)))))
    print("mean error vs OECS      %+.2f %%   (n=%d, rms %.2f %%)"
          % (np.mean(errs_o), len(errs_o), np.sqrt(np.mean(np.square(errs_o)))))
    rec("verify_summary", "all", "mean_err_vs_measured_pct",
        round(float(np.mean(errs_m)), 3), "%")
    rec("verify_summary", "all", "rms_err_vs_measured_pct",
        round(float(np.sqrt(np.mean(np.square(errs_m)))), 3), "%")
    rec("verify_summary", "all", "mean_err_vs_oecs_pct",
        round(float(np.mean(errs_o)), 3), "%")
    return base


def step2_ordinates(res_by_cfg, mode_label='1st vert. ASM'):
    hr("STEP 2.  MASS-NORMALISED ANCHORAGE ORDINATES AND mu_eff")
    out = {}
    for cfg, res in res_by_cfg.items():
        m, f = find_mode(res, mode_label)
        if m is None:
            continue
        ad = anchor_data(res, m, None)
        # one entry per (stay, tower); the two cable planes are identical
        seen = {}
        for a in ad:
            seen[(a['no'], round(a['xt'], 6))] = a
        out[cfg] = (f, seen)
    cfg0 = 'decoupled'
    f0, seen0 = out[cfg0]
    print("Tuned mode: %s of the decoupled model, FE %.3f Hz "
          "(measured 9.02 Hz, OECS 9.12 Hz)." % (mode_label, f0))
    print()
    print("Per stay, LEFT tower (the right tower is the mirror, |phi_a| equal):")
    print("%4s %9s %9s %11s %11s %10s %9s %9s %9s %9s" %
          ("stay", "x_a (mm)", "M_s (kg)", "phi_a", "1/phi_a^2",
           "mu_eff", "s_study", "v_deck", "v_tower", "s_gen"))
    for no in range(1, 13):
        a = seen0[(no, round(X_TOWER_L, 6))]
        print("%4d %9.1f %9.5f %11.5f %11.2f %10.3e %8.3f%% %9.5f %9.5f %8.3f%%" %
              (no, 1e3 * a['xa'], a['Ms'], a['phi_a'], a['Mmodal'],
               a['mu_eff'], 100 * a['s_study'], a['drive_deck'],
               a['drive_top'], 100 * a['s_gen']))
        rec("ordinates", "stay %d" % no, "phi_a", round(a['phi_a'], 6),
            "kg^-1/2", "%s, %s" % (cfg0, mode_label))
        rec("ordinates", "stay %d" % no, "modal_mass_1_over_phi2",
            round(a['Mmodal'], 4), "kg", mode_label)
        rec("ordinates", "stay %d" % no, "mu_eff", float("%.6g" % a['mu_eff']),
            "-", mode_label)
        rec("ordinates", "stay %d" % no, "s_study_pct",
            round(100 * a['s_study'], 4), "%", "one stay, rigid-pylon law")
        rec("ordinates", "stay %d" % no, "s_generalised_pct",
            round(100 * a['s_gen'], 4), "%", "includes tower-top motion")
        rec("ordinates", "stay %d" % no, "r1_second_route", round(a['r1'], 4), "",
            "(2/pi^2) sin(theta) (EA/T)(m g L/T)")
        rec("ordinates", "stay %d" % no, "s_two_route_pct",
            round(100 * a['s_two'], 4), "%", "s_study x |1 - r_1|")
        rec("ordinates", "stay %d" % no, "s_two_route_tower_pct",
            round(100 * a['s_gen_two'], 4), "%", "deck term x (1 - r_1) plus tower top")
    print()
    print("Sensitivity of phi_a to how the stay mass is treated:")
    print("%-14s %9s %s" % ("configuration", "f0 (Hz)",
                            "".join("%10s" % ("stay %d" % n)
                                    for n in (7, 8, 9, 10, 11, 12))))
    for cfg in ('decoupled', 'bare', 'quasistatic', 'rigid_pier', 'no_Kg'):
        if cfg not in out:
            continue
        fq, sq = out[cfg]
        print("%-14s %9.3f %s" % (cfg, fq,
              "".join("%10.5f" % sq[(n, round(X_TOWER_L, 6))]['phi_a']
                      for n in (7, 8, 9, 10, 11, 12))))
        for n in range(1, 13):
            rec("phi_a_sensitivity", cfg, "stay %d phi_a" % n,
                round(sq[(n, round(X_TOWER_L, 6))]['phi_a'], 6), "kg^-1/2")
        rec("phi_a_sensitivity", cfg, "f0", round(fq, 4), "Hz")
    print()
    tot = sum(res_by_cfg['decoupled']['fm'].lumped.values()) + sum(
        fr['m'] * res_by_cfg['decoupled']['fm']._dir(fr['i'], fr['j'])[0]
        for fr in res_by_cfg['decoupled']['fm'].frames)
    print("Physical mass of the modelled structure: %.2f kg "
          "(thesis states 'total mass of superstructure ~= 50 kg')." % tot)
    a8 = seen0[(8, round(X_TOWER_L, 6))]
    a9 = seen0[(9, round(X_TOWER_L, 6))]
    print("Implied structure modal mass at the stay-8 anchorage 1/phi_a^2 = "
          "%.2f kg, at stay 9 %.2f kg." % (a8['Mmodal'], a9['Mmodal']))
    print("Both are the right order against %.1f kg physical, as they must be: "
          "1/phi_a^2 is the mass an oscillator at that point would need to "
          "carry the whole modal kinetic energy, so it exceeds the physical "
          "mass wherever the mode ordinate is below its maximum." % tot)
    rec("modal_mass", "structure", "physical_mass_modelled", round(tot, 3), "kg")
    rec("modal_mass", "stay 8 anchorage", "1_over_phi_a2",
        round(a8['Mmodal'], 3), "kg")
    rec("modal_mass", "stay 9 anchorage", "1_over_phi_a2",
        round(a9['Mmodal'], 3), "kg")
    return out


ALL_OUT = {}


def step3_width(out, f0_meas=9.02):
    ALL_OUT.update(out)
    hr("STEP 3.  PREDICTED WIDTH")
    f_fe, seen = out['decoupled']
    stays = [seen[(n, round(X_TOWER_L, 6))] for n in range(1, 13)]
    print("Every stay exists in FOUR copies: two towers x two cable planes.")
    print("All four have the same |phi_a|, so each stay number contributes")
    print("sqrt(4) s_i = 2 s_i to the bright split and leaves three dark modes.")
    print()
    print("Detuning of each stay from the measured decoupled reference %.2f Hz:"
          % f0_meas)
    print("%4s %9s %9s %9s %9s %9s %9s" %
          ("stay", "f_Irvine", "d_Irv %", "f_FEM", "d_FEM %",
           "s_i (%)", "4-copy (%)"))
    for a in stays:
        d_i = 100 * (a['f1_irv'] - f0_meas) / f0_meas
        d_f = 100 * (a['f1_fem'] - f0_meas) / f0_meas
        print("%4d %9.2f %+9.1f %9.2f %+9.1f %9.3f %9.3f" %
              (a['no'], a['f1_irv'], d_i, a['f1_fem'], d_f,
               100 * a['s_study'], 200 * a['s_study']))
        rec("detuning", "stay %d" % a['no'], "d_irvine_pct", round(d_i, 3), "%")
        rec("detuning", "stay %d" % a['no'], "d_fem_pct", round(d_f, 3), "%")
    print()
    subsets = [("stay 8 only", [8]),
               ("stays 8, 9", [8, 9]),
               ("stays 8, 9, 10", [8, 9, 10]),
               ("stays 7-11", [7, 8, 9, 10, 11]),
               ("all 12 stays", list(range(1, 13)))]
    print("Exact-tuning width sqrt(sum s_i^2) over four copies of each stay in")
    print("the subset (the study's bordered-pencil rule at exact tuning):")
    print("%-18s %10s %12s %12s %12s %12s" % ("subset", "N copies", "s_study %", "s_gen %",
                                              "two-route %", "two+tower %"))
    for name, ss in subsets:
        sel = [a for a in stays if a['no'] in ss]
        w = 2.0 * np.sqrt(sum(a['s_study'] ** 2 for a in sel))
        wg = 2.0 * np.sqrt(sum(a['s_gen'] ** 2 for a in sel))
        w2 = 2.0 * np.sqrt(sum(a['s_two'] ** 2 for a in sel))
        wg2 = 2.0 * np.sqrt(sum(a['s_gen_two'] ** 2 for a in sel))
        print("%-18s %10d %11.3f%% %11.3f%% %11.3f%% %11.3f%%" % (name, 4 * len(sel),
                                                100 * w, 100 * wg, 100 * w2, 100 * wg2))
        rec("width_exact_tuning", name, "s_combined_two_route_pct",
            round(100 * w2, 4), "%", "deck-side coupling x (1 - r_1)")
        rec("width_exact_tuning", name, "s_combined_two_route_tower_pct",
            round(100 * wg2, 4), "%", "with tower-top drive as well")
        rec("width_exact_tuning", name, "s_combined_study_pct",
            round(100 * w, 4), "%", "sqrt(sum s_i^2), 4 copies per stay")
        rec("width_exact_tuning", name, "s_combined_generalised_pct",
            round(100 * wg, 4), "%")
    print()
    print("How much of this survives the modelling uncertainty?  The same")
    print("sqrt(sum s_i^2) over stays 8 and 9, computed in every FE variant:")
    for cfg in ('decoupled', 'bare', 'quasistatic', 'rigid_pier', 'no_Kg'):
        if cfg not in ALL_OUT:
            continue
        fq, sq = ALL_OUT[cfg]
        sel = [sq[(n, round(X_TOWER_L, 6))] for n in (8, 9)]
        v = 2.0 * np.sqrt(sum(a['s_study'] ** 2 for a in sel))
        vall = 2.0 * np.sqrt(sum(sq[(n, round(X_TOWER_L, 6))]['s_study'] ** 2
                                 for n in range(1, 13)))
        print("   %-16s f0 = %6.3f Hz   stays 8+9 %6.2f %%   all stays %6.2f %%"
              % (cfg, fq, 100 * v, 100 * vall))
        rec("width_variant", cfg, "s_stays_8_9_pct", round(100 * v, 3), "%")
        rec("width_variant", cfg, "s_all_stays_pct", round(100 * vall, 3), "%")

    print()
    print("Bordered pencil with the ACTUAL stay frequencies, so detuning is")
    print("carried exactly.  Twelve stay coordinates with coupling 2 s_i (the")
    print("in-phase combination of the four copies); the remaining 36 copies")
    print("are exactly dark and sit on their own uncoupled frequencies.")
    for tag, key in (("Irvine (taut string)", 'f1_irv'), ("FEM (thesis)", 'f1_fem')):
        fs = [a[key] for a in stays]
        ss = [2.0 * a['s_study'] for a in stays]
        fb, br = bordered_branches(f0_meas, fs, ss)
        print()
        print("  stay frequencies: %s" % tag)
        print("  %-9s %s" % ("branch f", "".join("%8.3f" % v for v in fb)))
        print("  %-9s %s" % ("bright", "".join("%8.3f" % v for v in br)))
        order = np.argsort(-br)[:2]
        bp = sorted(fb[order])
        print("  brightest two branches %.3f and %.3f Hz -> width %.3f Hz "
              "= %.2f %% of %.2f Hz" %
              (bp[0], bp[1], bp[1] - bp[0], 100 * (bp[1] - bp[0]) / f0_meas,
               f0_meas))
        rec("bordered_pencil", tag, "brightest_pair_lo", round(bp[0], 4), "Hz")
        rec("bordered_pencil", tag, "brightest_pair_hi", round(bp[1], 4), "Hz")
        rec("bordered_pencil", tag, "brightest_pair_width_pct",
            round(100 * (bp[1] - bp[0]) / f0_meas, 4), "%")
        for v, b in zip(fb, br):
            rec("bordered_pencil_branches", tag, "f", round(v, 4), "Hz",
                "bright fraction %.4f" % b)
    return stays


def step4_measured(stays, f0_meas=9.02):
    hr("STEP 4.  COMPARISON WITH THE MEASURED BRANCHES")
    tab = MEAS_VERT_ASM_TABLE
    sh = MEAS_VERT_ASM_SHAKER
    print("Measured 1st vertical ASM branches of the ORIGINAL model:")
    print("  shaking table (9): %s" % ", ".join("%.2f" % v for v in tab))
    print("  shaker        (8): %s" % ", ".join("%.2f" % v for v in sh))
    print("  decoupled build (1 peak, no beating): 9.02 Hz")
    print()
    print("WHICH TWO BRANCHES ARE THE BRIGHT PAIR IS A JUDGEMENT.  The thesis")
    print("does not report a structural participation for the measured modes,")
    print("so the pair cannot be read off the data.  Five defensible choices:")
    print()
    choices = []
    lo = [v for v in tab if v < f0_meas]
    hi = [v for v in tab if v > f0_meas]
    choices.append(("nearest straddling pair (table)", lo[-1], hi[0]))
    choices.append(("second straddling pair (table)", lo[-2], hi[1]))
    lo2 = [v for v in sh if v < f0_meas]
    hi2 = [v for v in sh if v > f0_meas]
    choices.append(("nearest straddling pair (shaker)", lo2[-1], hi2[0]))
    choices.append(("full table group extent", tab[0], tab[-1]))
    choices.append(("table group without the 10.9/11.18 pair",
                    tab[0], tab[-3]))
    print("%-40s %8s %8s %10s" % ("choice", "f- (Hz)", "f+ (Hz)", "width %"))
    for name, a, b in choices:
        w = 100 * (b - a) / f0_meas
        print("%-40s %8.2f %8.2f %9.2f%%" % (name, a, b, w))
        rec("measured_width", name, "f_lo", a, "Hz")
        rec("measured_width", name, "f_hi", b, "Hz")
        rec("measured_width", name, "width_pct", round(w, 3), "%")
    print()
    print("Predicted values to compare against (from step 3):")
    sel89 = [a for a in stays if a['no'] in (8, 9)]
    w89 = 200 * np.sqrt(sum(a['s_study'] ** 2 for a in sel89))
    sel8 = [a for a in stays if a['no'] == 8]
    w8 = 200 * np.sqrt(sum(a['s_study'] ** 2 for a in sel8))
    selall = stays
    wall = 200 * np.sqrt(sum(a['s_study'] ** 2 for a in selall))
    print("  one stay (8) alone, four copies, at exact tuning : %.2f %%" % w8)
    print("  stays 8 and 9, four copies each, exact tuning    : %.2f %%" % w89)
    print("  all twelve stays, exact tuning (upper bound)     : %.2f %%" % wall)
    print()
    print("Full branch set: compare the predicted pencil branches with the")
    print("nine measured shaking-table branches, matched in ascending order.")
    for tag, key in (("Irvine", 'f1_irv'), ("FEM", 'f1_fem')):
        fs = [a[key] for a in stays]
        ss = [2.0 * a['s_study'] for a in stays]
        fb, br = bordered_branches(f0_meas, fs, ss)
        inband = fb[(fb > 8.0) & (fb < 12.0)]
        print()
        print("  %s stay frequencies, predicted branches in 8-12 Hz (%d): %s"
              % (tag, len(inband), ", ".join("%.2f" % v for v in inband)))
        n = min(len(inband), len(tab))
        resid = np.array(inband[:n]) - np.array(tab[:n])
        print("  measured (9): %s" % ", ".join("%.2f" % v for v in tab))
        print("  residual over the first %d matched in order: rms %.3f Hz, "
              "max %.3f Hz" % (n, np.sqrt(np.mean(resid ** 2)),
                               np.max(np.abs(resid))))
        rec("branch_match", tag, "n_predicted_in_band", int(len(inband)), "-")
        rec("branch_match", tag, "rms_residual_hz",
            round(float(np.sqrt(np.mean(resid ** 2))), 4), "Hz")
    return choices


def maxent_weights(lam, lam0):
    """Least-committed weights b_k >= 0, sum b = 1, sum b lam = lam0."""
    lam = np.asarray(lam, float)
    lo, hi = -50.0, 50.0
    for _ in range(300):
        beta = 0.5 * (lo + hi)
        w = np.exp(-beta * (lam - lam.mean()))
        w = w / w.sum()
        if float(w @ lam) > lam0:
            lo = beta
        else:
            hi = beta
    beta = 0.5 * (lo + hi)
    w = np.exp(-beta * (lam - lam.mean()))
    return w / w.sum()


def step4b_moments(stays, f0_meas=9.02):
    hr("STEP 4b.  A CHOICE-FREE VERSION OF THE SAME TEST: THE SUM RULE")
    print("The bordered pencil obeys two exact sum rules in lambda = f^2.  If")
    print("b_k is the structural (bright) content of branch k, then")
    print("      sum_k b_k = 1,   sum_k b_k lambda_k = lambda_0,")
    print("      sum_k b_k (lambda_k - lambda_0)^2 = lambda_0^2 sum_i s_i^2.")
    print("So sqrt(sum_i s_i^2) is exactly the bright-weighted standard")
    print("deviation of the branch frequencies in lambda, divided by lambda_0.")
    print("It is a MOMENT of the whole measured group, not a pair of branches,")
    print("and lambda_0 is the MEASURED decoupled reference 9.02 Hz.")
    print()
    print("The thesis does not report b_k for the measured modes.  But b_k >= 0")
    print("and the mean rule pin the achievable spread between two extremes, so")
    print("the 'which pair is bright' judgement becomes a BOUND, not a choice.")
    print()
    lam0 = f0_meas ** 2
    for name, meas in (("shaking table (9 branches)", MEAS_VERT_ASM_TABLE),
                       ("shaker (8 branches)", MEAS_VERT_ASM_SHAKER)):
        lam = np.array(meas) ** 2
        lo = [i for i in range(len(lam)) if lam[i] < lam0]
        hi = [i for i in range(len(lam)) if lam[i] > lam0]
        rows = []
        # minimum spread: the two branches immediately straddling lambda0
        for tag, ia, ib in (("min: nearest straddling pair", lo[-1], hi[0]),
                            ("max: extreme straddling pair", lo[0], hi[-1])):
            da, db = lam0 - lam[ia], lam[ib] - lam0
            ba, bb = db / (da + db), da / (da + db)
            var = ba * da ** 2 + bb * db ** 2
            rows.append((tag, np.sqrt(var) / lam0,
                         "b(%.2f)=%.3f b(%.2f)=%.3f"
                         % (meas[ia], ba, meas[ib], bb)))
        w = maxent_weights(lam, lam0)
        var = float(w @ (lam - lam0) ** 2)
        rows.append(("max-entropy weights", np.sqrt(var) / lam0,
                     "b = " + " ".join("%.3f" % v for v in w)))
        print("-- %s" % name)
        print("   %-30s %12s   %s" % ("weighting", "sqrt(sum s^2)", "weights"))
        for tag, v, w_s in rows:
            print("   %-30s %11.2f%%   %s" % (tag, 100 * v, w_s))
            rec("sum_rule", name, tag, round(100 * v, 3), "%")
        print()

    print("A FOURTH WEIGHTING, AND THIS ONE IS MEASURED.  A hybrid branch")
    print("carries the structure's damping in proportion to its bright content")
    print("and the wire's damping in proportion to the rest, so to first order")
    print("     zeta_k = b_k zeta_struct + (1 - b_k) zeta_cable.")
    print("zeta_struct is MEASURED on the decoupled build: 0.46-0.49 % for the")
    print("1st vertical ASM (Table 7.19).  zeta_cable is taken as the smallest")
    print("damping in the group, which is the most nearly pure cable branch.")
    print("Inverting gives b_k from the thesis's own damping column.")
    print()
    zst = 0.475
    for name, rows, pick in (("shaking table, mid zeta", T717_VERT_ASM, 'mid'),
                             ("shaking table, low zeta", T717_VERT_ASM, 'lo'),
                             ("shaking table, high zeta", T717_VERT_ASM, 'hi')):
        if pick == 'mid':
            got = [(r[1][0], 0.5 * (r[1][1] + r[1][2])) for r in rows
                   if r[1][0] is not None]
        elif pick == 'lo':
            got = [(r[1][0], r[1][1]) for r in rows if r[1][0] is not None]
        else:
            got = [(r[1][0], r[1][2]) for r in rows if r[1][0] is not None]
        f = np.array([v for v, _ in got])
        z = np.array([v for _, v in got])
        zc = z.min()
        b = np.clip((z - zc) / (zst - zc), 0.0, None)
        b = b / b.sum()
        lam = f ** 2
        mean = float(b @ lam)
        var = float(b @ (lam - mean) ** 2)
        var0 = float(b @ (lam - lam0) ** 2)
        print("-- %s, damping-inferred weights" % name)
        if pick == 'mid':
            print("   %8s %9s %9s" % ("f (Hz)", "zeta %", "b_k"))
            for v, zz, bb in zip(f, z, b):
                print("   %8.2f %9.3f %9.3f" % (v, zz, bb))
                rec("damping_weights", "%.2f Hz" % v, "b_k",
                    round(float(bb), 4), "-", "zeta_mid %.3f pct" % zz)
        print("   weighted mean frequency %.3f Hz against the measured" % np.sqrt(mean))
        print("   decoupled reference %.2f Hz (the sum rule says they should"
              % f0_meas)
        print("   agree; the gap is a direct check on the branch list being")
        print("   complete).  Difference %+.2f %%."
              % (100 * (np.sqrt(mean) - f0_meas) / f0_meas))
        print("   sqrt(sum s^2) about the weighted mean : %.2f %%"
              % (100 * np.sqrt(var) / mean))
        print("   sqrt(sum s^2) about lambda_0 = 9.02^2 : %.2f %%"
              % (100 * np.sqrt(var0) / lam0))
        rec("sum_rule", name, "damping_weighted_about_mean_pct",
            round(100 * float(np.sqrt(var) / mean), 3), "%")
        rec("sum_rule", name, "damping_weighted_about_lam0_pct",
            round(100 * float(np.sqrt(var0) / lam0), 3), "%")
        rec("sum_rule", name, "damping_weighted_mean_freq",
            round(float(np.sqrt(mean)), 4), "Hz")
        # and without the 10.90 / 11.18 pair, which belongs to backstay tuning
        keep = f < 10.0
        b2 = b[keep] / b[keep].sum()
        lam2 = lam[keep]
        mean2 = float(b2 @ lam2)
        var2 = float(b2 @ (lam2 - mean2) ** 2)
        print("   same, excluding 10.90 and 11.18 Hz (cable 1 is tuned to")
        print("   11.5 Hz and drives that pair, not the 9 Hz group): %.2f %%"
              % (100 * np.sqrt(var2) / mean2))
        rec("sum_rule", name, "damping_weighted_below_10Hz_pct",
            round(100 * float(np.sqrt(var2) / mean2), 3), "%")
    print()

    print("PREDICTED sqrt(sum_i s_i^2), four copies of every stay:")
    for nm, ss in (("stays 8 and 9 (the near-tuned ones)", (8, 9)),
                   ("stays 7-11", (7, 8, 9, 10, 11)),
                   ("all twelve stays", tuple(range(1, 13)))):
        sel = [a for a in stays if a['no'] in ss]
        v = 2.0 * np.sqrt(sum(a['s_study'] ** 2 for a in sel))
        vg = 2.0 * np.sqrt(sum(a['s_gen'] ** 2 for a in sel))
        v2 = 2.0 * np.sqrt(sum(a['s_two'] ** 2 for a in sel))
        vg2 = 2.0 * np.sqrt(sum(a['s_gen_two'] ** 2 for a in sel))
        print("   %-36s %7.2f %%   (generalised drive %.2f %%;"
              " two-route %.2f %%, with tower %.2f %%)"
              % (nm, 100 * v, 100 * vg, 100 * v2, 100 * vg2))
        rec("sum_rule_prediction", nm, "sqrt_sum_s2_pct", round(100 * v, 3), "%")
        rec("sum_rule_prediction", nm, "sqrt_sum_s2_two_route_pct", round(100 * v2, 3), "%")
        rec("sum_rule_prediction", nm, "sqrt_sum_s2_two_route_tower_pct", round(100 * vg2, 3), "%")
    print()
    print("The sum rule counts EVERY stay, however detuned, so the honest")
    print("comparison for the measured group is the all-stay figure, reduced by")
    print("whatever the truncation of the measured branch list to 8.6-11.2 Hz")
    print("removes.  Stays 2-5 sit at 15-19 Hz and their branches are outside")
    print("that window, so the measured spread must fall short of the all-stay")
    print("prediction by construction.")


def step5b_cross(rows, name):
    """Was a mode seen by BOTH techniques, and does s/zeta predict it?"""
    print()
    print("-- %s, cross-technique test" % name)
    print("Physical mode list is the thesis's own row pairing.  A row's")
    print("frequency is the shaking-table value where it exists, else the")
    print("shaker value; its damping is the widest range across both columns.")
    seq = []
    for (sh, tb) in rows:
        f = tb[0] if tb[0] is not None else sh[0]
        zs = [v for v in (sh[1], sh[2], tb[1], tb[2]) if v is not None]
        both = (sh[0] is not None) and (tb[0] is not None)
        seq.append((f, min(zs), max(zs), both))
    seq.sort()
    print("%8s %8s %9s %9s %8s %8s %8s" %
          ("f (Hz)", "s (%)", "zeta_lo%", "zeta_hi%", "dip?", "3dB?", "both?"))
    a = b = c = d = 0
    for k, (f, zl, zh, both) in enumerate(seq):
        nbs = [seq[j][0] for j in (k - 1, k + 1) if 0 <= j < len(seq) and j != k]
        nb = min(nbs, key=lambda v: abs(v - f))
        s = abs(nb - f) / (0.5 * (nb + f))
        dip = s > K_DIP * zh / 100.0
        db3 = s > K_3DB * zh / 100.0
        print("%8.2f %8.3f %9.3f %9.3f %8s %8s %8s" %
              (f, 100 * s, zl, zh, "yes" if dip else "NO",
               "yes" if db3 else "no", "yes" if both else "NO"))
        rec("resolvability_cross", "%s %.2f Hz" % (name, f), "s_pct",
            round(100 * s, 4), "%", "seen by both: %d" % int(both))
        rec("resolvability_cross", "%s %.2f Hz" % (name, f), "three_db",
            int(db3), "-")
        if db3 and both:
            a += 1
        elif db3 and not both:
            b += 1
        elif both:
            c += 1
        else:
            d += 1
    n = len(seq)
    print("  3dB & both %d | 3dB & one %d | no3dB & both %d | no3dB & one %d"
          % (a, b, c, d))
    print("  agreement of the 3 dB criterion with 'seen by both': %d/%d = %.0f %%"
          % (a + d, n, 100.0 * (a + d) / n))
    rec("resolvability_cross_summary", name, "agreement_pct",
        round(100.0 * (a + d) / n, 1), "%",
        "3dB&both=%d 3dB&one=%d no3dB&both=%d no3dB&one=%d" % (a, b, c, d))


def step5_resolvability():
    hr("STEP 5.  RESOLVABILITY  s > 0.9717 zeta  AND  s > 2.280 zeta")
    print("The thresholds are applied to the MEASURED normalised spacing")
    print("between adjacent identified branches and the MEASURED damping of")
    print("those branches.  No model enters this test.")
    print()
    print("Caetano, p. 7.71: 'In certain cases it was not possible to identify")
    print("accurately some of the multiple modes simultaneously with both")
    print("excitation techniques, due to high modal interference AND")
    print("insufficient amplitude and/or frequency resolution of the FRF")
    print("estimates.'  The second clause is a confound and is carried below.")
    print()
    print("A SECOND AND LARGER CONFOUND.  The study's condition is about a DIP")
    print("between two peaks in a spectrum, i.e. about peak picking.  Caetano")
    print("identified modes with a rational-fraction-polynomial curve fit over")
    print("many FRFs, which separates modes well below the dip threshold.  The")
    print("Jindo data can therefore FALSIFY the condition in the direction")
    print("'predicted unresolvable but identified anyway'; it cannot confirm")
    print("it.  Read the tables below with that asymmetry in mind.")
    print()

    def run(name, rows, col):
        print()
        print("-- %s, %s column" % (name, col))
        idx = 0 if col == 'shaker' else 1
        seq = [(r[idx][0], r[idx][1], r[idx][2], r[1 - idx][0])
               for r in rows]
        got = [(f, zl, zh, other) for (f, zl, zh, other) in seq if f is not None]
        print("%8s %8s %9s %9s %9s %9s %9s %8s" %
              ("f (Hz)", "s (%)", "zeta_lo%", "zeta_hi%", "s/z_hi",
               "dip?", "3dB?", "other?"))
        for k, (f, zl, zh, other) in enumerate(got):
            if k == 0:
                nb = got[1][0]
            elif k == len(got) - 1:
                nb = got[-2][0]
            else:
                nb = min(got[k - 1][0], got[k + 1][0],
                         key=lambda v: abs(v - f))
            s = abs(nb - f) / (0.5 * (nb + f))
            zhi = zh / 100.0
            zlo = zl / 100.0
            dip = s > K_DIP * zhi
            db3 = s > K_3DB * zhi
            print("%8.2f %8.3f %9.3f %9.3f %9.2f %9s %9s %8s" %
                  (f, 100 * s, zl, zh, s / zhi if zhi > 0 else np.inf,
                   "yes" if dip else "NO", "yes" if db3 else "no",
                   "yes" if other is not None else "NO"))
            rec("resolvability", "%s %s %.2f Hz" % (name, col, f),
                "s_pct", round(100 * s, 4), "%")
            rec("resolvability", "%s %s %.2f Hz" % (name, col, f),
                "s_over_zeta_hi", round(s / zhi, 3) if zhi > 0 else None, "-")
            rec("resolvability", "%s %s %.2f Hz" % (name, col, f),
                "dip_predicted", int(dip), "-",
                "identified by the other technique: %d"
                % (1 if other is not None else 0))
            rec("resolvability", "%s %s %.2f Hz" % (name, col, f),
                "three_db_predicted", int(db3), "-")
        # contingency
        a = b = c = d = 0
        for k, (f, zl, zh, other) in enumerate(got):
            if k == 0:
                nb = got[1][0]
            elif k == len(got) - 1:
                nb = got[-2][0]
            else:
                nb = min(got[k - 1][0], got[k + 1][0],
                         key=lambda v: abs(v - f))
            s = abs(nb - f) / (0.5 * (nb + f))
            db3 = s > K_3DB * (zh / 100.0)
            both = other is not None
            if db3 and both:
                a += 1
            elif db3 and not both:
                b += 1
            elif (not db3) and both:
                c += 1
            else:
                d += 1
        print("  contingency on the 3 dB criterion vs 'seen by both techniques'")
        print("    3dB predicted & seen by both     : %d" % a)
        print("    3dB predicted & seen by one only : %d" % b)
        print("    no 3dB       & seen by both      : %d" % c)
        print("    no 3dB       & seen by one only  : %d" % d)
        rec("resolvability_contingency", "%s %s" % (name, col),
            "3db_and_both", a, "modes")
        rec("resolvability_contingency", "%s %s" % (name, col),
            "3db_and_one", b, "modes")
        rec("resolvability_contingency", "%s %s" % (name, col),
            "no3db_and_both", c, "modes")
        rec("resolvability_contingency", "%s %s" % (name, col),
            "no3db_and_one", d, "modes")

    print("THE GROUP THE BRIEF SINGLES OUT: 1st transversal ASM on the shaking")
    print("table, where s/zeta straddles both thresholds.")
    for (fa, fb) in ((10.35, 10.54), (10.54, 10.73)):
        za = dict((r[1][0], (r[1][1], r[1][2])) for r in T717_TRANSV_ASM
                  if r[1][0] is not None)
        s = (fb - fa) / (0.5 * (fa + fb))
        zh = max(za[fa][1], za[fb][1]) / 100.0
        zl = min(za[fa][0], za[fb][0]) / 100.0
        print("  pair %.2f / %.2f Hz : s = %.3f %%, zeta %.2f-%.2f %%, "
              "s/zeta = %.2f to %.2f" %
              (fa, fb, 100 * s, 100 * zl, 100 * zh, s / zh, s / zl))
        print("      dip (s > %.4f zeta): %s at the high zeta, %s at the low"
              % (K_DIP, "yes" if s > K_DIP * zh else "NO",
                 "yes" if s > K_DIP * zl else "NO"))
        print("      3 dB (s > %.3f zeta): %s at the high zeta, %s at the low"
              % (K_3DB, "yes" if s > K_3DB * zh else "no",
                 "yes" if s > K_3DB * zl else "no"))
        rec("resolvability_key_pair", "%.2f/%.2f" % (fa, fb), "s_pct",
            round(100 * s, 4), "%")
        rec("resolvability_key_pair", "%.2f/%.2f" % (fa, fb),
            "s_over_zeta_hi", round(s / zh, 3), "-")
        rec("resolvability_key_pair", "%.2f/%.2f" % (fa, fb),
            "s_over_zeta_lo", round(s / zl, 3), "-")

    run("1st transversal ASM", T717_TRANSV_ASM, 'shaking table')
    run("1st transversal ASM", T717_TRANSV_ASM, 'shaker')
    run("1st vertical ASM", T717_VERT_ASM, 'shaking table')
    run("1st vertical SYM", T717_VERT_SYM, 'shaker')


def step6_stay12(res_by_cfg):
    hr("STEP 6.  THE STAY-12 DISCRIMINATOR: mu_eff AGAINST PLAIN MASS RATIO")
    res = res_by_cfg['decoupled']
    m_asm, f_asm = find_mode(res, '1st vert. ASM')
    m_sym, f_sym = find_mode(res, '1st vert. SYM')
    ad_asm = anchor_data(res, m_asm, None)
    ad_sym = anchor_data(res, m_sym, None)

    def key(ad, no):
        return [a for a in ad
                if a['no'] == no and abs(a['xt'] - X_TOWER_L) < 1e-9][0]

    tot = sum(res['fm'].lumped.values()) + sum(
        fr['m'] * res['fm']._dir(fr['i'], fr['j'])[0]
        for fr in res['fm'].frames)
    a12s, a12a = key(ad_sym, 12), key(ad_asm, 12)
    a8s, a8a = key(ad_sym, 8), key(ad_asm, 8)

    print("Stay 12 anchors 71 mm from midspan: a NODE of the anti-symmetric")
    print("mode, an ANTINODE of the symmetric one.  It is the heaviest")
    print("forestay, M_s = %.5f kg, twice stay 8.  Its plain mass ratio"
          % a12a['Ms'])
    print("M_s / M_structure = %.5f is IDENTICAL for the two families, so a"
          % (a12a['Ms'] / tot))
    print("plain mass ratio predicts IDENTICAL coupling to both.")
    print()
    print("%-6s %-16s %10s %10s %12s %10s %12s" %
          ("stay", "family", "f0 FE Hz", "phi_a", "mu_eff", "s (%)",
           "M_s/M_tot"))
    for no in (8, 12):
        for tag, ad, fq in (("1st vert. SYM", ad_sym, f_sym),
                            ("1st vert. ASM", ad_asm, f_asm)):
            a = key(ad, no)
            print("%-6d %-16s %10.3f %10.5f %12.3e %9.3f%% %12.5f" %
                  (no, tag, fq, a['phi_a'], a['mu_eff'],
                   100 * a['s_study'], a['Ms'] / tot))
            rec("stay12_discriminator", "stay %d %s" % (no, tag), "phi_a",
                round(a['phi_a'], 6), "kg^-1/2")
            rec("stay12_discriminator", "stay %d %s" % (no, tag), "mu_eff",
                float("%.6g" % a['mu_eff']), "-")
            rec("stay12_discriminator", "stay %d %s" % (no, tag), "s_pct",
                round(100 * a['s_study'], 4), "%")
            rec("stay12_discriminator", "stay %d %s" % (no, tag),
                "plain_mass_ratio", round(a['Ms'] / tot, 6), "-")

    r12 = a12s['mu_eff'] / a12a['mu_eff']
    r8 = a8s['mu_eff'] / a8a['mu_eff']
    print()
    print("PREDICTION.  For stay 12 the mu_eff ratio SYM/ASM is %.1f, so the"
          % r12)
    print("width it drives is sqrt of that, %.1f times larger against the"
          % np.sqrt(r12))
    print("symmetric family than against the anti-symmetric one, at the same")
    print("M_s and the same plain mass ratio.  For stay 8 the SAME ratio is")
    print("%.2f, i.e. the OPPOSITE way round.  So the discriminator is a" % r8)
    print("property of WHERE a stay sits, not of the two families in general,")
    print("and no single plain mass ratio can produce both signs.")
    rec("stay12_discriminator", "stay 12", "mu_eff_ratio_sym_over_asm",
        round(r12, 3), "-")
    rec("stay12_discriminator", "stay 8", "mu_eff_ratio_sym_over_asm",
        round(r8, 3), "-")
    print()
    print("Stay 12's own first frequency is %.2f Hz (Irvine) / %.2f Hz (FEM),"
          % (a12a['f1_irv'], a12a['f1_fem']))
    print("which sits INSIDE the measured 1st vertical SYM group")
    print("6.25 / 6.69 / 7.12 Hz and %.0f %% below the 1st vertical ASM at"
          % (100 * (9.02 - a12a['f1_irv']) / 9.02))
    print("9.02 Hz.  Stay 12 is therefore both near-tuned to the symmetric")
    print("family and placed at its antinode, and both detuned from and")
    print("nodal to the anti-symmetric one.  The two effects point the same")
    print("way, so this is a compound test, not a clean single-variable one.")
    print()
    print("TEST AGAINST MEASUREMENT.")
    print("  (a) Table 7.16, 3-D MECS, mode 27 at 9.04 Hz, the 1st vertical")
    print("      ASM mode with the most deck motion (cable/beam Z-ratio 1.6,")
    print("      the smallest in Table 7.10): stays 8U/8D carry the DOUBLE")
    print("      underline, highest amplitude, while 12U/12D carry NONE, the")
    print("      lowest band.  Stay 12 barely moves in the anti-symmetric")
    print("      mode despite being the heaviest forestay.")
    print("  (b) Table 7.14, MEASURED cable movement under tuned sinusoidal")
    print("      excitation at 9.20 Hz (1st vertical ASM): stays 8 and 9 are")
    print("      listed at BOTH towers with double underlines; stay 12 does")
    print("      not appear at all.  It first appears at 9.50 Hz and then at")
    print("      one tower only.")
    print("  (c) The measured symmetric group 6.25 / 6.69 / 7.12 Hz spans")
    print("      %.1f %% about 6.69 Hz from three modes."
          % (100 * (7.12 - 6.25) / 6.69))
    w12 = 2.0 * a12s['s_study']
    w_sym_all = 2.0 * np.sqrt(sum(key(ad_sym, n)['s_study'] ** 2
                                  for n in range(1, 13)))
    print()
    print("  Predicted symmetric-family width, four copies of stay 12 alone,")
    print("  at exact tuning: %.2f %%.  Over all twelve stays: %.2f %%."
          % (100 * w12, 100 * w_sym_all))
    print("  Measured 6.25-7.12 Hz = %.2f %%."
          % (100 * (7.12 - 6.25) / 6.69))
    rec("stay12_discriminator", "stay 12 SYM", "width_4copies_pct",
        round(100 * w12, 4), "%")
    rec("stay12_discriminator", "all stays SYM", "width_pct",
        round(100 * w_sym_all, 4), "%")
    rec("stay12_discriminator", "measured SYM group", "width_pct",
        round(100 * (7.12 - 6.25) / 6.69, 3), "%")
    print()
    print("  Stay 1 is the same test in its strongest form.  It is by far the")
    print("  heaviest stay (M_s = %.4f kg, 6.7 times stay 8) but it anchors AT"
          % key(ad_asm, 1)['Ms'])
    print("  the abutment support, where the deck cannot move.  Computed:")
    print("  phi_a = %.2e, mu_eff = %.2e, s = %.4f %%.  A plain mass ratio"
          % (key(ad_asm, 1)['phi_a'], key(ad_asm, 1)['mu_eff'],
             100 * key(ad_asm, 1)['s_study']))
    print("  would make stay 1 the most strongly coupled stay in the bridge.")
    print("  The refinement that includes tower-top motion gives it a non-zero")
    print("  drive, s_gen = %.3f %%, because the backstay top rides on the"
          % (100 * key(ad_asm, 1)['s_gen']))
    print("  tower; that is the leading correction to the rigid-pylon law.")
    rec("stay1_discriminator", "stay 1 ASM", "phi_a",
        float("%.4g" % key(ad_asm, 1)['phi_a']), "kg^-1/2")
    rec("stay1_discriminator", "stay 1 ASM", "mu_eff",
        float("%.4g" % key(ad_asm, 1)['mu_eff']), "-")
    rec("stay1_discriminator", "stay 1 ASM", "s_pct",
        float("%.4g" % (100 * key(ad_asm, 1)['s_study'])), "%")
    rec("stay1_discriminator", "stay 1 ASM", "s_generalised_pct",
        float("%.4g" % (100 * key(ad_asm, 1)['s_gen'])), "%")


# =========================================================================
# 9.  MAIN
# =========================================================================

def main():
    hr("JINDO 1:150 PHYSICAL MODEL - MEASURED TEST OF THE AMPLITUDE LAW")
    print("Source: Caetano (2001), FEUP doctoral thesis, Chapter 7.")
    print("Tested here: the amplitude law and the resolvability condition.")
    print("NOT tested here: the tension-error law.  The stay tensions were set")
    print("by tuning each wire to its taut-string design frequency with a")
    print("magnetic pickup, and the thesis's '(Irvine)' column is reproduced")
    print("exactly by f = (1/2L) sqrt(T/m) from the tabulated (L, m, T).  The")
    print("tabulated tensions are the tuning targets, not measurements, so")
    print("there is no independent tension to test an inversion against.")

    g = stay_geometry()
    print()
    print("Arithmetic self-check on the transcription: the taut-string formula")
    print("from the tabulated (L, m, T) reproduces the thesis's Irvine column")
    print("to %.4f %% maximum." %
          max(100 * abs(s['f1_check'] - s['f1_irv']) / s['f1_irv'] for s in g))
    for s in g:
        rec("stay_geometry", "stay %d" % s['no'], "L", round(s['L'], 6), "m")
        rec("stay_geometry", "stay %d" % s['no'], "x_anchor",
            round(s['x'], 6), "m")
        rec("stay_geometry", "stay %d" % s['no'], "tower_anchor_height",
            round(s['h'], 6), "m")
        rec("stay_geometry", "stay %d" % s['no'], "theta_deg",
            round(float(np.degrees(s['theta'])), 4), "deg")
        rec("stay_geometry", "stay %d" % s['no'], "Ms", round(s['Ms'], 6), "kg")
        rec("stay_geometry", "stay %d" % s['no'], "T", s['T'], "N")
        rec("stay_geometry", "stay %d" % s['no'], "EA",
            round(s['EA'], 2), "N")
        rec("stay_geometry", "stay %d" % s['no'], "f1_irvine", s['f1_irv'], "Hz")
        rec("stay_geometry", "stay %d" % s['no'], "f1_fem", s['f1_fem'], "Hz")

    print()
    print("Independent check of the wire diameters read off Figure 7.2: the")
    print("thesis says the stay AXIAL stiffness was correctly scaled, so")
    print("EA_model should equal EA_prototype / (sE sL^2) = EA_p / 69075.")
    print("%5s %9s %12s %12s %9s" % ("stay", "dia (mm)", "EA model (N)",
                                     "EA scaled (N)", "diff %"))
    for (no, dia, EAm, EAp, err) in axial_scale_check():
        print("%5d %9.2f %12.0f %12.0f %+9.1f" % (no, dia, EAm, EAp, err))
        rec("axial_scale_check", "stay %d" % no, "EA_model", round(EAm, 1), "N")
        rec("axial_scale_check", "stay %d" % no, "EA_scaled_from_prototype",
            round(EAp, 1), "N")
        rec("axial_scale_check", "stay %d" % no, "diff_pct", round(err, 3), "%")

    res = {}
    for cfg in ('decoupled', 'bare', 'quasistatic'):
        res[cfg] = solve_model(build(cable_mass=cfg), 24)
    res_nokg = solve_model(build(cable_mass='decoupled', geom_stiff=False), 24)
    res_rigid = solve_model(build(cable_mass='decoupled', rigid_pier=True), 24)
    res_coarse = solve_model(build(cable_mass='decoupled', max_seg=0.080), 24)
    res_fine = solve_model(build(cable_mass='decoupled', max_seg=0.025), 24)

    variants = [("built (decoupled)", res['decoupled']),
                ("no geometric stiff.", res_nokg),
                ("rigid pier", res_rigid),
                ("mesh 80 mm", res_coarse),
                ("mesh 25 mm", res_fine),
                ("bare (no cable mass)", res['bare']),
                ("quasistatic cable mass", res['quasistatic'])]
    step1_verify(variants)
    res['rigid_pier'] = res_rigid
    res['no_Kg'] = res_nokg
    out = step2_ordinates(res)
    stays = step3_width(out)
    step4_measured(stays)
    step4b_moments(stays)
    step5_resolvability()
    step5b_cross(T717_TRANSV_ASM, "1st transversal ASM")
    step5b_cross(T717_VERT_ASM, "1st vertical ASM")
    step5b_cross(T717_VERT_SYM, "1st vertical SYM")
    step6_stay12(res)

    hr("WRITING data/jindo.csv")
    path = os.path.join(ROOT, "data", "jindo.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["block", "item", "quantity",
                                           "value", "unit", "note"])
        w.writeheader()
        for r in ROWS:
            w.writerow(r)
    print("%d rows -> %s" % (len(ROWS), path))


if __name__ == "__main__":
    main()
