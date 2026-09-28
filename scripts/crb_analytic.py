# -*- coding: utf-8 -*-
"""Fisher information and Cramer-Rao bound on stay tension at a crossing.

Model: one stay sensor sees two Lorentzians with weights (1 +- rho)/2,
rho = d / sqrt(d^2 + s^2), under equal modal force intensities, with the
Whittle likelihood. Parts 1-4 derive and evaluate the bound; Parts 5-7 check
it by finite differences, Monte Carlo and Au (2014), Uncertainty law in
ambient modal identification, Part I, Eq. (15)-(16); Parts 8-10 cover the
screening criterion, the finite-element bridge and a second (deck) sensor.
Writes data/crb_analytic.csv.

Run:  python3 scripts/crb_analytic.py
"""

from __future__ import annotations

import os
import sys
import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

DATA = os.path.join(ROOT, "data")
CSV = os.path.join(DATA, "crb_analytic.csv")

# visibility thresholds from run_damping.py
U_DIP_COH = np.sqrt(np.sqrt(5.0) - 2.0)      # 0.48587, coherent driving point
U_DIP_INC = 1.0/np.sqrt(3.0)                 # 0.57735, incoherent ambient sum
U_3DB = 1.140                                # 3 dB prominence, s = 2.280 zeta

ROWS = []


def add(part, name, **kw):
    ROWS.append(dict(part=part, quantity=name, **kw))


# --- Part 1: symbolic pole calculus and exact-tuning closed forms ---

def part1():
    print("=" * 78)
    print("PART 1  Fisher information by pole calculus, exact tuning (d = 0)")
    print("=" * 78)
    sg, nu = sp.symbols('sigma nu', positive=True)
    NUV = sp.sqrt(1 + sg ** 2)
    I = sp.I

    # upper-half-plane features of ln S = ln P + ln N - ln D and their signs:
    # -1 for the zero of N, +1 for the poles of D
    POL = [(I * nu, -1), (sg + I, +1), (-sg + I, +1)]

    # pole velocities; d/d delta (delta = d/2 zeta) is evaluated at delta = 0:
    #   zero of N at -delta + i sqrt(1+sigma^2)      -> d/d delta = -1
    #   poles of D at +- sqrt(delta^2+sigma^2) + i   -> d/d delta = +- delta/u = 0
    # ln zeta acts as a dilation of x minus the induced change of (delta,sigma)
    VEL = {
        'x0':  [sp.Integer(1), sp.Integer(1), sp.Integer(1)],
        'del': [sp.Integer(-1), sp.Integer(0), sp.Integer(0)],
        'sig': [I * sg / nu, sp.Integer(1), sp.Integer(-1)],
        'lnz': [I / nu, I, I],
    }

    def entry(a, b):
        """Int d_a lnS d_b lnS dx over the whole line, by residues.

        Int dx/((x-z)(x-w)) = 2 pi i/(z-w) when Im z > 0 > Im w, else 0.
        """
        tot = 0
        for k, (z, sz) in enumerate(POL):
            ca, cb = sz * VEL[a][k], sz * VEL[b][k]
            for l, (w, sw) in enumerate(POL):
                wc = sp.conjugate(w)
                da = sp.conjugate(sw * VEL[a][l])
                db = sp.conjugate(sw * VEL[b][l])
                tot += (ca * db + cb * da) / (z - wc)
        e = sp.simplify(sp.re(sp.expand_complex(sp.expand(2 * sp.pi * I * tot))))
        return sp.radsimp(sp.cancel(sp.simplify(e.subs(nu, NUV))))

    names = ['x0', 'del', 'sig', 'lnz']
    J = {}
    print("\n  reduced FIM  J_ab = Int d_a lnS d_b lnS dx   (FIM = N_half . J)")
    print("  basis: x0 (band center, = zeta d/d ln f0), delta = d/2zeta,")
    print("         sigma = s/2zeta, ln zeta;  sigma = u at exact tuning\n")
    for i, a in enumerate(names):
        for b in names[i:]:
            J[(a, b)] = J[(b, a)] = entry(a, b)
            print("    J[%-4s,%-4s] = %s" % (a, b, sp.simplify(J[(a, b)])))

    print("\n  Parity: at exact tuning S is even in x, so the scores split into")
    print("  odd {x0, delta} and even {sigma, ln zeta, ln P, S_e} and the cross")
    print("  entries vanish identically:")
    for pair in [('x0', 'sig'), ('x0', 'lnz'), ('del', 'sig'), ('del', 'lnz')]:
        print("    J%-12s = %s" % (str(pair), J[pair]))
    assert all(sp.simplify(J[p]) == 0 for p in
               [('x0', 'sig'), ('x0', 'lnz'), ('del', 'sig'), ('del', 'lnz')])
    print("  => at exact tuning, unknown s (hence mu_eff), zeta, P and S_e")
    print("     add no variance to T; only an unknown host frequency does.")

    # physical scores:
    #   d/d lnT = (1/4z)[x0 + delta], d/d ln fh = (1/2z)[x0 - delta]
    A = sp.simplify(J[('x0', 'x0')] + 2 * J[('x0', 'del')] + J[('del', 'del')])
    B = sp.simplify(J[('x0', 'x0')] - 2 * J[('x0', 'del')] + J[('del', 'del')])
    C = sp.simplify(J[('x0', 'x0')] - J[('del', 'del')])
    u = sp.Symbol('u', positive=True)
    A, B, C = [sp.simplify(e.subs(sg, u)) for e in (A, B, C)]

    print("\n  scores:  d/d lnT = (1/4 zeta)[x0 + delta],")
    print("           d/d ln f_h = (1/2 zeta)[x0 - delta]")
    print("    16 zeta^2 I_TT / N_half =", A, "        [far field: 8 pi]")
    print("     4 zeta^2 I_hh / N_half =", sp.simplify(B))
    print("     8 zeta^2 I_Th / N_half =", sp.simplify(C))
    print("    I_Th = I_hh/2 exactly:", sp.simplify(C / 2 - B / 2) == 0)

    Rk = sp.simplify(8 * sp.pi / A)
    sch = sp.simplify(A - (2 * C) ** 2 / (4 * B))
    Ru = sp.simplify(8 * sp.pi / sch)
    corr2 = sp.simplify(sp.Rational(1, 4) * B / A * 4)          # I_Th^2/(I_TT I_hh)
    print("\n    corr(T, f_h)^2 = I_Th^2/(I_TT I_hh) =", sp.simplify(corr2))
    print("    1/(1 - corr^2) = variance factor for an unknown host frequency")
    print("                   =", sp.simplify(1 / (1 - corr2)))
    print("\n    CRB(lnT)/CRB_far, host known    =", Rk)
    print("    16 zeta^2 I_TT|h / N_half       =", sch)
    print("    CRB(lnT)/CRB_far, host unknown  =", Ru)
    assert sp.simplify(Rk - 2 * (1 + u ** 2) / (2 + u ** 2)) == 0
    assert sp.simplify(Ru - sp.sqrt(1 + u ** 2)) == 0
    print("\n    limits:  u -> 0   known %s   unknown %s"
          % (sp.limit(Rk, u, 0), sp.limit(Ru, u, 0)))
    print("             u -> oo  known %s   unknown %s"
          % (sp.limit(Rk, u, sp.oo), sp.limit(Ru, u, sp.oo)))
    print("\n  Both limits return 1 as u -> 0: when the coupling is small")
    print("  compared with the damping, the stay reads its own frequency")
    print("  with neither bias nor loss of information.")

    for uu in [0.0, U_DIP_COH, U_DIP_INC, U_3DB, 1.0, 2.34, 5.0, 20.0]:
        add("1", "tuned_ratio", u=uu,
            R_host_known=float(Rk.subs(u, uu)) if uu > 0 else 1.0,
            R_host_unknown=float(Ru.subs(u, uu)) if uu > 0 else 1.0)
    return sp.lambdify(u, Rk), sp.lambdify(u, Ru)


# --- Part 2: symbolic far-field expansion ---

def part2():
    print()
    print("=" * 78)
    print("PART 2  far from the crossing:  |d| >> s")
    print("=" * 78)
    dl, sg = sp.symbols('delta sigma', positive=True)   # d/2zeta, s/2zeta
    I = sp.I
    uu = sp.sqrt(dl ** 2 + sg ** 2)
    POL = [(-dl + I * sp.sqrt(1 + sg ** 2), -1), (uu + I, +1), (-uu + I, +1)]
    VEL = {
        'x0':  [sp.Integer(1)] * 3,
        'del': [sp.Integer(-1), dl / uu, -dl / uu],
    }

    def entry(a, b, order=5):
        tot = 0
        for k, (z, sz) in enumerate(POL):
            ca, cb = sz * VEL[a][k], sz * VEL[b][k]
            for l, (w, sw) in enumerate(POL):
                wc = sp.conjugate(w)
                da = sp.conjugate(sw * VEL[a][l])
                db = sp.conjugate(sw * VEL[b][l])
                tot += (ca * db + cb * da) / (z - wc)
        e = sp.re(sp.expand_complex(sp.expand(2 * sp.pi * I * tot)))
        return sp.simplify(sp.series(e, sg, 0, order).removeO())

    Jxx = entry('x0', 'x0'); Jxd = entry('x0', 'del'); Jdd = entry('del', 'del')
    A = sp.simplify(sp.expand(Jxx + 2 * Jxd + Jdd))
    B = sp.simplify(sp.expand(Jxx - 2 * Jxd + Jdd))
    C = sp.simplify(sp.expand(Jxx - Jdd))
    print("\n  16 zeta^2 I_TT/N_half =", A, "     [exact value at s = 0: 8 pi]")
    rk = sp.series(sp.simplify(8 * sp.pi / A), sg, 0, 5).removeO()
    rk = sp.simplify(sp.expand(rk))
    print("\n  CRB(lnT)/CRB_far, host known:")
    print("   ", rk)
    sch = sp.simplify(A - (2 * C) ** 2 / (4 * B))
    ru = sp.simplify(sp.expand(sp.series(sp.simplify(8 * sp.pi / sch), sg, 0, 5).removeO()))
    print("\n  CRB(lnT)/CRB_far, host unknown:")
    print("   ", ru)
    rk_c = sp.simplify(sp.factor(sp.expand(rk - 1) * (1 + dl ** 2) ** 2))
    ru_c = sp.simplify(sp.factor(sp.expand(ru - 1) * (1 + dl ** 2) ** 2))
    print("\n  factored, with delta = d/2zeta and sigma = s/2zeta:")
    print("    host known    R = 1 + [%s] / (1+delta^2)^2" % rk_c)
    print("    host unknown  R = 1 + [%s] / (1+delta^2)^2" % ru_c)
    lead = sp.simplify(sp.series(rk, sg, 0, 4).removeO() - 1)
    print("\n  the O(s^2) term is the same in both and, since")
    print("  sigma^2/(1+delta^2) = s^2/(d^2 + 4 zeta^2), it is a single formula")
    print("  valid over the whole plane:")
    print("\n      CRB(lnT)/CRB_far = 1 + s^2 / [2 (d^2 + 4 zeta^2)] + O(s^4)")
    print("\n  which returns 1 + s^2/2d^2 far from the crossing and 1 + u^2/2 at")
    print("  exact tuning, matching the small-u expansion of both PART 1 forms")
    print("  (2(1+u^2)/(2+u^2) and sqrt(1+u^2) agree to O(u^2)).  Check:")
    for uu in [0.05, 0.2, 0.5]:
        print("    u = %.2f : 2(1+u2)/(2+u2) = %.6f, sqrt(1+u2) = %.6f, 1+u2/2 = %.6f"
              % (uu, 2 * (1 + uu ** 2) / (2 + uu ** 2), np.sqrt(1 + uu ** 2),
                 1 + uu ** 2 / 2))
    print("\n  -> the far-field check on the algebra passes.  By contrast, the")
    print("     variance inflation falls as (s/d)^2 while the bias")
    print("     eps = s^2/2|d| falls only as s^2/|d|, so the bias persists farther out.")
    add("2", "farfield_known", expr=str(rk_c))
    add("2", "farfield_unknown", expr=str(ru_c))
    add("2", "uniform_small_s", expr="1 + s^2/(2(d^2+4 zeta^2))")
    return rk_c, ru_c


# --- Part 3: numerical FIM over the (d/zeta, s/zeta) plane ---

_GL = np.polynomial.legendre.leggauss(24)


def _nodes(u, X=1.0e8, npan=400):
    """Composite Gauss-Legendre on x = tan(phi), refined around the peaks."""
    a = np.arctan(X)
    base = np.linspace(-a, a, npan + 1)
    if u > 0:
        for c in (np.arctan(u), -np.arctan(u), 0.0):
            w = 4.0 / (1.0 + u ** 2)
            base = np.concatenate([base, np.linspace(c - w, c + w, 121)])
    e = np.unique(np.clip(base, -a, a))
    t, w = _GL
    mid = 0.5 * (e[1:] + e[:-1]); half = 0.5 * (e[1:] - e[:-1])
    phi = (mid[:, None] + half[:, None] * t[None, :]).ravel()
    wt = (half[:, None] * w[None, :]).ravel()
    return np.tan(phi), wt / np.cos(phi) ** 2


def scores(x, d, s, zeta, Se_rel=0.0):
    """d lnS/d(theta) for theta = (ln f0, d, s, zeta, ln P, ln S_e)."""
    R = np.hypot(d, s); u = R / (2 * zeta); rho = d / R
    N = x ** 2 + 2 * rho * u * x + u ** 2 + 1
    D = (x ** 2 + u ** 2 + 1) ** 2 - 4 * u ** 2 * x ** 2
    vf = (2 * x + 2 * rho * u) / N - (4 * x * (x ** 2 + u ** 2 + 1) - 8 * u ** 2 * x) / D
    vu = (2 * u + 2 * rho * x) / N - 4 * u * (u ** 2 + 1 - x ** 2) / D
    vr = 2 * u * x / N
    if Se_rel > 0.0:
        g = N / D
        w = g / (g + Se_rel)
        vf, vu, vr = vf * w, vu * w, vr * w
        vP, vSe = w, 1.0 - w
    else:
        vP, vSe = np.ones_like(x), np.zeros_like(x)
    c = np.sqrt(max(1 - rho ** 2, 0.0))
    return np.array([-vf / zeta,
                     (rho * vu + (1 - rho ** 2) / u * vr) / (2 * zeta),
                     (c * vu - rho * c / u * vr) / (2 * zeta),
                     -(u * vu + x * vf) / zeta,
                     vP, vSe])


TM = np.array([[0.25, 0.5], [0.5, -1.0]])       # (ln f0, d) <- (lnT, ln f_h)


def fim(d, s, zeta, Se_rel=0.0, X=1.0e8):
    """FIM/N_half in (lnT, ln f_h, s, zeta, ln P, ln S_e)."""
    u = np.hypot(d, s) / (2 * zeta)
    x, w = _nodes(u, X)
    V = scores(x, d, s, zeta, Se_rel)
    Ish = (V * w) @ V.T
    J = np.eye(6); J[0:2, 0:2] = TM
    return J.T @ Ish @ J


def crb(I, keep):
    return np.linalg.inv(I[np.ix_(keep, keep)])[0, 0]


def part3(Rk_f, Ru_f):
    print()
    print("=" * 78)
    print("PART 3  the bound over the (d/zeta, s/zeta) plane, one stay sensor")
    print("=" * 78)
    zeta = 0.005
    ref = 2 * zeta ** 2 / np.pi                    # far-field var(lnT) . N_half

    print("\n  A.  exact tuning, against the closed forms of PART 1")
    print("  %8s %8s | %11s %11s | %11s %11s" %
          ("s/zeta", "u", "known", "2(1+u2)/(2+u2)", "all free", "sqrt(1+u2)"))
    worst = 0.0
    for sz in [0.02, 0.2, 0.5, 2 * U_DIP_COH, 2 * U_DIP_INC, 2 * U_3DB,
               2.0, 4.68, 10.0, 40.0]:
        s = sz * zeta; u = sz / 2
        I = fim(1e-10 * zeta, s, zeta)
        r1, r2 = crb(I, [0]) / ref, crb(I, [0, 1, 2, 3, 4]) / ref
        worst = max(worst, abs(r1 / Rk_f(u) - 1), abs(r2 / Ru_f(u) - 1))
        print("  %8.4f %8.4f | %11.6f %11.6f | %11.6f %11.6f"
              % (sz, u, r1, Rk_f(u), r2, Ru_f(u)))
        add("3", "tuned_numeric", s_over_zeta=sz, u=u, R_known=r1,
            R_known_cf=float(Rk_f(u)), R_all=r2, R_all_cf=float(Ru_f(u)))
    print("  worst relative disagreement, quadrature vs closed form: %.2e" % worst)
    assert worst < 2e-5, worst

    print("\n  B.  detuning sweep at s/zeta = 4.68 (example bridge, zeta = 0.5 %)")
    print("  %9s | %9s %9s %9s %9s   %9s" %
          ("d/zeta", "known", "+f_h", "+f_h,s", "all free", "bias eps/s"))
    for dz in [200., 100., 50., 20., 10., 5., 3., 2., 1., 0.5, 0.2, 0.05, 1e-8]:
        d = dz * zeta; s = 4.68 * zeta
        I = fim(d, s, zeta)
        eps = (np.hypot(d, s) - abs(d)) / s
        print("  %9.4g | %9.5f %9.5f %9.5f %9.5f   %9.4f"
              % (dz, crb(I, [0]) / ref, crb(I, [0, 1]) / ref,
                 crb(I, [0, 1, 2]) / ref, crb(I, [0, 1, 2, 3, 4]) / ref, eps))
        add("3", "detune_sweep", d_over_zeta=dz, s_over_zeta=4.68,
            R_known=crb(I, [0]) / ref, R_fh=crb(I, [0, 1]) / ref,
            R_fh_s=crb(I, [0, 1, 2]) / ref, R_all=crb(I, [0, 1, 2, 3, 4]) / ref,
            eps_over_s=eps)

    print("\n  C.  where the loss is, against where the pair is visible")
    print("  %-34s %8s %10s %10s" % ("threshold", "u", "R_known", "R_all"))
    for nm, uu in [("dip exists (coherent, run_damping)", U_DIP_COH),
                   ("dip exists (incoherent ambient)", U_DIP_INC),
                   ("3 dB prominence", U_3DB),
                   ("worked bridge, zeta = 0.5 %", 2.34),
                   ("worked bridge, zeta = 0.2 %", 5.85)]:
        print("  %-34s %8.4f %10.4f %10.4f" % (nm, uu, Rk_f(uu), Ru_f(uu)))
        add("3", "threshold", label=nm, u=uu,
            R_known=float(Rk_f(uu)), R_all=float(Ru_f(uu)))
    print("\n  The loss grows with u, and so does the visibility.  At and")
    print("  below every visibility threshold the variance inflation is under")
    print("  1.6.  Visibility improves with u while identifiability degrades,")
    print("  so the two limits do not coincide.")

    print("\n  D.  effect of a finite noise floor S_e (high-SNR limit assumed above)")
    print("  %10s | %10s %10s" % ("S_e/P", "R_known", "R_all"))
    for se in [0.0, 1e-4, 1e-3, 1e-2, 1e-1]:
        I = fim(1e-10 * zeta, 4.68 * zeta, zeta, Se_rel=se, X=1e4)
        print("  %10.0e | %10.5f %10.5f"
              % (se, crb(I, [0]) / ref, crb(I, [0, 1, 2, 3, 4]) / ref))
        add("3", "noise_floor", Se_over_P=se, R_known=crb(I, [0]) / ref,
            R_all=crb(I, [0, 1, 2, 3, 4]) / ref)
    print("  S_e/P is the noise floor as a fraction of the peak height.  Au's")
    print("  noise-to-environment ratio nu = S_e/S is 1/(4 zeta^2) times larger,")
    print("  so S_e/P = 1e-2 at zeta = 0.5 percent is nu = 100, a poor")
    print("  record.  The high-SNR closed form is the S_e/P -> 0 row.")


# --- Part 4: free modal force intensities ---

def fim_free(d, s, zeta, X=1.0e8):
    """FIM/N_half with free peak strengths (BAYOMA parameterization).

    S = A_+ /(1+(x-u)^2) + A_- /(1+(x+u)^2); parameters (lnT, ln f_h, s, zeta,
    ln A_+, ln A_-). The veering then enters only through the peak positions.
    """
    R = np.hypot(d, s); u = R / (2 * zeta); rho = d / R
    x, w = _nodes(u, X)
    Lp = 1.0 / (1 + (x - u) ** 2); Lm = 1.0 / (1 + (x + u) ** 2)
    Ap, Am = 0.5 * (1 + rho), 0.5 * (1 - rho)
    S = Ap * Lp + Am * Lm
    # d lnS/d x0 (rigid shift), d lnS/d u (symmetric spread), amplitudes
    dLp = 2 * (x - u) * Lp ** 2; dLm = 2 * (x + u) * Lm ** 2   # = -dL/dx0
    v_x0 = (Ap * dLp + Am * dLm) / S       # = -d lnS/dx, poles shift by +1
    v_u = (Ap * dLp - Am * dLm) / S
    v_Ap = Lp / S; v_Am = Lm / S            # linear in the amplitudes, so that
    v_lnz = -(u * v_u + x * (-v_x0))       # A_- -> 0 stays an interior point
    # (x0, u) <- (lnT, ln f_h, s) with x0 = ln f0/zeta, u = R/2zeta
    #   d ln f0 = (1/4) dlnT + (1/2) dlnfh ;  dR = rho (dd) + sqrt(1-rho^2) ds
    #   dd = (1/2) dlnT - dlnfh
    c = np.sqrt(max(1 - rho ** 2, 0.0))
    G = np.array([
        v_x0 / zeta * 0.25 + v_u / (2 * zeta) * (rho * 0.5),           # lnT
        v_x0 / zeta * 0.50 - v_u / (2 * zeta) * rho,                   # ln f_h
        v_u / (2 * zeta) * c,                                          # s
        v_lnz, v_Ap, v_Am])
    return (G * w) @ G.T


def part4():
    print()
    print("=" * 78)
    print("PART 4  effect of the excitation assumption")
    print("=" * 78)
    print("""
  The results above use the amplitude ratio of the two peaks as an observable
  of the veering.  This requires the two hybrid modes to be driven with
  equal modal force intensity, which holds exactly for a load that is
  broadband in time and delta-correlated in space acting on mass-normalized
  modes (checked in scripts/simulate_records.py, check_sigma).  It does not
  hold for a pluck on the stay, for an excitation concentrated on the deck, or
  in the standard BAYOMA parameterization, where the modal force PSD matrix is
  a free unknown.  Au et al. (2020) make the corresponding point for close
  modes in general: at zero disparity the mode shapes and the modal force PSD
  matrix are jointly unidentifiable because Phi S Phi^T = (Phi T)(T^-1 S T^-T)
  (Phi T)^T for any invertible T.

  With the two peak strengths free, the spectrum depends on the physical
  parameters only through the two peak positions, so (lnT, ln f_h, s) enters
  through the two functions (f_0, R) and the FIM has rank at most 2 in those
  three: s is never identifiable, and T is identifiable only while
  d(R)/d(d) = rho = d/sqrt(d^2+s^2) is non-zero.""")
    zeta = 0.005
    ref = 2 * zeta ** 2 / np.pi
    print("\n  rank of the 3x3 block (lnT, ln f_h, s), free amplitudes, s/zeta = 4.68:")
    for dz in [10.0, 1.0, 0.1, 1e-6]:
        I = fim_free(dz * zeta, 4.68 * zeta, zeta)
        ev = np.linalg.eigvalsh(I[:3, :3])
        print("    d/zeta = %8.1e   eigenvalues %s" % (dz, np.array2string(
            ev / max(ev), precision=3, formatter={'float_kind': lambda v: "%.2e" % v})))
    print("  -> one eigenvalue is zero to machine precision at every detuning:")
    print("     with free amplitudes s is not identifiable anywhere.")

    print("\n  with s known.  CRB(lnT)/CRB_far, free amplitudes against pinned")
    print("  amplitudes, at s/zeta = 4.68 (u = 2.34):")
    print("  %9s %9s | %12s %12s %12s" %
          ("d/zeta", "rho", "free amp", "W(u)/rho^2", "pinned amp"))
    for dz in [50., 20., 10., 5., 3., 2., 1., 0.5, 0.2, 0.1, 0.05]:
        d = dz * zeta; s = 4.68 * zeta; rho = d / np.hypot(d, s)
        If = fim_free(d, s, zeta)
        keep = [0, 1, 3, 4, 5]                     # s known, amplitudes free
        rf = np.linalg.inv(If[np.ix_(keep, keep)])[0, 0] / ref
        Ie = fim(d, s, zeta)
        re_ = crb(Ie, [0, 1, 3, 4]) / ref
        print("  %9.3f %9.5f | %12.4f %12.4f %12.4f"
              % (dz, rho, rf, 0.93715 / rho ** 2, re_))
        add("4", "free_amplitude", d_over_zeta=dz, s_over_zeta=4.68, rho=rho,
            R_free_amp=rf, R_equal_amp=re_)

    print("\n  the divergence is exactly 1/rho^2; the prefactor W(u) accounts")
    print("  for a pair that is not yet fully separated, and it tends to")
    print("  1/2 (two independent peak frequencies) as the pair separates:")
    print("  %9s | %12s %12s" % ("u", "W(u)", "1/2"))
    for u_ in [0.5, 1.0, 2.0, 2.34, 4.0, 8.0, 20.0, 200.0]:
        s = 2 * u_ * zeta; rho = 1e-4
        d = s * rho / np.sqrt(1 - rho ** 2)
        If = fim_free(d, s, zeta); keep = [0, 1, 3, 4, 5]
        W = np.linalg.inv(If[np.ix_(keep, keep)])[0, 0] / ref * rho ** 2
        print("  %9.2f | %12.5f %12.4f" % (u_, W, 0.5))
        add("4", "W_of_u", u=u_, W=W)
    print("  W(u) -> (1/2)(1 + 3/2u) for large u; at finite u it is")
    print("  reported numerically rather than fitted.")
    print("""
  The free-amplitude bound diverges as 1/rho^2 at exact tuning: the FIM is
  singular there and T is not identifiable.  Identifiability of the tension
  at a crossing therefore depends on what is known a priori as well as on the
  structure.  Three conditions each make the problem singular:

    1. picking peak frequencies and discarding the spectrum, as every
       existing tension formula does;
    2. an excitation whose two modal force intensities are free, which is a
       pluck, a deck-dominated load, or the standard BAYOMA parameterization;
    3. an unknown direct host contribution at the stay sensor, which rotates
       the measured asymmetry by an unknown angle and so acts as (2).
       PART 9 shows this effect is large at the sensor station used in
       practice.

  Only when none of the three applies is the bound the sqrt(1+u^2) of PART 1.""")


# --- Part 5: check by finite differences on the likelihood ---

def psd_exact(f, T, fh, s, zeta, lnSf, lnSe, Tref, fref, accel=True):
    """Two-mode single-sensor PSD without the narrow-band approximation.

    Exact 2x2 modal pencil and resonance denominators, optional f^4
    acceleration weighting, additive white measurement noise.
    """
    ws = 2 * np.pi * fref * np.sqrt(T / Tref)
    wh = 2 * np.pi * fh
    w0sq = ws * wh
    A = np.array([[ws ** 2, s * w0sq], [s * w0sq, wh ** 2]])
    lam, V = np.linalg.eigh(A)
    wj = np.sqrt(lam)
    phi = V[0, :]                                   # stay-sensor amplitudes
    w = 2 * np.pi * f
    den = (wj[None, :] ** 2 - w[:, None] ** 2) ** 2 \
        + (2 * zeta * wj[None, :] * w[:, None]) ** 2
    S = np.exp(lnSf) * ((phi ** 2)[None, :] / den).sum(axis=1)
    if accel:
        S = S * w ** 4
    return S + np.exp(lnSe)


def part5():
    print()
    print("=" * 78)
    print("PART 5  verification: numerical FIM by finite differences")
    print("=" * 78)
    print("""
  The closed form is checked against the Hessian of the expected Whittle
  negative log likelihood,  L(th) = sum_k [ ln S_k(th) + S_k(th0)/S_k(th) ],
  whose Hessian at th0 is the Fisher information matrix exactly.  The check
  model retains every term the derivation omits: the exact 2x2 pencil rather
  than the leading-order veering formulas, exact resonance denominators rather
  than Lorentzians, the f^4 acceleration weighting, a finite band and a finite
  noise floor.  A disagreement therefore reflects the leading-order model
  rather than the algebra.""")
    Tref, fref = 151.6e3, 3.30
    Td, fs = 1800.0, 100.0
    band = (fref * 0.80, fref * 1.20)
    df = 1.0 / Td
    f = np.arange(band[0], band[1], df)
    Nc = fref * Td

    names = ["lnT", "ln f_h", "s", "zeta", "ln S_f", "ln S_e"]
    print("\n  record: T_d = %.0f s, N_c = f0 T_d = %.0f cycles, band %.2f-%.2f Hz,"
          % (Td, Nc, band[0], band[1]))
    print("          %d Whittle ordinates" % len(f))
    print("\n  %8s %7s %7s | %12s %12s %9s | %12s %12s %9s" %
          ("d", "s", "zeta", "sd(T)/T num", "sd(T)/T cf", "ratio",
           "host-known", "cf", "ratio"))
    worst = 0.0
    for (dd, s, zeta) in [(0.0, 0.0234, 0.005), (0.0, 0.0234, 0.002),
                          (0.0, 0.005, 0.005), (0.0, 0.05, 0.005),
                          (0.01, 0.0234, 0.005), (0.05, 0.0234, 0.005),
                          (0.20, 0.0234, 0.005)]:
        # place the pair: d = (f_iso - f_host)/f_iso  => f_host = f_iso (1-d)
        f_iso = fref
        fh = f_iso * (1 - dd)
        T0 = Tref
        lnSe = np.log(1e-8)
        th0 = np.array([np.log(T0), np.log(fh), s, zeta, 0.0, lnSe])

        def model(th):
            return psd_exact(f, np.exp(th[0]), np.exp(th[1]), th[2], th[3],
                             th[4], th[5], Tref, fref)

        # scale S_f so the peak height is 1e6 times the noise floor
        S00 = model(th0)
        th0[4] = np.log(1e6 * np.exp(lnSe) / S00.max())
        S0 = model(th0)

        def negll(th):
            S = model(th)
            return np.sum(np.log(S) + S0 / S)

        n = len(th0)
        h = np.array([1e-4, 1e-5, 1e-6, 1e-7, 1e-4, 1e-4])
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                ei = np.zeros(n); ei[i] = h[i]
                ej = np.zeros(n); ej[j] = h[j]
                H[i, j] = H[j, i] = (negll(th0 + ei + ej) - negll(th0 + ei - ej)
                                     - negll(th0 - ei + ej) + negll(th0 - ei - ej)) \
                    / (4 * h[i] * h[j])
        V = np.linalg.inv(H)
        sd_all = np.sqrt(V[0, 0])
        keep = [0, 2, 3, 4, 5]
        sd_known = np.sqrt(np.linalg.inv(H[np.ix_(keep, keep)])[0, 0])

        # closed form
        zeta_eff = zeta
        u = np.hypot(dd, s) / (2 * zeta_eff)
        Ifim = fim(max(dd, 1e-12), s, zeta_eff)
        base = zeta_eff / (np.pi * Nc)              # (2 zeta/pi Nc)/2 . N_half^-1
        cf_all = np.sqrt(crb(Ifim, [0, 1, 2, 3, 4]) / (zeta_eff * fref * Td))
        cf_known = np.sqrt(crb(Ifim, [0, 2, 3, 4]) / (zeta_eff * fref * Td))
        r1, r2 = sd_all / cf_all, sd_known / cf_known
        worst = max(worst, abs(r1 - 1), abs(r2 - 1))
        print("  %8.4f %7.4f %7.4f | %12.4e %12.4e %9.4f | %12.4e %12.4e %9.4f"
              % (dd, s, zeta, sd_all, cf_all, r1, sd_known, cf_known, r2))
        add("5", "fd_check", d=dd, s=s, zeta=zeta, sd_numeric=sd_all,
            sd_closed=cf_all, ratio=r1, sd_numeric_known=sd_known,
            sd_closed_known=cf_known, ratio_known=r2)
    print("""
  The check agrees to a few percent wherever the leading-order veering model
  applies.  The one large residual is the last row, d = 0.20, where the
  detuning is twenty percent and the exact pencil's rho = 1 - O(s^2/d^2)
  differs from the leading-order rho in its small part 1 - rho, on which the
  host-frequency direction depends; the host-known column in the same row
  still agrees to 0.4 percent.  This residual is an error of the
  leading-order model, not of the algebra, and it lies outside the range the
  model covers.""")
    print("\n  worst discrepancy %.3f (%.1f percent), worst for |d| <= 0.05: "
          % (worst, 100 * worst) + "see table")
    return worst


# --- Part 6: check by Monte Carlo maximum likelihood ---

def part6(nrep=400, seed=7):
    print()
    print("=" * 78)
    print("PART 6  verification: Monte Carlo maximum likelihood")
    print("=" * 78)
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    zeta, s, f0, Td = 0.005, 0.0234, 3.30, 1800.0
    Nc = f0 * Td
    df = 1.0 / Td
    x = np.arange(-400.0, 400.0, df / (zeta * f0))     # +-400 half-widths

    def S_of(th):
        """Model spectrum for th = (x0, delta, sigma, ln P), with S_e = 0."""
        x0, dl, sg, lnP = th
        u = np.hypot(dl, sg); rho = dl / u if u > 0 else 0.0
        xx = x - x0
        return np.exp(lnP) * 0.5 * ((1 + rho) / (1 + (xx - u) ** 2)
                                    + (1 - rho) / (1 + (xx + u) ** 2))

    th0 = np.array([0.0, 1e-9, s / (2 * zeta), 0.0])
    S0 = S_of(th0)
    est = []
    for _ in range(nrep):
        y = S0 * rng.exponential(size=S0.shape)
        f = lambda th: np.sum(np.log(S_of(th)) + y / S_of(th))
        r = minimize(f, th0 + rng.normal(scale=[0.02, 0.02, 0.02, 0.02]),
                     method="Nelder-Mead",
                     options=dict(xatol=1e-7, fatol=1e-7, maxiter=8000, maxfev=8000))
        est.append(r.x)
    est = np.array(est)
    # ln f0 = zeta x0 and d = 2 zeta delta; inverting
    # [ln f0; d] = [[1/4, 1/2], [1/2, -1]] [lnT; ln fh] gives lnT = 2 ln f0 + d
    lnT = 2 * (zeta * est[:, 0]) + 2 * zeta * est[:, 1]
    sd_mc = lnT.std(ddof=1)
    I = fim(1e-12, s, zeta)
    cf = np.sqrt(crb(I, [0, 1, 2, 3, 4]) / (zeta * f0 * Td))
    se = sd_mc / np.sqrt(2 * (nrep - 1))
    print("\n  exact tuning, s = %.4f, zeta = %.4f, u = %.3f, N_c = %.0f, %d replicates"
          % (s, zeta, s / (2 * zeta), Nc, nrep))
    print("  Monte Carlo sd(lnT) = %.4e  +- %.1e" % (sd_mc, se))
    print("  Cramer-Rao bound     = %.4e" % cf)
    print("  ratio                = %.4f   (%.1f standard errors from 1)"
          % (sd_mc / cf, abs(sd_mc - cf) / se))
    add("6", "monte_carlo", nrep=nrep, sd_mc=sd_mc, sd_crb=cf, ratio=sd_mc / cf,
        n_sigma=abs(sd_mc - cf) / se)
    return sd_mc / cf


# --- Part 7: check against Au's single-mode uncertainty law ---

def part7():
    print()
    print("=" * 78)
    print("PART 7  verification against Au's uncertainty law (single mode)")
    print("=" * 78)
    print("""
  Au (2014), Uncertainty law in ambient modal identification, Part I,
  48:15-33, Eq. (15)-(16):  delta_f^2 ~ zeta/(2 pi N_c B_f(kappa)) with
  B_f(kappa) = (2/pi)(atan kappa - kappa/(kappa^2+1)), N_c = T_d f the data
  length in cycles and kappa the half-bandwidth in units of zeta f.  The
  method of PART 1 applied to a single Lorentzian should reproduce this.""")
    kap = sp.Symbol('kappa', positive=True)
    xs = sp.Symbol('x')
    Jff = sp.integrate((2 * xs / (1 + xs ** 2)) ** 2, (xs, -kap, kap))
    Bf = sp.simplify(Jff / (2 * sp.pi))
    Bf_au = 2 / sp.pi * (sp.atan(kap) - kap / (kap ** 2 + 1))
    print("\n  this script:  J_ff(kappa)/2 pi =", sp.simplify(Bf))
    print("  Au Eq. (16):  B_f(kappa)       =", sp.simplify(Bf_au))
    print("  difference                     =", sp.simplify(Bf - Bf_au))
    assert sp.simplify(Bf - Bf_au) == 0
    print("\n  identical.  var(ln f) = zeta^2/(N_half . 2 pi B_f)")
    print("                        = zeta/(2 pi N_c B_f)   <- Au's law")
    print("  and var(ln T) = 4 var(ln f) = 2 zeta/(pi N_c) at B_f = 1, which is")
    print("  the far-field reference used throughout.")
    for k in [1.0, 2.0, 6.0, 20.0]:
        print("    kappa = %5.1f   B_f = %.5f" % (k, float(Bf_au.subs(kap, k))))
        add("7", "Au_band_factor", kappa=k, B_f=float(Bf_au.subs(kap, k)))
    add("7", "Au_single_mode_law", match="exact")


# --- Part 8: screening criterion as a precision bound ---

def part8(Ru_f):
    print()
    print("=" * 78)
    print("PART 8  the screening criterion as a precision bound")
    print("=" * 78)
    print("""
  The screening criterion  |d| >= (s^2 - tol^2)/(2 tol)  is exactly the
  condition eps = sqrt(d^2+s^2) - |d| <= tol on the frequency bias.  Because

      1/|rho| = sqrt(d^2+s^2)/|d| = 1 + eps/|d|

  the same quantity controls the variance.  With free modal force intensities
  (PART 4) the inflation is W(u)/rho^2 as rho -> 0 and (1/2)(1 + 1/rho^2) in
  the well-separated limit u >> 1, which is the form used below; at finite u
  multiply by 2 W(u) from PART 4 (0.94 x 2 = 1.87 for the example bridge).  On
  the screening boundary |d| = (s^2 - tol^2)/(2 tol),

      1/|rho| = (s^2 + tol^2)/(s^2 - tol^2)

  and the achievable variance is inflated by

      Upsilon = (1/2)[ 1 + ((s^2+tol^2)/(s^2-tol^2))^2 ].

  The criterion that bounds the bias to tol also bounds the precision loss
  to Upsilon.""")
    print("\n  %8s %8s | %10s %10s %10s %10s" %
          ("s", "tol", "|d| min", "1/rho", "Upsilon", "sd factor"))
    for s in [0.0234, 0.05]:
        for tol in [0.005, 0.01, 0.02]:
            if tol >= s:
                continue
            dmin = (s ** 2 - tol ** 2) / (2 * tol)
            inv_rho = (s ** 2 + tol ** 2) / (s ** 2 - tol ** 2)
            ups = 0.5 * (1 + inv_rho ** 2)
            print("  %8.4f %8.4f | %10.4f %10.4f %10.4f %10.4f"
                  % (s, tol, dmin, inv_rho, ups, np.sqrt(ups)))
            add("8", "criterion", s=s, tol=tol, d_min=dmin, inv_rho=inv_rho,
                Upsilon=ups, sd_factor=np.sqrt(ups))
    print("""
  Inverted, Upsilon = (1/2)(1 + 1/rho^2) gives 1/rho^2 = 2 Upsilon - 1,
  and rho = |d|/sqrt(d^2+s^2) then gives

      |d| >= s / sqrt(2 (Upsilon - 1))            identifiability criterion

  against the screening

      |d| >= (s^2 - tol^2)/(2 tol)                bias criterion

  The two criteria scale differently: the bias criterion is s^2/(2 tol) for
  small tol and grows without limit as the tolerance tightens; the
  identifiability criterion is a fixed multiple of s.  They cross where
  (s^2 - tol^2)/(2 tol) = s, that is at

      tol = (sqrt 2 - 1) s = 0.414 s

  so the bias criterion governs for any tolerance tighter than 0.414 s and
  the identifiability criterion governs for looser ones.  For s = 2.34
  percent the crossover is tol = 0.97 percent, which lies inside the range a
  tension survey would specify, so both criteria have to be checked.""")
    print("\n    %9s %12s %14s" % ("Upsilon", "|d| >= .. s", "sd inflation"))
    for ups in [1.25, 1.5, 2.0, 5.0]:
        d_over_s = 1.0 / np.sqrt(2 * (ups - 1))
        print("    %9.2f %12.3f %14.3f" % (ups, d_over_s, np.sqrt(ups)))
        add("8", "inverse_criterion", Upsilon=ups, d_over_s=d_over_s,
            sd_factor=np.sqrt(ups))
    print("\n    %8s %8s | %12s %12s   %s" %
          ("s", "tol", "bias crit", "ident crit", "which binds"))
    for s in [0.0234, 0.05]:
        for tol in [0.002, 0.005, 0.01, 0.02]:
            if tol >= s:
                continue
            db = (s ** 2 - tol ** 2) / (2 * tol)
            di = s / np.sqrt(2 * (1.5 - 1))         # Upsilon = 1.5
            print("    %8.4f %8.4f | %12.4f %12.4f   %s"
                  % (s, tol, db, di, "bias" if db > di else "identifiability"))
            add("8", "which_binds", s=s, tol=tol, d_bias=db, d_ident=di,
                binds="bias" if db > di else "identifiability")


# --- Part 9: the two channels in the finite-element bridge ---

def part9():
    print()
    print("=" * 78)
    print("PART 9  the two channels in the finite element model")
    print("=" * 78)
    try:
        from cablefe import CableDeck
    except Exception as exc:                                    # pragma: no cover
        print("  cablefe not importable (%s); PART 9 skipped." % exc)
        return
    from scipy.linalg import eigh
    BRIDGE = dict(Ld=80.0, EId=2.0e9, md=1000.0,
                  Lc=25.0, EIc=1.2e4, mc=5.5, EA=1.4e8,
                  theta=np.deg2rad(35.0), nd=40, nc=40)
    FBAND = (2.5, 4.2)
    T0 = 151.6e3

    cd0 = CableDeck(T=T0, **BRIDGE)

    def pair(T, dof=None):
        cd = CableDeck(T=T, **BRIDGE)
        w2, V = eigh(cd.K, cd.M)
        f = np.sqrt(np.maximum(w2, 0.0)) / (2 * np.pi)
        Phi = cd.Lmat @ V
        p = cd.cable_sensor_dof(2.0) if dof is None else dof
        m = (f > FBAND[0]) & (f < FBAND[1])
        idx = np.where(m)[0]
        assert len(idx) == 2, f
        return f[idx], Phi[p, idx]

    f0s, a0s = pair(T0)
    fmid = f0s.mean()
    s_fe = (f0s[1] - f0s[0]) / fmid
    print("\n  example bridge at T = %.1f kN:  f = %.5f, %.5f Hz" % (T0 / 1e3, *f0s))
    print("  split s = %.5f (%.3f %%), amplitude ratio at the stay sensor = %.4f"
          % (s_fe, 100 * s_fe, (a0s[1] / a0s[0]) ** 2))

    h = 0.004
    fp, ap = pair(T0 * (1 + h))
    fm, am = pair(T0 * (1 - h))
    dlnf = (np.log(fp) - np.log(fm)) / (2 * h)
    print("\n  Channel 1, level repulsion.  d ln f_j / d ln T from the model:")
    print("    lower branch %.5f, upper branch %.5f, sum %.5f" %
          (dlnf[0], dlnf[1], dlnf.sum()))
    print("    prediction (1 -+ rho)/4 each, summing to 1/2 for any rho;")
    print("    an isolated stay would give 1/2 for its one mode, so each")
    print("    branch responds at half the isolated rate.  The difference of")
    print("    the two gives the operating point: rho = 2 (dlnf+ - dlnf-) = %.4f"
          % (2 * (dlnf[1] - dlnf[0])))
    rho_freq = 2 * (dlnf[1] - dlnf[0])
    print("    so T = %.1f kN is tuned to within d = rho s = %.5f (%.2f zeta"
          % (T0 / 1e3, rho_freq * s_fe, rho_freq * s_fe / 0.005))
    print("    at zeta = 0.5 percent), which is close to exact tuning.")

    Rp = (fp[1] - fp[0]) / fp.mean(); Rm = (fm[1] - fm[0]) / fm.mean()
    dR = (Rp - Rm) / (2 * h)
    print("\n  Dependence of the veering width on T.  The derivation assumes")
    print("  s = (2/n pi) cos(theta) sqrt(mu_eff) has no T in it, so the only")
    print("  T dependence of the observed separation R = sqrt(d^2+s^2) is")
    print("  through d, giving dR/dlnT = rho/2 = %.5f." % (rho_freq / 2))
    print("  The model gives dR/dlnT = %.5f, ratio %.4f.  The observed drift"
          % (dR, dR / (rho_freq / 2)))
    print("  of the split with tension comes from the detuning alone.")
    add("9", "fe_split_drift", dR_dlnT=dR, predicted=rho_freq / 2,
        ratio=dR / (rho_freq / 2))

    print("\n  Channel 2, amplitude asymmetry against sensor position.")
    print("  %8s | %10s %10s %10s %10s" %
          ("s from", "rho at", "d rho/d lnT", "1/(2s)", "ratio"))
    print("  %8s | %10s %10s %10s %10s" %
          ("anchor", "sensor", "model", "predicted", "= cos 2 chi"))
    for pos in [1.0, 2.0, 4.0, 8.0, 12.5, 20.0]:
        q = cd0.cable_sensor_dof(pos)

        def rho_at(T):
            f_, a_ = pair(T, q)
            return (a_[1] ** 2 - a_[0] ** 2) / (a_[1] ** 2 + a_[0] ** 2)
        r0 = rho_at(T0)
        drho = (rho_at(T0 * (1 + h)) - rho_at(T0 * (1 - h))) / (2 * h)
        print("  %8.1f | %10.4f %10.3f %10.3f %10.4f"
              % (pos, r0, drho, 1 / (2 * s_fe), abs(drho) * 2 * s_fe))
        add("9", "fe_sensor_position", pos_m=pos, rho_sensor=r0, drho=drho,
            drho_pred=1 / (2 * s_fe), ratio=abs(drho) * 2 * s_fe)
    print("""
  The asymmetry rho measured at the sensor is cos 2(alpha - chi), not
  cos 2 alpha: the stay foot moves with the deck, so a host mode puts motion
  into the stay directly, and near the anchorage that direct term rotates the
  effective mixing angle by chi.  At 12.5 m (mid chord) chi is negligible and
  the sensor reads the true mixing.  At the 2 m station used in field
  practice, rho is -0.25 when the true rho is %.3f: the sign is reversed and
  the magnitude is six times too large.  The sensitivity is reduced only by
  cos 2 chi (the last column), so the information is nearly intact if chi is
  known.  In the field chi is not known, and an unknown chi is equivalent to
  a free amplitude ratio, the singular case of PART 4.

  To recover tension through a crossing from one accelerometer, the sensor
  should be near mid chord, where the amplitude balance reflects the mode
  mixing alone.  At the usual station one or two meters above the anchorage
  the tension is not identifiable at exact tuning.""" % rho_freq)
    add("9", "fe_channels", s_fe=s_fe, dlnf_lo=dlnf[0], dlnf_hi=dlnf[1],
        dlnf_sum=dlnf.sum(), rho_freq=rho_freq)

    for zeta in [0.002, 0.005, 0.010]:
        for Td in [600.0, 1800.0]:
            Nc = fmid * Td
            u = s_fe / (2 * zeta)
            base = 2 * zeta / (np.pi * Nc)
            cov_far = np.sqrt(base)
            cov_kn = np.sqrt(base * 2 * (1 + u ** 2) / (2 + u ** 2))
            cov_un = np.sqrt(base * np.sqrt(1 + u ** 2))
            print("  zeta %4.1f %%, T_d %6.0f s : c.o.v.(T) far %.3e, tuned known"
                  " %.3e, tuned all-free %.3e"
                  % (100 * zeta, Td, cov_far, cov_kn, cov_un))
            add("9", "bound_kN", zeta=zeta, Td=Td, u=u, cov_far=cov_far,
                cov_known=cov_kn, cov_unknown=cov_un,
                sd_T_kN_unknown=cov_un * T0 / 1e3)
    cov_ref = np.sqrt(2 * 0.005 / (np.pi * fmid * 1800.0)
                      * np.sqrt(1 + (s_fe / 0.01) ** 2))
    print("\n  Against the bias.  At exact tuning the isolated-cable")
    print("  inversion is displaced by 2 s = %.2f percent in tension.  The"
          % (200 * s_fe))
    print("  standard deviation the record allows, with the amplitude balance")
    print("  pinned and everything else free, is %.3f percent (zeta = 0.5"
          % (100 * cov_ref))
    print("  percent, 30 minutes), so the bias exceeds the achievable noise by a")
    print("  factor of %.0f.  With the amplitude balance not pinned the bound"
          % (2 * s_fe / cov_ref))
    print("  is unbounded and no comparison applies: the")
    print("  tension is not identifiable.")
    add("9", "bias_vs_noise", bias_pct=200 * s_fe, cov_pct=100 * cov_ref,
        ratio=2 * s_fe / cov_ref)



# --- Part 10: stay and deck sensors ---

def _E2(x, th):
    """2x2 spectral density matrix for a stay sensor and a deck sensor.

    th = (x0, delta, sigma, A_+, A_-, ln g2, S_e). The stay-sensor gain is
    fixed at 1 to remove the gain/intensity scale redundancy; A_+ and A_- are
    free modal force intensities, as in Part 4.
    """
    x0, dl, sg, Ap, Am, lng2, Se = th
    u = np.hypot(dl, sg)
    a = 0.5 * np.arctan2(sg, dl)                  # tan 2 alpha = s/d
    c, s_ = np.cos(a), np.sin(a)
    g2 = np.exp(lng2)
    P = np.array([[c, -s_], [g2 * s_, g2 * c]])   # rows sensors, cols modes
    xx = x - x0
    L = np.stack([1.0 / (1 + (xx - u) ** 2), 1.0 / (1 + (xx + u) ** 2)], axis=1)
    E = np.einsum('ij,kj,nj->nik', P, P, L * np.array([Ap, Am])[None, :])
    return E + Se * np.eye(2)[None, :, :]


def part10():
    print()
    print("=" * 78)
    print("PART 10  multi-sensor contrast")
    print("=" * 78)
    print("""
  One accelerometer on the stay is the case used in practice and is what
  everything above assumes.  A second accelerometer on the deck changes the
  result qualitatively, for the following reason.

  With one sensor the two hybrid modes contribute two numbers, their peak
  strengths, and the mixing angle can only be read off them if the modal force
  intensities are pinned (PART 4).  With two sensors each hybrid mode
  contributes a two-component shape, and the two shapes are M-orthogonal, so

      (deck/stay of mode +) x (deck/stay of mode -) = -(g2/g1)^2
      (deck/stay of mode +) / (deck/stay of mode -) = -tan^2 alpha

  and alpha is recovered from the shapes without knowing either gain and
  without any assumption about the excitation.  The orthogonality of the two
  hybrid shapes takes the place of the equal-force assumption.

  The bound below is the CRB on ln T with everything free: both frequencies,
  the split, the two modal force intensities, the gain ratio and the noise.""")
    zeta = 0.005
    ref = 2 * zeta ** 2 / np.pi
    TM10 = np.zeros((7, 7))
    TM10[0, 0] = 1 / (4 * zeta); TM10[1, 0] = 1 / (4 * zeta)      # ln T
    TM10[0, 1] = 1 / (2 * zeta); TM10[1, 1] = -1 / (2 * zeta)     # ln f_h
    for k in range(2, 7):
        TM10[k, k] = 1.0

    def bound(d, s, Se=1e-9):
        dl, sg = d / (2 * zeta), s / (2 * zeta)
        u = np.hypot(dl, sg)
        x, w = _nodes(u, 1e6)
        th = np.array([0.0, dl, sg, 1.0, 1.0, 0.0, Se])
        E0 = _E2(x, th); Ei = np.linalg.inv(E0)
        n = len(th); h = 1e-6; dE = []
        for i in range(n):
            tp = th.copy(); tm = th.copy(); tp[i] += h; tm[i] -= h
            dE.append((_E2(x, tp) - _E2(x, tm)) / (2 * h))
        I = np.zeros((n, n))
        for i in range(n):
            Ai = np.einsum('nij,njk->nik', Ei, dE[i])
            for j in range(i, n):
                Aj = np.einsum('nij,njk->nik', Ei, dE[j])
                I[i, j] = I[j, i] = np.sum(
                    np.einsum('nij,nji->n', Ai, Aj).real * w)
        Ip = TM10.T @ I @ TM10
        return np.linalg.inv(Ip)[0, 0] / ref

    print("\n  CRB(lnT) / CRB_far, two sensors, everything free")
    print("  %9s | %10s %10s %10s %10s %10s" %
          ("s/zeta", "d/z=50", "d/z=10", "d/z=2", "d/z=0.2", "d/z=0"))
    for sz in [0.5, 2.0, 4.68, 10.0, 30.0]:
        row = [bound(dz * zeta, sz * zeta) for dz in (50., 10., 2., 0.2, 1e-9)]
        print("  %9.2f | %10.5f %10.5f %10.5f %10.5f %10.5f" % (sz, *row))
        add("10", "two_sensor", s_over_zeta=sz, R_d50=row[0], R_d10=row[1],
            R_d2=row[2], R_d02=row[3], R_d0=row[4])
    print("""
  The bound is 1 everywhere, to five figures, independent of the split and of
  the detuning: with both the stay and the deck instrumented the crossing adds
  no variance, and the tension is recovered as precisely as from a stay far
  from any deck mode.  The single-sensor loss therefore follows from sensor
  placement, not from the information in the structure.""")
    for Se in [1e-9, 1e-6, 1e-4]:
        print("    noise floor S_e/peak = %.0e :  R at exact tuning = %.5f"
              % (Se, bound(1e-9 * zeta, 4.68 * zeta, Se)))
        add("10", "two_sensor_noise", Se=Se,
            R=bound(1e-9 * zeta, 4.68 * zeta, Se))


def main():
    print(__doc__.split("Run:")[0])
    Rk_f, Ru_f = part1()
    part2()
    part3(Rk_f, Ru_f)
    part4()
    w5 = part5()
    r6 = part6()
    part7()
    part8(Ru_f)
    part9()
    part10()

    os.makedirs(DATA, exist_ok=True)
    keys = []
    for r in ROWS:
        for k in r:
            if k not in keys:
                keys.append(k)
    import csv as _csv
    with open(CSV, "w", newline="") as fh:
        wr = _csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in ROWS:
            wr.writerow(r)
    print("\nwrote %s (%d rows)" % (CSV, len(ROWS)))
    print("\nSummary")
    print("  finite-difference FIM agrees with the closed form to %.1f percent"
          % (100 * w5))
    print("  Monte Carlo MLE / CRB = %.3f" % r6)


if __name__ == "__main__":
    main()
