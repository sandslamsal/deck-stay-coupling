# -*- coding: utf-8 -*-
"""How precisely does a stressing sweep return the host mode's anchorage ordinate?

THE PROPOSAL UNDER TEST.  The veering width at a deck-stay crossing is

    s = (2 / (n pi)) cos(theta) sqrt(mu_eff),    mu_eff = M_s phi_a^2

with ``M_s = m_c L_c / 2`` the stay modal mass and ``phi_a`` the MASS-NORMALISED
host mode amplitude at the anchorage.  Every quantity but ``s`` is in the stay
schedule, so measuring the split inverts to ``phi_a``, and hence to the host
modal mass presented at that anchorage, ``1 / phi_a^2``.  Operational modal
analysis cannot scale mode shapes without a known perturbation; the claim is
that a stay under stressing IS the known perturbation, applied for free.

This script does not argue the mechanism -- ``scripts/run_veering.py`` and the
campaign already established it, and Part 0 here re-checks the inversion end to
end against the finite element model.  It asks the only question that decides
whether the reversal is a method: HOW PRECISE IS phi_a FROM A REALISTIC SWEEP,
and does that beat the incumbent mass-change method.

THE ABSOLUTE SPLIT, NOT THE RELATIVE ONE, IS THE NATURAL PARAMETER.  The
derivation behind ``veering_split`` gives the half split in absolute terms as
``x = cos(theta) sqrt(T / (2 L_c M_host))``, independent of the stay mode order
n; the ``1/n`` appears only on dividing by ``f_n``.  In Hz,

    g(T) = (cos(theta) / pi) phi_a sqrt(T / (2 L_c))                    (*)

so the coupling GROWS AS sqrt(T) across a stressing sweep.  Fitting a
constant-gap hyperbola is therefore wrong, and Part 0 measures what that costs.
Writing the fit in terms of ``phi_a`` through (*) makes the deliverable a fitted
parameter rather than a post-hoc conversion.

WHAT THE PARTS ESTABLISH.

  0  Ground truth.  phi_a and the host modal mass read directly off the
     deck-alone eigenproblem; the closed form reproduces the FE split to 0.02 %;
     the sqrt(T) hyperbola of (*) tracks both FE branches over a 70 kN sweep to
     0.4 mHz RMS, which is a factor 8 below the frequency noise assumed below.
     The fit model is therefore not the limiting error.
  A  Frequency precision, justified rather than assumed, from Au's uncertainty
     law, and swept because it is the dominant random error source.
  B  A closed form for the precision of phi_a, with its Fisher derivation, and
     the effective-step count N_eff that turns out to govern everything.
  C  The design grid: confidence intervals against step count, frequency
     precision, damping and sweep extent.
  D  What the nuisance parameters cost.  The stay's own f(T) law is not known
     exactly, and every parameter fitted alongside phi_a inflates its variance.
  E  Back-check against the project's one field instance.  The Ponte del Mare
     damper traverse returned a host modal mass between 91 and 959 tonnes; the
     law of Part B has to reproduce that bracket from that traverse's geometry,
     or it is not to be trusted on a sweep.
  F  Refinement (a): does the fit work when the pair is NEVER resolved?
  G  Refinement (b): does the 1/n structure across stay mode orders help?
  H  The practical obstacles, quantified.
  I  Head to head against the mass-change method at matched frequency precision.

Writes data/reversal.csv (long format, ``record`` column).

Run:  python3 scripts/reversal_precision.py
"""

from __future__ import annotations

import os
import sys

# The FE solves here are 160-odd DOF: individually far too small to cover
# multi-threaded BLAS dispatch overhead, and a threaded build spends more time
# in synchronisation than in the eigensolver.  Pinned before numpy is imported,
# which is the only point at which it takes effect.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402
from scipy.interpolate import CubicSpline                       # noqa: E402
from scipy.linalg import eigh                                   # noqa: E402
from scipy.optimize import brentq, least_squares               # noqa: E402
from scipy.signal import find_peaks                             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from cablefe import (CableDeck, chain, irvine_lambda2, mu_effective,  # noqa: E402
                     sag_ratio, tensioned_beam_freq, veering_split)
from run_damping import BRIDGE, DrivenBridge, FBAND, peak_census  # noqa: E402

DATA = os.path.join(ROOT, "data")
RNG = np.random.default_rng(20260814)

T_TUNE = 151.6e3                    # exact-tuning tension of the worked bridge
THETA = BRIDGE["theta"]
L_C = BRIDGE["Lc"]
M_S = 0.5 * BRIDGE["mc"] * L_C      # stay modal mass, kg
COS_T = np.cos(THETA)

ROWS: list[dict] = []


def row(**kw):
    ROWS.append(kw)


# ---------------------------------------------------------------------------
# the observation model
# ---------------------------------------------------------------------------

def g_of_T(phi_a, T):
    """Absolute veering gap in Hz at tension ``T``, equation (*).

    Order-independent, and proportional to sqrt(T): a stressing sweep does not
    hold the coupling fixed while it moves the detuning.
    """
    return COS_T / np.pi * phi_a * np.sqrt(T / (2.0 * L_C))


def phi_from_gap(g, T):
    """Invert (*).  The whole proposal in one line."""
    return np.pi * g / (COS_T * np.sqrt(T / (2.0 * L_C)))


def f_iso(T, c=1.0, EI=BRIDGE["EIc"], n=1, dT=0.0):
    """Isolated tensioned-beam frequency of stay mode ``n``.

    ``c`` is a multiplicative calibration on the whole law -- it absorbs a scale
    error in the assumed mass per length, the effective length, or the jack
    calibration, all of which act on f as a single factor.  ``dT`` is an
    additive tension offset (anchor set, friction), which does NOT act as a
    scale and so is a separate nuisance.
    """
    return c * tensioned_beam_freq(n, L_C, T + dT, EI, BRIDGE["mc"])


def branches(T, phi_a, f_h, c=1.0, EI=BRIDGE["EIc"], n=1, drift=0.0, dT=0.0):
    """Lower and upper hybrid branch, the avoided-crossing hyperbola.

    ``drift`` is a linear change of the host frequency with tension, in relative
    units per T_TUNE: the deck is compressed by the stay being stressed, and
    Part H shows that is not negligible against the split.
    """
    fi = f_iso(T, c, EI, n, dT)
    fh = f_h * (1.0 + drift * (T - T_TUNE) / T_TUNE)
    gg = g_of_T(phi_a, T)
    half = 0.5 * np.sqrt((fi - fh) ** 2 + gg ** 2)
    mid = 0.5 * (fi + fh)
    return mid - half, mid + half


# ---------------------------------------------------------------------------
# PART 0: ground truth, and whether the fit model is good enough to use
# ---------------------------------------------------------------------------

def host_mode_truth():
    """phi_a and the host modal mass from the deck-alone eigenproblem.

    The host is the deck carrying the stay's axial spring but not its transverse
    inertia -- the same reference ``CableDeck.deck_alone`` uses, extended to
    return the mass-normalised ordinate at the anchorage.
    """
    cd = CableDeck(T=T_TUNE, **BRIDGE)
    Kd, Md = chain(cd.Ld, cd.nd, cd.EId, cd.md, 0.0)
    Kd = Kd.copy()
    Kd[2 * cd.ia, 2 * cd.ia] += cd.k_ax
    keep = [i for i in range(Kd.shape[0]) if i not in (0, 2 * cd.nd)]
    w2, V = eigh(Kd[np.ix_(keep, keep)], Md[np.ix_(keep, keep)])
    f = np.sqrt(np.maximum(w2, 0.0)) / (2.0 * np.pi)
    pos = {d: j for j, d in enumerate(keep)}
    fi1 = f_iso(T_TUNE)
    j = int(np.argmin(np.abs(f - fi1)))
    return float(f[j]), float(abs(V[pos[2 * cd.ia], j])), cd


def fe_pair(T, band=(2.3, 4.5)):
    cd = CableDeck(T=T, **BRIDGE)
    f, _ = cd.modes(60)
    inb = f[(f > band[0]) & (f < band[1])]
    assert len(inb) == 2, f"T={T:.0f}: band holds {inb}"
    return float(inb[0]), float(inb[1])


def fe_split_at(T, f_h, halfband=0.25):
    """Relative split of the hybrid pair straddling ``f_h`` at tension ``T``.

    Used to check the ``1/n`` law against the model at the higher orders'
    crossings, where the band around f_h contains modes belonging to other stay
    orders as well.  The pair is selected as the two modes a stay-mounted
    accelerometer would rank highest, which is the operational definition.
    """
    cd = CableDeck(T=T, **BRIDGE)
    f, Phi = cd.modes(80)
    p = cd.cable_sensor_dof(2.0)
    sel = np.where((f > f_h - halfband) & (f < f_h + halfband))[0]
    if len(sel) < 2:
        return np.nan
    amp = np.abs(Phi[p, sel])
    top = sel[np.argsort(amp)[-2:]]
    lo, hi = float(f[top].min()), float(f[top].max())
    return (hi - lo) / (0.5 * (lo + hi))


def part0():
    print("=" * 78)
    print("PART 0   ground truth, and whether the fit model is the limit")
    print("=" * 78)

    f_h, phi_a, cd = host_mode_truth()
    M_host = 1.0 / phi_a ** 2
    mu = mu_effective(M_S, phi_a)
    s_pred = veering_split(mu, THETA, 1)
    lo, hi = fe_pair(T_TUNE)
    f0 = 0.5 * (lo + hi)
    s_fe = (hi - lo) / f0

    print(f"  host mode          f_h      = {f_h:.5f} Hz")
    print(f"  anchorage ordinate phi_a    = {phi_a:.6e} kg^-1/2   (TRUTH)")
    print(f"  host modal mass    1/phi_a^2= {M_host/1e3:.3f} t        (TRUTH)")
    print(f"  stay modal mass    M_s      = {M_S:.2f} kg")
    print(f"  mu_eff = M_s phi_a^2        = {mu:.4e}")
    print(f"  split: closed form {100*s_pred:.4f} %   FE {100*s_fe:.4f} %   "
          f"ratio {s_pred/s_fe:.5f}")
    # the inversion actually used downstream, run forwards then backwards
    phi_back = phi_from_gap(hi - lo, T_TUNE)
    print(f"  inverting the FE gap through (*): phi_a = {phi_back:.6e}  "
          f"({100*(phi_back/phi_a - 1):+.3f} % of truth)")

    row(record="truth", f_h=f_h, phi_a=phi_a, M_host=M_host, M_s=M_S,
        mu_eff=mu, s_closed_form=s_pred, s_fe=s_fe,
        phi_a_inverted=phi_back,
        inversion_err_pct=100 * (phi_back / phi_a - 1), T_tune=T_TUNE)

    # ---- model adequacy: does the sqrt(T) hyperbola track the FE branches? --
    print("\n  model adequacy -- fitting the FE loci with no noise:")
    Ts = np.linspace(120e3, 190e3, 21)
    obs = np.array([fe_pair(T) for T in Ts])

    def resid_factory(power):
        def r(p):
            ph, fh = p
            fi = f_iso(Ts)
            gg = COS_T / np.pi * ph * (Ts / T_TUNE) ** power \
                * np.sqrt(T_TUNE / (2.0 * L_C))
            half = 0.5 * np.sqrt((fi - fh) ** 2 + gg ** 2)
            mid = 0.5 * (fi + fh)
            return np.concatenate([mid - half - obs[:, 0],
                                   mid + half - obs[:, 1]])
        return r

    floor = np.nan
    for power, tag in ((0.0, "constant gap  "), (0.5, "gap ~ sqrt(T) ")):
        res = least_squares(resid_factory(power), [phi_a, f_h])
        rms = float(np.sqrt(np.mean(res.fun ** 2)))
        bias = 100 * (res.x[0] / phi_a - 1)
        if power == 0.5:
            floor = abs(bias)
        print(f"    {tag}: phi_a {res.x[0]:.6e} ({bias:+.3f} %)  "
              f"RMS residual {1e3*rms:.3f} mHz  worst {1e3*np.abs(res.fun).max():.3f} mHz")
        row(record="model_adequacy", gap_power=power, phi_a_fit=res.x[0],
            phi_a_bias_pct=bias, rms_resid_hz=rms,
            max_resid_hz=float(np.abs(res.fun).max()),
            n_tensions=len(Ts), T_lo=Ts[0], T_hi=Ts[-1])

    print("    The sqrt(T) law more than halves the residual, so it is worth")
    print("    carrying; but note that BOTH variants recover phi_a to about a")
    print(f"    tenth of a per cent.  That {floor:.2f} % is a SYSTEMATIC FLOOR: it is")
    print("    the two-mode idealisation failing to describe a 160 degree of")
    print("    freedom system, and no amount of averaging removes it.  Every")
    print("    random error quoted below should be read as sitting on top of it,")
    print(f"    so a claim of better than about {floor:.2f} % on phi_a (twice that on")
    print("    the modal mass) is not supported whatever the noise level.")
    row(record="model_floor", phi_a_floor_pct=floor, M_host_floor_pct=2 * floor)
    return f_h, phi_a, M_host, s_fe


# ---------------------------------------------------------------------------
# PART A: how precisely can a frequency actually be identified in the field
# ---------------------------------------------------------------------------

def B_f(kappa):
    """Au's data-length factor for the natural frequency, B_f(kappa)."""
    return 2.0 * (np.arctan(kappa) - kappa / (kappa ** 2 + 1.0))


def cov_f(zeta, f, T_d, kappa=6.0):
    """Posterior c.o.v. of an identified natural frequency.

    Au (2014), uncertainty law in ambient modal identification:

        delta_f^2 = zeta / (2 pi N_c B_f(kappa)),   N_c = f T_d

    with N_c the record length in cycles and kappa the half-bandwidth used for
    identification in units of zeta (the +-6 zeta band carries 90 % of the
    response variance).  This is a posterior floor for a well-separated mode
    under a correct model, so field repeatability is worse, not better; Part F
    measures the close-mode inflation that applies near a crossing.
    """
    return float(np.sqrt(zeta / (2.0 * np.pi * f * T_d * B_f(kappa))))


def partA(f0):
    print("\n" + "=" * 78)
    print("PART A   the frequency precision budget, and why 0.1 % is the baseline")
    print("=" * 78)
    print("  Au's uncertainty law, delta_f^2 = zeta / (2 pi f T_d B_f), "
          f"B_f(6) = {B_f(6.0):.4f}")
    print("\n  zeta     T_d=300 s   600 s    1200 s    3600 s")
    for z in (0.002, 0.005, 0.01, 0.02):
        vals = [cov_f(z, f0, td) for td in (300, 600, 1200, 3600)]
        print(f"  {100*z:4.1f} %   " + "  ".join(f"{100*v:7.3f} %" for v in vals))
        for td, v in zip((300, 600, 1200, 3600), vals):
            row(record="freq_precision_law", zeta=z, T_d=td, f=f0, cov_f=v)

    base = cov_f(0.01, f0, 600)
    print(f"\n  A hybrid branch at this crossing carries damping between the")
    print(f"  stay's (0.1-0.5 %) and the deck's (0.5-2 %).  At zeta = 1 % and a")
    print(f"  10 minute record the law gives {100*base:.3f} %.  That is a floor:")
    print(f"  it assumes the model is right, the process stationary and the mode")
    print(f"  well separated, and near a crossing it is not well separated.")
    print(f"  BASELINE ADOPTED: sigma_f/f = 0.1 %, about 1.8x the floor.")
    print(f"  The project's own identification campaign used 0.2 %; every result")
    print(f"  below is swept from 0.02 % to 1 % so the choice is never load-bearing.")
    row(record="freq_precision_baseline", cov_f_law=base, sigma_f_baseline=1e-3,
        inflation_over_law=1e-3 / base)
    return 1e-3


# ---------------------------------------------------------------------------
# PART B: the closed form, and its Fisher derivation
# ---------------------------------------------------------------------------

def fisher_se(Ts, phi_a, f_h, sigma_rel, free=("phi", "fh", "c"),
              n=1, single_branch=False):
    """Standard error of phi_a from the linearised (Fisher) covariance.

    Parameters are a subset of (phi_a, f_h, calibration c, tension offset dT,
    host drift a).  Returns (SE(phi_a), N_eff) where

        N_eff = sum_k g_k^2 / (D_k^2 + g_k^2)

    is the effective number of informative steps -- the group that governs
    everything downstream.  Steps far from the crossing contribute almost
    nothing to it however many of them there are.
    """
    p0 = dict(phi=phi_a, fh=f_h, c=1.0, dT=0.0, drift=0.0)

    def predict(p):
        lo, hi = branches(Ts, p["phi"], p["fh"], p["c"], BRIDGE["EIc"], n,
                          p["drift"], p["dT"])
        return hi if single_branch else np.concatenate([lo, hi])

    names = list(free)
    y0 = predict(p0)
    J = np.zeros((len(y0), len(names)))
    for i, nm in enumerate(names):
        h = {"phi": 1e-8, "fh": 1e-6, "c": 1e-7, "dT": 1.0, "drift": 1e-6}[nm]
        pp = dict(p0); pp[nm] = p0[nm] + h
        pm = dict(p0); pm[nm] = p0[nm] - h
        J[:, i] = (predict(pp) - predict(pm)) / (2 * h)

    sig = sigma_rel * f_h
    try:
        cov = sig ** 2 * np.linalg.inv(J.T @ J)
        se = float(np.sqrt(cov[names.index("phi"), names.index("phi")]))
    except np.linalg.LinAlgError:
        se = np.inf

    D = f_iso(Ts, n=n) - f_h
    gg = g_of_T(phi_a, Ts)
    n_eff = float(np.sum(gg ** 2 / (D ** 2 + gg ** 2)))
    return se, n_eff


def partB(f_h, phi_a, s_true, sigma_f):
    print("\n" + "=" * 78)
    print("PART B   the closed form for the precision of phi_a")
    print("=" * 78)
    print("""
  Track both branches over a sweep.  The measured gap is sqrt(D^2 + g^2) with
  D the detuning and g the coupling, so

      d(gap)/dg = g / sqrt(D^2+g^2),     d(gap)/dD = D / sqrt(D^2+g^2)

  and those two squared SUM TO ONE.  The sensitivity to the thing wanted and the
  sensitivity to the thing that drifts are exactly complementary: where the gap
  carries information about g it is first-order blind to the detuning, and where
  it responds to the detuning it says nothing about g.  Summing the first over a
  sweep gives the effective step count

      N_eff = sum_k  1 / (1 + (D_k/g_k)^2)

  and, with independent noise sigma_f f on each of the two branch readings,

      SE(phi_a)/phi_a = SE(s)/s = sqrt(2) (sigma_f/f0) / (s sqrt(N_eff))
      SE(M_host)/M_host = 2 x SE(phi_a)/phi_a          (mass goes as phi_a^-2)

  N_eff counts only the steps that land INSIDE the avoided crossing.  A step at
  |D| = 3g contributes 0.1; at |D| = 10g, 0.01.  Extending the sweep further
  either side buys nuisance-parameter leverage, not precision on phi_a.""")

    dT_c = 2.0 * s_true * T_TUNE
    print(f"\n  crossing width in tension: |D| < g for |T - T*| < 2 s T = "
          f"{dT_c/1e3:.2f} kN out of {T_TUNE/1e3:.0f} kN")
    row(record="crossing_width", dT_c_N=dT_c, dT_c_frac=dT_c / T_TUNE,
        s=s_true)

    print("\n  closed form vs Fisher vs Monte Carlo (sigma_f/f = 0.1 %):")
    print("  steps  half-width   N_eff   closed form   Fisher(3 par)   MC(3 par)")
    for nst, w in ((1, 0.0), (3, 1.0), (5, 1.0), (9, 2.0), (17, 2.0),
                   (17, 8.0), (33, 2.0)):
        Ts = (np.array([T_TUNE]) if nst == 1 else
              np.linspace(T_TUNE - w * dT_c, T_TUNE + w * dT_c, nst))
        se_or, n_eff = fisher_se(Ts, phi_a, f_h, sigma_f, free=("phi",))
        cf = np.sqrt(2) * sigma_f / (s_true * np.sqrt(n_eff))
        if nst >= 3:                       # 3 parameters need >= 2 tension steps
            se_3, _ = fisher_se(Ts, phi_a, f_h, sigma_f, free=("phi", "fh", "c"))
            c3 = f"{100*se_3/phi_a:10.2f} %"
        else:
            se_3, c3 = np.nan, "  degenerate"
        mc = mc_se(Ts, phi_a, f_h, sigma_f, nrep=400) if nst >= 5 else np.nan
        print(f"   {nst:3d}    {w:4.1f} dT_c  {n_eff:6.2f}   "
              f"{100*cf:8.2f} %   {c3}   "
              f"{'  n/a  ' if np.isnan(mc) else f'{100*mc/phi_a:7.2f} %'}")
        row(record="closed_form_check", n_steps=nst, half_width_dTc=w,
            N_eff=n_eff, se_closed_form_rel=cf, se_fisher_1par=se_or / phi_a,
            se_fisher_3par=(np.nan if np.isnan(se_3) else se_3 / phi_a),
            se_mc_3par=(np.nan if np.isnan(mc) else mc / phi_a))
    print("""
  Two things to read off this table.  First, Monte Carlo on the FE truth tracks
  the Fisher prediction, so the linearisation is safe and the closed form can be
  used for design.  Second, and less obvious: columns 4 and 5 barely differ.
  Fitting the host frequency and the stay's calibration alongside phi_a costs
  almost nothing, because those two act on the branch MIDPOINT while phi_a acts
  on the branch GAP, and by the complementarity above the gap is first-order
  blind to the detuning exactly where it carries the coupling.  A single hold is
  degenerate for three parameters -- with one tension there are two numbers and
  nothing to separate them -- which is the arithmetic reason a sweep is needed
  at all, quite apart from the variance.""")


# ---------------------------------------------------------------------------
# Monte Carlo on the FE truth
# ---------------------------------------------------------------------------

_SPL = {}


def fe_splines(n=1, lo=105e3, hi=205e3, npts=161):
    """Cubic splines through the FE branch frequencies, built once.

    The Monte Carlo draws its truth from the finite element model, not from the
    fitted hyperbola, so the recovered bias includes whatever the two-mode
    idealisation gets wrong.
    """
    if n in _SPL:
        return _SPL[n]
    Ts = np.linspace(lo, hi, npts)
    pr = np.array([fe_pair(T) for T in Ts])
    _SPL[n] = (CubicSpline(Ts, pr[:, 0]), CubicSpline(Ts, pr[:, 1]))
    return _SPL[n]


def mc_se(Ts, phi_a, f_h, sigma_rel, nrep=400, free=("phi", "fh", "c"),
          rng=None, drift_true=0.0, jitter_rel=0.0, single_branch=False,
          noise_ratio_hi=1.0, return_all=False):
    """Empirical SE of phi_a from repeated noisy sweeps fitted end to end.

    ``jitter_rel`` adds an unmodelled per-step wander of the detuning (the host
    mode moving between records because other stays are being tensioned, or the
    temperature changing), expressed as a relative frequency shift of the host.
    ``noise_ratio_hi`` inflates the noise on the host-dominated branch, which a
    stay-mounted accelerometer sees with a smaller residue.
    """
    rng = RNG if rng is None else rng
    sl, sh = fe_splines()
    lo_t, hi_t = sl(Ts), sh(Ts)
    sig = sigma_rel * f_h
    out = []
    for _ in range(nrep):
        dl, dh = lo_t.copy(), hi_t.copy()
        if drift_true or jitter_rel:
            # move the host frequency, and with it both branches, without
            # telling the fit: a systematic ramp plus a per-step wander
            shift = f_h * (drift_true * (Ts - T_TUNE) / T_TUNE
                           + jitter_rel * rng.standard_normal(len(Ts)))
            dl, dh = dl + shift, dh + shift
        obs_lo = dl + sig * rng.standard_normal(len(Ts))
        obs_hi = dh + sig * noise_ratio_hi * rng.standard_normal(len(Ts))

        def resid(p):
            q = dict(zip(free, p))
            lo, hi = branches(Ts, q.get("phi", phi_a), q.get("fh", f_h),
                              q.get("c", 1.0), BRIDGE["EIc"], 1,
                              q.get("drift", 0.0), q.get("dT", 0.0))
            if single_branch:
                return hi - obs_hi
            return np.concatenate([(lo - obs_lo),
                                   (hi - obs_hi) / noise_ratio_hi])

        x0 = [{"phi": phi_a, "fh": f_h, "c": 1.0, "dT": 0.0,
               "drift": 0.0}[k] for k in free]
        try:
            r = least_squares(resid, x0, method="lm", max_nfev=400)
            out.append(r.x[list(free).index("phi")])
        except Exception:
            continue
    out = np.array(out)
    out = out[np.isfinite(out) & (out > 0)]
    if return_all:
        return out
    return float(np.std(out, ddof=1)) if len(out) > 8 else np.nan


# ---------------------------------------------------------------------------
# PART C: the design grid
# ---------------------------------------------------------------------------

def partC(f_h, phi_a, M_host, s_true, sigma_base):
    print("\n" + "=" * 78)
    print("PART C   confidence intervals against sweep design")
    print("=" * 78)
    dT_c = 2.0 * s_true * T_TUNE

    print("\n  (i) against frequency precision and step count, half-width 2 dT_c")
    print("      cells are SE(phi_a)/phi_a  /  SE(M_host)/M_host, both 1 sigma")
    print("      sigma_f    5 steps        9 steps       17 steps       33 steps")
    for sf in (2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2):
        cells = []
        for nst in (5, 9, 17, 33):
            Ts = np.linspace(T_TUNE - 2 * dT_c, T_TUNE + 2 * dT_c, nst)
            se, n_eff = fisher_se(Ts, phi_a, f_h, sf)
            cells.append(f"{100*se/phi_a:5.1f}/{200*se/phi_a:5.1f} %")
            row(record="grid_sigma_steps", sigma_f=sf, n_steps=nst,
                half_width_dTc=2.0, N_eff=n_eff, se_phi_rel=se / phi_a,
                se_Mhost_rel=2 * se / phi_a,
                ci95_Mhost_lo=M_host * (1 - 2 * 1.96 * se / phi_a),
                ci95_Mhost_hi=M_host * (1 + 2 * 1.96 * se / phi_a))
        print(f"      {100*sf:5.2f} %  " + "  ".join(cells))

    print("\n  (ii) against how far the sweep extends either side of the crossing")
    print("       17 steps spread over the stated half-width, sigma_f = 0.1 %")
    print("       half-width   N_eff   SE(phi_a)   SE(M_host)   steps with |D|<g")
    for w in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        Ts = np.linspace(T_TUNE - w * dT_c, T_TUNE + w * dT_c, 17)
        se, n_eff = fisher_se(Ts, phi_a, f_h, sigma_base)
        inside = int(np.sum(np.abs(f_iso(Ts) - f_h) < g_of_T(phi_a, Ts)))
        print(f"       {w:5.1f} dT_c  {n_eff:6.2f}   {100*se/phi_a:7.2f} %   "
              f"{200*se/phi_a:8.2f} %      {inside:3d} of 17")
        row(record="grid_extent", half_width_dTc=w, n_steps=17,
            sigma_f=sigma_base, N_eff=n_eff, se_phi_rel=se / phi_a,
            se_Mhost_rel=2 * se / phi_a, steps_inside=inside)
    print("       Widening the sweep at fixed step count makes phi_a WORSE.")
    print("       The information is inside the crossing; the wings only pin")
    print("       down the nuisance parameters, and there is a trade-off.")

    print("\n  (iii) the two operating points that matter")
    schedules = {
        "as-built stressing schedule (8 holds, 0.4 to 1.25 T, no refinement)":
            np.linspace(0.40 * T_TUNE, 1.25 * T_TUNE, 8),
        "same, 12 holds": np.linspace(0.40 * T_TUNE, 1.25 * T_TUNE, 12),
        "designed experiment: 15 holds inside +-2 dT_c, 4 outside":
            np.concatenate([np.linspace(T_TUNE - 2 * dT_c, T_TUNE + 2 * dT_c, 15),
                            [T_TUNE - 8 * dT_c, T_TUNE - 5 * dT_c,
                             T_TUNE + 5 * dT_c, T_TUNE + 8 * dT_c]]),
    }
    for name, Ts in schedules.items():
        Ts = np.sort(Ts[(Ts > 105e3) & (Ts < 205e3)])
        if len(Ts) < 4:
            continue
        se, n_eff = fisher_se(Ts, phi_a, f_h, sigma_base)
        mc = mc_se(Ts, phi_a, f_h, sigma_base, nrep=600)
        print(f"    {name}")
        print(f"      {len(Ts)} usable holds, N_eff = {n_eff:.2f}")
        print(f"      SE(phi_a)   = {100*se/phi_a:6.2f} %  (Monte Carlo "
              f"{100*mc/phi_a:6.2f} %)")
        print(f"      SE(M_host)  = {200*se/phi_a:6.2f} %   ->  "
              f"{M_host/1e3:.1f} t  +- {2*1.96*se/phi_a*M_host/1e3:.1f} t (95 %)")
        row(record="schedule", name=name, n_holds=len(Ts), N_eff=n_eff,
            sigma_f=sigma_base, se_phi_rel=se / phi_a, se_phi_rel_mc=mc / phi_a,
            se_Mhost_rel=2 * se / phi_a, M_host=M_host,
            ci95_halfwidth_t=2 * 1.96 * se / phi_a * M_host / 1e3)
    print("\n    A NORMAL stressing schedule puts 0 or 1 hold inside the crossing.")
    print("    The precision is a design choice, not a free by-product: the")
    print("    difference between the two rows above is where the holds are put.")


# ---------------------------------------------------------------------------
# PART D: what the nuisance parameters cost
# ---------------------------------------------------------------------------

def partD(f_h, phi_a, s_true, sigma_base):
    print("\n" + "=" * 78)
    print("PART D   the cost of not knowing the stay's own f(T) law")
    print("=" * 78)
    dT_c = 2.0 * s_true * T_TUNE
    Ts = np.linspace(T_TUNE - 2 * dT_c, T_TUNE + 2 * dT_c, 17)
    Tw = np.concatenate([np.linspace(T_TUNE - 2 * dT_c, T_TUNE + 2 * dT_c, 15),
                         [T_TUNE - 8 * dT_c, T_TUNE + 8 * dT_c]])
    Tw = np.sort(Tw)
    variants = [
        (("phi",), "phi_a alone (oracle: f_h, mass, length, jack all exact)"),
        (("phi", "fh"), "+ host frequency"),
        (("phi", "fh", "c"), "+ scale calibration on f_iso(T)  [BASELINE]"),
        (("phi", "fh", "c", "dT"), "+ additive tension offset (anchor set)"),
        (("phi", "fh", "c", "drift"), "+ linear host drift with tension"),
        (("phi", "fh", "c", "dT", "drift"), "+ both"),
    ]
    base = None
    print("      free parameters                                    "
          "narrow sweep   with wings")
    for free, label in variants:
        se_n, _ = fisher_se(Ts, phi_a, f_h, sigma_base, free=free)
        se_w, _ = fisher_se(Tw, phi_a, f_h, sigma_base, free=free)
        if base is None:
            base = se_n
        print(f"    {label:52s} {100*se_n/phi_a:6.2f} %   {100*se_w/phi_a:6.2f} %")
        row(record="nuisance", free="+".join(free), label=label,
            se_phi_rel_narrow=se_n / phi_a, se_phi_rel_wings=se_w / phi_a,
            inflation_vs_oracle=se_n / base, sigma_f=sigma_base)
    print("""
    This is the most encouraging table in the study, and it was not the expected
    result.  Not knowing the stay's own f(T) law costs essentially NOTHING: the
    host frequency and a scale calibration are free, and the two harder nuisances
    together inflate the standard error by under a third on a narrow sweep and by
    a tenth once a couple of holds are taken in the wings.  The reason is the
    complementarity of Part B.  Every one of these nuisances moves where the
    branches sit; phi_a sets how far apart they are.  The measurement is
    differential, so the mass per length, the effective length, the jack
    calibration and the anchor set -- the recognised error sources that dog the
    incumbent tension inversion -- largely do not propagate into phi_a at all.
    A scale error on the jack is exactly absorbed; only an additive offset
    survives, and only weakly, because over a narrow tension range an offset and
    a scale are nearly collinear.  Holds in the wings cost N_eff but restore that
    separation, which is the one reason to take them.""")


# ---------------------------------------------------------------------------
# PART E: back-check against the Ponte del Mare traverse
# ---------------------------------------------------------------------------

def partE():
    print("\n" + "=" * 78)
    print("PART E   does the law reproduce the project's own field bracket?")
    print("=" * 78)
    print("""
  The damper traverse on the Ponte del Mare moved a global mode from 0.750 to
  0.930 Hz across stay N8E's fundamental at ~0.83 Hz, at constant stay tension.
  That is a two-point sweep, and both points sit deep in the WINGS: the stay's
  line was never inside the crossing.  If the law of Part B is any use it should
  put that traverse in the right order of magnitude from its geometry alone, and
  attribute the width of the resulting bracket to the geometry rather than to
  the idea.  Order of magnitude is all that can be asked: what follows compares
  a Gaussian standard error against the min-max envelope of a few pixel reads.""")
    f_iso_pdm = 0.8288                      # solved isolated line, data/traverse.csv
    s_pdm = 0.030                           # midpoint of the measured 1.6-5.2 %
    g_pdm = s_pdm * f_iso_pdm
    D = np.array([f_iso_pdm - 0.750, f_iso_pdm - 0.930])
    sigma_read = 0.005                      # pixel reads span ~0.5 % of 0.83 Hz
    # single line tracked (the stay's), so the sensitivity is half the pair's
    n_eff = float(np.sum(1.0 / (1.0 + (D / g_pdm) ** 2)))
    rel = 2.0 * sigma_read / (s_pdm * np.sqrt(n_eff))
    obs_ratio = 5.2 / 1.6
    obs_log = np.log(obs_ratio) / 2.0        # half-width of the envelope in ln s
    print(f"  |D|/g at the two damper states: {abs(D[0])/g_pdm:.2f} and "
          f"{abs(D[1])/g_pdm:.2f}   ->  N_eff = {n_eff:.3f}")
    print(f"  predicted SE(s)/s = 2 (sigma/f) / (s sqrt(N_eff)) = {100*rel:.0f} %")
    print(f"  observed envelope s = 1.6 to 5.2 %, a factor {obs_ratio:.2f}, i.e.")
    print(f"  +-{100*obs_log:.0f} % about its geometric centre.")
    print(f"\n  These agree to a factor {rel/obs_log:.1f}, and that is as close as the")
    print(f"  comparison can honestly be made: the predicted number is a Gaussian")
    print(f"  1-sigma, the observed one is the min-max envelope of a handful of")
    print(f"  pixel reads, and at an error this large the linearisation behind the")
    print(f"  prediction is itself past its range of validity.  The claim is only")
    print(f"  that the law puts that traverse in the right order of magnitude --")
    print(f"  tens of per cent, not per cent -- and it does.")
    print(f"\n  What carries is the ratio, which needs no such matching.  That")
    print(f"  traverse had N_eff = {n_eff:.2f}; a designed sweep reaches N_eff ~ 8-9.")
    print(f"  That alone is a factor {np.sqrt(8.5/n_eff):.0f} in standard error, before any")
    print(f"  improvement in the frequency reads over pixel measurement of a")
    print(f"  printed figure.  The factor of ten on the modal mass is therefore")
    print(f"  NOT evidence against the reversal: it is what a two-point traverse")
    print(f"  at |D|/g ~ 3-5 is arithmetically obliged to return.  Both damper")
    print(f"  states sat in the WINGS, where the branch displacement is g^2/(4|D|)")
    print(f"  and so QUADRATIC in the thing being measured -- a 30 % error on the")
    print(f"  displacement is a 15 % error on g only because it is halved by the")
    print(f"  square root, and the displacement error there was of order 100 %.")
    row(record="pontedelmare_backcheck", f_iso=f_iso_pdm, s_assumed=s_pdm,
        sigma_read=sigma_read, D1=D[0], D2=D[1], N_eff=n_eff,
        se_s_rel_pred=rel, obs_envelope_ratio=obs_ratio,
        obs_log_halfwidth=obs_log, pred_over_obs=rel / obs_log,
        N_eff_designed_sweep=8.5, se_ratio_designed=np.sqrt(8.5 / n_eff),
        bracket_Mhost_obs=959.0 / 91.0)


# ---------------------------------------------------------------------------
# PART F: refinement (a) -- the pair never separately resolved
# ---------------------------------------------------------------------------

def partF(f_h, phi_a, s_true, sigma_base):
    print("\n" + "=" * 78)
    print("PART F   refinement (a): does it work if the pair is never resolved?")
    print("=" * 78)
    print("""
  Claim under test: because s comes from a fit over many tensions rather than
  from one resolved doublet, the damping limit that stops the pair being seen as
  two peaks need not stop the measurement.

  There is an exact result against it.  At a stay sensor the two hybrid modes
  carry residues equal to their squared amplitudes there, and the residue-
  weighted mean of the pair's frequencies is the DIAGONAL of the pencil, i.e.
  exactly f_iso, the unperturbed stay frequency.  A perfectly merged pair whose
  single peak sits at the residue-weighted mean therefore reports the stay
  frequency the isolated formula would have predicted, and carries NO
  information about the coupling at leading order.  Verified numerically below,
  then tested against the real damped FRF.""")

    # ---- the centroid identity, on the FE model itself --------------------
    dT_c = 2.0 * s_true * T_TUNE
    print("\n  (i) the residue-weighted mean of the hybrid pair, FE model")
    Tg = np.linspace(T_TUNE - 3 * dT_c, T_TUNE + 3 * dT_c, 13)
    cen, brn, fis, gaps = [], [], [], []
    for T in Tg:
        cd = CableDeck(T=T, **BRIDGE)
        f, Phi = cd.modes(60)
        p = cd.cable_sensor_dof(2.0)
        sel = (f > 2.3) & (f < 4.5)
        w = Phi[p, sel] ** 2
        fs = f[sel]
        cen.append(float(np.sum(w * fs) / np.sum(w)))
        brn.append(float(fs[int(np.argmax(w))]))    # stay-dominated branch
        fis.append(f_iso(T))
        gaps.append(float(fs.max() - fs.min()))
    cen, brn, fis, gaps = map(np.array, (cen, brn, fis, gaps))

    # A merged-peak tracker can only fit a smooth stay law to what it sees.
    # Whatever residual survives that fit IS the whole available signal.
    def smooth_resid(y):
        r = least_squares(lambda p: p[0] * tensioned_beam_freq(
            1, L_C, Tg, max(p[1], 1.0), BRIDGE["mc"]) - y, [1.0, BRIDGE["EIc"]])
        return r.fun

    rc, rb = smooth_resid(cen), smooth_resid(brn)
    print("      T [kN]   f_iso     centroid   stay branch   (cen-f_iso)/gap")
    for i in (0, 6, 12):
        print(f"      {Tg[i]/1e3:6.1f}  {fis[i]:8.5f}  {cen[i]:8.5f}   "
              f"{brn[i]:9.5f}    {(cen[i]-fis[i])/gaps[i]:+8.3f}")
    print(f"\n      after fitting a smooth stay law (scale + EI) and removing it:")
    print(f"        residual left in the CENTROID     "
          f"{1e3*np.sqrt(np.mean(rc**2)):7.3f} mHz "
          f"= {100*np.sqrt(np.mean(rc**2))/gaps.mean():5.2f} % of the gap")
    print(f"        residual left in the STAY BRANCH  "
          f"{1e3*np.sqrt(np.mean(rb**2)):7.3f} mHz "
          f"= {100*np.sqrt(np.mean(rb**2))/gaps.mean():5.2f} % of the gap")
    print(f"      information loss on merging: a factor "
          f"{np.sqrt(np.mean(rb**2))/np.sqrt(np.mean(rc**2)):.0f} in signal.")
    print("      The centroid does not sit exactly on f_iso -- modes outside the")
    print("      pair contribute to the sum rule -- but it departs from it by a")
    print("      SMOOTH offset that a calibration parameter absorbs, and what is")
    print("      left is a small fraction of what the resolved branch carries.")
    row(record="centroid_identity", resid_centroid_hz=float(np.sqrt(np.mean(rc ** 2))),
        resid_branch_hz=float(np.sqrt(np.mean(rb ** 2))),
        mean_gap_hz=float(gaps.mean()),
        signal_loss_factor=float(np.sqrt(np.mean(rb ** 2))
                                 / np.sqrt(np.mean(rc ** 2))),
        n_tensions=len(Tg))

    # ---- can a stay sensor even SEE both branches? ------------------------
    print("\n  (ii) observability: can a stay sensor see BOTH branches?")
    print("       Part H's design rule will be to read the pair from one record,")
    print("       which needs the host-dominated branch to be visible AT THE STAY.")
    print("       A two-mode pencil says its stay content should collapse away")
    print("       from the crossing, as (1-|D|/sqrt(D^2+g^2))/2.  The model says")
    print("       otherwise, and the reason is the tie:")
    print("       |D|/g    weaker branch, relative to stronger, at the stay sensor")
    for k in (0.0, 1.0, 2.0, 3.0, 5.0, 8.0):
        Tk = T_TUNE + k * dT_c
        cd = CableDeck(T=Tk, **BRIDGE)
        f, Phi = cd.modes(60)
        p = cd.cable_sensor_dof(2.0)
        sel = (f > 2.3) & (f < 4.5)
        a = np.sort(Phi[p, sel] ** 2)
        ratio = float(a[0] / a[-1])
        print(f"       {k:5.1f}    {ratio:10.4f}   ({10*np.log10(max(ratio,1e-12)):+6.1f} dB)")
        row(record="observability", D_over_g=k, T=Tk, residue_ratio=ratio,
            residue_ratio_dB=10 * np.log10(max(ratio, 1e-12)))
    print("       The ratio does not collapse -- it saturates near -13 dB.  The")
    print("       2x2 prediction at |D| = 8g is -24 dB, so the pencil is wrong")
    print("       here, and wrong in the helpful direction.  The stay's lower end")
    print("       is TIED to the deck, v_stay = cos(theta) w_deck, so a")
    print("       deck-dominated mode drags the stay kinematically whatever the")
    print("       detuning; the sensor 2 m up the chord sees that entrainment")
    print("       even with no resonant participation at all.  That is the same")
    print("       tie this project added to the model precisely because the")
    print("       isolated-cable formulations omit it.")
    print("       CONSEQUENCE: a stay-mounted accelerometer alone does suffice to")
    print("       read both branches across the whole practical sweep, and the")
    print("       instrumentation premise survives.  It is the one obstacle")
    print("       expected here that did not materialise.")

    # ---- what the damped FRF actually shows -------------------------------
    print("\n  (iii) the driven FRF at the stay sensor across the sweep")
    print("       zeta    r = s/2zeta   peaks at exact tuning   resolvable at any T?")
    Ts_frf = np.linspace(T_TUNE - 3 * dT_c, T_TUNE + 3 * dT_c, 7)
    fgrid = np.linspace(FBAND[0], FBAND[1], 12001)
    for z in (0.002, 0.005, 0.010, 0.020, 0.040):
        anyres = False
        npk_tune = None
        for T in Ts_frf:
            br = DrivenBridge(T)
            c = peak_census(fgrid, br.frf(z, fgrid))
            if abs(T - T_TUNE) < 1.0:
                npk_tune = c["n_peaks_3db"]
            if c["n_peaks_3db"] >= 2:
                anyres = True
        r = s_true / (2 * z)
        print(f"       {100*z:4.1f} %  {r:9.2f}      {npk_tune:5d}              "
              f"{'yes' if anyres else 'NO -- merged throughout'}")
        row(record="frf_resolvability", zeta=z, r=r, n_peaks_at_tuning=npk_tune,
            resolvable_anywhere=int(anyres))

    # ---- can a two-mode curve fit beat the peak-picking wall? -------------
    print("\n  (iv) a two-mode curve fit of the noisy FRF, below the peak wall")
    print("        Rayleigh's criterion is not an information limit: a fit uses")
    print("        the whole complex response, not the location of a maximum.")
    print("        Empirical SE of the measured gap, 120 noise realisations,")
    print("        additive complex noise at 40 dB below the band peak.")
    print("        The spread is reported as a robust scale (1.4826 x median")
    print("        absolute deviation) and with the fraction of fits that land")
    print("        within +-50 % of the true gap, because once the pair merges")
    print("        the six-parameter fit stops being identifiable and returns")
    print("        garbage -- a sample standard deviation over failed fits is a")
    print("        number, not a measurement.")
    print("        zeta      r    robust SE(gap)/gap   usable fits   inflation")
    br0 = DrivenBridge(T_TUNE)
    fg = np.linspace(3.15, 3.50, 700)
    gtrue = br0.f_hi - br0.f_lo
    ref = None
    for z in (0.002, 0.005, 0.010, 0.020, 0.040):
        H0 = br0.frf(z, fg)
        noise = 10 ** (-40 / 20.0) * np.abs(H0).max()
        gaps = []
        for _ in range(120):
            H = H0 + noise * (RNG.standard_normal(len(fg))
                              + 1j * RNG.standard_normal(len(fg))) / np.sqrt(2)

            def resid(p):
                f1, f2, z1, z2, r1, r2 = p
                w = 2 * np.pi * fg
                w1, w2 = 2 * np.pi * f1, 2 * np.pi * f2
                Hm = (r1 / (w1 ** 2 - w ** 2 + 2j * z1 * w1 * w)
                      + r2 / (w2 ** 2 - w ** 2 + 2j * z2 * w2 * w))
                d = Hm - H
                return np.concatenate([d.real, d.imag]) / noise

            amp0 = float(np.max(np.abs(H0)) * (2 * np.pi * br0.f_lo) ** 2 * 2 * z)
            p0 = [br0.f_lo, br0.f_hi, z, z, amp0, amp0]
            try:
                r_ = least_squares(resid, p0, method="lm", max_nfev=800)
                gaps.append(abs(r_.x[1] - r_.x[0]))
            except Exception:
                pass
        gaps = np.array(gaps)
        gaps = gaps[np.isfinite(gaps)]
        usable = float(np.mean(np.abs(gaps / gtrue - 1) < 0.5)) if len(gaps) else 0.0
        mad = float(np.median(np.abs(gaps - np.median(gaps)))) * 1.4826
        se = mad / gtrue if len(gaps) > 8 else np.nan
        if ref is None:
            ref = se
        rr = s_true / (2 * z)
        if usable > 0.9:
            print(f"        {100*z:4.1f} % {rr:6.2f}      {100*se:8.3f} %        "
                  f"{100*usable:5.0f} %     {se/ref:7.1f} x")
        else:
            print(f"        {100*z:4.1f} % {rr:6.2f}         --                "
                  f"{100*usable:5.0f} %        --      <- unidentifiable")
        row(record="frf_curvefit", zeta=z, r=rr, se_gap_rel_robust=se,
            frac_usable=usable, inflation=se / ref, n_rep=len(gaps))

    print("""
    VERDICT ON REFINEMENT (a).  It does not hold in the form claimed.

    Peak-picking a merged pair is worthless, and the reason is exact rather
    than circumstantial: the residue-weighted mean of the pair AT A STAY SENSOR
    is the diagonal of the pencil, which is the isolated-cable frequency.  A
    merged peak therefore reports what the incumbent formula would have
    predicted anyway.  Measured above, after allowing a smooth calibration to
    be fitted out, the centroid retains three orders of magnitude less coupling
    signature than the resolved branch does.  Sweeping tension does not repair
    this, because the deficiency is at every tension.

    A two-mode CURVE FIT does better, and Rayleigh's criterion is genuinely not
    an information limit -- the fit uses the whole complex response rather than
    the location of a maximum, and it still works at r = 1.17 where the 3 dB
    peak test already sees one peak.  But it extends the usable range by
    roughly one factor of two in r, not indefinitely: by r ~ 0.6 the six
    parameters are no longer identifiable and the fit returns numbers unrelated
    to the truth.  The correct statement of the requirement is therefore
    s > about 1.2 zeta on the HYBRID branches -- a little below the project's
    own 3 dB peak-resolvability threshold of s > 2.28 zeta, and nothing like an
    exemption from it.  Damping stays a first-order limitation on the method,
    and since the hybrid branches carry a blend of the stay's low damping and
    the deck's higher damping, it is the deck's damping that decides.""")


# ---------------------------------------------------------------------------
# PART G: refinement (b) -- several stay mode orders
# ---------------------------------------------------------------------------

def partG(f_h, phi_a, sigma_base):
    print("\n" + "=" * 78)
    print("PART G   refinement (b): does the 1/n redundancy improve phi_a?")
    print("=" * 78)
    print("""
  Stay mode n meets the same host mode where f_n(T) = f_h, and f_n ~ n sqrt(T),
  so T_n ~ T_1 / n^2.  A single monotone stressing operation from slack to full
  load therefore passes EVERY order's crossing with that host mode, in the order
  n = 5, 4, 3, 2, 1.  The redundancy is real and it is free.  Whether it
  improves the ESTIMATE is a different question.""")

    EA = BRIDGE["EA"]
    A_stay = EA / 2.0e11
    print("\n    n   T_n [kN]   s_n closed   s_n from FE   s_n x n   "
          "lambda^2   sag/L   EI share   SE(phi_a)")
    ests = []
    for n in range(1, 6):
        try:
            Tn = brentq(lambda T: f_iso(T, n=n) - f_h, 1.0, 400e3)
        except ValueError:
            continue
        mu = mu_effective(M_S, phi_a)
        s_cf = veering_split(mu, THETA, n)
        # what fraction of the stay's restoring force is bending, not tension?
        EIk2 = BRIDGE["EIc"] * (n * np.pi / L_C) ** 2
        ei_share = EIk2 / (Tn + EIk2)
        # the 1/n law is the load-bearing claim; check it against the FE
        s_fe_n = fe_split_at(Tn, f_h)
        lam2 = irvine_lambda2(L_C, Tn, EA, BRIDGE["mc"], THETA)
        sag = sag_ratio(L_C, Tn, BRIDGE["mc"], THETA)
        # one resolved doublet at that crossing, same relative frequency noise
        se_rel = np.sqrt(2) * sigma_base / s_cf
        ests.append((n, Tn, s_cf, se_rel, lam2, sag, ei_share))
        print(f"    {n}   {Tn/1e3:7.2f}   {100*s_cf:9.4f} %  "
              f"{'      n/a  ' if np.isnan(s_fe_n) else f'{100*s_fe_n:9.4f} %'}  "
              f"{100*s_cf*n:7.4f}   {lam2:9.2f}  {sag:7.5f}  {100*ei_share:6.1f} %"
              f"   {100*se_rel:6.2f} %")
        row(record="mode_order", n=n, T_n=Tn, s_closed=s_cf, s_fe=s_fe_n,
            s_times_n=s_cf * n, lambda2=lam2, sag_ratio=sag,
            ei_share=ei_share, se_phi_rel_single=se_rel,
            stay_stress_MPa=Tn / A_stay / 1e6)

    # inverse-variance combination
    ns = np.array([e[0] for e in ests])
    se = np.array([e[3] for e in ests])
    lam2 = np.array([e[4] for e in ests])
    eis = np.array([e[6] for e in ests])
    w = 1.0 / se ** 2
    comb_all = float(1.0 / np.sqrt(np.sum(w)))
    ok = (lam2 < 1.0) & (eis < 0.10)
    comb_ok = float(1.0 / np.sqrt(np.sum(w[ok]))) if ok.any() else np.nan
    print(f"\n    single best order (n=1)              SE(phi_a) = {100*se[0]:6.2f} %")
    print(f"    all {len(ns)} orders, inverse variance      SE(phi_a) = "
          f"{100*comb_all:6.2f} %   ({se[0]/comb_all:.3f} x better)")
    print(f"    only orders still a taut cable       SE(phi_a) = "
          f"{100*comb_ok:6.2f} %   ({se[0]/comb_ok:.3f} x better)")
    print(f"       ({int(ok.sum())} of {len(ns)} orders pass lambda^2 < 1 and bending"
          f" share < 10 %)")
    print(f"    the ceiling, infinitely many orders: sum 1/n^2 -> pi^2/6, so the")
    print(f"    best possible gain is 1/sqrt(pi^2/6) = "
          f"{1/np.sqrt(np.pi**2/6):.3f}, i.e. {100*(1-1/np.sqrt(np.pi**2/6)):.0f} %.")
    print("\n    how taut must a stay be for order n to be usable at all?")
    print("    lambda^2 grows as roughly n^6 along the crossing sequence, so:")
    for i, n_ in enumerate(ns):
        if n_ == 1:
            continue
        need = lam2[0] / lam2[i]
        print(f"      order n = {n_}: needs lambda^2 < {need:.2e} at the n = 1"
              f" crossing   (this stay has {lam2[0]:.3f})")
        row(record="mode_order_criterion", n=int(n_),
            lambda2_1_required=need, lambda2_1_actual=lam2[0],
            usable=int(lam2[0] < need))
    row(record="mode_order_combined", se_n1=se[0], se_all=comb_all,
        se_lambda_ok=comb_ok, gain_all=se[0] / comb_all,
        gain_ok=se[0] / comb_ok, ceiling_gain=np.sqrt(np.pi ** 2 / 6),
        n_orders=len(ns), n_orders_lambda_ok=int(ok.sum()))
    print("""
    VERDICT ON REFINEMENT (b).  The 1/n law itself is confirmed to four figures
    against the FE at every order, which is worth having.  As a precision lever
    the refinement fails twice over.

    First, arithmetically.  s_n falls as 1/n while the frequency noise does not,
    so order n carries inverse-variance weight 1/n^2 and the sum converges: no
    number of extra orders can do better than a 22 % reduction in standard
    error, and five orders deliver 21 % of that 22 %.

    Second, and decisively, the extra orders are not available.  T_n goes as
    1/n^2, so H goes as 1/n^2 and the Irvine parameter, which goes as 1/H^3,
    grows as ROUGHLY n^6.  One order up multiplies lambda^2 by about sixty-four.
    The general criterion that follows is sharp: the n = 2 crossing is only
    usable if the stay has lambda^2 < about 0.015 at its n = 1 crossing, and
    n = 3 needs 3e-4.  This stay has 0.089, so of five crossings exactly one is
    on a taut cable -- by n = 3 the sag parameter is 89, and by n = 4 bending
    supplies a third of the restoring force and the object crossing the deck
    mode is a beam, not a cable.  A stressing operation does pass through all
    five crossings, but the straight-chord inversion the whole method rests on
    has stopped being true at four of them.

    What the redundancy is actually worth is FALSIFICATION.  Where two orders
    are both usable they must return the same phi_a and their splits must stand
    in the ratio 2:1.  That is the only internal check the method has against
    the third-mode contamination of H5 below, which otherwise biases phi_a
    silently, and it is worth designing for on that ground alone.""")


# ---------------------------------------------------------------------------
# PART H: the practical obstacles
# ---------------------------------------------------------------------------

def partH(f_h, phi_a, M_host, s_true, sigma_base):
    print("\n" + "=" * 78)
    print("PART H   the obstacles, quantified")
    print("=" * 78)
    dT_c = 2.0 * s_true * T_TUNE
    Ts = np.linspace(T_TUNE - 2 * dT_c, T_TUNE + 2 * dT_c, 17)

    # H1 -- the deck is compressed by the stay being stressed --------------
    print("\n  H1  the host mode does not hold still while the stay is stressed")
    P_euler = np.pi ** 2 * BRIDGE["EId"] / BRIDGE["Ld"] ** 2
    for span, tag in (((0.40 * T_TUNE, 1.25 * T_TUNE), "full stressing sweep"),
                      ((T_TUNE - 2 * dT_c, T_TUNE + 2 * dT_c), "crossing window only")):
        dP = (span[1] - span[0]) * COS_T
        d_f = -0.5 * dP / P_euler
        print(f"      {tag:22s}: axial thrust change {dP/1e3:6.1f} kN, "
              f"{100*dP/P_euler:5.2f} % of Euler load")
        print(f"      {'':22s}  -> host frequency moves {100*d_f:+.3f} % "
              f"({abs(d_f)/s_true*100:.0f} % of the split)")
        row(record="obstacle_deck_compression", case=tag, dP=dP,
            P_euler=P_euler, frac_euler=dP / P_euler, df_rel=d_f,
            frac_of_split=abs(d_f) / s_true)
    print("      The deck carries no axial force in this model, so these are hand")
    print("      calculations on the worked bridge, not FE results.  They are not")
    print("      small: over a full stressing sweep the host mode moves by most of")
    print("      the split it is being used to measure, and even confined to the")
    print("      crossing window it moves by a sixth of it.")

    # H2 -- temperature ----------------------------------------------------
    print("\n  H2  temperature")
    dTdC = 0.005            # 0.5 % of stay force per degree C, reported range
    dfdC_deck = -0.0007     # -0.07 % per degree C, concrete deck, typical
    print(f"      stay force  ~{100*dTdC:.1f} %/degC  ->  stay frequency "
          f"{100*dTdC/2:.2f} %/degC")
    print(f"      deck mode   ~{100*dfdC_deck:.2f} %/degC")
    print(f"      net detuning drift {100*abs(dTdC/2 - dfdC_deck):.2f} %/degC "
          f"= {abs(dTdC/2 - dfdC_deck)/s_true:.2f} of the split per degree C")
    print(f"      A 10 degC swing moves the detuning by "
          f"{10*abs(dTdC/2-dfdC_deck)/s_true:.1f} splits.")
    row(record="obstacle_temperature", stay_force_per_degC=dTdC,
        stay_freq_per_degC=dTdC / 2, deck_freq_per_degC=dfdC_deck,
        detune_per_degC=abs(dTdC / 2 - dfdC_deck),
        splits_per_degC=abs(dTdC / 2 - dfdC_deck) / s_true)

    # H3 -- and why H1 and H2 hurt far less than they look ------------------
    print("\n  H3  why H1 and H2 hurt far less than their size suggests")
    print("      Both move the DETUNING, not the gap.  The complementarity of")
    print("      Part B says the gap is first-order blind to the detuning exactly")
    print("      where it is most informative about the coupling.  Provided both")
    print("      branches are read from the SAME record, the gap is instantaneous")
    print("      and a drift between records only relabels which step sat where.")
    print("\n      Monte Carlo, 17 holds over +-2 dT_c, sigma_f = 0.1 %,")
    print("      with an UNMODELLED per-step wander of the host frequency:")
    print("      wander    SE(phi_a) two branches   SE(phi_a) one branch only")
    for jit in (0.0, 0.002, 0.005, 0.01, 0.02):
        se2 = mc_se(Ts, phi_a, f_h, sigma_base, nrep=400, jitter_rel=jit)
        se1 = mc_se(Ts, phi_a, f_h, sigma_base, nrep=400, jitter_rel=jit,
                    free=("phi", "fh", "c"), single_branch=True)
        print(f"      {100*jit:4.1f} %       {100*se2/phi_a:7.2f} %            "
              f"{100*se1/phi_a:9.2f} %")
        row(record="obstacle_jitter", jitter_rel=jit,
            se_phi_rel_two_branch=se2 / phi_a,
            se_phi_rel_one_branch=se1 / phi_a, sigma_f=sigma_base)
    print("      Two-branch tracking is nearly immune; single-branch tracking is")
    print("      not.  THIS IS THE DESIGN RULE: read both branches from one")
    print("      record.  It is also why refinement (a) matters so much -- the")
    print("      configuration that survives drift is the one that needs the pair")
    print("      resolved, and Part F says that needs s > about 2 zeta.")

    # H4 -- is phi_a even the same quantity in service? --------------------
    print("\n  H4  is the anchorage ordinate measured during stressing the one")
    print("      that is wanted in service?")
    base_cd = CableDeck(T=T_TUNE, **BRIDGE)
    k_full = base_cd.k_ax
    print("      (a) the stay's own axial restraint, still being built up:")
    print("          k_ax/k_full   f_h [Hz]   phi_a          M_host [t]   change")
    for frac in (0.0, 0.25, 0.5, 1.0, 1.5):
        Kd, Md = chain(base_cd.Ld, base_cd.nd, base_cd.EId, base_cd.md, 0.0)
        Kd = Kd.copy(); Kd[2 * base_cd.ia, 2 * base_cd.ia] += frac * k_full
        keep = [i for i in range(Kd.shape[0]) if i not in (0, 2 * base_cd.nd)]
        w2, V = eigh(Kd[np.ix_(keep, keep)], Md[np.ix_(keep, keep)])
        ff = np.sqrt(np.maximum(w2, 0)) / (2 * np.pi)
        pos = {d: j for j, d in enumerate(keep)}
        j = int(np.argmin(np.abs(ff - f_h)))
        pa = abs(V[pos[2 * base_cd.ia], j])
        print(f"          {frac:9.2f}   {ff[j]:8.4f}   {pa:.6e}   "
              f"{1/pa**2/1e3:8.2f}   {100*(1/pa**2/M_host - 1):+7.2f} %")
        row(record="obstacle_kax", k_frac=frac, f_h=ff[j], phi_a=pa,
            M_host=1 / pa ** 2, change_pct=100 * (1 / pa ** 2 / M_host - 1))
    print("      (b) mass added after stressing (surfacing, parapets, furniture):")
    for dm in (0.0, 0.05, 0.10, 0.20):
        B2 = dict(BRIDGE); B2["md"] = BRIDGE["md"] * (1 + dm)
        cd2 = CableDeck(T=T_TUNE, **B2)
        Kd, Md = chain(cd2.Ld, cd2.nd, cd2.EId, cd2.md, 0.0)
        Kd = Kd.copy(); Kd[2 * cd2.ia, 2 * cd2.ia] += cd2.k_ax
        keep = [i for i in range(Kd.shape[0]) if i not in (0, 2 * cd2.nd)]
        w2, V = eigh(Kd[np.ix_(keep, keep)], Md[np.ix_(keep, keep)])
        ff = np.sqrt(np.maximum(w2, 0)) / (2 * np.pi)
        pos = {d: j for j, d in enumerate(keep)}
        j = int(np.argmin(np.abs(ff - f_h * np.sqrt(1 / (1 + dm)))))
        pa = abs(V[pos[2 * cd2.ia], j])
        print(f"          deck mass +{100*dm:4.1f} %:  f_h {ff[j]:7.4f} Hz   "
              f"phi_a {pa:.6e}   M_host {1/pa**2/1e3:7.2f} t "
              f"({100*(1/pa**2/M_host - 1):+6.2f} %)")
        row(record="obstacle_added_mass", deck_mass_frac=dm, f_h=ff[j],
            phi_a=pa, M_host=1 / pa ** 2,
            change_pct=100 * (1 / pa ** 2 / M_host - 1))
    print("      A 10 % dead load added after stressing moves the host modal")
    print("      mass by 10 % -- comparable with the whole measurement error.")
    print("      The measurement is of the structure AS IT WAS AT THE CROSSING.")
    print("      That is not fatal: the change is a known, computable transport,")
    print("      and it is the same objection that applies to any mass-change")
    print("      test done during construction.  But it forbids using the number")
    print("      raw as a service-condition scaling factor, and it rules out the")
    print("      erection stages -- cantilever, falsework -- where the deck mode")
    print("      is not the service mode at all.  The defensible window is a")
    print("      re-stressing or adjustment operation on the completed structure.")

    # H5 -- a third mode in the band ---------------------------------------
    print("\n  H5  a second host mode near the crossing biases the 2x2 fit")
    print("      3x3 pencil: stay + two host modes, second host mode offset by")
    print("      delta from the first, coupled with the same phi_a.")
    print("      offset delta   bias in fitted phi_a")
    g1 = g_of_T(phi_a, T_TUNE)
    for delta in (0.02, 0.05, 0.10, 0.20, 0.50):
        f2 = f_h * (1 + delta)
        Tg = np.linspace(T_TUNE - 2 * dT_c, T_TUNE + 2 * dT_c, 17)
        lo3, hi3 = [], []
        for T in Tg:
            gg = g_of_T(phi_a, T)
            A = np.array([[f_iso(T), gg / 2, gg / 2],
                          [gg / 2, f_h, 0.0],
                          [gg / 2, 0.0, f2]])
            ev, evec = np.linalg.eigh(A)
            # the two a stay accelerometer would pick: largest stay component
            top = np.argsort(np.abs(evec[0, :]))[-2:]
            lo3.append(float(ev[top].min())); hi3.append(float(ev[top].max()))
        lo3, hi3 = np.array(lo3), np.array(hi3)

        def rr(p):
            ph, fh, c = p
            lo, hi = branches(Tg, ph, fh, c)
            return np.concatenate([lo - lo3, hi - hi3])
        r = least_squares(rr, [phi_a, f_h, 1.0])
        print(f"      {100*delta:6.1f} %       {100*(r.x[0]/phi_a - 1):+8.2f} %")
        row(record="obstacle_third_mode", delta=delta, f2=f2,
            phi_bias_pct=100 * (r.x[0] / phi_a - 1))
    print("      A second host mode within 5 % of the first biases phi_a by")
    print("      several per cent, and the 2x2 model gives no warning.  On a")
    print("      cable-stayed bridge with a dense global spectrum this is the")
    print("      obstacle most likely to be met and least likely to be noticed.")
    print("      The 1/n redundancy of Part G is the only available detector.")


# ---------------------------------------------------------------------------
# PART I: head to head with the mass-change method
# ---------------------------------------------------------------------------

def partI(f_h, phi_a, M_host, s_true, sigma_base):
    print("\n" + "=" * 78)
    print("PART I   head to head with the mass-change method")
    print("=" * 78)
    print("""
  The incumbent (Parloo et al. 2002; Brincker and Andersen 2003) adds a known
  mass, re-identifies, and scales from the frequency shift.  For added mass
  dM at a point of mass-normalised ordinate phi_a,

      relative frequency shift   d = dM phi_a^2 / 2 = dM / (2 M_host)
      alpha^2 ~ (w0^2 - w1^2),  so   SE(alpha)/alpha = (1/sqrt2)(sigma_f/f)/d

  The veering measurement, at one resolved doublet, gives

      SE(phi_a)/phi_a = sqrt(2) (sigma_f/f) / s

  so the two are equal when d = s/2.  A veering split s is worth a mass-change
  experiment that shifts the frequency by s/2, which needs an added mass of
  s x M_host.  Both are then divided by sqrt(N_eff) or by the number of
  independent mass configurations, which favours neither.""")

    dM_equiv = s_true * M_host
    mu = mu_effective(M_S, phi_a)
    amp = s_true / mu
    print(f"\n  For the worked crossing (s = {100*s_true:.2f} %, M_host = "
          f"{M_host/1e3:.1f} t):")
    print(f"    equivalent added mass  dM = s M_host = {dM_equiv:.0f} kg "
          f"({100*s_true:.2f} % of the modal mass)")
    print(f"    the stay that produces it is only M_s phi_a^2 = "
          f"{100*mu:.3f} % of the modal mass")
    print(f"    AMPLIFICATION = s / mu_eff = {amp:.1f}")
    print(f"    because veering goes as sqrt(mu_eff), not mu_eff.  A perturbation")
    print(f"    of {100*mu:.3f} % is read out as a {100*s_true:.2f} % frequency")
    print(f"    signature.  That resonant amplification is the whole case for the")
    print(f"    reversal, and it is a factor {amp:.0f} here.")
    row(record="equivalence", s=s_true, M_host=M_host, dM_equiv=dM_equiv,
        mu_eff=mu, amplification=amp)

    print("\n  What the incumbent would need, and what it would return, at the")
    print("  same frequency precision sigma_f/f = 0.1 %:")
    print("    added mass      shift d    SE(alpha)   SE(M_host)   feasible on a bridge?")
    for frac, note in ((0.002, "a 68 kg toolbox"),
                       (0.01, "70 kg"),
                       (0.0234, "800 kg -- matches this stay"),
                       (0.05, "1.7 t"),
                       (0.10, "3.4 t of kentledge")):
        d = frac / 2.0
        se_a = (1 / np.sqrt(2)) * sigma_base / d
        print(f"    {100*frac:5.2f} % M_host   {100*d:5.3f} %   {100*se_a:7.1f} %   "
              f"{200*se_a:8.1f} %    {note}")
        row(record="masschange_curve", dM_frac=frac, shift=d,
            se_alpha_rel=se_a, se_Mhost_rel=2 * se_a, sigma_f=sigma_base,
            note=note)
    print("    On a real deck mode of 200-500 t those percentages are 5-50 t of")
    print("    kentledge, craned on and off, with the bridge closed.  That is the")
    print("    practical reason mass change is not done on bridges, and it is the")
    print("    gap the reversal is aimed at.")

    dT_c = 2.0 * s_true * T_TUNE
    print("\n  THE DECISIVE COMPARISON, at sigma_f/f = 0.1 % throughout:")
    designs = [
        ("veering, one resolved doublet at exact tuning", 1, 0.0),
        ("veering, as-built 8-hold stressing schedule", None, None),
        ("veering, 15 holds inside +-2 dT_c + 2 wing holds", 17, 2.0),
    ]
    for label, nst, w in designs:
        if nst is None:
            Ts = np.linspace(0.40 * T_TUNE, 1.25 * T_TUNE, 8)
            Ts = Ts[(Ts > 105e3) & (Ts < 205e3)]
        elif nst == 1:
            Ts = np.array([T_TUNE])
        else:
            Ts = np.sort(np.concatenate([
                np.linspace(T_TUNE - w * dT_c, T_TUNE + w * dT_c, nst - 2),
                [T_TUNE - 8 * dT_c, T_TUNE + 8 * dT_c]]))
        free = ("phi",) if len(Ts) < 3 else ("phi", "fh", "c")
        se, n_eff = fisher_se(Ts, phi_a, f_h, sigma_base, free=free)
        print(f"    {label:48s} SE(phi_a) {100*se/phi_a:5.2f} %  "
              f"SE(M_host) {200*se/phi_a:5.2f} %")
        row(record="head_to_head", method="veering", design=label,
            n_holds=len(Ts), N_eff=n_eff, se_phi_rel=se / phi_a,
            se_Mhost_rel=2 * se / phi_a, sigma_f=sigma_base)
    for frac, lab in ((0.01, "mass change, 1 % of modal mass added"),
                      (0.05, "mass change, 5 % of modal mass added")):
        se_a = (1 / np.sqrt(2)) * sigma_base / (frac / 2)
        print(f"    {lab:48s} SE(alpha)  {100*se_a:5.2f} %  "
              f"SE(M_host) {200*se_a:5.2f} %")
        row(record="head_to_head", method="mass_change", design=lab,
            dM_frac=frac, se_phi_rel=se_a, se_Mhost_rel=2 * se_a,
            sigma_f=sigma_base)
    print("""
    On formula alone the reversal MATCHES a 2.3 %-of-modal-mass mass-change
    test and BEATS a 1 % one, at no added mass and no bridge closure.  Neither
    figure includes the mass-change method's own second error source, the mode
    shape at the mass locations, which enters squared and is reported in that
    literature as the reason optimised mass placement matters; the reversal has
    the mirror-image weakness in the stay schedule, which is known far better.""")


# ---------------------------------------------------------------------------

def verdict(f_h, phi_a, M_host, s_true, sigma_base):
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    dT_c = 2.0 * s_true * T_TUNE
    Ts_asb = np.linspace(0.40 * T_TUNE, 1.25 * T_TUNE, 8)
    Ts_asb = Ts_asb[(Ts_asb > 105e3) & (Ts_asb < 205e3)]
    Ts_des = np.sort(np.concatenate([
        np.linspace(T_TUNE - 2 * dT_c, T_TUNE + 2 * dT_c, 15),
        [T_TUNE - 8 * dT_c, T_TUNE + 8 * dT_c]]))
    se_a, _ = fisher_se(Ts_asb, phi_a, f_h, sigma_base)
    se_d, _ = fisher_se(Ts_des, phi_a, f_h, sigma_base)
    print(f"""
  HOW PRECISE IS phi_a FROM A REALISTIC SWEEP?  At sigma_f/f = 0.1 %, on the
  worked bridge's 2.34 % crossing:

      a stressing schedule as normally run       {100*se_a/phi_a:5.2f} % on phi_a,"""
          f" {200*se_a/phi_a:5.1f} % on the modal mass\n"
          f"      a schedule deliberately refined at the\n"
          f"        crossing (17 holds)                      {100*se_d/phi_a:5.2f} %"
          f" on phi_a, {200*se_d/phi_a:5.1f} % on the modal mass\n"
          f"""
  Both sit on a systematic floor of about 0.12 % from the two-mode idealisation.

  DOES IT BEAT THE INCUMBENT?  At matched frequency precision, a mass-change
  test would have to add {200*(1/np.sqrt(2))*sigma_base/(se_a/phi_a):.1f} % of the modal mass to match the schedule as
  normally run, and {200*(1/np.sqrt(2))*sigma_base/(se_d/phi_a):.1f} % to match the refined one -- {(1/np.sqrt(2))*sigma_base/(se_d/phi_a)*2*M_host/1e3:.1f} tonnes on this
  34 t modal mass, and 13-33 t on a real deck mode of 200-500 t.  The reason is
  not subtle and it is the substance of the proposal: veering responds to sqrt(mu_eff), so this stay's 0.20 % of the
  modal mass is read out as a 2.34 % frequency signature, an amplification of
  {s_true/mu_effective(M_S, phi_a):.0f}.  The equivalent kentledge is {s_true*M_host:.0f} kg here and 5-50 t on a
  real deck mode, craned on and off with the bridge closed.  That is a real
  advantage and it is the whole case.

  THE FACTOR OF TEN WAS NOT REPRESENTATIVE.  The Ponte del Mare traverse had
  N_eff = 0.15 against 8-9 for a designed sweep, because both of its two points
  sat in the wings where the response to the coupling is quadratic.  Its bracket
  is what that geometry owes, not what the method owes.

  WHAT ACTUALLY LIMITS IT, in order of severity.

    1  Damping, unchanged by the refinements.  The pair must be resolvable on
       the HYBRID branches, s > about 1.2 zeta with a curve fit.  On this
       crossing that caps the usable deck damping near 2 %.  Refinement (a) does
       not lift this; the merged-peak centroid is the isolated-cable frequency
       by an exact identity and carries essentially nothing.
    2  Where the holds are put.  N_eff counts only holds inside the crossing,
       and a normal stressing schedule puts in one.  The precision is bought by
       designing the schedule, not by reading a schedule someone else designed,
       and that means asking a contractor to hold tension at prescribed loads
       for a 10 minute record each.  That is the real cost of the method.
    3  A second host mode within 5 % of the first, which biases phi_a by several
       per cent with no warning from the fit.  Cable-stayed spectra are dense.
       The only detector is the 1/n redundancy, and refinement (b) shows that
       for a stay like this one exactly one order is on a taut cable.
    4  Whether the structure measured is the structure wanted.  A 10 % dead load
       added after stressing moves the modal mass by 10 %.  The number is
       transportable, but only from a configuration that can be modelled, which
       rules out erection stages and points at re-stressing or adjustment on a
       completed bridge as the defensible window.
    5  Not instrumentation.  This was expected to be a limit and is not: the
       host-dominated branch stays within about 13 dB of the other at a stay
       accelerometer even at |D| = 8g, because the tie drags the stay
       kinematically whether or not the mode is resonant with it.  A stay-
       mounted accelerometer alone can read both branches across the sweep.

  WHAT IS NOT A LIMIT, and this was the surprise.  Not knowing the stay's own
  f(T) law barely matters.  Mass per length, effective length, jack calibration
  and anchor set inflate the standard error by under a third, because they move
  where the branches sit while phi_a sets how far apart they are, and the gap is
  first-order blind to the detuning exactly where it carries the coupling.  The
  same complementarity makes the measurement nearly immune to host-frequency
  drift from deck compression and temperature -- the two obstacles that looked
  largest at the outset -- provided the pair is read simultaneously.

  SO: a method, not a curiosity, but a narrower one than proposed.  It is a
  designed experiment that a stressing operation can host, not a free by-product
  of stressing; it needs a low-damping crossing and a spectrum clean enough that
  no second host mode sits within 5 % of the one being measured.
  Where those hold it returns the host modal mass at an anchorage to about 4 %,
  which no mass-change campaign on a bridge deck will match.  Both of the
  refinements offered to widen it fail: (a) because a merged pair carries the
  isolated-cable frequency and nothing else, (b) because the higher orders cross
  where the stay is no longer a taut cable.""")
    row(record="verdict", se_phi_asbuilt=se_a / phi_a,
        se_phi_designed=se_d / phi_a, se_Mhost_asbuilt=2 * se_a / phi_a,
        se_Mhost_designed=2 * se_d / phi_a, sigma_f=sigma_base,
        amplification=s_true / mu_effective(M_S, phi_a),
        dM_equiv_kg=s_true * M_host)


def main():
    f_h, phi_a, M_host, s_true = part0()
    sigma_base = partA(f_h)
    partB(f_h, phi_a, s_true, sigma_base)
    partC(f_h, phi_a, M_host, s_true, sigma_base)
    partD(f_h, phi_a, s_true, sigma_base)
    partE()
    partF(f_h, phi_a, s_true, sigma_base)
    partG(f_h, phi_a, sigma_base)
    partH(f_h, phi_a, M_host, s_true, sigma_base)
    partI(f_h, phi_a, M_host, s_true, sigma_base)
    verdict(f_h, phi_a, M_host, s_true, sigma_base)

    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "reversal.csv")
    pd.DataFrame(ROWS).to_csv(path, index=False)
    print(f"\nwrote {path}  ({len(ROWS)} rows)")


if __name__ == "__main__":
    main()
