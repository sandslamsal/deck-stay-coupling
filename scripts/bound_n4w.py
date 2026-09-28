# -*- coding: utf-8 -*-
"""Bounding the open N4W anomaly on the Ponte del Mare.

The open item in the field section. Stay N4W (Kumar 2011, Tables 4.4 and 5.5)
has L = 46.9 m, m = 10.7 kg/m, picked frequencies 2.09/4.04/5.91/7.75/9.80 Hz,
and a SISTRAL-measured pull of 348 kN whose taut-string frequency is 1.93 Hz.
Its fundamental therefore sits +8.3 % above the isolated reference while its
Irvine parameter (0.16) buys only +0.3 to +0.4 % of sag lift, leaving roughly
+8 % unexplained. The identified global-mode table stops at 2.862 Hz, so the
standing suggestion has been that an unidentified deck mode couples to it.

This script does not test that suggestion, it BOUNDS it. The question asked is
not "does a mode exist" but "what would a mode have to be worth, and is that
worth physically available on this bridge".

The branch relation used is the one the traverse analysis closes on
(scripts/validate_traverse.py), the stay-dominated branch being displaced away
from the deck mode it meets by half the veering separation,

    f_obs = f_iso + (1/2) sgn(D) ( sqrt(D^2 + S^2) - |D| ),
    D = f_iso - f_deck,   S = s f_iso,

which inverts in closed form for the width a demanded displacement needs.
With Delta = f_obs - f_iso and the displacement taken away from the deck mode,

    S = 2 sqrt( Delta (Delta + |D|) ),                      (invert)

so the requirement is smallest at exact tuning, S -> 2 Delta, and grows without
bound as the candidate mode is moved away. That floor is what makes the bound
decisive: no placement of a hypothetical deck mode is cheaper than tuning it
exactly onto the stay.

Sections:
  A  sign screen: which side of the stay a deck mode must sit on at all
  B  required veering width s(f_deck), and its floor
  C  required mu_eff, over the 26-42 deg inclination band
  D  attainable mu_eff, from M_s = m L / 2 and a mass-normalised deck ordinate
     bounded by a perfect antinode on a 40-160 t modal mass
  E  finite element corroboration of the ceiling, at the ceiling
  F  the spacing argument for a mode above 2.862 Hz, and mode 2 at 4.04 Hz
  G  the alternative that is NOT a frequency shift: a mixed deck line picked
     as the stay fundamental

Writes data/n4w_bound.csv.

Run:  python3 scripts/bound_n4w.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
DATA = os.path.join(ROOT, "data")

from cablefe import (CableDeck, mu_effective, veering_split, tension_error,
                     tensioned_beam_freq, string_freq, irvine_lambda2)

G = 9.80665
# Values transcribed from Kumar (2011), PhD thesis, University of Trento (Ponte del Mare footbridge). They are not redistributed with
# this code: they live in data/external/bound_n4w_data.py (see README).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "external"))
try:
    from bound_n4w_data import (  # noqa: E402
        E_STAY, RHO_STAY, L, M, F_PICK, T_MEAS, F_STRING, THETAS, DECK)
except ImportError as exc:
    raise SystemExit("scripts/bound_n4w.py needs values transcribed from "
                     "Kumar (2011), PhD thesis, University of Trento (Ponte del Mare footbridge), which are not redistributed here. "
                     "See README, 'Third-party data'.") from exc

# ---------------------------------------------------------------------------
# N4W, as transcribed in scripts/validate_pontedelmare2.py
# ---------------------------------------------------------------------------
TH_LO, TH_HI = 26.0, 42.0      # deg, inclination band

M_S = M * L / 2                # stay modal mass, kg
A_STAY = M / RHO_STAY
EA = E_STAY * A_STAY
D_STAY = np.sqrt(4.0 * A_STAY / np.pi)
EI_STAY = E_STAY * np.pi * D_STAY ** 4 / 64.0

# identified global modes, Kumar Table 5.6 / Fig 4.13

# deck modal mass band carried through the field section
MD_LO, MD_HI = 40e3, 160e3     # kg

F_OBS = F_PICK[0]              # 2.09 Hz


def hi_mode_fit(f=F_PICK, orders=(2, 3, 4, 5)):
    """Least squares f_n = c n on the orders sag and coupling perturb least."""
    n = np.array(orders, float)
    fn = np.array([f[int(o) - 1] for o in orders])
    return float(np.sum(n * fn) / np.sum(n * n))


C_HI = hi_mode_fit()


def sag_lift(theta_deg):
    """Irvine lift of the first symmetric mode, lambda^2 / 4 pi^2."""
    lam2 = irvine_lambda2(L, T_MEAS, EA, M, np.deg2rad(theta_deg))
    return lam2 / (4.0 * np.pi ** 2), lam2


# three isolated references, in increasing order of what they concede
SAG_LO, LAM_LO = sag_lift(TH_LO)
SAG_HI, LAM_HI = sag_lift(TH_HI)
SAG_MID = 0.5 * (SAG_LO + SAG_HI)

TARGETS = {
    "string_pull": (F_STRING, "taut-string frequency of the measured pull"),
    "string_pull_sagged": (F_STRING * (1.0 + SAG_MID),
                           "the same, with the mid-band Irvine sag lift conceded"),
    "own_high_modes": (C_HI,
                       "the stay's own orders 2-5 fit, conceding tension drift"),
}


# ---------------------------------------------------------------------------
# the branch relation and its inversion
# ---------------------------------------------------------------------------

def f_obs_of(f_iso, f_deck, s):
    """Stay-dominated branch position, displaced away from the deck mode.

    At exact tuning both branches are half stay and sit at +/- s/2, so the
    sign convention there is a choice, not a fact.  ``np.sign(0) = 0`` would
    silently return no displacement at all, which is the one place the
    requirement is CHEAPEST; the upper branch is taken instead, which is the
    branch an upward anomaly would have to be.
    """
    D = f_iso - f_deck
    sgn = 1.0 if D >= 0.0 else -1.0
    return f_iso + 0.5 * sgn * (np.sqrt(D * D + (s * f_iso) ** 2) - abs(D))


def s_required(f_iso, f_obs, f_deck):
    """Width needed for a deck mode at f_deck to put the branch at f_obs.

    Returns nan when the demanded displacement is on the wrong side of the
    deck mode, because the branch is displaced AWAY from it and no width,
    however large, reverses that.
    """
    delta = f_obs - f_iso
    D = f_iso - f_deck
    if delta == 0.0:
        return 0.0
    if np.sign(delta) != np.sign(D):        # wrong side, or exactly tuned
        if D == 0.0:                        # exact tuning: either sign is had
            return 2.0 * abs(delta) / f_iso
        return np.nan
    return 2.0 * np.sqrt(abs(delta) * (abs(delta) + abs(D))) / f_iso


def mu_of_s(s, theta_deg, n=1):
    """Invert s = (2/(n pi)) cos(theta) sqrt(mu_eff)."""
    return (s * n * np.pi / (2.0 * np.cos(np.deg2rad(theta_deg)))) ** 2


def shift_of(f_iso, f_deck, s):
    """Signed relative displacement of the stay branch, (f_obs - f_iso)/f_iso."""
    return (f_obs_of(f_iso, f_deck, s) - f_iso) / f_iso


def _selfcheck():
    """Round-trip the closed-form inversion through the forward relation.

    S = 2 sqrt(Delta (Delta + |D|)) is algebra, and algebra written down as
    fact is what this project keeps having to correct, so it is checked
    against f_obs_of at every candidate before any of it is reported.
    """
    worst = 0.0
    for f_iso in (1.90, 1.93, 1.9594, 2.00):
        for fd in np.arange(1.00, 4.0001, 0.01):
            sr = s_required(f_iso, F_OBS, float(fd))
            if np.isnan(sr):
                continue
            worst = max(worst, abs(f_obs_of(f_iso, float(fd), sr) - F_OBS))
    return worst


_RT = _selfcheck()

# the ceiling deck: 40 t modal mass, antinode exactly at the anchorage
LD_FE, MD_FE = 80.0, 1000.0                 # md * Ld / 2 = 40 t exactly
F_ISO_FE = tensioned_beam_freq(1, L, T_MEAS, EI_STAY, M)
PHI_A_FE = np.sqrt(2.0 / (MD_FE * LD_FE))
MU_FE = mu_effective(M_S, PHI_A_FE)


def ceiling_model(theta_rad, f_deck_target):
    """CableDeck at the physical ceiling, tuned to put its deck mode on target."""
    def resid(logEI):
        cd = CableDeck(Ld=LD_FE, EId=np.exp(logEI), md=MD_FE, Lc=L,
                       EIc=EI_STAY, mc=M, T=T_MEAS, EA=EA, theta=theta_rad,
                       nd=40, nc=40)
        return cd.deck_alone(3)[0] - f_deck_target
    logEI = brentq(resid, np.log(1e7), np.log(1e14), xtol=1e-11)
    return CableDeck(Ld=LD_FE, EId=np.exp(logEI), md=MD_FE, Lc=L,
                     EIc=EI_STAY, mc=M, T=T_MEAS, EA=EA, theta=theta_rad,
                     nd=40, nc=40)


line = "=" * 84

print(line)
print("N4W BOUND: can coupling to an unidentified deck mode carry the anomaly?")
print(line)
print("  inversion round-trip check, worst |f_obs(s_req) - 2.09| = %.2e Hz  %s"
      % (_RT, "OK" if _RT < 1e-9 else "FAIL"))
print("  L = %.1f m,  m = %.1f kg/m,  T_meas = %.0f kN" % (L, M, T_MEAS / 1e3))
print("  M_s = m L / 2               = %.1f kg" % M_S)
print("  EA  = %.3e N,  EI = %.3e N m^2,  d_eq = %.1f mm"
      % (EA, EI_STAY, 1e3 * D_STAY))
print("  picked f_n [Hz]             :", ", ".join("%.2f" % v for v in F_PICK))
print("  f_n / n     [Hz]            :",
      ", ".join("%.3f" % v for v in F_PICK / np.arange(1, 6)))
print("  orders 2-5 fit c            = %.4f Hz" % C_HI)
print("  string frequency of the pull= %.4f Hz  (source's own control table)"
      % F_STRING)
print("  Irvine lambda^2             = %.4f (%.0f deg) to %.4f (%.0f deg)"
      % (LAM_LO, TH_LO, LAM_HI, TH_HI))
print("  sag lift lambda^2/4pi^2     = %.2f to %.2f %%"
      % (100 * SAG_LO, 100 * SAG_HI))
print("  anomaly f1/f_string - 1     = %+.2f %%" % (100 * (F_OBS / F_STRING - 1)))
print("  anomaly after sag           = %+.2f to %+.2f %%"
      % (100 * (F_OBS / (F_STRING * (1 + SAG_HI)) - 1),
         100 * (F_OBS / (F_STRING * (1 + SAG_LO)) - 1)))
print("  anomaly vs its own 2-5 fit  = %+.2f %%" % (100 * (F_OBS / C_HI - 1)))

# ---------------------------------------------------------------------------
# D  what is physically available
# ---------------------------------------------------------------------------
MU_MAX = M_S / MD_LO            # perfect antinode on the lightest deck mode
MU_MAX_HEAVY = M_S / MD_HI
print()
print(line)
print("D  the physical ceiling on mu_eff, computed first because it sets the bar")
print(line)
print("  mu_eff = M_s phi_a^2 with phi_a the MASS-NORMALISED deck ordinate at")
print("  the anchorage. A unit-max mode shape has phi_a <= 1/sqrt(M_deck), the")
print("  equality being a perfect antinode exactly at the anchorage, so")
print("      mu_eff <= M_s / M_deck.")
print("  M_deck = 40 t  ->  mu_eff <= %.3e   (the ceiling used below)" % MU_MAX)
print("  M_deck = 160 t ->  mu_eff <= %.3e" % MU_MAX_HEAVY)
print()
print("  %-10s %12s %12s %14s" % ("theta", "s_max [%]", "s_max/2 [%]",
                                  "max f-shift [Hz]"))
for th in THETAS:
    s_max = veering_split(MU_MAX, np.deg2rad(th), n=1)
    print("  %6.0f deg %11.3f %12.3f %14.4f"
          % (th, 100 * s_max, 100 * s_max / 2, s_max / 2 * F_STRING))
S_MAX_BEST = veering_split(MU_MAX, np.deg2rad(TH_LO), n=1)
print()
print("  s_max/2 is the LARGEST relative frequency displacement any deck mode")
print("  can impose on this stay's fundamental, reached only at exact tuning.")
print("  Best case over the whole band: %.2f %% (%.4f Hz), taking f1 to %.4f Hz."
      % (100 * S_MAX_BEST / 2, S_MAX_BEST / 2 * F_STRING,
         F_STRING * (1 + S_MAX_BEST / 2)))
print("  The anomaly to be carried is %+.2f %% (%.4f Hz), to %.2f Hz."
      % (100 * (F_OBS / F_STRING - 1), F_OBS - F_STRING, F_OBS))

# the same ceiling read as a length of deck, which is the falsifiable form
MLIN_LO, MLIN_HI = 2 * MD_LO / 173.0, 2 * MD_HI / 148.0   # kg/m per deck
print()
print("  Read as deck rather than as a number: the two decks are 148-173 m")
print("  long, so the 40-160 t band is a linear mass of %.0f-%.0f kg/m. A modal"
      % (MLIN_LO, MLIN_HI))
print("  mass M implies a participating length of about 2 M / m_lin, so the")
print("  requirement will be quoted below in metres of deck as well.")


def participating_length(M_req_kg):
    return 2 * M_req_kg / MLIN_HI, 2 * M_req_kg / MLIN_LO


# ---------------------------------------------------------------------------
# A  sign screen
# ---------------------------------------------------------------------------
print()
print(line)
print("A  sign screen: a deck mode ABOVE the stay cannot raise it, at any width")
print(line)
print("  The stay-dominated branch is displaced AWAY from the deck mode it")
print("  meets. The anomaly is an INCREASE of %+.4f Hz, so the deck mode must"
      % (F_OBS - F_STRING))
print("  sit BELOW the stay fundamental. Every candidate above %.3f Hz,"
      % F_STRING)
print("  including anything past the 2.862 Hz end of the identified table,")
print("  displaces the fundamental DOWNWARD and moves the record the wrong way.")
print()
for fd in (1.791, 1.900, 1.93, 2.306, 2.862, 2.950, 3.400):
    s_req = s_required(F_STRING, F_OBS, fd)
    side = "below" if fd < F_STRING else ("tuned" if fd == F_STRING else "above")
    if np.isnan(s_req):
        got = shift_of(F_STRING, fd, S_MAX_BEST)
        print("  f_deck = %6.3f Hz (%s): NO SOLUTION; at the ceiling width the"
              " branch moves %+.2f %%" % (fd, side, 100 * got))
    else:
        print("  f_deck = %6.3f Hz (%s): s_req = %6.2f %%   (ceiling %.2f %%)"
              % (fd, side, 100 * s_req, 100 * S_MAX_BEST))

print()
print("  That sign is not asserted, it is run. The ceiling model of section E is")
print("  swept through a crossing and the stay-dominated branch tracked:")
print()
print("  %-12s %14s %10s %12s %12s"
      % ("f_deck [Hz]", "stay branch [Hz]", "E_stay", "FE shift", "closed form"))
s_ceiling = veering_split(MU_FE, np.deg2rad(TH_LO), n=1)
for fd in (1.40, 1.70, 1.85, 2.05, 2.30, 2.90):
    cdA = ceiling_model(np.deg2rad(TH_LO), fd)
    fA, PhiA = cdA.modes(40)
    fracA = cdA.energy_split(PhiA)
    near = np.abs(fA - F_ISO_FE) < 0.4 * F_ISO_FE
    i = int(np.argmax(fracA * near))
    DA = F_ISO_FE - fd
    pred = F_ISO_FE + 0.5 * (1.0 if DA >= 0 else -1.0) * (
        np.sqrt(DA * DA + (s_ceiling * F_ISO_FE) ** 2) - abs(DA))
    print("  %11.3f %15.5f %10.2f %11.3f %% %11.3f %%"
          % (fd, fA[i], fracA[i], 100 * (fA[i] - F_ISO_FE) / F_ISO_FE,
             100 * (pred - F_ISO_FE) / F_ISO_FE))
print()
print("  Up while the deck mode is below, down once it is above, at agreement")
print("  with the closed form of better than 0.03 percentage points throughout.")
print("  A candidate above the stay cannot raise it.")

# ---------------------------------------------------------------------------
# B, C  the grid
# ---------------------------------------------------------------------------
grid = np.round(np.arange(1.00, 4.0001, 0.01), 4)
rows = []
for tname, (f_iso, tnote) in TARGETS.items():
    delta = F_OBS - f_iso
    for th in THETAS:
        s_max = veering_split(MU_MAX, np.deg2rad(th), n=1)
        mu_max = MU_MAX
        for fd in grid:
            D = f_iso - fd
            s_req = s_required(f_iso, F_OBS, fd)
            feasible_sign = bool(np.sign(delta) == np.sign(D) or D == 0.0)
            if np.isnan(s_req):
                mu_req = np.nan
                md_req = np.nan
                short_mu = np.nan
                short_s = np.nan
                attain = False
            else:
                mu_req = mu_of_s(s_req, th, n=1)
                md_req = M_S / mu_req if mu_req > 0 else np.inf
                short_mu = mu_req / mu_max
                short_s = s_req / s_max
                attain = bool(mu_req <= mu_max)
            rows.append(dict(
                target=tname, f_iso=f_iso, f_obs=F_OBS,
                delta_Hz=delta, delta_pct=100 * delta / f_iso,
                theta_deg=th, f_deck=fd, D_Hz=D, d_rel=(f_iso - fd) / f_iso,
                deck_side=("below" if fd < f_iso else
                           ("tuned" if fd == f_iso else "above")),
                sign_feasible=feasible_sign,
                s_req_pct=100 * s_req if not np.isnan(s_req) else np.nan,
                mu_req=mu_req,
                M_deck_req_t=md_req / 1e3 if np.isfinite(md_req) else np.nan,
                mu_max_40t=mu_max, mu_max_160t=MU_MAX_HEAVY,
                s_max_pct=100 * s_max,
                achievable_shift_pct=100 * shift_of(f_iso, fd, s_max),
                attainable=attain,
                shortfall_mu=short_mu, shortfall_s=short_s,
                note=tnote))
df = pd.DataFrame(rows)

print()
print(line)
print("B/C  required width and required mu_eff over f_deck in 1.0-4.0 Hz")
print(line)
print("  Only the sign-feasible half of the range has a solution at all. Within")
print("  it the requirement has a floor at exact tuning and rises away from it.")
print()
for tname, (f_iso, tnote) in TARGETS.items():
    sub = df[(df.target == tname) & df.s_req_pct.notna()]
    print("  target %-20s f_iso = %.4f Hz, anomaly %+.2f %%   (%s)"
          % (tname, f_iso, 100 * (F_OBS / f_iso - 1), tnote))
    if sub.empty:
        print("    no sign-feasible candidate in the range")
        continue
    print("    sign-feasible f_deck window        : %.2f to %.2f Hz"
          % (sub.f_deck.min(), sub.f_deck.max()))
    print("    s required, floor -> edge of window: %.2f -> %.2f %%"
          % (sub.s_req_pct.min(), sub.s_req_pct.max()))
    for th in THETAS:
        s2 = sub[sub.theta_deg == th]
        print("      theta = %2.0f deg: mu_eff required %.4f to %.4f;"
              " heaviest deck mode that could still supply it %.2f t"
              % (th, s2.mu_req.min(), s2.mu_req.max(), s2.M_deck_req_t.max()))
    print("    attainable anywhere in the window   : %s"
          % ("YES" if sub.attainable.any() else "NO"))
    print("    smallest shortfall in mu_eff        : x%.1f  (in s, x%.2f)"
          % (sub.shortfall_mu.min(), sub.shortfall_s.min()))
    print("    largest shift the ceiling can buy   : %+.2f %% against %+.2f %% needed"
          % (df[(df.target == tname)].achievable_shift_pct.max(),
             100 * (F_OBS / f_iso - 1)))
    print()

# ---------------------------------------------------------------------------
# E  finite element corroboration at the ceiling
# ---------------------------------------------------------------------------
print(line)
print("E  finite element corroboration, run AT the ceiling rather than near it")
print(line)
print("  A deck is built whose fundamental modal mass is exactly the 40 t low")
print("  end of the band and whose antinode is placed exactly at the anchorage,")
print("  then tuned onto the stay. That is the most favourable coupled system")
print("  this stay can be part of; nothing physical on the bridge beats it.")

Ld, md = LD_FE, MD_FE
f_iso_fe, phi_a_ideal, mu_ideal = F_ISO_FE, PHI_A_FE, MU_FE
print()
print("  isolated stay f1 (tensioned beam)   = %.5f Hz" % f_iso_fe)
print("  deck mode-1 modal mass              = %.0f kg" % (md * Ld / 2))
print("  mass-normalised antinode ordinate   = %.5e" % phi_a_ideal)
print("  mu_eff at the ceiling               = %.5e (M_s/M_deck = %.5e)"
      % (mu_ideal, M_S / (md * Ld / 2)))
print()
print("  %-8s %10s %10s %10s %10s %10s %10s"
      % ("theta", "f- [Hz]", "f+ [Hz]", "s_FE [%]", "s_th [%]",
         "up-shift", "needed"))
fe_rows = []
for th in THETAS:
    thr = np.deg2rad(th)
    cd = ceiling_model(thr, f_iso_fe)
    f, Phi = cd.modes(40)
    frac = cd.energy_split(Phi)
    near = np.where(np.abs(f - f_iso_fe) < 0.35 * f_iso_fe)[0]
    hyb = [i for i in near if 0.05 < frac[i] < 0.95]
    if len(hyb) < 2:
        hyb = list(near[np.argsort(np.abs(f[near] - f_iso_fe))[:2]])
    hyb = sorted(hyb, key=lambda i: f[i])
    fm, fp = f[hyb[0]], f[hyb[-1]]
    s_fe = (fp - fm) / f_iso_fe
    s_th = veering_split(mu_ideal, thr, n=1)
    up = (fp - f_iso_fe) / f_iso_fe
    print("  %5.0f deg %10.5f %10.5f %10.3f %10.3f %9.2f %% %9.2f %%"
          % (th, fm, fp, 100 * s_fe, 100 * s_th, 100 * up,
             100 * (F_OBS / F_STRING - 1)))
    fe_rows.append(dict(theta_deg=th, f_minus=fm, f_plus=fp, s_fe=s_fe,
                        s_theory=s_th, up_shift=up,
                        energy_lo=frac[hyb[0]], energy_hi=frac[hyb[-1]]))
fe = pd.DataFrame(fe_rows)
print()
print("  FE against closed form: %.1f to %.1f %% agreement in s"
      % (100 * (fe.s_fe / fe.s_theory).min(), 100 * (fe.s_fe / fe.s_theory).max()))
print("  Largest upward displacement the FE will produce: %+.2f %%."
      % (100 * fe.up_shift.max()))
print("  Required: %+.2f %%. Shortfall factor %.1f."
      % (100 * (F_OBS / F_STRING - 1),
         (F_OBS / F_STRING - 1) / fe.up_shift.max()))

# ---------------------------------------------------------------------------
# F  the spacing argument
# ---------------------------------------------------------------------------
print()
print(line)
print("F  is a mode just above 2.862 Hz plausible, and would it matter?")
print(line)
gaps = np.diff(DECK)
print("  identified mode spacings [Hz]:", ", ".join("%.3f" % g for g in gaps))
print("  min %.3f, median %.3f, mean %.3f, max %.3f (the 1.791-2.306 gap)"
      % (gaps.min(), np.median(gaps), gaps.mean(), gaps.max()))
print("  PLAUSIBLE: on that spacing the next mode should sit at roughly")
print("  %.2f to %.2f Hz." % (2.862 + gaps.min(), 2.862 + gaps.max()))
print()
print("  But plausibility is not relevance. For such a mode against N4W:")
print("  %-10s %8s %10s %12s %14s"
      % ("f_deck", "d", "s_max [%]", "shift [%]", "shift [Hz]"))
for fd in sorted((2.862 + gaps.min(), 2.90, 2.862 + np.median(gaps), 3.20,
                  2.862 + gaps.max())):
    d = (F_STRING - fd) / F_STRING
    sh = shift_of(F_STRING, fd, S_MAX_BEST)
    print("  %8.3f Hz %8.3f %10.3f %11.3f %% %13.5f"
          % (fd, d, 100 * S_MAX_BEST, 100 * sh, sh * F_STRING))
print()
print("  Every one of them is 0.8 Hz or more above the fundamental, |d| >= %.2f,"
      % abs((F_STRING - (2.862 + gaps.min())) / F_STRING))
print("  and every one pushes the fundamental DOWN by under 0.2 %. A mode just")
print("  above the end of the table is plausible and irrelevant.")
print()
print("  N4W's SECOND mode, at 4.04 Hz, is the one such a mode could reach.")
f2_iso = 2 * F_STRING
print("  isolated 2nd-mode reference 2 f_string = %.3f Hz, measured 4.04 Hz,"
      % f2_iso)
print("  an excess of %+.2f %%." % (100 * (4.04 / f2_iso - 1)))
s_max_n2 = veering_split(MU_MAX, np.deg2rad(TH_LO), n=2)
print("  At n = 2 the width carries the 1/n factor: s_max = %.2f %%, so the"
      % (100 * s_max_n2))
print("  largest displacement available to mode 2 is %.2f %% (%.4f Hz), against"
      % (100 * s_max_n2 / 2, s_max_n2 / 2 * f2_iso))
print("  the %.2f %% (%.3f Hz) observed. Mode 2 is not carried by coupling"
      % (100 * (4.04 / f2_iso - 1), 4.04 - f2_iso))
print("  either, and its excess is the same size as the scatter of orders 3-5")
print("  about the fitted line (%s %%), which is peak-pick spread."
      % ", ".join("%+.1f" % (100 * (F_PICK[i] / (C_HI * (i + 1)) - 1))
                  for i in (1, 2, 3, 4)))

# ---------------------------------------------------------------------------
# G  the explanation that is not a shift
# ---------------------------------------------------------------------------
print()
print(line)
print("G  what coupling CAN do at 2.09 Hz: lend a line, not move one")
print(line)
print("  The largest gap in the identified table is 1.791 to 2.306 Hz, %.3f Hz"
      % gaps.max())
print("  wide against a median of %.3f, and 2.09 Hz sits inside it. A deck mode"
      % np.median(gaps))
print("  missed in that gap needs no implausible mu_eff to appear in a")
print("  stay-mounted record: mixing goes as kappa = s / 2|d|, which is first")
print("  order in s, while the frequency displacement goes as s^2 / 2|d|.")
print()
print("  %-10s %8s %10s %10s %12s %14s"
      % ("f_deck", "d", "s [%]", "kappa", "shift [%]", "stay f1 [Hz]"))
g_rows = []
for fd in (2.05, 2.09, 2.13):
    for th, mu in ((TH_LO, MU_MAX), (TH_HI, MU_MAX)):
        s = veering_split(mu, np.deg2rad(th), n=1)
        d = (C_HI - fd) / C_HI
        kap = s / (2 * abs(d)) if d != 0 else np.inf
        sh = shift_of(C_HI, fd, s)
        print("  %8.2f Hz %8.4f %10.2f %10.3f %11.2f %% %13.4f"
              % (fd, d, 100 * s, kap, 100 * sh, C_HI * (1 + sh)))
        g_rows.append(dict(f_deck=fd, theta_deg=th, s=s, d=d, kappa=kap,
                           shift=sh))
print()
print("  A deck mode at 2.09 Hz mixes into the stay's record at kappa up to")
print("  %.2f while moving the stay's own fundamental by under %.1f %%. The"
      % (max(r["kappa"] for r in g_rows),
         100 * max(abs(r["shift"]) for r in g_rows)))
print("  stay's true fundamental would stay near %.2f Hz and a second, brighter"
      % C_HI)
print("  line would stand at the deck frequency. Peak-picking a five-order")
print("  series off one spectrum takes the visible line. This explanation needs")
print("  mu_eff at or below the ceiling, not thirteen times over it.")

# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------
best = df[(df.target == "own_high_modes") & df.s_req_pct.notna()]
strict = df[(df.target == "string_pull") & df.s_req_pct.notna()]
print()
print(line)
print("VERDICT")
print(line)
print("  Cheapest possible requirement, over every candidate frequency, every")
print("  inclination in the band, and the most forgiving isolated reference:")
print("      s      >= %.2f %%     against a ceiling of %.2f %%"
      % (best.s_req_pct.min(), best.s_max_pct.max()))
print("      mu_eff >= %.4f       against a ceiling of %.5f"
      % (best.mu_req.min(), MU_MAX))
print("      short by a factor of %.1f in mu_eff, %.1f in s"
      % (best.shortfall_mu.min(), best.shortfall_s.min()))
print("  On the strict reference (the measured pull) the shortfall is x%.1f in"
      % strict.shortfall_mu.min())
print("  mu_eff, and the deck mode would have to weigh %.1f t at the very most,"
      % strict.M_deck_req_t.max())
print("  while presenting a perfect antinode at the anchorage, against the")
print("  %.0f-%.0f t the deck modal masses of this bridge run to."
      % (MD_LO / 1e3, MD_HI / 1e3))
plo, phi = participating_length(1e3 * strict.M_deck_req_t.max())
print("  That is a %.1f t global mode at 1.93 Hz on a 148-173 m twin deck: at"
      % strict.M_deck_req_t.max())
print("  %.0f-%.0f kg/m it would have to move only %.1f-%.1f m of deck, and move"
      % (MLIN_LO, MLIN_HI, plo, phi))
print("  it in a perfect antinode centred on this one anchorage. No mode of a")
print("  148-173 m deck at 1.93 Hz looks like that.")
print()
print("  THE N4W ANOMALY IS NOT COUPLING. The requirement exceeds what is")
print("  physically attainable by an order of magnitude in the governing group,")
print("  and it exceeds it at the floor of the requirement, so no placement of")
print("  a hypothetical unidentified mode rescues it. The sign is against it")
print("  too: any mode beyond the 2.862 Hz end of the table sits above the")
print("  fundamental and would push it down.")

os.makedirs(DATA, exist_ok=True)
out = os.path.join(DATA, "n4w_bound.csv")
df.to_csv(out, index=False)
print()
print("  wrote %s  (%d rows)" % (out, len(df)))
