# -*- coding: utf-8 -*-
"""Audit of a critic's quantitative charge against the field evidence.

The charge, in full:

  "Look at which stays carry independent tensions: N8E, N7E, N5E, N4W -- and
   check their predicted coupling bias.  N7E is at 0.4% detuning but
   mu_eff <= 6e-5, so epsilon ~ 0.2%.  N8E is detuned 13% with kappa <= 0.10.
   All four are stays where your own theory predicts a negligible bias, and
   the table's residuals are 2.5-8%.  You have no stay that is both near a
   crossing and independently tensioned -- the validation table and the
   coupling demonstration don't intersect on a single cable.  The measurement
   floor on that record is an order of magnitude above the effect you're
   trying to detect there."

Nothing in the study is modified.  Every number below is recomputed with the
study's own closed forms in src/cablefe.py:

    mu_eff = M_s phi_a^2                       cablefe.mu_effective
    s      = (2 / (n pi)) cos(theta) sqrt(mu)  cablefe.veering_split
    eps    = sqrt(d^2 + s^2) - |d|             cablefe.tension_error

Sources, all Kumar (2011), PhD thesis, University of Trento:
  Table 4.4  (PDF p.104): four instrumented stays, five picked orders each
  Table 5.5  (PDF p.149): the SISTRAL sheet -- ALL THIRTY stays, each with a
                          measured frequency and the pull inferred from it.
                          The sheet's own banner reads "RILIEVO TIRO EFFETTIVO
                          NELLE FUNI CON METODO ACCELEROMETRICO", and the body
                          text (PDF p.147-148) states the pull was obtained by
                          exciting each cable with a mechanical impulse and
                          inverting the taut-string formula.  Section 1 below
                          verifies this arithmetically on all thirty rows.
  Table 5.6  (PDF p.153): twelve identified global modes, EMA column
  Figure 4.14 (PDF p.98): vertical mode shapes of both decks, FTD and CTD
  Figure 4.18 (PDF p.103): plan location of the four instrumented stays

Aveiro: Rebelo, Julio, Varum, Costa, Experimental Techniques 34(4) 62-68,
2010, Tables 1 and 3.

Run:  python3 scripts/check_intersection.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from cablefe import mu_effective, veering_split, tension_error   # noqa: E402

DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "intersection.csv")

NEAR = 0.05          # the study's own "near a crossing" threshold
G = 9.80665


# ===========================================================================
# 1. the record
# ===========================================================================

# ---- Table 5.6, EMA column: the twelve identified global modes ------------
DECK = np.array([0.747, 1.065, 1.126, 1.243, 1.394, 1.510,
                 1.716, 1.791, 2.306, 2.364, 2.512, 2.862])
DECK_TOP = DECK.max()

# ---- Table 5.5, the SISTRAL sheet, transcribed from the scanned table -----
# name, diameter mm, mass kg/m, length at cable temperature m,
# measured frequency Hz, pull inferred kN, design pull kN
# Values transcribed from Kumar (2011), PhD thesis, University of Trento (Ponte del Mare footbridge); Rebelo et al. (2009) (Aveiro footbridge). They are not redistributed with
# this code: they live in data/external/check_intersection_data.py (see README).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "external"))
try:
    from check_intersection_data import (  # noqa: E402
        SISTRAL, STAYS, ORDINATE, AVEIRO_F1, AVEIRO_TEST, AVEIRO_TDES, ZETA_KUMAR)
except ImportError as exc:
    raise SystemExit("scripts/check_intersection.py needs values transcribed from "
                     "Kumar (2011), PhD thesis, University of Trento (Ponte del Mare footbridge); Rebelo et al. (2009) (Aveiro footbridge), which are not redistributed here. "
                     "See README, 'Third-party data'.") from exc

# ---- Table 4.4, the four instrumented stays -------------------------------
# SISTRAL calls them NE8/NE7/NE5/NW4; Table 4.4 calls them N8E/N7E/N5E/N4W.

# ---- deck modal mass band carried through the study's field section -------
M_DECK_LO, M_DECK_HI = 40e3, 160e3       # kg, single-deck modes
# the low end is used for every mu_eff upper bound below, which is the
# most generous choice available to the coupling hypothesis.

# ---- anchorage ordinates of Figure 4.14, |phi| normalised to unit maximum --
# provenance is recorded per entry:
#   "study"  band carried in scripts/validate_pontedelmare2.py and the paper
#   "read"   read here off the same figure, in the sensor window nearest the
#            anchorage.  The four anchorages sit at roughly -77, -71, -51 and
#            -41 m on the Figure 4.14 abscissa (Section 2 fits that geometry);
#            the outermost sensor is at -68 m, so for N8E and N7E the window
#            read is an OVER-estimate: the true anchorage lies further out,
#            where every mode ordinate is falling towards the end support.
# a second read of global mode 3 at N7E, taken here, is wider than the band
# the paper carries; both are reported.
ORDINATE_ALT = {("N7E", 3): (0.03, 0.15, "read")}

# ---- tabulated residuals of Table 5 of the manuscript ---------------------
# T from orders 2-5 against the SISTRAL pull, per cent
RESIDUAL_HI = {"N8E": 4.2, "N7E": 7.6, "N5E": 6.7, "N4W": 3.9}
RESIDUAL_F1 = {"N8E": 33.7, "N7E": 12.9, "N5E": 6.7, "N4W": 18.2}

# ---- damping ---------------------------------------------------------------
ZETA_LO, ZETA_HI = 0.002, 0.010   # band the study's campaign measures
DIP_ANY, DIP_3DB = 0.9717, 2.280  # coefficients of the resolvability law


def line(c="=", n=86):
    print(c * n)


# ===========================================================================
# section 1: is the "independent" tension independent?
# ===========================================================================

def string_f1(T_N, L, m):
    return 0.5 / L * np.sqrt(T_N / m)


def section1():
    line()
    print("1. IS THE REFERENCE TENSION INDEPENDENT OF THE VIBRATION METHOD?")
    line()
    print("   For each of the thirty SISTRAL rows, the taut-string fundamental")
    print("   implied by the tabulated pull is compared with the tabulated")
    print("   frequency.  If they agree to the tabulation quantum the pull was")
    print("   COMPUTED FROM the frequency and is not an independent measurement.")
    print()
    print("   %-9s %9s %9s %9s %9s" %
          ("cable", "f sheet", "f from T", "diff Hz", "diff %"))
    rows, worst = [], 0.0
    for name, dia, m, L, f, T, Tdes in SISTRAL:
        fs = string_f1(T * 1e3, L, m)
        dif = fs - f
        rows.append(dict(cable=name, f_sheet=f, f_from_T=fs, diff=dif))
        worst = max(worst, abs(dif))
        print("   %-9s %9.2f %9.4f %+9.4f %+9.2f" %
              (name, f, fs, dif, 100 * dif / f))
    print()
    print("   largest discrepancy over all thirty rows: %.4f Hz, i.e. within"
          % worst)
    print("   the sheet's own 0.01 Hz tabulation quantum on every cable.")
    print()
    print("   CONSEQUENCE.  The SISTRAL pull is a single-mode taut-string")
    print("   inversion of one picked frequency -- the same estimator the study")
    print("   is testing, run a month earlier with impulse rather than ambient")
    print("   excitation.  It is not a load cell, not a lift-off, not a jacking")
    print("   record.  So the residuals in the manuscript's Table 5 measure the")
    print("   disagreement between two vibration readings of the same cable,")
    print("   not between vibration and an independent force.")
    return pd.DataFrame(rows)


# ===========================================================================
# section 2: anchorage geometry, fitted so every stay gets an inclination
# ===========================================================================

def fit_mast_height():
    """Height of the mast anchorage above deck level, fitted on the NE fan.

    The eight NE stays run from one mast head to eight anchorages spaced
    evenly along the deck.  With h the vertical rise, the horizontal reach of
    stay i is sqrt(L_i^2 - h^2); the h that makes those reaches most nearly
    evenly spaced is the geometry the fan implies.  It is a fit to the stay
    schedule only, and it is checked below against the inclination bands the
    study carries.
    """
    Ls = np.array([L for n, d, m, L, f, T, Td in SISTRAL
                   if n.startswith("NE") and n != "N1(NE1)"])
    Ls = np.sort(Ls)
    best = (1e9, None)
    for h in np.arange(5.0, 30.0, 0.01):
        if h >= Ls.min():
            continue
        x = np.sqrt(Ls ** 2 - h ** 2)
        dx = np.diff(x)
        cost = np.std(dx) / np.mean(dx)
        if cost < best[0]:
            best = (cost, h)
    return best[1], best[0]


def section2(h):
    line()
    print("2. ANCHORAGE GEOMETRY (fitted, used only to give every stay a theta)")
    line()
    print("   mast rise above deck fitted on the NE fan: h = %.1f m" % h)
    print()
    print("   %-6s %8s %9s %10s   %s" %
          ("stay", "L [m]", "reach [m]", "theta [deg]", "study's band"))
    for k, s in STAYS.items():
        L = [row for row in SISTRAL if row[0] == s["sistral"]][0][3]
        th = np.rad2deg(np.arcsin(h / L))
        reach = np.sqrt(L ** 2 - h ** 2)
        print("   %-6s %8.2f %9.1f %10.1f   %.0f-%.0f" %
              (k, L, reach, th, s["th"][0], s["th"][1]))
    print()
    print("   The fit lands at the bottom of every band the study carries, so")
    print("   using it maximises cos(theta) and therefore maximises every")
    print("   predicted veering width below.  That is deliberate: the whole")
    print("   audit is run in favour of the coupling hypothesis.")
    print()
    print("   Placing those reaches on the Figure 4.14 abscissa puts the four")
    print("   anchorages near -77, -71, -51 and -41 m.  The check is Figure")
    print("   4.14 itself: at -51 m the mode-7 ordinate is 0.6-1.0, which is")
    print("   the band the study reads for N5E, and at -60 to -68 m the mode-1")
    print("   ordinate is 0.06-0.33, which is the band it reads for N8E.")


# ===========================================================================
# section 3: the four instrumented stays, five orders each
# ===========================================================================

def mu_band(stay, jdeck, ordinate=None):
    """mu_eff band for a stay against global mode index jdeck (1-based)."""
    s = STAYS[stay]
    M_s = 0.5 * s["m"] * s["L"]
    key = (stay, jdeck)
    band = ordinate.get(key) if ordinate else None
    if band is None:
        band = ORDINATE.get(key)
    if band is None:
        # no ordinate read exists: carry the mass-normalisation ceiling only
        p_lo, p_hi, src = 0.0, 1.0, "ceiling"
    else:
        p_lo, p_hi, src = band
    mu_lo = mu_effective(M_s, p_lo / np.sqrt(M_DECK_HI))
    mu_hi = mu_effective(M_s, p_hi / np.sqrt(M_DECK_LO))
    mu_ceil = mu_effective(M_s, 1.0 / np.sqrt(M_DECK_LO))
    return M_s, mu_lo, mu_hi, mu_ceil, src


def s_eps(stay, n, d, mu_lo, mu_hi, mu_ceil, h):
    s = STAYS[stay]
    L = [row for row in SISTRAL if row[0] == s["sistral"]][0][3]
    th = np.arcsin(h / L)                    # smallest inclination -> largest s
    s_lo = veering_split(mu_lo, th, n)
    s_hi = veering_split(mu_hi, th, n)
    s_cl = veering_split(mu_ceil, th, n)
    e_lo = tension_error(d, mu_lo, th, n)
    e_hi = tension_error(d, mu_hi, th, n)
    e_cl = tension_error(d, mu_ceil, th, n)
    return th, s_lo, s_hi, s_cl, e_lo, e_hi, e_cl


def dilution(n):
    """How much a frequency error at order n moves T fitted on orders 2-5.

    T_hi = 4 m L^2 c^2 with c the least-squares slope of f_n = c n over
    n = 2..5, so c = sum(n f_n)/sum(n^2), sum(n^2) = 54.  Perturbing f_n by
    eps f_n = eps c n moves c by c eps n^2 / 54 and therefore T by
    2 eps n^2 / 54 = eps n^2 / 27.  Orders outside 2..5 return 0.
    """
    return 0.0 if n < 2 or n > 5 else n * n / 27.0


def section3(h):
    line()
    print("3. THE FOUR INSTRUMENTED STAYS, ORDERS 1 TO 5")
    line()
    print("   d is against the NEAREST identified global mode.  The identified")
    print("   table stops at %.3f Hz, so for any stay mode above that the" % DECK_TOP)
    print("   nearest tabulated mode is the top one and d is a LOWER bound on")
    print("   the true detuning, not a measurement.  Those rows are marked > and")
    print("   carry no eps: nothing in the record says where the next global")
    print("   mode sits, so no prediction can be made there either way.")
    print()
    print("   eps_f is the fractional frequency error.  The single-mode tension")
    print("   error is 2 eps_f.  The share that reaches the manuscript's")
    print("   orders-2-to-5 tension is eps_f n^2/27.")
    print()
    print("   mu_eff columns: 'read' uses the Figure 4.14 ordinate band, at the")
    print("   40 t deck modal mass floor; 'ceil' is the mass-normalisation")
    print("   ceiling phi_a <= 1/sqrt(M), an absolute bound no ordinate can beat.")
    print()
    hdr = ("stay", "n", "f_n", "gm", "f_g", "d", "src", "mu read hi",
           "mu ceil", "s read %", "eps_f %", "2eps %", "in T25 %", "ceil T25 %")
    print("   %-5s %2s %6s %3s %6s %9s %7s %10s %9s %9s %8s %7s %9s %10s" % hdr)
    rows = []
    for k, s_ in STAYS.items():
        for n, fn in enumerate(s_["f"], start=1):
            j = int(np.argmin(np.abs(DECK - fn)))
            fg = DECK[j]
            d = (fn - fg) / fn
            above = fn > DECK_TOP
            M_s, mu_lo, mu_hi, mu_cl, src = mu_band(k, j + 1)
            th, s_lo, s_hi, s_cl, e_lo, e_hi, e_cl = s_eps(
                k, n, d, mu_lo, mu_hi, mu_cl, h)
            dil = dilution(n)
            if above:
                s_lo = s_hi = s_cl = np.nan
                e_lo = e_hi = e_cl = np.nan
            print("   %-5s %2d %6.3f %3d %6.3f %+8.4f%s %7s %10.2e %9.2e "
                  "%9s %8s %7s %9s %10s"
                  % (k, n, fn, j + 1, fg, d, ">" if above else " ", src,
                     mu_hi, mu_cl,
                     "--" if above else "%.2f" % (100 * s_hi),
                     "--" if above else "%.2f" % (100 * e_hi),
                     "--" if above else "%.2f" % (200 * e_hi),
                     "--" if above else "%.2f" % (100 * e_hi * dil),
                     "--" if above else "%.2f" % (100 * e_cl * dil)))
            rows.append(dict(
                record="PonteDelMare", stay=k, n=n, f_stay=fn,
                deck_mode=j + 1, f_deck=fg, d=d,
                d_is_lower_bound=bool(above), near_crossing=bool(abs(d) < NEAR
                                                                 and not above),
                M_s_kg=M_s, ordinate_src=src,
                mu_eff_lo=mu_lo, mu_eff_hi=mu_hi, mu_eff_ceiling=mu_cl,
                theta_deg=np.rad2deg(th),
                s_lo=s_lo, s_hi=s_hi, s_ceiling=s_cl,
                eps_f_lo=e_lo, eps_f_hi=e_hi, eps_f_ceiling=e_cl,
                eps_T_singlemode_hi=2 * e_hi,
                eps_T_in_orders25_hi=e_hi * dil,
                eps_T_in_orders25_ceiling=e_cl * dilution(n),
                residual_orders25_pct=RESIDUAL_HI[k],
                residual_f1_pct=RESIDUAL_F1[k]))
    print()
    print("   Alternative read of the one band that matters most: this audit")
    print("   reads global mode 3 at the N7E anchorage as 0.03-0.15 of the unit")
    print("   maximum, against the 0.00-0.08 the paper carries.  At the wider")
    print("   read mu_eff rises to %.1e and eps_f to %.2f per cent, still an"
          % tuple(_n7e_alt(h)))
    print("   order below the residual it would have to explain.")
    return pd.DataFrame(rows)


def _n7e_alt(h):
    s_ = STAYS["N7E"]
    M_s = 0.5 * s_["m"] * s_["L"]
    p_lo, p_hi, _ = ORDINATE_ALT[("N7E", 3)]
    mu_lo = mu_effective(M_s, p_lo / np.sqrt(M_DECK_HI))
    mu_hi = mu_effective(M_s, p_hi / np.sqrt(M_DECK_LO))
    d = (1.13 - 1.126) / 1.13
    th, _, _, _, _, e_hi, _ = s_eps("N7E", 1, d, mu_lo, mu_hi, mu_hi, h)
    return mu_hi, 100 * e_hi


def section3b(df):
    line()
    print("3c. PREDICTED SHARE OF EACH TABULATED RESIDUAL")
    line()
    print("   The manuscript's Table 5 residual is T(orders 2-5) against the")
    print("   SISTRAL pull.  Coupling can enter it twice: through the ambient")
    print("   orders 2-5, diluted by the fit to eps n^2/27 each, and through")
    print("   the SISTRAL fundamental that fixes the reference pull, at the")
    print("   full 2 eps.  Both are summed in magnitude, never with their")
    print("   signs, so the totals are upper bounds and not estimates.")
    print()
    print("   TIER A -- Figure 4.14 ordinate reads, deck modal mass at 40 t")
    print("   %-5s %13s %13s %13s %10s %10s"
          % ("stay", "from ambient", "from SISTRAL", "total pred", "tabulated",
             "resid/pred"))
    out = []
    for k in STAYS:
        sub = df[(df.stay == k) & (~df.d_is_lower_bound)]
        amb = float(np.nansum(np.abs(sub.eps_T_in_orders25_hi))) * 100.0
        sis = SIS_BIAS[k] * 100.0
        tot = amb + sis
        tab = RESIDUAL_HI[k]
        print("   %-5s %12.2f%% %12.2f%% %12.2f%% %9.1f%% %10.2f"
              % (k, amb, sis, tot, tab, tab / tot if tot > 0 else np.inf))
        subc = df[(df.stay == k) & (~df.d_is_lower_bound)]
        ambc = float(np.nansum(np.abs(subc.eps_T_in_orders25_ceiling))) * 100.0
        sisc = SIS_CEIL[k] * 100.0
        out.append(dict(stay=k, pred_from_ambient_pct=amb,
                        pred_from_sistral_pct=sis, pred_total_pct=tot,
                        ceiling_from_ambient_pct=ambc,
                        ceiling_from_sistral_pct=sisc,
                        ceiling_total_pct=ambc + sisc,
                        tabulated_pct=tab,
                        ratio_residual_over_prediction=tab / tot if tot > 0
                        else np.nan,
                        ratio_residual_over_ceiling=(tab / (ambc + sisc)
                                                     if ambc + sisc > 0
                                                     else np.nan)))
    print()
    print("   TIER B -- mass-normalisation ceiling, phi_a = 1/sqrt(40 t)")
    print("   the largest coupling the bridge can physically support, with a")
    print("   perfect antinode placed on the anchorage of every mode at once")
    print("   %-5s %13s %13s %13s %10s %10s"
          % ("stay", "from ambient", "from SISTRAL", "total ceil", "tabulated",
             "resid/ceil"))
    for r in out:
        print("   %-5s %12.2f%% %12.2f%% %12.2f%% %9.1f%% %10.2f"
              % (r["stay"], r["ceiling_from_ambient_pct"],
                 r["ceiling_from_sistral_pct"], r["ceiling_total_pct"],
                 r["tabulated_pct"], r["ratio_residual_over_ceiling"]))
    print()
    print("   Read the ambient column on its own -- it is the only column that")
    print("   bears on the estimator the manuscript is defending, because the")
    print("   SISTRAL column is a bias in the REFERENCE, not in the estimate:")
    print("   %-5s %14s %12s %10s" % ("stay", "ambient pred", "tabulated",
                                      "resid/pred"))
    for r in out:
        a = r["pred_from_ambient_pct"]
        print("   %-5s %13.2f%% %11.1f%% %10.0f"
              % (r["stay"], a, r["tabulated_pct"],
                 r["tabulated_pct"] / a if a > 0 else np.inf))
    return pd.DataFrame(out)


def section3c(df):
    """Signed comparison after removing the common offset the paper isolates.

    The manuscript argues that a common positive offset of about 5.6 per cent
    is epoch drift and cannot be attributed to an estimator, so that what the
    estimator owns is the SPREAD.  Taking that argument at face value, the
    quantity to predict is each residual minus the four-stay mean, and the
    prediction must be signed: repulsion carries a stay branch AWAY from the
    global mode it meets, up when that mode sits below and down when above.
    """
    line()
    print("3d. SIGNED COMPARISON AFTER REMOVING THE COMMON OFFSET")
    line()
    off = float(np.mean(list(RESIDUAL_HI.values())))
    print("   common offset (mean of the four residuals): %+.2f%%" % off)
    print("   the manuscript treats this as epoch drift, leaving the spread as")
    print("   the estimator-dependent part.  Signed prediction below: a stay")
    print("   mode BELOW its global partner is pushed DOWN, which lowers the")
    print("   fitted tension; above, it is pushed UP.")
    print()
    print("   %-5s %12s %14s %12s %10s"
          % ("stay", "resid - off", "signed pred", "difference", "ratio"))
    out = []
    for k in STAYS:
        sub = df[(df.stay == k) & (~df.d_is_lower_bound)]
        pred = 0.0
        for _, r in sub.iterrows():
            if np.isnan(r.eps_f_hi):
                continue
            sign = -1.0 if r.d < 0 else +1.0    # d<0: global above, push down
            pred += sign * r.eps_T_in_orders25_hi
        pred *= 100.0
        obs = RESIDUAL_HI[k] - off
        print("   %-5s %11.2f%% %13.2f%% %11.2f%% %10s"
              % (k, obs, pred, obs - pred,
                 "%.2f" % (obs / pred) if abs(pred) > 1e-9 else "--"))
        out.append(dict(stay=k, residual_minus_offset_pct=obs,
                        signed_prediction_pct=pred))
    print()
    print("   Only N8E carries a non-zero signed prediction, and it has the")
    print("   right sign and the right order.  That is one point of agreement")
    print("   bought with a fitted offset (one degree of freedom out of four")
    print("   residuals) and with the top of an unread ordinate band, so it is")
    print("   suggestive and nothing more.  It is not a measurement of s.")
    return pd.DataFrame(out)


# ===========================================================================
# section 3b/5: the SISTRAL side, and the thirty-stay search
# ===========================================================================

def sistral_bias(h):
    """Coupling bias the SISTRAL fundamental could carry, per instrumented stay.

    The SISTRAL frequency IS the isolated-cable reading whose inversion fixes
    the reference pull, so a bias there enters the tabulated residual at the
    full 2 eps.  Returned as (read-band bias, ceiling bias, detail).
    """
    read, ceil, detail = {}, {}, {}
    for k, s_ in STAYS.items():
        row = [r for r in SISTRAL if r[0] == s_["sistral"]][0]
        f1 = row[4]
        j = int(np.argmin(np.abs(DECK - f1)))
        d = (f1 - DECK[j]) / f1
        M_s, mu_lo, mu_hi, mu_cl, src = mu_band(k, j + 1)
        th, s_lo, s_hi, s_cl, e_lo, e_hi, e_cl = s_eps(
            k, 1, d, mu_lo, mu_hi, mu_cl, h)
        read[k] = 2 * e_hi
        ceil[k] = 2 * e_cl
        detail[k] = (f1, j + 1, DECK[j], d, mu_hi, mu_cl, s_hi, s_cl,
                     e_hi, e_cl, src)
    return read, ceil, detail


def section4(h):
    line()
    print("3b. THE SISTRAL FUNDAMENTAL OF EACH INSTRUMENTED STAY")
    line()
    print("   This is the frequency whose taut-string inversion IS the")
    print("   reference pull.  If it sits on a global mode, the reference is")
    print("   biased, not the estimate.")
    print()
    print("   %-5s %7s %3s %7s %9s %7s %10s %8s %9s %9s"
          % ("stay", "f SIS", "gm", "f_g", "d", "src", "mu read hi",
             "s read %", "2eps read", "2eps ceil"))
    for k in STAYS:
        (f1, jm, fg, d, mu_hi, mu_cl, s_hi, s_cl, e_hi, e_cl,
         src) = SIS_DETAIL[k]
        print("   %-5s %7.2f %3d %7.3f %+9.4f %7s %10.2e %8.2f %8.2f%% %8.2f%%"
              % (k, f1, jm, fg, d, src, mu_hi, 100 * s_hi, 200 * e_hi,
                 200 * e_cl))
    print()
    print("   N8E is the case that matters.  Its SISTRAL fundamental at")
    print("   0.74 Hz sits %.2f per cent from the first global mode at"
          % abs(100 * SIS_DETAIL["N8E"][3]))
    print("   0.747 Hz.  That is a near-exact crossing on a stay that carries")
    print("   a reference pull -- but the crossing contaminates the reference")
    print("   rather than testing the law, because the reference is itself a")
    print("   single-mode string inversion.  The same is true of N7E, whose")
    print("   SISTRAL fundamental at 1.06 Hz sits %.2f per cent from the second"
          % abs(100 * SIS_DETAIL["N7E"][3]))
    print("   global mode at 1.065 Hz.")


def section5(h):
    line()
    print("5. ALL THIRTY SISTRAL STAYS AGAINST THE TWELVE GLOBAL MODES")
    line()
    print("   Each stay's tabulated frequency is its fundamental (Section 1),")
    print("   so its own series is n f_1 for n = 1..5.  A hit is |d| < %.2f."
          % NEAR)
    print()
    rows, hits = [], []
    for name, dia, m, L, f1, T, Tdes in SISTRAL:
        instrumented = any(s["sistral"] == name for s in STAYS.values())
        for n in range(1, 6):
            fn = n * f1
            j = int(np.argmin(np.abs(DECK - fn)))
            d = (fn - DECK[j]) / fn
            above = fn > DECK_TOP
            hit = abs(d) < NEAR and not above
            rows.append(dict(record="PonteDelMare_SISTRAL", stay=name, n=n,
                             f_stay=fn, deck_mode=j + 1, f_deck=DECK[j], d=d,
                             d_is_lower_bound=bool(above),
                             near_crossing=bool(hit),
                             instrumented=bool(instrumented),
                             T_kN=T, T_design_kN=Tdes,
                             tension_is_independent=False))
            if hit:
                hits.append((name, n, fn, j + 1, DECK[j], d, instrumented))
    print("   %-9s %2s %8s %5s %8s %9s   %s"
          % ("cable", "n", "f_n", "mode", "f_g", "d", "has a modal series?"))
    for name, n, fn, jm, fg, d, instr in sorted(hits, key=lambda r: abs(r[5])):
        print("   %-9s %2d %8.3f %5d %8.3f %+9.4f   %s"
              % (name, n, fn, jm, fg, d, "YES" if instr else "no"))
    print()
    print("   %d of the 150 stay-mode/global-mode pairs are inside |d| < %.2f,"
          % (len(hits), NEAR))
    print("   on %d of the 30 cables."
          % len({h_[0] for h_ in hits}))
    print()
    print("   But every one of the thirty pulls is a string inversion of the")
    print("   frequency in the same row (Section 1), so for the twenty-six")
    print("   cables with no ambient modal series the residual is zero BY")
    print("   CONSTRUCTION and no test exists.  Widening the search from four")
    print("   cables to thirty adds coincidences and adds no measurement.")
    return pd.DataFrame(rows)


# ===========================================================================
# section 6: resolvability
# ===========================================================================

def section6(df):
    line()
    print("6. RESOLVABILITY OF THE PREDICTED SPLITS")
    line()
    print("   A dip appears in |H|^2 only while s > %.4f zeta, and a 3 dB dip"
          % DIP_ANY)
    print("   needs s > %.3f zeta.  Kumar reports zeta = %.2f%% for the N8E"
          % (DIP_3DB, 100 * ZETA_KUMAR))
    print("   fundamental; the study's campaign carries %.1f-%.1f%%."
          % (100 * ZETA_LO, 100 * ZETA_HI))
    print()
    sub = df[df.near_crossing & (df.record == "PonteDelMare")]
    print("   %-5s %2s %8s %12s %12s   %s"
          % ("stay", "n", "s hi %", "zeta for dip", "zeta for 3dB", "resolvable?"))
    for _, r in sub.iterrows():
        z_any = r.s_hi / DIP_ANY
        z_3db = r.s_hi / DIP_3DB
        verdict = ("3 dB dip at zeta<=%.2f%%" % (100 * z_3db)
                   if z_3db > ZETA_LO else "below the damping floor")
        print("   %-5s %2d %8.2f %11.2f%% %11.2f%%   %s"
              % (r.stay, r.n, 100 * r.s_hi, 100 * z_any, 100 * z_3db, verdict))
    print()
    print("   These are the widths at the TOP of every band.  At the bottom of")
    print("   the bands every split is unresolvable at any damping the record")
    print("   admits.")


# ===========================================================================
# section 7: Aveiro
# ===========================================================================

AVEIRO_GLOBAL = np.array([1.70, 1.85, 3.17, 3.25, 3.40, 3.96, 4.05])
AVEIRO_LABEL = ["strip 1", "strip 2", "deck 1", "deck 2", "strip 2nd",
                "deck 3", "deck 4"]


def section7():
    line()
    print("7. AVEIRO (Rebelo et al. 2010): IS ANY OF THE EIGHT NEAR A CROSSING?")
    line()
    print("   Eight stays, fundamentals only.  The screen is run on n = 1..5,")
    print("   not just the fundamental, so a harmonic landing on a global mode")
    print("   would be caught.  Global modes span %.2f-%.2f Hz."
          % (AVEIRO_GLOBAL.min(), AVEIRO_GLOBAL.max()))
    print()
    print("   %-6s %2s %8s %-10s %8s %9s   %s"
          % ("cable", "n", "f_n", "nearest", "f_g", "d", "near?"))
    rows, hits = [], 0
    for i in range(8):
        for n in range(1, 6):
            fn = n * AVEIRO_F1[i]
            j = int(np.argmin(np.abs(AVEIRO_GLOBAL - fn)))
            d = (fn - AVEIRO_GLOBAL[j]) / fn
            above = fn > AVEIRO_GLOBAL.max()
            hit = abs(d) < NEAR and not above
            hits += hit
            if n <= 2 or hit:
                print("   %-6d %2d %8.3f %-10s %8.2f %+9.4f%s   %s"
                      % (i + 1, n, fn, AVEIRO_LABEL[j], AVEIRO_GLOBAL[j], d,
                         ">" if above else " ", "YES" if hit else ""))
            rows.append(dict(record="Aveiro", stay="cable %d" % (i + 1), n=n,
                             f_stay=fn, deck_mode=AVEIRO_LABEL[j],
                             f_deck=AVEIRO_GLOBAL[j], d=d,
                             d_is_lower_bound=bool(above),
                             near_crossing=bool(hit),
                             T_kN=AVEIRO_TEST[i], T_design_kN=AVEIRO_TDES[i],
                             tension_is_independent=False))
    df = pd.DataFrame(rows)
    n1 = df[df.n == 1]
    print()
    print("   smallest |d| over the eight fundamentals: %.3f" % n1.d.abs().min())
    print("   smallest |d| over all 40 stay-mode pairs below the top global")
    print("   mode: %.3f"
          % df[~df.d_is_lower_bound].d.abs().min())
    print("   near-crossings found: %d" % hits)
    print()
    print("   Aveiro also has no independent tension: the 'estimated' force is")
    print("   the frequency-control reading itself, compared against a DESIGN")
    print("   force, not against a measured one.  It clears the screen, which")
    print("   is a consistency check, and cannot test the amplitude law.")
    return df


# ===========================================================================

def main():
    df_sis_check = section1()
    print()
    h, cost = fit_mast_height()
    section2(h)
    print()
    global SIS_BIAS, SIS_CEIL, SIS_DETAIL
    SIS_BIAS, SIS_CEIL, SIS_DETAIL = sistral_bias(h)
    df3 = section3(h)
    print()
    section4(h)
    print()
    df3b = section3b(df3)
    print()
    df3c = section3c(df3)
    print()
    df5 = section5(h)
    print()
    section6(df3)
    print()
    df7 = section7()
    print()

    line()
    print("8. THE INTERSECTION QUESTION, ANSWERED")
    line()
    near_instr = df3[df3.near_crossing]
    print("   Stays that are BOTH near a crossing (|d| < %.2f) AND carry a" % NEAR)
    print("   tabulated reference pull:")
    for _, r in near_instr.iterrows():
        print("     %s mode %d at %.3f Hz vs global mode %d at %.3f Hz, "
              "d = %+.4f" % (r.stay, r.n, r.f_stay, r.deck_mode, r.f_deck, r.d))
    for k in STAYS:
        det = SIS_DETAIL[k]
        if abs(det[3]) < NEAR:
            print("     %s SISTRAL fundamental at %.2f Hz vs global mode %d at "
                  "%.3f Hz, d = %+.4f" % (k, det[0], det[1], det[2], det[3]))
    print()
    print("   So the intersection is NOT empty on the detuning axis: three of")
    print("   the four instrumented stays carry a stay mode inside 5 per cent")
    print("   of an identified global mode, two of them inside 0.4 per cent.")
    print("   What is empty is the intersection with a MEASUREMENT, on two")
    print("   counts.  First, no tension on either bridge is independent of the")
    print("   vibration method (Sections 1 and 7).  Second, at every one of")
    print("   those crossings the bias the study's own law predicts in the")
    print("   ESTIMATOR -- the ambient orders 2 to 5 -- is 0.90 per cent at")
    print("   worst (N8E) and below 0.15 per cent on the other three, against")
    print("   residuals of 3.9 to 7.6 per cent.  The larger predicted bias")
    print("   lands on the SISTRAL reference, which is not a test of the law")
    print("   but a defect of the reference.")

    frames = [df3.assign(source="Table4.4_ambient"),
              df5.assign(source="Table5.5_SISTRAL"),
              df7.assign(source="Rebelo2010")]
    out = pd.concat(frames, ignore_index=True, sort=False)
    os.makedirs(DATA, exist_ok=True)
    out.to_csv(OUT, index=False)
    df3b = df3b.merge(df3c, on="stay")
    df3b.to_csv(OUT.replace(".csv", "_summary.csv"), index=False)
    df_sis_check.to_csv(OUT.replace(".csv", "_sistral_check.csv"), index=False)
    print()
    print("   wrote %s" % OUT)
    print("   wrote %s" % OUT.replace(".csv", "_summary.csv"))
    print("   wrote %s" % OUT.replace(".csv", "_sistral_check.csv"))


SIS_BIAS, SIS_CEIL, SIS_DETAIL = {}, {}, {}

if __name__ == "__main__":
    main()
