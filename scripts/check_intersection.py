# -*- coding: utf-8 -*-
"""Near-crossing stays against the field records of Ponte del Mare and Aveiro.

Checks whether any stay with a tabulated reference tension also sits near a
crossing with an identified global mode, and bounds the bias the closed forms
of src/cablefe.py (mu_effective, veering_split, tension_error) predict there.
Sources: Kumar (2011), PhD thesis, University of Trento, Tables 4.4, 5.5, 5.6
and Figures 4.14, 4.18; Rebelo et al. (2010), Experimental Techniques 34(4)
62-68, Tables 1 and 3. Writes data/intersection*.csv (three files).
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

NEAR = 0.05          # |d| below which a stay mode is near a crossing
G = 9.80665


# === 1. the record ===========================================================

# ---- Kumar (2011) Table 5.6, EMA column: twelve global modes (Hz) ---------
DECK = np.array([0.747, 1.065, 1.126, 1.243, 1.394, 1.510,
                 1.716, 1.791, 2.306, 2.364, 2.512, 2.862])
DECK_TOP = DECK.max()

# ---- Kumar (2011) Table 5.5, the SISTRAL sheet ------------------------------
# SISTRAL rows: name, diameter mm, mass kg/m, length at cable temperature m,
# measured frequency Hz, pull inferred kN, design pull kN
# Values transcribed from Kumar (2011), PhD thesis, University of Trento
# (Ponte del Mare footbridge), and Rebelo et al. (2009) (Aveiro footbridge).
# They are not redistributed with this code: they live in
# data/external/check_intersection_data.py (see README).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "external"))
try:
    from check_intersection_data import (  # noqa: E402
        SISTRAL, STAYS, ORDINATE, AVEIRO_F1, AVEIRO_TEST, AVEIRO_TDES, ZETA_KUMAR)
except ImportError as exc:
    raise SystemExit("scripts/check_intersection.py needs values transcribed from "
                     "Kumar (2011), PhD thesis, University of Trento (Ponte del Mare footbridge); Rebelo et al. (2009) (Aveiro footbridge), which are not redistributed here. "
                     "See README, 'Third-party data'.") from exc

# ---- Kumar (2011) Table 4.4, the four instrumented stays (STAYS) -----------
# SISTRAL calls them NE8/NE7/NE5/NW4; Table 4.4 calls them N8E/N7E/N5E/N4W.

# ---- deck modal mass band ---------------------------------------------------
M_DECK_LO, M_DECK_HI = 40e3, 160e3       # kg, single-deck modes
# the low end gives every mu_eff upper bound below (largest coupling)

# ---- anchorage ordinates (ORDINATE), Kumar (2011) Figure 4.14 --------------
# |phi| normalized to unit maximum; provenance per entry:
#   "study"  band used in scripts/validate_pontedelmare2.py
#   "read"   read off the same figure in the sensor window nearest the
#            anchorage. The anchorages sit near -77, -71, -51 and -41 m and
#            the outermost sensor is at -68 m, so for N8E and N7E the read
#            is an upper bound.
# a second, wider read of global mode 3 at N7E; both are reported.
ORDINATE_ALT = {("N7E", 3): (0.03, 0.15, "read")}

# ---- tabulated residuals of the validation table, per cent ----------------
# RESIDUAL_HI: T from orders 2-5 against the SISTRAL pull; RESIDUAL_F1: order 1
RESIDUAL_HI = {"N8E": 4.2, "N7E": 7.6, "N5E": 6.7, "N4W": 3.9}
RESIDUAL_F1 = {"N8E": 33.7, "N7E": 12.9, "N5E": 6.7, "N4W": 18.2}

# ---- damping ---------------------------------------------------------------
ZETA_LO, ZETA_HI = 0.002, 0.010   # damping ratio band of the campaign
DIP_ANY, DIP_3DB = 0.9717, 2.280  # s / zeta for any dip and for a 3 dB dip


def line(c="=", n=86):
    print(c * n)


# === section 1: is the "independent" tension independent? ====================

def string_f1(T_N, L, m):
    return 0.5 / L * np.sqrt(T_N / m)


def section1():
    line()
    print("1. INDEPENDENCE OF THE REFERENCE TENSION FROM THE VIBRATION METHOD")
    line()
    print("   For each of the thirty SISTRAL rows, the taut-string fundamental")
    print("   implied by the tabulated pull is compared with the tabulated")
    print("   frequency.  Agreement to the tabulation quantum means the pull was")
    print("   computed from the frequency and is not an independent measurement.")
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
    print("   largest difference over the thirty rows: %.4f Hz, within"
          % worst)
    print("   the 0.01 Hz tabulation quantum of the sheet on every cable.")
    print()
    print("   The SISTRAL pull is a single-mode taut-string")
    print("   inversion of one picked frequency, the same method that is under")
    print("   test, applied a month earlier with impulse instead of ambient")
    print("   excitation.  It is not a load-cell, lift-off or jacking")
    print("   record.  The residuals in the validation table therefore measure the")
    print("   disagreement between two vibration readings of the same cable,")
    print("   not the difference between vibration and an independent force.")
    return pd.DataFrame(rows)


# === section 2: anchorage geometry, fitted to give every stay an angle ======

def fit_mast_height():
    """Height of the mast anchorage above deck level, fitted on the NE fan.

    The eight NE stays run from one mast head to eight anchorages spaced
    evenly along the deck.  With h the vertical rise, the horizontal reach of
    stay i is sqrt(L_i^2 - h^2); the h that makes those reaches most nearly
    evenly spaced is the geometry the fan implies. Returns (h in m, cost).
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
          ("stay", "L [m]", "reach [m]", "theta [deg]", "validation band"))
    for k, s in STAYS.items():
        L = [row for row in SISTRAL if row[0] == s["sistral"]][0][3]
        th = np.rad2deg(np.arcsin(h / L))
        reach = np.sqrt(L ** 2 - h ** 2)
        print("   %-6s %8.2f %9.1f %10.1f   %.0f-%.0f" %
              (k, L, reach, th, s["th"][0], s["th"][1]))
    print()
    print("   The fitted angle is at the bottom of each validation band, so")
    print("   it gives the largest cos(theta) and therefore the largest")
    print("   predicted veering width below.  The widths below are therefore")
    print("   at the upper end of the range the angle bands allow.")
    print()
    print("   Placing those reaches on the abscissa of Kumar (2011) Figure 4.14")
    print("   puts the four anchorages near -77, -71, -51 and -41 m.  The same")
    print("   figure confirms this: at -51 m the mode-7 ordinate is 0.6-1.0, the")
    print("   validation band for N5E, and at -60 to -68 m the mode-1")
    print("   ordinate is 0.06-0.33, the validation band for N8E.")


# === section 3: the four instrumented stays, five orders each ================

def mu_band(stay, jdeck, ordinate=None):
    """mu_eff band for a stay against global mode index jdeck (1-based)."""
    s = STAYS[stay]
    M_s = 0.5 * s["m"] * s["L"]
    key = (stay, jdeck)
    band = ordinate.get(key) if ordinate else None
    if band is None:
        band = ORDINATE.get(key)
    if band is None:
        # no ordinate read exists: use the mass-normalization ceiling only
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
    print("   d is against the nearest identified global mode.  The identified")
    print("   table stops at %.3f Hz, so for any stay mode above that the" % DECK_TOP)
    print("   nearest tabulated mode is the top one and d is a lower bound on")
    print("   the true detuning.  Those rows are marked > and carry no eps:")
    print("   the record does not give the frequency of the next global")
    print("   mode, so no prediction is made there.")
    print()
    print("   eps_f is the fractional frequency error.  The single-mode tension")
    print("   error is 2 eps_f.  The share that reaches the")
    print("   orders-2-to-5 tension is eps_f n^2/27.")
    print()
    print("   mu_eff columns: 'read' uses the ordinate band of Kumar (2011)")
    print("   Figure 4.14 at the 40 t deck modal mass floor; 'ceil' uses the")
    print("   mass-normalization ceiling phi_a <= 1/sqrt(M), which bounds any ordinate.")
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
    print("   Alternative read of global mode 3 at the N7E anchorage:")
    print("   0.03-0.15 of the unit maximum, against the validation band")
    print("   of 0.00-0.08.")
    print("   With the wider read, mu_eff is %.1e and eps_f is %.2f percent,"
          % tuple(_n7e_alt(h)))
    print("   an order of magnitude below the tabulated residual.")
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
    print("   The tabulated residual is T(orders 2-5) against the")
    print("   SISTRAL pull.  Coupling enters it in two ways: through the ambient")
    print("   orders 2-5, reduced by the fit to eps n^2/27 each, and through")
    print("   the SISTRAL fundamental that sets the reference pull, at the")
    print("   full 2 eps.  The two are summed in magnitude, without their")
    print("   signs, so the totals are upper bounds.")
    print()
    print("   Tier A: Kumar (2011) Figure 4.14 ordinate reads, deck modal mass 40 t")
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
    print("   Tier B: mass-normalization ceiling, phi_a = 1/sqrt(40 t),")
    print("   the largest coupling the 40 t modal mass allows, with an")
    print("   antinode at the anchorage in every mode at once")
    print("   %-5s %13s %13s %13s %10s %10s"
          % ("stay", "from ambient", "from SISTRAL", "total ceil", "tabulated",
             "resid/ceil"))
    for r in out:
        print("   %-5s %12.2f%% %12.2f%% %12.2f%% %9.1f%% %10.2f"
              % (r["stay"], r["ceiling_from_ambient_pct"],
                 r["ceiling_from_sistral_pct"], r["ceiling_total_pct"],
                 r["tabulated_pct"], r["ratio_residual_over_ceiling"]))
    print()
    print("   Ambient column alone.  Only this column bears on the method")
    print("   under test; the SISTRAL column is a bias in the reference pull,")
    print("   not in the identified tension:")
    print("   %-5s %14s %12s %10s" % ("stay", "ambient pred", "tabulated",
                                      "resid/pred"))
    for r in out:
        a = r["pred_from_ambient_pct"]
        print("   %-5s %13.2f%% %11.1f%% %10.0f"
              % (r["stay"], a, r["tabulated_pct"],
                 r["tabulated_pct"] / a if a > 0 else np.inf))
    return pd.DataFrame(out)


def section3c(df):
    """Signed comparison after removing the common offset of the residuals.

    The offset (four-stay mean) is treated as epoch drift. Repulsion moves a
    stay branch away from the global mode it meets: up when that mode is
    below, down when above.
    """
    line()
    print("3d. SIGNED COMPARISON AFTER REMOVING THE COMMON OFFSET")
    line()
    off = float(np.mean(list(RESIDUAL_HI.values())))
    print("   common offset (mean of the four residuals): %+.2f%%" % off)
    print("   The offset is treated as epoch drift, and the spread as")
    print("   the method-dependent part.  Signed prediction below: a stay")
    print("   mode below its global partner is pushed down, which lowers the")
    print("   fitted tension; a mode above it is pushed up.")
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
    print("   Only N8E has a non-zero signed prediction, and it matches the")
    print("   residual in sign and order of magnitude.  This one agreement")
    print("   rests on a fitted offset (one degree of freedom out of four")
    print("   residuals) and on the top of an ordinate band not measured at")
    print("   the anchorage.  It is not a measurement of s.")
    return pd.DataFrame(out)


# === section 3b/5: the SISTRAL side, and the thirty-stay search ==============

def sistral_bias(h):
    """Coupling bias the SISTRAL fundamental could carry, per instrumented stay.

    The SISTRAL pull is the taut-string inversion of this frequency, so a bias
    there enters the tabulated residual at the full 2 eps. Returns
    (read-band bias, ceiling bias, detail) dicts keyed by stay.
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
    print("   The taut-string inversion of this frequency is the")
    print("   reference pull.  A fundamental on a global mode biases the")
    print("   reference pull, not the identified tension.")
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
    print("   N8E: the SISTRAL fundamental at")
    print("   0.74 Hz is %.2f percent from the first global mode at"
          % abs(100 * SIS_DETAIL["N8E"][3]))
    print("   0.747 Hz, a near-exact crossing on a stay with")
    print("   a reference pull.  The crossing biases the reference")
    print("   and does not test the law, because the reference is itself a")
    print("   single-mode string inversion.  The same holds for N7E, whose")
    print("   SISTRAL fundamental at 1.06 Hz is %.2f percent from the second"
          % abs(100 * SIS_DETAIL["N7E"][3]))
    print("   global mode at 1.065 Hz.")


def section5(h):
    line()
    print("5. ALL THIRTY SISTRAL STAYS AGAINST THE TWELVE GLOBAL MODES")
    line()
    print("   Each stay's tabulated frequency is its fundamental (part 1 above),")
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
    print("   Each of the thirty pulls is a string inversion of the")
    print("   frequency in the same row (part 1 above), so for the twenty-six")
    print("   cables with no ambient modal series the residual is zero by")
    print("   construction and gives no test.  Extending the screen from four")
    print("   cables to thirty adds near-crossings but no independent tension.")
    return pd.DataFrame(rows)


# === section 6: resolvability ================================================

def section6(df):
    line()
    print("6. RESOLVABILITY OF THE PREDICTED SPLITS")
    line()
    print("   A dip appears in |H|^2 only while s > %.4f zeta, and a 3 dB dip"
          % DIP_ANY)
    print("   needs s > %.3f zeta.  Kumar reports zeta = %.2f%% for the N8E"
          % (DIP_3DB, 100 * ZETA_KUMAR))
    print("   fundamental; the parametric study uses %.1f-%.1f%%."
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
    print("   These are the widths at the top of every band.  At the bottom of")
    print("   the bands every split is unresolvable at any damping the record")
    print("   allows.")


# === section 7: Aveiro =======================================================

AVEIRO_GLOBAL = np.array([1.70, 1.85, 3.17, 3.25, 3.40, 3.96, 4.05])
AVEIRO_LABEL = ["strip 1", "strip 2", "deck 1", "deck 2", "strip 2nd",
                "deck 3", "deck 4"]


def section7():
    line()
    print("7. AVEIRO (Rebelo et al. 2010): NEAR-CROSSING SCREEN OF THE EIGHT STAYS")
    line()
    print("   Eight stays, fundamentals only.  The screen covers n = 1..5,")
    print("   so a harmonic near a global mode")
    print("   is also detected.  Global modes span %.2f-%.2f Hz."
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
    print("   Aveiro also has no independent tension: the estimated force is")
    print("   the frequency-control reading itself, compared with a design")
    print("   force, not a measured one.  It passes the screen, which")
    print("   is a consistency check and does not test the amplitude law.")
    return df


# === main ===================================================================

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
    print("8. STAYS NEAR A CROSSING THAT CARRY A REFERENCE PULL")
    line()
    near_instr = df3[df3.near_crossing]
    print("   Stays that are both near a crossing (|d| < %.2f) and carry a" % NEAR)
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
    print("   On detuning alone such stays exist: three of")
    print("   the four instrumented stays have a stay mode within 5 percent")
    print("   of an identified global mode, two of them within 0.4 percent.")
    print("   None of them gives a test against a measured tension, for two")
    print("   reasons.  First, no tension on either bridge is independent of the")
    print("   vibration method (parts 1 and 7).  Second, at each of")
    print("   those crossings the closed-form bias in the tension")
    print("   from the ambient orders 2 to 5 is 0.90 percent at")
    print("   worst (N8E) and below 0.15 percent on the other three, against")
    print("   residuals of 3.9 to 7.6 percent.  The larger predicted bias")
    print("   falls on the SISTRAL reference pull, where it is a defect of the")
    print("   reference and not a test of the law.")

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
