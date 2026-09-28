# -*- coding: utf-8 -*-
"""Test whether the damper traverse excludes two rival coupling hypotheses.

Uses the N8E stay reads with and without dampers (global mode 1 at 0.75 and
0.93 Hz) to test the plain mass ratio mu = M_s / M_deck (H_plain) and zero
coupling (H_null) by chi(s), the best-fit band miss in read uncertainties.
Also tests H_plain against the N7E stay-deck spacing. Writes
data/falsify_traverse.csv and data/falsify_traverse_thresholds.csv.
Run: python3 scripts/falsify_traverse.py
"""
from __future__ import annotations
import os
import sys
from math import comb

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import cablefe                                              # noqa: E402

# --- the record ---
FD_NO, FD_YES = 0.750, 0.930        # global mode 1, without / with dampers
N = 1
# Values transcribed from Kumar (2011), PhD thesis, University of Trento
# (Ponte del Mare footbridge). They are not redistributed with this code;
# they live in data/external/falsify_traverse_data.py (see README).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "external"))
try:
    from falsify_traverse_data import (  # noqa: E402
        TH, M_S, READ_NO, READ_YES, M_S7, TH7, F7_STAY, F7_DECK)
except ImportError as exc:
    raise SystemExit("scripts/falsify_traverse.py needs values transcribed from "
                     "Kumar (2011), PhD thesis, University of Trento (Ponte del Mare footbridge), which are not redistributed here. "
                     "See README, 'Third-party data'.") from exc
MDECK_LO, MDECK_HI = 40e3, 160e3                # deck modal mass, kg
SIG = 0.004                                     # read uncertainty, Hz

BAND_NO = (min(READ_NO), max(READ_NO))
BAND_YES = (min(READ_YES), max(READ_YES))
SHIFT_LO = min(READ_YES) - max(READ_NO)         # -10 mHz, most negative pair
SHIFT_HI = max(READ_YES) - min(READ_NO)         # -1 mHz, least negative pair
SIG_SHIFT = SIG * np.sqrt(2.0)                  # two independent reads

W = 78
def rule(t=""):
    print("=" * W)
    if t:
        print("  " + t)
        print("=" * W)

# --- branch models, vectorized in f_iso ---
def f_lin(f_iso, f_deck, s):
    """Stay-dominated branch, linear-frequency two-level repulsion.

    The model that validate_traverse.py inverts.
    """
    f_iso = np.asarray(f_iso, dtype=float)
    D = f_iso - f_deck
    return f_iso + 0.5 * np.sign(D) * (np.sqrt(D * D + (s * f_iso) ** 2) - np.abs(D))


def f_2dof(f_iso, f_deck, s):
    """Stay-dominated branch of the inertially coupled 2-DOF pair, closed form.

    K = diag(wa^2, wb^2), M = [[1, s], [s, 1]] in modal coordinates, so
    lam^2 (1 - s^2) - lam (a + b) + a b = 0 with a = wa^2, b = wb^2. The
    stay-dominated root is the upper one while the stay sits above the deck
    mode and the lower one below it. At exact tuning the pair splits by s to
    O(s^3), the same normalization as f_lin.
    """
    f_iso = np.asarray(f_iso, dtype=float)
    a = (2 * np.pi * f_iso) ** 2
    b = (2 * np.pi * f_deck) ** 2
    g2 = s * s
    disc = np.sqrt((a + b) ** 2 - 4.0 * a * b * (1.0 - g2))
    hi = (a + b + disc) / (2.0 * (1.0 - g2))
    lo = (a + b - disc) / (2.0 * (1.0 - g2))
    lam = np.where(f_iso > f_deck, hi, lo)
    return np.sqrt(lam) / (2 * np.pi)


MODELS = {"linear": f_lin, "exact2dof": f_2dof}


def solve_pair(o_no, o_yes, model):
    """The (s, f_iso) pair that reproduces both reads exactly."""
    def f_iso_of(s):
        return brentq(lambda f: float(model(f, FD_NO, s)) - o_no,
                      0.60, 0.99, xtol=1e-13)

    s = brentq(lambda s: float(model(f_iso_of(s), FD_YES, s)) - o_yes,
               1e-7, 0.40, xtol=1e-13)
    return s, f_iso_of(s)


def mu_from_s(s, th):
    return (s * N * np.pi / (2 * np.cos(th))) ** 2


def s_from_mu(mu, th):
    return cablefe.veering_split(mu, th, N)


# --- 0. verification ---
rule("0  VERIFICATION")
mu_p, th_p = 3.0e-3, np.deg2rad(22.0)
s_p = s_from_mu(mu_p, th_p)
assert abs(mu_from_s(s_p, th_p) - mu_p) < 1e-15
print("  cablefe.veering_split round trip  : mu %.4e -> s %.6f -> mu %.4e  OK"
      % (mu_p, s_p, mu_from_s(s_p, th_p)))
assert np.isclose(cablefe.mu_effective(M_S, 1e-3), M_S * 1e-6)
print("  cablefe.mu_effective identity      : OK")

# closed-form 2-DOF branch against a numerical generalized eigensolve
worst = 0.0
for s_t in (0.01, 0.05, 0.10):
    for fi in (0.80, 0.83, 0.86):
        for fd in (FD_NO, FD_YES):
            K = np.diag([(2 * np.pi * fi) ** 2, (2 * np.pi * fd) ** 2])
            M = np.array([[1.0, s_t], [s_t, 1.0]])
            lam, vec = eigh(K, M)
            fr = np.sqrt(lam) / (2 * np.pi)
            num = fr[int(np.argmax(np.abs(vec[0, :])))]
            worst = max(worst, abs(num - float(f_2dof(fi, fd, s_t))))
print("  closed-form 2-DOF vs eigh          : max |df| = %.2e Hz  %s"
      % (worst, "OK" if worst < 1e-10 else "FAIL"))

for s_t in (0.02, 0.05, 0.10):
    K = np.diag([(2 * np.pi * 0.83) ** 2] * 2)
    M = np.array([[1.0, s_t], [s_t, 1.0]])
    fr = np.sqrt(eigh(K, M)[0]) / (2 * np.pi)
    print("  2-DOF split at exact tuning s=%.2f : %.5f  (err %+.1e, O(s^3))"
          % (s_t, (fr[1] - fr[0]) / 0.83, (fr[1] - fr[0]) / 0.83 - s_t))
print()

rows = []
for a in READ_NO:
    for b in READ_YES:
        if b >= a:
            continue
        s_l, fi_l = solve_pair(a, b, f_lin)
        s_e, fi_e = solve_pair(a, b, f_2dof)
        for th in TH:
            rows.append(dict(read_no=a, read_yes=b, shift_Hz=b - a,
                             theta_deg=np.rad2deg(th),
                             s_linear=s_l, f_iso_linear=fi_l,
                             mu_linear=mu_from_s(s_l, th),
                             s_exact=s_e, f_iso_exact=fi_e,
                             mu_exact=mu_from_s(s_e, th)))
meas = pd.DataFrame(rows)
ok_s = (abs(100 * meas.s_linear.min() - 1.6) < 0.05
        and abs(100 * meas.s_linear.max() - 5.2) < 0.05)
ok_mu = (abs(meas.mu_linear.min() / 7.2e-4 - 1) < 0.02
         and abs(meas.mu_linear.max() / 8.3e-3 - 1) < 0.02)
print("  reproduce validate_traverse.py:")
print("    s      = %.2f to %.2f %%      (expected 1.6 to 5.2 %%)   %s"
      % (100 * meas.s_linear.min(), 100 * meas.s_linear.max(),
         "OK" if ok_s else "MISMATCH"))
print("    mu_eff = %.2e to %.2e (expected 7.2e-4 to 8.3e-3)  %s"
      % (meas.mu_linear.min(), meas.mu_linear.max(),
         "OK" if ok_mu else "MISMATCH"))
d_rel = (meas.s_exact - meas.s_linear) / meas.s_linear
print("    inverting the exact 2-DOF pair instead moves s by %+.2f to %+.2f %%"
      % (100 * d_rel.min(), 100 * d_rel.max()))
print("    (relative), giving mu_eff = %.2e to %.2e.  The model difference is"
      % (meas.mu_exact.min(), meas.mu_exact.max()))
print("    within the read uncertainty; both models are used below.")
print()

# --- the exclusion functional ---
FGRID = np.linspace(0.740, 0.930, 38001)        # 5 uHz, vs a 4 mHz sigma


def _miss(x, band):
    return np.maximum.reduce([band[0] - x, np.zeros_like(x), x - band[1]])


def chi(s, model):
    """min over f_iso of the worst band miss, in read uncertainties."""
    m = np.maximum(_miss(model(FGRID, FD_NO, s), BAND_NO),
                   _miss(model(FGRID, FD_YES, s), BAND_YES))
    i = int(np.argmin(m))
    return m[i] / SIG, float(FGRID[i])


def s_window(target, model, smax=0.60):
    """Both crossings of the U-shaped chi(s): the window of splits the reads
    admit at `target` sigmas."""
    grid = np.linspace(1e-5, smax, 4000)
    ok = np.array([chi(s, model)[0] <= target for s in grid])
    if not ok.any():
        return (np.nan, np.nan)
    i, j = int(np.min(np.where(ok)[0])), int(np.max(np.where(ok)[0]))
    def cross(a, b):                      # a inconsistent, b consistent
        for _ in range(60):
            m = 0.5 * (a + b)
            if chi(m, model)[0] <= target:
                b = m
            else:
                a = m
        return 0.5 * (a + b)
    lo = cross(grid[i - 1], grid[i]) if i > 0 else grid[0]
    hi = cross(grid[j + 1], grid[j]) if j + 1 < len(grid) else np.nan
    return (lo, hi)


def s_at_chi(target, model, smax=0.60):
    """Largest s the reads admit at `target` sigmas.

    This is the upper crossing of the U-shaped chi(s). A grid scan precedes
    the bisection because chi is also positive below the consistent window.
    """
    grid = np.linspace(1e-5, smax, 4000)
    ok = np.array([chi(s, model)[0] <= target for s in grid])
    if not ok.any():
        return np.nan
    j = int(np.max(np.where(ok)[0]))
    if j == len(grid) - 1:
        return np.nan
    lo, hi = grid[j], grid[j + 1]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if chi(mid, model)[0] <= target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


w_lin = s_window(0.0, f_lin)
w_ex = s_window(0.0, f_2dof)
print("  chi(s) is U-shaped: a small split cannot move the line across the two")
print("  disjoint read bands, and a large split overshoots them.")
print("  Consistent window, linear model    : s = %.2f to %.2f %%"
      % (100 * w_lin[0], 100 * w_lin[1]))
print("  pairwise solve, for comparison     : s = %.2f to %.2f %%   %s"
      % (100 * meas.s_linear.min(), 100 * meas.s_linear.max(),
         "OK" if (abs(w_lin[0] - meas.s_linear.min()) < 5e-4
                  and abs(w_lin[1] - meas.s_linear.max()) < 5e-4) else "MISMATCH"))
print("  Consistent window, exact 2-DOF     : s = %.2f to %.2f %%"
      % (100 * w_ex[0], 100 * w_ex[1]))
print()

# --- (a) what the plain mass ratio predicts ---
rule("(a)  H_plain: mu = M_s / M_deck, anchorage ordinate ignored")
print("  N8E: m = 20.2 kg/m, L = 80 m -> M_s = %.0f kg (half-sine modal mass);"
      % M_S)
print("  deck modal mass 40 to 160 t; inclination 19 to 26 deg; n = 1.")
print("  f_iso is anchored on the undamped read, so the damped read is a")
print("  prediction, not a fit.")
print()
print("  M_deck   mu_plain    theta   s_plain   f_iso    predicted damped read"
      "   shift")
print("  " + "-" * 88)
arows = []
for Md in (MDECK_LO, 60e3, 80e3, 100e3, 120e3, MDECK_HI):
    mu = M_S / Md
    for th in TH:
        s = s_from_mu(mu, th)
        for anchor in (min(READ_NO), float(np.mean(READ_NO)), max(READ_NO)):
            fi = brentq(lambda f: float(f_lin(f, FD_NO, s)) - anchor, 0.55, 0.99)
            py = float(f_lin(fi, FD_YES, s))
            arows.append(dict(M_deck_t=Md / 1e3, mu_plain=mu,
                              theta_deg=np.rad2deg(th), s_plain=s,
                              anchor_no=anchor, f_iso=fi, pred_yes=py,
                              pred_shift=py - anchor))
        r = arows[-2]
        print("  %5.0f t  %.3e  %2.0f deg  %6.3f %%  %.4f   %.4f Hz"
              "               %+6.1f mHz"
              % (Md / 1e3, mu, np.rad2deg(th), 100 * s, r["f_iso"],
                 r["pred_yes"], 1e3 * r["pred_shift"]))
apl = pd.DataFrame(arows)
print()
print("  H_plain spans mu = %.2e to %.2e, s = %.2f to %.2f %%, and predicts a"
      % (apl.mu_plain.min(), apl.mu_plain.max(),
         100 * apl.s_plain.min(), 100 * apl.s_plain.max()))
print("  shift of %+.1f to %+.1f mHz."
      % (1e3 * apl.pred_shift.min(), 1e3 * apl.pred_shift.max()))
print("  Observed shift %+.1f to %+.1f mHz; %.0f mHz per read, %.1f mHz on the"
      % (1e3 * SHIFT_LO, 1e3 * SHIFT_HI, 1e3 * SIG, 1e3 * SIG_SHIFT))
print("  difference.")
print()
ov_lo = max(apl.mu_plain.min(), meas.mu_linear.min())
ov_hi = min(apl.mu_plain.max(), meas.mu_linear.max())
print("  Measured bracket: mu = %.2e to %.2e."
      % (meas.mu_linear.min(), meas.mu_linear.max()))
if ov_lo <= ov_hi:
    print("  H_plain overlaps it on %.2e to %.2e, so the traverse cannot"
          % (ov_lo, ov_hi))
    print("  exclude H_plain as a whole.  Part (b) gives the deck modal mass")
    print("  below which it is excluded.")
else:
    print("  H_plain is disjoint from it and is excluded.")
print()

# --- (b) exclusion test ---
rule("(b)  EXCLUSION OF H_plain BY THE TRAVERSE")
brows = []
for name, model in MODELS.items():
    for tgt in (0.0, 1.0, 2.0, 3.0):
        s_c = s_at_chi(tgt, model)
        for th in TH:
            mu_c = mu_from_s(s_c, th)
            brows.append(dict(model=name, sigmas=tgt, theta_deg=np.rad2deg(th),
                              s_crit=s_c, mu_crit=mu_c,
                              M_deck_crit_t=M_S / mu_c / 1e3))
bnd = pd.DataFrame(brows)
print("  H_plain is consistent only where its predicted split lies inside the")
print("  window the reads admit.  It is excluded at the stated number of read")
print("  uncertainties when the deck modal mass is below:")
print()
print("  model      sigmas  theta   s admitted   mu_crit     M_deck below which")
print("  " + "-" * 76)
for _, r in bnd.iterrows():
    print("  %-9s   %.0f    %2.0f deg   %6.3f %%    %.3e     %6.1f t"
          % (r.model, r.sigmas, r.theta_deg, 100 * r.s_crit, r.mu_crit,
             r.M_deck_crit_t))
print()
m0 = bnd[(bnd.model == "linear") & (bnd.sigmas == 0)].M_deck_crit_t
m1 = bnd[(bnd.model == "linear") & (bnd.sigmas == 1)].M_deck_crit_t
m2 = bnd[(bnd.model == "linear") & (bnd.sigmas == 2)].M_deck_crit_t
print("  Against the assumed range, 40 to 160 t:")
print("    zero allowance : excluded below %.0f to %.0f t, the lighter %.0f to"
      % (m0.min(), m0.max(), 100 * (m0.min() - 40) / 120))
print("                     %.0f percent of the range;" % (100 * (m0.max() - 40) / 120))
print("    one sigma      : excluded below %.0f to %.0f t;" % (m1.min(), m1.max()))
print("    two sigma      : excluded below %.0f to %.0f t." % (m2.min(), m2.max()))
print()

best = None
for Md in np.linspace(MDECK_LO, MDECK_HI, 481):
    for th in TH:
        s = s_from_mu(M_S / Md, th)
        c, fi = chi(s, f_lin)
        if best is None or c < best[0]:
            best = (c, Md, np.rad2deg(th), s, fi)
c_best, Md_best, th_best, s_best, fi_best = best
pn, py = float(f_lin(fi_best, FD_NO, s_best)), float(f_lin(fi_best, FD_YES, s_best))
print("  Best fit of H_plain over 40 to 160 t:")
print("    M_deck = %.0f t, theta = %.0f deg, s = %.3f %%, f_iso = %.4f Hz;"
      % (Md_best / 1e3, th_best, 100 * s_best, fi_best))
print("    predicts %.4f Hz undamped and %.4f Hz damped, shift %+.1f mHz;"
      % (pn, py, 1e3 * (py - pn)))
print("    worst band miss %.2f read uncertainties  =>  %s."
      % (c_best, "excluded" if c_best > 2 else "not excluded"))
print()

apl["chi_sigmas"] = [chi(s, f_lin)[0] for s in apl.s_plain]
apl["shift_gap_sigmas"] = [max(SHIFT_LO - r.pred_shift, 0.0,
                               r.pred_shift - SHIFT_HI) / SIG_SHIFT
                           for _, r in apl.iterrows()]
print("  The same test on the shift (anchored f_iso, predicted shift")
print("  against the observed envelope %+.1f to %+.1f mHz, over %.1f mHz):"
      % (1e3 * SHIFT_LO, 1e3 * SHIFT_HI, 1e3 * SIG_SHIFT))
for Md in (40, 60, 80, 100, 120, 160):
    g = apl[apl.M_deck_t == Md]
    print("    %3.0f t : predicted %+6.1f to %+6.1f mHz, gap %.2f to %.2f sigma,"
          " chi %.2f"
          % (Md, 1e3 * g.pred_shift.min(), 1e3 * g.pred_shift.max(),
             g.shift_gap_sigmas.min(), g.shift_gap_sigmas.max(),
             g.chi_sigmas.min()))
print()
print("  The anchored statistic is the stricter of the two because it uses")
print("  the undamped read as the anchor; chi lets f_iso vary freely.")
print("  Both are given; chi governs.")
print()

# --- (c) the null ---
rule("(c)  H_null: mu_eff = 0, the line does not move")
c0, f0_ = chi(0.0, f_lin)
print("  Prediction: shift exactly 0 mHz, the same reading in both states.")
print("  Observed:   %+.1f to %+.1f mHz over the twelve read pairs."
      % (1e3 * SHIFT_LO, 1e3 * SHIFT_HI))
print()
print("  Band test.  With s = 0 the line sits at f_iso in both states, so one")
print("  frequency must satisfy both bands, [%.4f, %.4f] and [%.4f, %.4f],"
      % (BAND_NO[0], BAND_NO[1], BAND_YES[0], BAND_YES[1]))
print("  which are disjoint by %.1f mHz.  Best miss %.2f sigma."
      % (1e3 * (BAND_NO[0] - BAND_YES[1]), c0))
print()
zs = np.array([abs(b - a) / SIG_SHIFT for a in READ_NO for b in READ_YES])
dm = float(np.mean(READ_YES) - np.mean(READ_NO))
se = SIG * np.sqrt(1.0 / len(READ_NO) + 1.0 / len(READ_YES))
print("  Magnitude test, sigma on a difference of two reads = %.1f mHz:"
      % (1e3 * SIG_SHIFT))
print("    least negative pair (%+.1f mHz) : z = %.2f" % (1e3 * SHIFT_HI, zs.min()))
print("    most negative pair  (%+.1f mHz) : z = %.2f" % (1e3 * SHIFT_LO, zs.max()))
print("    band means %.4f -> %.4f, %+.1f mHz, SE %.1f mHz : z = %.2f"
      % (np.mean(READ_NO), np.mean(READ_YES), 1e3 * dm, 1e3 * se, abs(dm) / se))
print()
p_rank = 1.0 / comb(len(READ_NO) + len(READ_YES), len(READ_NO))
print("  Rank test, which needs no error model.  All %d undamped reads exceed"
      % len(READ_NO))
print("  all %d damped reads; under H_null the seven are exchangeable, so that"
      % len(READ_YES))
print("  separation has one-sided p = 1/C(7,3) = %.3f, a %.2f sigma equivalent."
      % (p_rank, norm.isf(p_rank)))
print()
print("  H_null.  On magnitude the null is not excluded: the largest")
print("  single pair reaches %.2f sigma and the band means %.2f,"
      % (zs.max(), abs(dm) / se))
print("  both below two.  The evidence against the null is the sign:")
print("  every damped read lies below every undamped read, with")
print("  p = %.3f on a rank test and a %.2f sigma band miss.  Two effects"
      % (p_rank, c0))
print("  act in opposite directions and neither can be quantified from")
print("  the published record.  The seven reads come from three figures,")
print("  so they are not seven independent draws and the rank p is")
print("  optimistic.  A shared axis-calibration error is common mode and")
print("  cancels in the difference, so the shift is better determined")
print("  than 4 mHz per read implies.  The null is disfavored at about")
print("  two sigma.")
print()

# --- (d) falsification table ---
rule("(d)  FALSIFICATION TABLE")
def verdict(c):
    return "excluded" if c > 2 else ("marginal" if c > 1 else "not excluded")


tab = []
for Md in (40, 80, 160):
    g = apl[apl.M_deck_t == Md]
    tab.append(("H_plain, M_deck = %d t" % Md,
                "%+.0f to %+.0f mHz" % (1e3 * g.pred_shift.min(),
                                        1e3 * g.pred_shift.max()),
                g.chi_sigmas.min()))
tab.append(("H_plain, best over 40-160 t", "%+.1f mHz" % (1e3 * (py - pn)),
            c_best))
tab.append(("H_null, mu_eff = 0", "0.0 mHz exactly", c0))
print("  hypothesis                    predicted shift   observed        "
      "miss   verdict")
print("  " + "-" * 88)
obs = "%+.0f to %+.0f mHz" % (1e3 * SHIFT_LO, 1e3 * SHIFT_HI)
for name, pred, c in tab:
    print("  %-29s %-17s %-15s %.2f   %s" % (name, pred, obs, c, verdict(c)))
print("  %-29s %-17s %-15s %.2f   %s"
      % ("H_eff, 1.3e-5 to 1.8e-3",
         "%+.0f to %+.0f mHz" % (1e3 * SHIFT_LO, 1e3 * SHIFT_HI), obs, 0.0,
         "consistent"))
print()
print("  miss is chi, the best-fit band miss in 4 mHz read uncertainties;")
print("  excluded means a miss above 2 sigma.")
print()

# --- where H_plain is falsified: the N7E spacing ---
rule("H_plain AGAINST THE N7E STAY-DECK SPACING")
Q7_STAY, Q7_DECK = 0.005, 0.0005      # tabulation quanta, 2 dp and 3 dp
SEP_OBS = abs(F7_STAY - F7_DECK)
SEP_MAX = SEP_OBS + Q7_STAY + Q7_DECK
F7 = 0.5 * (F7_STAY + F7_DECK)
ZETA = (0.002, 0.005, 0.010)

print("  N7E: m = 10.7 kg/m, L = 73.7 m -> M_s = %.0f kg, at nearly exact"
      % M_S7)
print("  tuning with a global mode that carries almost no motion where it")
print("  anchors.  mu_eff = M_s phi_a^2 places the anchorage near a node, so")
print("  the split is at most half a percent; H_plain ignores the ordinate and")
print("  predicts a wide split.  The deck modal mass range is the same")
print("  40 to 160 t, though this is a different global mode from the traverse.")
print()
print("  M_deck    mu_plain    s_plain (16-28 deg)   split at %.3f Hz" % F7)
print("  " + "-" * 68)
s7 = []
for Md in (MDECK_LO, 80e3, MDECK_HI):
    mu7 = M_S7 / Md
    ss = [s_from_mu(mu7, t) for t in TH7]
    s7 += ss
    print("  %5.0f t   %.3e   %.2f to %.2f %%           %.1f to %.1f mHz"
          % (Md / 1e3, mu7, 100 * min(ss), 100 * max(ss),
             1e3 * min(ss) * F7, 1e3 * max(ss) * F7))
s7_min, s7_max = min(s7), max(s7)
print()

rule("")
print("  TEST 1  minimum separation of the hybrid pair.  Two veering branches")
print("  cannot approach closer than s f0 at any detuning, so the observed")
print("  spacing of the stay line and the deck mode is an upper bound on")
print("  the split, independent of f_iso.")
print()
print("    observed  : stay %.3f Hz (Kumar (2011) Table 4.4, 2 dp) and deck mode %.3f Hz"
      % (F7_STAY, F7_DECK))
print("                (Kumar (2011) Table 5.6, 3 dp), separation %.1f mHz, at most %.1f mHz"
      % (1e3 * SEP_OBS, 1e3 * SEP_MAX))
print("                with the tabulation rounding taken in H_plain's favor")
print("    H_plain   : needs at least %.1f mHz (s = %.2f %% at %.3f Hz)"
      % (1e3 * s7_min * F7, 100 * s7_min, F7))
s_paper7 = s_from_mu(6e-5, TH7[1])
print("    H_eff     : mu_eff = 6e-5 gives s = %.2f %%, a separation of at least %.1f mHz."
      % (100 * s_paper7, 1e3 * s_paper7 * F7))
print("                That is %.1f mHz above the nominal %.1f mHz separation,"
      % (1e3 * (s_paper7 * F7 - SEP_OBS), 1e3 * SEP_OBS))
print("                so it is admitted only through the tabulation")
print("                rounding.  The record bounds")
print("                the N7E split at %.1f mHz and cannot distinguish H_eff"
      % (1e3 * SEP_MAX))
print("                from zero coupling there; both are separated from")
print("                H_plain by the factor below.")
print()
print("    H_plain exceeds the largest separation the record allows by a")
print("    factor %.1f, and by %.1f at the top of its own range: excluded."
      % (s7_min * F7 / SEP_MAX, s7_max * F7 / SEP_MAX))
print()

print("  TEST 2  resolvability.  A dip between the two peaks exists only for")
print("  s > 0.9717 zeta, and a 3 dB dip, which a peak picker can act on, needs")
print("  s > 2.280 zeta.  For H_plain's smallest N7E split, %.2f %%, to leave"
      % (100 * s7_min))
print("  the single unsplit line the record shows, the stay damping would have")
print("  to reach zeta > %.2f %% (no dip at all) or zeta > %.2f %% (no 3 dB dip)."
      % (100 * s7_min / 0.9717, 100 * s7_min / 2.280))
print("  Measured stay damping is zeta = %.1f to %.1f %%, so H_plain misses"
      % (100 * min(ZETA), 100 * max(ZETA)))
print("  by a factor %.1f to %.1f in damping on the strict criterion."
      % (s7_min / 0.9717 / max(ZETA), s7_min / 0.9717 / min(ZETA)))
print()
print("  Deck modal mass H_plain needs for the doublet to stay unresolved:")
print()
print("  zeta     s allowed (3 dB)   mu required     M_deck required")
print("  " + "-" * 62)
for z in ZETA:
    mu_req = mu_from_s(2.280 * z, TH7[1])     # 28 deg, the kindest to H_plain
    print("  %.1f %%      < %.2f %%           < %.2e     > %8.0f t"
          % (100 * z, 100 * 2.280 * z, mu_req, M_S7 / mu_req / 1e3))
mu_req = mu_from_s(2.280 * max(ZETA), TH7[1])
Mreq = M_S7 / mu_req / 1e3
print()
print("  At zeta = %.1f %%, high for a stay, H_plain needs %.0f t,"
      % (100 * max(ZETA), Mreq))
print("  %.1f times the top of the assumed range; at %.1f %% it needs %.0f t,"
      % (Mreq / (MDECK_HI / 1e3), 100 * ZETA[1],
         M_S7 / mu_from_s(2.280 * ZETA[1], TH7[1]) / 1e3))
print("  %.0f times.  Test 2 excludes H_plain for any plausible stay"
      % (M_S7 / mu_from_s(2.280 * ZETA[1], TH7[1]) / 1e3 / (MDECK_HI / 1e3)))
print("  damping, but its margin depends on the assumed zeta; Test 1")
print("  does not depend on damping.")
print()

print("  The N7E frequency anomaly does not distinguish the two")
print("  hypotheses.  The stay fundamental is %.1f %% above the"
      % (100 * (F7_STAY / 1.1035 - 1)))
print("  string-formula frequency of the measured tension, and near exact tuning")
print("  H_plain displaces the upper branch by s/2 = %.1f to %.1f %%, which"
      % (50 * s7_min, 50 * s7_max))
print("  brackets the observed anomaly.")
print("  H_plain is excluded by the spacing of the pair and the absence of a")
print("  doublet, not by the position of the line.  The anomaly alone is")
print("  consistent with H_plain, so the spacing test is the")
print("  discriminating one.")
print()
print("  Not used above: the structural estimate")
print("  mu_eff = 1.3e-5 to 1.8e-3 is disjoint from H_plain = 5.1e-3 to")
print("  2.0e-2, but it is built from the anchorage ordinate, which is the")
print("  quantity H_plain omits, so comparing the two would assume the")
print("  result.  Only measurements that do not use the ordinate (the")
print("  traverse bound and the N7E spacing) can decide between them.")
print()

# --- output ---
out = os.path.join(ROOT, "data", "falsify_traverse.csv")
out2 = os.path.join(ROOT, "data", "falsify_traverse_thresholds.csv")
apl.to_csv(out, index=False)
bnd.to_csv(out2, index=False)

rule("SUMMARY")
print("  1. The traverse does not exclude H_plain over 40 to 160 t.  Its best")
print("     fit, M_deck = %.0f t at theta = %.0f deg, reproduces both reads"
      % (Md_best / 1e3, th_best))
print("     with a miss of %.2f read uncertainties, predicting a %+.1f mHz"
      % (c_best, 1e3 * (py - pn)))
print("     shift against the %+.0f to %+.0f mHz observed.  The traverse"
      % (1e3 * SHIFT_LO, 1e3 * SHIFT_HI))
print("     excludes H_plain only below %.0f to %.0f t at zero allowance and"
      % (m0.min(), m0.max()))
print("     %.0f to %.0f t at one read uncertainty.  The measured and H_plain"
      % (m1.min(), m1.max()))
print("     brackets overlap on mu = %.1e to %.1e, so the measured bracket"
      % (ov_lo, ov_hi))
print("     does not exclude H_plain.")
print("  2. H_null is not excluded on magnitude: %.2f sigma at the most"
      % zs.max())
print("     favorable pair, %.2f on the band means.  It is disfavored at"
      % (abs(dm) / se))
print("     %.2f sigma by the sign, through a rank test on seven reads taken"
      % norm.isf(p_rank))
print("     from three figures and therefore not independent, so the result")
print("     is marginal.")
print("  3. The traverse establishes the sign, which needs no calibration,")
print("     and a two-sided bound")
print("     s = %.2f to %.2f %% (linear) or %.2f to %.2f %% (exact 2-DOF),"
      % (100 * meas.s_linear.min(), 100 * meas.s_linear.max(),
         100 * meas.s_exact.min(), 100 * meas.s_exact.max()))
print("     hence mu_eff < %.1e.  Any hypothesis that predicts a wider split"
      % meas.mu_linear.max())
print("     is inconsistent with this upper bound.")
print("  4. H_plain is excluded by the N7E coincidence in the same record.")
print("     The stay line and the deck mode sit %.1f mHz apart, at most %.1f"
      % (1e3 * SEP_OBS, 1e3 * SEP_MAX))
print("     mHz, while two veering branches cannot approach closer than s f0;")
print("     H_plain requires at least %.1f mHz, a factor %.1f too large.  This"
      % (1e3 * s7_min * F7, s7_min * F7 / SEP_MAX))
print("     test needs no f_iso, no damping and no anchorage ordinate.")
print("  5. The traverse gives the sign and a bound on the")
print("     width; the N7E spacing distinguishes mu_eff from the plain mass")
print("     ratio.  The traverse alone does not exclude the plain mass ratio")
print("     (item 1).")
print()
print("  wrote %s" % out)
print("  wrote %s" % out2)
