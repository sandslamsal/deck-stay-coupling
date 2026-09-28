# -*- coding: utf-8 -*-
"""Design of a bench-scale cable-beam rig for testing the veering split and
the tension-error law.

The rig is a simply supported steel blade bending in the horizontal plane,
with a near-vertical music wire (theta = 0) pinned to it at a movable station
and tensioned through a load cell and a soft spring. Using the closed forms
and coupled model of src/cablefe.py, the script computes the rig's
frequencies, the crossings in the tension sweep, the predicted split, the
detuning and damping ladders, and instrumentation checks. Writes
data/rig_design.csv.

Run:  python3 scripts/rig_design.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from cablefe import (CableDeck, beam_freq, chain, invert_string,  # noqa: E402
                     irvine_lambda2, mu_effective, tensioned_beam_freq,
                     tension_error, veering_split, xi_param)

DATA = os.path.join(ROOT, "data")

# ---------------------------------------------------------------------------
# 1.  the rig
# ---------------------------------------------------------------------------

# -- beam ("deck") ---------------------------------------------------------
# S355 flat bar on edge: 75 mm tall (stiff axis, carries gravity) by 10 mm
# thick (soft axis, the coupled plane), 900 mm between pin centers.
E_B, RHO_B = 200.0e9, 7850.0
LB, B_TALL, B_THICK = 0.900, 0.075, 0.010

AB = B_TALL * B_THICK
MB_LIN = RHO_B * AB
MB = MB_LIN * LB
IB_FLEX = B_TALL * B_THICK ** 3 / 12.0
IB_STIFF = B_THICK * B_TALL ** 3 / 12.0
EIB = E_B * IB_FLEX

# -- wire ("stay") ---------------------------------------------------------
# ASTM A228 music wire, 1.50 mm, 1.500 m between pin centers. UTS about
# 2200 MPa keeps the sweep below 14 % of breaking; low internal damping;
# ferromagnetic, so it can be driven by a non-contact electromagnet.
E_C, RHO_C = 207.0e9, 7850.0
LC, DC = 1.500, 0.00150

AC = np.pi * DC ** 2 / 4.0
MC_LIN = RHO_C * AC
IC = np.pi * DC ** 4 / 64.0
EIC = E_C * IC
EA_WIRE = E_C * AC
M_STAY = 0.5 * MC_LIN * LC
UTS_C = 2200e6

# -- tensioner -------------------------------------------------------------
# soft spring in series with the wire: cuts thermal tension drift and the
# wire's axial restraint on the beam
K_SPRING = 5.0e3
K_AX_WIRE = EA_WIRE / LC
K_SERIES = 1.0 / (1.0 / K_SPRING + 1.0 / K_AX_WIRE)
EA_EFF = K_SERIES * LC
ALPHA_STEEL = 11.5e-6

# -- test matrix -----------------------------------------------------------
# anchorage stations x_a / L_b: mu_eff varies as sin^2(j pi x_a / L_b) while
# M_s / M_b stays fixed
STATIONS = {"S50": 0.500, "S33": 1.0 / 3.0, "S25": 0.250,
            "S15": 0.150, "S05": 0.050}
THETAS_DEG = [0.0, 30.0, 45.0]
ND, NC = 120, 120
T_MIN, T_MAX = 20.0, 520.0
NMODE_BEAM, NMODE_WIRE = 3, 12

D_LADDER = np.array([-0.16, -0.08, -0.04, -0.02, -0.01, -0.005, -0.0025,
                     0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16])
ZETA_LADDER = np.array([0.0015, 0.0025, 0.004, 0.006, 0.009, 0.014,
                        0.020, 0.030, 0.045])
ZETA_BASE = 0.0015                    # the rig's own damping, assumed

C_DIP = 2.0 * np.sqrt(np.sqrt(5.0) - 2.0)     # 0.97169: a dip needs s > C_DIP zeta
C_3DB = 2.0 * 1.13985                         # 2.2797: a 3 dB dip needs s > C_3DB zeta

# sensor stations for the shape fit, x measured from the ground end, the
# anchorage at x = L; clustered at both ends because the evanescent term
# decays over L/xi, 20-50 mm on this wire
SHAPE_X = np.array([0.005, 0.015, 0.030, 0.060, 0.120, 0.200, 0.300, 0.420,
                    0.550, 0.680, 0.800, 0.880, 0.940, 0.970, 0.995])


# ---------------------------------------------------------------------------
# 2.  closed forms and finite element references for the rig
# ---------------------------------------------------------------------------

def phi_anchor(j, xfrac):
    """Mass-normalized beam ordinate at the anchorage, analytic.

    Simply supported uniform beam: phi_j(x) = sqrt(2/(m L)) sin(j pi x/L).
    In the rig phi_a^2 is measured as the driving-point residue of the beam
    at the anchorage station.
    """
    return np.sqrt(2.0 / MB) * np.sin(j * np.pi * xfrac)


def analytic_beam_freqs():
    return np.array([beam_freq(j, LB, EIB, MB_LIN)
                     for j in range(1, NMODE_BEAM + 1)])


def fe_wire_freq(n, T):
    """FE frequency of the wire alone, both ends pinned."""
    K, M = chain(LC, NC, EIC, MC_LIN, T)
    keep = [i for i in range(K.shape[0]) if i not in (0, 2 * NC)]
    w2, _ = eigh(K[np.ix_(keep, keep)], M[np.ix_(keep, keep)])
    f = np.sqrt(np.maximum(w2, 0.0)) / (2.0 * np.pi)
    return f[n - 1]


def build(T, xfrac, theta):
    return CableDeck(Ld=LB, EId=EIB, md=MB_LIN, Lc=LC, EIc=EIC, mc=MC_LIN,
                     T=T, EA=EA_EFF, theta=theta, x_anchor=xfrac * LB,
                     nd=ND, nc=NC)


def fe_beam_freqs(xfrac, theta):
    """Beam alone with the stay present only as its axial spring."""
    return build(100.0, xfrac, theta).deck_alone(NMODE_BEAM)


def crossing_tension(n, f_target):
    """Tension at which FE wire mode n sits exactly on f_target."""
    lo, hi = 1.0, 4000.0
    g = lambda T: fe_wire_freq(n, T) - f_target        # noqa: E731
    if g(lo) * g(hi) > 0:
        return np.nan
    return brentq(g, lo, hi, xtol=1e-8, rtol=1e-12)


def zeta_limits(s):
    return s / C_DIP, s / C_3DB


def dip_depth_db(u):
    """Depth (dB) of the notch between two equal-residue Lorentzians.

    Merged-peak spectrum at rho = 1, u = s / (2 zeta):
    |H|^2 = 4(x^2+1) / ((x^2 - 1 - u^2)^2 + 4x^2), x measured from the pair
    center in units of zeta omega_0.  Positive means a visible notch.
    """
    h2 = lambda x: 4.0 * (x * x + 1.0) / ((x * x - 1.0 - u * u) ** 2  # noqa
                                          + 4.0 * x * x)
    xs2 = u * np.sqrt(u * u + 4.0) - 1.0
    if xs2 <= 0.0:
        return 0.0
    return 10.0 * np.log10(h2(np.sqrt(xs2)) / h2(0.0))


def merged_peak(u, zeta, d=0.0):
    """Peak position x_* and tension error once the doublet has merged.

    At equal residues the maximum of |H|^2 sits at
    x_*^2 = u sqrt(u^2 + 4) - 1, real only when the dip exists, and the
    tension error is eps_m = 2 zeta x_* - d.  Below the threshold x_* = 0,
    the single peak sits at the pair center, and the error is -d.
    """
    xs2 = u * np.sqrt(u * u + 4.0) - 1.0
    xs = np.sqrt(xs2) if xs2 > 0.0 else 0.0
    return xs, 2.0 * zeta * xs - d


def resolution(f0, s, zeta):
    """Line spacing and record length one crossing demands.

    At least eight lines across the split and six across each branch's
    half-power band (the damping is measured); the tighter governs.
    """
    df = min(s * f0 / 8.0, 2.0 * zeta * f0 / 6.0)
    return df, 1.0 / df


# ---------------------------------------------------------------------------
# 3.  what the finite element says the rig will do
# ---------------------------------------------------------------------------

def pick_window(n):
    """Relative half-width of the band a peak picker searches.

    Wire modes are spaced f0/n apart, so the window shrinks with n to keep
    neighboring orders out.
    """
    return min(0.20, 0.35 / n)


def fe_split(T, xfrac, theta, f0):
    """Fractional gap between the two coupled modes straddling f0."""
    cd = build(T, xfrac, theta)
    f, Phi = cd.modes(120)
    win = f[(f > (1 - 0.25) * f0) & (f < (1 + 0.25) * f0)]
    if len(win) < 2:
        return np.nan, np.nan
    two = np.sort(win[np.argsort(np.abs(win - f0))[:2]])
    # residue ratio at the wire sensor, which sets the notch shape
    dof = cd.cable_sensor_dof(0.15 * LC)
    amp = np.abs(Phi[dof, :])
    a = [amp[np.argmin(np.abs(f - t))] for t in two]
    gap = (two[1] - two[0]) / f0
    # rho is undefined when one branch carries no wire motion at the sensor
    rho = (a[1] / a[0]) ** 2 if (a[0] > 0 and gap > 1e-4) else np.nan
    return gap, rho


def fe_pick(T, xfrac, theta, n, f_iso):
    """Frequency picked by a wire sensor, and the tensions inferred from it.

    The largest response in the search window at the sensor 0.15 L_c from
    the anchorage wins, as in peak picking.  Returns (f_pick, T_str, T_tb),
    the tensions from the string and tensioned-beam formulas.
    """
    cd = build(T, xfrac, theta)
    f, Phi = cd.modes(120)
    dof = cd.cable_sensor_dof(0.15 * LC)
    amp = np.abs(Phi[dof, :])
    w = pick_window(n)
    sel = (f > (1 - w) * f_iso) & (f < (1 + w) * f_iso)
    if not sel.any():
        return np.nan, np.nan, np.nan
    idx = np.where(sel)[0]
    f_pick = float(f[idx[np.argmax(amp[idx])]])
    T_str = float(invert_string(np.array([f_pick]), LC, MC_LIN,
                                np.array([n]))[0])
    T_tb = (4.0 * MC_LIN * LC ** 2 * f_pick ** 2 / n ** 2
            - EIC * (n * np.pi / LC) ** 2)
    return f_pick, T_str, T_tb


# ---------------------------------------------------------------------------
# 4.  the test program
# ---------------------------------------------------------------------------

HEADLINE = [("S50", 0.0, 1, 1), ("S15", 0.0, 1, 1), ("S05", 0.0, 1, 1),
            ("S25", 0.0, 2, 2), ("S50", 0.0, 2, 2), ("S25", 0.0, 2, 4),
            ("S33", 0.0, 3, 4)]
DAMPING_SET = [("S50", 0.0, 1, 1), ("S25", 0.0, 2, 2), ("S15", 0.0, 1, 2),
               ("S05", 0.0, 1, 1)]


def main():
    fb_an = analytic_beam_freqs()
    fb0 = fe_beam_freqs(0.5, 0.0)          # theta = 0: k_ax = 0 identically

    print("=" * 78)
    print("BENCH-SCALE CABLE-BEAM RIG:  computed design")
    print("=" * 78)
    print("\nBEAM (the 'deck'):  S355 flat bar on edge, %.0f x %.0f mm, "
          "span %.0f mm" % (1e3 * B_TALL, 1e3 * B_THICK, 1e3 * LB))
    print("  m_b = %.4f kg/m, span mass M_b = %.4f kg, modal mass M_b/2 = "
          "%.4f kg" % (MB_LIN, MB, MB / 2))
    print("  EI(soft) = %.1f N m^2;  EI(stiff)/EI(soft) = %.1f, so the "
          "stiff-axis modes sit at %.1f x the soft ones"
          % (EIB, IB_STIFF / IB_FLEX, np.sqrt(IB_STIFF / IB_FLEX)))
    for j in range(NMODE_BEAM):
        print("  beam mode %d: %8.3f Hz (analytic %8.3f Hz)"
              % (j + 1, fb0[j], fb_an[j]))
    print("  first stiff-axis mode ~ %.0f Hz, out of the coupled plane"
          % (fb0[0] * np.sqrt(IB_STIFF / IB_FLEX)))
    print("  self-weight deflection on the stiff axis: %.3f mm"
          % (1e3 * 5 * MB_LIN * 9.80665 * LB ** 4
             / (384 * E_B * IB_STIFF)))

    print("\nWIRE (the 'stay'):  ASTM A228 music wire, d = %.2f mm, "
          "L_c = %.3f m" % (1e3 * DC, LC))
    print("  m_c = %.6f kg/m, wire mass %.2f g, M_s = %.5f kg"
          % (MC_LIN, 1e3 * MC_LIN * LC, M_STAY))
    print("  EI_c = %.5f N m^2, EA = %.1f kN, EA_eff through the soft "
          "spring = %.2f kN" % (EIC, 1e-3 * EA_WIRE, 1e-3 * EA_EFF))
    print("  stress %.0f MPa at %.0f N (%.1f %% UTS) to %.0f MPa at %.0f N "
          "(%.1f %% UTS)"
          % (1e-6 * T_MIN / AC, T_MIN, 100 * T_MIN / AC / UTS_C,
             1e-6 * T_MAX / AC, T_MAX, 100 * T_MAX / AC / UTS_C))
    print("  xi = L sqrt(T/EI) runs %.0f to %.0f over the tension range"
          % (xi_param(LC, T_MIN, EIC), xi_param(LC, T_MAX, EIC)))
    print("  self weight along the chord: %.3f N, so the tension varies "
          "%.2f %% end to end at %.0f N"
          % (MC_LIN * LC * 9.80665,
             100 * MC_LIN * LC * 9.80665 / 100.0, 100.0))

    print("\nGOVERNING GROUP")
    print("  plain mass ratio M_s/M_b = %.4e, the same at every station "
          "and every beam mode" % (M_STAY / MB))
    print("  station  x_a/L_b      mu_eff(j=1)   mu_eff(j=2)   mu_eff(j=3)")
    for name, xf in STATIONS.items():
        mus = [mu_effective(M_STAY, phi_anchor(j, xf)) for j in (1, 2, 3)]
        print("  %s      %.4f   %11.4e   %11.4e   %11.4e"
              % (name, xf, *mus))
    print("  design set (699 graded designs, data/campaign.csv): mu_eff "
          "5th-95th pct 3.5e-5 to 1.5e-2, median 6.5e-4")

    print("\nTENSIONER, THERMAL AND STATIC CHECKS")
    print("  tension drift: %.3f N/K with the soft spring, %.2f N/K without"
          % (ALPHA_STEEL * LC * K_SERIES, ALPHA_STEEL * LC * K_AX_WIRE))
    print("  at 100 N that is %.3f %%/K in tension and %.3f %%/K in "
          "frequency" % (100 * ALPHA_STEEL * LC * K_SERIES / 100.0,
                         50 * ALPHA_STEEL * LC * K_SERIES / 100.0))
    mu_mid = mu_effective(M_STAY, phi_anchor(1, 0.5))
    for th in THETAS_DEG:
        fbt = fe_beam_freqs(0.5, np.deg2rad(th))
        kax = EA_EFF * np.sin(np.deg2rad(th)) ** 2 / LC
        print("  theta = %4.1f deg: static soft-axis deflection %.3f mm "
              "(= 2 L_c mu_eff sin theta / pi^2), k_ax = %5.0f N/m, "
              "beam mode 1 moves %6.3f Hz -> %.3f Hz"
              % (th, 1e3 * 2 * LC * mu_mid * np.sin(np.deg2rad(th))
                 / np.pi ** 2, kax, fbt[0] - fb0[0], fbt[0]))

    rows = []

    # ---- 4a. the isolated-wire sweep ------------------------------------
    print("\nISOLATED WIRE FREQUENCIES OVER THE TENSION RANGE (Hz)")
    Tsweep = np.array([20, 25, 36, 50, 60, 80, 96, 110, 150, 175, 216, 250,
                       317, 400, 460, 520], float)
    hdr = "   T[N]  " + "".join("   n=%-2d" % n for n in range(1, 9))
    print(hdr)
    for T in Tsweep:
        fs = [tensioned_beam_freq(n, LC, T, EIC, MC_LIN) for n in range(1, 13)]
        print("  %5.0f  " % T + "".join("%7.2f" % v for v in fs[:8]))
        for n, fv in enumerate(fs, start=1):
            rows.append(dict(block="sweep", config="", station="",
                             x_over_Lb=np.nan, theta_deg=0.0,
                             beam_mode=np.nan, wire_mode=n, T_N=T,
                             f_iso_Hz=fv,
                             sigma_wire_MPa=1e-6 * T / AC,
                             xi=xi_param(LC, T, EIC)))
    print("  beam modes for comparison: %.2f, %.2f, %.2f Hz"
          % tuple(fb0[:3]))

    # ---- 4b. every crossing, every station, every tilt -------------------
    cross = []
    for j in range(1, NMODE_BEAM + 1):
        for n in range(1, NMODE_WIRE + 1):
            T = crossing_tension(n, fb0[j - 1])
            if np.isfinite(T) and T_MIN <= T <= T_MAX:
                cross.append((j, n, float(fb0[j - 1]), float(T)))
    cross.sort(key=lambda r: -r[3])

    print("\nCROSSINGS INSIDE THE TENSION RANGE (%.0f to %.0f N): %d of them"
          % (T_MIN, T_MAX, len(cross)))
    print("   j   n   T_cross[N]   f0[Hz]   sigma[MPa]     xi")
    for j, n, f0, T in cross:
        print("  %2d  %2d   %9.2f  %8.2f   %8.0f  %6.0f"
              % (j, n, T, f0, 1e-6 * T / AC, xi_param(LC, T, EIC)))

    print("\nPREDICTED SPLIT: closed form against the coupled FE model")
    for name, xf in STATIONS.items():
        for th_deg in THETAS_DEG:
            if th_deg != 0.0 and name != "S50":
                continue
            th = np.deg2rad(th_deg)
            fbj = fe_beam_freqs(xf, th)
            print("  station %s (x_a/L_b = %.4f), theta = %.0f deg"
                  % (name, xf, th_deg))
            for j, n, _, _ in cross:
                f0 = float(fbj[j - 1])
                Tc = crossing_tension(n, f0)
                if not np.isfinite(Tc):
                    continue
                mu = mu_effective(M_STAY, phi_anchor(j, xf))
                s = veering_split(mu, th, n)
                s_fe, rho = fe_split(Tc, xf, th, f0)
                zd, z3 = zeta_limits(s) if s > 0 else (0.0, 0.0)
                if s > 1e-6:
                    df, tr = resolution(f0, s, ZETA_BASE)
                else:
                    df, tr = np.nan, np.nan
                lam2 = (irvine_lambda2(LC, Tc, EA_EFF, MC_LIN,
                                       np.pi / 2 - th) if th > 0 else 0.0)
                rows.append(dict(
                    block="crossing", config="%s_th%02d" % (name, int(th_deg)),
                    station=name, x_over_Lb=xf, theta_deg=th_deg,
                    beam_mode=j, wire_mode=n, T_N=Tc, f0_Hz=f0,
                    mass_ratio=M_STAY / MB, mu_eff=mu,
                    s_pred_pct=100 * s, s_fe_pct=100 * s_fe, rho_wire=rho,
                    zeta_dip_max_pct=100 * zd, zeta_3dB_max_pct=100 * z3,
                    df_req_Hz=df, T_record_s=tr,
                    sigma_wire_MPa=1e-6 * Tc / AC,
                    xi=xi_param(LC, Tc, EIC), lam2=lam2))
                print("    j=%d n=%2d  T=%7.2f N  f0=%7.2f Hz  "
                      "mu_eff=%.4e  s=%6.3f %%  s_FE=%6.3f %%  "
                      "rho=%5.2f  zeta<%.3f %%(3dB)  df<%s  T_rec>%s"
                      % (j, n, Tc, f0, mu, 100 * s, 100 * s_fe, rho,
                         100 * z3,
                         ("%.4f Hz" % df) if np.isfinite(df) else "   n/a",
                         ("%5.0f s" % tr) if np.isfinite(tr) else "  n/a"))

    # ---- 4c. detuning ladders -------------------------------------------
    print("\nDETUNING LADDERS  (eps is the tension error; the branch moves "
          "by eps/2 in frequency)")
    for name, th_deg, j, n in HEADLINE:
        xf, th = STATIONS[name], np.deg2rad(th_deg)
        f0 = float(fe_beam_freqs(xf, th)[j - 1])
        mu = mu_effective(M_STAY, phi_anchor(j, xf))
        s = veering_split(mu, th, n)
        print("  %s theta=%.0f  j=%d n=%d   f0=%.3f Hz  mu_eff=%.4e  "
              "s_pred=%.4f %%" % (name, th_deg, j, n, f0, mu, 100 * s))
        print("       d[%]     T[N]  f_iso[Hz]   eps_pred[%]  eps_f_pred[%]"
              "  eps_f_FE[%]  epsT_FE_str[%]  epsT_FE_tb[%]  branch")
        for d in D_LADDER:
            f_iso = f0 / (1.0 - d)
            T = crossing_tension(n, f_iso)
            if not np.isfinite(T) or not (10.0 <= T <= 900.0):
                continue
            eps = tension_error(d, mu, th, n)
            f_pick, T_str, T_tb = fe_pick(T, xf, th, n, f_iso)
            br = ("upper" if f_pick > f0 else "lower") if np.isfinite(
                f_pick) else ""
            rows.append(dict(
                block="ladder", config="%s_th%02d" % (name, int(th_deg)),
                station=name, x_over_Lb=xf, theta_deg=th_deg,
                beam_mode=j, wire_mode=n, T_N=T, f0_Hz=f0,
                mass_ratio=M_STAY / MB, mu_eff=mu, s_pred_pct=100 * s,
                d_pct=100 * d, f_iso_Hz=f_iso,
                eps_pred_pct=100 * eps, eps_f_pred_pct=50 * eps,
                eps_f_fe_pct=100 * (f_pick - f_iso) / f_iso,
                eps_T_fe_string_pct=100 * (T_str - T) / T,
                eps_T_fe_tbeam_pct=100 * (T_tb - T) / T,
                branch=br, sigma_wire_MPa=1e-6 * T / AC,
                xi=xi_param(LC, T, EIC)))
            r = rows[-1]
            print("    %7.3f  %7.2f  %9.3f   %10.3f   %11.3f  %11.3f  "
                  "%13.3f  %12.3f   %s"
                  % (r["d_pct"], T, f_iso, r["eps_pred_pct"],
                     r["eps_f_pred_pct"], r["eps_f_fe_pct"],
                     r["eps_T_fe_string_pct"], r["eps_T_fe_tbeam_pct"], br))

    # ---- 4d. resolvability ladder ---------------------------------------
    print("\nRESOLVABILITY LADDER  (dip depth in dB between the two peaks)")
    print("  thresholds: a dip exists while s > %.4f zeta, a 3 dB dip "
          "while s > %.4f zeta" % (C_DIP, C_3DB))
    for name, th_deg, j, n in DAMPING_SET:
        xf, th = STATIONS[name], np.deg2rad(th_deg)
        f0 = float(fe_beam_freqs(xf, th)[j - 1])
        mu = mu_effective(M_STAY, phi_anchor(j, xf))
        s = veering_split(mu, th, n)
        zd, z3 = zeta_limits(s)
        Tc = crossing_tension(n, f0)
        print("  %s j=%d n=%d  f0=%.2f Hz  T=%.1f N  s=%.4f %%  ->  dip "
              "vanishes at zeta=%.3f %%, 3 dB lost at zeta=%.3f %%"
              % (name, j, n, f0, Tc, 100 * s, 100 * zd, 100 * z3))
        print("        zeta[%]      u     dip[dB]  eps_merged[%]  df_req[Hz]  T_rec[s]")
        for z in ZETA_LADDER:
            u = s / (2.0 * z)
            dip = dip_depth_db(u)
            xs, eps_m = merged_peak(u, z, 0.0)
            df, tr = resolution(f0, s, z)
            rows.append(dict(
                block="damping", config="%s_th%02d" % (name, int(th_deg)),
                station=name, x_over_Lb=xf, theta_deg=th_deg, beam_mode=j,
                wire_mode=n, T_N=Tc, f0_Hz=f0, mu_eff=mu, s_pred_pct=100 * s,
                zeta_pct=100 * z, u=u, dip_dB=dip,
                eps_merged_pct=100 * eps_m,
                zeta_dip_max_pct=100 * zd, zeta_3dB_max_pct=100 * z3,
                df_req_Hz=df, T_record_s=tr))
            print("        %6.3f  %6.3f   %8.3f   %12.3f   %9.4f  %8.0f" % (100 * z, u, dip, 100 * eps_m, df, tr))

    # ---- 4e. the negative control ---------------------------------------
    print("\nNEGATIVE CONTROL: mu_eff against the plain mass ratio")
    for j, ctrl, live in ((2, "S50", "S25"), (3, "S33", "S25")):
        for jj, n, _, _ in cross:
            if jj != j:
                continue
            f0c = float(fe_beam_freqs(STATIONS[ctrl], 0.0)[j - 1])
            f0l = float(fe_beam_freqs(STATIONS[live], 0.0)[j - 1])
            Tc, Tl = crossing_tension(n, f0c), crossing_tension(n, f0l)
            sc, _ = fe_split(Tc, STATIONS[ctrl], 0.0, f0c)
            sl, _ = fe_split(Tl, STATIONS[live], 0.0, f0l)
            mc_ = mu_effective(M_STAY, phi_anchor(j, STATIONS[ctrl]))
            ml_ = mu_effective(M_STAY, phi_anchor(j, STATIONS[live]))
            print("  beam mode %d, wire mode %2d, T = %6.1f N:  %s "
                  "mu_eff=%.2e s_FE=%.4f %%   |   %s mu_eff=%.2e "
                  "s_FE=%.4f %%   ratio %.0f:1   [M_s/M_b = %.3e in both]"
                  % (j, n, Tc, ctrl, mc_, 100 * sc, live, ml_, 100 * sl,
                     sl / max(sc, 1e-12), M_STAY / MB))
            break
    print("  node placement tolerance (beam mode 2 at midspan):")
    for dx in (0.5, 1.0, 2.0, 5.0):
        e = dx * 1e-3 / LB
        mul = mu_effective(M_STAY, phi_anchor(2, 0.5 + e))
        print("    %4.1f mm off the node gives mu_eff = %.3e, s = %.4f %%"
              % (dx, mul, 100 * veering_split(mul, 0.0, 2)))

    # ---- 4f. shape-fit stations -----------------------------------------
    print("\nSHAPE-FIT SENSOR STATIONS")
    for T in (24.0, 99.5, 397.8):
        p = np.sqrt((np.sqrt(T ** 2 + 0.0) + T) / (2.0 * EIC))
        print("  T = %6.1f N: xi = %5.1f, evanescent decay length L/xi = "
              "%.1f mm, so the two stations nearest each end (%.1f and "
              "%.1f mm) resolve it"
              % (T, xi_param(LC, T, EIC), 1e3 / p,
                 1e3 * SHAPE_X[0] * LC, 1e3 * SHAPE_X[1] * LC))
    print("  15 stations at x/L_c = " + ", ".join("%.3f" % v
                                                  for v in SHAPE_X))
    print("  in mm from the ground end: " + ", ".join(
        "%.1f" % (1e3 * v * LC) for v in SHAPE_X))
    print("  data/sensitivity.csv: 5 stations identify T; shape "
          "noise 0.5 % of peak gives 0.74 % RMSE in tension, 1 % gives "
          "1.39 %, 2 % gives 2.73 %")
    print("  chord length enters squared and mass per meter one for one, so "
          "L_c to +-0.5 mm (%.3f %%) and m_c to +-0.1 %% bound the systematic "
          "error at %.2f %% and %.2f %%"
          % (100 * 0.0005 / LC, 2 * 100 * 0.0005 / LC, 0.1))

    # ---- 4g. instrumentation arithmetic ---------------------------------
    print("\nINSTRUMENTATION ARITHMETIC")
    m_wire = MC_LIN * LC
    for name, m_acc in (("PCB 352A92", 0.6e-3), ("Endevco 25B", 0.2e-3),
                        ("retro-reflective dot", 1e-6)):
        print("  %-22s %5.2f g on a %.2f g wire = %5.2f %% of the wire "
              "mass, so a mode antinode moves by up to %.2f %%; the split "
              "being measured is %.2f to %.2f %%"
              % (name, 1e3 * m_acc, 1e3 * m_wire, 100 * m_acc / m_wire,
                 100 * m_acc / m_wire, 0.15, 3.99))
    print("  -> the wire is measured without contact; accelerometers "
          "go on the beam, where %.1f g on %.2f kg is %.3f %%"
          % (5.0, MB, 100 * 5e-3 / MB))
    fs = 2048.0
    nblk = 2 ** 18
    print("  DAQ: fs = %.0f Hz, block %d samples = %.1f s, df = %.4f Hz, "
          "finer than the tightest requirement in the test program (%.4f Hz)"
          % (fs, nblk, nblk / fs, fs / nblk, 0.0110))
    print("  anti-alias below %.0f Hz; the highest wire mode of interest is "
          "n = 8 at %.0f N, %.0f Hz"
          % (0.4 * fs, T_MAX,
             tensioned_beam_freq(8, LC, T_MAX, EIC, MC_LIN)))
    v1 = 2 * np.pi * 28.26 * 1e-3
    v2 = 2 * np.pi * 254.3 * 10e-6
    print("  LDV velocity at 1 mm peak and %.1f Hz: %.0f mm/s; at 10 um "
          "peak and %.0f Hz: %.1f mm/s.  Both are far above a %.0f um/s "
          "noise floor" % (28.26, 1e3 * v1, 254.3, 1e3 * v2, 1.0))
    print("  load cell: 500 N full scale at class 0.03 %% FS is %.2f N, "
          "= %.3f %% at the lowest crossing tension (%.1f N) and %.3f %% at "
          "the highest (%.1f N)"
          % (0.0003 * 500, 100 * 0.0003 * 500 / 21.5, 21.5,
             100 * 0.0003 * 500 / 501.1, 501.1))

    df = pd.DataFrame(rows)
    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, "rig_design.csv")
    df.to_csv(out, index=False)
    print("\nwrote %s  (%d rows, blocks: %s)"
          % (out, len(df), ", ".join(sorted(df.block.unique()))))


if __name__ == "__main__":
    main()
