# -*- coding: utf-8 -*-
"""What each identification method returns through a deck-stay crossing.

THE CLAIM UNDER TEST.  The manuscript prices the tension error of the
incumbent inversion with two closed forms.  The branch law,
``eps = sqrt(d^2 + s^2) - |d|``, prices a resolved pair with one branch
read.  The merged-peak law prices the single maximum left when the pair is
not resolved, and it says that maximum lies OUTSIDE the pair, reaching
``sqrt(5)/2`` of the branch value at ``u = 2/sqrt(3)`` before collapsing
toward the mean as damping rises further.  Both assume the frequency is
PICKED from a spectrum as a maximum.  The manuscript then asserts, without
evidence, that a method which fits modes rather than picking maxima "can
separate a pair the spectrum shows as one, in which case the branch law
returns in full".  Nothing in the study had run such a method.

This script runs six of them on the same records: peak picking, half power,
FDD, EFDD, covariance-driven SSI and data-driven SSI, over a traverse of
tension through the crossing, six damping ratios, three noise levels, three
record lengths and repeated realisations, and reports the tension each one
returns beside the truth, the branch law and the merged-peak law.

NOTHING HERE REIMPLEMENTS THE PHYSICS OR THE ESTIMATORS.  The model, the
Rayleigh calibration, the excitation and the measurement chain come from
``simulate_records``; the four spectral estimators come from ``oma_fdd``;
the two subspace estimators come from ``oma_ssi``; the two laws come from
``run_merged`` and ``run_dangerband``; the inversion comes from
``cablefe``.  The verification below asserts that the readings produced
here reproduce ``oma_fdd.run_one`` on a shared state to the last digit, so
this campaign cannot drift from the arms it summarises.

WHAT IS SWEPT, AND WHY

tension    seventeen values, placed by inverting the detuning rather than by
           stepping the tension, so the traverse is even in the variable the
           laws are written in.  The targets run to ``|d| = 4 per cent``,
           which is where ``run_dangerband`` measures the seasonal swing, and
           thirteen of the seventeen sit inside ``|d| <= s`` where the branch
           exchange happens.
damping    0.1, 0.2, 0.5, 1, 2 and 3 per cent.  At the crossing these give
           ``u = s / 2 zeta`` of 11.7, 5.8, 2.3, 1.17, 0.58 and 0.39, so the
           sweep crosses ``u = 2/sqrt(3) = 1.155``, where the merged-peak law
           is worst, and ``u = 0.4859``, below which the dip in the spectrum
           does not exist at all.
noise      clean (no sensor noise), then 40, 20, 10, 6 and 3 dB broadband.
           A force-balance accelerometer on a stay sits nearer 50 dB and a
           cheap MEMS unit nearer 15 dB, so 20 dB is already pessimistic and
           3 dB is a deliberately hostile case.
length     150 s, 600 s and 3600 s.  600 s is the ordinary ambient stay
           record; 3600 s is what a monitoring system can be asked for; 150 s
           is what a survey crew has time for, and at that length the Welch
           resolution alone is coarser than the split, which separates
           "the spectrum cannot resolve it" from "the damping has merged it".
seeds      five realisations in the main grid, three in the sub-arms.  Every
           number below is reported with its scatter across realisations.

THE READINGS.  Each method returns one frequency, and the incumbent
inversion turns it into a tension by ``T = 4 m L^2 f^2``.  Two error
measures are carried for each, because they answer different questions.
``eps`` is the coupling error ``(f/f_iso1)^2 - 1``, which is what the two
laws predict; ``errT`` is the error against the true tension, which is what
the engineer suffers and which also carries the 0.13 per cent the string
formula loses to bending stiffness.  For the subspace methods, which can
return two poles, four readings are priced, because "SSI separated the
pair" and "the engineer got the right tension" are different claims:

  amplitude   the pole with the larger stay-channel spectral density, which
              is what an engineer would take with no other information
  shape       the pole with the larger stay-to-deck amplitude ratio in the
              identified mode shape, which a two-sensor analyst can apply
              and a one-sensor analyst cannot
  oracle      the pole that IS the stay-dominated branch, which nobody can
              know in the field but which is what the branch law prices
  trace       both poles used together, ``f_s^2 = f_+^2 + f_-^2 - f_d^2``,
              which is exact for the two-mode pencil and needs the deck
              frequency as an input.  Its sensitivity to that input is
              measured and reported, not assumed away.

Writes ``data/oma.csv`` (one row per state) and ``data/oma_verify.csv``.

Run:  python3 scripts/run_oma.py --verify
      python3 scripts/run_oma.py --campaign [--nproc 6]
      python3 scripts/run_oma.py --report
      python3 scripts/run_oma.py --all
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from cablefe import CableDeck, tensioned_beam_freq  # noqa: E402
from run_damping import BRIDGE, FBAND, T_TUNE  # noqa: E402
from run_merged import (eps_branch_law, eps_merged_law,  # noqa: E402
                        eps_wrong_law, rho_2dof, xstar)
from run_dangerband import picked_error  # noqa: E402
from simulate_records import RecordSimulator  # noqa: E402
from oma_fdd import (SMOOTH_BINS, _smooth_db, cpsd_welch,  # noqa: E402
                     efdd, errors, fdd, half_power, peak_pick, sv_decomp)
from oma_ssi import identify, prepare  # noqa: E402

DATA = os.path.join(ROOT, "data")

# ---------------------------------------------------------------------------
# the design
# ---------------------------------------------------------------------------

# detuning targets; the tensions that realise them are solved for below
D_TARGETS = (-0.040, -0.030, -0.022, -0.016, -0.011, -0.007, -0.004, -0.002,
             0.000, 0.002, 0.004, 0.007, 0.011, 0.016, 0.022, 0.030, 0.040)
D_CORE = (-0.022, -0.007, 0.000, 0.007, 0.022)   # the sub-arms' tensions

ZETAS = (0.001, 0.002, 0.005, 0.010, 0.020, 0.030)
DUR_MAIN = (600.0, 3600.0)
DUR_SHORT = 150.0
SNR_MAIN = (None, 20.0)                # clean and the realistic default
SNR_LADDER = (40.0, 10.0, 6.0, 3.0)    # 20 dB is already in the main grid
SEEDS_MAIN = (0, 1, 2, 3, 4)
SEEDS_SUB = (0, 1, 2)

TAU = 16.0                             # s, the SSI correlation lag
NPERSEG_LONG = 8192                    # 0.0122 Hz at fs = 100 Hz
NPERSEG_SHORT = 2048                   # 0.0488 Hz, forced by a 150 s record
MATCH_TOL = 0.01                       # a pole matches a branch within 1 %
# The tolerance is nearly half the split at the worked crossing, so a
# separation rate quoted at one tolerance says as much about the tolerance as
# about the method. Every rate below is reported across this ladder.
TOL_LADDER = (0.010, 0.005, 0.003, 0.002)
PROM_DB = 3.0                          # the manuscript's peak-count rule
U_DIP = np.sqrt(np.sqrt(5.0) - 2.0)    # 0.48587
U_3DB = 1.1401                         # from run_damping's numeric bisection
U_WORST = 2.0 / np.sqrt(3.0)           # 1.1547, where the merged peak is worst

SSI_CONFIGS = (("cov2", "cov", "sd"), ("dat2", "data", "sd"),
               ("cov1", "cov", "s"), ("dat1", "data", "s"))


# ---------------------------------------------------------------------------
# the truth, and the tension grid that realises a wanted detuning
# ---------------------------------------------------------------------------

def model_truth(T, cache={}):
    """Frequencies, branch identities and detuning of the coupled model.

    ``d`` is the relative detuning of the two uncoupled resonators.  It is
    measured two ways.  ``d`` itself is ``2 (f_iso1 - f0) / f0``, which is
    exact for the two-mode pencil because coupling leaves the mean of the
    pair where the mean of the uncoupled pair was; ``d_deck`` is the direct
    ``(f_iso1 - f_deck) / f0`` using the deck with the stay present only as
    its axial spring.  They agree to 0.02 percentage points across the whole
    traverse, which is checked rather than assumed.
    """
    key = round(float(T), 3)
    if key in cache:
        return cache[key]
    cd = CableDeck(T=float(T), **BRIDGE)
    f, Phi = cd.modes(60)
    f_iso = float(tensioned_beam_freq(1, BRIDGE["Lc"], float(T),
                                      BRIDGE["EIc"], BRIDGE["mc"]))
    near = np.sort(np.argsort(np.abs(f - f_iso))[:2])
    f_lo, f_hi = float(f[near[0]]), float(f[near[1]])
    f0 = 0.5 * (f_lo + f_hi)
    es = cd.energy_split(Phi)
    f_deck_all = cd.deck_alone(8)
    j = int(np.argmin(np.abs(f_deck_all - f_iso)))
    amp = np.abs(Phi[cd.cable_sensor_dof(2.0), near])
    out = dict(T=float(T), f_lo=f_lo, f_hi=f_hi, f0=f0,
               Delta=(f_hi - f_lo) / f0, f_iso1=f_iso,
               f_deck=float(f_deck_all[j]),
               d=2.0 * (f_iso - f0) / f0,
               d_deck=(f_iso - float(f_deck_all[j])) / f0,
               e_lo=float(es[near[0]]), e_hi=float(es[near[1]]),
               amp_lo=float(amp[0]), amp_hi=float(amp[1]))
    out["f_stay_energy"] = f_hi if out["e_hi"] >= out["e_lo"] else f_lo
    out["f_stay_amp"] = f_hi if out["amp_hi"] >= out["amp_lo"] else f_lo
    out["branch_is_upper"] = bool(out["e_hi"] >= out["e_lo"])
    if len(cache) > 400:
        cache.clear()
    cache[key] = out
    return out


def s_at_tuning(lo=1.40e5, hi=1.62e5, it=60):
    """The split at exact tuning, by golden-section on the observed split.

    Taken from the model rather than from the closed form, so the number the
    laws are evaluated with is the number this bridge actually has.
    """
    g = 0.5 * (np.sqrt(5.0) - 1.0)
    a, b = lo, hi
    c, e = b - g * (b - a), a + g * (b - a)
    fc, fe = model_truth(c)["Delta"], model_truth(e)["Delta"]
    for _ in range(it):
        if fc < fe:
            b, e, fe = e, c, fc
            c = b - g * (b - a)
            fc = model_truth(c)["Delta"]
        else:
            a, c, fc = c, e, fe
            e = a + g * (b - a)
            fe = model_truth(e)["Delta"]
    Tm = 0.5 * (a + b)
    return model_truth(Tm)["Delta"], Tm


def tension_for_d(d_target, lo=1.30e5, hi=1.80e5, it=60):
    """Tension at which the detuning is ``d_target``, by bisection."""
    fa = model_truth(lo)["d"] - d_target
    fb = model_truth(hi)["d"] - d_target
    if fa * fb > 0:
        raise ValueError("d = %g not bracketed" % d_target)
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        fm = model_truth(mid)["d"] - d_target
        if fa * fm <= 0:
            hi, fb = mid, fm
        else:
            lo, fa = mid, fm
    return 0.5 * (lo + hi)


def tension_grid(targets=D_TARGETS):
    return [tension_for_d(d) for d in targets]


# ---------------------------------------------------------------------------
# grading, shared by every method so the comparison is like for like
# ---------------------------------------------------------------------------

def grade_pair(cands, f_lo, f_hi, tol=MATCH_TOL):
    """How many of the two true branches a candidate list resolves.

    Each true branch is matched to its nearest candidate; the branch counts
    as found if that candidate is within ``tol``, and the pair counts as
    separated only when the two branches are matched by DIFFERENT
    candidates.  One frequency sitting between two close branches therefore
    scores one, not two, which is the whole point of the count.
    """
    c = np.asarray([v for v in np.atleast_1d(cands) if np.isfinite(v)],
                   dtype=float)
    if not len(c):
        return 0, False, np.nan
    hit = {}
    for name, ft in (("lo", f_lo), ("hi", f_hi)):
        k = int(np.argmin(np.abs(c - ft)))
        if abs(c[k] - ft) / ft < tol:
            hit[name] = k
    both = len(set(hit.values())) == 2
    sep = (abs(c[hit["hi"]] - c[hit["lo"]]) / (0.5 * (c[hit["hi"]]
                                                      + c[hit["lo"]]))
           if both else np.nan)
    return len(set(hit.values())), both, sep


def count_peaks_3db(f, P, window, half_width_hz, prom_db=PROM_DB):
    """Peaks of the smoothed decibel spectrum inside a window.

    The manuscript's merge convention is a 3 dB prominence, so the count that
    decides "the spectrum shows one peak or two" uses that rule and not the
    statistical floor the frequency picker uses.  The smoothing is the same
    three-bin decibel average ``oma_fdd`` applies before reading heights.
    """
    m = (f >= window[0]) & (f <= window[1])
    if m.sum() < 5:
        return 0, np.array([])
    db, _ = _smooth_db(f[m], P[m], half_width_hz)
    pk, _ = find_peaks(db, prominence=prom_db)
    return len(pk), f[m][pk]


def pair_window(tr, zeta):
    """A window around the true pair, wide enough to hold a merged peak.

    The merged-peak law puts the surviving maximum OUTSIDE the pair, up to
    ``sqrt(5)/2`` of the half separation beyond a branch, so a window clipped
    to ``[f_lo, f_hi]`` would miss exactly the reading under study.  The
    margin is therefore the larger of half the separation and two half-power
    bandwidths, and the window is clipped to the analysis band.
    """
    sep = tr["f_hi"] - tr["f_lo"]
    marg = max(0.5 * sep, 4.0 * zeta * tr["f0"])
    return (max(tr["f_lo"] - marg, FBAND[0]),
            min(tr["f_hi"] + marg, FBAND[1]))


def add_reading(out, tag, f_hat, T_true, f_iso):
    """One method's frequency, tension and both error measures."""
    T_hat, e_ts, e_cp = errors(f_hat, T_true, f_iso)
    out["f_" + tag] = f_hat
    out["T_" + tag + "_kN"] = T_hat / 1e3 if np.isfinite(T_hat) else np.nan
    out["eps_" + tag + "_pct"] = e_cp
    out["errT_" + tag + "_pct"] = e_ts
    return out


# ---------------------------------------------------------------------------
# the subspace readings
# ---------------------------------------------------------------------------

def ssi_block(out, tag, y, fs, sd, method, tr, f_spec, db_stay, T_true):
    """One SSI configuration: the poles it returns and the four readings.

    ``sd`` is the per-channel standard deviation AFTER the pipeline's
    decimation and detrending.  ``oma_ssi.prepare`` scales every channel to
    unit variance before the decomposition, so the identified mode shape is
    in normalised units and its stay-to-deck ratio is meaningless until the
    scaling is put back; multiplying component ``k`` by ``sd[k]`` restores
    the physical shape.  Without that step the shape rule below would be
    reading the normalisation, not the mode.

    ``db_stay`` is the SMOOTHED decibel spectrum of the stay channel, the
    same curve the peak picker reads.  The amplitude rule below is an
    engineer looking at a plot and taking the pole that stands taller there,
    so it must read that curve and not the raw periodogram: a Welch density
    scatters by about 27 per cent per line, which is enough to reverse the
    rule between two poles of nearly equal height and would show up as a
    property of the identification method rather than of the spectrum it is
    being compared with.
    """
    pref = tag
    out["fail_" + pref] = ""
    try:
        sel = identify(y, fs, method=method, tau=TAU)
    except Exception as exc:                       # noqa: BLE001
        out["fail_" + pref] = "%s: %s" % (type(exc).__name__, exc)
        out["n_" + pref] = 0
        out["both_" + pref] = False
        out["split_" + pref] = np.nan
        for suff in ("", "_or", "_sh", "_tr"):
            add_reading(out, pref + suff, np.nan, T_true, tr["f_iso1"])
        out["f_%s_p1" % pref] = out["f_%s_p2" % pref] = np.nan
        out["z_%s_p1" % pref] = out["z_%s_p2" % pref] = np.nan
        out["ratio_%s_p1" % pref] = out["ratio_%s_p2" % pref] = np.nan
        out["errT_%s_tr1_pct" % pref] = np.nan
        return out

    inb = sel[(sel["f"] > FBAND[0]) & (sel["f"] < FBAND[1])]
    fg = inb["f"].to_numpy(dtype=float)
    zg = inb["zeta"].to_numpy(dtype=float)
    n_match, both, sep = grade_pair(fg, tr["f_lo"], tr["f_hi"])
    out["n_" + pref] = len(fg)
    out["n_matched_" + pref] = n_match
    out["both_" + pref] = both
    out["split_" + pref] = sep

    # physical stay-to-deck amplitude ratio of each pole, two channels only
    ratio = np.full(len(fg), np.nan)
    if len(fg) and len(sd) > 1 and "phi" in inb:
        for k, phi in enumerate(inb["phi"].to_numpy()):
            p = np.asarray(phi, dtype=complex).ravel()
            if len(p) >= 2:
                pp = np.abs(p[:2]) * sd[:2]
                ratio[k] = pp[0] / pp[1] if pp[1] > 0 else np.inf

    dens = (np.interp(fg, f_spec, db_stay) if len(fg) else np.array([]))
    order = np.argsort(-dens) if len(fg) else np.array([], dtype=int)
    for j, nm in enumerate(("p1", "p2")):
        if len(fg) > j:
            k = int(order[j])
            out["f_%s_%s" % (pref, nm)] = float(fg[k])
            out["z_%s_%s" % (pref, nm)] = float(zg[k])
            out["ratio_%s_%s" % (pref, nm)] = float(ratio[k])
        else:
            out["f_%s_%s" % (pref, nm)] = np.nan
            out["z_%s_%s" % (pref, nm)] = np.nan
            out["ratio_%s_%s" % (pref, nm)] = np.nan

    # (1) amplitude rule
    f_amp = float(fg[int(np.argmax(dens))]) if len(fg) else np.nan
    add_reading(out, pref, f_amp, T_true, tr["f_iso1"])
    # (2) oracle branch
    f_or = (float(fg[int(np.argmin(np.abs(fg - tr["f_stay_energy"])))])
            if len(fg) else np.nan)
    add_reading(out, pref + "_or", f_or, T_true, tr["f_iso1"])
    # (3) shape rule
    f_sh = np.nan
    if len(fg) and np.isfinite(ratio).any():
        f_sh = float(fg[int(np.nanargmax(ratio))])
    add_reading(out, pref + "_sh", f_sh, T_true, tr["f_iso1"])
    # (4) trace rule on the two strongest poles
    f_tr = np.nan
    f_tr1 = np.nan
    if len(fg) >= 2:
        a, b = float(fg[order[0]]), float(fg[order[1]])
        q = a * a + b * b - tr["f_deck"] ** 2
        f_tr = float(np.sqrt(q)) if q > 0 else np.nan
        q1 = a * a + b * b - (1.01 * tr["f_deck"]) ** 2
        f_tr1 = float(np.sqrt(q1)) if q1 > 0 else np.nan
    add_reading(out, pref + "_tr", f_tr, T_true, tr["f_iso1"])
    out["errT_%s_tr1_pct" % pref] = errors(f_tr1, T_true, tr["f_iso1"])[1]
    # (5) the trace rule as an engineer would actually be scored on it.
    # Where the fitter returns fewer than two poles it returns no second
    # frequency, and a tension must still be reported, so the fallback is the
    # one-pole amplitude reading. Scoring only the records where two poles
    # came back selects on the outcome and flatters the rule.
    f_trf = f_tr if np.isfinite(f_tr) else f_amp
    add_reading(out, pref + "_trf", f_trf, T_true, tr["f_iso1"])
    out["n_%s_poles" % pref] = int(len(fg))
    out["fellback_%s" % pref] = bool(not np.isfinite(f_tr))
    return out


# ---------------------------------------------------------------------------
# one state: one record, every method
# ---------------------------------------------------------------------------

def analyse_state(sim, tr, s_tune, arm, duration, snr_db, seed, nperseg):
    """Every method on one record, and the two laws beside them."""
    T_true = sim.T
    zeta = sim.zeta
    rec = sim.record(duration=duration, snr_db=snr_db, seed=seed)
    X = np.vstack([rec["a_stay"], rec["a_deck"]])
    fs = sim.fs

    f, G, n_seg, k_eff = cpsd_welch(X, fs, nperseg=nperseg)
    df = fs / nperseg
    hw = SMOOTH_BINS * df
    S, U = sv_decomp(G)
    P_stay = G[:, 0, 0].real
    db_stay, _ = _smooth_db(f, P_stay, SMOOTH_BINS * fs / nperseg)

    win = pair_window(tr, zeta)
    Delta = tr["Delta"]
    u = Delta / (2.0 * zeta)

    out = dict(
        arm=arm, T_true=T_true, T_kN=T_true / 1e3, zeta=zeta,
        snr_db=(np.inf if snr_db is None else float(snr_db)),
        duration=duration, seed=seed, fs=fs, nperseg=nperseg, df_hz=df,
        n_seg=n_seg, k_eff=k_eff, tau=TAU,
        f_lo=tr["f_lo"], f_hi=tr["f_hi"], f0=tr["f0"], Delta=Delta,
        d=tr["d"], d_deck=tr["d_deck"], u=u, s_tune=s_tune,
        f_iso1=tr["f_iso1"], f_deck=tr["f_deck"],
        f_stay_energy=tr["f_stay_energy"], f_stay_amp=tr["f_stay_amp"],
        branch_is_upper=tr["branch_is_upper"],
        e_lo=tr["e_lo"], e_hi=tr["e_hi"],
        zeta_lo=float(sim.zj[sim.pair_idx[0]]),
        zeta_hi=float(sim.zj[sim.pair_idx[1]]),
        resolvable=bool(u > U_DIP), dip3db=bool(u > U_3DB),
        bins_per_split=Delta * tr["f0"] / df,
        bins_per_hpbw=2.0 * zeta * tr["f0"] / df,
        win_lo=win[0], win_hi=win[1],
        rms_stay_mg=rec["meta"]["rms_stay_mg"],
        rms_deck_mg=rec["meta"]["rms_deck_mg"],
        snr_inband_db=rec["meta"]["snr_inband_db"],
        T_iso_kN=4.0 * BRIDGE["mc"] * BRIDGE["Lc"] ** 2
        * tr["f_iso1"] ** 2 / 1e3)

    # -- the laws -------------------------------------------------------
    d = tr["d"]
    for tag, eps in (("branch", eps_branch_law(d, s_tune)),
                     ("wrong", eps_wrong_law(d, s_tune)),
                     ("merged", eps_merged_law(d, s_tune, zeta)),
                     ("mfrf", picked_error(d, s_tune, zeta))):
        out["eps_%s_pct" % tag] = 100.0 * eps
        f_law = tr["f_iso1"] * np.sqrt(max(1.0 + eps, 0.0))
        out["f_%s" % tag] = f_law
        Th = 4.0 * BRIDGE["mc"] * BRIDGE["Lc"] ** 2 * f_law ** 2
        out["T_%s_kN" % tag] = Th / 1e3
        out["errT_%s_pct" % tag] = 100.0 * (Th - T_true) / T_true
    # the branch law signed by which branch actually carries the stay
    # energy, rather than by the sign of d.  The two agree everywhere except
    # at exact tuning, where sgn(d) has no content: the pair is symmetric
    # there and which branch is called the stay branch is a label.
    sg = 1.0 if tr["f_stay_energy"] >= tr["f0"] else -1.0
    eb = sg * (np.hypot(d, s_tune) - abs(d))
    out["eps_branche_pct"] = 100.0 * eb
    f_be = tr["f_iso1"] * np.sqrt(max(1.0 + eb, 0.0))
    out["f_branche"] = f_be
    out["T_branche_kN"] = (4.0 * BRIDGE["mc"] * BRIDGE["Lc"] ** 2
                           * f_be ** 2 / 1e3)
    out["errT_branche_pct"] = 100.0 * (1e3 * out["T_branche_kN"]
                                       - T_true) / T_true
    # what the model itself gives when the right branch, and the wrong one,
    # are read exactly.  The laws approximate these two numbers; an
    # identification estimates them.
    other = (tr["f_lo"] if tr["f_stay_energy"] == tr["f_hi"] else tr["f_hi"])
    add_reading(out, "tbranch", tr["f_stay_energy"], T_true, tr["f_iso1"])
    add_reading(out, "twrong", other, T_true, tr["f_iso1"])
    out["rho"] = rho_2dof(d, s_tune)
    out["kfactor"] = xstar(np.hypot(d, s_tune) / (2.0 * zeta),
                           out["rho"]) / (np.hypot(d, s_tune) / (2.0 * zeta))

    # -- 1, 2: peak picking and half power on the stay auto-spectrum ------
    pp = peak_pick(f, P_stay, k_eff, hw, band=FBAND)
    n3, f3 = count_peaks_3db(f, P_stay, win, hw)
    n_pair_stat = int(np.sum((pp["f"] >= win[0]) & (pp["f"] <= win[1])))
    add_reading(out, "pp", pp["f_top"], T_true, tr["f_iso1"])
    out["n_pp_band"] = pp["n"]
    out["n_pp_pair"] = n_pair_stat
    out["n_pp_pair3db"] = n3
    nm, both, sep = grade_pair(pp["f"], tr["f_lo"], tr["f_hi"])
    out["n_matched_pp"] = nm
    out["both_pp"] = both
    out["split_pp"] = sep
    for tl in TOL_LADDER:
        _, bt, _ = grade_pair(pp["f"], tr["f_lo"], tr["f_hi"], tol=tl)
        out["both_pp_tol%g" % (1000 * tl)] = bt
    out["prom_db"] = pp["prom_db"]
    # The control the campaign lacked. The two-pole trace identity is not the
    # property of a subspace fitter: wherever the SPECTRUM resolves the pair,
    # a single-channel peak picker supplies the same two frequencies. Applying
    # the identity to them separates what the identity buys from what mode
    # fitting buys, which is the whole of the comparison below.
    fs_pp = np.sort(np.asarray(pp.get("f_sorted", pp["f"]), dtype=float))
    fs_pp = fs_pp[(fs_pp >= win[0]) & (fs_pp <= win[1])]
    f_pp_tr = f_pp_tr1 = np.nan
    if len(fs_pp) >= 2:
        a, b = float(fs_pp[0]), float(fs_pp[-1])
        q = a * a + b * b - tr["f_deck"] ** 2
        f_pp_tr = float(np.sqrt(q)) if q > 0 else np.nan
        q1 = a * a + b * b - (1.01 * tr["f_deck"]) ** 2
        f_pp_tr1 = float(np.sqrt(q1)) if q1 > 0 else np.nan
    add_reading(out, "pp_tr", f_pp_tr, T_true, tr["f_iso1"])
    out["errT_pp_tr1_pct"] = errors(f_pp_tr1, T_true, tr["f_iso1"])[1]
    out["n_pp_trace"] = int(len(fs_pp))

    hp = half_power(f, P_stay, band=FBAND)
    add_reading(out, "hp", hp["f0"], T_true, tr["f_iso1"])
    out["hp_ok"] = bool(hp["ok"])
    out["zeta_hp"] = hp["zeta"]
    out["zeta_hp_ratio"] = hp["zeta"] / zeta if np.isfinite(hp["zeta"]) \
        else np.nan

    # -- 3, 4: FDD and EFDD ----------------------------------------------
    res = fdd(f, S, U, k_eff, hw, band=FBAND)
    add_reading(out, "fdd", res["sv1"]["f_top"], T_true, tr["f_iso1"])
    cands = np.concatenate([res["sv1"]["f"], res["sv2"]["f"]]) \
        if (res["sv1"]["n"] + res["sv2"]["n"]) else np.array([])
    nm, both, sep = grade_pair(cands, tr["f_lo"], tr["f_hi"])
    out["n_sv1_band"] = res["sv1"]["n"]
    out["n_sv2_band"] = res["sv2"]["n"]
    out["n_sv1_pair"] = int(np.sum((res["sv1"]["f"] >= win[0])
                                   & (res["sv1"]["f"] <= win[1])))
    out["n_sv2_pair"] = int(np.sum((res["sv2"]["f"] >= win[0])
                                   & (res["sv2"]["f"] <= win[1])))
    out["n_matched_fdd"] = nm
    out["both_fdd"] = both
    out["split_fdd"] = sep
    out["sv2_peak_ratio"] = res["sv2_peak_over_sv1_peak"]
    out["f_sv2"] = res["sv2"]["f_top"]
    # the second reading FDD offers, if sv1 shows two peaks
    f2 = np.nan
    if res["sv1"]["n"] >= 2:
        f2 = float(res["sv1"]["f"][1])
    add_reading(out, "fdd2", f2, T_true, tr["f_iso1"])

    fb, Sb, Ub = res["f"], res["S"], res["U"]
    ef = dict(ok=False, f_efdd=np.nan, zeta_efdd=np.nan, n_bell=0,
              r2=np.nan, stop_lo=None, stop_hi=None)
    if res["sv1"]["n"]:
        ip = int(np.argmin(np.abs(fb - res["sv1"]["f"][0])))
        ef = efdd(fb, Sb, Ub, ip, which=0, fs=fs, nperseg=nperseg)
    add_reading(out, "efdd", ef["f_efdd"], T_true, tr["f_iso1"])
    out["efdd_ok"] = bool(ef["ok"])
    out["zeta_efdd"] = ef["zeta_efdd"]
    out["zeta_efdd_ratio"] = (ef["zeta_efdd"] / zeta
                              if np.isfinite(ef["zeta_efdd"]) else np.nan)
    out["n_bell"] = ef["n_bell"]
    out["efdd_r2"] = ef["r2"]
    out["efdd_stop"] = "%s/%s" % (ef["stop_lo"], ef["stop_hi"])

    # -- 5, 6: the two subspace methods, two instrument configurations ----
    chans = {"sd": X, "s": X[:1]}
    sds = {}
    for key, y in chans.items():
        yp, _ = prepare(y, fs, normalise=False)
        sds[key] = yp.std(axis=1)
    for tag, method, cfg in SSI_CONFIGS:
        ssi_block(out, tag, chans[cfg], fs, sds[cfg], method, tr,
                  f, db_stay, T_true)
    return out


# ---------------------------------------------------------------------------
# the campaign
# ---------------------------------------------------------------------------

def _init_worker():
    """One thread per worker.

    The linear algebra in a single cell is small enough that its own
    threading costs more than it buys, and twelve threads in each of six
    workers oversubscribe the machine.  Workers are started with the SPAWN
    context rather than fork: this parent has already run LAPACK before the
    pool is built, and forking a process whose BLAS thread pool is live
    deadlocks the child on its first call, which is what a first version of
    this script did for ten minutes before it was killed.
    """
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:                              # noqa: BLE001
        pass


def _task_rows(task):
    """One (tension, damping, length, noise) cell, all its seeds."""
    T, zeta, duration, snr_db, seeds, arm, s_tune = task
    rows = []
    sim = RecordSimulator(T, zeta)
    tr = model_truth(T)
    nperseg = NPERSEG_SHORT if duration < 600.0 else NPERSEG_LONG
    for sd in seeds:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows.append(analyse_state(sim, tr, s_tune, arm, duration,
                                      snr_db, sd, nperseg))
    return rows


def build_tasks(s_tune, quick=False):
    """Every cell of the design, cheapest first so a partial run is useful."""
    Ts = tension_grid(D_TARGETS if not quick else D_CORE)
    Tc = {d: tension_for_d(d) for d in D_CORE}
    zetas = ZETAS if not quick else (0.005, 0.020)
    seeds_main = SEEDS_MAIN if not quick else (0,)
    seeds_sub = SEEDS_SUB if not quick else (0,)
    tasks = []
    for zeta in zetas:
        for T in Ts:
            for dur in (DUR_MAIN if not quick else (600.0,)):
                for snr in SNR_MAIN:
                    tasks.append((T, zeta, dur, snr, seeds_main, "grid",
                                  s_tune))
    if not quick:
        for zeta in zetas:
            for d in D_CORE:
                for snr in SNR_LADDER:
                    tasks.append((Tc[d], zeta, 600.0, snr, seeds_sub,
                                  "noise", s_tune))
                for snr in SNR_MAIN:
                    tasks.append((Tc[d], zeta, DUR_SHORT, snr, seeds_sub,
                                  "short", s_tune))
    # short records first, so a run stopped early still spans the design
    tasks.sort(key=lambda t: (t[2], t[1], t[0]))
    return tasks


def campaign(args):
    """Run the design, or the part of it ``--dur`` selects.

    The campaign is splittable by record length because that is what its
    memory demand scales with: an hour-long record at eight times
    oversampling is a 3.4 million sample integration per channel, and eight
    of those at once will be killed by the memory manager on a loaded
    machine, which is what happened to the first attempt at 80 cells of 588.
    Running the long records in their own pass at a lower worker count keeps
    the peak bounded; ``--merge`` puts the parts back together, and the
    result is identical to a single pass because every cell is independent
    and every record is seeded.
    """
    s_tune, T_tune = s_at_tuning()
    tasks = build_tasks(s_tune, quick=args.quick)
    if args.dur:
        keep = [float(v) for v in args.dur]
        tasks = [t for t in tasks if t[2] in keep]
    if args.zeta:
        keep = [float(v) for v in args.zeta]
        tasks = [t for t in tasks if any(abs(t[1] - v) < 1e-12
                                         for v in keep)]
    n_rec = sum(len(t[4]) for t in tasks)
    print("=" * 78)
    print("THE CAMPAIGN")
    print("=" * 78)
    print("  split at exact tuning s = %.5f (%.3f per cent) at T = %.3f kN"
          % (s_tune, 100 * s_tune, T_tune / 1e3))
    print("  %d cells, %d records, %d workers" % (len(tasks), n_rec,
                                                  args.nproc))
    t0 = time.time()
    rows = []
    if args.nproc > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.nproc, initializer=_init_worker) as pool:
            for k, part in enumerate(pool.imap_unordered(_task_rows, tasks),
                                     1):
                rows.extend(part)
                if k % 20 == 0 or k == len(tasks):
                    print("    %4d/%d cells  %6.0f s" % (k, len(tasks),
                                                         time.time() - t0))
    else:
        for k, task in enumerate(tasks, 1):
            rows.extend(_task_rows(task))
            if k % 10 == 0 or k == len(tasks):
                print("    %4d/%d cells  %6.0f s" % (k, len(tasks),
                                                     time.time() - t0))
    d = pd.DataFrame(rows)
    d = d.sort_values(["arm", "duration", "zeta", "snr_db", "d", "seed"],
                      kind="mergesort").reset_index(drop=True)
    path = os.path.join(DATA, args.out)
    d.to_csv(path, index=False)
    print("  wrote %s  (%d rows, %d columns, %.0f s)"
          % (path, len(d), d.shape[1], time.time() - t0))
    return d


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def verify(args):
    """Checks that must pass before any number here is quoted."""
    rows = []

    def add(check, **kw):
        rows.append(dict(check=check, **kw))
        val = kw.get("value")
        tol = kw.get("tol")
        ok = kw.get("ok")
        print("  [%s] %-58s %s" % (check, kw.get("what", ""),
                                   ("%.6g" % val) if isinstance(val, float)
                                   else val))
        if ok is not None and not ok:
            print("        FAILED (tol %s)" % tol)

    print("=" * 78)
    print("VERIFICATION")
    print("=" * 78)

    # [A] the split at tuning, and the tension grid
    s_tune, T_tune = s_at_tuning()
    add("A", what="split at exact tuning s", value=s_tune, tol="quoted 2.34 %",
        ok=abs(100 * s_tune - 2.34) < 0.01)
    add("A", what="tension at exact tuning, kN", value=T_tune / 1e3,
        ok=abs(T_tune - T_TUNE) / T_TUNE < 0.01)
    Ts = tension_grid()
    got = [model_truth(T)["d"] for T in Ts]
    err = max(abs(g - t) for g, t in zip(got, D_TARGETS))
    add("A", what="tension grid hits its detuning targets, max |error|",
        value=err, tol=1e-7, ok=err < 1e-7)
    add("A", what="tensions spanned, kN",
        value="%.2f to %.2f" % (min(Ts) / 1e3, max(Ts) / 1e3))

    # [B] the two detuning measures, and the two-mode pencil itself
    dev, hyp = [], []
    for T in Ts:
        tr = model_truth(T)
        dev.append(abs(tr["d"] - tr["d_deck"]))
        hyp.append(abs(np.hypot(tr["d"], s_tune) - tr["Delta"]))
    add("B", what="|d - d_deck| over the traverse, max", value=max(dev),
        tol=5e-4, ok=max(dev) < 5e-4)
    add("B", what="|hypot(d, s) - observed split|, max", value=max(hyp),
        tol=1e-3, ok=max(hyp) < 1e-3)

    # [C] the trace rule against the model, before any identification
    tre = [abs((np.sqrt(model_truth(T)["f_lo"] ** 2
                        + model_truth(T)["f_hi"] ** 2
                        - model_truth(T)["f_deck"] ** 2)
                / model_truth(T)["f_iso1"]) ** 2 - 1.0) for T in Ts]
    add("C", what="trace rule on EXACT model frequencies, max |eps|",
        value=max(tre), tol=2e-3, ok=max(tre) < 2e-3)
    add("C", what="  the same, as a fraction of the branch bias s",
        value=max(tre) / s_tune)
    # its sensitivity to the deck frequency it needs
    T0 = tension_for_d(0.0)
    t0 = model_truth(T0)
    base = t0["f_lo"] ** 2 + t0["f_hi"] ** 2 - t0["f_deck"] ** 2
    pert = t0["f_lo"] ** 2 + t0["f_hi"] ** 2 - (1.01 * t0["f_deck"]) ** 2
    add("C", what="1 % error in the deck frequency costs, in tension",
        value=abs(pert / base - 1.0))

    # [D] the laws: the closed forms against their own definitions
    add("D", what="branch law at d = 0 equals s",
        value=abs(eps_branch_law(0.0, s_tune) - s_tune), tol=1e-15,
        ok=abs(eps_branch_law(0.0, s_tune) - s_tune) < 1e-15)
    # at rho = 1 the two maxima are mirror images and which one the root
    # finder returns is a convention, so the magnitude is what is checked
    k = abs(xstar(U_WORST, 1.0) / U_WORST)
    add("D", what="merged peak |k| at u = 2/sqrt(3)", value=k,
        tol="sqrt(5)/2 = 1.118034",
        ok=abs(k - np.sqrt(5) / 2) < 1e-6)
    poly = U_DIP ** 4 + 4 * U_DIP ** 2 - 1.0
    add("D", what="dip threshold solves u^4 + 4u^2 - 1", value=abs(poly),
        tol=1e-12, ok=abs(poly) < 1e-12)
    # The closed-form merged law is a linearisation, 2 zeta x* - d, of the
    # maximum of the exact two-mode receptance.  The two are compared here in
    # MAGNITUDE, because at d = 0 the pair is symmetric and which of the two
    # equal maxima a root finder returns is a convention with no content.
    # They agree to 0.01 percentage points while the pair is resolved and
    # part company in the deep merged regime, which is where the
    # linearisation of a peak that is collapsing toward the mean should be
    # expected to fail; both columns are carried in the output.
    dev_hi, dev_lo = [], []
    for zeta in ZETAS:
        for T in Ts:
            dd = model_truth(T)["d"]
            e = abs(abs(eps_merged_law(dd, s_tune, zeta))
                    - abs(picked_error(dd, s_tune, zeta)))
            (dev_hi if np.hypot(dd, s_tune) / (2 * zeta) > 1.0
             else dev_lo).append(e)
    add("D", what="merged law vs 2-DOF FRF maximum, u > 1, max",
        value=max(dev_hi), tol=1e-3, ok=max(dev_hi) < 1e-3)
    add("D", what="merged law vs 2-DOF FRF maximum, u < 1, max",
        value=max(dev_lo), tol=1e-2, ok=max(dev_lo) < 1e-2)

    # [E] the readings reproduce oma_fdd.run_one on a shared state
    import oma_fdd as FDD
    sim = RecordSimulator(T_TUNE, 0.005)
    tr = model_truth(T_TUNE)
    mine = analyse_state(sim, tr, s_tune, "verify", 600.0, 20.0, 0,
                         NPERSEG_LONG)
    theirs = FDD.run_one(sim, T_TUNE, 0.005, 600.0, 20.0, 0, NPERSEG_LONG,
                         "per_channel")
    for a, b in (("pp", "pp"), ("hp", "hp"), ("fdd", "sv1"),
                 ("efdd", "efdd1")):
        dv = abs(mine["f_" + a] - theirs["f_" + b])
        add("E", what="f_%s matches oma_fdd.run_one" % a, value=dv,
            tol=0.0, ok=dv == 0.0)

    # [F] the same record, identified twice, is identified identically
    r1 = sim.record(duration=600.0, snr_db=20.0, seed=0)
    r2 = sim.record(duration=600.0, snr_db=20.0, seed=0)
    same = float(np.max(np.abs(r1["a_stay"] - r2["a_stay"])))
    add("F", what="record(seed) is reproducible, max |difference|",
        value=same, tol=0.0, ok=same == 0.0)
    rc = sim.record(duration=600.0, snr_db=None, seed=0)
    dv = float(np.max(np.abs(rc["a_stay_clean"] - r1["a_stay_clean"])))
    add("F", what="clean and noisy records share the excitation draw",
        value=dv, tol=0.0, ok=dv == 0.0)

    # [G] the noise-free long-record limit: does the pipeline find the truth
    simq = RecordSimulator(T_TUNE, 0.002)
    trq = model_truth(T_TUNE)
    q = analyse_state(simq, trq, s_tune, "verify", 3600.0, None, 0,
                      NPERSEG_LONG)
    for tag in ("cov2", "dat2", "cov1", "dat1"):
        e = max(abs(q["f_%s_p1" % tag] - trq["f_hi"]),
                abs(q["f_%s_p2" % tag] - trq["f_lo"]))
        e = min(e, max(abs(q["f_%s_p1" % tag] - trq["f_lo"]),
                       abs(q["f_%s_p2" % tag] - trq["f_hi"])))
        add("G", what="%s recovers both branches on a clean hour, max error"
            % tag, value=e / trq["f0"], tol=2e-3, ok=e / trq["f0"] < 2e-3)

    # [H] the shape rule reads the mode and not the normalisation
    add("H", what="stay/deck ratio, stay-dominated pole (clean hour)",
        value=max(q["ratio_cov2_p1"], q["ratio_cov2_p2"]))
    add("H", what="stay/deck ratio, deck-dominated pole",
        value=min(q["ratio_cov2_p1"], q["ratio_cov2_p2"]))

    df = pd.DataFrame(rows)
    path = os.path.join(DATA, "oma_verify.csv")
    df.to_csv(path, index=False)
    print("\n  wrote %s (%d checks)" % (path, len(df)))
    bad = df[df["ok"] == False] if "ok" in df else df.iloc[:0]  # noqa: E712
    print("  %d checks, %d failed" % (len(df), len(bad)))
    return df


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

METHODS = (("pp", "peak picking"), ("hp", "half power"),
           ("fdd", "FDD"), ("efdd", "EFDD"),
           ("cov2", "SSI-COV stay+deck"), ("dat2", "SSI-DATA stay+deck"),
           ("cov1", "SSI-COV stay only"), ("dat1", "SSI-DATA stay only"))


def _mad(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x)) if len(x) else np.nan


def _pm(x):
    """mean +- standard deviation over realisations, as text."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return "   none"
    return "%6.2f+-%.2f" % (np.mean(x), np.std(x))


def dev_from_law(s, tag, law):
    """Mean distance from a method's reading to a law's prediction.

    ``branch`` takes the nearer of the two exact branch errors the model
    itself carries, because a method that resolves the pair must return one
    of the two and either is a branch reading.  ``merged`` takes the signed
    merged-peak prediction, except at exact tuning, where the pair is
    symmetric and the sign of that prediction carries no information, so the
    nearer sign is used there.
    """
    e = s["eps_%s_pct" % tag].to_numpy(dtype=float)
    if law == "branch":
        a = np.abs(e - s["eps_tbranch_pct"].to_numpy(dtype=float))
        b = np.abs(e - s["eps_twrong_pct"].to_numpy(dtype=float))
        v = np.minimum(a, b)
    else:
        m = s["eps_mfrf_pct"].to_numpy(dtype=float)
        v = np.abs(e - m)
        tune = np.abs(s["d"].to_numpy(dtype=float)) < 1e-6
        v[tune] = np.minimum(v[tune], np.abs(e[tune] + m[tune]))
    return float(np.nanmean(v))


def field_trace(d, tag="cov2"):
    """The trace rule with the deck frequency taken from the campaign itself.

    ``f_s^2 = f_+^2 + f_-^2 - f_d^2`` needs the deck frequency, and the
    version priced in the main table takes it from the model, which no
    engineer has.  The field version takes it from the same instrument at the
    same damping, noise and record length, at the most detuned tension in the
    traverse, where the two poles are far apart and the deck-dominated one is
    the pole with the SMALLER stay-to-deck amplitude ratio.  A stay's tension
    moves with temperature over a season and the deck mode does not, so this
    is an ordinary monitoring record and not a special experiment.
    """
    rows = []
    key = ["zeta", "snr_db", "duration", "seed", "arm"]
    for k, g in d.groupby(key):
        # one estimate from each end of the traverse, averaged.  A season
        # carries the tension through both ends, so both records exist, and
        # two estimates halve the error of the one input this rule needs.
        ests = []
        for side in (g[g["d"] < 0], g[g["d"] > 0]):
            if not len(side):
                continue
            far = side.loc[side["d"].abs().idxmax()]
            r1 = far.get("ratio_%s_p1" % tag, np.nan)
            r2 = far.get("ratio_%s_p2" % tag, np.nan)
            f1 = far.get("f_%s_p1" % tag, np.nan)
            f2 = far.get("f_%s_p2" % tag, np.nan)
            if np.isfinite(r1) and np.isfinite(r2):
                ests.append(f1 if r1 < r2 else f2)
        if not ests:
            continue
        fd = float(np.mean(ests))
        for _, r in g.iterrows():
            a, b = r["f_%s_p1" % tag], r["f_%s_p2" % tag]
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            q = a * a + b * b - fd * fd
            if q <= 0:
                continue
            fh = np.sqrt(q)
            T_hat, ets, ecp = errors(fh, r["T_true"], r["f_iso1"])
            rows.append(dict(zeta=r["zeta"], d=r["d"], snr_db=r["snr_db"],
                             duration=r["duration"], seed=r["seed"],
                             arm=r["arm"], f_deck_used=fd,
                             f_deck_err_pct=100 * (fd / r["f_deck"] - 1.0),
                             errT_pct=ets, eps_pct=ecp))
    return pd.DataFrame(rows)



def _rms(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else np.nan


def tune_at_tuning(d, dur, snr=20.0):
    """The exact-tuning table at one record length."""
    q = d[(d["duration"] == dur) & (d["snr_db"] == snr)
          & (np.abs(d["d"]) < 1e-6)]
    if not len(q):
        return
    print("\n  %.0f s records at %.0f dB, n = %d per row"
          % (dur, snr, q.groupby("zeta").size().max()))
    print("  %6s %6s %6s %6s | %6s %6s %6s %6s %6s | %-14s %-14s"
          % ("zeta%", "u", "branch", "merged", "pp", "fdd", "efdd", "COVor",
             "trace", "pp signed", "COV oracle signed"))
    for z in sorted(q["zeta"].unique()):
        s = q[q["zeta"] == z]
        print("  %6.1f %6.2f %6.2f %6.2f | %6.2f %6.2f %6.2f %6.2f %6.2f |"
              " %-14s %-14s"
              % (100 * z, s["u"].iloc[0], abs(s["eps_branch_pct"].iloc[0]),
                 abs(s["eps_mfrf_pct"].iloc[0]),
                 _rms(s["eps_pp_pct"]), _rms(s["eps_fdd_pct"]),
                 _rms(s["eps_efdd_pct"]), _rms(s["eps_cov2_or_pct"]),
                 _rms(s["eps_cov2_tr_pct"]),
                 _pm(s["eps_pp_pct"]), _pm(s["eps_cov2_or_pct"])))


def report(d):
    tune = d[np.abs(d["d"]) < 1e-6]
    print("\n" + "=" * 78)
    print("1  THE DESIGN, AND WHAT THE SPECTRUM SHOWS")
    print("=" * 78)
    base = d[(d["arm"] == "grid") & (d["duration"] == 600.0)
             & (d["snr_db"] == 20.0)]
    g = base[np.abs(base["d"]) < 1e-6]                # exact tuning
    # Rates are quoted over the five tensions within 0.4 per cent of exact
    # tuning rather than over the single tuned one.  Across that band u
    # changes by 1.5 per cent, so the physics is the same and the sample is
    # five times larger; error magnitudes stay at exact tuning, where the
    # laws have one value rather than five.
    gn = base[np.abs(base["d"]) <= 0.0041]
    print("  600 s records at 20 dB, |d| <= 0.4 per cent")
    print("  The two peak counts differ at light damping and the reason is")
    print("  resolution, not merging: at zeta = 0.1 per cent the resonance")
    print("  is half a Welch line wide, so where the peak falls between")
    print("  lines moves its apparent height by several decibels and a")
    print("  3 dB prominence rule loses one of two peaks that are plainly")
    print("  there.  The bandwidth in lines is printed so that the reader")
    print("  can see which counts are estimator artefacts.")
    print("  %6s %6s %8s %7s %7s %7s %7s"
          % ("zeta%", "u", "regime", "hpbw/df", "3 dB", "stat", "SSI-COV"))
    for z in sorted(gn["zeta"].unique()):
        s = gn[gn["zeta"] == z]
        reg = ("resolved" if s["u"].iloc[0] > U_3DB else
               ("dip only" if s["u"].iloc[0] > U_DIP else "merged"))
        print("  %6.1f %6.2f %8s %7.2f %7.2f %7.2f %7.2f"
              % (100 * z, s["u"].iloc[0], reg, s["bins_per_hpbw"].mean(),
                 s["n_pp_pair3db"].mean(), s["n_pp_pair"].mean(),
                 s["n_cov2"].mean()))

    print("\n" + "=" * 78)
    print("2  DOES MODE FITTING SEPARATE A PAIR THE SPECTRUM SHOWS AS ONE?")
    print("=" * 78)
    print("  fraction of records returning BOTH branches, within 1 per cent")
    print("  600 s, 20 dB, |d| <= 0.4 per cent (%d records per row)"
          % gn.groupby("zeta").size().max())
    hdr = "  %6s %6s" % ("zeta%", "u")
    for tag, _ in METHODS:
        hdr += " %7s" % tag
    print(hdr)
    for z in sorted(gn["zeta"].unique()):
        s = gn[gn["zeta"] == z]
        line = "  %6.1f %6.2f" % (100 * z, s["u"].iloc[0])
        for tag, _ in METHODS:
            col = "both_" + tag if "both_" + tag in s else None
            line += " %6.0f%%" % (100 * s[col].mean()) if col else "     na"
        print(line)
    print("  and the separation they return, as a fraction of the true one")
    print("  %6s %6s %10s %10s %10s %10s" % ("zeta%", "u", "pp", "fdd",
                                             "cov2", "cov1"))
    for z in sorted(gn["zeta"].unique()):
        s = gn[gn["zeta"] == z]
        row = "  %6.1f %6.2f" % (100 * z, s["u"].iloc[0])
        for tag in ("pp", "fdd", "cov2", "cov1"):
            v = (s["split_" + tag] / s["Delta"]).to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            row += " %10s" % ("%.3f" % np.mean(v) if len(v) else "  -")
        print(row)

    print("\n" + "=" * 78)
    print("2b IN THE MERGED REGIME, WHAT DOES A SEPARATING METHOD RETURN?")
    print("=" * 78)
    print("  Records whose stay spectrum shows ONE peak in the pair window")
    print("  under the 3 dB rule.  For those records: how often each method")
    print("  still resolves two poles, and, on the records where SSI-COV")
    print("  does resolve them, what the reading costs against what peak")
    print("  picking cost on the SAME record.  |error| in tension, per cent.")
    mg = base[base["n_pp_pair3db"] <= 1]
    print("  %6s %6s %6s %7s %7s | %8s %8s %8s %8s"
          % ("zeta%", "u", "n", "cov2", "cov1", "branch", "merged", "COV or",
             "pp"))
    for z in sorted(mg["zeta"].unique()):
        s = mg[mg["zeta"] == z]
        both = s[s["both_cov2"].astype(bool)]
        print("  %6.1f %6.2f %6d %6.0f%% %6.0f%% | %8.2f %8.2f %8s %8s"
              % (100 * z, s["u"].median(), len(s),
                 100 * s["both_cov2"].mean(), 100 * s["both_cov1"].mean(),
                 np.nanmean(np.abs(both["errT_branch_pct"])),
                 np.nanmean(np.abs(both["errT_merged_pct"])),
                 _pm(np.abs(both["errT_cov2_or_pct"])),
                 _pm(np.abs(both["errT_pp_pct"]))))

    print("\n" + "=" * 78)
    print("3  THE TENSION EACH METHOD RETURNS AT THE CROSSING")
    print("=" * 78)
    print("  Coupling error at exact tuning, per cent of tension, over %d"
          % g["seed"].nunique())
    print("  realisations.  RMS, not the mean of the magnitudes: at exact")
    print("  tuning the two branches are symmetric, a method returns one or")
    print("  the other with the noise deciding, and the root mean square is")
    print("  the one scalar that adds a bias and a scatter the way an")
    print("  engineer suffers them.  The signed mean and its scatter follow")
    print("  for the two readings whose difference is the whole question.")
    for dur in sorted(d["duration"].unique()):
        tune_at_tuning(d, dur)

    print("\n" + "=" * 78)
    print("4  WHICH LAW DOES EACH METHOD OBEY?")
    print("=" * 78)
    print("  Two distances, both in percentage points of tension.  BRANCH is")
    print("  the distance from the reading to the nearer of the two exact")
    print("  branch errors of the model, which is what 'the branch law")
    print("  returns in full' means for a method that has to pick one of two")
    print("  poles.  MERGED is the distance to the merged-peak prediction;")
    print("  at exact tuning, where the pair is symmetric and the sign of")
    print("  that prediction is a label, the nearer of the two signs is used.")
    gg = d[(d["arm"] == "grid") & (d["duration"] == 600.0)
           & (d["snr_db"] == 20.0)]
    print("  600 s at 20 dB, whole traverse, mean over %d records per cell"
          % gg.groupby(["zeta", "d"]).size().max())
    print("  %6s %6s | %-27s | %-27s"
          % ("zeta%", "u@0", "distance to a BRANCH", "distance to MERGED law"))
    tags4 = ("pp", "fdd", "efdd", "cov2", "cov1")
    print("  %6s %6s | %s | %s" % ("", "", " ".join("%5s" % t for t in tags4),
                                   " ".join("%5s" % t for t in tags4)))
    for z in sorted(gg["zeta"].unique()):
        s_ = gg[gg["zeta"] == z]
        u0 = s_[np.abs(s_["d"]) < 1e-6]["u"]
        row = "  %6.1f %6.2f |" % (100 * z, u0.iloc[0] if len(u0) else np.nan)
        for law in ("branch", "merged"):
            for tag in tags4:
                row += " %5.2f" % dev_from_law(s_, tag, law)
            row += " |" if law == "branch" else ""
        print(row)

    print("\n" + "=" * 78)
    print("5  THE DANGER BAND, MEASURED PER METHOD")
    print("=" * 78)
    print("  Half the peak-to-peak of the signed coupling error over the")
    print("  traverse, per cent: the swing a seasonal tension cycle would")
    print("  put through the reading.  It is formed WITHIN each realisation")
    print("  and then averaged, because near exact tuning the branch a")
    print("  method returns flips with the noise, and averaging the readings")
    print("  before taking the range would cancel the very excursion the")
    print("  band is meant to measure.  The two law columns are the same")
    print("  statistic evaluated on the model, on the same 17 tensions;")
    print("  run_dangerband uses 321 and reports 4 per cent more swing.")
    print("  branch value s = %.2f per cent" % (100 * d["s_tune"].iloc[0]))

    def swing(s, tag):
        vals = []
        for _, gsd in s.groupby("seed"):
            v = gsd.sort_values("d")["eps_%s_pct" % tag].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if len(v) > 2:
                vals.append(0.5 * (v.max() - v.min()))
        return _pm(vals)

    for dur in (600.0, 3600.0):
        q = d[(d["arm"] == "grid") & (d["duration"] == dur)
              & (d["snr_db"] == 20.0)]
        if not len(q):
            continue
        print("\n  %.0f s at 20 dB" % dur)
        print("  %6s %6s %9s %9s | %11s %11s %11s %11s"
              % ("zeta%", "u@0", "law:pick", "law:branch", "pp", "fdd",
                 "cov2 amp", "cov2 oracle"))
        for z in sorted(q["zeta"].unique()):
            s = q[q["zeta"] == z]
            m = s.groupby("d", as_index=False).mean(numeric_only=True)
            u0 = s[np.abs(s["d"]) < 1e-6]["u"]
            row = "  %6.1f %6.2f %9.3f %9.3f |" % (
                100 * z, u0.iloc[0] if len(u0) else np.nan,
                0.5 * (m["eps_mfrf_pct"].max() - m["eps_mfrf_pct"].min()),
                0.5 * (m["eps_branch_pct"].max() - m["eps_branch_pct"].min()))
            for tag in ("pp", "fdd", "cov2", "cov2_or"):
                row += " %11s" % swing(s, tag)
            print(row)

    print("\n" + "=" * 78)
    print("6  RECORD LENGTH AND NOISE")
    print("=" * 78)
    print("  fraction returning both branches, SSI-COV s+d / SSI-COV stay "
          "/ FDD")
    # the five tensions and three seeds every record length shares, so the
    # three columns differ in record length and in nothing else
    shr = d[d["d"].round(4).isin([round(v, 4) for v in D_CORE])
            & d["seed"].isin(SEEDS_SUB) & (d["snr_db"] == 20.0)]
    print("  %d tensions x %d seeds per cell" % (len(D_CORE), len(SEEDS_SUB)))
    print("  %6s %6s | %-16s %-16s %-16s"
          % ("zeta%", "u@0", "150 s", "600 s", "3600 s"))
    for z in sorted(shr["zeta"].unique()):
        row = "  %6.1f" % (100 * z)
        u0 = shr[(shr["zeta"] == z) & (np.abs(shr["d"]) < 1e-6)]["u"]
        row += " %6.2f |" % (u0.iloc[0] if len(u0) else np.nan)
        for dur in (150.0, 600.0, 3600.0):
            s = shr[(shr["zeta"] == z) & (shr["duration"] == dur)]
            if not len(s):
                row += " %-16s" % "  -"
                continue
            row += " %-16s" % ("%.0f/%.0f/%.0f%%"
                               % (100 * s["both_cov2"].mean(),
                                  100 * s["both_cov1"].mean(),
                                  100 * s["both_fdd"].mean()))
        print(row)

    nz = d[(d["arm"].isin(("grid", "noise"))) & (d["duration"] == 600.0)
           & (np.abs(d["d"]) <= 0.0041)]
    print("\n  against noise, 600 s at exact tuning (both branches, "
          "SSI-COV s+d / stay only)")
    snrs = sorted(nz["snr_db"].unique())
    print("  %6s |" % "zeta%" + "".join(" %11s" % ("%.0f dB" % s if
                                                   np.isfinite(s) else "clean")
                                        for s in snrs))
    for z in sorted(nz["zeta"].unique()):
        row = "  %6.1f |" % (100 * z)
        for s_ in snrs:
            t = nz[(nz["zeta"] == z) & (nz["snr_db"] == s_)]
            row += " %11s" % ("%.0f/%.0f%%" % (100 * t["both_cov2"].mean(),
                                               100 * t["both_cov1"].mean())
                              if len(t) else "-")
        print(row)

    print("\n" + "=" * 78)
    print("7  USING BOTH POLES: THE TRACE RULE")
    print("=" * 78)
    print("  RMS tension error, per cent, at exact tuning, 20 dB")
    for dur in sorted(d["duration"].unique()):
        q = d[(d["duration"] == dur) & (d["snr_db"] == 20.0)
              & (np.abs(d["d"]) < 1e-6)]
        if not len(q):
            continue
        print("\n  %.0f s" % dur)
        print("  %6s %10s %10s %10s %10s %10s %8s"
              % ("zeta%", "pp", "cov2 amp", "cov2 oracle", "cov2 trace",
                 "trace f_d+1%", "n trace"))
        for z in sorted(q["zeta"].unique()):
            s = q[q["zeta"] == z]
            print("  %6.1f %10.2f %10.2f %10.2f %10.2f %10.2f %8d"
                  % (100 * z, _rms(s["errT_pp_pct"]),
                     _rms(s["errT_cov2_pct"]), _rms(s["errT_cov2_or_pct"]),
                     _rms(s["errT_cov2_tr_pct"]),
                     _rms(s["errT_cov2_tr1_pct"]),
                     int(np.isfinite(s["errT_cov2_tr_pct"]).sum())))

    print("\n" + "=" * 78)
    print("8  THE TRACE RULE WITH A DECK FREQUENCY THE CAMPAIGN MEASURED")
    print("=" * 78)
    ft = field_trace(d[d["arm"] == "grid"])
    if len(ft):
        print("  deck frequency identified at the most detuned tension of "
              "each traverse")
        print("  its own error: %.3f +- %.3f per cent"
              % (ft["f_deck_err_pct"].mean(), ft["f_deck_err_pct"].std()))
        print("  %6s %8s | %-22s %-22s"
              % ("zeta%", "records", "|errT| at tuning, %", "|errT| whole "
                 "traverse, %"))
        for z in sorted(ft["zeta"].unique()):
            t = ft[(ft["zeta"] == z) & (ft["duration"] == 600.0)
                   & (ft["snr_db"] == 20.0)]
            tt = t[np.abs(t["d"]) < 1e-6]
            print("  %6.1f %8d | %-22s %-22s"
                  % (100 * z, len(t),
                     _pm(np.abs(tt["errT_pct"])) if len(tt) else "none",
                     _pm(np.abs(t["errT_pct"])) if len(t) else "none"))
    else:
        print("  no traverse had two poles at its most detuned tension")

    print("\n" + "=" * 78)
    print("9  WHAT THE DAMPING ESTIMATES DO AT THE CROSSING")
    print("=" * 78)
    print("  estimate divided by the true damping ratio, 600 s at 20 dB")
    print("  %6s %6s %14s %14s %14s"
          % ("zeta%", "u", "half power", "EFDD", "SSI-COV pole"))
    for z in sorted(g["zeta"].unique()):
        s_ = g[g["zeta"] == z]
        zc = (s_["z_cov2_p1"] / z).to_numpy(dtype=float)
        print("  %6.1f %6.2f %14s %14s %14s"
              % (100 * z, s_["u"].iloc[0], _pm(s_["zeta_hp_ratio"]),
                 _pm(s_["zeta_efdd_ratio"]), _pm(zc)))

    print("\n" + "=" * 78)
    print("9b IS ANY METHOD CLOSER TO THE TRUTH THAN PEAK PICKING?")
    print("=" * 78)
    print("  paired on the same record: fraction of records where the")
    print("  method's |tension error| is smaller than peak picking's, and")
    print("  the mean of |method| - |peak picking| in percentage points.")
    print("  600 s at 20 dB, whole traverse")
    print("  %-22s %10s %12s %12s" % ("", "closer", "mean gain", "at tuning"))
    for tag, name in (("fdd", "FDD"), ("efdd", "EFDD"),
                      ("cov2", "SSI-COV s+d, amplitude"),
                      ("cov2_or", "SSI-COV s+d, oracle"),
                      ("cov2_sh", "SSI-COV s+d, shape"),
                      ("cov2_tr", "SSI-COV s+d, trace"),
                      ("cov1", "SSI-COV stay, amplitude"),
                      ("dat2", "SSI-DATA s+d, amplitude")):
        a = np.abs(gg["errT_%s_pct" % tag].to_numpy(dtype=float))
        b = np.abs(gg["errT_pp_pct"].to_numpy(dtype=float))
        m = np.isfinite(a) & np.isfinite(b)
        t = gg[np.abs(gg["d"]) < 1e-6]
        at = np.abs(t["errT_%s_pct" % tag].to_numpy(dtype=float))
        bt = np.abs(t["errT_pp_pct"].to_numpy(dtype=float))
        mt = np.isfinite(at) & np.isfinite(bt)
        print("  %-22s %9.0f%% %12.2f %12.2f"
              % (name, 100 * np.mean(a[m] < b[m]), np.mean(a[m] - b[m]),
                 np.mean(at[mt] - bt[mt]) if mt.any() else np.nan))

    print("\n" + "=" * 78)
    print("10 WHAT FAILED")
    print("=" * 78)
    for tag, name in METHODS:
        col = "f_" + tag
        if col not in d:
            continue
        miss = 100.0 * (~np.isfinite(d[col])).mean()
        extra = ""
        if "fail_" + tag in d:
            nf = int((d["fail_" + tag].fillna("") != "").sum())
            extra = ", %d exceptions" % nf
        print("  %-20s returned nothing on %5.1f %% of records%s"
              % (name, miss, extra))
    print("  half power failed to find two -3 dB crossings on %.1f %%"
          % (100.0 * (~d["hp_ok"].astype(bool)).mean()))
    print("  EFDD bell fit failed on %.1f %%"
          % (100.0 * (~d["efdd_ok"].astype(bool)).mean()))
    return d


def merge_parts(parts, out="oma.csv"):
    """Concatenate the part files of a split run into one table."""
    d = pd.concat([pd.read_csv(os.path.join(DATA, p)) for p in parts],
                  ignore_index=True)
    d = d.sort_values(["arm", "duration", "zeta", "snr_db", "d", "seed"],
                      kind="mergesort").reset_index(drop=True)
    path = os.path.join(DATA, out)
    d.to_csv(path, index=False)
    print("  merged %d parts into %s (%d rows, %d columns)"
          % (len(parts), path, len(d), d.shape[1]))
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--campaign", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="a reduced design, for a smoke test")
    p.add_argument("--nproc", type=int, default=6)
    p.add_argument("--dur", action="append",
                   help="run only these record lengths, repeatable")
    p.add_argument("--zeta", action="append",
                   help="run only these damping ratios, repeatable")
    p.add_argument("--out", default="oma.csv")
    p.add_argument("--merge", nargs="+", metavar="PART",
                   help="concatenate part files into --out and report")
    args = p.parse_args()
    if args.merge:
        d = merge_parts(args.merge, args.out)
        report(d)
        return
    if not any((args.verify, args.campaign, args.report, args.all)):
        args.all = True
    if args.verify or args.all:
        verify(args)
    d = None
    if args.campaign or args.all:
        d = campaign(args)
    if args.report or args.all:
        if d is None:
            d = pd.read_csv(os.path.join(DATA, "oma.csv"))
        report(d)


if __name__ == "__main__":
    main()
