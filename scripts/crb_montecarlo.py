# -*- coding: utf-8 -*-
"""Cramer-Rao bound and Monte Carlo precision of stay tension at a crossing.

Fits a two-DOF stay-host model (see model_spectrum) to the stay-sensor
spectrum by Whittle maximum likelihood and reports the bound on T, the
profile-likelihood half-width, and the sample spread and bias of the estimate
for exact periodogram draws (arm 0), simulated records (arm A) and finite
element records (arm B). Writes data/crb_*.csv.
Run:  python3 scripts/crb_montecarlo.py --verify --bound --criterion
      python3 scripts/crb_montecarlo.py --arm0 --armA --armB
"""

from __future__ import annotations

import argparse
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")   # before numpy; avoids oversubscription

import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
from scipy.signal import fftconvolve, firwin, lfilter
from scipy.stats import kstest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from cablefe import tensioned_beam_freq                        # noqa: E402
from run_damping import BRIDGE, T_TUNE                         # noqa: E402
from simulate_records import (AA_CUTOFF_FRAC, AA_KAISER_BETA,   # noqa: E402
                              AA_TAPS_PER_R, FS_DEFAULT, FULL_SCALE_G,
                              G0, NBITS, OVERSAMPLE)

DATA = os.path.join(ROOT, "data")

# --- configuration -----------------------------------------------------------

FS = FS_DEFAULT                  # Hz, the logger rate of simulate_records
DUR = 600.0                      # s, record length (ten minutes)
BAND_HALF = 0.35                 # Hz either side of the pair midpoint
SENSOR_X = 2.0                   # m up the chord, as run_damping.py uses

# true values, calibrated against the finite element bridge (check [5])
F_HOST = 3.322130                # Hz, from the FE pair by the trace identity
S_TUNE = 0.023380                # coupling at exact tuning
RH = 0.0025                      # G_h / G_s, fitted to the FE spectrum
ZETA_REF = 0.005                 # damping the noise floor is calibrated at
PNR_REF_DB = 21.8                # peak over noise floor of the worked record

ZETAS = (0.002, 0.005, 0.010, 0.020, 0.040)
DGRID = (-0.060, -0.040, -0.025, -0.015, -0.008, -0.004, -0.002, 0.0,
         0.002, 0.004, 0.008, 0.015, 0.025, 0.040, 0.060)
DGRID_MC = (-0.040, -0.015, -0.008, -0.002, 0.0, 0.002, 0.008, 0.015, 0.040)

NREAL0 = 250                     # realizations per configuration, by arm
NREALA = 120
NREALB = 40
NPROC = max(1, (os.cpu_count() or 4) - 2)

MC, LC, EIC, THETA_DEG = BRIDGE["mc"], BRIDGE["Lc"], BRIDGE["EIc"], 35.0
MS_STAY = 0.5 * MC * LC


def _c_b(x_p=SENSOR_X, n=1):
    """Observation gain factor c_b in rb = c_b s.

    c_b = -n pi (1 - x_p/L_c) / (2 sqrt(M_s) phi_s(x_p)), from the kinematic
    tie v_stay(0) = cos(theta) w_deck at the anchorage. The sign is negative
    because the model writes this inertia coupling as a positive stiffness
    coupling kappa. check_cb() compares c_b with the finite element shapes.
    """
    phis = np.sqrt(2.0 / (MC * LC)) * np.sin(n * np.pi * x_p / LC)
    return -n * np.pi * (1.0 - x_p / LC) / (2.0 * np.sqrt(MS_STAY) * phis)


CB = _c_b()


def T_of_fs(f_s):
    """Tension of a pinned-pinned tensioned beam with fundamental ``f_s``."""
    return 4.0 * MC * LC ** 2 * np.asarray(f_s) ** 2 \
        - EIC * (np.pi / LC) ** 2


def fs_of_T(T):
    """Inverse of :func:`T_of_fs`, the isolated stay frequency f_iso.

    No dtype in ``np.asarray``, so a complex tension passes through for the
    complex-step derivatives.
    """
    return tensioned_beam_freq(1, LC, np.asarray(T), EIC, MC)


# --- the model ---------------------------------------------------------------

# parameter vector theta; rb = c_b s + rb_extra
PARAMS = ("T", "f_h", "s", "zeta", "lnGs", "lnGn", "rb_extra")
# sets that also get a profile-likelihood half-width
PROFILE_SETS = ("host_unknown", "host_unknown_rb")
# free parameters for each state of knowledge about the host
FREE_SETS = {
    "host_known": ("T", "zeta", "lnGs", "lnGn"),
    "fh_known": ("T", "s", "zeta", "lnGs", "lnGn"),
    "host_unknown": ("T", "f_h", "s", "zeta", "lnGs", "lnGn"),
    "host_unknown_rb": ("T", "f_h", "s", "zeta", "lnGs", "lnGn", "rb_extra"),
}


def _kernel(theta, cb):
    """Frequency-independent quantities of the model, shared by every call.

    Runs unchanged on a complex parameter vector (no branch, abs or comparison
    on the differentiated path), so the information matrix can be built by
    complex-step differentiation. Central differences cannot resolve the
    smallest scaled eigenvalue, about 1e-7 of the largest (check [2]).
    """
    T, f_h, s, zeta = theta[0], theta[1], theta[2], theta[3]
    ws2 = (2.0 * np.pi * fs_of_T(T)) ** 2
    wh2 = (2.0 * np.pi * f_h) ** 2
    w0sq = 0.5 * (ws2 + wh2)
    kap = s * w0sq
    half = np.sqrt((0.5 * (ws2 - wh2)) ** 2 + kap ** 2)
    wp = np.sqrt(w0sq + half)
    wm = np.sqrt(w0sq - half)
    alpha = 2.0 * zeta * wp * wm / (wp + wm)
    beta = 2.0 * zeta / (wp + wm)
    rb = cb * s + theta[6]
    return ws2, wh2, kap, alpha, beta, rb


def model_spectrum(f, theta, cb=CB, rh=RH):
    """One-sided acceleration PSD at the stay sensor.

    Model: u'' + C u' + K u = p, K = [[w_s^2, kappa], [kappa, w_h^2]],
    kappa = s (w_s^2 + w_h^2) / 2, Rayleigh C with damping zeta on both modes,
    white forces G_s and rh G_s, sensor y = u_s'' + rb u_h'' + noise G_n:
    S = w^4 (|A22 - rb A12|^2 + rh |A12 - rb A11|^2) G_s / |det A|^2 + G_n,
    A = K - w^2 I + i w C. Written in real arithmetic for speed.
    """
    ws2, wh2, kap, alpha, beta, rb = _kernel(theta, cb)
    om = 2.0 * np.pi * np.asarray(f, float)
    om2 = om * om
    A11r, A11i = ws2 - om2, om * (alpha + beta * ws2)
    A22r, A22i = wh2 - om2, om * (alpha + beta * wh2)
    A12r, A12i = kap, kap * beta * om
    dr = A11r * A22r - A11i * A22i - (A12r * A12r - A12i * A12i)
    di = A11r * A22i + A11i * A22r - 2.0 * A12r * A12i
    n1r, n1i = A22r - rb * A12r, A22i - rb * A12i
    num = n1r * n1r + n1i * n1i
    if rh:
        n2r, n2i = rb * A11r - A12r, rb * A11i - A12i
        num = num + rh * (n2r * n2r + n2i * n2i)
    return om2 * om2 * num * np.exp(theta[4]) / (dr * dr + di * di) \
        + np.exp(theta[5])


def model_spectrum_2ch(f, theta, cb=CB, rh=RH, gd=1.0):
    """2 by 2 one-sided spectral matrix of the stay and deck accelerations.

    The deck channel reads the host coordinate with gain ``gd`` and does not
    see the stay coordinate, which has a node at the anchorage. Sensor noise
    is independent between channels and at the same level.
    """
    ws2, wh2, kap, alpha, beta, rb = _kernel(theta, cb)
    om = 2.0 * np.pi * np.asarray(f, float)
    A11 = ws2 - om ** 2 + 1j * om * (alpha + beta * ws2)
    A22 = wh2 - om ** 2 + 1j * om * (alpha + beta * wh2)
    A12 = kap * (1.0 + 1j * beta * om)
    det = A11 * A22 - A12 ** 2
    Hs1 = (A22 - rb * A12) / det          # stay channel, stay force
    Hs2 = (rb * A11 - A12) / det          # stay channel, host force
    Hd1 = gd * (-A12) / det               # deck channel, stay force
    Hd2 = gd * A11 / det                  # deck channel, host force
    w4 = (om ** 2) ** 2
    Gs, Gn = np.exp(theta[4]), np.exp(theta[5])
    S = np.empty((len(om), 2, 2), dtype=complex)
    S[:, 0, 0] = w4 * Gs * (np.abs(Hs1) ** 2 + rh * np.abs(Hs2) ** 2) + Gn
    S[:, 1, 1] = w4 * Gs * (np.abs(Hd1) ** 2 + rh * np.abs(Hd2) ** 2) + Gn
    S[:, 0, 1] = w4 * Gs * (Hs1 * np.conj(Hd1) + rh * Hs2 * np.conj(Hd2))
    S[:, 1, 0] = np.conj(S[:, 0, 1])
    return S


def pair_from_theta(theta):
    """(f_lo, f_hi, f0, d) of the coupled pair, for reporting."""
    ws2 = (2.0 * np.pi * fs_of_T(theta[0])) ** 2
    wh2 = (2.0 * np.pi * theta[1]) ** 2
    w0sq = 0.5 * (ws2 + wh2)
    half = np.sqrt((0.5 * (ws2 - wh2)) ** 2 + (theta[2] * w0sq) ** 2)
    f_lo = np.sqrt(max(w0sq - half, 1e-9)) / (2 * np.pi)
    f_hi = np.sqrt(w0sq + half) / (2 * np.pi)
    return f_lo, f_hi, 0.5 * (f_lo + f_hi), 0.5 * (ws2 - wh2) / w0sq


def theta_from_d(d, s=S_TUNE, zeta=ZETA_REF, f_h=F_HOST, lnGs=0.0,
                 lnGn=-30.0, rb_extra=0.0):
    """Parameter vector at detuning d = (f_iso - f_host) / f_iso."""
    return np.array([T_of_fs(f_h / (1.0 - d)), f_h, s, zeta, lnGs, lnGn,
                     rb_extra])


def band_grid(theta, dur=DUR, fs=FS, half=BAND_HALF):
    """The Fourier bins the likelihood is evaluated on."""
    _, _, f0, _ = pair_from_theta(theta)
    n = int(round(dur * fs))
    f = np.fft.rfftfreq(n, 1.0 / fs)[1:-1]
    return f[(f >= f0 - half) & (f <= f0 + half)]


def set_levels(theta, pnr_ref_db=PNR_REF_DB, dur=DUR, fs=FS, half=BAND_HALF,
               cb=CB, rh=RH):
    """Set G_s to unity and G_n to an absolute noise floor.

    The floor gives peak-to-noise ratio ``pnr_ref_db`` at the reference
    damping ZETA_REF and is then held as zeta varies, so the instrument noise
    does not change with damping. The common scale cancels from every result.
    """
    th = np.array(theta, dtype=float)
    th[4], th[5] = 0.0, -60.0
    ref = np.array(th)
    ref[3] = ZETA_REF
    peak = model_spectrum(band_grid(ref, dur, fs, half), ref, cb, rh).max()
    th[5] = np.log(peak * 10.0 ** (-pnr_ref_db / 10.0))
    return th


# --- the Whittle likelihood --------------------------------------------------

def periodogram(y, fs):
    """One-sided periodogram with E[I_k] = S(f_k), positive frequencies."""
    n = len(y)
    Y = np.fft.rfft(y - y.mean())
    return (np.fft.rfftfreq(n, 1.0 / fs)[1:-1],
            ((2.0 / (fs * n)) * np.abs(Y) ** 2)[1:-1])


def periodogram_2ch(Y2, fs):
    """2 by 2 periodogram matrices from a (2, n) record."""
    n = Y2.shape[1]
    Z = np.fft.rfft(Y2 - Y2.mean(axis=1, keepdims=True), axis=1)[:, 1:-1]
    return (np.fft.rfftfreq(n, 1.0 / fs)[1:-1],
            (2.0 / (fs * n)) * np.einsum('ik,jk->kij', Z, np.conj(Z)))


def nll(theta, f, I, cb=CB, rh=RH):
    """Negative Whittle log-likelihood, constants dropped."""
    with np.errstate(all="ignore"):
        S = model_spectrum(f, theta, cb, rh)
        if not np.all(np.isfinite(S)) or np.any(S <= 0):
            return 1e300
        v = float(np.sum(np.log(S) + I / S))
    return v if np.isfinite(v) else 1e300


def nll_2ch(theta, f, I, cb=CB, rh=RH):
    with np.errstate(all="ignore"):
        S = model_spectrum_2ch(f, theta, cb, rh)
        det = (S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] * S[:, 1, 0]).real
        if np.any(det <= 0) or not np.all(np.isfinite(det)):
            return 1e300
        tr = (S[:, 1, 1] * I[:, 0, 0] - S[:, 1, 0] * I[:, 0, 1]
              - S[:, 0, 1] * I[:, 1, 0] + S[:, 0, 0] * I[:, 1, 1]).real / det
        v = float(np.sum(np.log(det) + tr))
    return v if np.isfinite(v) else 1e300


# --- fitting -----------------------------------------------------------------

LOGGED = ("T", "s", "zeta")


def bounds_for(theta_ref):
    f0 = fs_of_T(theta_ref[0])
    return {"T": (T_of_fs(f0 - 0.60), T_of_fs(f0 + 0.60)),
            "f_h": (f0 - 0.60, f0 + 0.60),
            "s": (1.0e-5, 0.30),
            "zeta": (1.0e-4, 0.20),
            "lnGs": (theta_ref[4] - 12.0, theta_ref[4] + 12.0),
            "lnGn": (theta_ref[5] - 25.0, theta_ref[5] + 25.0),
            "rb_extra": (-2.0, 2.0)}


def _to_x(theta, free):
    return np.array([np.log(theta[PARAMS.index(n)]) if n in LOGGED
                     else theta[PARAMS.index(n)] for n in free])


def _from_x(x, free, base):
    theta = np.array(base, dtype=float)
    for xi, n in zip(x, free):
        theta[PARAMS.index(n)] = np.exp(xi) if n in LOGGED else xi
    return theta


def _xbounds(free, bnds):
    out = []
    for n in free:
        lo, hi = bnds[n]
        out.append((np.log(lo), np.log(hi)) if n in LOGGED else (lo, hi))
    return out


def fit_once(f, I, start, free, base, bnds, cb=CB, rh=RH, two_ch=False):
    obj = nll_2ch if two_ch else nll
    xb = _xbounds(free, bnds)
    x0 = np.clip(_to_x(start, free), [b[0] for b in xb], [b[1] for b in xb])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(lambda x: obj(_from_x(x, free, base), f, I, cb, rh),
                       x0, method="L-BFGS-B", bounds=xb,
                       options=dict(maxiter=1200, maxfun=12000,
                                    ftol=1e-15, gtol=1e-12))
    at_bound = any(min(abs(xi - lo), abs(xi - hi))
                   < 1e-6 * (1.0 + abs(xi)) for xi, (lo, hi)
                   in zip(res.x, xb))
    return dict(theta=_from_x(res.x, free, base), nll=float(res.fun),
                ok=bool(res.success), at_bound=bool(at_bound))


def peak_start(f, I, theta_true, nsm=9):
    """A data-driven start: two smoothed maxima and the notch between them."""
    sm = np.convolve(I, np.ones(nsm) / nsm, mode="same")
    k = int(np.argmax(sm))
    f_pk = f[k]
    mask = np.abs(f - f_pk) > 3 * nsm * (f[1] - f[0])
    f2 = f[mask][int(np.argmax(sm[mask]))] if mask.any() else f_pk
    lo, hi = min(f_pk, f2), max(f_pk, f2)
    inner = (f > lo) & (f < hi)
    f_notch = (f[inner][int(np.argmin(sm[inner]))] if inner.sum() > 3
               else 0.5 * (lo + hi))
    f0 = 0.5 * (lo + hi)
    ws2 = (2 * np.pi * lo) ** 2 + (2 * np.pi * hi) ** 2 \
        - (2 * np.pi * f_notch) ** 2
    f_s = np.sqrt(max(ws2, (2 * np.pi * 0.5) ** 2)) / (2 * np.pi)
    th = np.array(theta_true, dtype=float)
    th[0] = T_of_fs(np.clip(f_s, f0 - 0.5, f0 + 0.5))
    th[1] = np.clip(f_notch, f0 - 0.5, f0 + 0.5)
    th[2] = np.clip(max((hi - lo) / f0, 1e-4), 1e-4, 0.2)
    return th


def fit_mle(f, I, theta_true, free, cb=CB, rh=RH, two_ch=False):
    """Multi-start maximum likelihood fit with bimodality diagnostics.

    Four starts: the truth (exposes optimizer failures), its mirror with stay
    and host swapped (the competing explanation of the two peaks), a
    data-driven start from the peaks and the notch, and a start with the
    coupling ten times too small.
    """
    base = np.array(theta_true, dtype=float)
    bnds = bounds_for(theta_true)
    f_s_t = fs_of_T(theta_true[0])
    th_mirror = np.array(theta_true, dtype=float)
    th_mirror[0] = T_of_fs(theta_true[1])       # stay and host swapped
    th_mirror[1] = f_s_t
    th_small = np.array(theta_true, dtype=float)
    th_small[2] = max(theta_true[2] * 0.1, 1e-4)

    starts = [("truth", theta_true), ("mirror", th_mirror),
              ("peaks", peak_start(f, I, theta_true)), ("small_s", th_small)]
    fits = {}
    for name, st in starts:
        fits[name] = fit_once(f, I, st, free, base, bnds, cb, rh, two_ch)
    best_name = min(fits, key=lambda k: fits[k]["nll"])
    best = fits[best_name]

    order = sorted(fits.values(), key=lambda r: r["nll"])
    runner = next((r for r in order[1:]
                   if abs(r["theta"][0] - best["theta"][0])
                   > 0.005 * abs(best["theta"][0])), None)
    return dict(T_hat=best["theta"][0], nll_best=best["nll"],
                theta=best["theta"], start_won=best_name, ok=best["ok"],
                at_bound=best["at_bound"],
                n_fail=sum(0 if r["ok"] else 1 for r in fits.values()),
                n_bound=sum(1 if r["at_bound"] else 0 for r in fits.values()),
                truth_gap=fits["truth"]["nll"] - best["nll"],
                mirror_gap=fits["mirror"]["nll"] - best["nll"],
                peaks_gap=fits["peaks"]["nll"] - best["nll"],
                runner_gap=(runner["nll"] - best["nll"]) if runner else np.inf,
                runner_T=runner["theta"][0] if runner else np.nan)


# --- Fisher information ------------------------------------------------------

FD_REL = {"T": 3e-5, "f_h": 3e-6, "s": 3e-4, "zeta": 3e-4,
          "lnGs": 1e-4, "lnGn": 1e-4, "rb_extra": 3e-4}
CS_H = 1e-30                      # complex step; exact, no cancellation


def _steps(theta, free, scale=1.0):
    """Central-difference steps for the observed Hessian and check [2].

    The observed likelihood contains the periodogram, which is data, so the
    complex step does not apply to it.
    """
    h = []
    for n in free:
        r = FD_REL[n] * scale
        if n in ("lnGs", "lnGn"):
            h.append(r * 1e4)
        elif n == "rb_extra":
            h.append(r * 1e2)
        else:
            h.append(r * abs(theta[PARAMS.index(n)]))
    return np.array(h)


def _dlogS(theta, f, free, cb, rh, scale=1.0, method="cs"):
    """d log S / d theta at every bin.

    ``method="cs"`` uses the complex step Im log S(theta + i h) / h, exact
    to machine precision; ``method="fd"`` uses central differences (check [2]).
    """
    if method == "fd":
        h = _steps(theta, free, scale)
        D = np.empty((len(free), len(f)))
        for i, n in enumerate(free):
            j = PARAMS.index(n)
            tp = np.array(theta); tp[j] += h[i]
            tm = np.array(theta); tm[j] -= h[i]
            D[i] = (np.log(model_spectrum(f, tp, cb, rh))
                    - np.log(model_spectrum(f, tm, cb, rh))) / (2 * h[i])
        return D
    D = np.empty((len(free), len(f)))
    for i, n in enumerate(free):
        tp = np.array(theta, dtype=complex)
        tp[PARAMS.index(n)] += 1j * CS_H
        D[i] = np.imag(np.log(model_spectrum(f, tp, cb, rh))) / CS_H
    return D


def fim_expected(theta, f, free, cb=CB, rh=RH, scale=1.0, two_ch=False,
                 gd=1.0, method="cs"):
    """Expected Whittle information matrix.

    One channel: sum_k dlog S_k dlog S_k^T. Two channels: the Slepian-Bangs
    form sum_k tr(S^-1 dS_i S^-1 dS_j), with central differences.
    """
    if not two_ch:
        D = _dlogS(theta, f, free, cb, rh, scale, method)
        return D @ D.T
    h = _steps(theta, free, scale)
    S = model_spectrum_2ch(f, theta, cb, rh, gd)
    det = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] * S[:, 1, 0]
    Si = np.empty_like(S)
    Si[:, 0, 0] = S[:, 1, 1] / det
    Si[:, 1, 1] = S[:, 0, 0] / det
    Si[:, 0, 1] = -S[:, 0, 1] / det
    Si[:, 1, 0] = -S[:, 1, 0] / det
    A = []
    for i, n in enumerate(free):
        j = PARAMS.index(n)
        tp = np.array(theta); tp[j] += h[i]
        tm = np.array(theta); tm[j] -= h[i]
        dS = (model_spectrum_2ch(f, tp, cb, rh, gd)
              - model_spectrum_2ch(f, tm, cb, rh, gd)) / (2 * h[i])
        A.append(Si @ dS)
    m = len(free)
    F = np.empty((m, m))
    for i in range(m):
        for j in range(i, m):
            F[i, j] = F[j, i] = np.real(
                np.trace(A[i] @ A[j], axis1=1, axis2=2)).sum()
    return F


def fim_observed(theta, f, I, free, cb=CB, rh=RH, scale=1.0, two_ch=False):
    """Minus the finite-difference Hessian of the log-likelihood."""
    h = _steps(theta, free, scale)
    obj = nll_2ch if two_ch else nll
    idx = [PARAMS.index(n) for n in free]
    m = len(free)
    f0 = obj(theta, f, I, cb, rh)
    H = np.empty((m, m))
    for i in range(m):
        for j in range(i, m):
            tpp = np.array(theta); tpp[idx[i]] += h[i]; tpp[idx[j]] += h[j]
            tmm = np.array(theta); tmm[idx[i]] -= h[i]; tmm[idx[j]] -= h[j]
            if i == j:
                v = (obj(tpp, f, I, cb, rh) - 2 * f0
                     + obj(tmm, f, I, cb, rh)) / (4 * h[i] * h[i])
            else:
                tpm = np.array(theta); tpm[idx[i]] += h[i]; tpm[idx[j]] -= h[j]
                tmp = np.array(theta); tmp[idx[i]] -= h[i]; tmp[idx[j]] += h[j]
                v = (obj(tpp, f, I, cb, rh) - obj(tpm, f, I, cb, rh)
                     - obj(tmp, f, I, cb, rh) + obj(tmm, f, I, cb, rh)) \
                    / (4 * h[i] * h[j])
            H[i, j] = H[j, i] = v
    return H


SING_TOL = 1e-11                  # relative eigenvalue below which the
                                  # information matrix is called singular


def crb_T(F, tol=SING_TOL):
    """Bound on T in newtons squared, and the scaled condition number.

    The matrix is rescaled by its diagonal before inversion to remove the
    spread of units. An eigenvalue below ``tol`` times the largest is treated
    as zero, and the bound is then returned as infinite.
    """
    d = np.diag(F).copy()
    if np.any(d <= 0) or not np.all(np.isfinite(F)):
        return np.inf, np.inf
    sc = 1.0 / np.sqrt(d)
    Fs = F * np.outer(sc, sc)
    w, V = np.linalg.eigh(Fs)
    cond = abs(w[-1] / w[0]) if w[0] != 0 else np.inf
    if w[0] <= tol * w[-1]:
        return np.inf, np.inf
    v = float((V[0] ** 2 / w).sum()) * sc[0] ** 2
    return (v if v > 0 else np.inf), cond


# --- expected log-likelihood ratio and profile half-width --------------------

def expected_lr(theta, theta0, f, cb=CB, rh=RH):
    """E[l(theta0) - l(theta)] for the Whittle likelihood, exactly.

    This is the Kullback-Leibler divergence sum_k [log(S/S_0) + S_0/S - 1].
    It stays finite where the information matrix is singular.
    """
    S = model_spectrum(f, theta, cb, rh)
    S0 = model_spectrum(f, theta0, cb, rh)
    r = S0 / S
    return float(np.sum(np.log(1.0 / r) + r - 1.0))


def profile_halfwidth_T(theta0, f, free, cb=CB, rh=RH, level=0.5,
                        rel_max=0.30, start=2e-4):
    """Half-width in T of the expected profile likelihood at ``level``.

    The range of T over which the expected log-likelihood ratio, minimized
    over the other free parameters, stays below ``level`` (0.5 gives one
    standard error). Returns the mean relative half-width and the upper and
    lower sides.
    """
    others = [n for n in free if n != "T"]
    bnds = bounds_for(theta0)
    xb = _xbounds(others, bnds)
    warm = {}

    def prof(rel, sgn):
        base = np.array(theta0, dtype=float)
        base[0] = theta0[0] * (1.0 + sgn * rel)
        if not others:
            return expected_lr(base, theta0, f, cb, rh)
        x0 = warm.get(sgn, _to_x(theta0, others))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = minimize(lambda x: expected_lr(_from_x(x, others, base),
                                               theta0, f, cb, rh),
                         x0, method="L-BFGS-B", bounds=xb,
                         options=dict(maxiter=400, ftol=1e-13))
        warm[sgn] = r.x
        return float(r.fun)

    out = []
    for sgn in (+1.0, -1.0):
        warm.pop(sgn, None)
        lo, hi = 0.0, start
        while prof(hi, sgn) < level and hi < rel_max:
            lo, hi = hi, hi * 2.0
        if hi >= rel_max:
            out.append(np.nan)
            continue
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            if prof(mid, sgn) < level:
                lo = mid
            else:
                hi = mid
        out.append(0.5 * (lo + hi))
    return float(np.nanmean(out)), out[0], out[1]


def bound_row(theta, dur=DUR, fs=FS, half=BAND_HALF, cb=CB, rh=RH,
              two_ch=False, profile=True):
    """Every bound for one configuration, as a dict."""
    f = band_grid(theta, dur, fs, half)
    S = model_spectrum(f, theta, cb, rh)
    f_lo, f_hi, f0, d = pair_from_theta(theta)
    Slo = float(model_spectrum(np.array([f_lo]), theta, cb, rh)[0])
    Shi = float(model_spectrum(np.array([f_hi]), theta, cb, rh)[0])
    Gn = np.exp(theta[5])
    weak = min(Slo, Shi) - Gn
    row = dict(T=theta[0], f_s=fs_of_T(theta[0]), f_h=theta[1], s=theta[2],
               zeta=theta[3], d=d, r_res=theta[2] / (2 * theta[3]),
               dur=dur, fs=fs, band_half=half, nbins=len(f), cb=cb, rh=rh,
               pnr_db=10 * np.log10(S.max() / np.exp(theta[5])),
               f_lo=f_lo, f_hi=f_hi, f0=f0,
               eps=np.sqrt(d ** 2 + theta[2] ** 2) - abs(d),
               S_lo=Slo, S_hi=Shi,
               weak_peak_snr_db=(10 * np.log10(weak / Gn) if weak > 0
                                 else -np.inf),
               peak_ratio_db=10 * np.log10(max(Slo, Shi) / max(Slo, Shi)
                                           if weak <= 0 else
                                           (max(Slo, Shi) - Gn) / weak))
    for key, free in FREE_SETS.items():
        v, cond = crb_T(fim_expected(theta, f, free, cb, rh, two_ch=two_ch))
        row[f"cv_{key}"] = np.sqrt(v) / theta[0] if np.isfinite(v) else np.inf
        row[f"cond_{key}"] = cond
        if profile and not two_ch and key in PROFILE_SETS:
            hw, hp, hm = profile_halfwidth_T(theta, f, free, cb, rh)
            row[f"prof_{key}"] = hw
            row[f"prof_up_{key}"] = hp
            row[f"prof_dn_{key}"] = hm
    row["ratio_unknown_known"] = (row["cv_host_unknown"]
                                  / row["cv_host_known"]) ** 2
    row["ratio_rb_known"] = (row["cv_host_unknown_rb"]
                             / row["cv_host_known"]) ** 2
    return row


# --- arm A: sampled records of the two-DOF model -----------------------------

class TwoDofRecorder:
    """Sampled acceleration records of the two-DOF model.

    Uses the measurement chain of simulate_records: exact zero-order hold on
    each modal oscillator at ``oversample`` times the logger rate, a
    linear-phase Kaiser anti-alias filter at 0.4 fs with its group delay
    removed, decimation, white sensor noise and uniform quantization.
    Rayleigh damping is classical, so the modal form is exact.
    """

    def __init__(self, theta, cb=CB, rh=RH, fs=FS, oversample=OVERSAMPLE):
        from scipy.linalg import expm
        from scipy.signal import ss2tf
        self.theta = np.array(theta, dtype=float)
        self.cb, self.rh = cb, rh
        self.fs, self.oversample = float(fs), int(oversample)
        self.fs_int = self.fs * self.oversample
        self.dt = 1.0 / self.fs_int
        ws2, wh2, kap, alpha, beta, rb = _kernel(self.theta, cb)
        lam, V = np.linalg.eigh(np.array([[ws2, kap], [kap, wh2]]))
        self.w = np.sqrt(np.maximum(lam, 1e-9))
        self.zj = 0.5 * (alpha / self.w + beta * self.w)
        Gs = np.exp(self.theta[4])
        self.Lf = V.T @ np.diag(np.sqrt(0.5 * np.array([Gs, rh * Gs])))
        self.obs = np.array([1.0, rb]) @ V      # y = u_s'' + rb u_h''
        self.Gn = np.exp(self.theta[5])
        self.b = np.zeros((2, 3))
        self.a = np.zeros((2, 3))
        for j in range(2):
            w, z = self.w[j], self.zj[j]
            aug = np.zeros((3, 3))
            aug[:2, :2] = np.array([[0.0, 1.0], [-w * w, -2 * z * w]]) * self.dt
            aug[:2, 2:] = np.array([[0.0], [1.0]]) * self.dt
            E = expm(aug)
            num, den = ss2tf(E[:2, :2], E[:2, 2:],
                             np.array([[-w * w, -2 * z * w]]),
                             np.array([[1.0]]))
            self.b[j], self.a[j] = num[0], den
        self.aa = firwin(AA_TAPS_PER_R * self.oversample + 1,
                         AA_CUTOFF_FRAC * self.fs,
                         window=("kaiser", AA_KAISER_BETA), fs=self.fs_int)
        self.burn_in = 8.0 / np.min(self.zj * self.w)

    def record(self, duration=DUR, seed=0, quantise=True, noise=True,
               target_rms_mg=5.0):
        rng = np.random.default_rng(seed)
        nkeep = int(round(duration * self.fs))
        pad = len(self.aa) // 2 + self.oversample
        nburn = int(round(self.burn_in * self.fs_int))
        nint = nburn + nkeep * self.oversample + 2 * pad
        F = self.Lf @ (rng.standard_normal((2, nint)) / np.sqrt(self.dt))
        Y = np.empty_like(F)
        for j in range(2):
            Y[j] = lfilter(self.b[j], self.a[j], F[j])
        af = fftconvolve(self.obs @ Y, self.aa, mode="same")
        y = af[nburn + pad:nburn + pad + nkeep * self.oversample][
            ::self.oversample]
        if noise:
            y = y + np.sqrt(self.Gn * self.fs / 2.0) \
                * rng.standard_normal(len(y))
        if quantise:
            sc = target_rms_mg * 1e-3 * G0 / y.std()
            lsb = 2.0 * FULL_SCALE_G * G0 / 2 ** NBITS
            y = np.round(y * sc / lsb) * lsb / sc
        return y


# --- Monte Carlo arms --------------------------------------------------------

def _one_realisation(job):
    """One record, one set of fits.  Top level so it can be pickled."""
    (arm, theta, cb, rh, dur, fs, half, seed, free_keys, T_true,
     T_pseudo) = job
    th = np.array(theta, dtype=float)
    if arm == "arm0":
        f = band_grid(th, dur, fs, half)
        S = model_spectrum(f, th, cb, rh)
        I = S * np.random.default_rng(seed).exponential(size=len(f))
    elif arm == "armA":
        rec = TwoDofRecorder(th, cb, rh, fs).record(dur, seed=seed)
        fa, Ia = periodogram(rec, fs)
        _, _, f0, _ = pair_from_theta(th)
        m = (fa >= f0 - half) & (fa <= f0 + half)
        f, I = fa[m], Ia[m]
    else:
        raise ValueError(arm)
    out = dict(seed=seed, T_true=T_true, T_pseudo=T_pseudo)
    for key in free_keys:
        r = fit_mle(f, I, th, FREE_SETS[key], cb, rh)
        Fo = fim_observed(r["theta"], f, I, FREE_SETS[key], cb, rh)
        v, cond = crb_T(Fo)
        out.update({f"T_{key}": r["T_hat"], f"ok_{key}": r["ok"],
                    f"bound_{key}": r["at_bound"],
                    f"nbound_{key}": r["n_bound"],
                    f"nfail_{key}": r["n_fail"],
                    f"mirror_gap_{key}": r["mirror_gap"],
                    f"peaks_gap_{key}": r["peaks_gap"],
                    f"runner_gap_{key}": r["runner_gap"],
                    f"runner_T_{key}": r["runner_T"],
                    f"won_{key}": r["start_won"],
                    f"s_{key}": r["theta"][2], f"zeta_{key}": r["theta"][3],
                    f"fh_{key}": r["theta"][1],
                    f"crbobs_{key}": np.sqrt(v) / T_true
                    if np.isfinite(v) else np.inf})
    return out


def _summarise(draws, theta, cb, rh, dur, fs, half, free_keys, T_true,
               T_pseudo, arm):
    d = pd.DataFrame(draws)
    row = bound_row(theta, dur, fs, half, cb, rh)
    row.update(arm=arm, nreal=len(d), T_true=T_true, T_pseudo=T_pseudo)
    for key in free_keys:
        t = d[f"T_{key}"].to_numpy()
        var = float(np.var(t, ddof=1))
        crb = (row[f"cv_{key}"] * T_true) ** 2
        row[f"samp_cv_{key}"] = np.sqrt(var) / T_true
        row[f"var_over_crb_{key}"] = var / crb if np.isfinite(crb) else np.nan
        row[f"bias_pct_{key}"] = 100 * (t.mean() - T_true) / T_true
        row[f"bias_pseudo_pct_{key}"] = 100 * (t.mean() - T_pseudo) / T_pseudo
        row[f"rmse_pct_{key}"] = 100 * np.sqrt(
            np.mean((t - T_true) ** 2)) / T_true
        row[f"med_pct_{key}"] = 100 * (np.median(t) - T_true) / T_true
        row[f"frac_fail_{key}"] = float((d[f"nfail_{key}"] > 0).mean())
        row[f"frac_bound_{key}"] = float(d[f"bound_{key}"].mean())
        row[f"frac_truthlost_{key}"] = float((d[f"won_{key}"] != "truth").mean())
        row[f"frac_bimodal_{key}"] = float(
            (d[f"runner_gap_{key}"] < 2.0).mean())
        row[f"med_mirror_gap_{key}"] = float(d[f"mirror_gap_{key}"].median())
        row[f"crbobs_med_{key}"] = float(np.median(d[f"crbobs_{key}"]))
        # dispersion robust to bimodal fits
        q = np.percentile(t, [15.865, 84.135])
        row[f"robust_cv_{key}"] = 0.5 * (q[1] - q[0]) / T_true
    return row, d


def run_mc(arm, configs, free_keys, nreal, tag, nproc=NPROC):
    """Run one arm over a list of (theta, cb, rh, dur, fs, half, T_true,
    T_pseudo, label) configurations."""
    rows, alldraws = [], []
    t0 = time.time()
    for ci, cfg in enumerate(configs):
        theta, cb, rh, dur, fs, half, T_true, T_pseudo, label = cfg
        jobs = [(arm, theta, cb, rh, dur, fs, half, 10000 * ci + k,
                 free_keys, T_true, T_pseudo) for k in range(nreal)]
        if nproc > 1:
            with Pool(nproc) as p:
                draws = p.map(_one_realisation, jobs, chunksize=4)
        else:
            draws = [_one_realisation(j) for j in jobs]
        row, dd = _summarise(draws, theta, cb, rh, dur, fs, half, free_keys,
                             T_true, T_pseudo, arm)
        row["label"] = label
        rows.append(row)
        dd["label"] = label
        dd["d"] = row["d"]
        dd["zeta"] = row["zeta"]
        alldraws.append(dd)
        k0 = free_keys[-1]
        print(f"  {label:26s} d {row['d']:+.4f} zeta {100*row['zeta']:5.2f} % "
              f"| CRB {100*row[f'cv_{k0}']:.4f} %  sample "
              f"{100*row[f'samp_cv_{k0}']:.4f} %  var/CRB "
              f"{row[f'var_over_crb_{k0}']:6.2f}  bias "
              f"{row[f'bias_pct_{k0}']:+.3f} %  bimodal "
              f"{row[f'frac_bimodal_{k0}']:.2f}  bound "
              f"{row[f'frac_bound_{k0}']:.2f}")
    print(f"  ({time.time()-t0:.0f} s)")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA, f"crb_{tag}.csv"), index=False)
    pd.concat(alldraws, ignore_index=True).to_csv(
        os.path.join(DATA, f"crb_{tag}_draws.csv"), index=False)
    print(f"  wrote data/crb_{tag}.csv and data/crb_{tag}_draws.csv")
    return out


# --- arm B: finite element records fitted with the two-DOF model -------------

def fe_calibrate(T, zeta, half=BAND_HALF, dur=DUR, fs=FS, cb=CB,
                 snr_db=20.0, sim=None):
    """Fit the two-DOF model to the exact finite element spectrum.

    The result is the Kullback-Leibler projection of the truth onto the
    model. ``T_pseudo`` is the tension the estimate converges to, and
    T_pseudo - T is the misspecification bias.
    """
    from simulate_records import RecordSimulator
    sim = RecordSimulator(T, zeta) if sim is None else sim
    n = int(round(dur * fs))
    fa = np.fft.rfftfreq(n, 1.0 / fs)[1:-1]
    f = fa[(fa >= sim.f0 - half) & (fa <= sim.f0 + half)]
    # the truth the record is drawn from: recorded spectrum plus sensor noise
    S = sim.psd_recorded(f)
    if snr_db is not None:
        var = sim.variance_recorded(sim.dof_stay)
        S = S + var * 10.0 ** (-snr_db / 10.0) / (fs / 2.0)

    def resid(x, rh):
        th = np.array([T_of_fs(x[0]), x[1], np.exp(x[2]), np.exp(x[3]),
                       x[4], x[5], 0.0])
        return np.log(model_spectrum(f, th, cb, rh)) - np.log(S)

    x0 = np.array([sim.f_iso1, sim.f0, np.log(max(sim.s_split, 1e-4)),
                   np.log(zeta), np.log(S.max() * 4 * zeta ** 2),
                   np.log(S.max() * 1e-4)])
    best = None
    for rh in (0.0, RH, 0.005):
        r = least_squares(resid, x0, args=(rh,), xtol=1e-15, ftol=1e-15,
                          max_nfev=20000)
        v = float(np.sqrt(np.mean(r.fun ** 2)))
        if best is None or v < best[1]:
            best = (r, v, rh)
    r, rms, rh = best
    theta = np.array([T_of_fs(r.x[0]), r.x[1], np.exp(r.x[2]),
                      np.exp(r.x[3]), r.x[4], r.x[5], 0.0])
    return dict(theta=theta, rh=rh, rms_log=rms,
                max_log=float(np.abs(r.fun).max()), sim=sim, f=f,
                T_pseudo=theta[0])


def _one_fe_realisation(job):
    (T, zeta, theta_p, cb, rh, dur, fs, half, seed, free_keys, snr_db,
     f0) = job
    from simulate_records import RecordSimulator
    global _FE_CACHE
    key = (T, zeta)
    if _FE_CACHE.get("key") != key:
        _FE_CACHE["key"] = key
        _FE_CACHE["sim"] = RecordSimulator(T, zeta)
    sim = _FE_CACHE["sim"]
    rec = sim.record(duration=dur, snr_db=snr_db, seed=seed)
    fa, Ia = periodogram(rec["a_stay"], rec["fs"])
    m = (fa >= f0 - half) & (fa <= f0 + half)
    f, I = fa[m], Ia[m]
    # rescale the pseudo-true level parameters to this record's absolute scale
    th = np.array(theta_p, dtype=float)
    sc = np.log(np.median(I) / np.median(model_spectrum(f, th, cb, rh)))
    th[4] += sc
    th[5] += sc
    out = dict(seed=seed, T_true=T, T_pseudo=theta_p[0])
    for key_ in free_keys:
        r = fit_mle(f, I, th, FREE_SETS[key_], cb, rh)
        Fo = fim_observed(r["theta"], f, I, FREE_SETS[key_], cb, rh)
        v, _ = crb_T(Fo)
        out.update({f"T_{key_}": r["T_hat"], f"ok_{key_}": r["ok"],
                    f"bound_{key_}": r["at_bound"],
                    f"nbound_{key_}": r["n_bound"],
                    f"nfail_{key_}": r["n_fail"],
                    f"mirror_gap_{key_}": r["mirror_gap"],
                    f"peaks_gap_{key_}": r["peaks_gap"],
                    f"runner_gap_{key_}": r["runner_gap"],
                    f"runner_T_{key_}": r["runner_T"],
                    f"won_{key_}": r["start_won"],
                    f"s_{key_}": r["theta"][2], f"zeta_{key_}": r["theta"][3],
                    f"fh_{key_}": r["theta"][1],
                    f"crbobs_{key_}": np.sqrt(v) / T
                    if np.isfinite(v) else np.inf})
    return out


_FE_CACHE = {}


def run_armB(tensions, zetas, nreal=NREALB, free_keys=("host_unknown",),
             snr_db=20.0, nproc=NPROC):
    rows, alldraws = [], []
    t0 = time.time()
    for zeta in zetas:
        for ci, T in enumerate(tensions):
            cal = fe_calibrate(T, zeta, snr_db=snr_db)
            th_p, sim = cal["theta"], cal["sim"]
            jobs = [(T, zeta, th_p, CB, cal["rh"], DUR, FS, BAND_HALF,
                     20000 + 1000 * ci + k, free_keys, snr_db, sim.f0)
                    for k in range(nreal)]
            if nproc > 1:
                with Pool(min(nproc, 6)) as p:
                    draws = p.map(_one_fe_realisation, jobs, chunksize=2)
            else:
                draws = [_one_fe_realisation(j) for j in jobs]
            row, dd = _summarise(draws, th_p, CB, cal["rh"], DUR, FS,
                                 BAND_HALF, free_keys, T, cal["T_pseudo"],
                                 "armB")
            row.update(label=f"T={T/1e3:.1f}kN", T_fe=T,
                       rms_log_misspec=cal["rms_log"],
                       max_log_misspec=cal["max_log"],
                       rh_fitted=cal["rh"], s_fe=sim.s_split,
                       f_iso_fe=sim.f_iso1, f_lo_fe=sim.f_lo,
                       f_hi_fe=sim.f_hi, f0_fe=sim.f0,
                       d_fe=(sim.f_iso1 - cal["theta"][1]) / sim.f_iso1,
                       misspec_bias_pct=100 * (cal["T_pseudo"] - T) / T)
            rows.append(row)
            dd["label"] = row["label"]
            dd["zeta"] = zeta
            alldraws.append(dd)
            k0 = free_keys[-1]
            print(f"  T {T/1e3:6.1f} kN zeta {100*zeta:5.2f} % | "
                  f"misspec {row['misspec_bias_pct']:+.3f} %  "
                  f"resid {row['rms_log_misspec']:.4f}  CRB "
                  f"{100*row[f'cv_{k0}']:.4f} %  sample "
                  f"{100*row[f'samp_cv_{k0}']:.4f} %  bias(T) "
                  f"{row[f'bias_pct_{k0}']:+.3f} %  bias(pseudo) "
                  f"{row[f'bias_pseudo_pct_{k0}']:+.3f} %")
    print(f"  ({time.time()-t0:.0f} s)")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA, "crb_armB.csv"), index=False)
    pd.concat(alldraws, ignore_index=True).to_csv(
        os.path.join(DATA, "crb_armB_draws.csv"), index=False)
    print("  wrote data/crb_armB.csv and data/crb_armB_draws.csv")
    return out


# --- bound map, no Monte Carlo -----------------------------------------------

def run_bound():
    rows = []
    print("bound map: detuning by damping, at the nominal record")
    print("     d     zeta   r=s/2z  PNR   cv_known  cv_unknown  cv_+rb   "
          "ratio  ratio_rb   2 eps")
    for z in ZETAS:
        for d in DGRID:
            th = set_levels(theta_from_d(d, zeta=z))
            r = bound_row(th)
            r.update(sweep="d_zeta")
            rows.append(r)
            if d >= 0:
                print(f"  {d:+.4f}  {100*z:5.2f}  {r['r_res']:6.2f} "
                      f"{r['pnr_db']:5.1f}  {100*r['cv_host_known']:.4f}   "
                      f"{100*r['cv_host_unknown']:.4f}   "
                      f"{100*r['cv_host_unknown_rb']:.4f}  "
                      f"{r['ratio_unknown_known']:6.2f} "
                      f"{r['ratio_rb_known']:9.1f}  {200*r['eps']:.3f}")
    print("\ncoupling sweep at zeta = 0.5 %, d = 0")
    for s in (0.001, 0.002, 0.005, 0.010, 0.0232, 0.050, 0.100):
        th = set_levels(theta_from_d(0.0, s=s))
        r = bound_row(th)
        r.update(sweep="s")
        rows.append(r)
        print(f"  s {s:.4f}  r {r['r_res']:6.2f}  cv_known "
              f"{100*r['cv_host_known']:.4f} %  cv_unknown "
              f"{100*r['cv_host_unknown']:.4f} %  ratio "
              f"{r['ratio_unknown_known']:.2f}")
    print("\nnoise sweep at zeta = 0.5 % and 2 %, d = 0")
    for z in (0.005, 0.020):
        for p in (40, 30, 20, 15, 10, 5, 0):
            th = set_levels(theta_from_d(0.0, zeta=z), pnr_ref_db=p)
            r = bound_row(th)
            r.update(sweep="pnr", pnr_ref_db=p)
            rows.append(r)
            print(f"  zeta {100*z:4.1f} %  PNR {r['pnr_db']:5.1f} dB  "
                  f"cv_known {100*r['cv_host_known']:.4f} %  cv_unknown "
                  f"{100*r['cv_host_unknown']:.4f} %  ratio "
                  f"{r['ratio_unknown_known']:.2f}")
    print("\nrecord length sweep at zeta = 0.5 %, d = 0 and 0.02")
    for d in (0.0, 0.02):
        for dur in (60.0, 150.0, 300.0, 600.0, 1200.0, 3600.0):
            th = set_levels(theta_from_d(d), dur=dur)
            r = bound_row(th, dur=dur)
            r.update(sweep="dur")
            rows.append(r)
            print(f"  d {d:+.3f}  {dur:6.0f} s  {r['nbins']:5d} bins  "
                  f"cv_known {100*r['cv_host_known']:.4f} %  cv_unknown "
                  f"{100*r['cv_host_unknown']:.4f} %")
    print("\nband half-width sweep at zeta = 0.5 %, d = 0")
    for half in (0.03, 0.05, 0.08, 0.12, 0.20, 0.35, 0.60, 0.85):
        th = set_levels(theta_from_d(0.0), half=half)
        r = bound_row(th, half=half)
        r.update(sweep="band")
        rows.append(r)
        print(f"  half {half:.2f} Hz  {r['nbins']:5d} bins  cv_known "
              f"{100*r['cv_host_known']:.4f} %  cv_unknown "
              f"{100*r['cv_host_unknown']:.4f} %  ratio "
              f"{r['ratio_unknown_known']:.2f}")
    print("\ntwo sensors (stay and deck) against one, zeta = 0.5 %")
    for d in (0.0, 0.008, 0.025, 0.060):
        th = set_levels(theta_from_d(d))
        r1 = bound_row(th)
        r2 = bound_row(th, two_ch=True)
        r2.update(sweep="two_channel")
        r1.update(sweep="one_channel")
        rows += [r1, r2]
        print(f"  d {d:+.3f}  one sensor cv_unknown "
              f"{100*r1['cv_host_unknown']:.4f} %   two sensors "
              f"{100*r2['cv_host_unknown']:.4f} %   gain "
              f"{(r1['cv_host_unknown']/r2['cv_host_unknown'])**2:.2f}")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA, "crb_bound.csv"), index=False)
    print("\nwrote data/crb_bound.csv")
    return out


# --- verification ------------------------------------------------------------

VERIFY_NOTES = """
[1] the Whittle assumption: I_k/S_k on finite element records, mean, variance
    and a Kolmogorov-Smirnov test against Exp(1)
[2] the finite-difference step for the information matrix, a factor of ten
    either side of the choice
[3] the expected information against the observed information averaged over
    realizations, and against the sample covariance of the score
[4] a case with a known answer: one isolated mode, where the bound is
    checked against the sample variance of the estimate
[5] the two DOF model against the exact finite element spectrum, which is
    the size of arm B's misspecification
[6] invariance of the bound on T under reparametrization of the nuisances
[7] the observation gain factor c_b against the finite element mode shapes
[8] the arm A simulator: its periodogram against its own analytic spectrum
[9] the null direction of the information matrix: that the log spectrum
    changes only at second order along it, so the singularity is real and
    not a rounding artifact, and that the divergence along it grows as the
    fourth power of the step
[10] the cost in tension of an error in the observation gain factor c_b,
    which is the external knowledge that identifiability depends on
"""


def check_whittle_premise(rows, zetas=(0.005, 0.02), nseed=4):
    from simulate_records import RecordSimulator
    for z in zetas:
        sim = RecordSimulator(T_TUNE, z)
        st = []
        for sd in range(nseed):
            rec = sim.record(duration=DUR, snr_db=20.0, seed=100 + sd)
            f, I = periodogram(rec["a_stay"], rec["fs"])
            m = (f >= sim.f0 - BAND_HALF) & (f <= sim.f0 + BAND_HALF)
            sig = rec["meta"]["noise_rms"][0]
            S = sim.psd_recorded(f[m]) * rec["meta"]["intensity"] \
                + sig ** 2 / (rec["fs"] / 2)
            u = I[m] / S
            ks = kstest(u, "expon")
            st.append((u.mean(), u.var(), ks.statistic, ks.pvalue, m.sum()))
        a = np.array(st)
        rows.append(dict(check="1_whittle_premise", zeta=z,
                         mean_I_over_S=a[:, 0].mean(),
                         var_I_over_S=a[:, 1].mean(), ks_stat=a[:, 2].mean(),
                         ks_p_min=a[:, 3].min(), nbins=int(a[0, 4]),
                         nseed=nseed))
        print(f"  [1] zeta {100*z:.1f} %: mean I/S {a[:,0].mean():.4f}  var "
              f"{a[:,1].mean():.4f}  KS {a[:,2].mean():.4f}  min p "
              f"{a[:,3].min():.3f}  ({int(a[0,4])} bins)")


def check_fd_step(rows):
    """Complex step against central differences for the expected information."""
    for z in (0.002, 0.02):
        for d in (0.0, 0.02):
            th = set_levels(theta_from_d(d, zeta=z))
            f = band_grid(th)
            cs = {}
            for key, free in FREE_SETS.items():
                v, _ = crb_T(fim_expected(th, f, free, method="cs"))
                cs[key] = np.sqrt(v) / th[0]
            for sc in (0.1, 0.3, 1.0, 3.0, 10.0):
                fd = {}
                for key, free in FREE_SETS.items():
                    v, _ = crb_T(fim_expected(th, f, free, scale=sc,
                                              method="fd"))
                    fd[key] = np.sqrt(v) / th[0]
                rows.append(dict(check="2_fd_step", zeta=z, d=d, scale=sc,
                                 **{f"cv_fd_{k}": x for k, x in fd.items()},
                                 **{f"cv_cs_{k}": x for k, x in cs.items()}))
            sub = [r for r in rows if r.get("check") == "2_fd_step"
                   and r["zeta"] == z and r["d"] == d]
            a = np.array([r["cv_fd_host_unknown"] for r in sub])
            b = np.array([r["cv_fd_host_unknown_rb"] for r in sub])
            print(f"  [2] zeta {100*z:.1f} % d {d:+.3f}: tied model, central "
                  f"differences over steps x0.1 to x10 give "
                  f"{a.min():.4e} to {a.max():.4e} against complex step "
                  f"{cs['host_unknown']:.4e}")
            print(f"      observation gain free: central differences "
                  f"{b.min():.3e} to {b.max():.3e} against complex step "
                  f"{cs['host_unknown_rb']:.3e}")


def check_information_agreement(rows, nreal=200, seed=0):
    rng = np.random.default_rng(seed)
    for z in (0.005, 0.02):
        for d in (0.0, 0.02):
            th = set_levels(theta_from_d(d, zeta=z))
            f = band_grid(th)
            S = model_spectrum(f, th)
            free = FREE_SETS["host_unknown"]
            Fe = fim_expected(th, f, free)
            D = _dlogS(th, f, free, CB, RH)
            Fo = np.zeros_like(Fe)
            Sc = np.zeros((nreal, len(free)))
            for r in range(nreal):
                I = S * rng.exponential(size=len(f))
                Fo += fim_observed(th, f, I, free)
                Sc[r] = D @ (I / S - 1.0)
            Fo /= nreal
            e = np.sqrt(np.diag(Fe))
            rel_o = np.abs(Fo - Fe) / np.outer(e, e)
            rel_s = np.abs(Sc.T @ Sc / nreal - Fe) / np.outer(e, e)
            rows.append(dict(check="3_fim_agreement", zeta=z, d=d,
                             nreal=nreal, max_rel_obs=rel_o.max(),
                             max_rel_score=rel_s.max(),
                             cv_expected=np.sqrt(crb_T(Fe)[0]) / th[0],
                             cv_observed=np.sqrt(crb_T(Fo)[0]) / th[0]))
            print(f"  [3] zeta {100*z:.1f} % d {d:+.3f}: observed vs expected "
                  f"max {rel_o.max():.4f}, score covariance vs expected max "
                  f"{rel_s.max():.4f}  (n = {nreal}, expected scatter "
                  f"{np.sqrt(2/nreal):.3f})")


def check_single_mode(rows, nreal=400, seed=1):
    for zeta in (0.005, 0.02):
        rng = np.random.default_rng(seed)
        th = set_levels(theta_from_d(0.0, s=1e-5, zeta=zeta))
        f = band_grid(th)
        free = ("T", "zeta", "lnGs", "lnGn")
        crb = crb_T(fim_expected(th, f, free))[0]
        S = model_spectrum(f, th)
        That = np.array([fit_mle(f, S * rng.exponential(size=len(f)), th,
                                 free)["T_hat"] for _ in range(nreal)])
        v = That.var(ddof=1)
        rows.append(dict(check="4_single_mode", zeta=zeta, nreal=nreal,
                         crb_cv=np.sqrt(crb) / th[0],
                         sample_cv=np.sqrt(v) / th[0], ratio=v / crb,
                         bias_pct=100 * (That.mean() - th[0]) / th[0]))
        print(f"  [4] one mode, zeta {100*zeta:.1f} %: CRB cv "
              f"{100*np.sqrt(crb)/th[0]:.4f} %, sample cv "
              f"{100*np.sqrt(v)/th[0]:.4f} %, var/CRB {v/crb:.3f} "
              f"(scatter {np.sqrt(2/nreal):.3f}), bias "
              f"{100*(That.mean()-th[0])/th[0]:+.4f} %")


def check_misspecification(rows):
    for z in (0.002, 0.005, 0.010, 0.020):
        cal = fe_calibrate(T_TUNE, z, snr_db=None)
        th, sim = cal["theta"], cal["sim"]
        rows.append(dict(check="5_misspecification", zeta=z,
                         rms_log_resid=cal["rms_log"],
                         max_log_resid=cal["max_log"], rh_fitted=cal["rh"],
                         f_s_fit=fs_of_T(th[0]), f_h_fit=th[1], s_fit=th[2],
                         zeta_fit=th[3], T_pseudo=th[0], T_true=T_TUNE,
                         bias_pct=100 * (th[0] - T_TUNE) / T_TUNE,
                         s_fe=sim.s_split, f_iso_fe=sim.f_iso1))
        print(f"  [5] zeta {100*z:.1f} %: rms log residual "
              f"{cal['rms_log']:.5f}, max {cal['max_log']:.5f}; fitted s "
              f"{th[2]:.5f} vs FE split {sim.s_split:.5f}; pseudo-true T is "
              f"{100*(th[0]-T_TUNE)/T_TUNE:+.3f} % from the truth")


def check_invariance(rows):
    th = set_levels(theta_from_d(0.005))
    f = band_grid(th)
    free = FREE_SETS["host_unknown"]
    c1 = crb_T(fim_expected(th, f, free))[0]
    D = _dlogS(th, f, free, CB, RH)
    J = np.eye(len(free))
    J[1, 1], J[1, 2], J[2, 1], J[2, 2], J[3, 5] = 2.0, 0.5, -1.0, 3.0, 0.7
    D2 = np.linalg.solve(J.T, D)
    c2 = crb_T(D2 @ D2.T)[0]
    rows.append(dict(check="6_invariance", crb_1=c1, crb_2=c2,
                     rel_diff=abs(c2 - c1) / c1))
    print(f"  [6] bound on T invariant under nuisance mixing: relative "
          f"difference {abs(c2-c1)/c1:.2e}")


def check_cb(rows):
    """c_b against the finite element coupled mode shapes, sign-safe.

    Each coupled shape is normalized to a positive anchorage amplitude. The
    two-DOF model uses the same frequencies, from the trace and determinant
    identities w_h^2 = w_+^2 + w_-^2 - w_s^2 and
    kappa^2 = w_s^2 w_h^2 - w_+^2 w_-^2. The ratio of the two modes' sensor
    amplitudes is a Mobius function of rb, inverted in closed form.
    """
    from simulate_records import RecordSimulator
    vals = []
    for T in (140e3, 146e3, T_TUNE, 157e3, 164e3):
        sim = RecordSimulator(T, 0.005)
        ws2 = (2 * np.pi * sim.f_iso1) ** 2
        wp2, wm2 = (2 * np.pi * sim.f_hi) ** 2, (2 * np.pi * sim.f_lo) ** 2
        wh2 = wp2 + wm2 - ws2
        kap = np.sqrt(max(ws2 * wh2 - wp2 * wm2, 0.0))
        sfe = kap / (0.5 * (ws2 + wh2))
        _, V = np.linalg.eigh(np.array([[ws2, kap], [kap, wh2]]))
        dofa = 2 * sim.cd.ia
        A = []
        for i in sim.pair_idx:
            v = sim.Phi[:, i] * np.sign(sim.Phi[dofa, i])
            A.append((v[sim.dof_stay], v[dofa]))
        r = (A[0][0] / A[0][1]) / (A[1][0] / A[1][1])
        P = [V[:, j] * np.sign(V[1, j]) for j in (0, 1)]
        a0, b0, a1, b1 = P[0][0], P[0][1], P[1][0], P[1][1]
        rb = (r * a1 * b0 - a0 * b1) / (b0 * b1 * (1 - r))
        vals.append((T, float(np.sqrt(wh2) / (2 * np.pi)), float(sfe),
                     float(rb), float(rb / sfe)))
        rows.append(dict(check="7_cb", T=T, f_h_fe=vals[-1][1],
                         s_fe=vals[-1][2], rb_fe=vals[-1][3],
                         cb_fe=vals[-1][4], cb_model=CB))
    a = np.array(vals)
    print(f"  [7] observation gain from the FE mode shapes over 140 to 164 kN: "
          f"rb = {a[:,3].min():+.5f} to {a[:,3].max():+.5f}, c_b = "
          f"{a[:,4].min():+.3f} to {a[:,4].max():+.3f} (mean "
          f"{a[:,4].mean():+.3f}); the kinematic value used is {CB:+.3f}")
    print(f"      host frequency from the trace identity: "
          f"{a[:,1].mean():.5f} Hz, spread {a[:,1].std():.5f}; "
          f"the module uses {F_HOST:.5f}")


def check_armA_simulator(rows, nseed=6):
    for z in (0.005, 0.02):
        th = set_levels(theta_from_d(0.0, zeta=z))
        rec = TwoDofRecorder(th)
        u = []
        for sd in range(nseed):
            y = rec.record(duration=DUR, seed=500 + sd)
            f, I = periodogram(y, FS)
            _, _, f0, _ = pair_from_theta(th)
            m = (f >= f0 - BAND_HALF) & (f <= f0 + BAND_HALF)
            u.append(I[m] / model_spectrum(f[m], th))
        u = np.concatenate(u)
        ks = kstest(u, "expon")
        rows.append(dict(check="8_armA_simulator", zeta=z, nseed=nseed,
                         mean_I_over_S=u.mean(), var_I_over_S=u.var(),
                         ks_stat=ks.statistic, ks_p=ks.pvalue, n=len(u)))
        print(f"  [8] arm A simulator, zeta {100*z:.1f} %: mean I/S "
              f"{u.mean():.4f} (nominal error "
              f"{1/np.sqrt(len(u)):.4f}), var {u.var():.4f}, KS "
              f"{ks.statistic:.4f}, p {ks.pvalue:.3f}")


def check_null_direction(rows):
    """Along the null direction the spectrum must move only at second order."""
    th = set_levels(theta_from_d(0.0))
    f = band_grid(th)
    free = FREE_SETS["host_unknown_rb"]
    F = fim_expected(th, f, free)
    sc = 1.0 / np.sqrt(np.diag(F))
    w, V = np.linalg.eigh(F * np.outer(sc, sc))
    step = V[:, 0] * sc
    S0 = model_spectrum(f, th)
    rec = []
    for tpct in (0.1, 0.3, 1.0, 3.0):
        k = tpct / 100.0 * th[0] / abs(step[0])
        t2 = np.array(th)
        for i, n in enumerate(free):
            t2[PARAMS.index(n)] += k * step[i]
        dl = np.log(model_spectrum(f, t2) / S0)
        rec.append((tpct, float(np.sqrt(np.mean(dl ** 2))),
                    expected_lr(t2, th, f)))
    a = np.array(rec)
    p_log = np.polyfit(np.log(a[:, 0]), np.log(a[:, 1]), 1)[0]
    p_kl = np.polyfit(np.log(a[:, 0]), np.log(a[:, 2]), 1)[0]
    rows.append(dict(check="9_null_direction", lam_min=w[0], lam_max=w[-1],
                     lam_ratio=w[0] / w[-1], slope_logS=p_log, slope_kl=p_kl,
                     dT_pct=a[:, 0].tolist(), rms_dlogS=a[:, 1].tolist(),
                     kl=a[:, 2].tolist()))
    print(f"  [9] smallest scaled eigenvalue {w[0]:.2e} of {w[-1]:.2e}; along "
          f"that direction rms dlog S goes as step^{p_log:.2f} and the "
          f"divergence as step^{p_kl:.2f} (2 and 4 expected)")
    print("      " + "  ".join(f"dT {t:.1f} %: KL {k:.4f}"
                               for t, _, k in rec))


def check_cb_error(rows):
    """What an error in c_b costs in tension, at fixed spectrum."""
    for d in (0.0, 0.02):
        th = set_levels(theta_from_d(d))
        f = band_grid(th)
        free = [n for n in FREE_SETS["host_unknown"] if n != "T"]
        bnds = bounds_for(th)
        for err in (-0.20, -0.06, 0.06, 0.20):
            cb2 = CB * (1.0 + err)

            def obj(x):
                t = _from_x(x[1:], free, np.array(th))
                t[0] = th[0] * np.exp(x[0])
                return expected_lr_cross(t, th, f, cb2, CB)
            x0 = np.r_[0.0, _to_x(th, free)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = minimize(obj, x0, method="Nelder-Mead",
                             options=dict(maxiter=8000, xatol=1e-10,
                                          fatol=1e-12))
            rows.append(dict(check="10_cb_error", d=d, cb_err=err,
                             T_bias_pct=100 * (np.exp(r.x[0]) - 1.0),
                             kl_at_opt=float(r.fun)))
            print(f"  [10] d {d:+.3f}, c_b in error by {100*err:+.0f} %: the "
                  f"best fit moves T by {100*(np.exp(r.x[0])-1):+.4f} % "
                  f"(residual divergence {r.fun:.4f})")


def expected_lr_cross(theta, theta0, f, cb_fit, cb_true, rh=RH):
    """Divergence of a model with the wrong c_b from the truth."""
    S = model_spectrum(f, theta, cb_fit, rh)
    S0 = model_spectrum(f, theta0, cb_true, rh)
    r = S0 / S
    return float(np.sum(np.log(1.0 / r) + r - 1.0))


def run_verify():
    print("verification" + VERIFY_NOTES)
    rows = []
    check_whittle_premise(rows)
    check_fd_step(rows)
    check_information_agreement(rows)
    check_single_mode(rows)
    check_misspecification(rows)
    check_invariance(rows)
    check_cb(rows)
    check_armA_simulator(rows)
    check_null_direction(rows)
    check_cb_error(rows)
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "crb_verify.csv"),
                              index=False)
    print("\nwrote data/crb_verify.csv")


# --- screening criterion beside the bound ------------------------------------


def run_criterion(tols=(0.02, 0.05, 0.10), zetas=ZETAS, s=S_TUNE):
    """Screening criterion beside the precision a coupled fit can reach.

    |d| >= (s^2 - tol^2)/(2 tol) is exactly eps <= tol, with
    eps = sqrt(d^2 + s^2) - |d| the tension bias of the isolated-cable
    inversion. Tabulates the bound on T at |d| = 0, |d|_crit and 2 |d|_crit.
    Writes data/crb_criterion.csv.
    """
    rows = []
    print("screening criterion and bound (percent of T, 600 s, one sensor)")
    print("  zeta   tol   |d|_crit   eps at |d|_crit   cv at 0   cv at "
          "|d|_crit   cv at 2|d|_crit")
    for z in zetas:
        for tol in tols:
            dcrit = max((s ** 2 - tol ** 2) / (2 * tol), 0.0)
            r0 = bound_row(set_levels(theta_from_d(0.0, s=s, zeta=z)),
                           profile=False)
            r1 = bound_row(set_levels(theta_from_d(dcrit, s=s, zeta=z)),
                           profile=False)
            r2 = bound_row(set_levels(theta_from_d(2 * dcrit, s=s, zeta=z)),
                           profile=False)
            row = dict(zeta=z, tol=tol, d_crit=dcrit,
                       eps_at_dcrit=r1["eps"],
                       cv_at_0=r0["cv_host_unknown"],
                       cv_at_dcrit=r1["cv_host_unknown"],
                       cv_at_2dcrit=r2["cv_host_unknown"],
                       weak_snr_0=r0["weak_peak_snr_db"],
                       weak_snr_dcrit=r1["weak_peak_snr_db"])
            rows.append(row)
            print(f"  {100*z:4.1f}  {100*tol:4.1f}   {dcrit:8.5f}   "
                  f"{100*r1['eps']:12.3f}   {100*r0['cv_host_unknown']:8.3f}  "
                  f"{100*r1['cv_host_unknown']:12.3f}  "
                  f"{100*r2['cv_host_unknown']:14.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA, "crb_criterion.csv"), index=False)
    print("\nwrote data/crb_criterion.csv")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bound", action="store_true")
    ap.add_argument("--arm0", action="store_true")
    ap.add_argument("--armA", action="store_true")
    ap.add_argument("--armB", action="store_true")
    ap.add_argument("--criterion", action="store_true")
    ap.add_argument("--nreal0", type=int, default=NREAL0)
    ap.add_argument("--nrealA", type=int, default=NREALA)
    ap.add_argument("--nrealB", type=int, default=NREALB)
    ap.add_argument("--nproc", type=int, default=NPROC)
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)

    if a.verify:
        run_verify()
    if a.bound:
        run_bound()
    if a.criterion:
        run_criterion()
    if a.arm0:
        print("\narm 0: exact Whittle draws, information content only")
        cfgs = []
        for z in (0.002, 0.005, 0.020):
            for d in DGRID_MC:
                th = set_levels(theta_from_d(d, zeta=z))
                cfgs.append((th, CB, RH, DUR, FS, BAND_HALF, th[0], th[0],
                             f"d={d:+.3f} zeta={100*z:.1f}%"))
        run_mc("arm0", cfgs, ("host_known", "host_unknown",
                              "host_unknown_rb"), a.nreal0, "arm0", a.nproc)
    if a.armA:
        print("\narm A: sampled records from the same model, full chain")
        cfgs = []
        for z in (0.002, 0.005, 0.020):
            for d in (-0.025, -0.008, 0.0, 0.008, 0.025):
                th = set_levels(theta_from_d(d, zeta=z))
                cfgs.append((th, CB, RH, DUR, FS, BAND_HALF, th[0], th[0],
                             f"d={d:+.3f} zeta={100*z:.1f}%"))
        run_mc("armA", cfgs, ("host_known", "host_unknown"), a.nrealA,
               "armA", a.nproc)
    if a.armB:
        print("\narm B: the finite element bridge, fitted with two modes")
        run_armB((140e3, 146e3, 151.6e3, 157e3, 164e3), (0.005, 0.020),
                 a.nrealB, ("host_known", "host_unknown"), 20.0, a.nproc)


if __name__ == "__main__":
    main()
