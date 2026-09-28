# -*- coding: utf-8 -*-
"""Frequency domain decomposition through a deck-stay crossing.

The manuscript states, without evidence, that a method which fits modes
rather than picking maxima "can separate a pair the spectrum shows as a
single peak, in which case the branch law returns in full".  This script
tests that statement for the decomposition half of it.  Four estimators are
run on the same synthetic records:

  1. PEAK PICKING          the baseline the manuscript already assumes: the
                           tallest maximum of the Welch auto-spectrum of the
                           stay accelerometer
  2. HALF-POWER            the same maximum, plus the damping a practitioner
                           reads from the -3 dB width of that peak
  3. FDD                   singular value decomposition of the cross-power
                           spectral density matrix of stay and deck sensors,
                           with peaks read from the singular value curves
  4. EFDD                  the singular value bell around each peak,
                           selected by modal assurance criterion against the
                           peak singular vector, inverse transformed to a
                           correlation function and fitted for frequency and
                           damping

Nothing here re-derives the physics.  The records come from
``scripts/simulate_records.py`` unchanged, the bridge and the damping
calibration from ``scripts/run_damping.py``, the branch and merged-peak laws
from ``scripts/run_merged.py``, and the baseline peak picker is
``simulate_records.refine_peaks``, which that module already verifies against
the analytic spectrum.

WHY FDD IS THE RIGHT TEST
-------------------------
Frequency domain decomposition is the standard answer to closely spaced
modes in output-only modal analysis.  Its claim is structural rather than
statistical: if the modal coordinates are uncorrelated, the response
cross-spectral matrix at frequency f is

    G(f) = sum_j  g_j |H_j(f)|^2  psi_j psi_j^H                        (1)

a sum of rank-one terms, one per mode.  Two modes therefore make G rank two,
the first singular value tracks whichever mode is larger and the second
tracks the other, and a pair the auto-spectrum shows as one peak is supposed
to appear as two features across the two singular value curves.  That is the
claim being tested.

WHAT BREAKS IT HERE, AND IT IS NOT THE SENSORS
----------------------------------------------
For a stationary load with modal force cross-spectral matrix Gamma, the
exact output cross-spectral matrix is

    G(f) = A(f) Gamma A(f)^H,   A[p,j] = phi_j(p) H_j(f)               (2)

and (1) is the special case Gamma diagonal.  At a veering crossing the two
hybrid modes are, to leading order, the sum and the difference of one stay
mode and one deck mode,

    psi_+- = (u_s +- u_d) / sqrt(2)

so any load field p produces modal forces (p.u_s +- p.u_d)/sqrt(2) whose
correlation coefficient is

    corr = (sigma_s^2 - sigma_d^2) / (sigma_s^2 + sigma_d^2)           (3)

with sigma_s, sigma_d the root-mean-square generalised forces on the two
constituent modes.  The correlation is +-1 at both extremes of the load
balance and vanishes only where the two are equal.  A stay carries a few
kilogrammes per metre against a deck's tonne, so a spatially uniform load
intensity drives the stay mode far harder, Gamma is near rank one, G is near
rank one whatever the sensors do, and the second singular value has nothing
to carry.  This script measures that correlation, measures the second
singular value it produces, and sweeps the load balance to find where FDD
would in fact work.

The competing explanation, that two sensors simply cannot tell the two
hybrid modes apart, is measured too: the modal assurance criterion between
the two true mode shapes restricted to the sensor set is reported beside the
force correlation, and a positive control with the same frequencies, damping
and mode shapes but uncorrelated modal forces separates the two causes.

WHAT A SINGLE CHANNEL DOES
--------------------------
With one sensor G is one by one, its singular value decomposition returns
the auto-spectrum itself and a unit singular vector, and there is no second
singular value at all.  FDD then IS peak picking, bit for bit, and this is
checked rather than argued.  Single-channel stay measurement is the practice
the study is about, so the honest statement is that FDD is not available to
it without a second sensor somewhere on the structure.

PROCESSING, AND THE CHOICES IT NEEDS
------------------------------------
``nperseg = 8192`` at ``fs = 100 Hz`` gives a bin of 0.0122 Hz, a sixth of
the 0.0777 Hz split of the worked bridge, with 26 Hann segments at 75 per
cent overlap in a 600 s record.  Those 26 are worth 13.8 independent
averages, not 26, and the difference is not cosmetic: it sets the
prominence floor that decides whether a shoulder is called a peak, and
:func:`effective_averages` computes it rather than assuming it.  Overlap
beyond 50 per cent buys almost nothing, 12.4 independent averages at 50 per
cent against 13.8 at 75 and 13.6 at 87.5.  Resolution and
variance pull against each other and the trade is stated with every result:
the number of bins across a half-power bandwidth is reported for every case,
and a case with fewer than about three is resolution limited rather than
method limited.  A supplementary run at 3600 s and ``nperseg = 32768``
separates the two.

Peaks are found by ``simulate_records.refine_peaks`` with a smoothing width
of three bins, which is a data-driven choice that does not presuppose the
damping, and a prominence floor of four times the chi-squared scatter of the
unsmoothed decibel curve.  The same procedure is applied to the analytic
spectrum so that the comparison is like for like.

WHAT WAS FOUND, IN ONE PLACE
----------------------------
Every number below is printed by this script and written to the files it
names.  On the worked bridge at its crossing, T = 151.6 kN, split
s = 2.34 per cent, pair at 3.2833 and 3.3609 Hz:

* the second singular value never shows the second branch.  On the exact
  cross-spectrum it carries exactly one peak, at 3.327 to 3.330 Hz, between
  the two branches, at every damping from 0.2 to 3 per cent and at every
  tension tried, and that peak stands 33 to 39 dB below the first singular
  value's peak.  The two curves agree on how many peaks there are in every
  one of the 15 exact cases and in 310 of the 315 records, so where the
  auto-spectrum shows one peak the first singular value shows one too, and
  the second adds a feature at the wrong frequency rather than the missing
  one (``data/oma_fdd_exact.csv``, ``data/oma_fdd_sweep.csv``).

* the reason is the excitation and not the sensors.  The two hybrid modes
  take modal forces with correlation -0.988 at the crossing, because both
  contain the same stay shape and a stay 180 times lighter per metre than
  the deck takes far more generalised force from the same load field.  The
  smaller eigenvalue of their 2 by 2 force matrix is 0.6 per cent of the
  larger.  Forced to be exactly rank one, the cross-spectral matrix has
  sv2/sv1 = 5e-16 at 2 sensors and 8e-16 at 7; with the bridge's own forces
  it is 5.6e-3 at 2 sensors and 2.0e-3 at 7, so ADDING SENSORS MAKES IT
  SMALLER.  The two hybrid shapes at the two sensors have MAC 0.86, so the
  sensors do tell them apart.

* the load field this uses is delta-correlated in space, which is the
  favourable case.  A spatially coherent load, a uniform gust, gives modal
  force correlation exactly -1.000000 and sv2/sv1 = 4e-16.

* moving the second sensor cannot fix it.  A second accelerometer on the
  same stay leaves the two hybrid shapes at MAC 0.996 and drops the sv2 peak
  to 2.8e-5 of the sv1 peak, against 4.7e-4 for a deck sensor at the
  anchorage (``data/oma_fdd_sensors.csv``).

* the load balance that would satisfy FDD's premise exists and is narrow.
  The force correlation passes through zero at a stay-to-deck load intensity
  ratio of 0.0055, which is the mass ratio mc/md, and even there the sv2
  peak is 20.5 dB down at 0.5 per cent damping and 14.7 dB down at 2 per
  cent, and the first singular value still shows one peak in the merged
  regime (``data/oma_fdd_loadbalance.csv``).

* it is not a record-length, resolution or noise limit.  At 3600 s,
  nperseg 32768, 43 to 65 bins across a half-power bandwidth and no sensor
  noise at all, the merged cases still give one peak on the auto-spectrum
  and one on the first singular value (``data/oma_fdd_long.csv``).

* the positive control says the same.  A synthetic pair with the same
  frequencies, damping and sensor shapes but INDEPENDENT modal forces still
  gives one peak on both curves once the pair is merged, and the nearest
  FDD feature to the upper branch is then 0.8 per cent low.

* FDD returns the tension peak picking returns.  On the exact spectrum over
  21 tensions the two agree to 0.0008 to 0.014 percentage points of tension,
  and on 600 s records the median difference is 0.004 percentage points.
  The first singular value is the argument of a maximum, so it obeys the
  MERGED-PEAK law and not the branch law: root mean square difference from
  the merged law 0.005 to 0.21 percentage points against 0.011 to 0.64 from
  the branch law, the gap widening with damping.  Its worst error over the
  tension sweep is 1.01, 1.04, 1.08, 0.86 and 0.55 times the branch value
  at 0.2, 0.5, 1, 2 and 3 per cent damping, so it takes the same merged-peak
  relief peak picking does (``data/oma_fdd_exactlaw.csv``).

* the second singular value carries no tension at all.  Its single peak sits
  within 0.3 per cent of the DECK mode's own frequency, which does not
  depend on the stay tension, so over 140 to 165 kN it moves 0.0011 Hz while
  the isolated stay frequency moves 0.273 Hz, and a tension read from it
  runs from +8.6 to -7.8 per cent.

* resolvability on a record is not the closed-form resolvability.  With 26
  segments worth 14 independent averages the prominence floor is 4.7 dB, so
  the 3 dB dip the closed form allows at u = 1.14 cannot be called: the
  lowest u at which any of the 315 rows shows two peaks is 2.34.  At 3 per
  cent damping the seed-to-seed scatter of the picked frequency is 0.5 per
  cent, worth 1.1 percentage points of tension on one record, which is the
  same size as the bias being looked for.

* the shapes are not what fails.  The first singular vector at the surviving
  peak matches its nearer true hybrid shape at MAC 0.96 to 1.00 at every
  tension and damping tried, and EFDD's enhanced shape at 0.97 to 1.00,
  while the second singular vector matches the nearer of the two at only
  0.05 to 0.41.  FDD returns a good shape for the mode it sees and nothing
  for the mode it does not (``data/oma_fdd_shapes.csv``).

* the crossing announces itself in the DAMPING, not in the frequency.  At
  the crossing the half-power damping is 1.45 and 1.54 times what the same
  estimator returns on the same records far from it, at 2 and 3 per cent
  damping, and EFDD's is 1.43 and 1.31 times.  Below 1 per cent, where the
  pair resolves, neither inflates.

VERIFICATION
------------
``python3 scripts/oma_fdd.py --verify`` runs the checks listed in
``VERIFICATION_NOTES`` and writes ``data/oma_fdd_verify.csv``.  The two that
matter most are [B], a synthetic pair with known frequencies, damping and
mode shapes and genuinely uncorrelated modal forces, where FDD must succeed
if it works at all, and [D], the Welch cross-spectral estimate of long
noise-free bridge records against the exact cross-spectrum of the same
model convolved with the Welch window kernel.

Run:  python3 scripts/oma_fdd.py --verify
      python3 scripts/oma_fdd.py --exact
      python3 scripts/oma_fdd.py --sweep
      python3 scripts/oma_fdd.py --extra
      python3 scripts/oma_fdd.py --report      (re-report from the csv)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.signal import fftconvolve, freqz, get_window, lfilter, ss2tf, welch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from cablefe import invert_string  # noqa: E402
from run_damping import BRIDGE, FBAND, T_TUNE  # noqa: E402
from run_merged import rho_2dof, xstar  # noqa: E402
from simulate_records import (OVERSAMPLE, RecordSimulator,  # noqa: E402
                              refine_peaks)

DATA = os.path.join(ROOT, "data")

NPERSEG = 8192                   # 0.0122 Hz at fs = 100 Hz
OVERLAP_FRAC = 0.75              # 26 segments worth 13.8 independent
SMOOTH_BINS = 3                  # peak-picking smoothing, in bins
MAC_MIN = 0.80                   # EFDD bell membership
BAND = FBAND                     # 2.5 to 4.2 Hz, the band with the pair
ZETAS = (0.002, 0.005, 0.010, 0.020, 0.030)
T_SWEEP = np.arange(140.0e3, 165.0e3 + 1.0, 1.25e3)


# ===========================================================================
# spectral estimation
# ===========================================================================

def cpsd_welch(X, fs, nperseg=NPERSEG, overlap_frac=OVERLAP_FRAC,
               window="hann", detrend=True):
    """One-sided cross-power spectral density matrix of ``X`` (channels, n).

    Returns ``(f, G, n_seg, k_eff)`` with ``G`` of shape ``(nf, m, m)``,
    Hermitian and positive semidefinite at every line, in the convention
    ``G[p, q] = E[Y_p conj(Y_q)]``.  ``k_eff`` is the number of INDEPENDENT
    averages the overlapped segments are worth, which is what every
    prominence floor downstream is set from.  The diagonal is checked
    against ``scipy.signal.welch`` and the off-diagonal against
    ``scipy.signal.csd`` in the verification, so the scaling is not asserted
    here but measured.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    m, n = X.shape
    nperseg = int(min(nperseg, n))
    nover = int(round(overlap_frac * nperseg))
    step = max(1, nperseg - nover)
    starts = np.arange(0, n - nperseg + 1, step)
    win = get_window(window, nperseg)
    scale = 1.0 / (fs * np.sum(win ** 2))

    nf = nperseg // 2 + 1
    G = np.zeros((nf, m, m), dtype=complex)
    for a in starts:
        seg = X[:, a:a + nperseg]
        if detrend:
            seg = seg - seg.mean(axis=1, keepdims=True)
        Y = np.fft.rfft(seg * win, axis=1)                 # (m, nf)
        G += np.einsum('pk,qk->kpq', Y, Y.conj())
    G *= scale / len(starts)
    if nperseg % 2 == 0:                       # a Nyquist line exists
        G[1:-1] *= 2.0
    else:
        G[1:] *= 2.0
    f = np.fft.rfftfreq(nperseg, 1.0 / fs)
    return f, G, len(starts), effective_averages(win, step, len(starts))


def effective_averages(win, step, n_seg):
    """Independent averages an overlapped Welch estimate is worth.

    Overlapping segments share data, so ``n_seg`` is not the number of
    independent looks and using it as one sets a peak-picking prominence
    floor that is too low, which manufactures peaks.  For a Gaussian process
    the correlation between the periodograms of two segments offset by
    ``m`` steps is

        rho(m) = |sum_n w(n) w(n + m S)|^2 / (sum_n w(n)^2)^2

    and the variance of the average of ``K`` of them is
    ``[K + 2 sum_m (K - m) rho(m)] / K^2``, so the estimate is worth
    ``K^2 / [K + 2 sum_m (K - m) rho(m)]`` independent averages.  At
    87.5 per cent Hann overlap this is 0.27 of the segment count, and the
    difference decides whether a 2 dB feature is called a peak.
    """
    w = np.asarray(win, dtype=float)
    N, den = len(w), np.sum(w ** 2) ** 2
    tot = float(n_seg)
    for m in range(1, n_seg):
        sh = m * step
        if sh >= N:
            break
        rho = np.sum(w[:N - sh] * w[sh:]) ** 2 / den
        tot += 2.0 * (n_seg - m) * rho
    return n_seg ** 2 / tot


def sv_decomp(G):
    """Singular values and vectors of a Hermitian positive matrix stack.

    ``G`` is Hermitian by construction, so the singular value decomposition
    is its eigendecomposition and ``numpy.linalg.eigh`` is used, which
    returns real eigenvalues and an orthonormal basis without the sign
    ambiguity a general singular value routine carries.  Agreement with
    ``numpy.linalg.svd`` is checked in the verification.

    Returns ``(S, U)`` with ``S`` of shape ``(nf, m)`` in descending order
    and ``U`` of shape ``(nf, m, m)``, column ``c`` the vector of ``S[:, c]``.
    """
    ev, V = np.linalg.eigh(0.5 * (G + np.conj(np.swapaxes(G, -1, -2))))
    S = np.maximum(ev[..., ::-1], 0.0)
    U = V[..., ::-1]
    return S, U


def mac(a, b):
    """Modal assurance criterion between two complex vectors."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    den = (np.vdot(a, a).real * np.vdot(b, b).real)
    if den <= 0:
        return 0.0
    return float(abs(np.vdot(a, b)) ** 2 / den)


# ===========================================================================
# the exact cross-spectrum of the model, at three levels of the chain
# ===========================================================================

def _modal_A(sim, fgrid, dofs, mode, modes=None):
    """Acceleration transfer matrix ``A[p, j]`` at the requested sensors.

    ``mode="cont"`` is the continuous system, ``mode="disc"`` the
    zero-order-hold sections that ``lfilter`` actually runs, so that the
    hold and every alias the internal sampling folds in are already inside
    the answer.  The two mirror ``RecordSimulator._quad`` exactly.
    """
    f = np.asarray(fgrid, dtype=float)
    j = slice(None) if modes is None else np.asarray(modes, dtype=int)
    ph = sim.Phi[np.asarray(dofs, dtype=int), :][:, j]     # (m, nm)
    if mode == "cont":
        om = 2.0 * np.pi * f
        den = (sim.w[j] ** 2)[None, :] - (om ** 2)[:, None] \
            + 2j * (sim.zj[j] * sim.w[j])[None, :] * om[:, None]
        h = (-om ** 2)[:, None] / den                      # (nf, nm)
    else:
        e = np.exp(-2j * np.pi * f * sim.dt_int)[:, None]
        num = (sim.b_iir[j, 0][None, :] + sim.b_iir[j, 1][None, :] * e
               + sim.b_iir[j, 2][None, :] * e ** 2)
        den = (sim.a_iir[j, 0][None, :] + sim.a_iir[j, 1][None, :] * e
               + sim.a_iir[j, 2][None, :] * e ** 2)
        h = num / den
    return ph[None, :, :] * h[:, None, :]                  # (nf, m, nm)


def cpsd_model(sim, fgrid, dofs, mode="cont", modes=None):
    """One-sided cross-spectrum ``2 A Gamma A^H`` at unit load intensity.

    The diagonal reproduces ``RecordSimulator.psd_ideal`` for ``mode="cont"``
    and ``psd_internal`` for ``mode="disc"``, which the verification asserts
    to machine precision rather than trusting.
    """
    A = _modal_A(sim, fgrid, dofs, mode, modes)
    Gam = (sim.Gamma if modes is None
           else sim.Gamma[np.ix_(np.asarray(modes, dtype=int),
                                 np.asarray(modes, dtype=int))])
    B = A @ Gam.astype(complex)
    return 2.0 * (B @ np.conj(np.swapaxes(A, -1, -2)))


def cpsd_recorded(sim, fgrid, dofs, aa=True, decim=True):
    """One-sided cross-spectrum of the RECORDED signals, unit intensity.

    The internal cross-spectrum, weighted by the anti-alias filter and
    folded by the decimation.  A cross-spectrum of real signals satisfies
    ``G(-f) = conj(G(f))``, so an alias landing at a negative frequency
    contributes its conjugate; a power spectrum has no such term, which is
    why this cannot simply reuse ``psd_recorded``.  The diagonal is checked
    against ``psd_recorded`` in the verification.
    """
    f = np.asarray(fgrid, dtype=float)
    dofs = np.asarray(dofs, dtype=int)
    R = sim.oversample if decim else 1
    fs_out = sim.fs if decim else sim.fs_int
    nyq = sim.fs_int / 2.0 + 1e-9
    out = np.zeros((f.size, len(dofs), len(dofs)), dtype=complex)
    for k in range(-R - 1, R + 2):
        g = f + k * fs_out
        m = np.abs(g) <= nyq
        if not m.any():
            continue
        gg = np.abs(g[m])
        Gk = cpsd_model(sim, gg, dofs, mode="disc")
        neg = g[m] < 0.0
        Gk[neg] = Gk[neg].conj()
        if aa:
            _, H = freqz(sim.aa_taps, worN=gg, fs=sim.fs_int)
            Gk = Gk * (np.abs(H) ** 2)[:, None, None]
        out[m] += Gk
    return out


def welch_kernel(nperseg, fs, window="hann", os_fine=16, half_bins=96):
    """The kernel a Welch estimate convolves the true spectrum with.

    ``|W(f)|^2`` normalised to unit area on a grid ``os_fine`` times finer
    than the bin.  Returned separately from any spectrum so that the same
    kernel can be applied to a matrix of cross-spectra and to an analytic
    single-mode curve, which is what makes the estimator bias in check [F]
    a prediction rather than a fitted correction.
    """
    df = fs / nperseg
    dfine = df / os_fine
    win = get_window(window, nperseg)
    npad = nperseg * os_fine
    K = np.fft.fftshift(np.abs(np.fft.fft(win, npad)) ** 2)
    gk = (np.arange(npad) - npad // 2) * (fs / npad)
    K = K[np.abs(gk) <= half_bins * df]
    return K / (K.sum() * dfine), dfine


def expected_welch_cpsd(sim, fbins, dofs, nperseg=NPERSEG, window="hann",
                        os_fine=16, half_bins=96):
    """Expected Welch cross-spectrum: the exact one, smeared by the window.

    A Welch estimate of a resonance a few bins wide estimates the true
    spectrum convolved with ``|W(f)|^2`` normalised to unit area, not the
    true spectrum.  The smearing is linear, so it applies entry by entry to
    a cross-spectral matrix, real and imaginary parts alike.  Comparing a
    Welch estimate with an unsmeared prediction measures the window.
    """
    fbins = np.asarray(fbins, dtype=float)
    dofs = np.asarray(dofs, dtype=int)
    df = sim.fs / nperseg
    K, dfine = welch_kernel(nperseg, sim.fs, window, os_fine, half_bins)

    lo = fbins.min() - half_bins * df - dfine
    hi = fbins.max() + half_bins * df + dfine
    n = int(np.ceil((hi - lo) / dfine)) + 1
    gf = lo + dfine * np.arange(n)
    Gf = cpsd_recorded(sim, np.abs(gf), dofs)
    Gf[gf < 0.0] = Gf[gf < 0.0].conj()

    m = len(dofs)
    out = np.zeros((len(fbins), m, m), dtype=complex)
    for p in range(m):
        for q in range(m):
            re = np.convolve(Gf[:, p, q].real, K * dfine, mode="same")
            im = np.convolve(Gf[:, p, q].imag, K * dfine, mode="same")
            out[:, p, q] = (np.interp(fbins, gf, re)
                            + 1j * np.interp(fbins, gf, im))
    return out


# ===========================================================================
# the four estimators
# ===========================================================================

def _smooth_db(f, P, half_width_hz):
    """The decibel curve ``refine_peaks`` smooths, reproduced for heights."""
    db = 10.0 * np.log10(np.maximum(np.asarray(P, dtype=float), 1e-300))
    nsm = max(1, int(round(half_width_hz / (f[1] - f[0]))))
    return np.convolve(db, np.ones(nsm) / nsm, mode="same"), nsm


def pick_curve(f, P, n_seg, half_width_hz, n_sigma=4.0):
    """Peaks of one spectral curve, with heights, in a stated band.

    Frequencies come from ``simulate_records.refine_peaks`` unchanged, so
    the baseline picker here is the one that module already verified against
    the analytic spectrum.  Heights are read off the same smoothed decibel
    curve at the refined frequencies.
    """
    fp, prom = refine_peaks(f, P, n_seg, half_width_hz, n_sigma=n_sigma)
    sm, _ = _smooth_db(f, P, half_width_hz)
    h = np.interp(fp, f, sm) if len(fp) else np.array([])
    return fp, h, prom


def peak_pick(f, P, n_seg, half_width_hz, band=BAND):
    """The baseline: every peak of an auto-spectrum, tallest first."""
    m = (f >= band[0]) & (f <= band[1])
    fp, h, prom = pick_curve(f[m], P[m], n_seg, half_width_hz)
    if not len(fp):
        return dict(n=0, f=np.array([]), h=np.array([]), f_top=np.nan,
                    prom_db=prom)
    o = np.argsort(-h)
    return dict(n=len(fp), f=fp, h=h, f_top=float(fp[o[0]]), prom_db=prom,
                f_sorted=fp[o], h_sorted=h[o])


def half_power(f, P, band=BAND, refine=True, smooth_bins=SMOOTH_BINS):
    """Frequency and damping from the -3 dB width of the tallest peak.

    A power spectral density is already a squared magnitude, so the
    half-power points are where the density falls to half its maximum, and
    ``zeta = (f2 - f1) / (2 f0)``.

    The width is measured on the density smoothed over ``smooth_bins``
    lines, which is what a practitioner does and what the method needs: a
    Welch density is chi-squared about its mean with a scatter of about
    27 per cent per bin here, so on the raw curve the first line that
    happens to fall below half the maximum arrives early and the width comes
    out about a third too small.  Smoothing is arithmetic on the density,
    not on its logarithm, because the arithmetic mean of a chi-squared
    variable is unbiased and the geometric mean is not.  The residual bias
    the smoothing itself leaves is measured in check [F].
    """
    m = np.where((f >= band[0]) & (f <= band[1]))[0]
    fb = f[m]
    Pb = np.asarray(P, dtype=float)[m]
    nsm = max(1, int(smooth_bins))
    if nsm > 1:
        Pb = np.convolve(Pb, np.ones(nsm) / nsm, mode="same")
    i = int(np.argmax(Pb))
    f0 = float(fb[i])
    if refine and 0 < i < len(fb) - 1:
        y = np.log(np.maximum(Pb[i - 1:i + 2], 1e-300))
        den = y[0] - 2 * y[1] + y[2]
        if den < 0:
            f0 = float(fb[i] - 0.5 * (fb[1] - fb[0]) * (y[2] - y[0]) / den)
    half = 0.5 * Pb[i]
    j = i
    while j > 0 and Pb[j] > half:
        j -= 1
    k = i
    while k < len(Pb) - 1 and Pb[k] > half:
        k += 1
    if Pb[j] > half or Pb[k] > half:
        return dict(ok=False, f0=f0, zeta=np.nan, f1=np.nan, f2=np.nan,
                    bw=np.nan)
    f1 = float(np.interp(half, [Pb[j], Pb[j + 1]], [fb[j], fb[j + 1]]))
    f2 = float(np.interp(half, [Pb[k], Pb[k - 1]], [fb[k], fb[k - 1]]))
    return dict(ok=True, f0=f0, zeta=(f2 - f1) / (2.0 * f0), f1=f1, f2=f2,
                bw=f2 - f1)


def fdd(f, S, U, n_seg, half_width_hz, band=BAND, n_sv=2):
    """Peaks of the singular value curves, with their singular vectors."""
    m = np.where((f >= band[0]) & (f <= band[1]))[0]
    fb = f[m]
    out = dict(f=fb, S=S[m], U=U[m], n_sv=min(n_sv, S.shape[1]))
    for c in range(out["n_sv"]):
        y = S[m, c]
        fp, h, prom = pick_curve(fb, y, n_seg, half_width_hz)
        o = np.argsort(-h)
        out["sv%d" % (c + 1)] = dict(
            n=len(fp), f=fp[o] if len(fp) else fp, h=h[o] if len(fp) else h,
            f_top=float(fp[o[0]]) if len(fp) else np.nan, prom_db=prom,
            i_top=(int(np.argmin(np.abs(fb - fp[o[0]]))) if len(fp)
                   else int(np.argmax(y))),
            f_argmax=float(fb[int(np.argmax(y))]),
            shape=U[m][int(np.argmax(y)), :, c])
    out["sv2_over_sv1_max"] = (float(np.max(S[m, 1] / np.maximum(S[m, 0],
                                                                1e-300)))
                               if S.shape[1] > 1 else 0.0)
    # what an analyst reads off the plot: the height of the sv2 hump against
    # the height of the sv1 peak, which is not the same as the largest
    # pointwise ratio and is the honest measure of whether sv2 shows a mode
    out["sv2_peak_over_sv1_peak"] = (float(S[m, 1].max() / S[m, 0].max())
                                     if S.shape[1] > 1 else 0.0)
    return out


def efdd(f, S, U, i_peak, which=0, mac_min=MAC_MIN, fs=None, nperseg=None,
         lo_frac=0.20, hi_frac=0.95, min_bins=5, min_extrema=6,
         db_floor=20.0, rise_db=2.0, smooth_bins=SMOOTH_BINS):
    """The enhanced step: an SDOF bell, its correlation function, a fit.

    The bell is grown outward from the peak line while three conditions
    hold, and which of them stops it is recorded, because it says what
    limited the estimate:

      ``mac``     no singular vector at that line still matches the peak
                  vector to ``mac_min``.  The match is allowed to jump
                  between singular values, which is the standard refinement
                  and matters here: where two singular value curves veer
                  they exchange their vectors, so a bell locked to index
                  zero would be cut in half at exactly the frequency where
                  the pair is interesting.
      ``valley``  the curve has risen ``rise_db`` above the lowest point
                  reached so far, so the bell has passed the trough between
                  this resonance and the next, and the bell is trimmed back
                  to that trough.  Without this bound a two-channel bell
                  runs across the whole band, takes in the neighbouring mode
                  and returns one merged estimate with several times the
                  true damping.  The margin is needed because a Welch
                  singular value is chi-squared about its mean and a bare
                  "the curve turned up" test fires on the first upward
                  wiggle, which cut the bell to seven lines and halved the
                  damping in an early version of this function.
      ``level``   the curve has fallen ``db_floor`` below the peak.

    The bell is inverse transformed to a correlation function, the frequency
    comes from a regression of the extremum times on the extremum index and
    the damping from a regression of the log envelope on time, which is the
    logarithmic decrement written as a least squares fit.
    """
    nf, m = S.shape
    u_ref = U[i_peak, :, which]

    # the best-matching singular value at every line, once
    vsel = np.zeros(nf)
    csel = np.zeros(nf, dtype=int)
    msel = np.zeros(nf)
    for k in range(nf):
        cs = [mac(U[k, :, c], u_ref) for c in range(m)]
        c = int(np.argmax(cs))
        csel[k], msel[k], vsel[k] = c, cs[c], S[k, c]
    nsm = max(1, int(smooth_bins))
    vsm = np.convolve(vsel, np.ones(nsm) / nsm, mode="same")

    val = np.zeros(nf)
    inb = np.zeros(nf, dtype=bool)
    val[i_peak], inb[i_peak] = vsel[i_peak], True
    peak_val = vsel[i_peak]
    floor = peak_val * 10.0 ** (-db_floor / 10.0)
    stops = {}
    rise = 10.0 ** (rise_db / 10.0)
    for step in (-1, 1):
        k, why = i_peak + step, "edge"
        vmin, kmin, run = vsm[i_peak], i_peak, []
        while 0 <= k < nf:
            if msel[k] < mac_min:
                why = "mac"
                break
            if vsel[k] < floor:
                why = "level"
                break
            if vsm[k] > rise * vmin:
                why = "valley"
                break
            run.append(k)
            if vsm[k] < vmin:
                vmin, kmin = vsm[k], k
            k += step
        if why == "valley":                       # trim back to the trough
            run = [q for q in run if abs(q - i_peak) <= abs(kmin - i_peak)]
        for q in run:
            val[q], inb[q] = vsel[q], True
        stops["lo" if step < 0 else "hi"] = why

    lo = int(np.argmax(inb))
    hi = nf - 1 - int(np.argmax(inb[::-1]))
    n_bell = int(inb.sum())
    res = dict(n_bell=n_bell, f_lo_bell=float(f[lo]), f_hi_bell=float(f[hi]),
               mac_min_in_bell=float(msel[inb].min()) if n_bell else np.nan,
               stop_lo=stops.get("lo"), stop_hi=stops.get("hi"),
               f_efdd=np.nan, zeta_efdd=np.nan, n_extrema=0, r2=np.nan,
               f_zc=np.nan, ok=False, shape=None)
    if n_bell < min_bins or fs is None:
        return res

    # the bell as a one-sided spectrum on the full bin grid, back to time
    nfft = nperseg if nperseg is not None else 2 * (len(f) - 1)
    Sb = np.zeros(nfft // 2 + 1)
    i0 = int(round(f[0] * nfft / fs))
    Sb[i0:i0 + nf] = val
    R = np.fft.irfft(Sb, n=nfft)
    R = R / R[0]
    t = np.arange(nfft) / fs

    d = np.diff(R)
    ext = np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0] + 1
    if len(ext) < min_extrema:
        return res
    a = np.abs(R[ext])
    keep = (a <= hi_frac) & (a >= lo_frac)
    if keep.sum() < min_extrema:
        return res
    i_first = int(np.argmax(keep))
    run = i_first
    while run + 1 < len(keep) and keep[run + 1]:
        run += 1
    sel = np.arange(i_first, run + 1)
    if len(sel) < min_extrema:
        return res

    te, ae, ke = t[ext[sel]], a[sel], np.arange(len(sel), dtype=float)
    sl_a, ic_a = np.polyfit(te, np.log(ae), 1)
    yy = np.log(ae)
    ss = np.sum((yy - yy.mean()) ** 2)
    rr = np.sum((yy - (sl_a * te + ic_a)) ** 2)
    sl_t = np.polyfit(ke, te, 1)[0]                 # seconds per extremum
    wd = np.pi / sl_t
    ratio = -sl_a / wd                              # zeta / sqrt(1 - zeta^2)
    zeta = ratio / np.sqrt(1.0 + ratio ** 2)
    wn = wd / np.sqrt(max(1.0 - zeta ** 2, 1e-12))

    zc = np.where(np.sign(R[:-1]) * np.sign(R[1:]) < 0)[0]
    zc = zc[zc <= ext[sel[-1]]]
    if len(zc) > 2:
        f_zc = 0.5 / np.polyfit(np.arange(len(zc), dtype=float), t[zc], 1)[0]
    else:
        f_zc = np.nan

    # enhanced shape: singular values as weights over the bell
    Ub = np.zeros((m,), dtype=complex)
    for k in np.where(inb)[0]:
        v = U[k, :, csel[k]]
        Ub += vsel[k] * v * np.exp(-1j * np.angle(np.vdot(u_ref, v)))
    nb = np.linalg.norm(Ub)
    res.update(ok=True, f_efdd=float(wn / (2.0 * np.pi)),
               zeta_efdd=float(zeta), n_extrema=int(len(sel)),
               r2=float(1.0 - rr / ss) if ss > 0 else np.nan,
               f_zc=float(f_zc), shape=(Ub / nb if nb > 0 else Ub))
    return res


# ===========================================================================
# a synthetic case whose answer is known exactly
# ===========================================================================

class ModalTruth:
    """Records from a modal model with stated frequencies, damping, shapes.

    The point of this class is that its answer is not computed, it is
    imposed: the frequencies, the damping ratios, the mode shapes at the
    sensors and the modal force cross-spectral matrix are all inputs, and
    the exact response cross-spectrum ``2 A Gamma A^H`` follows in closed
    form.  Any estimator can therefore be scored against a truth that owes
    nothing to the finite element model.

    The chain is the one ``simulate_records`` uses and for the same reasons:
    exact zero-order-hold second-order sections, eightfold oversampling, a
    linear-phase Kaiser anti-alias filter at 0.4 fs, decimation, then white
    sensor noise.
    """

    def __init__(self, freqs, zetas, shapes, Gamma=None, fs=100.0,
                 oversample=OVERSAMPLE):
        self.f = np.asarray(freqs, dtype=float)
        self.z = np.asarray(zetas, dtype=float)
        self.Psi = np.asarray(shapes, dtype=float)          # (m, nm)
        self.w = 2.0 * np.pi * self.f
        self.fs = float(fs)
        self.oversample = int(oversample)
        self.fs_int = self.fs * self.oversample
        self.dt_int = 1.0 / self.fs_int
        nm = len(self.f)
        self.Gamma = np.eye(nm) if Gamma is None else np.asarray(Gamma,
                                                                 dtype=float)
        self.Gchol = np.linalg.cholesky(
            self.Gamma + 1e-14 * np.trace(self.Gamma) / nm * np.eye(nm))

        self.b = np.zeros((nm, 3))
        self.a = np.zeros((nm, 3))
        for j in range(nm):
            w, z = self.w[j], self.z[j]
            A = np.array([[0.0, 1.0], [-w * w, -2.0 * z * w]])
            B = np.array([[0.0], [1.0]])
            Cm = np.array([[-w * w, -2.0 * z * w]])
            aug = np.zeros((3, 3))
            aug[:2, :2] = A * self.dt_int
            aug[:2, 2:] = B * self.dt_int
            E = expm(aug)
            num, den = ss2tf(E[:2, :2], E[:2, 2:], Cm, np.array([[1.0]]))
            self.b[j], self.a[j] = num[0], den

        from scipy.signal import firwin
        self.aa_taps = firwin(40 * self.oversample + 1, 0.4 * self.fs,
                              window=("kaiser", 8.6), fs=self.fs_int)

    def cpsd_exact(self, fgrid, mode="cont"):
        """Exact one-sided response cross-spectrum, unit force intensity."""
        f = np.asarray(fgrid, dtype=float)
        if mode == "cont":
            om = 2.0 * np.pi * f
            den = (self.w ** 2)[None, :] - (om ** 2)[:, None] \
                + 2j * (self.z * self.w)[None, :] * om[:, None]
            h = (-om ** 2)[:, None] / den
        else:
            e = np.exp(-2j * np.pi * f * self.dt_int)[:, None]
            h = ((self.b[:, 0] + self.b[:, 1] * e + self.b[:, 2] * e ** 2)
                 / (self.a[:, 0] + self.a[:, 1] * e + self.a[:, 2] * e ** 2))
        A = self.Psi[None, :, :] * h[:, None, :]
        B = A @ self.Gamma.astype(complex)
        return 2.0 * (B @ np.conj(np.swapaxes(A, -1, -2)))

    def record(self, duration=600.0, seed=0, snr_db=None, burn=120.0):
        nkeep = int(round(duration * self.fs))
        pad = len(self.aa_taps) // 2 + self.oversample
        nburn = int(round(burn * self.fs_int))
        nint = nburn + nkeep * self.oversample + 2 * pad
        rng = np.random.default_rng(seed)
        nm = len(self.f)
        F = (self.Gchol @ rng.standard_normal((nm, nint))) \
            / np.sqrt(self.dt_int)
        Y = np.empty_like(F)
        for j in range(nm):
            Y[j] = lfilter(self.b[j], self.a[j], F[j])
        y = self.Psi @ Y
        yf = np.vstack([fftconvolve(r, self.aa_taps, mode="same") for r in y])
        yf = yf[:, nburn + pad:nburn + pad + nkeep * self.oversample]
        y = yf[:, ::self.oversample]
        if snr_db is not None:
            s = y.std(axis=1) / 10.0 ** (snr_db / 20.0)
            y = y + s[:, None] * rng.standard_normal(y.shape)
        return np.arange(nkeep) / self.fs, y


# ===========================================================================
# helpers tying the estimators to the bridge
# ===========================================================================

def string_tension(f, bridge=None, n=1):
    """The incumbent inversion, one estimate from one frequency."""
    b = BRIDGE if bridge is None else bridge
    return float(invert_string(np.array([f]), b["Lc"], b["mc"],
                               np.array([n]))[0])


def errors(f_hat, T_true, f_iso, bridge=None):
    """Both error measures the study uses, in per cent."""
    if not np.isfinite(f_hat):
        return np.nan, np.nan, np.nan
    T_hat = string_tension(f_hat, bridge)
    return (T_hat, 100.0 * (T_hat - T_true) / T_true,
            100.0 * ((f_hat / f_iso) ** 2 - 1.0))


def two_channel_record(sim, duration, snr_db, seed, noise_ref="per_channel",
                       quantise=True):
    """The default sensor pair: stay accelerometer and deck anchorage."""
    r = sim.record(duration=duration, snr_db=snr_db, seed=seed,
                   noise_ref=noise_ref, quantise=quantise)
    return np.vstack([r["a_stay"], r["a_deck"]]), r


def extra_channel_record(sim, dof, duration, snr_db, seed,
                         noise_ref="per_channel", quantise=True):
    """A record whose second channel sits at ``dof`` instead of the deck.

    The modal forces are drawn before any sensor is touched and the sensor
    noise is drawn from the same generator at the same shape, so the stay
    channel of this record is bit identical to the stay channel of
    :func:`two_channel_record` at the same seed.  That identity is asserted
    in the verification, and it is what lets channels from separate calls be
    stacked into one consistent multi-sensor record without reimplementing
    the measurement chain.
    """
    old = sim.dofs
    sim.dofs = np.array([sim.dof_stay, int(dof)])
    try:
        r = sim.record(duration=duration, snr_db=snr_db, seed=seed,
                       noise_ref=noise_ref, quantise=quantise)
    finally:
        sim.dofs = old
    return np.vstack([r["a_stay"], r["a_deck"]]), r


def pair_geometry(sim):
    """Everything about the pair that explains what FDD can and cannot do."""
    i, j = sim.pair_idx
    G = sim.Gamma
    corr = float(G[i, j] / np.sqrt(G[i, i] * G[j, j]))
    ev = np.linalg.eigvalsh(np.array([[G[i, i], G[i, j]],
                                      [G[i, j], G[j, j]]]))
    dofs = [sim.dof_stay, sim.dof_deck]
    v1, v2 = sim.Phi[dofs, i], sim.Phi[dofs, j]
    return dict(gamma_corr=corr, gamma_rank2_ratio=float(ev[0] / ev[1]),
                mac_sensor=mac(v1, v2),
                mac_full=mac(sim.Phi[:, i], sim.Phi[:, j]),
                deck_over_stay_lo=float(abs(v1[1] / v1[0])),
                deck_over_stay_hi=float(abs(v2[1] / v2[0])))


VERIFICATION_NOTES = """
[A] the cross-spectral estimator: diagonal against scipy.signal.welch, off
    diagonal against scipy.signal.csd, and the singular value decomposition
    against numpy.linalg.svd
[B] a synthetic modal truth with stated frequencies, damping, mode shapes
    and UNCORRELATED modal forces: FDD, EFDD, half power and peak picking
    against the imposed answer, for a separated triple and for a pair as
    close as the worked bridge's
[C] one channel: the singular value equals the auto-spectrum and FDD
    reduces to peak picking, bit for bit
[D] the exact model cross-spectrum: diagonal against
    RecordSimulator.psd_ideal, psd_internal and psd_recorded, then the
    Welch estimate of long noise-free bridge records against the exact
    cross-spectrum convolved with the Welch window kernel
[E] the multi-channel construction: the stay channel is bit identical
    across calls at the same seed
[F] half power on an exact Lorentzian, and EFDD on an exact analytic bell,
    then on the same bell smeared by the Welch kernel, which predicts the
    resolution bias of check [G] without a Monte Carlo
[G] a SINGLE isolated mode at the sweep's own processing settings, which
    measures what each estimator does when there is no crossing at all and
    is the baseline every merged case has to be read against
[H] the rank argument: modal forces forced to rank one give a
    cross-spectral matrix of rank one at every line, for two sensors and
    for seven, and a spatially coherent load does the same
"""


def _fmt(x, n=5):
    return "nan" if not np.isfinite(x) else ("%%.%df" % n) % x


# ===========================================================================
# verification
# ===========================================================================

def verify(args):
    rows = []

    def add(check, **kw):
        rows.append(dict(check=check, **kw))

    print("FDD / EFDD VERIFICATION")
    print("=" * 74)

    # -- [A] estimator plumbing -------------------------------------------
    rng = np.random.default_rng(0)
    n = 200000
    x = rng.standard_normal(n)
    y = np.convolve(rng.standard_normal(n + 8), np.ones(9) / 3.0,
                    mode="valid")[:n] + 0.3 * x
    X = np.vstack([x, y])
    fw, G, nseg, keff = cpsd_welch(X, 100.0, nperseg=4096)
    from scipy.signal import csd as scsd
    nov = int(OVERLAP_FRAC * 4096)
    f1, P0 = welch(x, fs=100.0, nperseg=4096, noverlap=nov,
                   window="hann", detrend="constant")
    _, P1 = welch(y, fs=100.0, nperseg=4096, noverlap=nov,
                  window="hann", detrend="constant")
    _, C01 = scsd(y, x, fs=100.0, nperseg=4096, noverlap=nov,
                  window="hann", detrend="constant")
    e00 = float(np.max(np.abs(G[:, 0, 0].real - P0)) / P0.max())
    e11 = float(np.max(np.abs(G[:, 1, 1].real - P1)) / P1.max())
    e01 = float(np.max(np.abs(G[:, 0, 1] - C01)) / np.abs(C01).max())
    S, U = sv_decomp(G)
    Ssvd = np.linalg.svd(G, compute_uv=False)
    esv = float(np.max(np.abs(S - Ssvd)) / S.max())
    her = float(np.max(np.abs(G - np.conj(np.swapaxes(G, -1, -2))))
                / np.abs(G).max())
    print("\n[A] cross-spectral estimator, %d segments of 4096" % nseg)
    print("    diag vs scipy.signal.welch : %.2e , %.2e  of band peak"
          % (e00, e11))
    print("    off  vs scipy.signal.csd   : %.2e  (convention "
          "G[p,q] = E[Y_p conj(Y_q)] = csd(x_q, x_p))" % e01)
    print("    eigh vs numpy.linalg.svd   : %.2e ; Hermitian to %.2e"
          % (esv, her))
    add("A_estimator", welch_00=e00, welch_11=e11, csd_01=e01, svd=esv,
        hermitian=her, n_seg=nseg)

    # -- [B] synthetic modal truth ----------------------------------------
    print("\n[B] synthetic modal truth: frequencies, damping, mode shapes and")
    print("    modal force correlation are all IMPOSED, so the answer is")
    print("    known and not computed")
    sim0 = RecordSimulator(T_TUNE, 0.005)
    i, j = sim0.pair_idx
    Psi2 = sim0.Phi[[sim0.dof_stay, sim0.dof_deck], :][:, [i, j]]
    Gb = np.array([[sim0.Gamma[i, i], sim0.Gamma[i, j]],
                   [sim0.Gamma[i, j], sim0.Gamma[j, j]]])
    Gb = Gb / np.sqrt(Gb[0, 0] * Gb[1, 1])
    fpair = np.array([sim0.f_lo, sim0.f_hi])
    Psi3 = np.array([[1.0, 1.0, 1.0], [0.4, -0.7, 0.2]])

    cases = [("separated triple, independent forces",
              np.array([2.10, 3.30, 5.05]), np.full(3, 0.005), Psi3,
              np.eye(3))]
    for z in (0.005, 0.020, 0.030):
        cases.append(("pair zeta %.1f %%, independent forces" % (100 * z),
                      fpair, np.full(2, z), Psi2, np.eye(2)))
        cases.append(("pair zeta %.1f %%, bridge force correlation"
                      % (100 * z), fpair, np.full(2, z), Psi2, Gb))

    dur, nps = 3600.0, 32768
    for name, fr, zt, Psi, Gam in cases:
        mt = ModalTruth(fr, zt, Psi, Gamma=Gam)
        t, Y = mt.record(duration=dur, seed=1, snr_db=None)
        fw, G, nseg, keff = cpsd_welch(Y, mt.fs, nperseg=nps)
        Sv, Uv = sv_decomp(G)
        df = mt.fs / nps
        band = (fr.min() - 0.35, fr.max() + 0.35)
        res = fdd(fw, Sv, Uv, keff, SMOOTH_BINS * df, band=band)
        pp = peak_pick(fw, G[:, 0, 0].real, keff, SMOOTH_BINS * df,
                       band=band)
        hp = half_power(fw, G[:, 0, 0].real, band=band)
        fx = np.linspace(band[0], band[1], 40001)
        Gx = mt.cpsd_exact(fx)
        Sx, _ = sv_decomp(Gx)
        sv2_exact = float(np.max(Sx[:, 1] / Sx[:, 0])) if len(fr) > 1 else 0.0
        sv2pk_exact = float(Sx[:, 1].max() / Sx[:, 0].max()) \
            if len(fr) > 1 else 0.0
        u = ((fr[1] - fr[0]) / fr.mean() / (2 * zt[0])) if len(fr) == 2 \
            else np.nan
        print("\n    %s" % name)
        print("      truth %s Hz, zeta %s, u = %s"
              % (np.round(fr, 5), np.round(zt, 4), _fmt(u, 3)))
        print("      stay auto-spectrum peaks : %s Hz"
              % np.round(np.sort(pp["f"]), 5))
        print("      FDD sv1 peaks            : %s Hz"
              % np.round(np.sort(res["sv1"]["f"]), 5))
        print("      FDD sv2 peaks            : %s Hz"
              % np.round(np.sort(res["sv2"]["f"]), 5))
        print("      max sv2/sv1: exact %.3e, from the record %.3e ; "
              "sv2 PEAK over sv1 PEAK: exact %.3e, record %.3e"
              % (sv2_exact, res["sv2_over_sv1_max"], sv2pk_exact,
                 res["sv2_peak_over_sv1_peak"]))
        fb, Sb, Ub = res["f"], res["S"], res["U"]
        eff = []
        for c in (0, 1):
            k = "sv%d" % (c + 1)
            for fpk in np.sort(res[k]["f"]):
                ip = int(np.argmin(np.abs(fb - fpk)))
                e = efdd(fb, Sb, Ub, ip, which=c, fs=mt.fs, nperseg=nps)
                if not e["ok"]:
                    continue
                eff.append((c + 1, fpk, e))
                print("      EFDD from sv%d at %.5f  : f %.5f Hz, zeta "
                      "%.5f  (%3d bins %.3f-%.3f Hz, stops %s/%s, R2 %.4f)"
                      % (c + 1, fpk, e["f_efdd"], e["zeta_efdd"],
                         e["n_bell"], e["f_lo_bell"], e["f_hi_bell"],
                         e["stop_lo"], e["stop_hi"], e["r2"]))
        print("      half power on the stay channel: f %.5f Hz, zeta %.5f "
              "(%+.1f %% of the true %.4f)"
              % (hp["f0"], hp["zeta"], 100 * (hp["zeta"] / zt[0] - 1),
                 zt[0]))
        got = np.sort(np.unique(np.round(np.concatenate(
            [res["sv1"]["f"], res["sv2"]["f"]]), 6))) \
            if (res["sv1"]["n"] + res["sv2"]["n"]) else np.array([])
        for k, ft in enumerate(fr):
            near = got[np.argmin(np.abs(got - ft))] if len(got) else np.nan
            ze = [e for e in eff if abs(e[1] - ft) < 0.5 * (fr.max()
                                                            - fr.min() + 1e-9)]
            ze = sorted(ze, key=lambda e: abs(e[1] - ft))
            add("B_synthetic", case=name, mode=k + 1, f_true=ft,
                zeta_true=zt[k], u=u, f_nearest_fdd=near,
                f_err_pct=(100 * (near / ft - 1) if np.isfinite(near)
                           else np.nan),
                f_efdd=ze[0][2]["f_efdd"] if ze else np.nan,
                zeta_efdd=ze[0][2]["zeta_efdd"] if ze else np.nan,
                zeta_err_pct=(100 * (ze[0][2]["zeta_efdd"] / zt[k] - 1)
                              if ze else np.nan),
                n_bell=ze[0][2]["n_bell"] if ze else 0,
                n_peaks_auto=pp["n"], n_peaks_sv1=res["sv1"]["n"],
                n_peaks_sv2=res["sv2"]["n"],
                sv2_over_sv1=res["sv2_over_sv1_max"],
                sv2_over_sv1_exact=sv2_exact,
                sv2_peak_ratio=res["sv2_peak_over_sv1_peak"],
                sv2_peak_ratio_exact=sv2pk_exact,
                f_peakpick=pp["f_top"], f_halfpower=hp["f0"],
                zeta_halfpower=hp["zeta"], duration=dur, nperseg=nps,
                n_seg=nseg, k_eff=keff)

    # -- [C] one channel ---------------------------------------------------
    mt = ModalTruth(np.array([3.30]), np.array([0.005]),
                    np.array([[1.0]]), Gamma=np.eye(1))
    t, Y = mt.record(duration=600.0, seed=2, snr_db=40.0)
    f1c, G1, ns1, k1 = cpsd_welch(Y, mt.fs, nperseg=8192)
    S1, U1 = sv_decomp(G1)
    d = float(np.max(np.abs(S1[:, 0] - G1[:, 0, 0].real)))
    dr = d / G1[:, 0, 0].real.max()
    b = (3.0, 3.6)
    pk_fdd = fdd(f1c, S1, U1, k1, SMOOTH_BINS * mt.fs / 8192, band=b,
                 n_sv=1)["sv1"]["f"]
    pk_pp = peak_pick(f1c, G1[:, 0, 0].real, k1, SMOOTH_BINS * mt.fs / 8192,
                      band=b)["f"]
    same = bool(len(pk_fdd) == len(pk_pp)
                and np.array_equal(np.sort(pk_fdd), np.sort(pk_pp)))
    print("\n[C] one channel: sv1 minus auto-spectrum = %.2e (%.2e of peak);"
          % (d, dr))
    print("    FDD peaks %s and peak-pick peaks %s are the same array: %s"
          % (np.round(np.sort(pk_fdd), 6), np.round(np.sort(pk_pp), 6), same))
    print("    there is no second singular value, so the classical FDD "
          "argument is unavailable to a single stay accelerometer")
    add("C_single_channel", sv1_minus_psd=d, rel=dr, identical=same,
        n_sv=int(S1.shape[1]))

    # -- [D] the exact bridge cross-spectrum -------------------------------
    print("\n[D] the exact model cross-spectrum")
    sim = RecordSimulator(T_TUNE, 0.005)
    dofs = [sim.dof_stay, sim.dof_deck]
    fg = np.linspace(2.6, 4.1, 601)
    Gc = cpsd_model(sim, fg, dofs, "cont")
    Gi = cpsd_model(sim, fg, dofs, "disc")
    Gr = cpsd_recorded(sim, fg, dofs)
    e_c = float(np.max(np.abs(Gc[:, 0, 0].real - sim.psd_ideal(fg))
                       / sim.psd_ideal(fg).max()))
    e_i = float(np.max(np.abs(Gi[:, 0, 0].real - sim.psd_internal(fg))
                       / sim.psd_internal(fg).max()))
    e_r = float(np.max(np.abs(Gr[:, 0, 0].real - sim.psd_recorded(fg))
                       / sim.psd_recorded(fg).max()))
    e_h = float(np.max(np.abs(Gr - np.conj(np.swapaxes(Gr, -1, -2))))
                / np.abs(Gr).max())
    print("    diagonal vs psd_ideal %.2e , psd_internal %.2e , "
          "psd_recorded %.2e" % (e_c, e_i, e_r))
    print("    recorded matrix Hermitian to %.2e" % e_h)
    add("D_exact_cpsd", vs_psd_ideal=e_c, vs_psd_internal=e_i,
        vs_psd_recorded=e_r, hermitian=e_h)

    nps = 16384
    seeds = tuple(11 + 7 * k for k in range(args.nseed))
    Gs = []
    for sd in seeds:
        r = sim.record(duration=args.dur, snr_db=None, seed=sd,
                       quantise=False)
        X = np.vstack([r["a_stay_clean"], r["a_deck_clean"]])
        fwb, Gw, nsegb, _ = cpsd_welch(X, sim.fs, nperseg=nps)
        Gs.append(Gw)
    Gw = np.mean(Gs, axis=0)
    inten = r["meta"]["intensity"]
    mb = (fwb >= 2.6) & (fwb <= 4.1)
    Ge = expected_welch_cpsd(sim, fwb[mb], dofs, nperseg=nps) * inten
    r00 = (Gw[mb, 0, 0].real / Ge[:, 0, 0].real)
    r11 = (Gw[mb, 1, 1].real / Ge[:, 1, 1].real)
    r01 = (np.abs(Gw[mb, 0, 1]) / np.abs(Ge[:, 0, 1]))
    ph = np.angle(Gw[mb, 0, 1] * np.conj(Ge[:, 0, 1]))
    mw = (fwb > 0.4) & (fwb < 0.39 * sim.fs)
    Gew = expected_welch_cpsd(sim, fwb[mw], dofs, nperseg=nps) * inten
    w00 = Gw[mw, 0, 0].real / Gew[:, 0, 0].real
    w11 = Gw[mw, 1, 1].real / Gew[:, 1, 1].real
    w01 = np.abs(Gw[mw, 0, 1]) / np.abs(Gew[:, 0, 1])
    phw = np.angle(Gw[mw, 0, 1] * np.conj(Gew[:, 0, 1]))
    kt = len(seeds) * nsegb * 0.53                 # 75 %% overlap, measured
    print("    %d records of %.0f s, nperseg %d, 0.4-%.0f Hz, %d bins:"
          % (len(seeds), args.dur, nps, 0.39 * sim.fs, mw.sum()))
    coh = (np.abs(Gew[:, 0, 1]) ** 2
           / (Gew[:, 0, 0].real * Gew[:, 1, 1].real))
    hi = coh > 0.5
    print("      stay auto %.4f +- %.4f , deck auto %.4f , cross |G| %.4f ,"
          " phase %.4f rad rms"
          % (w00.mean(), w00.std() / np.sqrt(mw.sum()), w11.mean(),
             w01.mean(), np.sqrt((phw ** 2).mean())))
    print("      the wideband cross figure is the known finite-average bias "
          "of |G_xy| where the two channels are incoherent; over the %d "
          "bins with model coherence above 0.5 it is %.4f with phase %.4f "
          "rad rms" % (hi.sum(), w01[hi].mean(),
                       np.sqrt((phw[hi] ** 2).mean())))
    add("D_welch_vs_exact_wide", n_seeds=len(seeds), duration=args.dur,
        nperseg=nps, n_bins=int(mw.sum()), stay=w00.mean(),
        stay_se=w00.std() / np.sqrt(mw.sum()), stay_sd=w00.std(),
        deck=w11.mean(), cross=w01.mean(),
        cross_coherent=w01[hi].mean(), n_bins_coherent=int(hi.sum()),
        phase_rms=float(np.sqrt((phw ** 2).mean())),
        phase_rms_coherent=float(np.sqrt((phw[hi] ** 2).mean())))
    print("    %d records of %.0f s, nperseg %d (%d bins in 2.6-4.1 Hz):"
          % (len(seeds), args.dur, nps, mb.sum()))
    print("      stay auto  mean ratio %.4f (sd %.4f)"
          % (r00.mean(), r00.std()))
    print("      deck auto  mean ratio %.4f (sd %.4f)"
          % (r11.mean(), r11.std()))
    print("      cross |G|  mean ratio %.4f (sd %.4f); phase error "
          "%.4f rad rms" % (r01.mean(), r01.std(), np.sqrt((ph ** 2).mean())))
    add("D_welch_vs_exact", n_seeds=len(seeds), duration=args.dur,
        nperseg=nps, n_bins=int(mb.sum()), stay=r00.mean(),
        stay_sd=r00.std(), deck=r11.mean(), deck_sd=r11.std(),
        cross=r01.mean(), cross_sd=r01.std(),
        phase_rms=float(np.sqrt((ph ** 2).mean())))

    # -- [E] the multi-channel construction --------------------------------
    Xa, ra = two_channel_record(sim, 120.0, 20.0, 4)
    Xb, rb = extra_channel_record(sim, sim.cd.cable_sensor_dof(8.0),
                                  120.0, 20.0, 4)
    dstay = float(np.max(np.abs(Xa[0] - Xb[0])))
    print("\n[E] stay channel across two calls at seed 4: max difference "
          "%.2e m/s^2" % dstay)
    add("E_channels", stay_max_diff=dstay)

    # -- [F] estimators on exact curves ------------------------------------
    f0, zt = 3.3, 0.005
    fe = np.linspace(3.0, 3.6, 600001)
    om, w0 = 2 * np.pi * fe, 2 * np.pi * f0
    Pl = 1.0 / np.abs(w0 ** 2 - om ** 2 + 2j * zt * w0 * om) ** 2
    hp = half_power(fe, Pl, band=(3.0, 3.6))
    Pa = om ** 4 * Pl
    hpa = half_power(fe, Pa, band=(3.0, 3.6))
    print("\n[F] half power on an exact single mode, f0 = %.3f Hz, "
          "zeta = %.5f" % (f0, zt))
    print("    receptance   : f %.6f Hz, zeta %.6f (%+.3f %%)"
          % (hp["f0"], hp["zeta"], 100 * (hp["zeta"] / zt - 1)))
    print("    accelerance  : f %.6f Hz, zeta %.6f (%+.3f %%)   this is the "
          "quantity an ambient survey measures"
          % (hpa["f0"], hpa["zeta"], 100 * (hpa["zeta"] / zt - 1)))
    add("F_halfpower_exact", f0_true=f0, zeta_true=zt,
        f0_receptance=hp["f0"], zeta_receptance=hp["zeta"],
        zeta_err_receptance_pct=100 * (hp["zeta"] / zt - 1),
        f0_accelerance=hpa["f0"], zeta_accelerance=hpa["zeta"],
        zeta_err_accelerance_pct=100 * (hpa["zeta"] / zt - 1))

    # EFDD on an exact analytic bell of the same single mode
    nps = 32768
    fs = 100.0
    fb = np.fft.rfftfreq(nps, 1.0 / fs)
    om = 2 * np.pi * fb
    Pb = (om ** 4 / np.abs(w0 ** 2 - om ** 2 + 2j * zt * w0 * om) ** 2)
    Sst = Pb[:, None, None].astype(float)
    Ust = np.ones((len(fb), 1, 1))
    mband = (fb > 3.0) & (fb < 3.6)
    idx = np.where(mband)[0]
    ip = int(np.argmax(Pb[idx]))
    e = efdd(fb[idx], Sst[idx, :, 0], Ust[idx], ip, which=0, fs=fs,
             nperseg=nps, mac_min=0.0)
    print("    EFDD on the exact accelerance bell of that mode: f = %.6f Hz "
          "(%+.4f %%), zeta = %.6f (%+.3f %%), %d bins, %d extrema"
          % (e["f_efdd"], 100 * (e["f_efdd"] / f0 - 1), e["zeta_efdd"],
             100 * (e["zeta_efdd"] / zt - 1), e["n_bell"], e["n_extrema"]))
    add("F_efdd_exact", f0_true=f0, zeta_true=zt, f_efdd=e["f_efdd"],
        f_err_pct=100 * (e["f_efdd"] / f0 - 1), zeta_efdd=e["zeta_efdd"],
        zeta_err_pct=100 * (e["zeta_efdd"] / zt - 1), n_bell=e["n_bell"],
        n_extrema=e["n_extrema"], r2=e["r2"], nperseg=nps)

    # -- [G] the estimators' own bias at the sweep's processing settings ---
    print("\n[G] a SINGLE isolated mode at the settings the sweep uses")
    print("    600 s, nperseg 8192, 20 dB, two channels; the answer is")
    print("    imposed, so what is measured here is the estimator, not the")
    print("    structure, and it is the baseline every merged case is read")
    print("    against")
    print("    %-7s %6s %11s %11s %11s %11s"
          % ("zeta", "bins/bw", "f error %", "hp zeta", "hp err %",
             "EFDD zeta"))
    for zt in (0.005, 0.010, 0.020, 0.030):
        f0 = 3.32
        mt = ModalTruth(np.array([f0]), np.array([zt]),
                        np.array([[1.0], [0.25]]), Gamma=np.eye(1))
        fe, ze, hz, ee = [], [], [], []
        for sd in range(args.nseed + 2):
            t, Y = mt.record(duration=600.0, seed=100 + sd, snr_db=20.0)
            fw, G, nseg, keff = cpsd_welch(Y, mt.fs, nperseg=8192)
            Sv, Uv = sv_decomp(G)
            df = mt.fs / 8192
            r = fdd(fw, Sv, Uv, keff, SMOOTH_BINS * df, band=(3.0, 3.7))
            hp = half_power(fw, G[:, 0, 0].real, band=(3.0, 3.7))
            fe.append(r["sv1"]["f_top"])
            hz.append(hp["zeta"])
            ip = int(np.argmin(np.abs(r["f"] - r["sv1"]["f_top"])))
            e = efdd(r["f"], r["S"], r["U"], ip, which=0, fs=mt.fs,
                     nperseg=8192)
            ee.append(e["zeta_efdd"] if e["ok"] else np.nan)
        fe, hz, ee = np.array(fe), np.array(hz), np.array(ee)
        print("    %-7.3f %6.2f %11.4f %11.5f %11.2f %11s"
              % (zt, 2 * zt * f0 / df, 100 * (np.mean(fe) / f0 - 1),
                 np.nanmean(hz), 100 * (np.nanmean(hz) / zt - 1),
                 _fmt(np.nanmean(ee), 5)))
        add("G_single_mode_bias", zeta_true=zt, f_true=f0,
            bins_per_hpbw=2 * zt * f0 / df,
            f_fdd=float(np.mean(fe)), f_err_pct=100 * (np.mean(fe) / f0 - 1),
            f_sd=float(np.std(fe, ddof=1)),
            zeta_hp=float(np.nanmean(hz)),
            zeta_hp_err_pct=100 * (np.nanmean(hz) / zt - 1),
            zeta_hp_sd=float(np.nanstd(hz, ddof=1)),
            zeta_efdd=float(np.nanmean(ee)),
            zeta_efdd_err_pct=100 * (np.nanmean(ee) / zt - 1),
            n_seeds=len(fe), duration=600.0, nperseg=8192, snr_db=20.0)

    print("    the same bell smeared by the Welch kernel, which is what a")
    print("    record's bell actually is, with no Monte Carlo in the answer:")
    for f0s, zts in ((3.30, 0.005), (3.30, 0.020)):
        w0s = 2 * np.pi * f0s
        for nps2 in (32768, 8192):
            dfb = fs / nps2
            fb2 = np.fft.rfftfreq(nps2, 1.0 / fs)
            Kk, dfine = welch_kernel(nps2, fs, half_bins=48)
            half = 0.5 * (len(Kk) - 1) * dfine
            gf = f0s - 1.2 - half + dfine * np.arange(
                int(round(2.4 / dfine)) + len(Kk))
            omg = 2 * np.pi * gf
            Sf = omg ** 4 / np.abs(w0s ** 2 - omg ** 2
                                   + 2j * zts * w0s * omg) ** 2
            Sc = np.convolve(Sf, Kk * dfine, mode="same")
            Sm = np.interp(fb2, gf, Sc)
            mm = (fb2 > f0s - 0.6) & (fb2 < f0s + 0.6)
            ii = np.where(mm)[0]
            e2 = efdd(fb2[ii], Sm[ii][:, None], np.ones((len(ii), 1, 1)),
                      int(np.argmax(Sm[ii])), which=0, fs=fs, nperseg=nps2,
                      mac_min=0.0, smooth_bins=1)
            print("      zeta %.3f, nperseg %5d (%5.2f bins per half-power "
                  "bandwidth): f %.5f Hz (%+.4f %%), zeta %.5f (%+.1f %%)"
                  % (zts, nps2, 2 * zts * f0s / dfb, e2["f_efdd"],
                     100 * (e2["f_efdd"] / f0s - 1), e2["zeta_efdd"],
                     100 * (e2["zeta_efdd"] / zts - 1)))
            add("F_efdd_smeared", nperseg=nps2, f0_true=f0s, zeta_true=zts,
                bins_per_hpbw=2 * zts * f0s / dfb, f_efdd=e2["f_efdd"],
                f_err_pct=100 * (e2["f_efdd"] / f0s - 1),
                zeta_efdd=e2["zeta_efdd"],
                zeta_err_pct=100 * (e2["zeta_efdd"] / zts - 1),
                n_bell=e2["n_bell"])

    # -- [H] the rank argument, exactly ------------------------------------
    print("\n[H] why the second singular value is small: an exact statement")
    sim = RecordSimulator(T_TUNE, 0.005)
    fgx = np.linspace(3.05, 3.60, 4001)
    many = ([sim.cd.cable_sensor_dof(x) for x in (2.0, 6.0, 10.0, 14.0)]
            + [2 * k for k in (10, 20, 30)])
    ev = np.linalg.eigvalsh(sim.Gamma)
    g1 = sim.Gamma.copy()
    w_, V_ = np.linalg.eigh(sim.Gamma)
    rank1 = np.outer(V_[:, -1], V_[:, -1]) * w_[-1]
    for tag, dofset in (("2 sensors", [sim.dof_stay, sim.dof_deck]),
                        ("%d sensors" % len(many), many)):
        sim.Gamma = rank1
        G = cpsd_model(sim, fgx, dofset, "cont")
        S, _ = sv_decomp(G)
        r1 = float(np.max(S[:, 1] / S[:, 0]))
        sim.Gamma = g1
        G = cpsd_model(sim, fgx, dofset, "cont")
        S, _ = sv_decomp(G)
        r2 = float(np.max(S[:, 1] / S[:, 0]))
        print("    %-11s: modal forces forced to rank one -> max sv2/sv1 "
              "%.2e ; the bridge's own forces -> %.2e"
              % (tag, r1, r2))
        add("H_rank", sensors=tag, n_sensors=len(dofset),
            sv2_over_sv1_rank1_forces=r1, sv2_over_sv1_true_forces=r2)
    print("    a perfectly correlated modal force makes the response one")
    print("    complex vector times one scalar, so the cross-spectral matrix")
    print("    is exactly rank one at every line however many sensors are")
    print("    added; the bridge is 0.6 per cent away from that, and no")
    print("    array can recover what the excitation did not put in.")
    print("    the load modelled here is delta-correlated in space, which is")
    print("    the FAVOURABLE case.  A load coherent along the members, a")
    print("    uniform gust for instance, has a rank-one force matrix by")
    print("    construction, and that is not an argument but a number:")
    le_d = sim.bridge["Ld"] / sim.cd.nd
    le_c = sim.bridge["Lc"] / sim.cd.nc
    fvec = np.zeros(sim.cd.N)
    for e in range(sim.cd.nd):
        fvec[[2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]] += np.array(
            [le_d / 2, le_d ** 2 / 12, le_d / 2, -le_d ** 2 / 12])
    for e in range(sim.cd.nc):
        b0 = sim.cd.nD + 2 * e
        fvec[[b0, b0 + 1, b0 + 2, b0 + 3]] += np.array(
            [le_c / 2, le_c ** 2 / 12, le_c / 2, -le_c ** 2 / 12])
    Gcoh = sim.Phi.T @ np.outer(fvec, fvec) @ sim.Phi
    i_, j_ = sim.pair_idx
    ccoh = float(Gcoh[i_, j_] / np.sqrt(Gcoh[i_, i_] * Gcoh[j_, j_]))
    sim.Gamma = Gcoh
    Sc, _ = sv_decomp(cpsd_model(sim, fgx, [sim.dof_stay, sim.dof_deck],
                                 "cont"))
    rcoh = float(np.max(Sc[:, 1] / Sc[:, 0]))
    sim.Gamma = g1
    print("      a spatially uniform coherent load gives pair force "
          "correlation %.6f and max sv2/sv1 %.2e" % (ccoh, rcoh))
    add("H_rank", sensors="2 sensors, coherent uniform load", n_sensors=2,
        gamma_corr=ccoh, sv2_over_sv1_true_forces=rcoh)

    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, "oma_fdd_verify.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("\nwrote %s" % out)
    return rows


# ===========================================================================
# the exact answer FDD converges to, and where the load balance puts it
# ===========================================================================

def exact_study(args):
    """Singular values of the EXACT cross-spectrum, free of estimation.

    Separating what FDD can do in principle from what a 600 s record allows
    is the point of this section: every limit reported here is a property of
    the structure and the load, not of the record.  The pair is also taken
    on its own, with every other mode removed, so that the residual
    flexibility of the deck modes cannot be mistaken for rank the pair does
    not have.
    """
    rows = []
    print("\nEXACT CROSS-SPECTRUM THROUGH THE CROSSING")
    print("=" * 74)
    print("  peaks and heights of the singular value curves of the exact")
    print("  2 by 2 cross-spectrum at the stay and deck sensors, 3.05-3.60 Hz")
    fg = np.linspace(3.05, 3.60, 22001)
    hw = 3.0 * (fg[1] - fg[0])
    for zeta in ZETAS:
        for T in (145.0e3, T_TUNE, 158.0e3):
            sim = RecordSimulator(T, zeta)
            dofs = [sim.dof_stay, sim.dof_deck]
            geo = pair_geometry(sim)
            u = sim.s_split / (2.0 * zeta)
            da = sim.cd.deck_alone(8)
            f_deck = float(da[int(np.argmin(np.abs(da - sim.f0)))])
            row = dict(T=T, zeta=zeta, f_lo=sim.f_lo, f_hi=sim.f_hi,
                       f0=sim.f0, s=sim.s_split, u=u, f_iso=sim.f_iso1,
                       f_deck_alone=f_deck, **geo)
            for tag, mset in (("all", None), ("pair", list(sim.pair_idx))):
                G = cpsd_model(sim, fg, dofs, "cont", modes=mset)
                S, U = sv_decomp(G)
                res = fdd(fg, S, U, 10 ** 9, hw)
                pp = peak_pick(fg, G[:, 0, 0].real, 10 ** 9, hw)
                row.update({
                    "n_pp_%s" % tag: pp["n"], "f_pp_%s" % tag: pp["f_top"],
                    "n_sv1_%s" % tag: res["sv1"]["n"],
                    "n_sv2_%s" % tag: res["sv2"]["n"],
                    "f_sv1_%s" % tag: res["sv1"]["f_top"],
                    "f_sv2_%s" % tag: res["sv2"]["f_top"],
                    "f_sv1_all_%s" % tag: ";".join(
                        "%.5f" % v for v in np.sort(res["sv1"]["f"])),
                    "f_sv2_all_%s" % tag: ";".join(
                        "%.5f" % v for v in np.sort(res["sv2"]["f"])),
                    "sv2_over_sv1_%s" % tag: res["sv2_over_sv1_max"],
                    "sv2_peak_ratio_%s" % tag:
                        res["sv2_peak_over_sv1_peak"]})
            rows.append(row)
            if T == T_TUNE:
                print("  zeta %5.3f u %5.3f | all modes: sv1 %-18s sv2 %-9s "
                      "sv2peak/sv1peak %.2e | pair alone: sv1 %-18s sv2 "
                      "%-9s %.2e"
                      % (zeta, u, row["f_sv1_all_all"], row["f_sv2_all_all"],
                         row["sv2_peak_ratio_all"], row["f_sv1_all_pair"],
                         row["f_sv2_all_pair"], row["sv2_peak_ratio_pair"]))
    d = pd.DataFrame(rows)
    c = d[d["T"] == T_TUNE].iloc[0]
    print("\n  at the crossing, T = %.1f kN: eigenvalues %.5f / %.5f Hz, "
          "split %.3f %%" % (T_TUNE / 1e3, c.f_lo, c.f_hi, 100 * c.s))
    print("  the pair's modal forces have correlation %.4f there and their "
          "2 by 2 force matrix has a smaller eigenvalue %.4f of the larger, "
          "so the excitation is one source and not two"
          % (c.gamma_corr, c.gamma_rank2_ratio))
    print("  the two hybrid mode shapes at the two sensors have MAC %.4f, "
          "so the SENSORS distinguish them; what does not is the excitation"
          % c.mac_sensor)
    print("  across the whole table the force correlation runs %.4f to %.4f "
          "and the sv2 peak stays %.1f to %.1f dB below the sv1 peak"
          % (d.gamma_corr.min(), d.gamma_corr.max(),
             10 * np.log10(d.sv2_peak_ratio_all.max()),
             10 * np.log10(d.sv2_peak_ratio_all.min())))
    rel = 100.0 * (d.f_sv2_all - d.f_deck_alone) / d.f_deck_alone
    print("  and it does not sit on either branch: over tensions %.0f to "
          "%.0f kN the sv2 peak moves only %.4f Hz while the branches move "
          "%.4f Hz, staying within %.2f to %.2f per cent of the DECK mode's "
          "own frequency %.4f Hz, which does not depend on the stay tension "
          "at all"
          % (d["T"].min() / 1e3, d["T"].max() / 1e3,
             d.f_sv2_all.max() - d.f_sv2_all.min(),
             d.f_hi.max() - d.f_lo.min(), rel.min(), rel.max(),
             d.f_deck_alone.iloc[0]))
    print("  a second singular value read as a stay mode therefore returns "
          "a tension that does not move when the tension does")
    d.to_csv(os.path.join(DATA, "oma_fdd_exact.csv"), index=False)
    print("  wrote data/oma_fdd_exact.csv")
    return d


def exact_law_sweep(args):
    """What law does the FDD frequency obey, with the record taken out?

    The sweep on records answers this with noise in it.  Here the exact
    cross-spectrum is built at every tension, its first singular value is
    maximised, and the maximum is compared with the branch value and with
    the merged-peak law of ``run_merged.xstar``.  Because FDD's frequency is
    still the argument of a maximum, the prediction is that it obeys the
    merged-peak law and not the branch law, and this measures whether it
    does.
    """
    rows = []
    print("\nWHICH LAW THE FDD FREQUENCY OBEYS, ON THE EXACT SPECTRUM")
    print("=" * 74)
    fg = np.linspace(3.02, 3.62, 60001)          # 1e-5 Hz
    hw = 3.0 * (fg[1] - fg[0])
    for zeta in ZETAS:
        for T in T_SWEEP:
            sim = RecordSimulator(float(T), zeta)
            dofs = [sim.dof_stay, sim.dof_deck]
            G = cpsd_model(sim, fg, dofs, "cont")
            S, U = sv_decomp(G)
            f_sv1 = float(fg[int(np.argmax(S[:, 0]))])
            f_pp = float(fg[int(np.argmax(G[:, 0, 0].real))])
            f_sv2 = float(fg[int(np.argmax(S[:, 1]))])
            res = fdd(fg, S, U, 10 ** 9, hw)
            d_meas = 2.0 * (sim.f_iso1 - sim.f0) / sim.f0
            u = sim.s_split / (2.0 * zeta)
            rho = rho_2dof(d_meas, sim.s_split)
            f_law = sim.f0 * (1.0 + zeta * xstar(u, rho))
            stay_up = bool(sim.energy_split[sim.pair_idx[1]]
                           >= sim.energy_split[sim.pair_idx[0]])
            f_branch = sim.f_hi if stay_up else sim.f_lo
            row = dict(T=float(T), zeta=zeta, u=u, d=d_meas, rho=rho,
                       f_lo=sim.f_lo, f_hi=sim.f_hi, f0=sim.f0,
                       f_iso=sim.f_iso1, f_sv1=f_sv1, f_pp=f_pp,
                       f_sv2=f_sv2, f_law=f_law, f_branch=f_branch,
                       n_sv1=res["sv1"]["n"], n_sv2=res["sv2"]["n"],
                       sv2_peak_ratio=res["sv2_peak_over_sv1_peak"])
            for tag, fh in (("sv1", f_sv1), ("pp", f_pp), ("sv2", f_sv2),
                            ("law", f_law), ("branch", f_branch)):
                _, ets, ecp = errors(fh, float(T), sim.f_iso1)
                row["eps_cp_%s_pct" % tag] = ecp
                row["eps_ts_%s_pct" % tag] = ets
            rows.append(row)
    d = pd.DataFrame(rows)

    def rms(a, b):
        v = (a - b).values
        v = v[np.isfinite(v)]
        return float(np.sqrt((v ** 2).mean())) if len(v) else np.nan

    print("  root mean square difference over the %d tensions, in "
          "percentage points of tension" % len(T_SWEEP))
    print("  %-7s %7s %14s %14s %14s %14s"
          % ("zeta", "u tune", "sv1 vs law", "sv1 vs branch", "sv1 vs pick",
             "sv2 peak dB"))
    for z in ZETAS:
        g = d[d.zeta == z]
        k = int(np.argmin(np.abs(g["d"].values)))
        print("  %-7.3f %7.2f %14.4f %14.4f %14.5f %14.1f"
              % (z, g.u.iloc[k], rms(g.eps_cp_sv1_pct, g.eps_cp_law_pct),
                 rms(g.eps_cp_sv1_pct, g.eps_cp_branch_pct),
                 rms(g.eps_cp_sv1_pct, g.eps_cp_pp_pct),
                 10 * np.log10(g.sv2_peak_ratio.mean())))
    print("  the first singular value is a maximum of a spectrum, so it "
          "obeys the merged-peak law, not the branch law")
    print("\n  worst tension error over the same tensions, per cent, and "
          "as a multiple of the branch value; this is the danger band with "
          "the record taken out")
    print("  %-7s %10s %10s %10s %10s %10s"
          % ("zeta", "FDD sv1", "pick", "merged law", "branch", "FDD/br"))
    for z in ZETAS:
        g = d[d.zeta == z]
        wb = g.eps_cp_branch_pct.abs().max()
        print("  %-7.3f %10.4f %10.4f %10.4f %10.4f %10.4f"
              % (z, g.eps_cp_sv1_pct.abs().max(),
                 g.eps_cp_pp_pct.abs().max(), g.eps_cp_law_pct.abs().max(),
                 wb, g.eps_cp_sv1_pct.abs().max() / wb))
    print("  FDD takes the same merged-peak relief peak picking does: the "
          "ratio rises to about 1.08 near u = 1.15 and falls below 1 once "
          "the pair has merged, which is what the branch law returning in "
          "full would forbid")
    g2 = d[d.zeta == 0.020]
    print("  and the second singular value carries no tension at all: over "
          "%.0f to %.0f kN at zeta = 2 per cent its peak moves %.4f Hz "
          "while the isolated stay frequency moves %.4f Hz, so the tension "
          "read from it runs from %+.1f to %+.1f per cent"
          % (g2["T"].min() / 1e3, g2["T"].max() / 1e3,
             g2.f_sv2.max() - g2.f_sv2.min(),
             g2.f_iso.max() - g2.f_iso.min(),
             g2.eps_cp_sv2_pct.max(), g2.eps_cp_sv2_pct.min()))
    d.to_csv(os.path.join(DATA, "oma_fdd_exactlaw.csv"), index=False)
    print("  wrote data/oma_fdd_exactlaw.csv")
    return d


def load_balance_study(args):
    """Where in the load balance the second singular value comes alive.

    ``stay_load_ratio`` is the spatial intensity of the ambient load on the
    stay divided by that on the deck.  Equation (3) of the header says the
    modal force correlation passes through zero where the two constituent
    modes take equal generalised force, and that is the only place FDD's
    premise holds.  This locates it, at the exact cross-spectrum so that
    nothing here is an estimation artefact, and reports the pair on its own
    beside the whole model because at very small ratios the deck modes'
    residual flexibility dominates the deck channel and inflates the second
    singular value for a reason that has nothing to do with the pair.
    """
    rows = []
    print("\nLOAD BALANCE AND THE SECOND SINGULAR VALUE")
    print("=" * 74)
    print("  r = stay load intensity / deck load intensity, exact spectrum")
    print("  %-8s %-6s %8s %9s %11s %11s %6s %6s"
          % ("r", "zeta", "corr", "Gam2/Gam1", "sv2pk/sv1pk", "(pair only)",
             "n_sv1", "n_sv2"))
    fg = np.linspace(3.05, 3.60, 22001)
    hw = 3.0 * (fg[1] - fg[0])
    for zeta in (0.005, 0.020):
        for r in (1e-4, 3e-4, 1e-3, 3e-3, 5.5e-3, 1e-2, 3e-2, 1e-1, 1.0,
                  10.0):
            sim = RecordSimulator(T_TUNE, zeta, stay_load_ratio=r)
            dofs = [sim.dof_stay, sim.dof_deck]
            geo = pair_geometry(sim)
            row = dict(stay_load_ratio=r, zeta=zeta, T=T_TUNE,
                       f_lo=sim.f_lo, f_hi=sim.f_hi, **geo)
            for tag, mset in (("all", None), ("pair", list(sim.pair_idx))):
                G = cpsd_model(sim, fg, dofs, "cont", modes=mset)
                S, U = sv_decomp(G)
                res = fdd(fg, S, U, 10 ** 9, hw)
                row.update({
                    "n_sv1_%s" % tag: res["sv1"]["n"],
                    "n_sv2_%s" % tag: res["sv2"]["n"],
                    "f_sv1_all_%s" % tag: ";".join(
                        "%.5f" % v for v in np.sort(res["sv1"]["f"])),
                    "f_sv2_all_%s" % tag: ";".join(
                        "%.5f" % v for v in np.sort(res["sv2"]["f"])),
                    "sv2_peak_ratio_%s" % tag:
                        res["sv2_peak_over_sv1_peak"]})
            rows.append(row)
            print("  %-8.4g %-6.3f %8.4f %9.2e %11.3e %11.3e %6d %6d"
                  % (r, zeta, geo["gamma_corr"], geo["gamma_rank2_ratio"],
                     row["sv2_peak_ratio_all"], row["sv2_peak_ratio_pair"],
                     row["n_sv1_all"], row["n_sv2_all"]))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(DATA, "oma_fdd_loadbalance.csv"), index=False)
    print("  wrote data/oma_fdd_loadbalance.csv")
    return d


# ===========================================================================
# the sweep: four estimators through the crossing, on records
# ===========================================================================

def run_one(sim, T, zeta, duration, snr_db, seed, nperseg, noise_ref,
            band=BAND):
    """All four estimators on one record."""
    X, r = two_channel_record(sim, duration, snr_db, seed,
                              noise_ref=noise_ref)
    f, G, nseg, keff = cpsd_welch(X, sim.fs, nperseg=nperseg)
    df = sim.fs / nperseg
    hw = SMOOTH_BINS * df
    S, U = sv_decomp(G)

    P_stay = G[:, 0, 0].real
    pp = peak_pick(f, P_stay, keff, hw, band=band)
    hp = half_power(f, P_stay, band=band)
    res = fdd(f, S, U, keff, hw, band=band)

    fb, Sb, Ub = res["f"], res["S"], res["U"]
    ef = {}
    for c in (0, 1):
        k = "sv%d" % (c + 1)
        if res[k]["n"] == 0:
            continue
        ip = int(np.argmin(np.abs(fb - res[k]["f"][0])))
        ef[k] = efdd(fb, Sb, Ub, ip, which=c, fs=sim.fs, nperseg=nperseg)

    f_iso = sim.f_iso1
    d_meas = 2.0 * (f_iso - sim.f0) / sim.f0
    u = sim.s_split / (2.0 * zeta)
    stay_up = bool(sim.energy_split[sim.pair_idx[1]]
                   >= sim.energy_split[sim.pair_idx[0]])
    f_branch = sim.f_hi if stay_up else sim.f_lo
    rho = rho_2dof(d_meas, sim.s_split) if abs(sim.s_split) > 0 else np.nan
    f_law = sim.f0 * (1.0 + zeta * xstar(u, rho)) if np.isfinite(rho) \
        else np.nan

    out = dict(T=T, zeta=zeta, seed=seed, duration=duration, snr_db=snr_db,
               nperseg=nperseg, df=df, n_seg=nseg, k_eff=keff,
               noise_ref=noise_ref,
               f_lo=sim.f_lo, f_hi=sim.f_hi, f0=sim.f0, s=sim.s_split,
               u=u, d=d_meas, f_iso=f_iso, rho=rho,
               f_branch=f_branch, f_law=f_law,
               bins_per_hpbw=2.0 * zeta * sim.f0 / df,
               bins_per_split=sim.s_split * sim.f0 / df,
               rms_stay_mg=r["meta"]["rms_stay_mg"],
               rms_deck_mg=r["meta"]["rms_deck_mg"],
               snr_inband_db=r["meta"]["snr_inband_db"])
    out.update(pair_geometry(sim))


    for tag, fh in (("pp", pp["f_top"]), ("hp", hp["f0"]),
                    ("sv1", res["sv1"]["f_top"]), ("sv2", res["sv2"]["f_top"]),
                    ("branch", f_branch), ("law", f_law),
                    ("efdd1", ef.get("sv1", {}).get("f_efdd", np.nan)),
                    ("efdd2", ef.get("sv2", {}).get("f_efdd", np.nan))):
        Th, ets, ecp = errors(fh, T, f_iso)
        out["f_%s" % tag] = fh
        out["T_%s" % tag] = Th
        out["eps_ts_%s_pct" % tag] = ets
        out["eps_cp_%s_pct" % tag] = ecp

    # the same decomposition on channels scaled to unit variance.  Scaling
    # cannot manufacture a second source, but it is the obvious objection to
    # a second singular value that is small because one channel is small, so
    # it is measured rather than argued away.
    Xn = X / X.std(axis=1, keepdims=True)
    _, Gn, _, _ = cpsd_welch(Xn, sim.fs, nperseg=nperseg)
    Sn, Un = sv_decomp(Gn)
    resn = fdd(f, Sn, Un, keff, hw, band=band)

    # the white sensor noise adds its own level to every singular value, so
    # the floor sv2 cannot go below is stated beside sv2 itself
    nrms = r["meta"]["noise_rms"]
    noise_psd = min(nrms) ** 2 / (0.5 * sim.fs) if min(nrms) > 0 else 0.0
    mband = (f >= band[0]) & (f <= band[1])

    out.update(n_pp=pp["n"], n_sv1=res["sv1"]["n"], n_sv2=res["sv2"]["n"],
               sv2_peak_ratio=res["sv2_peak_over_sv1_peak"],
               n_sv1_norm=resn["sv1"]["n"], n_sv2_norm=resn["sv2"]["n"],
               f_sv1_norm=resn["sv1"]["f_top"],
               f_sv2_norm=resn["sv2"]["f_top"],
               sv2_peak_ratio_norm=resn["sv2_peak_over_sv1_peak"],
               noise_psd=noise_psd,
               sv2_peak=float(S[mband, 1].max()),
               sv1_peak=float(S[mband, 0].max()),
               sv2_over_noise_db=(10 * np.log10(S[mband, 1].max()
                                                / noise_psd)
                                  if noise_psd > 0 else np.inf),
               prom_db=pp["prom_db"],
               f_pp_all=";".join("%.5f" % v for v in np.sort(pp["f"])),
               f_sv1_all=";".join("%.5f" % v
                                  for v in np.sort(res["sv1"]["f"])),
               f_sv2_all=";".join("%.5f" % v
                                  for v in np.sort(res["sv2"]["f"])),
               sv2_over_sv1=res["sv2_over_sv1_max"],
               zeta_hp=hp["zeta"], zeta_hp_ratio=hp["zeta"] / zeta,
               zeta_efdd1=ef.get("sv1", {}).get("zeta_efdd", np.nan),
               zeta_efdd2=ef.get("sv2", {}).get("zeta_efdd", np.nan),
               n_bell1=ef.get("sv1", {}).get("n_bell", 0),
               n_bell2=ef.get("sv2", {}).get("n_bell", 0),
               r2_efdd1=ef.get("sv1", {}).get("r2", np.nan))

    # does anything the two singular value curves show land on f_hi?
    cands = (np.concatenate([np.sort(res["sv1"]["f"]),
                             np.sort(res["sv2"]["f"])])
             if (res["sv1"]["n"] + res["sv2"]["n"]) else np.array([]))
    for nm_, ft in (("lo", sim.f_lo), ("hi", sim.f_hi)):
        if len(cands):
            k = int(np.argmin(np.abs(cands - ft)))
            out["best_%s_hz" % nm_] = float(cands[k])
            out["best_%s_err_pct" % nm_] = 100.0 * (cands[k] / ft - 1.0)
        else:
            out["best_%s_hz" % nm_] = np.nan
            out["best_%s_err_pct" % nm_] = np.nan
    return out


def sweep(args):
    rows = []
    seeds = tuple(range(args.nseed))
    print("\nFOUR ESTIMATORS THROUGH THE CROSSING")
    print("=" * 74)
    print("  %d tensions x %d damping ratios x %d seeds, %.0f s records at "
          "%.0f dB" % (len(T_SWEEP), len(ZETAS), len(seeds), args.dur,
                       args.snr))
    for zeta in ZETAS:
        for T in T_SWEEP:
            sim = RecordSimulator(float(T), zeta)
            for sd in seeds:
                rows.append(run_one(sim, float(T), zeta, args.dur, args.snr,
                                    sd, args.nperseg, args.noise_ref))
            last = rows[-1]
            print("  T %6.1f kN  zeta %5.3f  u %5.2f : peaks pp %d sv1 %d "
                  "sv2 %d | f_pp %.4f  f_sv1 %.4f  f_sv2 %s  | true "
                  "%.4f/%.4f" % (T / 1e3, zeta, last["u"], last["n_pp"],
                                 last["n_sv1"], last["n_sv2"], last["f_pp"],
                                 last["f_sv1"], _fmt(last["f_sv2"], 4),
                                 last["f_lo"], last["f_hi"]))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(DATA, "oma_fdd_sweep.csv"), index=False)
    print("  wrote data/oma_fdd_sweep.csv  (%d rows)" % len(d))
    return d


def sweep_report(d):
    print("\n" + "=" * 74)
    print("REPORT")
    print("=" * 74)
    g = d.groupby(["zeta", "T"], as_index=False).mean(numeric_only=True)
    zs = sorted(d.zeta.unique())

    def at_tune(s):
        """Position, not label, of the row nearest exact tuning."""
        return int(np.argmin(np.abs(s["d"].values)))

    print("\n1. how often each curve shows two peaks in %.1f-%.1f Hz, over "
          "all %d rows" % (BAND[0], BAND[1], len(d)))
    print("   %-7s %7s %14s %14s %14s"
          % ("zeta", "u tune", "peak picking", "FDD sv1", "FDD sv1 scaled"))
    for z in zs:
        s = d[d.zeta == z]
        print("   %-7.3f %7.2f %10d /%3d %10d /%3d %10d /%3d"
              % (z, s.u.iloc[at_tune(s)], int((s.n_pp >= 2).sum()),
                 len(s), int((s.n_sv1 >= 2).sum()), len(s),
                 int((s.n_sv1_norm >= 2).sum()), len(s)))
    a = d.n_pp >= 2
    b = d.n_sv1 >= 2
    c = d.n_sv1_norm >= 2
    print("   FDD's first singular value and peak picking disagree on "
          "whether the pair is resolved in %d of the %d rows, %d where FDD "
          "resolves and picking does not and %d the other way; scaling the "
          "channels to equal variance disagrees with picking in %d rows, "
          "%d and %d each way, so it trades resolutions at 0.2 per cent "
          "damping for losses at 0.5 and changes nothing at 1 per cent "
          "and above"
          % (int((a != b).sum()), len(d), int(((~a) & b).sum()),
             int((a & (~b)).sum()), int((a != c).sum()),
             int(((~a) & c).sum()), int((a & (~c)).sum())))
    print("   no processing choice tried here resolves the pair at 1 per "
          "cent damping or above, where u falls below 1.17")

    print("\n2. does FDD return a different frequency from peak picking?")
    ok = d[np.isfinite(d.f_sv1) & np.isfinite(d.f_pp)]
    dd = (100.0 * (ok.f_sv1 / ok.f_pp - 1.0)).abs()
    de = (ok.eps_cp_sv1_pct - ok.eps_cp_pp_pct).abs()
    print("   over %d rows the sv1 peak differs from the picked peak by "
          "%.5f %% of frequency in the median, %.4f %% at the 99th "
          "percentile and %.4f %% at worst"
          % (len(ok), dd.median(), dd.quantile(0.99), dd.max()))
    print("   the two tension errors differ by %.5f percentage points in "
          "the median, %.4f at the 99th percentile and %.4f at worst; the "
          "worst case is the one row in %d where the taller of two resolved "
          "peaks is not the same peak on the two curves"
          % (de.median(), de.quantile(0.99), de.max(), len(ok)))

    print("\n3. when a record actually shows two peaks, against the "
          "closed-form criterion")
    two = d[d.n_pp >= 2]
    one = d[d.n_pp < 2]
    print("   two peaks are found in %d of %d rows; the lowest u at which "
          "any row shows two is %.2f and the highest at which a row shows "
          "one is %.2f"
          % (len(two), len(d), two.u.min() if len(two) else np.nan,
             one.u.max() if len(one) else np.nan))
    print("   the closed forms put the dip at u = 0.486 and a 3 dB dip at "
          "u = 1.140; on a 600 s record with %d effective averages the "
          "prominence floor is %.2f dB, so a 3 dB dip cannot be called and "
          "the working threshold is about twice the closed-form one"
          % (round(d.k_eff.iloc[0]), d.prom_db.iloc[0]))

    print("\n4. the second singular value, FDD's claim to fame")
    print("   %-7s %12s %12s %12s %14s"
          % ("zeta", "sv2pk/sv1pk", "in dB", "over noise", "f_sv2 vs f_hi %"))
    for z in zs:
        s = d[d.zeta == z]
        h = s[np.isfinite(s.f_sv2)]
        print("   %-7.3f %12.2e %12.1f %9.1f dB %14s"
              % (z, s.sv2_peak_ratio.mean(),
                 10 * np.log10(s.sv2_peak_ratio.mean()),
                 s.sv2_over_noise_db.mean(),
                 _fmt(np.nanmean(100 * (h.f_sv2 / h.f_hi - 1)), 2)
                 if len(h) else "n/a"))
    print("   the sv2 peak sits between the branches, not on the upper one,")
    print("   and it never carries a second frequency the sv1 curve lacks")

    print("\n5. tension error at the crossing, per cent of the true tension")
    print("   %-7s %9s %9s %9s %9s %9s %9s"
          % ("zeta", "pick", "FDD sv1", "EFDD", "half pwr", "branch",
             "merged"))
    for z in zs:
        s = d[d.zeta == z]
        k = at_tune(s)
        print("   %-7.3f %9.3f %9.3f %9s %9.3f %9.3f %9.3f"
              % (z, s.eps_cp_pp_pct.iloc[k], s.eps_cp_sv1_pct.iloc[k],
                 _fmt(s.eps_cp_efdd1_pct.iloc[k], 3),
                 s.eps_cp_hp_pct.iloc[k], s.eps_cp_branch_pct.iloc[k],
                 s.eps_cp_law_pct.iloc[k]))

    print("\n6. worst tension error over the tension sweep, per cent, and "
          "as a multiple of the branch value")
    print("   %-7s %9s %9s %9s %9s | %8s %8s"
          % ("zeta", "pick", "FDD sv1", "EFDD", "branch", "pick/br",
             "FDD/br"))
    for z in zs:
        s = g[g.zeta == z]
        wp = s.eps_cp_pp_pct.abs().max()
        wf = s.eps_cp_sv1_pct.abs().max()
        we = s.eps_cp_efdd1_pct.abs().max()
        wb = s.eps_cp_branch_pct.abs().max()
        print("   %-7.3f %9.3f %9.3f %9s %9.3f | %8.3f %8.3f"
              % (z, wp, wf, _fmt(we, 3), wb, wp / wb, wf / wb))

    print("\n7. half-power damping at the crossing, against the SAME "
          "estimator on the same sweep far from it")
    print("   %-7s %7s %9s %9s %7s %9s %9s %7s"
          % ("zeta", "u tune", "hp tune", "hp far", "ratio", "EFDD tune",
             "EFDD far", "ratio"))
    for z in zs:
        s = d[d.zeta == z]
        k = at_tune(s)
        far = s[np.abs(s["d"]) > 0.6 * np.abs(s["d"]).max()]
        rf = far.zeta_hp_ratio.mean()
        ez = s.zeta_efdd1.iloc[k] / z
        ef = np.nanmean(far.zeta_efdd1) / z
        print("   %-7.3f %7.2f %9.3f %9.3f %7.3f %9s %9s %7s"
              % (z, s.u.iloc[k], s.zeta_hp_ratio.iloc[k], rf,
                 s.zeta_hp_ratio.iloc[k] / rf, _fmt(ez, 3), _fmt(ef, 3),
                 _fmt(ez / ef, 3)))
    print("   the far columns are the estimator's own bias at this "
          "resolution, so only the ratios are statements about the crossing;"
          " both damping estimates inflate where the pair cannot be "
          "resolved, which makes the damping the warning sign the frequency "
          "is not")

    print("\n8. which law describes what each estimator returns, as a root "
          "mean square over the whole sweep, in percentage points of "
          "tension")
    print("   %-7s %12s %12s %12s %12s"
          % ("zeta", "pick vs law", "pick vs br", "FDD vs law",
             "FDD vs br"))
    for z in zs:
        s = g[g.zeta == z]

        def rms(a, b):
            v = (a - b).values
            v = v[np.isfinite(v)]
            return float(np.sqrt((v ** 2).mean())) if len(v) else np.nan
        print("   %-7.3f %12.3f %12.3f %12.3f %12.3f"
              % (z, rms(s.eps_cp_pp_pct, s.eps_cp_law_pct),
                 rms(s.eps_cp_pp_pct, s.eps_cp_branch_pct),
                 rms(s.eps_cp_sv1_pct, s.eps_cp_law_pct),
                 rms(s.eps_cp_sv1_pct, s.eps_cp_branch_pct)))

    print("\n10. resolution budget at nperseg %d (%d effective averages)"
          % (d.nperseg.iloc[0], round(d.k_eff.iloc[0])))
    print("   %-7s %16s %16s" % ("zeta", "bins per hpbw", "bins per split"))
    for z in zs:
        s = d[d.zeta == z]
        print("   %-7.3f %16.2f %16.2f"
              % (z, s.bins_per_hpbw.mean(), s.bins_per_split.mean()))


# ===========================================================================
# supplementary: does a second stay sensor help, and a longer record
# ===========================================================================

def extra_sensor_study(args):
    """Two sensors on the stay, against one on the stay and one on the deck.

    The classical prescription for closely spaced modes is more sensors.
    This measures whether that is available here, and the answer is decided
    by the excitation rather than by the array.
    """
    rows = []
    print("\nWHERE THE SECOND SENSOR GOES")
    print("=" * 74)
    for zeta in (0.005, 0.020):
        sim = RecordSimulator(T_TUNE, zeta)
        sets = [("stay + deck anchorage", sim.dof_deck),
                ("stay + stay 8 m up", sim.cd.cable_sensor_dof(8.0)),
                ("stay + stay midchord", sim.cd.cable_sensor_dof(12.5)),
                ("stay + deck quarter span",
                 2 * int(round(0.25 * sim.cd.nd)))]
        for name, dof in sets:
            X, r = extra_channel_record(sim, dof, args.dur, args.snr, 0)
            f, G, nseg, keff = cpsd_welch(X, sim.fs, nperseg=args.nperseg)
            S, U = sv_decomp(G)
            df = sim.fs / args.nperseg
            res = fdd(f, S, U, keff, SMOOTH_BINS * df)
            v1 = sim.Phi[[sim.dof_stay, dof], sim.pair_idx[0]]
            v2 = sim.Phi[[sim.dof_stay, dof], sim.pair_idx[1]]
            Ge = cpsd_model(sim, np.linspace(3.05, 3.6, 8001),
                            [sim.dof_stay, dof], "cont")
            Se, _ = sv_decomp(Ge)
            rows.append(dict(zeta=zeta, sensors=name, dof=dof,
                             mac_sensor=mac(v1, v2),
                             n_sv1=res["sv1"]["n"], n_sv2=res["sv2"]["n"],
                             sv2_over_sv1=res["sv2_over_sv1_max"],
                             sv2_peak_ratio=res["sv2_peak_over_sv1_peak"],
                             sv2_peak_ratio_exact=float(
                                 Se[:, 1].max() / Se[:, 0].max()),
                             sv2_over_sv1_exact=float(
                                 np.max(Se[:, 1] / Se[:, 0])),
                             f_sv1=res["sv1"]["f_top"],
                             f_sv2=res["sv2"]["f_top"],
                             f_lo=sim.f_lo, f_hi=sim.f_hi))
            print("  zeta %5.3f  %-26s MAC %.4f  sv2pk/sv1pk exact %.2e "
                  "estimated %.2e  peaks sv1 %d sv2 %d"
                  % (zeta, name, rows[-1]["mac_sensor"],
                     rows[-1]["sv2_peak_ratio_exact"],
                     rows[-1]["sv2_peak_ratio"], res["sv1"]["n"],
                     res["sv2"]["n"]))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(DATA, "oma_fdd_sensors.csv"), index=False)
    print("  wrote data/oma_fdd_sensors.csv")
    return d


def shape_study(args):
    """The mode shapes the singular vectors return, against the truth.

    A singular vector is defined only up to a complex scale, so the modal
    assurance criterion is the comparison, and each identified shape is
    scored against BOTH hybrid modes rather than against the nearer one, so
    that a shape matching the wrong branch is visible instead of hidden.
    The ceiling is set by the structure: the two true shapes are themselves
    similar at any small sensor set, and how similar is reported beside the
    estimates.
    """
    rows = []
    print("\nTHE MODE SHAPES THE SINGULAR VECTORS RETURN")
    print("=" * 74)
    print("  %-7s %8s %6s %8s %8s %8s %8s %8s"
          % ("zeta", "T kN", "u", "true MAC", "sv1-lo", "sv1-hi", "sv2-lo",
             "sv2-hi"))
    for zeta in ZETAS:
        for T in (145.0e3, 148.75e3, T_TUNE, 155.0e3, 158.0e3):
            sim = RecordSimulator(float(T), zeta)
            Psi = sim.Phi[[sim.dof_stay, sim.dof_deck], :][:, sim.pair_idx]
            for sd in range(args.nseed):
                X, r = two_channel_record(sim, args.dur, args.snr, sd)
                f, G, nseg, keff = cpsd_welch(X, sim.fs,
                                              nperseg=args.nperseg)
                S, U = sv_decomp(G)
                df = sim.fs / args.nperseg
                res = fdd(f, S, U, keff, SMOOTH_BINS * df)
                ip = int(np.argmin(np.abs(res["f"] - res["sv1"]["f_top"]))) \
                    if res["sv1"]["n"] else int(np.argmax(res["S"][:, 0]))
                e = efdd(res["f"], res["S"], res["U"], ip, which=0,
                         fs=sim.fs, nperseg=args.nperseg)
                row = dict(T=float(T), zeta=zeta, seed=sd,
                           u=sim.s_split / (2.0 * zeta),
                           mac_true_pair=mac(Psi[:, 0], Psi[:, 1]),
                           mac_true_pair_full=mac(sim.Phi[:, sim.pair_idx[0]],
                                                  sim.Phi[:, sim.pair_idx[1]]))
                for tag, v in (("sv1", res["sv1"]["shape"]),
                               ("sv2", res["sv2"]["shape"]),
                               ("efdd1", e.get("shape"))):
                    row["mac_%s_lo" % tag] = (mac(v, Psi[:, 0])
                                              if v is not None else np.nan)
                    row["mac_%s_hi" % tag] = (mac(v, Psi[:, 1])
                                              if v is not None else np.nan)
                rows.append(row)
            q = pd.DataFrame(rows[-args.nseed:]).mean(numeric_only=True)
            print("  %-7.3f %8.2f %6.2f %8.4f %8.4f %8.4f %8.4f %8.4f"
                  % (zeta, T / 1e3, q.u, q.mac_true_pair, q.mac_sv1_lo,
                     q.mac_sv1_hi, q.mac_sv2_lo, q.mac_sv2_hi))
    d = pd.DataFrame(rows)
    print("  the two TRUE hybrid shapes have MAC %.4f with each other at "
          "these two sensors and %.4f over all degrees of freedom, so a "
          "shape estimate that matches one perfectly still matches the "
          "other that well; the sensors are not what fails"
          % (d.mac_true_pair.mean(), d.mac_true_pair_full.mean()))
    d.to_csv(os.path.join(DATA, "oma_fdd_shapes.csv"), index=False)
    print("  wrote data/oma_fdd_shapes.csv")
    return d


def long_record_study(args):
    """The same question with the record length and resolution removed."""
    rows = []
    print("\nLONGER RECORDS, FINER RESOLUTION")
    print("=" * 74)
    for zeta in (0.005, 0.020, 0.030):
        sim = RecordSimulator(T_TUNE, zeta)
        for dur, nps in ((600.0, 8192), (3600.0, 32768)):
            for snr in (args.snr, None):
                X, r = two_channel_record(sim, dur, snr, 0,
                                          quantise=(snr is not None))
                f, G, nseg, keff = cpsd_welch(X, sim.fs, nperseg=nps)
                S, U = sv_decomp(G)
                df = sim.fs / nps
                res = fdd(f, S, U, keff, SMOOTH_BINS * df)
                pp = peak_pick(f, G[:, 0, 0].real, keff, SMOOTH_BINS * df)
                rows.append(dict(zeta=zeta, duration=dur, nperseg=nps,
                                 snr_db=(np.nan if snr is None else snr),
                                 n_seg=nseg, df=df,
                                 bins_per_hpbw=2 * zeta * sim.f0 / df,
                                 n_pp=pp["n"], n_sv1=res["sv1"]["n"],
                                 n_sv2=res["sv2"]["n"],
                                 sv2_over_sv1=res["sv2_over_sv1_max"],
                                 sv2_peak_ratio=res[
                                     "sv2_peak_over_sv1_peak"],
                                 f_sv1=res["sv1"]["f_top"],
                                 f_sv2=res["sv2"]["f_top"],
                                 f_lo=sim.f_lo, f_hi=sim.f_hi))
                print("  zeta %5.3f  %6.0f s  nperseg %5d (%3d seg, "
                      "%4.1f bins/hpbw)  snr %4s : peaks pp %d sv1 %d sv2 "
                      "%d  sv2pk/sv1pk %.2e"
                      % (zeta, dur, nps, nseg, rows[-1]["bins_per_hpbw"],
                         "none" if snr is None else "%.0f" % snr, pp["n"],
                         res["sv1"]["n"], res["sv2"]["n"],
                         rows[-1]["sv2_peak_ratio"]))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(DATA, "oma_fdd_long.csv"), index=False)
    print("  wrote data/oma_fdd_long.csv")
    return d


# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--extra", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--report", action="store_true",
                    help="re-read data/oma_fdd_sweep.csv and report only")
    ap.add_argument("--dur", type=float, default=600.0)
    ap.add_argument("--snr", type=float, default=20.0)
    ap.add_argument("--nseed", type=int, default=3)
    ap.add_argument("--nperseg", type=int, default=NPERSEG)
    ap.add_argument("--noise_ref", default="per_channel")
    args = ap.parse_args()

    if args.report:
        sweep_report(pd.read_csv(os.path.join(DATA, "oma_fdd_sweep.csv")))
        return
    if not any((args.verify, args.sweep, args.exact, args.extra, args.all)):
        args.all = True
    if args.verify or args.all:
        verify(args)
    if args.exact or args.all:
        exact_study(args)
        exact_law_sweep(args)
        load_balance_study(args)
    if args.sweep or args.all:
        d = sweep(args)
        sweep_report(d)
    if args.extra or args.all:
        extra_sensor_study(args)
        shape_study(args)
        long_record_study(args)
    print(VERIFICATION_NOTES)


if __name__ == "__main__":
    main()
