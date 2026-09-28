# -*- coding: utf-8 -*-
"""Synthetic acceleration records from the coupled cable-deck model.

Everything the study says about identification through a crossing has so far
been said about an analytic FRF, or about eigenvalues read straight from the
finite element model.  A subspace or decomposition method cannot be run on
either.  It needs a record: a sampled time series, with a stated excitation,
a stated damping, a stated measurement chain and a stated noise floor.  This
module produces one, so that the manuscript's claim that a method which fits
modes "can separate a pair the spectrum shows as one" can be tested rather
than asserted.

Nothing here re-derives the physics.  The mass and stiffness matrices, the
tie constraint, the sensor position and the Rayleigh calibration are taken
from ``src/cablefe.py`` and ``scripts/run_damping.py`` unchanged, and the
frequency response computed here is checked against ``DrivenBridge.frf``
before any record is generated.

WHAT IS SIMULATED
-----------------
The reduced coupled system ``M q'' + C q' + K q = F(t)`` with

  * ``K``, ``M``   from :class:`cablefe.CableDeck` at the requested tension,
    already reduced by the constraint that ties the stay foot to the deck;
  * ``C``          classical, so the real modes diagonalise it exactly.  Two
    models are offered.  ``rayleigh`` (default) is ``alpha M + beta K`` with
    ``alpha``, ``beta`` placing exactly the requested ``zeta`` on both modes
    of the hybrid pair, which is the calibration ``run_damping.py`` uses and
    is reproduced here to the last digit so the two studies describe the same
    structure.  ``uniform`` puts the same ``zeta`` on every mode, which is
    the assumption most operational modal analysis benchmarks make and which
    removes the Rayleigh model's steep rise of damping with frequency.
  * ``F(t)``       one of two excitations, below.

INTEGRATION, AND WHY
--------------------
Modal state space, discretised exactly by zero-order hold, one second-order
IIR section per mode, run with ``scipy.signal.lfilter``.

The damping is classical by construction, so the real modes decouple the
equations exactly and modal superposition is not an approximation but a
change of basis; the only error is modal truncation, measured in check [3]
and worth 4e-6 of the continuous acceleration spectrum in the band of
interest.  Given decoupled modes, the zero-order-hold discretisation of each
2-state modal oscillator is exact for a force held constant across each
sample interval: unlike Newmark it has no period elongation and no
algorithmic damping, which matters when the object of study is a 2.3 per
cent frequency split and a 0.5 per cent damping ratio.  It is also cheap,
because 50 second-order sections cost less per step than one dense solve on
160 degrees of freedom.  The price of the modal form is that it is available
only because the damping is classical; a non-classically damped model would
need the complex modes of ``(K + i w C - w^2 M)`` or a direct integrator,
and neither is used here.

The force really is held constant across each internal sample interval,
which is a modelling choice and not an approximation of something else: the
excitation process IS piecewise constant, and the response is its exact
response.  Everything the hold and the sampling do to the spectrum, the
``sinc^2(f dt)`` of the hold and every alias the sampling folds in, is
already contained in the discrete transfer function of the sections that
are run, so the analytic spectrum below is written from those sections and
is exact.  An earlier version folded the continuous spectrum instead and
was 0.7 per cent wrong in the record variance; the Parseval check found it.

EXCITATION
----------
``ambient``  Gaussian white noise applied as a distributed transverse load
    on both members, uncorrelated in space as well as in time.  This is the
    standard operational modal analysis assumption, and the spatial part of
    it matters: a load fully correlated along the span cannot excite a mode
    whose shape integrates to zero, so a spatially coherent idealisation
    would silently suppress modes that a real ambient field excites.  For a
    load delta-correlated in space the consistent nodal force covariance is

        Sigma = S_d (M_deck / m_deck) + S_c (M_stay / m_stay)

    because the consistent load covariance of an element is ``S`` times
    ``\\int N^T N dx``, and for a uniform chain that integral is exactly the
    consistent mass matrix divided by the mass per unit length.  That
    identity is verified numerically against Gauss-Legendre quadrature of the
    Hermite shape functions rather than assumed.  The default puts the same
    spatial intensity on the deck and on the stay, which is the neutral
    assumption; ``stay_load_ratio`` moves it.

``pluck``  a half-sine transverse force pulse at one point on the stay,
    the operation a stay test actually performs.  The pulse duration sets
    the excitation bandwidth (a half sine of duration ``Tp`` is flat to
    about ``0.5/Tp`` and first vanishes at ``1.5/Tp``); the default 0.05 s
    is a hammer or a sharp release rather than a slow hand pull.  The two
    excitations pose different identification problems, which is why both
    are here: the ambient record is stationary and output-only, the pluck is
    a transient with a deterministic input, and a method that reads one well
    need not read the other.

MEASUREMENT CHAIN
-----------------
The chain is simulated in the order a real one runs.

  1. The modal equations are integrated at an internal rate
     ``oversample * fs``.  Oversampling is not cosmetic: an accelerometer on
     a stay driven by broadband load sees a response whose acceleration
     spectrum does not roll off, so sampling the continuous response
     directly at 100 Hz folds everything above 50 Hz back into the band.
     Measured on the worked bridge, sampling straight at 100 Hz moves the
     spectrum by up to 312 per cent at the antiresonance notch between the
     two hybrid peaks, which is the one feature of the spectrum this study
     cannot afford to corrupt, and by 84 per cent on average across the
     band.  At 8x oversampling those fall to 1.4 and 0.4 per cent, or
     0.06 dB at worst.
  2. An anti-alias FIR low pass at ``0.4 fs``, linear phase, Kaiser window,
     applied with ``fftconvolve`` in ``same`` mode so the group delay is
     removed exactly, then decimation to ``fs``.  This is what a sigma-delta
     digitiser does, and using a linear-phase FIR rather than an IIR keeps
     the chain from introducing poles that a subspace method would identify
     as structural modes.
  3. Additive Gaussian sensor noise at a stated signal-to-noise ratio.  The
     noise is added at the recorded rate, so it is white across the whole
     recorded band; a real chain would band-limit it with the same
     anti-alias filter, which changes only the top tenth of the band and
     nothing in the 2.5-4.2 Hz band of interest.
  4. Uniform quantisation over a stated full-scale range.  At the default
     24 bits over +-2 g, and a stay responding at 5 mg RMS, the
     quantisation noise is 97 dB below the signal.  It is immaterial, and
     so, on this chain, is 16-bit quantisation at 49 dB down: both sit far
     under any sensor noise worth modelling, and the sober conclusion is
     that word length is not what limits a stay-cable ambient survey.  The
     step is kept in the chain and its number reported because
     "immaterial" should be a measurement and not an opinion.

DEFAULTS, AND THE REASONING
---------------------------
``fs = 100 Hz``  the anti-alias corner at 40 Hz leaves the first twelve
    harmonics of a 3.3 Hz stay in the record, so the harmonic-comb screening
    the study relies on has something to screen; 100 to 256 Hz is the
    ordinary setting of a 24-bit ambient logger.
``duration = 600 s``  ten minutes is the usual length of an ambient stay
    record.  It is 2000 cycles of the fundamental, which is what a subspace
    method needs for its covariances to settle, and its raw frequency
    resolution of 1/600 Hz is a 46th of the 2.3 per cent split on the worked
    bridge, so the record length is not what limits resolution.
``snr_db = 20``  the broadband ratio of signal RMS to noise RMS over the
    whole recorded band.  A good force-balance accelerometer on a stay
    responding at a few milli-g sits nearer 50 dB and a cheap MEMS unit
    nearer 15 dB; 20 dB is deliberately at the pessimistic end so that a
    method which survives it survives a field record.  Broadband is not
    in-band, and on this model the gap is large: a spatially white load
    excites the stay's higher harmonics as hard as its first, so only 0.26
    per cent of the recorded power falls in 2.5 to 4.2 Hz and a broadband
    20 dB is an in-band 8.8 dB, confirmed at 8.7 dB by measuring the noise
    on a 600 s record.  Both numbers are put in every record's
    metadata as ``snr_db`` and ``snr_inband_db`` so that downstream work
    states which one it means.  A real ambient field rolls off with
    frequency and would put more of its power in the band, so the in-band
    figure quoted here is pessimistic; that is a limitation of the white
    idealisation, not a property of stays.
``target RMS = 5 mg``  a stay under ordinary ambient wind and traffic.  The
    excitation intensity is scaled to hit this RMS analytically, not
    empirically, so that the scale factor is deterministic and the analytic
    spectrum is comparable with the record's without a random offset.
``mode cutoff = 3 fs/2``  every mode below 150 Hz, 50 of the 160.  Keeping
    the other 110 would change the continuous acceleration spectrum in the
    band by 4e-6 and the recorded spectrum at the two hybrid peaks by 5.4e-5,
    but the recorded spectrum at the antiresonance notch by 3.8e-2, because
    whatever modes are retained above the internal Nyquist alias into the
    band and the notch is where the true spectrum is smallest.  The cutoff
    is therefore a choice with a size, and the ground for making it is that
    a 40-element chain does not represent its own modes above about the
    twentieth: they are discretisation artefacts, and carrying artefacts
    into the record is worse than dropping them.  Note that the discarded
    modes are NOT negligible in the receptance, where they carry a residual
    flexibility worth 8.5e-4 of the band peak; the difference is the factor
    ``omega^4`` between the two, and it is the acceleration that is
    recorded.

WHAT A RECORD CARRIES
---------------------
:func:`simulate_records` returns ``t``, ``a_stay`` and ``a_deck`` in m/s^2
with the sensor noise and the quantiser applied, ``a_stay_clean`` and
``a_deck_clean`` without them, ``fs``, and a ``meta`` dict.  The metadata is
the interface every other arm of this study reads, so it carries the truth
as well as the settings: ``f_modes`` and ``zeta_modes`` for every retained
mode, ``energy_split`` to say which are stay modes and which deck modes,
``f_lo``, ``f_hi``, ``f0`` and ``s`` for the hybrid pair, ``f_iso1`` for the
isolated stay fundamental the incumbent formula would invert, and
``snr_db`` beside ``snr_inband_db`` so that a result is never quoted against
the wrong one.  The deck channel is 41 dB below the stay channel under equal
load intensity, and by default each channel's noise is scaled to its own
signal; ``noise_ref="absolute"`` gives both channels the same noise floor
instead, which is what one instrument model on two cables would do and which
leaves the deck channel far noisier.

VERIFICATION
------------
``python3 scripts/simulate_records.py --verify`` runs eight checks, listed
in ``VERIFICATION_NOTES`` below, and writes
``data/simulate_records_verify.csv``.

The one that matters most is the fourth: the Welch spectrum of long
noise-free records against the analytic spectrum of the same model.
"Analytic" there means the exact discrete spectrum of the zero-order-hold
sections, multiplied by the anti-alias filter, folded by the decimation and
finally convolved with the Welch window kernel.  The last step is not
fussiness.  A Welch estimate of a resonance three bins wide is a smoothed
thing, and comparing it with an unsmoothed prediction is comparing two
different quantities: at ``nperseg = 4096`` the unsmoothed comparison is
2.6 per cent out on average and 96 per cent out in the worst bin, while the
smoothed one agrees to a few parts in ten thousand.

That check is reported over two bands, and the two say different things.
Over 0.4 to 39 Hz, which holds some fifteen resonances and 1581 bins, eight
hour-long records give a mean ratio of 1.0003 with a nominal standard error
of 0.0006.  Over 2.6 to 4.1 Hz alone, 61 bins, the same records give 0.991
with a nominal error of 0.003, and a different set of six records gives
0.999.  The nominal errors are optimistic, because neighbouring Welch bins
share a window main lobe and are not independent, so the in-band figure is
consistent with unity at roughly the one per cent precision that a 1.5 Hz
band affords, and one per cent is the honest claim for it.  The wideband
figure is the strong one, and it covers resonances of exactly the same kind.

Run:  python3 scripts/simulate_records.py --verify
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.linalg import eigh, expm
from scipy.signal import (butter, cont2discrete, fftconvolve, filtfilt,
                          find_peaks, firwin, freqz, get_window, hilbert,
                          lfilter, ss2tf, welch)

try:                                     # optional, and worth having
    from threadpoolctl import threadpool_limits
except ImportError:                      # pragma: no cover
    from contextlib import contextmanager

    @contextmanager
    def threadpool_limits(*a, **k):
        yield

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from cablefe import CableDeck, chain, tensioned_beam_freq  # noqa: E402
from run_damping import BRIDGE, FBAND, T_TUNE  # noqa: E402

DATA = os.path.join(ROOT, "data")
G0 = 9.80665                     # m/s^2 per g, for reporting in milli-g

FS_DEFAULT = 100.0               # Hz
DUR_DEFAULT = 600.0              # s
SNR_DEFAULT = 20.0               # dB, broadband
OVERSAMPLE = 8
MODE_CUTOFF_FACTOR = 3.0         # keep modes below this times the Nyquist
AA_CUTOFF_FRAC = 0.4             # anti-alias corner as a fraction of fs
AA_TAPS_PER_R = 40               # FIR length = AA_TAPS_PER_R * oversample + 1
AA_KAISER_BETA = 8.6             # about -80 dB stopband
TARGET_RMS_MG = 5.0              # ambient stay acceleration, milli-g RMS
TARGET_PEAK_MG = 200.0           # pluck peak acceleration, milli-g
NBITS = 24
FULL_SCALE_G = 2.0
BURN_TAU = 8.0                   # burn-in, in slowest modal decay times
CHUNK = 200000                   # internal samples per integration chunk


# ---------------------------------------------------------------------------
# the model, its damping, and the excitation covariance
# ---------------------------------------------------------------------------

class RecordSimulator:
    """The worked bridge, ready to emit records at one tension and damping.

    The eigensolution is computed once in the constructor, so a Monte Carlo
    over seeds or noise levels at fixed tension costs one solve.
    """

    def __init__(self, T, zeta, bridge=None, damping_model="rayleigh",
                 fs=FS_DEFAULT, oversample=OVERSAMPLE,
                 mode_cutoff_factor=MODE_CUTOFF_FACTOR,
                 sensor_from_anchor=2.0, deck_station=None,
                 stay_load_ratio=1.0, aa_cutoff_frac=AA_CUTOFF_FRAC):
        self.bridge = dict(BRIDGE if bridge is None else bridge)
        self.T = float(T)
        self.zeta = float(zeta)
        self.damping_model = damping_model
        self.fs = float(fs)
        self.oversample = int(oversample)
        self.fs_int = self.fs * self.oversample
        self.dt_int = 1.0 / self.fs_int
        self.stay_load_ratio = float(stay_load_ratio)

        self.cd = CableDeck(T=self.T, **self.bridge)
        w2, V = eigh(self.cd.K, self.cd.M)
        w_all = np.sqrt(np.maximum(w2, 0.0))
        Phi_all = self.cd.Lmat @ V           # mass-normalised, full DOF
        self.n_modes_full = len(w_all)
        self.f_all = w_all / (2.0 * np.pi)

        # -- damping ------------------------------------------------------
        self.alpha, self.beta, self.f_cal = self._calibrate(w_all)
        if damping_model == "rayleigh":
            z_all = 0.5 * (self.alpha / w_all + self.beta * w_all)
        elif damping_model == "uniform":
            z_all = np.full_like(w_all, self.zeta)
        else:
            raise ValueError("damping_model must be rayleigh or uniform")

        # -- modal truncation ---------------------------------------------
        self.mode_cutoff = mode_cutoff_factor * 0.5 * self.fs
        keep = self.f_all < self.mode_cutoff
        self.idx = np.where(keep)[0]
        self.w = w_all[self.idx]
        self.zj = z_all[self.idx]
        self.Phi = Phi_all[:, self.idx]
        self.f = self.w / (2.0 * np.pi)
        self.w_all, self.z_all, self.Phi_all = w_all, z_all, Phi_all

        # -- sensors ------------------------------------------------------
        self.dof_stay = self.cd.cable_sensor_dof(sensor_from_anchor)
        if deck_station is None:
            self.dof_deck = 2 * self.cd.ia          # the anchorage node
            self.x_deck = self.cd.x_anchor
        else:
            i = int(round(deck_station / self.bridge["Ld"] * self.cd.nd))
            i = max(0, min(self.cd.nd, i))
            self.dof_deck = 2 * i
            self.x_deck = i * self.bridge["Ld"] / self.cd.nd
        self.dofs = np.array([self.dof_stay, self.dof_deck])

        # -- excitation covariance ----------------------------------------
        Md_full, Mc_full = self.cd._mass_blocks()
        self.Sigma = (Md_full / self.bridge["md"]
                      + self.stay_load_ratio * Mc_full / self.bridge["mc"])
        self.Gamma = self.Phi.T @ self.Sigma @ self.Phi
        self.Gchol = np.linalg.cholesky(
            self.Gamma + 1e-14 * np.trace(self.Gamma) / len(self.Gamma)
            * np.eye(len(self.Gamma)))

        # -- discretisation and anti-alias filter --------------------------
        self._discretise()
        ntaps = AA_TAPS_PER_R * self.oversample + 1
        self.aa_taps = firwin(ntaps, aa_cutoff_frac * self.fs,
                              window=("kaiser", AA_KAISER_BETA),
                              fs=self.fs_int)
        self.aa_cutoff = aa_cutoff_frac * self.fs

        # -- pair diagnostics, carried into every record's metadata --------
        self.f_iso1 = float(tensioned_beam_freq(
            1, self.bridge["Lc"], self.T, self.bridge["EIc"],
            self.bridge["mc"]))
        near = np.sort(np.argsort(np.abs(self.f - self.f_iso1))[:2])
        self.f_lo, self.f_hi = float(self.f[near[0]]), float(self.f[near[1]])
        self.f0 = 0.5 * (self.f_lo + self.f_hi)
        self.s_split = (self.f_hi - self.f_lo) / self.f0
        self.pair_idx = near
        self.energy_split = self.cd.energy_split(self.Phi)

        # slowest modal decay sets the burn-in
        self.burn_in = BURN_TAU / np.min(self.zj * self.w)

    # -- setup helpers ----------------------------------------------------

    def _calibrate(self, w_all):
        """Rayleigh coefficients placing zeta on both modes of the pair.

        The pair is found as the two modes nearest the isolated stay
        fundamental, which on the worked bridge is the same choice as
        ``run_damping.DrivenBridge`` makes by taking the two modes inside
        FBAND, and is checked to be so in the verification.
        """
        f_iso = tensioned_beam_freq(1, self.bridge["Lc"], self.T,
                                    self.bridge["EIc"], self.bridge["mc"])
        fa = w_all / (2.0 * np.pi)
        near = np.sort(np.argsort(np.abs(fa - f_iso))[:2])
        wa, wb = w_all[near[0]], w_all[near[1]]
        alpha = 2.0 * self.zeta * wa * wb / (wa + wb)
        beta = 2.0 * self.zeta / (wa + wb)
        return alpha, beta, (float(fa[near[0]]), float(fa[near[1]]))

    def _discretise(self):
        """One exact zero-order-hold second-order section per mode.

        State ``[q, q']``, input the modal force, output the modal
        acceleration ``q'' = f - 2 zeta w q' - w^2 q``, so the direct
        feedthrough is exactly one.  ``Ad`` and ``Bd`` come from the
        exponential of the augmented matrix, which gives the input integral
        without a separate quadrature, and are checked against
        ``scipy.signal.cont2discrete`` in the verification.
        """
        nm = len(self.w)
        self.b_iir = np.zeros((nm, 3))
        self.a_iir = np.zeros((nm, 3))
        self.Ad = np.zeros((nm, 2, 2))
        self.Bd = np.zeros((nm, 2, 1))
        for j in range(nm):
            w, z = self.w[j], self.zj[j]
            A = np.array([[0.0, 1.0], [-w * w, -2.0 * z * w]])
            B = np.array([[0.0], [1.0]])
            Cm = np.array([[-w * w, -2.0 * z * w]])
            aug = np.zeros((3, 3))
            aug[:2, :2] = A * self.dt_int
            aug[:2, 2:] = B * self.dt_int
            E = expm(aug)
            Ad, Bd = E[:2, :2], E[:2, 2:]
            self.Ad[j], self.Bd[j] = Ad, Bd
            num, den = ss2tf(Ad, Bd, Cm, np.array([[1.0]]))
            self.b_iir[j] = num[0]
            self.a_iir[j] = den

    # -- frequency response and spectra -----------------------------------

    def receptance(self, fgrid, dof=None, truncated=True):
        """Driving-point receptance, the sum run_damping.frf also uses."""
        dof = self.dof_stay if dof is None else dof
        w, z, Phi = ((self.w, self.zj, self.Phi) if truncated
                     else (self.w_all, self.z_all, self.Phi_all))
        om = 2.0 * np.pi * np.asarray(fgrid, dtype=float)
        den = (w ** 2)[None, :] - (om ** 2)[:, None] \
            + 2j * (z * w)[None, :] * om[:, None]
        return (Phi[dof, :] ** 2 / den).sum(axis=1)

    def _quad(self, fgrid, dof, mode="cont"):
        """``h(f)^H Gamma h(f)`` for the response row at ``dof``.

        ``mode="cont"`` uses the continuous-time modal receptance, giving
        the response of the structure to an ideal white force.
        ``mode="disc"`` uses the zero-order-hold discrete transfer function
        of the very sections ``lfilter`` runs, giving the EXACT spectrum of
        the sampled sequence: the hold, and every alias the sampling folds
        in, are already inside it, so nothing has to be folded or corrected
        afterwards.  The two differ by 0.7 per cent in variance on the
        default chain, and that difference is not noise; it was found as an
        unexplained 0.65 per cent excess in the Parseval check, and the
        folded-continuous form was the thing that was wrong.

        Evaluated as ``sum |h L|^2`` with ``Gamma = L L^T``, which is the
        same number as the double modal sum over every pair of modes and
        costs one matrix product instead of ``n^2`` inner products.  The
        real and imaginary parts are multiplied separately because a
        complex-by-real product in numpy does not reach BLAS and runs three
        orders of magnitude slower than the two real products that replace
        it.  The thread limit is not superstition either: on this machine a
        201 by 50 product costs 106 ms with MKL threading and 0.08 ms
        without, because the matrices are far too small to repay the
        synchronisation, and the whole verification spends its time here.
        """
        f = np.asarray(fgrid, dtype=float)
        out = np.empty(f.shape, dtype=float)
        step = max(1, int(4e6 // max(len(self.w), 1)))
        ph = self.Phi[dof, :][None, :]
        with threadpool_limits(1):
            for a in range(0, f.size, step):
                b = min(a + step, f.size)
                if mode == "cont":
                    om = 2.0 * np.pi * f[a:b]
                    den = (self.w ** 2)[None, :] - (om ** 2)[:, None] \
                        + 2j * (self.zj * self.w)[None, :] * om[:, None]
                    h = ph / den
                else:
                    e = np.exp(-2j * np.pi * f[a:b] * self.dt_int)[:, None]
                    num = (self.b_iir[:, 0][None, :]
                           + self.b_iir[:, 1][None, :] * e
                           + self.b_iir[:, 2][None, :] * e ** 2)
                    den = (self.a_iir[:, 0][None, :]
                           + self.a_iir[:, 1][None, :] * e
                           + self.a_iir[:, 2][None, :] * e ** 2)
                    h = ph * num / den
                wr = np.ascontiguousarray(h.real) @ self.Gchol
                wi = np.ascontiguousarray(h.imag) @ self.Gchol
                out[a:b] = np.einsum('ij,ij->i', wr, wr) \
                    + np.einsum('ij,ij->i', wi, wi)
        return out

    def psd_ideal(self, fgrid, dof=None):
        """One-sided acceleration PSD of the continuous system, unit intensity.

        What the structure would show under a genuinely white force, with no
        hold, no sampling and no filter.  It is the reference the sampled
        chain is measured against, not the thing the chain produces.
        """
        dof = self.dof_stay if dof is None else dof
        om = 2.0 * np.pi * np.asarray(fgrid, dtype=float)
        return 2.0 * om ** 4 * self._quad(fgrid, dof, "cont")

    def psd_internal(self, fgrid, dof=None):
        """One-sided PSD of the internally sampled sequence, unit intensity.

        Exact, for ``|f| <= fs_int/2``.  The modal forces are white with
        covariance ``Gamma / dt``, and a discrete white input of covariance
        ``Q`` through a discrete system of response ``h`` gives a PSD of
        ``dt h^H Q h`` in units per Hz, so the ``dt`` cancels and the answer
        is ``h^H Gamma h`` with ``h`` the discrete response.
        """
        dof = self.dof_stay if dof is None else dof
        return 2.0 * self._quad(fgrid, dof, "disc")

    def psd_recorded(self, fgrid, dof=None, aa=True, decim=True):
        """One-sided PSD of the RECORDED signal, unit intensity.

        The internal spectrum, multiplied by the anti-alias filter and
        folded by the decimation.  Decimation folding is exact folding of a
        discrete spectrum, so this expression is exact and not a truncated
        series; the earlier version of this method summed aliases of the
        CONTINUOUS spectrum instead, converged only as ``1/n_alias``, and
        was 0.7 per cent wrong in the variance however many terms it took.
        """
        dof = self.dof_stay if dof is None else dof
        f = np.asarray(fgrid, dtype=float)
        R = self.oversample if decim else 1
        fs_out = self.fs if decim else self.fs_int
        nyq = self.fs_int / 2.0 + 1e-9

        gs = []
        for m in range(-R, R + 1):
            for sgn in (1.0, -1.0):
                g = np.abs(sgn * f + m * fs_out)
                if np.all(g > nyq):
                    continue
                if any(np.allclose(g, gp, atol=1e-12) for gp in gs):
                    continue
                gs.append(g)

        # one pass over every frequency the answer needs: the quadratic form
        # is a BLAS product whose per-call overhead dwarfs its arithmetic
        allg = np.concatenate(gs)
        S = self.psd_internal(np.minimum(allg, self.fs_int / 2.0),
                              dof).reshape(len(gs), f.size)
        if aa:
            _, H = freqz(self.aa_taps, worN=allg, fs=self.fs_int)
            S = S * (np.abs(H) ** 2).reshape(len(gs), f.size)
        out = np.zeros_like(f)
        for g, si in zip(gs, S):
            m_in = g <= nyq
            out[m_in] += si[m_in]
        return out

    def _fine_grid(self, lo, hi, npts, span=200.0, ncl=4001):
        """A grid that resolves every resonance between ``lo`` and ``hi``.

        A uniform grid over 0 to 400 Hz cannot resolve a resonance 0.03 Hz
        wide, and integrating one that cannot would under-count the variance
        by an order of magnitude.
        """
        g = [np.linspace(max(lo, 1e-6), hi, npts)]
        for fj, zj in zip(self.f, self.zj):
            hw = max(zj * fj, 1e-4)
            if lo - span * hw < fj < hi + span * hw:
                g.append(fj + hw * np.linspace(-span, span, ncl))
        return np.unique(np.clip(np.concatenate(g), max(lo, 1e-6), hi))

    def variance_recorded(self, dof=None, npts=100000):
        """Analytic variance of the recorded signal, unit intensity.

        Subsampling does not change the variance of a stationary process, so
        the recorded variance is the variance of the anti-alias-filtered
        internal sequence, and the decimation fold never has to be written
        down: it is one integral over the internal band.
        """
        dof = self.dof_stay if dof is None else dof
        key = (dof, npts)
        if getattr(self, "_varcache", None) is None:
            self._varcache = {}
        if key in self._varcache:
            return self._varcache[key]
        gg = self._fine_grid(0.0, self.fs_int / 2.0, npts)
        _, H = freqz(self.aa_taps, worN=gg, fs=self.fs_int)
        v = float(np.trapz(self.psd_internal(gg, dof) * np.abs(H) ** 2, gg))
        self._varcache[key] = v
        return v

    def band_fraction(self, band=FBAND, dof=None, npts=6000):
        """Share of the recorded power that falls in ``band``.

        Reported because a broadband signal-to-noise ratio is not an in-band
        one, and on a stay under white load the two differ by 10 dB.
        """
        dof = self.dof_stay if dof is None else dof
        key = (dof, tuple(band), npts)
        if getattr(self, "_bandcache", None) is None:
            self._bandcache = {}
        if key in self._bandcache:
            return self._bandcache[key]
        gg = self._fine_grid(band[0], band[1], npts)
        pb = float(np.trapz(self.psd_recorded(gg, dof), gg))
        fr = pb / self.variance_recorded(dof)
        self._bandcache[key] = fr
        return fr

    def snr_inband_db(self, snr_db, band=FBAND, dof=None):
        """In-band signal-to-noise ratio implied by a broadband one.

        The sensor noise is white over the recorded band, so its share of
        ``band`` is the bandwidth ratio, and the conversion needs no
        absolute level:

            SNR_band = SNR_broad + 10 log10( P_band/P_total * (fs/2)/B ).
        """
        if snr_db is None:
            return np.inf
        fr = self.band_fraction(band, dof)
        return snr_db + 10.0 * np.log10(
            fr * (0.5 * self.fs) / (band[1] - band[0]))

    # -- integration -------------------------------------------------------

    def _integrate(self, nint, modal_force):
        """Run the modal sections over ``nint`` internal samples.

        ``modal_force(a, b)`` returns the (n_modes, b-a) block of modal
        forces.  Blocking keeps memory bounded on hour-long records; the
        filter states are carried across blocks so the result is identical
        to filtering the whole record at once.
        """
        nm = len(self.w)
        zi = [np.zeros(2) for _ in range(nm)]
        Cs = self.Phi[self.dofs, :]
        out = np.zeros((len(self.dofs), nint))
        for a in range(0, nint, CHUNK):
            b = min(a + CHUNK, nint)
            F = modal_force(a, b)
            Y = np.empty_like(F)
            for j in range(nm):
                Y[j], zi[j] = lfilter(self.b_iir[j], self.a_iir[j], F[j],
                                      zi=zi[j])
            out[:, a:b] = Cs @ Y
        return out

    def _ambient_force(self, rng, nint):
        """Modal forces of a spatially and temporally white distributed load.

        The nodal forces have covariance ``Sigma / dt`` so that the force
        spectrum is independent of the internal rate; the modal forces are
        their projection, generated directly from the Cholesky factor of
        ``Gamma = Phi^T Sigma Phi`` rather than through the 164 nodal
        components, which is the same distribution at a fraction of the cost.
        """
        scale = 1.0 / np.sqrt(self.dt_int)

        def f(a, b):
            z = rng.standard_normal((len(self.w), b - a))
            return scale * (self.Gchol @ z)
        return f

    def _pluck_force(self, nint, pluck_from_anchor, t_pulse, dur_pulse):
        """Modal forces of a half-sine pulse at one point on the stay."""
        dof = self.cd.cable_sensor_dof(pluck_from_anchor)
        phi = self.Phi[dof, :][:, None]
        k0 = int(round(t_pulse / self.dt_int))
        k1 = int(round((t_pulse + dur_pulse) / self.dt_int))

        def f(a, b):
            k = np.arange(a, b)
            p = np.zeros(b - a)
            m = (k >= k0) & (k < k1)
            p[m] = np.sin(np.pi * (k[m] - k0) / max(k1 - k0, 1))
            return phi * p[None, :]
        return f, dof

    # -- the public entry point -------------------------------------------

    def record(self, duration=DUR_DEFAULT, snr_db=SNR_DEFAULT, seed=0,
               excitation="ambient", target_rms_mg=TARGET_RMS_MG,
               target_peak_mg=TARGET_PEAK_MG, nbits=NBITS,
               full_scale_g=FULL_SCALE_G, noise_ref="per_channel",
               pluck_from_anchor=None, t_pulse=1.0, dur_pulse=0.05,
               ambient_frac=0.0, quantise=True):
        """One record.  Returns a dict of arrays and metadata.

        ``snr_db`` is broadband: the ratio of signal RMS to noise RMS over
        the whole recorded band, not over the band of interest.  The in-band
        ratio is reported alongside it because for a stay under white load
        most of the signal power sits above 10 Hz, so the two differ.
        ``snr_db=None`` adds no sensor noise at all, which is what the
        verification uses and what no field record ever is.
        """
        rng = np.random.default_rng(seed)
        nkeep = int(round(duration * self.fs))
        pad = len(self.aa_taps) // 2 + self.oversample
        burn = 0.0 if excitation == "pluck" else self.burn_in
        nburn = int(round(burn * self.fs_int))
        nint = nburn + nkeep * self.oversample + 2 * pad

        def chain(y):
            """Anti-alias, trim the burn-in and the filter edges, decimate."""
            yf = np.vstack([fftconvolve(row, self.aa_taps, mode="same")
                            for row in y])
            yf = yf[:, nburn + pad:nburn + pad + nkeep * self.oversample]
            return yf[:, ::self.oversample]

        # amplitudes are set on the RECORDED signal, after the anti-alias
        # filter and the decimation, so that the target means what it says:
        # the filter removes a tenth of a sharp pluck's peak, and scaling
        # before it would leave the record short of the figure asked for
        if excitation == "ambient":
            a_clean = chain(self._integrate(nint,
                                            self._ambient_force(rng, nint)))
            sc = np.sqrt((target_rms_mg * 1e-3 * G0) ** 2
                         / self.variance_recorded(self.dof_stay))
            a_clean = a_clean * sc
            intensity = sc ** 2
            dof_pluck = None
        elif excitation in ("pluck", "pluck+ambient"):
            pfa = (0.33 * self.bridge["Lc"] if pluck_from_anchor is None
                   else pluck_from_anchor)
            ff, dof_pluck = self._pluck_force(
                nint, pfa, t_pulse + burn + pad * self.dt_int, dur_pulse)
            a_clean = chain(self._integrate(nint, ff))
            a_clean = a_clean * (target_peak_mg * 1e-3 * G0
                                 / np.max(np.abs(a_clean[0])))
            intensity = np.nan
            if excitation == "pluck+ambient":
                ab = chain(self._integrate(nint,
                                           self._ambient_force(rng, nint)))
                sa = np.sqrt((ambient_frac * target_peak_mg * 1e-3 * G0) ** 2
                             / self.variance_recorded(self.dof_stay))
                a_clean = a_clean + sa * ab
        else:
            raise ValueError("excitation must be ambient, pluck or "
                             "pluck+ambient")

        # sensor noise
        rms = a_clean.std(axis=1)
        if snr_db is None:
            nsig = np.zeros(len(rms))
        elif noise_ref == "per_channel":
            nsig = rms / (10.0 ** (snr_db / 20.0))
        elif noise_ref == "absolute":
            nsig = np.full(len(rms), rms[0] / (10.0 ** (snr_db / 20.0)))
        else:
            raise ValueError("noise_ref must be per_channel or absolute")
        noise = nsig[:, None] * rng.standard_normal(a_clean.shape)
        a = a_clean + noise

        # quantisation
        lsb = 2.0 * full_scale_g * G0 / 2 ** nbits
        n_clip = 0
        if quantise:
            fsv = full_scale_g * G0
            n_clip = int(np.sum(np.abs(a) > fsv))
            a = np.clip(np.round(a / lsb) * lsb, -fsv, fsv)

        t = np.arange(nkeep) / self.fs
        meta = dict(
            T=self.T, zeta=self.zeta, damping_model=self.damping_model,
            fs=self.fs, duration=duration, oversample=self.oversample,
            snr_db=snr_db, seed=seed, excitation=excitation,
            nbits=nbits, full_scale_g=full_scale_g, lsb=lsb, n_clip=n_clip,
            burn_in_s=burn, aa_cutoff_hz=self.aa_cutoff,
            n_modes=len(self.w), mode_cutoff_hz=self.mode_cutoff,
            dof_stay=self.dof_stay, dof_deck=self.dof_deck,
            x_deck=self.x_deck, dof_pluck=dof_pluck,
            intensity=intensity,
            rms_stay=float(rms[0]), rms_deck=float(rms[1]),
            rms_stay_mg=float(rms[0] / G0 * 1e3),
            rms_deck_mg=float(rms[1] / G0 * 1e3),
            noise_rms=[float(v) for v in nsig],
            band=FBAND,
            band_power_frac=(self.band_fraction(FBAND)
                             if excitation == "ambient" else np.nan),
            snr_inband_db=(self.snr_inband_db(snr_db)
                           if excitation == "ambient" else np.nan),
            quant_rms=lsb / np.sqrt(12.0),
            f_modes=self.f.copy(), zeta_modes=self.zj.copy(),
            energy_split=self.energy_split.copy(),
            f_lo=self.f_lo, f_hi=self.f_hi, f0=self.f0, s=self.s_split,
            f_iso1=self.f_iso1, alpha=self.alpha, beta=self.beta)
        return dict(t=t, a_stay=a[0], a_deck=a[1], a_stay_clean=a_clean[0],
                    a_deck_clean=a_clean[1], fs=self.fs, meta=meta)


def simulate_records(T, zeta, duration=DUR_DEFAULT, fs=FS_DEFAULT,
                     snr_db=SNR_DEFAULT, excitation="ambient", seed=0,
                     bridge=None, damping_model="rayleigh", **kw):
    """Acceleration records at the stay sensor and a deck station.

    The five arguments the rest of the study needs are the first five:
    tension, damping ratio, record length, sampling rate and noise level.
    Everything else has a default that :mod:`simulate_records` documents and
    verifies.

    Returns a dict with ``t``, ``a_stay``, ``a_deck`` (all m/s^2, the first
    two with sensor noise and quantisation applied), ``a_stay_clean`` and
    ``a_deck_clean`` (before the noise), ``fs``, and ``meta``.
    """
    ctor = {k: kw.pop(k) for k in
            ("oversample", "mode_cutoff_factor", "sensor_from_anchor",
             "deck_station", "stay_load_ratio", "aa_cutoff_frac")
            if k in kw}
    sim = RecordSimulator(T, zeta, bridge=bridge,
                          damping_model=damping_model, fs=fs, **ctor)
    return sim.record(duration=duration, snr_db=snr_db, seed=seed,
                      excitation=excitation, **kw)


# ---------------------------------------------------------------------------
# what a Welch estimate of a sharp resonance actually estimates
# ---------------------------------------------------------------------------

def expected_welch(sim, fbins, nperseg, window="hann", dof=None, os_fine=16,
                   half_bins=96):
    """Expected Welch spectrum: the true PSD convolved with the window kernel.

    A Welch periodogram does not estimate ``S(f)``; it estimates ``S``
    smeared by ``|W(f)|^2 / \\int |W|^2``, with ``W`` the window transform.
    On a resonance a few bins wide the difference is tens of per cent, so
    comparing a Welch estimate with an unsmoothed analytic spectrum measures
    the window, not the simulator.
    """
    dof = sim.dof_stay if dof is None else dof
    fbins = np.asarray(fbins, dtype=float)
    df = sim.fs / nperseg
    dfine = df / os_fine
    win = get_window(window, nperseg)
    npad = nperseg * os_fine
    W = np.fft.fft(win, npad)
    K = np.abs(W) ** 2
    K = np.fft.fftshift(K)
    gk = (np.arange(npad) - npad // 2) * (sim.fs / npad)
    m = np.abs(gk) <= half_bins * df
    K, gk = K[m], gk[m]
    K = K / (K.sum() * dfine)                     # unit area on the fine grid

    lo = fbins.min() - half_bins * df - dfine
    hi = fbins.max() + half_bins * df + dfine
    n = int(np.ceil((hi - lo) / dfine)) + 1
    gf = lo + dfine * np.arange(n)
    Sf = sim.psd_recorded(np.abs(gf), dof)
    conv = np.convolve(Sf, K * dfine, mode="same")
    return np.interp(fbins, gf, conv)


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def _shape_integral(le):
    """``\\int N^T N dx`` for one Hermite beam element, by quadrature."""
    xg, wg = np.polynomial.legendre.leggauss(8)
    x = 0.5 * le * (xg + 1.0)
    s = x / le
    N = np.vstack([1 - 3 * s ** 2 + 2 * s ** 3,
                   le * (s - 2 * s ** 2 + s ** 3),
                   3 * s ** 2 - 2 * s ** 3,
                   le * (-s ** 2 + s ** 3)])
    return (N * (wg * 0.5 * le)) @ N.T


def check_sigma(sim):
    """Nodal load covariance equals the mass matrix over mass per length."""
    out = {}
    for name, L, nel, mpl in (("deck", sim.bridge["Ld"], sim.cd.nd,
                               sim.bridge["md"]),
                              ("stay", sim.bridge["Lc"], sim.cd.nc,
                               sim.bridge["mc"])):
        ndof = 2 * (nel + 1)
        A = np.zeros((ndof, ndof))
        ae = _shape_integral(L / nel)
        for e in range(nel):
            i = np.array([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3])
            A[np.ix_(i, i)] += ae
        _, M = chain(L, nel, 1.0, mpl, 0.0)
        out[name] = float(np.max(np.abs(A - M / mpl))
                          / np.max(np.abs(M / mpl)))
    return out


def check_zoh(sim):
    """The expm discretisation against scipy's own."""
    worst_a = worst_b = 0.0
    for j in range(len(sim.w)):
        w, z = sim.w[j], sim.zj[j]
        A = np.array([[0.0, 1.0], [-w * w, -2.0 * z * w]])
        B = np.array([[0.0], [1.0]])
        Ad, Bd, _, _, _ = cont2discrete(
            (A, B, np.array([[1.0, 0.0]]), np.array([[0.0]])), sim.dt_int,
            method="zoh")
        sa = max(np.max(np.abs(Ad)), 1e-30)
        sb = max(np.max(np.abs(Bd)), 1e-30)
        worst_a = max(worst_a, np.max(np.abs(Ad - sim.Ad[j])) / sa)
        worst_b = max(worst_b, np.max(np.abs(Bd - sim.Bd[j])) / sb)
    return worst_a, worst_b


def check_frf(sim, fgrid):
    """The simulator's receptance against run_damping.DrivenBridge."""
    from run_damping import DrivenBridge
    br = DrivenBridge(sim.T)
    Href = br.frf(sim.zeta, fgrid)
    Hall = sim.receptance(fgrid, truncated=False)
    Htru = sim.receptance(fgrid, truncated=True)
    sc = np.abs(Href).max()
    return (float(np.max(np.abs(Hall - Href)) / sc),
            float(np.max(np.abs(Htru - Href)) / sc),
            br, Href)


def _decay_zeta(a, fs, f0, half_band=0.2, lo_frac=0.03, hi_frac=0.7):
    """Damping from a free decay: band pass, Hilbert envelope, log fit.

    The fit window runs from the first sample after the peak at which the
    envelope has fallen to ``hi_frac`` of it, which skips the pulse itself
    and the filter's own rise, to the first at which it reaches
    ``lo_frac``, which stops before the noise floor bends the log envelope.
    The window is one contiguous run, not the union of every sample in the
    amplitude range, so a beat that dips into the range late in the record
    cannot rejoin the fit.
    """
    b, aa = butter(4, [(f0 - half_band) / (fs / 2), (f0 + half_band)
                       / (fs / 2)], btype="band")
    x = filtfilt(b, aa, a)
    env = np.abs(hilbert(x))
    k0 = int(np.argmax(env))
    env = env[k0:]
    pk = env[0]
    below = np.where(env < hi_frac * pk)[0]
    if len(below) == 0:
        return np.nan, np.nan, 0
    i0 = int(below[0])
    under = np.where(env[i0:] < lo_frac * pk)[0]
    i1 = i0 + int(under[0]) if len(under) else len(env)
    if i1 - i0 < 50:
        return np.nan, np.nan, 0
    tt = np.arange(i0, i1) / fs
    yy = np.log(env[i0:i1])
    coef, res = np.polyfit(tt, yy, 1, full=True)[:2]
    ss = np.sum((yy - yy.mean()) ** 2)
    r2 = 1.0 - (res[0] / ss if len(res) and ss > 0 else np.nan)
    return -coef[0] / (2.0 * np.pi * f0), float(r2), int(i1 - i0)


VERIFICATION_NOTES = """
[1] the excitation covariance identity, against Gauss-Legendre quadrature
[2] the zero-order-hold discretisation, against scipy.signal.cont2discrete
[3] the receptance, against run_damping.DrivenBridge, and the cost of the
    modal truncation in both the receptance and the acceleration spectrum
[4] the Welch spectrum of long noise-free records, against the analytic
    spectrum of the same model convolved with the Welch window kernel, plus
    a Parseval check of the record variance
[5] peak-picked frequencies, against the analytic spectrum picked the same
    way, against the driving-point receptance and against the eigenvalues,
    and the width of the picked pair against the merged-peak law
[6] the damping identified from a pluck decay, against the modal damping
    that was put in
[7] the aliasing, quantisation and signal-to-noise budget of the chain
[8] reproducibility, and the pluck amplitude
"""


def refine_peaks(f, P, n_seg, half_width_hz, n_sigma=4.0, n_fit=None):
    """Peaks of a noisy spectrum, located to better than a bin.

    A Welch spectrum of a random process is chi-squared about its mean, so
    raw local maxima are meaningless: at 43 segments every third bin is a
    local maximum.  The spectrum is therefore smoothed over one half-power
    half-width before peaks are looked for, the prominence floor is set at
    ``n_sigma`` times the chi-squared scatter expressed in dB, and the
    surviving peaks are refined by a parabola through the smoothed decibel
    curve.  Applying exactly the same procedure to the analytic spectrum
    makes the comparison like for like, so whatever bias the smoothing
    carries cancels out of it.
    """
    f = np.asarray(f, dtype=float)
    db = 10.0 * np.log10(np.asarray(P, dtype=float))
    dfb = f[1] - f[0]
    nsm = max(1, int(round(half_width_hz / dfb)))
    sm = np.convolve(db, np.ones(nsm) / nsm, mode="same")
    prom = n_sigma * (10.0 / np.log(10.0)) / np.sqrt(max(n_seg, 1))
    pk, _ = find_peaks(sm, prominence=prom)
    nf = max(2, nsm) if n_fit is None else n_fit
    out = []
    for k in pk:
        a, b = max(0, k - nf), min(len(f), k + nf + 1)
        if b - a < 3:
            continue
        c = np.polyfit(f[a:b] - f[k], sm[a:b], 2)
        if c[0] < 0:
            out.append(f[k] - c[1] / (2.0 * c[0]))
        else:
            out.append(f[k])
    return np.array(out), prom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--dur", type=float, default=3600.0,
                    help="record length for the spectral check, s")
    ap.add_argument("--nseed", type=int, default=4,
                    help="independent records for the spectral check")
    ap.add_argument("--demo", action="store_true",
                    help="write one 600 s record to data/ as .npz")
    args = ap.parse_args()

    import pandas as pd
    rows = []

    def add(check, **kw):
        rows.append(dict(check=check, **kw))

    zeta = 0.005
    print("RESPONSE SIMULATOR VERIFICATION")
    print("worked bridge, T = %.1f kN, zeta = %.1f %%, fs = %.0f Hz, "
          "oversample %d" % (T_TUNE / 1e3, 100 * zeta, FS_DEFAULT, OVERSAMPLE))

    sim = RecordSimulator(T_TUNE, zeta)
    print("\n[model] %d modes below %.0f Hz of %d; pair %.4f / %.4f Hz, "
          "s = %.3f %%" % (len(sim.w), sim.mode_cutoff, sim.n_modes_full,
                           sim.f_lo, sim.f_hi, 100 * sim.s_split))
    print("        burn-in %.0f s; stay sensor DOF %d, deck station x = "
          "%.1f m" % (sim.burn_in, sim.dof_stay, sim.x_deck))
    add("model", n_modes=len(sim.w), n_modes_full=sim.n_modes_full,
        f_lo=sim.f_lo, f_hi=sim.f_hi, s=sim.s_split, burn_in_s=sim.burn_in,
        alpha=sim.alpha, beta=sim.beta)

    # -- 1 excitation covariance ------------------------------------------
    c = check_sigma(sim)
    print("\n[1] load covariance Sigma = M / m (Gauss-Legendre check):")
    print("    deck %.2e   stay %.2e   relative" % (c["deck"], c["stay"]))
    add("sigma_identity", deck_rel=c["deck"], stay_rel=c["stay"])

    # -- 2 discretisation --------------------------------------------------
    wa, wb = check_zoh(sim)
    print("\n[2] zero-order hold vs scipy.signal.cont2discrete, %d modes:"
          % len(sim.w))
    print("    Ad %.2e   Bd %.2e   relative" % (wa, wb))
    add("zoh_vs_scipy", Ad_rel=wa, Bd_rel=wb, n_modes=len(sim.w))

    # -- 3 FRF against run_damping ----------------------------------------
    fg = np.linspace(FBAND[0], FBAND[1], 17001)
    e_all, e_tru, br, Href = check_frf(sim, fg)
    print("\n[3] receptance at the stay sensor vs run_damping.DrivenBridge:")
    print("    all %d modes      %.2e of band peak"
          % (sim.n_modes_full, e_all))
    print("    truncated to %d   %.2e of band peak" % (len(sim.w), e_tru))
    sim_full = RecordSimulator(T_TUNE, zeta, mode_cutoff_factor=60.0)
    fbt = np.linspace(FBAND[0], FBAND[1], 341)
    t_cont = float(np.max(np.abs(sim.psd_ideal(fbt)
                                 / sim_full.psd_ideal(fbt) - 1.0)))
    t_rec = float(np.max(np.abs(sim.psd_recorded(fbt)
                                / sim_full.psd_recorded(fbt) - 1.0)))
    fpk2 = np.array([sim.f_lo, sim.f_hi])
    t_pk = float(np.max(np.abs(sim.psd_recorded(fpk2)
                               / sim_full.psd_recorded(fpk2) - 1.0)))
    print("    keeping all %d modes instead of %d moves the CONTINUOUS "
          "acceleration spectrum by %.1e, and the RECORDED one by %.1e at "
          "worst but only %.1e at the two peaks"
          % (sim_full.n_modes_full, len(sim.w), t_cont, t_rec, t_pk))
    print("    the worst case is the antiresonance notch, where the true "
          "spectrum is smallest and the aliased floor of modes above the "
          "internal Nyquist shows: a %d element chain does not represent "
          "those modes, which is why they are dropped rather than kept"
          % sim.cd.nc)
    add("frf_vs_run_damping", all_modes_rel=e_all, truncated_rel=e_tru,
        n_modes_full=sim_full.n_modes_full, psd_trunc_continuous=t_cont,
        psd_trunc_recorded=t_rec, psd_trunc_at_peaks=t_pk)

    # -- 4 spectrum of long records ----------------------------------------
    print("\n[4] Welch spectrum of %d noise-free records of %.0f s vs "
          "analytic" % (args.nseed, args.dur))
    seeds = tuple(1 + 10 * i for i in range(args.nseed))
    recs = [sim.record(duration=args.dur, snr_db=None, seed=sd,
                       quantise=False) for sd in seeds]
    xs = [r["a_stay_clean"] for r in recs]
    rec = recs[0]
    print("    record RMS %s mg (target %.1f mg); deck %.4f mg, "
          "%.1f dB below"
          % (np.round([r["meta"]["rms_stay_mg"] for r in recs], 3),
             TARGET_RMS_MG, rec["meta"]["rms_deck_mg"],
             20 * np.log10(rec["meta"]["rms_stay"]
                           / rec["meta"]["rms_deck"])))
    var_an = sim.variance_recorded(sim.dof_stay) * rec["meta"]["intensity"]
    var_c = sim.variance_recorded(sim.dof_stay, npts=200000) \
        * rec["meta"]["intensity"]
    print("    analytic variance integral is converged to %.1e "
          "(grid doubled)" % abs(var_c / var_an - 1.0))
    vr = np.array([x.var() for x in xs]) / var_an
    print("    [Parseval] variance / analytic = %s, mean %.4f, "
          "sd %.4f" % (np.round(vr, 4), vr.mean(), vr.std(ddof=1)))
    add("parseval", var_analytic=float(var_an), ratio=float(vr.mean()),
        ratio_sd=float(vr.std(ddof=1)), n_seeds=len(vr), duration=args.dur)

    welch_cache = {}
    for nps in (4096, 16384):
        Ps = []
        for x in xs:
            f, P = welch(x, fs=sim.fs, nperseg=nps, window="hann",
                         noverlap=nps // 2)
            Ps.append(P)
        P = np.mean(Ps, axis=0)
        welch_cache[nps] = (f, Ps)
        navg = int(2 * len(xs[0]) / nps - 1) * len(xs)
        # Welch (1967): half-overlapped Hann segments are not independent,
        # and the variance of the average falls as 9/(11 K), not 1/K
        pred = np.sqrt(9.0 / (11.0 * navg))
        mw = (f > 0.4) & (f < 0.39 * sim.fs)
        Ew_w = expected_welch(sim, f[mw], nps) * rec["meta"]["intensity"]
        rw = P[mw] / Ew_w
        m = (f >= 2.6) & (f <= 4.1)
        fb, Pb = f[m], P[m]
        Ew = expected_welch(sim, fb, nps) * rec["meta"]["intensity"]
        Eu = sim.psd_recorded(fb) * rec["meta"]["intensity"]
        r = Pb / Ew
        print("    nperseg %5d (df %.4f Hz, %4d segments): per-bin scatter "
              "%.4f, Welch predicts %.4f"
              % (nps, sim.fs / nps, navg, rw.std(), pred))
        print("      0.4-%.0f Hz  %4d bins: mean ratio %.4f +- %.4f"
              % (0.39 * sim.fs, mw.sum(), rw.mean(),
                 pred / np.sqrt(mw.sum())))
        print("      2.6-4.1 Hz  %4d bins: mean ratio %.4f +- %.4f"
              % (m.sum(), r.mean(), pred / np.sqrt(m.sum())))
        print("      against the UNSMOOTHED spectrum, 2.6-4.1 Hz: mean "
              "ratio %.4f, worst bin %.2f -- the window kernel is not "
              "optional at this resolution"
              % ((Pb / Eu).mean(), np.max(np.abs(Pb / Eu - 1))))
        add("psd_vs_analytic", nperseg=nps, df=sim.fs / nps, n_seg=navg,
            mean_ratio_wide=float(rw.mean()),
            se_wide=float(pred / np.sqrt(mw.sum())), n_bins_wide=int(mw.sum()),
            mean_ratio=float(r.mean()),
            mean_ratio_se=float(pred / np.sqrt(m.sum())),
            rms_scatter=float(rw.std()), welch_predicted=pred,
            n_bins=int(m.sum()), duration=args.dur, n_seeds=len(xs),
            mean_ratio_unsmoothed=float((Pb / Eu).mean()),
            worst_bin_unsmoothed=float(np.max(np.abs(Pb / Eu - 1))))

    # -- 5 peak frequencies -------------------------------------------------
    print("\n[5] peak-picked frequency, noise-free, against the model")
    nps = 16384
    hw = zeta * sim.f0
    ffine = np.linspace(3.10, 3.55, 45001)
    San = sim.psd_recorded(ffine)
    pa, _ = find_peaks(San, prominence=San.max() * 1e-4)
    f_true_pk = ffine[pa]
    Hf = np.abs(br.frf(zeta, ffine))
    pf, _ = find_peaks(Hf)
    f_frf = ffine[pf]
    f, Ps = welch_cache[nps]
    m = (f >= 3.10) & (f <= 3.55)
    nseg = int(2 * len(xs[0]) / nps - 1)
    got = []
    for sd, P in zip(seeds, Ps):
        fp, prom = refine_peaks(f[m], P[m], nseg, hw)
        got.append(fp)
        print("    seed %2d: %s Hz" % (sd, np.round(fp, 5)))
    Ew = expected_welch(sim, f[m], nps)
    fa_w, _ = refine_peaks(f[m], Ew, 10 ** 9, hw)
    print("    analytic spectrum, same picking procedure : %s Hz"
          % np.round(fa_w, 5))
    print("    analytic spectrum, unsmoothed             : %s Hz"
          % np.round(f_true_pk, 5))
    print("    driving-point receptance peaks            : %s Hz"
          % np.round(f_frf, 5))
    print("    eigenvalues                               : %.5f / %.5f Hz"
          % (sim.f_lo, sim.f_hi))
    print("    prominence floor %.2f dB, bin %.5f Hz" % (prom, sim.fs / nps))
    n_ok = [g for g in got if len(g) == len(fa_w)]
    if len(n_ok) == len(got) and len(fa_w) == 2:
        A = np.array(n_ok)
        d = A.mean(axis=0) - fa_w
        print("    record vs analytic, mean over %d seeds: %+.5f / %+.5f Hz "
              "(%.3f / %.3f bins, %.4f / %.4f %% of f)"
              % (len(A), d[0], d[1], d[0] / (sim.fs / nps),
                 d[1] / (sim.fs / nps), 100 * d[0] / fa_w[0],
                 100 * d[1] / fa_w[1]))
        print("    seed-to-seed scatter: %.5f / %.5f Hz"
              % (A[:, 0].std(ddof=1), A[:, 1].std(ddof=1)))
        add("peak_pick", n_peaks=2, bin_hz=sim.fs / nps, nperseg=nps,
            duration=args.dur, n_seeds=len(A), prominence_db=prom,
            f1_record=A[:, 0].mean(), f2_record=A[:, 1].mean(),
            f1_analytic=fa_w[0], f2_analytic=fa_w[1],
            f1_analytic_raw=f_true_pk[0] if len(f_true_pk) > 1 else np.nan,
            f2_analytic_raw=f_true_pk[1] if len(f_true_pk) > 1 else np.nan,
            f1_frf=f_frf[0] if len(f_frf) > 1 else np.nan,
            f2_frf=f_frf[1] if len(f_frf) > 1 else np.nan,
            f_lo=sim.f_lo, f_hi=sim.f_hi, err1_hz=d[0], err2_hz=d[1],
            sd1_hz=A[:, 0].std(ddof=1), sd2_hz=A[:, 1].std(ddof=1))
    else:
        print("    peak counts differ between seeds: %s"
              % [len(g) for g in got])
        add("peak_pick", n_peaks=len(got[0]), bin_hz=sim.fs / nps)

    # the picked peaks are not the eigenvalues, and the amount by which
    # they are not is an established result of this study, so it is a check
    if len(f_true_pk) > 1 and len(f_frf) > 1:
        u = sim.s_split / (2.0 * zeta)
        over = np.sqrt(u * np.sqrt(u ** 2 + 4.0) - 1.0) / u
        gap_e = sim.f_hi - sim.f_lo
        gap_p = f_true_pk[1] - f_true_pk[0]
        gap_f = f_frf[1] - f_frf[0]
        print("    the picked pair is WIDER than the eigenvalue pair, as "
              "the merged-peak law requires:")
        print("      u = s/2zeta = %.4f, law predicts |x*|/u = %.5f"
              % (u, over))
        print("      eigenvalue gap %.5f Hz; ambient spectrum %.5f Hz "
              "(ratio %.5f); receptance %.5f Hz (ratio %.5f)"
              % (gap_e, gap_p, gap_p / gap_e, gap_f, gap_f / gap_e))
        print("      law vs simulated ambient spectrum: %.3f %% apart"
              % (100 * (gap_p / gap_e / over - 1)))
        add("merged_peak_law", u=u, overshoot_law=over,
            gap_eigen_hz=gap_e, gap_psd_hz=gap_p, gap_frf_hz=gap_f,
            ratio_psd=gap_p / gap_e, ratio_frf=gap_f / gap_e,
            law_vs_psd_pct=100 * (gap_p / gap_e / over - 1),
            law_vs_frf_pct=100 * (gap_f / gap_e / over - 1))
    # -- 6 damping from a decay --------------------------------------------
    print("\n[6] damping identified from a pluck decay, detuned tension")
    T_det = 120.0e3
    for model in ("rayleigh", "uniform"):
        s2 = RecordSimulator(T_det, zeta, damping_model=model)
        j = int(np.argmin(np.abs(s2.f - s2.f_iso1)))
        z_true, f_true = s2.zj[j], s2.f[j]
        for snr in (None, 40.0, 20.0):
            r = s2.record(duration=120.0, snr_db=snr, seed=3,
                          excitation="pluck", quantise=(snr is not None))
            zh, r2, npts = _decay_zeta(r["a_stay"], s2.fs, f_true)
            tag = "clean" if snr is None else "%.0f dB" % snr
            print("    %-8s %-7s mode %.4f Hz: zeta_true %.5f, "
                  "identified %.5f (%+.2f %%), R2 %.5f"
                  % (model, tag, f_true, z_true, zh,
                     100 * (zh / z_true - 1), r2))
            add("decay_damping", damping_model=model, T=T_det,
                snr_db=(np.nan if snr is None else snr),
                f_mode=f_true, zeta_true=z_true, zeta_identified=zh,
                err_pct=100 * (zh / z_true - 1), r2=r2, n_points=npts)

    # -- 7 aliasing, noise and quantisation budget --------------------------
    print("\n[7] chain budget at the default settings")
    fb2 = np.linspace(FBAND[0], FBAND[1], 341)
    ratio = np.abs(sim.psd_recorded(fb2) / sim.psd_ideal(fb2) - 1.0)
    sim1 = RecordSimulator(T_TUNE, zeta, oversample=1)
    r1 = np.abs(sim1.psd_recorded(fb2) / sim1.psd_ideal(fb2) - 1.0)
    print("    departure of the recorded spectrum from the ideal "
          "continuous one, %.1f-%.1f Hz:\n      oversample %d -> max %.2e "
          "mean %.2e"
          % (FBAND[0], FBAND[1], sim.oversample, ratio.max(),
             ratio.mean()))
    print("      oversample 1 -> max %.2e mean %.2e   (hold and aliasing "
          "together)" % (r1.max(), r1.mean()))
    add("alias_budget", oversample=sim.oversample, max_rel=float(ratio.max()),
        mean_rel=float(ratio.mean()), max_rel_os1=float(r1.max()),
        mean_rel_os1=float(r1.mean()))

    r = sim.record(duration=60.0, snr_db=SNR_DEFAULT, seed=5)
    md = r["meta"]
    for nb in (24, 16):
        lsb = 2.0 * FULL_SCALE_G * G0 / 2 ** nb
        q = lsb / np.sqrt(12.0)
        print("    %2d bit over +-%.0f g: LSB %.3e m/s^2, quantisation RMS "
              "%.2e = %.1f dB below signal"
              % (nb, FULL_SCALE_G, lsb, q,
                 20 * np.log10(md["rms_stay"] / q)))
        add("quantisation", nbits=nb, lsb=lsb, quant_rms=q,
            db_below_signal=20 * np.log10(md["rms_stay"] / q),
            signal_rms=md["rms_stay"])
    print("    sensor noise at %.0f dB: %.2e m/s^2 RMS; clipped samples %d"
          % (SNR_DEFAULT, md["noise_rms"][0], md["n_clip"]))

    # in-band versus broadband signal-to-noise, and a direct measurement
    # of the conversion rather than a trust in the formula
    fr = sim.band_fraction(FBAND)
    snr_b = sim.snr_inband_db(SNR_DEFAULT)
    rlong = sim.record(duration=600.0, snr_db=SNR_DEFAULT, seed=9,
                       quantise=False)
    # the noise sequence itself, not the difference of two spectra: the
    # difference is a chi-squared residual clipped at zero and biases the
    # answer by nearly a decibel
    fw, Pc = welch(rlong["a_stay_clean"], fs=sim.fs, nperseg=8192,
                   window="hann", noverlap=4096)
    _, Pn = welch(rlong["a_stay"] - rlong["a_stay_clean"], fs=sim.fs,
                  nperseg=8192, window="hann", noverlap=4096)
    mb = (fw >= FBAND[0]) & (fw <= FBAND[1])
    p_sig = np.trapz(Pc[mb], fw[mb])
    p_noi = np.trapz(Pn[mb], fw[mb])
    snr_meas = 10.0 * np.log10(p_sig / p_noi)
    print("    signal power in %.1f-%.1f Hz is %.2f %% of the total; "
          "in-band SNR %.1f dB against %.0f dB broadband (measured on a "
          "600 s record: %.1f dB)"
          % (FBAND[0], FBAND[1], 100 * fr, snr_b, SNR_DEFAULT, snr_meas))
    add("snr_budget", snr_db=SNR_DEFAULT, band_power_frac=fr,
        snr_inband_db=snr_b, snr_inband_measured_db=snr_meas,
        noise_rms=md["noise_rms"][0], rms_stay=md["rms_stay"],
        rms_deck=md["rms_deck"],
        deck_below_stay_db=20 * np.log10(md["rms_stay"] / md["rms_deck"]))

    # -- 8 reproducibility and the pluck spectrum ---------------------------
    r1_ = sim.record(duration=30.0, snr_db=20.0, seed=7)
    r2_ = sim.record(duration=30.0, snr_db=20.0, seed=7)
    same = float(np.max(np.abs(r1_["a_stay"] - r2_["a_stay"])))
    r3_ = sim.record(duration=30.0, snr_db=20.0, seed=8)
    diff = float(np.max(np.abs(r1_["a_stay"] - r3_["a_stay"])))
    print("\n[8] same seed reproduces bit for bit (%.1e), different seed "
          "does not (%.1e)" % (same, diff))
    add("reproducibility", same_seed_max_diff=same,
        diff_seed_max_diff=diff)

    p = sim.record(duration=60.0, snr_db=None, excitation="pluck",
                   quantise=False)
    print("    pluck: peak %.1f mg at the stay sensor, RMS %.2f mg"
          % (np.max(np.abs(p["a_stay"])) / G0 * 1e3,
             p["meta"]["rms_stay_mg"]))
    add("pluck", peak_mg=float(np.max(np.abs(p["a_stay"])) / G0 * 1e3),
        rms_mg=p["meta"]["rms_stay_mg"], dof_pluck=p["meta"]["dof_pluck"])

    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, "simulate_records_verify.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("\nwrote %s" % out)

    if args.demo:
        d = sim.record(duration=600.0, snr_db=20.0, seed=0)
        dp = os.path.join(DATA, "record_demo.npz")
        np.savez_compressed(dp, t=d["t"], a_stay=d["a_stay"],
                            a_deck=d["a_deck"], fs=d["fs"],
                            f_modes=d["meta"]["f_modes"],
                            zeta_modes=d["meta"]["zeta_modes"])
        print("wrote %s" % dp)


if __name__ == "__main__":
    main()
