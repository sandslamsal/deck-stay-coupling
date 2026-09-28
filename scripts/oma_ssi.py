# -*- coding: utf-8 -*-
"""Stochastic subspace identification through a deck-stay crossing.

WHY THIS SCRIPT EXISTS
----------------------
Every result in this study about the bias of the isolated-cable tension
inversion assumes the frequency is PICKED from a spectrum as a maximum.  The
branch law prices a resolved pair, the merged-peak law prices the single
maximum that survives when the pair is not resolved, and the danger band
follows the picked peak across damping.  Against all of that the manuscript
asserts, without evidence, that a method which FITS modes rather than picking
maxima "can separate a pair the spectrum shows as one, in which case the
branch law returns in full".

That sentence is the one an ordinary reviewer of a signal-processing journal
will ask for evidence of, and there is none.  This script supplies it, or
refuses to, by running covariance-driven and data-driven stochastic subspace
identification on records from ``scripts/simulate_records.py`` and reporting
what they return.

The answer is not a single yes or no, and it is less favourable to the
manuscript's sentence than the sentence implies.  Every number below is
produced by this script on the worked bridge, whose pair splits by s = 2.338
per cent at a crossing near 151.6 kN.

WHERE THE SPECTRUM STILL SHOWS TWO PEAKS (u = s/2zeta at or above 2.34 here,
which is zeta at or below 0.5 per cent), the claim holds and holds easily.
Every instrument configuration and both algorithms returned both branches on
every one of the records tried.  The split came back to within 0.2 per cent
of its true value with a deck channel and within 9 per cent with a stay
sensor alone, and the tension error agreed with the branch law in magnitude
to within 4 per cent.  There the branch law does return in full.

WHERE THE SPECTRUM SHOWS ONE PEAK, which on this bridge begins at zeta = 1.0
per cent and not at the 2.4 per cent that the dip-existence threshold alone
would suggest, because the study's own 3 dB prominence convention needs a dip
of finite depth, the claim is true at first and then fails.  Twelve records
per case, at the exact crossing, taking the best of correlation lags 4, 16
and 32 s, the fraction in which the automatic rule returned BOTH branches is

    zeta      u        stay only   two on stay   stay + deck
    1.0 %    1.169        83 %         83 %          100 %
    2.0 %    0.585        25 %         17 %           83 %
    2.5 %    0.468         8 %         17 %           58 %
    3.0 %    0.390         0 %         25 %           58 %
    4.0 %    0.292         0 %         17 %           42 %

So a method that fits modes CAN separate a pair the spectrum shows as one,
which is what the manuscript asserts, and at 1 per cent damping it does so
almost always and even from a single stay sensor.  Below u = 0.5, which is
the dip threshold the study derives, a deck channel still succeeds in about
half of records and stay-mounted sensors essentially never do.  The sentence
in the manuscript is therefore right in the regime just past peak picking's
failure and wrong as a general statement.

SEPARATING THE PAIR DOES NOT REMOVE THE BIAS, and this is worth being plain
about because it is easy to read the manuscript's sentence as though it did.
When SSI returns two poles the engineer still has to choose one, and the
tension error is then the branch-law error, not zero: at the exact crossing
the mean absolute error of the reading is 2.4 to 3.3 per cent against a
branch-law value of 2.29 per cent.  What subspace identification buys is not
a smaller error but a PREDICTABLE one, governed by the branch law instead of
by where a broad maximum happened to land.

WHAT THE INSTRUMENT SEES DECIDES MORE THAN WHICH ALGORITHM IS USED.  SSI-COV
and SSI-DATA agree with each other to 0.06 per cent in frequency and differ
by less than the spread between sensor configurations.  With stay-mounted
sensors only, which is the instrumentation the incumbent tension method
actually uses, the pair is separated only when the fitted correlation lag
reaches about one beat period of the split: at 0.5 per cent damping the
success rate over lag runs 0, 0, 33, 67, 100, 100, 100 per cent at lags of
2, 4, 8, 12, 16, 24 and 32 s against a beat period of 12.9 s.  With a deck
channel it is 100 per cent at every one of those lags including 2 s, which
is a sixth of a beat period, because the two hybrid modes are then told apart
by shape rather than by waiting for the beat.  In the merged regime the
stay-only configuration managed a quarter of records at 2 per cent damping
and none at all at 3 per cent and above, at any lag tried.

A SECOND SENSOR ON THE SAME STAY DOES NOT HELP: its success rates sit within
the sampling error of twelve records of the single-sensor ones, above and
below them without pattern.  The reason is worth stating because it is not
obvious.  At the crossing the two hybrid modes have a MAC of 0.996 over two
stay stations and 0.9966 over three, so they are the same shape along the
stay to well inside the 2 per cent MAC criterion, and no clustering rule can
be expected to tell them apart on that.  Over one stay station and one deck
station the MAC is 0.859.  The information that separates the pair is in the
deck motion, not in more of the stay.

RECORD LENGTH RESCUES THE TWO-SENSOR CASE AND NOT THE ONE-SENSOR CASE, which
says the single-sensor failure is structural and not statistical.  At 3 per
cent damping, going from 600 s to 3600 s takes the stay-plus-deck success
rate from 33 to 67 per cent at a 16 s lag and from 58 to 83 per cent at 32 s,
and the number of model orders at which the pair is visible at all from 21.7
to 28.8 of 39; the stay-only configuration stays at zero per cent and its
pair visibility does not move, 4.4 orders against 4.2.  Six records per case
there, so those rates carry about 20 percentage points of sampling error, but
the contrast between the two configurations is far larger than that.

TWO FINDINGS FELL OUT THAT ARE NOT ABOUT SUBSPACE METHODS AT ALL.  First,
which branch a stay-mounted accelerometer reads larger depends on where it is
clamped: at exact tuning the modal kinetic energy makes the upper branch the
stay-dominated one, 0.5084 against 0.4917, and so does a sensor at mid-chord,
while a sensor 1 or 2 m above the anchorage, where field practice puts it,
reads the LOWER branch larger by 1.5 or 1.3 times.  The sign of the tension
error therefore depends on sensor position within about |d| < s.  Second, the
seed-to-seed scatter of the peak-picked tension grows from 0.11 per cent at
zeta = 0.2 per cent to 2.3 per cent at zeta = 4 per cent, so beyond about
zeta = 2 per cent the random error of locating a broad peak in a 600 s record
exceeds the entire systematic bias the merged-peak law describes.  The
danger-band result says the systematic bias falls with damping; these records
say what replaces it grows.

WHAT IS IMPLEMENTED
-------------------
Both classical algorithms, written out rather than called from a library, so
that every choice is visible.

SSI-COV.  Output correlations ``R_k = E[y_{t+k} y_t^T]`` for ``k = 0 .. 2i``
are formed by FFT.  The block Toeplitz matrix ``T1`` of size ``(l i, l i)``
carries ``R_{i+p-q}`` in block ``(p, q)``, so it uses lags 1 to 2i-1; the
shifted ``T2`` carries ``R_{i+1+p-q)``, lags 2 to 2i.  One singular value
decomposition ``T1 = U S V^T`` serves every model order: at order ``n`` the
observability matrix is ``O = U_n S_n^(1/2)`` and the controllability-like
factor is ``G = S_n^(1/2) V_n^T``.  ``C`` is the first block row of ``O``.
``A`` is obtained two independent ways, and their agreement is one of the
verification checks:

    shift invariance      A = pinv(O[:-l]) O[l:]
    Toeplitz realisation  A = S_n^(-1/2) U_n^T T2 V_n S_n^(-1/2)

SSI-DATA.  The block Hankel matrix of past and future outputs, ``i`` block
rows each and ``j = N - 2i + 1`` columns, scaled by ``1/sqrt(j)``, is reduced
by an LQ factorisation (a QR of its transpose).  With the standard
partitioning the orthogonal projection of the future row space onto the past
is ``P_i = L21 Q1^T``.  Because ``Q1`` has orthonormal rows, the singular
values and left singular vectors of ``P_i`` are those of ``L21``, so only
``L21`` is decomposed.  Three weightings are available: UPC (unweighted,
``W1 = W2 = I``, the default), PC, and CVA, which whitens by the future
output covariance ``(L21 L21^T + L22 L22^T)^(1/2)``.  ``O = W1^(-1) U_n
S_n^(1/2)`` and ``A``, ``C`` follow by shift invariance as above.

Eigenvalues of ``A`` become modal parameters through the exact inverse of the
zero-order hold, ``lambda = ln(mu)/dt``, so ``f = |lambda|/2pi`` and
``zeta = -Re(lambda)/|lambda|``.  The principal branch of the logarithm keeps
every identified frequency below the Nyquist rate automatically.  Mode shapes
at the sensors are ``C psi``.

STABILISATION DIAGRAM AND AUTOMATIC SELECTION
---------------------------------------------
Model orders ``n = 2, 4, ..., n_max`` are swept, ``n_max = 80`` by default and
capped at ``l(i-1)`` so the shift-invariance system stays overdetermined.
Each pole at order ``n`` is matched to its nearest neighbour in frequency at
order ``n - 2`` and graded on three criteria:

    frequency   |df|/f  < 1 per cent
    damping     |dz|/z  < 10 per cent
    shape       1 - MAC < 2 per cent

The damping tolerance is the one that decides how much of the diagram is
called stable, and 10 rather than the equally common 5 per cent was chosen
on the same synthetic calibration that set the clustering threshold, never
on the bridge.  Across those systems, requiring 5 per cent recovers 90.0 per
cent of the modes that are really there while admitting 1.1 per cent of the
spurious poles; 10 per cent recovers 95.8 per cent for 3.0 per cent; 20 per
cent recovers 96.7 per cent for 5.0 per cent.  Five per cent is too tight
for a reason that is visible in the diagrams rather than statistical: a pole
whose frequency repeats to five decimal places from one order to the next
has a damping estimate that wanders by 15 per cent, so the criterion rejects
poles that are not in any doubt.  The MAC criterion, by contrast, changes
almost nothing here, 127 stable poles against 127 with it switched off on
one synthetic case and 137 against 138 on another; it is kept because it
costs nothing and is standard, and because its being vacuous at one channel
would otherwise make the single-sensor case look as though it were held to
the same standard as the two-sensor case when it is not.

with the standard hard filters first: a conjugate pair with positive
imaginary part, ``0 < zeta < 20`` per cent, and a frequency inside the
analysis band.  Two further indicators are computed per pole:

    EMAC   the modulus of the complex correlation between the observability
           column ``O psi`` and the geometric sequence ``mu^k C psi`` it would
           be if the pole were exact.  It is 1 for a pole the data support and
           falls for one the least-squares residual invented.  It is defined
           for a single channel, which MAC and MPC are not.
    MPC    modal phase collinearity, the fraction of the mode shape's variance
           on the dominant axis of the (real, imaginary) scatter, computed as
           ((l1 - l2)/(l1 + l2))^2 from the eigenvalues of the 2 by 2 scatter
           matrix.  It is identically 1 for one channel and is reported as
           such rather than quietly used.

Selection is agglomerative clustering of the stable poles under the standard
distance ``d(p,q) = |f_p - f_q| / max(f_p, f_q) + (1 - MAC(phi_p, phi_q))``,
average linkage, cut at 2 per cent; with one channel the MAC term is
identically zero and the cut is on frequency alone at 1 per cent.  A cluster
survives if it holds poles from at least ``MIN_ORDER_FRAC`` of the swept
orders.  Its representative frequency and damping are the medians over the
cluster, which is robust to the one or two orders at which a pole wanders.

THE SINGLE CHOICE THAT MATTERS, AND HOW IT WAS MADE
---------------------------------------------------
The number of block rows ``i`` sets the maximum correlation lag
``tau = 2 i dt`` that the fit sees, and it is the only parameter that changes
the answer qualitatively.  It is NOT tuned against the truth here.  The
headline runs use ``tau = 16`` s, chosen from the analysis band alone: 16 s is
40 cycles of the 2.5 Hz bottom of the band and gives a lag-window resolution
``1/tau = 0.0625`` Hz, about 2 per cent of the band centre, which is the order
of split the study is about.  Because that choice is the load-bearing one, the
outcome is also reported as an explicit function of ``tau`` over 2 to 32 s,
and that sweep is a result in its own right rather than a robustness footnote.

Records are decimated to 20 Hz before identification, by a linear-phase
Kaiser FIR low pass at 8 Hz and downsampling by 5.  Twenty hertz is 4.8 times
the top of the analysis band.  This is not cosmetic and it is not free, and
check [6] of the verification measures both sides of it.  The 100 Hz record
carries 24 modes below the Nyquist rate against 6 below 8 Hz, and a state
space model must account for all of them before it reaches the pair, so with
a stay channel alone the pair does not appear anywhere in the sweep to order
80 that the rest of this study uses, and emerges only by order 240, where its
frequencies are out by 0.48 and 0.20 per cent; the decimated record returns
the same pair by order 80, out by 0.13 and 0.07 per cent.  With a deck
channel both rates find the pair and agree to 0.06 per cent.  So decimation
buys a three to fourfold accuracy gain and, for the single-sensor case that
this study turns on, the difference between working at an ordinary model
order and not working at all.  Each channel is then scaled to unit variance,
without which the deck channel, 41 dB below the stay channel, would be
beneath the singular values of the stay channel's noise.

VERIFICATION, ALL BEFORE ANY RESULT IS QUOTED
---------------------------------------------
No established OMA implementation is installed on this machine and the package
index is unreachable, so ``pip install pyoma2`` fails with "No matching
distribution found"; the same is true of pyOMA, sdypy, koma, pyEMA, sippy,
sysidentpy and python-control.  The cross-check against an outside
implementation therefore could not be run, and is not claimed.  In its place
are five checks against answers known independently of the code being tested:

    [1] EXACT CORRELATIONS.  Given the analytic output correlation sequence of
        a stochastic state-space model, built from the discrete Lyapunov
        solution rather than from data, SSI-COV must return the model's own
        eigenvalues exactly.  Agreement is at the level of the arithmetic.
    [2] WELL-SEPARATED PAIR, the check the brief asks for by name.  A synthetic
        four-mode system with prescribed frequencies and damping ratios,
        driven by white noise and sampled, identified from a finite record.
    [3] THE TWO ESTIMATORS OF A.  Shift invariance against the Toeplitz
        realisation, which use different parts of the same decomposition.
    [4] SSI-COV AGAINST SSI-DATA.  Two different reductions of the same record.
    [5] THE WORKED BRIDGE AWAY FROM THE CROSSING, where the pair is wide and
        the finite element truth is known mode by mode.
    [6] THE DECIMATION, identified both ways on one record, so that the
        preprocessing choice is measured rather than asserted.

Outputs, all new files:

    data/oma_ssi_verify.csv      the six checks
    data/oma_ssi_lag.csv         outcome against maximum correlation lag
    data/oma_ssi_crossing.csv    the crossing, per damping, detuning, seed
    data/oma_ssi_merged.csv      the merged regime, twelve seeds per case
    data/oma_ssi_longrec.csv     the same on hour-long records
    data/oma_ssi_branch.csv      which branch a stay sensor reads larger
    data/oma_ssi_stab.csv        one full stabilisation diagram, for a figure

Run:  python3 scripts/oma_ssi.py --verify
      python3 scripts/oma_ssi.py --lag
      python3 scripts/oma_ssi.py --crossing
      python3 scripts/oma_ssi.py --merged
      python3 scripts/oma_ssi.py --longrec
      python3 scripts/oma_ssi.py --branch
      python3 scripts/oma_ssi.py --all
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import eig, lstsq, qr, solve_discrete_lyapunov, svd
from scipy.signal import fftconvolve, firwin, welch, find_peaks
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from cablefe import tensioned_beam_freq                      # noqa: E402
from run_damping import BRIDGE, FBAND, T_TUNE                # noqa: E402
from run_merged import eps_merged_law                        # noqa: E402
from simulate_records import RecordSimulator                 # noqa: E402

DATA = os.path.join(ROOT, "data")

# ---- the analysis pipeline, fixed here and not tuned against the truth ----
FS_DEC = 20.0            # Hz, identification rate; 4.8 x the top of FBAND
TAU_MAX = 16.0           # s, maximum correlation lag of the headline runs
N_MAX = 80               # highest model order swept
N_STEP = 2               # order increment
ORD_MIN = 4              # lowest order that can carry a pole

# stability criteria of the diagram
TOL_F = 0.01             # 1 per cent on frequency
TOL_Z = 0.10             # 10 per cent on damping, calibrated below
TOL_MAC = 0.02           # 1 - MAC below 2 per cent
ZETA_MAX = 0.20          # hard physical filter
ZETA_MIN = 1e-4

# clustering
CLUST_CUT = 0.02         # 2 per cent in the combined distance
CLUST_CUT_1CH = 0.01     # 1 per cent in frequency alone, single channel
MIN_ORDER_FRAC = 0.50    # a cluster must span this fraction of the orders

# the analysis band.  Wider than FBAND so that a pole just outside is seen
# to be outside rather than silently discarded at the edge.
BAND = (2.0, 5.0)


# ===========================================================================
# preprocessing
# ===========================================================================

def decimate_fir(y, fs, fs_target, taps_per_q=80, beta=8.6):
    """Linear-phase Kaiser FIR low pass, then downsample.

    ``fftconvolve`` in "same" mode with an odd-length symmetric kernel removes
    the group delay exactly, so no phase is introduced that a subspace method
    could read as a state.  The filter transient is trimmed from both ends
    rather than left in the record.
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    q = int(round(fs / fs_target))
    if q <= 1:
        return y.copy(), fs
    ntaps = taps_per_q * q + 1
    taps = firwin(ntaps, 0.4 * fs / q, window=("kaiser", beta), fs=fs)
    yf = np.vstack([fftconvolve(row, taps, mode="same") for row in y])
    edge = ntaps // 2
    yf = yf[:, edge:yf.shape[1] - edge]
    return np.ascontiguousarray(yf[:, ::q]), fs / q


def prepare(y, fs, fs_target=FS_DEC, normalise=True):
    """Decimate, remove the mean and a linear trend, scale each channel.

    Channel scaling is not cosmetic.  The deck channel of this bridge sits
    41 dB below the stay channel under equal load intensity, so an unscaled
    two-channel Toeplitz matrix is numerically a one-channel Toeplitz matrix
    with a perturbation, and the deck information that separates the pair is
    below the stay channel's own noise floor in the singular value spectrum.
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    y, fsd = decimate_fir(y, fs, fs_target)
    n = y.shape[1]
    t = np.arange(n) - 0.5 * (n - 1)
    for r in range(y.shape[0]):                    # remove mean and trend
        y[r] -= y[r].mean()
        y[r] -= t * (t @ y[r]) / (t @ t)
    if normalise:
        y = y / y.std(axis=1, keepdims=True)
    return y, fsd


def output_correlations(y, kmax, biased=True):
    """``R[k] = E[y_{t+k} y_t^T]`` for ``k = 0 .. kmax``, by FFT.

    The biased normalisation (divide by ``N`` at every lag) is the default
    because it is the one that keeps the block Toeplitz matrix positive
    semidefinite, which the realisation theory assumes.  Its cost is a
    triangular taper ``1 - k/N`` on the correlation, an apparent extra decay
    rate of ``1/(N dt)``; at the record lengths used here that is 1.7 per cent
    of the true decay rate of the pair and is reported rather than ignored.
    """
    y = np.atleast_2d(y)
    l, N = y.shape
    nfft = 1 << int(np.ceil(np.log2(2 * N + 1)))
    Y = np.fft.rfft(y, n=nfft, axis=1)
    S = Y[:, None, :] * np.conj(Y[None, :, :])
    c = np.fft.irfft(S, n=nfft, axis=2)[:, :, :kmax + 1]
    R = np.moveaxis(c, 2, 0)                       # (kmax+1, l, l)
    if biased:
        return R / float(N)
    return R / (N - np.arange(kmax + 1))[:, None, None]


# ===========================================================================
# the two algorithms
# ===========================================================================

class _Subspace:
    """Common machinery: one decomposition, every model order off it."""

    def __init__(self, dt, l, i):
        self.dt, self.l, self.i = float(dt), int(l), int(i)
        self.U = self.S = self.V = None

    @property
    def n_max_possible(self):
        return min(self.l * (self.i - 1), len(self.S))

    def observability(self, n):
        return self.U[:, :n] * np.sqrt(self.S[:n])

    def state_matrix(self, n, route="shift"):
        raise NotImplementedError

    @staticmethod
    def _lsq(a, b):
        """Least squares by the LAPACK complete-orthogonal driver.

        ``numpy.linalg.lstsq`` calls ``gelsd`` on this machine and takes 1.9 s
        on the 318 by 80 systems the order sweep generates, which is 12 s of a
        13 s stabilisation diagram; ``gelsy`` returns the same answer to
        3e-15 in 1 ms.  The order sweep runs a few thousand of these, so the
        difference decides whether the crossing study takes minutes or a day.
        """
        return lstsq(a, b, lapack_driver="gelsy")[0]

    def modal(self, n, route="shift"):
        """Modal parameters at model order ``n``."""
        O = self.observability(n)
        A = self.state_matrix(n, route)
        C = O[:self.l]
        return modal_from_AC(A, C, self.dt, O=O)


class SSICov(_Subspace):
    """Covariance-driven SSI.

    ``y`` is ``(l, N)``.  ``i`` block rows means correlation lags up to
    ``2 i`` are used, a maximum lag of ``2 i dt`` seconds.
    """

    def __init__(self, y, fs, i, biased=True):
        y = np.atleast_2d(y)
        super().__init__(1.0 / fs, y.shape[0], i)
        l = self.l
        R = output_correlations(y, 2 * i, biased=biased)
        self.R = R
        T1 = np.empty((l * i, l * i))
        T2 = np.empty((l * i, l * i))
        for p in range(i):
            for q in range(i):
                T1[p * l:(p + 1) * l, q * l:(q + 1) * l] = R[i + p - q]
                T2[p * l:(p + 1) * l, q * l:(q + 1) * l] = R[i + 1 + p - q]
        self.T1, self.T2 = T1, T2
        self.U, self.S, Vt = svd(T1)
        self.V = Vt.T

    def state_matrix(self, n, route="shift"):
        l = self.l
        if route == "shift":
            O = self.observability(n)
            return self._lsq(O[:-l], O[l:])
        if route == "realisation":
            s = 1.0 / np.sqrt(self.S[:n])
            return (s[:, None] * (self.U[:, :n].T @ self.T2 @ self.V[:, :n])
                    * s[None, :])
        raise ValueError("route must be shift or realisation")


class SSIData(_Subspace):
    """Data-driven SSI by LQ factorisation of the block Hankel matrix.

    Weightings: ``UPC`` (unweighted principal component, the default),
    ``PC``, ``CVA``.  UPC and PC differ only in a right weighting, which does
    not change the left singular vectors and therefore does not change the
    observability matrix or the poles; both are provided because the
    literature names them separately, and the code makes the equivalence
    visible instead of hiding it.
    """

    def __init__(self, y, fs, i, weighting="UPC"):
        y = np.atleast_2d(y)
        super().__init__(1.0 / fs, y.shape[0], i)
        l, N = self.l, y.shape[1]
        j = N - 2 * i + 1
        if j < 2 * l * i:
            raise ValueError("record too short for %d block rows" % i)
        self.j = j
        # block Hankel, past on top of future, columns scaled by 1/sqrt(j)
        H = np.empty((2 * l * i, j))
        for b in range(2 * i):
            H[b * l:(b + 1) * l] = y[:, b:b + j]
        H /= np.sqrt(j)
        # LQ by QR of the transpose; only the triangular factor is needed
        Rq = qr(H.T, mode="r")[0]
        L = Rq[:2 * l * i].T
        li = l * i
        L21 = L[li:, :li]
        L22 = L[li:, li:]
        self.L21, self.L22 = L21, L22
        if weighting in ("UPC", "PC"):
            self.W1inv = None
            M = L21
        elif weighting == "CVA":
            Ryy = L21 @ L21.T + L22 @ L22.T
            w, Q = np.linalg.eigh(0.5 * (Ryy + Ryy.T))
            w = np.maximum(w, 1e-14 * w.max())
            W1 = Q @ np.diag(w ** -0.5) @ Q.T
            self.W1inv = Q @ np.diag(w ** 0.5) @ Q.T
            M = W1 @ L21
        else:
            raise ValueError("weighting must be UPC, PC or CVA")
        self.weighting = weighting
        self.U, self.S, Vt = svd(M, full_matrices=False)
        self.V = Vt.T

    def observability(self, n):
        O = self.U[:, :n] * np.sqrt(self.S[:n])
        return O if self.W1inv is None else self.W1inv @ O

    def state_matrix(self, n, route="shift"):
        O = self.observability(n)
        return self._lsq(O[:-self.l], O[self.l:])


# ===========================================================================
# eigenvalues to modal parameters, and the per-pole indicators
# ===========================================================================

def _rcmul(A, B):
    """Real matrix times complex matrix, as two real products.

    ``numpy``'s mixed real-complex ``@`` takes the complex path on this build
    and is 74 times slower than doing the two real products by hand: 177 ms
    against 2.4 ms on the 80 by 78 times 78 by 78 case the order sweep runs
    forty times per diagram.  Casting the real operand to complex first is
    worse again, 320 ms, so it is the complex GEMM itself that is slow here
    and not the mixed dispatch.  The two forms agree to 1.2e-14.
    """
    return A @ B.real + 1j * (A @ B.imag)


def modal_from_AC(A, C, dt, O=None):
    """Frequencies, damping ratios, shapes and indicators from ``A``, ``C``.

    Only one member of each conjugate pair is returned, the one with positive
    imaginary part.  Real eigenvalues are overdamped or spurious and are
    dropped here rather than filtered later, because they have no frequency.
    """
    mu, psi = eig(A)
    keep = np.imag(mu) > 0
    mu, psi = mu[keep], psi[:, keep]
    if mu.size == 0:
        z = np.zeros(0)
        return dict(f=z, zeta=z, phi=np.zeros((C.shape[0], 0), complex),
                    lam=np.zeros(0, complex), emac=z, mpc=z,
                    mu=np.zeros(0, complex))
    lam = np.log(mu) / dt
    absl = np.abs(lam)
    f = absl / (2.0 * np.pi)
    zeta = -np.real(lam) / absl
    phi = _rcmul(C, psi)
    emac = _emac(O, psi, mu, C.shape[0]) if O is not None else np.ones(len(mu))
    mpc = np.array([_mpc(phi[:, k]) for k in range(phi.shape[1])])
    return dict(f=f, zeta=zeta, phi=phi, lam=lam, emac=emac, mpc=mpc, mu=mu)


def _emac(O, psi, mu, l):
    """Consistency of the observability column with a geometric progression.

    If ``(A, C)`` were exact, the column ``O psi_k`` would be
    ``[C psi_k, mu_k C psi_k, mu_k^2 C psi_k, ...]`` exactly, because
    ``O = [C; CA; CA^2; ...]``.  ``A`` here is a least-squares fit to the
    shift of ``O``, so the column departs from that progression by the
    residual, and the modulus of the complex correlation between the two is a
    number between 0 and 1 that says how much of the pole the data support.

    Unlike MAC between orders and unlike MPC, this is meaningful with a
    single channel, which is exactly the case this study needs graded.

    The geometric factors are formed as ``exp(k log mu)`` with the real part
    of the exponent floored, rather than as ``mu**k``.  A heavily damped
    spurious pole has ``|mu|`` well below one, and at 160 block rows
    ``|mu|^159`` underflows to a subnormal, where complex arithmetic on this
    processor runs about two orders of magnitude slower.  That single detail
    was 93 per cent of the cost of a stabilisation diagram before it was
    fixed, 6.3 s of 6.8 s.  Flooring the exponent at -300 changes nothing:
    the terms it discards are below 1e-130 of the leading one and cannot
    move a correlation coefficient.
    """
    n_blocks = O.shape[0] // l
    n = psi.shape[1]
    if n == 0:
        return np.zeros(0)
    V = _rcmul(O, psi).reshape(n_blocks, l, n)
    k = np.arange(n_blocks)
    E = np.outer(k, np.log(mu))                    # (n_blocks, n)
    W = np.exp(np.clip(E.real, -300.0, 300.0) + 1j * E.imag)
    W = W[:, None, :] * V[0][None, :, :]           # (n_blocks, l, n)
    num = np.abs(np.einsum("bln,bln->n", np.conj(W), V))
    den = (np.sqrt(np.einsum("bln,bln->n", np.conj(V), V).real)
           * np.sqrt(np.einsum("bln,bln->n", np.conj(W), W).real))
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def _mpc(phi):
    """Modal phase collinearity, as the anisotropy of the (Re, Im) scatter.

    ``((l1 - l2)/(l1 + l2))^2`` with ``l1 >= l2`` the eigenvalues of the 2 by 2
    scatter matrix of the real and imaginary parts.  One for a monophase mode.

    The scatter is taken about the ORIGIN, not about the mean of the shape
    components.  A monophase mode has its components on a line through the
    origin, because a mode shape and its negative are the same mode, so the
    origin is where the line belongs.  The mean-removed variant, which some
    codes use, is degenerate for two channels: after centring the two points
    are equal and opposite, the scatter is rank one, and MPC is identically
    one no matter what the phases are.  With the scatter about the origin it
    is informative from two channels and degenerate only at one, where the
    single point is trivially collinear with itself.
    """
    x, y = np.real(phi), np.imag(phi)
    if x.size < 2:
        return 1.0
    Sxx, Syy, Sxy = x @ x, y @ y, x @ y
    tr, det = Sxx + Syy, Sxx * Syy - Sxy ** 2
    if tr <= 0:
        return 0.0
    disc = max(tr * tr / 4.0 - det, 0.0) ** 0.5
    l1, l2 = tr / 2.0 + disc, tr / 2.0 - disc
    return float(((l1 - l2) / (l1 + l2)) ** 2)


def mac(a, b):
    """Modal assurance criterion between two complex shapes."""
    na, nb = np.vdot(a, a).real, np.vdot(b, b).real
    if na <= 0 or nb <= 0:
        return 0.0
    return float(abs(np.vdot(a, b)) ** 2 / (na * nb))


# ===========================================================================
# the stabilisation diagram
# ===========================================================================

def stabilisation(est, orders=None, band=BAND, route="shift",
                  tol_f=TOL_F, tol_z=TOL_Z, tol_mac=TOL_MAC):
    """Poles at every model order, graded against the order below.

    Returns a DataFrame with one row per pole per order and the columns
    ``order, f, zeta, emac, mpc, stab_f, stab_z, stab_mac, stable``.  The
    hard physical filters are applied first, so a row that exists is already
    a conjugate pole with a positive damping ratio below 20 per cent and a
    frequency inside the analysis band; the three ``stab_`` flags then say
    whether it repeated at the previous order.

    Matching between orders is to the NEAREST pole in frequency at the
    previous swept order, among poles that survived the hard filters.  That is
    the usual rule and it is deliberately generous: it can only make a pole
    look more stable than a stricter matching would, so a pole this diagram
    calls unstable is unstable under any of the usual variants.
    """
    if orders is None:
        orders = range(ORD_MIN, min(N_MAX, est.n_max_possible) + 1, N_STEP)
    orders = [int(n) for n in orders if n >= ORD_MIN]
    prev = None
    rows = []
    for n in orders:
        m = est.modal(n, route=route)
        ok = ((m["f"] > band[0]) & (m["f"] < band[1])
              & (m["zeta"] > ZETA_MIN) & (m["zeta"] < ZETA_MAX))
        f, z = m["f"][ok], m["zeta"][ok]
        phi = m["phi"][:, ok]
        emac, mpc = m["emac"][ok], m["mpc"][ok]
        order_ = np.argsort(f)
        f, z, phi = f[order_], z[order_], phi[:, order_]
        emac, mpc = emac[order_], mpc[order_]
        for k in range(len(f)):
            sf = sz = sm = False
            if prev is not None and len(prev[0]):
                jj = int(np.argmin(np.abs(prev[0] - f[k])))
                sf = abs(prev[0][jj] - f[k]) / f[k] < tol_f
                sz = abs(prev[1][jj] - z[k]) / z[k] < tol_z
                sm = (1.0 - mac(prev[2][:, jj], phi[:, k])) < tol_mac
            rows.append(dict(order=n, f=f[k], zeta=z[k], emac=emac[k],
                             mpc=mpc[k], stab_f=sf, stab_z=sz, stab_mac=sm,
                             stable=bool(sf and sz and sm),
                             idx=len(rows), _phi=phi[:, k]))
        prev = (f, z, phi)
    df = pd.DataFrame(rows)
    return df


def select_poles(df, l, min_order_frac=MIN_ORDER_FRAC, cut=None,
                 use_stable=True, compact=False):
    """Cluster the stable poles and return one representative per cluster.

    The distance is the standard one,

        d(p, q) = |f_p - f_q| / max(f_p, f_q) + (1 - MAC(phi_p, phi_q)),

    cut at 2 per cent under average linkage.  With one channel every MAC is
    identically 1, so the second term vanishes and the cut is on frequency
    alone; the threshold is halved there to 1 per cent so that the criterion
    is not quietly loosened by the loss of the shape term.

    A cluster is then kept on one condition: its poles must come from at
    least ``min_order_frac`` of the swept model orders.  Counting DISTINCT
    orders rather than poles is what makes the rule mean "this pole survived
    as the order grew" rather than "this pole appeared often", which one
    order producing several near duplicates would otherwise satisfy.

    HOW THAT THRESHOLD WAS SET, AND WHAT ELSE WAS TRIED.  It was calibrated
    on synthetic systems only, never on the bridge, so that the rule was
    fixed before it was pointed at the question this study asks.  Seven
    modal systems of two to four modes, at two noise levels, three seeds and
    one and two channels, gave 742 clusters at ``N_MAX = 40`` and 1917 at
    ``N_MAX = 80``; each was labelled real or spurious by whether its
    frequency fell within 1 per cent of a mode the system was built with.
    Requiring half the orders keeps 86 and 83 per cent of the real poles
    while admitting 1 per cent of the spurious ones at both maximum orders.
    Nothing else tried did better.  Requiring the longest CONSECUTIVE run of
    orders to be half the sweep admits no spurious poles at all but loses
    more than half the real ones.  Measuring persistence from the order at
    which a pole first appears, which sounds fairer to a mode that needs a
    high order before it emerges, is worse on both counts, admitting 8 to 22
    per cent of the spurious poles for fewer real ones.  Screening on the
    cluster's internal scatter of frequency and damping, or on EMAC, admits
    10 to 30 per cent of the spurious poles at a comparable loss.

    THE SCATTER SCREEN, LEFT IN AND SWITCHED OFF.  ``compact=True`` requires
    the cluster's median absolute deviation in frequency to be within
    ``TOL_F`` of its median and in damping within ``TOL_Z``.  It is what a
    reader would reach for first and the reason it is not used should be
    inspectable rather than asserted: real poles on the worked bridge have a
    damping scatter of 9 to 11 per cent of their own median, because the
    damping estimate genuinely wanders with model order, so a screen tight
    enough to reject the spurious poles rejects the real ones with them.

    ``n_clusters_raw`` in the returned frame records how many clusters passed
    persistence before any scatter screen, so a rejection is visible rather
    than showing up as a pole that was never there.
    """
    cols = ["f", "zeta", "n_orders", "order_frac", "emac", "mpc",
            "f_std", "zeta_std", "f_mad", "zeta_mad", "n_poles", "phi"]
    empty = pd.DataFrame(columns=cols)
    empty.attrs["n_clusters_raw"] = 0
    if df is None or not len(df):
        return empty
    d = df[df["stable"]] if use_stable else df
    if not len(d):
        return empty
    n_orders_total = df["order"].nunique()
    if cut is None:
        cut = CLUST_CUT_1CH if l == 1 else CLUST_CUT
    f = d["f"].to_numpy()
    m = len(f)
    if m == 1:
        lab = np.array([1])
    else:
        fa = np.maximum(f[:, None], f[None, :])
        D = np.abs(f[:, None] - f[None, :]) / fa
        if l > 1 and "_phi" in d:
            P = np.array(list(d["_phi"])).T          # (l, m)
            P = P / np.linalg.norm(P, axis=0, keepdims=True)
            D = D + (1.0 - np.abs(P.conj().T @ P) ** 2)
        np.fill_diagonal(D, 0.0)
        D = 0.5 * (D + D.T)
        lab = fcluster(linkage(squareform(D, checks=False), method="average"),
                       t=cut, criterion="distance")
    out = []
    n_raw = 0
    for c in np.unique(lab):
        sel = d[lab == c]
        n_ord = sel["order"].nunique()
        frac = n_ord / n_orders_total
        if frac < min_order_frac:
            continue
        n_raw += 1
        fm = float(np.median(sel["f"]))
        zm = float(np.median(sel["zeta"]))
        fmad = float(np.median(np.abs(sel["f"] - fm)))
        zmad = float(np.median(np.abs(sel["zeta"] - zm)))
        if compact and (fmad / fm > TOL_F or zmad / max(zm, 1e-12) > TOL_Z):
            continue
        rep_i = int(np.argmin(np.abs(sel["f"].to_numpy() - fm)))
        phi_rep = (sel["_phi"].iloc[rep_i] if "_phi" in sel
                   else np.ones(1, complex))
        out.append(dict(f=fm, zeta=zm, n_orders=int(n_ord),
                        order_frac=float(frac), phi=phi_rep,
                        emac=float(np.median(sel["emac"])),
                        mpc=float(np.median(sel["mpc"])),
                        f_std=float(np.std(sel["f"])),
                        zeta_std=float(np.std(sel["zeta"])),
                        f_mad=fmad, zeta_mad=zmad, n_poles=int(len(sel))))
    if not out:
        res = empty
    else:
        res = pd.DataFrame(out).sort_values("f").reset_index(drop=True)
    res.attrs["n_clusters_raw"] = n_raw
    return res


def identify(y, fs, method="cov", tau=TAU_MAX, fs_target=FS_DEC,
             band=BAND, weighting="UPC", route="shift", orders=None,
             return_diagram=False):
    """The whole pipeline: prepare, decompose, sweep orders, select poles.

    ``tau`` is the maximum correlation lag in seconds; the block row count
    follows as ``i = round(tau fs_dec / 2)``.
    """
    yp, fsd = prepare(y, fs, fs_target)
    i = max(2, int(round(tau * fsd / 2.0)))
    l = yp.shape[0]
    if method == "cov":
        est = SSICov(yp, fsd, i)
    elif method == "data":
        est = SSIData(yp, fsd, i, weighting=weighting)
    else:
        raise ValueError("method must be cov or data")
    df = stabilisation(est, orders=orders, band=band, route=route)
    sel = select_poles(df, l)
    if return_diagram:
        return sel, df, est
    return sel


# ===========================================================================
# a synthetic stochastic state-space model with prescribed modal parameters
# ===========================================================================

class ModalSystem:
    """A discrete stochastic system built from frequencies and damping ratios.

    Each mode is one exactly discretised second-order oscillator in the state
    ``[q, q']``, driven by white process noise, observed through prescribed
    mode shapes as displacement.  Because the continuous poles are written
    down rather than solved for, the modal parameters this system possesses
    are known to the last digit, which is what makes it a verification case
    and not another simulation to be trusted.
    """

    def __init__(self, f, zeta, shapes, dt, q_scale=None, seed=0):
        f = np.asarray(f, float)
        zeta = np.asarray(zeta, float)
        self.f, self.zeta, self.dt = f, zeta, float(dt)
        self.Psi = np.asarray(shapes, float)          # (l, nmodes)
        nm = len(f)
        w = 2 * np.pi * f
        A = np.zeros((2 * nm, 2 * nm))
        B = np.zeros((2 * nm, nm))
        for k in range(nm):
            wk, zk = w[k], zeta[k]
            wd = wk * np.sqrt(1 - zk ** 2)
            e = np.exp(-zk * wk * dt)
            c, s = np.cos(wd * dt), np.sin(wd * dt)
            A[2 * k:2 * k + 2, 2 * k:2 * k + 2] = e * np.array([
                [c + zk * wk / wd * s, s / wd],
                [-wk ** 2 / wd * s, c - zk * wk / wd * s]])
            B[2 * k + 1, k] = 1.0
        self.A, self.B = A, B
        C = np.zeros((self.Psi.shape[0], 2 * nm))
        C[:, ::2] = self.Psi
        self.C = C
        self.q = np.ones(nm) if q_scale is None else np.asarray(q_scale, float)
        self.seed = seed

    def simulate(self, n, seed=None, noise_std=0.0):
        rng = np.random.default_rng(self.seed if seed is None else seed)
        nm = len(self.f)
        x = np.zeros(2 * nm)
        # burn in eight of the slowest decay times
        nb = int(8.0 / (np.min(self.zeta * 2 * np.pi * self.f) * self.dt))
        out = np.empty((self.C.shape[0], n))
        Bq = self.B * self.q[None, :]
        for k in range(nb):
            x = self.A @ x + Bq @ rng.standard_normal(nm)
        for k in range(n):
            out[:, k] = self.C @ x
            x = self.A @ x + Bq @ rng.standard_normal(nm)
        if noise_std:
            out = out + noise_std * out.std(axis=1, keepdims=True) \
                * rng.standard_normal(out.shape)
        return out

    def exact_correlations(self, kmax):
        """``R_k`` from the discrete Lyapunov solution, no data involved.

        ``Sigma = A Sigma A^T + Q`` with ``Q = B diag(q^2) B^T``, then
        ``R_0 = C Sigma C^T`` and ``R_k = C A^(k-1) G`` with ``G = A Sigma
        C^T``.  This is the sequence the realisation theory says the block
        Toeplitz matrix factors, so feeding it to SSI-COV tests the algorithm
        against its own premise with no estimation error anywhere.
        """
        Q = self.B @ np.diag(self.q ** 2) @ self.B.T
        Sig = solve_discrete_lyapunov(self.A, Q)
        G = self.A @ Sig @ self.C.T
        R = np.empty((kmax + 1, self.C.shape[0], self.C.shape[0]))
        R[0] = self.C @ Sig @ self.C.T
        M = np.eye(self.A.shape[0])
        for k in range(1, kmax + 1):
            R[k] = self.C @ M @ G
            M = M @ self.A
        return R


def _ssicov_from_R(R, dt, i, l):
    """SSI-COV driven by a supplied correlation sequence, not by a record."""
    est = _Subspace.__new__(SSICov)
    _Subspace.__init__(est, dt, l, i)
    T1 = np.empty((l * i, l * i))
    T2 = np.empty((l * i, l * i))
    for p in range(i):
        for q in range(i):
            T1[p * l:(p + 1) * l, q * l:(q + 1) * l] = R[i + p - q]
            T2[p * l:(p + 1) * l, q * l:(q + 1) * l] = R[i + 1 + p - q]
    est.R, est.T1, est.T2 = R, T1, T2
    est.U, est.S, Vt = svd(T1)
    est.V = Vt.T
    return est


def _match(f_true, z_true, f_got, z_got):
    """Pair identified poles to true ones by nearest frequency."""
    ef, ez = [], []
    for ft, zt in zip(f_true, z_true):
        if not len(f_got):
            ef.append(np.nan)
            ez.append(np.nan)
            continue
        k = int(np.argmin(np.abs(np.asarray(f_got) - ft)))
        ef.append((f_got[k] - ft) / ft)
        ez.append((z_got[k] - zt) / zt)
    return np.array(ef), np.array(ez)


# ===========================================================================
# verification
# ===========================================================================

def verify(dur=600.0, verbose=True):
    """Five checks against answers known independently of this code."""
    rows = []

    def add(check, **kw):
        rows.append(dict(check=check, **kw))
        if verbose:
            print("  " + check + ": " +
                  ", ".join("%s=%s" % (k, _fmt(v)) for k, v in kw.items()))

    # -- [1] exact correlations ------------------------------------------
    if verbose:
        print("\n[1] EXACT CORRELATIONS -> exact poles")
    dt = 1.0 / 20.0
    f_t = np.array([1.10, 3.28, 3.36, 5.55])
    z_t = np.array([0.008, 0.005, 0.005, 0.006])
    shp = np.array([[1.0, 1.0, -0.9, 0.4],
                    [0.3, 0.8, 1.0, -1.0]])
    sysm = ModalSystem(f_t, z_t, shp, dt, q_scale=[1.0, 1.0, 1.0, 1.0])
    i = 40
    R = sysm.exact_correlations(2 * i + 1)
    est = _ssicov_from_R(R, dt, i, 2)
    for route in ("shift", "realisation"):
        m = est.modal(8, route=route)
        o = np.argsort(m["f"])
        ef, ez = _match(f_t, z_t, m["f"][o], m["zeta"][o])
        add("exact_correlations", route=route, n=8,
            max_rel_err_f=float(np.max(np.abs(ef))),
            max_rel_err_zeta=float(np.max(np.abs(ez))))

    # -- [2] the well-separated pair at low damping, from a finite record --
    if verbose:
        print("\n[2] WELL-SEPARATED PAIR AT LOW DAMPING, finite record")
    f_s = np.array([2.00, 3.50])
    z_s = np.array([0.005, 0.005])
    shp2 = np.array([[1.0, 1.0], [0.6, -0.8]])
    for nch in (2, 1):
        errs_f, errs_z, nfound, det = [], [], [], []
        for seed in range(8):
            sysm2 = ModalSystem(f_s, z_s, shp2[:nch], dt, seed=seed)
            y = sysm2.simulate(int(dur / dt), noise_std=0.10)
            sel = identify(y, 1.0 / dt, method="cov", tau=TAU_MAX,
                           fs_target=1.0 / dt, band=(1.0, 5.0))
            nfound.append(len(sel))
            fg = sel["f"].to_numpy()
            zg = sel["zeta"].to_numpy()
            # detection and error are different things and are kept apart:
            # a mode that was not returned at all is a miss, not a large
            # error, and averaging the two together would hide both
            for ft, zt in zip(f_s, z_s):
                if len(fg) and np.min(np.abs(fg - ft)) / ft < 0.01:
                    k = int(np.argmin(np.abs(fg - ft)))
                    det.append(True)
                    errs_f.append((fg[k] - ft) / ft)
                    errs_z.append((zg[k] - zt) / zt)
                else:
                    det.append(False)
        add("separated_pair", channels=nch, seeds=8,
            detection_rate=float(np.mean(det)),
            n_found_min=int(np.min(nfound)), n_found_max=int(np.max(nfound)),
            max_rel_err_f=float(np.max(np.abs(errs_f))) if errs_f else np.nan,
            max_rel_err_zeta=(float(np.max(np.abs(errs_z))) if errs_z
                              else np.nan),
            rms_rel_err_f=float(np.sqrt(np.mean(np.array(errs_f) ** 2)))
            if errs_f else np.nan,
            rms_rel_err_zeta=float(np.sqrt(np.mean(np.array(errs_z) ** 2)))
            if errs_z else np.nan)

    # -- [3] and [4] the two A routes and the two algorithms ---------------
    if verbose:
        print("\n[3] SHIFT INVARIANCE vs TOEPLITZ REALISATION, and\n"
              "[4] SSI-COV vs SSI-DATA, on one bridge record")
    sim = RecordSimulator(T_TUNE, 0.005)
    rec = sim.record(duration=dur, snr_db=20.0, seed=0)
    y2 = np.vstack([rec["a_stay"], rec["a_deck"]])
    yp, fsd = prepare(y2, rec["fs"])
    ii = int(round(TAU_MAX * fsd / 2.0))
    ecov = SSICov(yp, fsd, ii)
    d_sh = stabilisation(ecov, route="shift")
    d_re = stabilisation(ecov, route="realisation")
    s_sh = select_poles(d_sh, 2)
    s_re = select_poles(d_re, 2)
    band = (s_sh["f"] > FBAND[0]) & (s_sh["f"] < FBAND[1])
    band_re = (s_re["f"] > FBAND[0]) & (s_re["f"] < FBAND[1])
    fa, fb = s_sh["f"][band].to_numpy(), s_re["f"][band_re].to_numpy()
    za, zb = s_sh["zeta"][band].to_numpy(), s_re["zeta"][band_re].to_numpy()
    add("A_routes_agree", n_shift=len(fa), n_realisation=len(fb),
        max_rel_df=_safe_max_rel(fa, fb),
        max_rel_dzeta=_safe_max_rel(za, zb))

    edat = SSIData(yp, fsd, ii)
    s_da = select_poles(stabilisation(edat), 2)
    bd = (s_da["f"] > FBAND[0]) & (s_da["f"] < FBAND[1])
    fd_, zd_ = s_da["f"][bd].to_numpy(), s_da["zeta"][bd].to_numpy()
    add("cov_vs_data", n_cov=len(fa), n_data=len(fd_),
        max_rel_df=_safe_max_rel(fa, fd_),
        max_rel_dzeta=_safe_max_rel(za, zd_))

    # -- [5] the worked bridge away from the crossing ----------------------
    if verbose:
        print("\n[5] THE WORKED BRIDGE AWAY FROM THE CROSSING")
    for T, lab in ((1.30e5, "detuned low"), (1.75e5, "detuned high")):
        s5 = RecordSimulator(T, 0.005)
        r5 = s5.record(duration=dur, snr_db=20.0, seed=1)
        ft = np.array([s5.f_lo, s5.f_hi])
        zt = np.array([s5.zj[s5.pair_idx[0]], s5.zj[s5.pair_idx[1]]])
        # which of the two the stay sensor can see at all: the stay energy
        # fraction, and the mode-shape amplitude at the sensor itself
        es = s5.energy_split[s5.pair_idx]
        amp = np.abs(s5.Phi[s5.dof_stay, s5.pair_idx])
        k_stay = int(np.argmax(amp))                 # stay-dominated branch
        for nch, name in ((2, "stay+deck"), (1, "stay only")):
            yy = (np.vstack([r5["a_stay"], r5["a_deck"]]) if nch == 2
                  else r5["a_stay"][None, :])
            for meth in ("cov", "data"):
                sel = identify(yy, r5["fs"], method=meth, tau=TAU_MAX)
                inb = sel[(sel["f"] > BAND[0]) & (sel["f"] < BAND[1])]
                fg = inb["f"].to_numpy()
                zg = inb["zeta"].to_numpy()
                ef, ez = _match(ft, zt, fg, zg)
                add("bridge_detuned", case=lab, T_kN=T / 1e3, channels=name,
                    method=meth,
                    split_pct=100 * (ft[1] - ft[0]) / np.mean(ft),
                    f_true_lo=ft[0], f_true_hi=ft[1],
                    stay_energy_lo=es[0], stay_energy_hi=es[1],
                    amp_ratio_at_sensor=float(amp.min() / amp.max()),
                    n_in_band=len(inb),
                    err_f_stay_branch=float(ef[k_stay]),
                    err_zeta_stay_branch=float(ez[k_stay]),
                    err_f_other_branch=float(ef[1 - k_stay]),
                    max_rel_err_f=float(np.nanmax(np.abs(ef))),
                    max_rel_err_zeta=float(np.nanmax(np.abs(ez))))
    # -- [6] the decimation ------------------------------------------------
    if verbose:
        print("\n[6] DECIMATION TO 20 Hz AGAINST THE RAW 100 Hz RECORD")
    sim6 = RecordSimulator(T_TUNE, 0.005)
    r6 = sim6.record(duration=dur, snr_db=20.0, seed=0)
    ft6 = np.array([sim6.f_lo, sim6.f_hi])
    for cfg, nch in (("stay+deck", 2), ("stay only", 1)):
        yy = (np.vstack([r6["a_stay"], r6["a_deck"]]) if nch == 2
              else r6["a_stay"][None, :])
        # the SAME order grid in every row, N_STEP as everywhere else, so
        # that the three rows differ only in the sampling rate and the
        # highest order and not in how finely the sweep was taken
        for fst, nmax, lab in ((FS_DEC, N_MAX, "decimated 20 Hz"),
                               (r6["fs"], N_MAX, "raw 100 Hz"),
                               (r6["fs"], 240, "raw 100 Hz, order to 240")):
            sel = identify(yy, r6["fs"], method="cov", tau=TAU_MAX,
                           fs_target=fst,
                           orders=range(ORD_MIN, nmax + 1, N_STEP))
            inb = sel[(sel["f"] > FBAND[0]) & (sel["f"] < FBAND[1])]
            ef, _ = _match(ft6, [0.005, 0.005], inb["f"].to_numpy(),
                           inb["zeta"].to_numpy())
            add("decimation", channels=cfg, rate=lab, n_max=nmax,
                n_in_band=len(inb),
                rel_err_f_lo=float(ef[0]) if len(inb) else np.nan,
                rel_err_f_hi=float(ef[1]) if len(inb) else np.nan,
                max_rel_err_f=(float(np.nanmax(np.abs(ef)))
                               if len(inb) else np.nan))
    return pd.DataFrame(rows)


def _safe_max_rel(a, b):
    """Largest relative difference between two equal-length vectors.

    Returns NaN rather than raising when the two identifications did not
    return the same number of poles, because that case is a finding and not
    an error: it means one method found something the other did not, and the
    ``n_`` columns beside this one say which.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b) or not len(a):
        return np.nan
    return float(np.max(np.abs(a - b) / np.abs(a)))


def _fmt(v):
    if isinstance(v, float):
        return "%.6g" % v
    return str(v)


# ===========================================================================
# the incumbent inversion, the laws it is measured against, and peak picking
# ===========================================================================

def eps_from_freq(f_hat, f_iso):
    """Relative tension error of the isolated-cable inversion.

    ``T = 4 m L^2 f^2 / n^2``, so with the isolated stay fundamental as the
    reference the relative tension error of a reading ``f_hat`` is
    ``(f_hat/f_iso)^2 - 1``.  This is the same quantity
    ``run_merged.picked_error`` returns and the same one ``eps_branch_law``
    and ``eps_merged_law`` predict, so the three are directly comparable.
    """
    return (f_hat / f_iso) ** 2 - 1.0


def eps_branch_law(d, s):
    """Stay-dominated branch error, signed.  Copied in form from run_merged."""
    sg = 1.0 if d >= 0 else -1.0
    return sg * (np.hypot(d, s) - abs(d))


def peak_pick(y, fs, band=FBAND, nperseg=8192, prom_db=3.0, smooth=3,
               window=None):
    """What an engineer reads off the spectrum of the same record.

    Welch spectrum of the stay channel, smoothed across ``smooth`` bins by a
    Hann kernel, peaks above a 3 dB prominence floor inside the band, the
    largest returned as the reading.  This is the incumbent method, run on
    the identical record the subspace methods see, so the comparison is
    between methods and not between datasets.

    THE SMOOTHING IS NOT COSMETIC AND IS NOT FREE.  A raw Welch estimate of a
    600 s record at ``nperseg = 8192`` averages thirteen segments, so each
    bin carries about 28 per cent standard error, and a 3 dB prominence rule
    then counts noise wiggles as peaks: four in the band at 0.5 per cent
    damping where the pair is plainly two, twenty-five at ``nperseg =
    16384``.  Averaging three bins with a Hann kernel widens the effective
    resolution to about 0.03 Hz, still a third of the 0.078 Hz separation of
    the worked bridge's pair, and returns the count to what the eye sees.
    Two counts are returned.  ``n_band`` is every peak in the analysis band,
    which is what a blind engineer would tally and which stays noisy however
    the spectrum is smoothed, because the band is 1.7 Hz wide and mostly
    empty.  ``n_window`` counts only the peaks inside a window supplied by
    the caller and set around the pair, and it is that count the dip
    criterion predicts.  The distinction matters: at 0.2 per cent damping the
    band count is three or four while the window count is two, and the pair
    is unambiguously two peaks.
    """
    f, P = welch(y, fs=fs, nperseg=min(nperseg, len(y)), noverlap=None)
    if smooth and smooth > 1:
        w = np.hanning(smooth + 2)[1:-1]
        P = np.convolve(P, w / w.sum(), mode="same")
    m = (f >= band[0]) & (f <= band[1])
    fb, Pb = f[m], P[m]
    db = 10.0 * np.log10(np.maximum(Pb, 1e-300))
    pk, _ = find_peaks(db, prominence=prom_db)
    if not len(pk):
        k = int(np.argmax(Pb))
        return fb[k], 1, fb[[k]], 1
    kbest = pk[int(np.argmax(Pb[pk]))]
    fp = fb[pk]
    n_win = (int(np.sum((fp >= window[0]) & (fp <= window[1])))
             if window is not None else len(fp))
    return fb[kbest], len(pk), fp, n_win


def _pair_truth(sim):
    """Truth carried by the simulator: branches, damping, detuning, split."""
    f_deck = sim.cd.deck_alone(8)
    j = int(np.argmin(np.abs(f_deck - sim.f_iso1)))
    d = (sim.f_iso1 - f_deck[j]) / sim.f_iso1
    return dict(f_lo=sim.f_lo, f_hi=sim.f_hi, f0=sim.f0, s_obs=sim.s_split,
                f_iso1=sim.f_iso1, f_deck=float(f_deck[j]), d=float(d),
                z_lo=float(sim.zj[sim.pair_idx[0]]),
                z_hi=float(sim.zj[sim.pair_idx[1]]),
                e_lo=float(sim.energy_split[sim.pair_idx[0]]),
                e_hi=float(sim.energy_split[sim.pair_idx[1]]))


def _channels(sim, rec, config):
    """The three instrument configurations compared.

    ``stay`` is one accelerometer on the stay, which is what the incumbent
    tension method uses.  ``stay+deck`` adds a deck channel at the anchorage,
    which is what a bridge monitoring system has.  ``stay2`` is two
    accelerometers on the SAME stay, at 2 m and mid-chord, which is what an
    engineer would reach for if told that one sensor is not enough; it is in
    the comparison because the answer is not obvious and turns out to be
    negative for a reason worth stating.
    """
    if config == "stay":
        return rec["a_stay"][None, :]
    if config == "stay+deck":
        return np.vstack([rec["a_stay"], rec["a_deck"]])
    if config == "stay2":
        return np.vstack([rec["a_stay"], rec["a_deck"]])
    raise ValueError(config)


def _simulator(T, zeta, config, cache={}):
    """A simulator for one tension, damping and instrument configuration.

    For ``stay2`` the output DOFs are set to two points on the stay before
    the record is generated.  ``RecordSimulator`` exposes the DOF list it
    observes through, so this needs no change to that module: the second
    channel is moved from the deck anchorage to mid-chord on the stay.
    """
    key = (round(T, 3), round(zeta, 8), config == "stay2")
    if key in cache:
        return cache[key]
    sim = RecordSimulator(T, zeta)
    if config == "stay2":
        sim.dofs = np.array([sim.cd.cable_sensor_dof(2.0),
                             sim.cd.cable_sensor_dof(0.5 * BRIDGE["Lc"])])
    if len(cache) > 4:
        cache.clear()
    cache[key] = sim
    return sim


# ===========================================================================
# study 1: the outcome against maximum correlation lag
# ===========================================================================

def lag_study(zetas=(0.005, 0.02, 0.03), taus=(2.0, 4.0, 8.0, 12.0, 16.0,
                                              24.0, 32.0),
              seeds=(0, 1, 2), dur=600.0, snr_db=20.0, verbose=True):
    """Does the pair separate, as a function of the fitted correlation lag?

    Run at the exact crossing, where the split is smallest and the question
    hardest.  The prediction under test is that a stay-only instrument needs
    a lag of the order of one beat period ``1/(f_hi - f_lo)`` before it can
    tell two poles from one, because that is when the two-mode and one-mode
    descriptions of the correlation stop being interchangeable, while a
    stay-plus-deck instrument does not, because the two hybrid modes have
    different deck components and are told apart by shape at any lag.
    """
    rows = []
    for zeta in zetas:
        sim = _simulator(T_TUNE, zeta, "stay+deck")
        tr = _pair_truth(sim)
        beat = 1.0 / (tr["f_hi"] - tr["f_lo"])
        for seed in seeds:
            rec = sim.record(duration=dur, snr_db=snr_db, seed=seed)
            for config in ("stay", "stay+deck"):
                y = _channels(sim, rec, config)
                for tau in taus:
                    if tau * FS_DEC / 2.0 * y.shape[0] > 0.25 * dur * FS_DEC:
                        continue
                    try:
                        sel = identify(y, rec["fs"], method="cov", tau=tau)
                    except Exception:
                        continue
                    inb = sel[(sel["f"] > FBAND[0]) & (sel["f"] < FBAND[1])]
                    got = _grade(inb, tr)
                    rows.append(dict(zeta=zeta, u=tr["s_obs"] / (2 * zeta),
                                     seed=seed, config=config, tau=tau,
                                     tau_over_beat=tau / beat, beat_s=beat,
                                     **got))
    df = pd.DataFrame(rows)
    if verbose and len(df):
        print("\nOUTCOME AGAINST MAXIMUM CORRELATION LAG, at the crossing")
        print("beat period 1/(f_hi - f_lo) = %.2f s\n" % df["beat_s"].iloc[0])
        g = df.groupby(["config", "zeta", "tau"]).agg(
            both=("both_found", "mean"), n=("n_in_band", "mean"),
            tob=("tau_over_beat", "first")).reset_index()
        for cfg in ("stay", "stay+deck"):
            print("  %s" % cfg)
            sub = g[g["config"] == cfg]
            print("    %6s %8s " % ("tau", "tau/beat") +
                  " ".join("z=%.1f%%" % (100 * z)
                           for z in sorted(df["zeta"].unique())))
            for tau in sorted(sub["tau"].unique()):
                r = sub[sub["tau"] == tau]
                cells = []
                for z in sorted(df["zeta"].unique()):
                    v = r[r["zeta"] == z]["both"]
                    cells.append("  %4.0f%%" % (100 * v.iloc[0])
                                 if len(v) else "     -")
                print("    %6.1f %8.2f " % (tau, r["tob"].iloc[0])
                      + " ".join(cells))
    return df


def _grade(inb, tr, tol=0.01):
    """Compare the poles found in the band with the two true branches."""
    f = inb["f"].to_numpy()
    z = inb["zeta"].to_numpy()
    out = dict(n_in_band=len(f))
    for name, ft, zt in (("lo", tr["f_lo"], tr["z_lo"]),
                         ("hi", tr["f_hi"], tr["z_hi"])):
        if len(f):
            k = int(np.argmin(np.abs(f - ft)))
            out["f_" + name] = float(f[k])
            out["zeta_" + name] = float(z[k])
            out["err_f_" + name] = float((f[k] - ft) / ft)
            out["err_z_" + name] = float((z[k] - zt) / zt)
        else:
            out["f_" + name] = np.nan
            out["zeta_" + name] = np.nan
            out["err_f_" + name] = np.nan
            out["err_z_" + name] = np.nan
    matched = set()
    for ft in (tr["f_lo"], tr["f_hi"]):
        if len(f):
            k = int(np.argmin(np.abs(f - ft)))
            if abs(f[k] - ft) / ft < tol:
                matched.add(k)
    out["n_matched"] = len(matched)
    out["both_found"] = bool(len(matched) == 2)
    out["split_found"] = (float((max(f) - min(f)) / np.mean(f))
                          if len(f) >= 2 else 0.0)
    return out


# ===========================================================================
# study 2: through the crossing
# ===========================================================================

TENSIONS = (1.430e5, 1.470e5, 1.516e5, 1.560e5, 1.610e5)
ZETAS = (0.002, 0.005, 0.010, 0.020, 0.025, 0.030, 0.040)
CONFIGS = ("stay", "stay2", "stay+deck")
U_SPLIT = np.sqrt(np.sqrt(5.0) - 2.0)      # 0.48587, below which the dip goes


def _psd_at(y, fs, freqs, nperseg=8192):
    """Welch density of the stay channel at given frequencies."""
    f, P = welch(y, fs=fs, nperseg=min(nperseg, len(y)))
    return np.array([P[int(np.argmin(np.abs(f - ff)))] for ff in freqs])


def crossing_study(tensions=TENSIONS, zetas=ZETAS, seeds=(0, 1, 2),
                   configs=CONFIGS, methods=("cov", "data"),
                   taus=(4.0, 16.0), dur=600.0, snr_db=20.0, verbose=True):
    """What each identification method returns through the crossing.

    For every tension, damping ratio, seed, instrument configuration, method
    and correlation lag: the poles found in the band, whether they match the
    two true branches, and the tension the incumbent inversion would return
    from them, beside the tension peak picking returns from the same record
    and beside the branch law and the merged-peak law.

    Three readings are priced, because "SSI separates the pair" and "the
    engineer gets the right tension" are different claims:

        eps_ssi_amp     the pole an engineer would take, being the one whose
                        spectral density at the stay channel is larger
        eps_ssi_oracle  the pole that IS the stay-dominated branch, which no
                        engineer can know but which is what the branch law
                        prices
        eps_peak        the maximum of the stay channel's own spectrum, the
                        incumbent method run on the identical record
    """
    rows = []
    t0 = time.time()
    for zeta in zetas:
        for T in tensions:
            sims = {}
            for config in configs:
                sims[config] = _simulator(T, zeta, config)
            tr = _pair_truth(sims[configs[0]])
            s_tune = 0.0234
            u = np.hypot(tr["d"], tr["s_obs"]) / (2.0 * zeta)
            sim0 = sims[configs[0]]
            amp = np.abs(sim0.Phi[sim0.cd.cable_sensor_dof(2.0),
                                  sim0.pair_idx])
            f_stay_branch = tr["f_lo"] if amp[0] > amp[1] else tr["f_hi"]
            for seed in seeds:
                # one record per distinct sensor set, shared by the
                # configurations that read the same instrument
                recs = {}
                for config in configs:
                    key = id(sims[config])
                    if key not in recs:
                        recs[key] = sims[config].record(
                            duration=dur, snr_db=snr_db, seed=seed)
                rec0 = recs[id(sims[configs[0]])]
                half = 2.5 * max(tr["s_obs"], 0.005) * tr["f0"]
                f_pk, n_pk, _, n_pk_win = peak_pick(
                    rec0["a_stay"], rec0["fs"],
                    window=(tr["f0"] - half, tr["f0"] + half))
                for config in configs:
                    rec = recs[id(sims[config])]
                    y = _channels(sims[config], rec, config)
                    for method in methods:
                        for tau in taus:
                            try:
                                sel = identify(y, rec["fs"], method=method,
                                               tau=tau)
                            except Exception as exc:
                                rows.append(dict(T=T, zeta=zeta, seed=seed,
                                                 config=config, method=method,
                                                 tau=tau, failed=str(exc)))
                                continue
                            inb = sel[(sel["f"] > FBAND[0])
                                      & (sel["f"] < FBAND[1])]
                            g = _grade(inb, tr)
                            fg = inb["f"].to_numpy()
                            eps_amp = eps_or = np.nan
                            f_amp = f_or = np.nan
                            if len(fg):
                                dens = _psd_at(rec0["a_stay"], rec0["fs"], fg)
                                f_amp = float(fg[int(np.argmax(dens))])
                                eps_amp = eps_from_freq(f_amp, tr["f_iso1"])
                                f_or = float(fg[int(np.argmin(
                                    np.abs(fg - f_stay_branch)))])
                                eps_or = eps_from_freq(f_or, tr["f_iso1"])
                            rows.append(dict(
                                T=T, zeta=zeta, seed=seed, config=config,
                                method=method, tau=tau, u=u, d=tr["d"],
                                s_obs=tr["s_obs"], s_tune=s_tune,
                                resolvable=bool(u > U_SPLIT),
                                f_lo_true=tr["f_lo"], f_hi_true=tr["f_hi"],
                                f_iso1=tr["f_iso1"], f_deck=tr["f_deck"],
                                z_lo_true=tr["z_lo"], z_hi_true=tr["z_hi"],
                                e_lo=tr["e_lo"], e_hi=tr["e_hi"],
                                n_peaks_spectrum=n_pk,
                                n_peaks_pair=n_pk_win, f_peak=f_pk,
                                eps_merged=eps_merged_law(tr["d"], s_tune,
                                                          zeta),
                                eps_peak=eps_from_freq(f_pk, tr["f_iso1"]),
                                f_ssi_amp=f_amp, eps_ssi_amp=eps_amp,
                                f_ssi_oracle=f_or, eps_ssi_oracle=eps_or,
                                eps_branch=eps_branch_law(tr["d"], s_tune),
                                T_hat_peak=4 * BRIDGE["mc"] * BRIDGE["Lc"] ** 2
                                * f_pk ** 2,
                                T_hat_ssi=(4 * BRIDGE["mc"] * BRIDGE["Lc"] ** 2
                                           * f_amp ** 2 if len(fg)
                                           else np.nan),
                                failed="", **g))
            if verbose:
                print("    zeta %.3f  T %.1f kN   %.0f s"
                      % (zeta, T / 1e3, time.time() - t0))
    return pd.DataFrame(rows)


def summarise_crossing(df, verbose=True):
    """The two tables the question actually asks for."""
    ok = df[df["failed"] == ""]
    tune = ok[np.abs(ok["d"]) < 0.005]           # the exact crossing
    if verbose:
        print("\n" + "=" * 78)
        print("AT THE CROSSING: does the method see one pole or two?")
        print("=" * 78)
        print("  u = s/2zeta below %.4f means the SPECTRUM shows one peak.\n"
              % U_SPLIT)
        hdr = ("  %-6s %-6s %-9s %-10s %-6s %8s %8s %9s"
               % ("zeta", "u", "config", "method", "tau", "both", "n_pole",
                  "n_peak"))
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for (z, cfg, meth, tau), g in tune.groupby(
                ["zeta", "config", "method", "tau"]):
            print("  %-6.3f %-6.2f %-9s %-10s %-6.0f %7.0f%% %8.2f %9.2f"
                  % (z, g["u"].mean(), cfg, meth, tau,
                     100 * g["both_found"].mean(), g["n_in_band"].mean(),
                     g["n_peaks_spectrum"].mean()))
    return tune


def pair_visibility(df, tr, tol=0.01):
    """Was the pair ever in the diagram, whatever the selection rule did?

    The automatic rule is deliberately conservative: it keeps only poles that
    persist over half the swept orders.  That is the right rule for an
    analyst who must decide without knowing the answer, but it conflates two
    very different failures, and the difference matters to the claim under
    test.  A pair that never appears at any order means the record does not
    carry the information.  A pair that appears cleanly at orders 12 to 20
    and then dissolves means the information is there and the rule threw it
    away, which is a statement about the rule and not about the physics.

    Returns the number of swept orders at which TWO poles simultaneously sit
    within ``tol`` of the two true branches, the first and last such order,
    and the longest run of consecutive such orders.  Stability is ignored
    here on purpose; this is a question about what the diagram contains.
    """
    if df is None or not len(df):
        return dict(n_orders_pair=0, first_order_pair=np.nan,
                    last_order_pair=np.nan, run_orders_pair=0)
    orders = np.array(sorted(df["order"].unique()))
    hit = []
    for o in orders:
        f = df[df["order"] == o]["f"].to_numpy()
        if len(f) < 2:
            continue
        a = np.min(np.abs(f - tr["f_lo"])) / tr["f_lo"]
        b = np.min(np.abs(f - tr["f_hi"])) / tr["f_hi"]
        ka = int(np.argmin(np.abs(f - tr["f_lo"])))
        kb = int(np.argmin(np.abs(f - tr["f_hi"])))
        if a < tol and b < tol and ka != kb:
            hit.append(o)
    if not hit:
        return dict(n_orders_pair=0, first_order_pair=np.nan,
                    last_order_pair=np.nan, run_orders_pair=0)
    pos = np.searchsorted(orders, hit)
    run = best = 1
    for k in range(1, len(pos)):
        run = run + 1 if pos[k] == pos[k - 1] + 1 else 1
        best = max(best, run)
    return dict(n_orders_pair=len(hit), first_order_pair=int(hit[0]),
                last_order_pair=int(hit[-1]), run_orders_pair=int(best))


def merged_study(zetas=(0.010, 0.020, 0.025, 0.030, 0.040), seeds=range(12),
                 configs=CONFIGS, methods=("cov", "data"),
                 taus=(4.0, 16.0, 32.0), dur=600.0, snr_db=20.0,
                 T=T_TUNE, verbose=True):
    """The merged regime, with enough seeds to quote a rate.

    Everything here is at the exact crossing and at damping ratios where the
    spectrum of this bridge shows ONE maximum under the study's own 3 dB
    prominence convention, which the noise-free frequency response puts at
    1.0 per cent damping and above.  The lag study answers the question at
    three seeds, which is enough to see a pattern and not enough to quote a
    number; this answers it at twelve, per damping ratio, configuration,
    method and correlation lag.

    Reported per case: the fraction of records in which BOTH branches were
    returned within 1 per cent, the fraction in which the split was
    recovered to within a quarter of its true value, and the tension error
    of the reading an engineer would take.
    """
    rows = []
    t0 = time.time()
    for zeta in zetas:
        sims = {c: _simulator(T, zeta, c) for c in configs}
        tr = _pair_truth(sims[configs[0]])
        u = np.hypot(tr["d"], tr["s_obs"]) / (2.0 * zeta)
        for seed in seeds:
            recs = {}
            for c in configs:
                k = id(sims[c])
                if k not in recs:
                    recs[k] = sims[c].record(duration=dur, snr_db=snr_db,
                                             seed=seed)
            rec0 = recs[id(sims[configs[0]])]
            half = 2.5 * tr["s_obs"] * tr["f0"]
            f_pk, n_pk, _, n_pk_win = peak_pick(
                rec0["a_stay"], rec0["fs"],
                window=(tr["f0"] - half, tr["f0"] + half))
            for c in configs:
                rec = recs[id(sims[c])]
                y = _channels(sims[c], rec, c)
                for method in methods:
                    for tau in taus:
                        try:
                            sel, diag, _ = identify(y, rec["fs"],
                                                    method=method, tau=tau,
                                                    return_diagram=True)
                        except Exception:
                            continue
                        inb = sel[(sel["f"] > FBAND[0])
                                  & (sel["f"] < FBAND[1])]
                        g = _grade(inb, tr)
                        g.update(pair_visibility(diag, tr))
                        g["n_orders_swept"] = int(diag["order"].nunique())
                        fg = inb["f"].to_numpy()
                        eps = np.nan
                        if len(fg):
                            dens = _psd_at(rec0["a_stay"], rec0["fs"], fg)
                            eps = eps_from_freq(
                                float(fg[int(np.argmax(dens))]), tr["f_iso1"])
                        rows.append(dict(
                            zeta=zeta, u=u, seed=seed, config=c,
                            method=method, tau=tau, dur=dur,
                            n_peaks_pair=n_pk_win, eps_peak=eps_from_freq(
                                f_pk, tr["f_iso1"]),
                            eps_ssi_amp=eps, s_true=tr["s_obs"],
                            eps_branch=eps_branch_law(tr["d"], 0.0234),
                            eps_merged=eps_merged_law(tr["d"], 0.0234, zeta),
                            **g))
        if verbose:
            print("    zeta %.3f  u %.3f   %.0f s"
                  % (zeta, u, time.time() - t0))
    df = pd.DataFrame(rows)
    if verbose and len(df):
        print("\nTHE MERGED REGIME: fraction of records returning BOTH "
              "branches\n")
        print("  %-7s %-6s %-10s %-6s %s" % ("zeta", "u", "config", "method",
              " ".join("tau=%-5.0f" % t for t in sorted(df["tau"].unique()))))
        for z in sorted(df["zeta"].unique()):
            for c in configs:
                for m in sorted(df["method"].unique()):
                    sub = df[(df["zeta"] == z) & (df["config"] == c)
                             & (df["method"] == m)]
                    if not len(sub):
                        continue
                    cells = []
                    for t in sorted(df["tau"].unique()):
                        v = sub[sub["tau"] == t]["both_found"]
                        cells.append("%6.0f%%  " % (100 * v.mean())
                                     if len(v) else "      -  ")
                    print("  %-7.3f %-6.2f %-10s %-6s %s"
                          % (z, sub["u"].iloc[0], c, m, "".join(cells)))
    return df


def branch_dominance(tensions=TENSIONS, stations=(1.0, 2.0, 4.0, 6.0, 12.5,
                                                20.0), verbose=True):
    """Which branch a stay-mounted sensor reads larger, against its position.

    This costs no records and is not an identification result at all; it is a
    property of the mode shapes.  It is here because the crossing study threw
    up a sign that did not match the branch law and this is the explanation.

    The branch law and the merged-peak law both descend from a two degree of
    freedom reduction in which the stay is ONE coordinate, so "the residue at
    the stay" and "the stay-dominated branch" are the same thing there.  On a
    stay with distributed mass they are not.  The two hybrid modes have
    slightly different shapes ALONG the chord, because each carries a
    component forced through the tie at the anchorage as well as its own
    modal motion, and the two add with opposite signs.  Near the anchorage
    the tie-driven part is a large fraction of the total and the ordering of
    the two residues can invert.

    On the worked bridge at exact tuning the modal kinetic energy says the
    upper branch is the stay-dominated one, by 0.5084 to 0.4917, and so does
    a sensor at mid-chord or beyond.  A sensor 1 or 2 m above the anchorage,
    which is where field practice puts it because that is what can be
    reached, reads the LOWER branch larger by a factor of 1.5 or 1.3.  The
    sign of the tension error therefore depends on where the accelerometer is
    clamped, and only within about ``|d| < s``: by a detuning of 1.5 per cent
    every station agrees again.
    """
    rows = []
    for T in tensions:
        sim = _simulator(T, 0.005, "stay+deck")
        tr = _pair_truth(sim)
        k = sim.pair_idx
        for x in stations:
            dof = sim.cd.cable_sensor_dof(x)
            r = abs(sim.Phi[dof, k[0]]) / abs(sim.Phi[dof, k[1]])
            rows.append(dict(T=T, d=tr["d"], station_m=x, ratio_lo_hi=r,
                             lower_reads_larger=bool(r > 1.0),
                             e_lo=tr["e_lo"], e_hi=tr["e_hi"],
                             energy_says_lower=bool(tr["e_lo"] > tr["e_hi"]),
                             f_lo=tr["f_lo"], f_hi=tr["f_hi"]))
    df = pd.DataFrame(rows)
    if verbose:
        print("\nWHICH BRANCH A STAY SENSOR READS LARGER\n")
        print("  %-9s %9s %8s %8s  %s" % ("T kN", "d", "E_lo", "E_hi",
              " ".join("%6.1fm" % x for x in stations)))
        for T in tensions:
            g = df[df["T"] == T]
            print("  %-9.1f %+9.5f %8.4f %8.4f  %s"
                  % (T / 1e3, g["d"].iloc[0], g["e_lo"].iloc[0],
                     g["e_hi"].iloc[0],
                     " ".join("%7.3f" % v for v in g["ratio_lo_hi"])))
        print("\n  ratio above 1 means the LOWER branch is read larger there")
    return df


def stab_diagram_dump(T=T_TUNE, zeta=0.03, seed=0, dur=600.0, snr_db=20.0,
                      taus=(4.0, 16.0), configs=("stay", "stay+deck"),
                      method="cov"):
    """One full stabilisation diagram per configuration, for plotting.

    Written at a damping ratio inside the merged regime, because that is the
    case the whole question turns on: the spectrum there has one maximum, and
    the diagram is where a second pole either appears or does not.
    """
    rows = []
    for config in configs:
        sim = _simulator(T, zeta, config)
        tr = _pair_truth(sim)
        rec = sim.record(duration=dur, snr_db=snr_db, seed=seed)
        y = _channels(sim, rec, config)
        for tau in taus:
            sel, df, _ = identify(y, rec["fs"], method=method, tau=tau,
                                  return_diagram=True)
            for _, r in df.iterrows():
                rows.append(dict(config=config, tau=tau, order=int(r["order"]),
                                 f=r["f"], zeta=r["zeta"], emac=r["emac"],
                                 mpc=r["mpc"], stab_f=bool(r["stab_f"]),
                                 stab_z=bool(r["stab_z"]),
                                 stab_mac=bool(r["stab_mac"]),
                                 stable=bool(r["stable"]),
                                 selected=bool(len(sel) and np.min(
                                     np.abs(sel["f"].to_numpy() - r["f"]))
                                     / r["f"] < 0.005),
                                 f_lo_true=tr["f_lo"], f_hi_true=tr["f_hi"],
                                 T=T, zeta_true=zeta, seed=seed))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--lag", action="store_true")
    ap.add_argument("--crossing", action="store_true")
    ap.add_argument("--stab", action="store_true")
    ap.add_argument("--merged", action="store_true")
    ap.add_argument("--longrec", action="store_true")
    ap.add_argument("--branch", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dur", type=float, default=600.0)
    ap.add_argument("--snr", type=float, default=20.0)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    if not any((a.verify, a.lag, a.crossing, a.stab, a.merged,
                a.longrec, a.branch, a.all)):
        a.all = True
    seeds = tuple(range(a.seeds))
    os.makedirs(DATA, exist_ok=True)

    if a.verify or a.all:
        print("=" * 78)
        print("VERIFICATION")
        print("=" * 78)
        print("  no established OMA package is installed and the package "
              "index is\n  unreachable, so the outside cross-check could not "
              "be run; see the\n  module docstring.")
        v = verify(dur=a.dur)
        v.to_csv(os.path.join(DATA, "oma_ssi_verify.csv"), index=False)
        print("\n  -> data/oma_ssi_verify.csv")

    if a.lag or a.all:
        print("\n" + "=" * 78)
        print("MAXIMUM CORRELATION LAG")
        print("=" * 78)
        lg = lag_study(seeds=seeds, dur=a.dur, snr_db=a.snr)
        lg.to_csv(os.path.join(DATA, "oma_ssi_lag.csv"), index=False)
        print("\n  -> data/oma_ssi_lag.csv")

    if a.crossing or a.all:
        print("\n" + "=" * 78)
        print("THE CROSSING")
        print("=" * 78)
        cr = crossing_study(seeds=seeds, dur=a.dur, snr_db=a.snr)
        cr.to_csv(os.path.join(DATA, "oma_ssi_crossing.csv"), index=False)
        summarise_crossing(cr)
        print("\n  -> data/oma_ssi_crossing.csv")

    if a.merged or a.all:
        print("\n" + "=" * 78)
        print("THE MERGED REGIME")
        print("=" * 78)
        mg = merged_study(dur=a.dur, snr_db=a.snr)
        mg.to_csv(os.path.join(DATA, "oma_ssi_merged.csv"), index=False)
        print("\n  -> data/oma_ssi_merged.csv")

    if a.longrec:
        print("\n" + "=" * 78)
        print("THE MERGED REGIME ON AN HOUR-LONG RECORD")
        print("=" * 78)
        print("  600 s is the ordinary length of an ambient stay record.  If\n"
              "  the pair separates at 3600 s and not at 600 s, the limit is\n"
              "  the record and not the method, and that is worth knowing\n"
              "  before anyone concludes that subspace identification cannot\n"
              "  do it.")
        lr = merged_study(zetas=(0.020, 0.030), seeds=range(6),
                          configs=("stay", "stay+deck"), methods=("cov",),
                          taus=(16.0, 32.0), dur=3600.0, snr_db=a.snr)
        lr.to_csv(os.path.join(DATA, "oma_ssi_longrec.csv"), index=False)
        print("\n  -> data/oma_ssi_longrec.csv")

    if a.branch or a.all:
        bd = branch_dominance()
        bd.to_csv(os.path.join(DATA, "oma_ssi_branch.csv"), index=False)
        print("\n  -> data/oma_ssi_branch.csv")

    if a.stab or a.all:
        sd = stab_diagram_dump(dur=a.dur, snr_db=a.snr)
        sd.to_csv(os.path.join(DATA, "oma_ssi_stab.csv"), index=False)
        print("\n  -> data/oma_ssi_stab.csv")


if __name__ == "__main__":
    main()
