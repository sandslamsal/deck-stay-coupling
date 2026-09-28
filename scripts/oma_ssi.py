# -*- coding: utf-8 -*-
"""Stochastic subspace identification through a deck-stay crossing.

Runs SSI-COV and SSI-DATA, with a stabilization diagram and automatic pole
clustering, on records from scripts/simulate_records.py, and compares the
result with peak picking, the branch law and the merged-peak law. --verify
checks the code against known answers; no outside OMA package was available.
Writes data/oma_ssi_*.csv.  Run: python3 scripts/oma_ssi.py --all
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

# --- identification settings ---
FS_DEC = 20.0            # Hz, identification rate; 4.8 x the top of FBAND
TAU_MAX = 16.0           # s, maximum correlation lag of the headline runs
N_MAX = 80               # highest model order swept
N_STEP = 2               # order increment
ORD_MIN = 4              # lowest order that can carry a pole

# stability criteria of the diagram
TOL_F = 0.01             # 1 percent on frequency
TOL_Z = 0.10             # 10 percent on damping, set on synthetic systems
TOL_MAC = 0.02           # 1 - MAC below 2 percent
ZETA_MAX = 0.20          # hard physical filter
ZETA_MIN = 1e-4

# clustering
CLUST_CUT = 0.02         # 2 percent in the combined distance
CLUST_CUT_1CH = 0.01     # 1 percent in frequency alone, single channel
MIN_ORDER_FRAC = 0.50    # a cluster must span this fraction of the orders

# analysis band, wider than FBAND so that poles near its edges are kept
BAND = (2.0, 5.0)


# --- preprocessing ---

def decimate_fir(y, fs, fs_target, taps_per_q=80, beta=8.6):
    """Linear-phase Kaiser FIR low pass, then downsample.

    ``fftconvolve`` in "same" mode with an odd-length symmetric kernel adds no
    group delay. The filter transient is trimmed from both ends.
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

    Scaling to unit variance is needed because the deck channel is about
    41 dB below the stay channel and would otherwise sit below the stay
    channel's noise in the singular values.
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

    Returns shape ``(kmax + 1, l, l)``. The biased estimate (divide by ``N``)
    keeps the block Toeplitz matrix positive semidefinite, at the cost of an
    apparent extra decay rate of ``1/(N dt)``.
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


# --- the two algorithms ---

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
        """Least squares by the fast LAPACK ``gelsy`` driver."""
        return lstsq(a, b, lapack_driver="gelsy")[0]

    def modal(self, n, route="shift"):
        """Modal parameters at model order ``n``."""
        O = self.observability(n)
        A = self.state_matrix(n, route)
        C = O[:self.l]
        return modal_from_AC(A, C, self.dt, O=O)


class SSICov(_Subspace):
    """Covariance-driven SSI.

    ``y`` is ``(l, N)``. ``i`` block rows use correlation lags up to ``2 i``,
    a maximum lag of ``2 i dt`` seconds. ``T1`` holds ``R[i+p-q]`` in block
    ``(p, q)`` and ``T2`` is shifted by one lag. ``A`` comes from shift
    invariance, ``pinv(O[:-l]) O[l:]``, or from the Toeplitz realization,
    ``S^(-1/2) U^T T2 V S^(-1/2)``.
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
    """Data-driven SSI by LQ factorization of the block Hankel matrix.

    The projection of future outputs onto past outputs is ``L21 Q1^T``, so
    only ``L21`` is decomposed. Weightings: ``UPC`` (default), ``PC`` and
    ``CVA``. UPC and PC give the same poles, because the right weighting does
    not change the left singular vectors.
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


# --- eigenvalues to modal parameters, and per-pole indicators ---

def _rcmul(A, B):
    """Real matrix times complex matrix, as two real products (faster here)."""
    return A @ B.real + 1j * (A @ B.imag)


def modal_from_AC(A, C, dt, O=None):
    """Frequencies, damping ratios, shapes and indicators from ``A``, ``C``.

    ``lambda = ln(mu)/dt``, ``f = |lambda|/2pi`` and
    ``zeta = -Re(lambda)/|lambda|``; shapes at the sensors are ``C psi``.
    Only the member of each conjugate pair with positive imaginary part is
    kept; real eigenvalues are dropped.
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
    """Consistency of each observability column with a geometric sequence.

    For exact ``(A, C)`` the column ``O psi_k`` equals ``[C psi_k,
    mu_k C psi_k, mu_k^2 C psi_k, ...]``. Returns the modulus of the complex
    correlation between the two, from 0 to 1. Unlike MAC and MPC, it is
    defined for a single channel. The exponent of ``mu^k`` is floored at -300
    to avoid slow subnormal arithmetic; the terms it drops are negligible.
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
    """Modal phase collinearity, ``((l1 - l2)/(l1 + l2))^2``.

    ``l1 >= l2`` are the eigenvalues of the 2 by 2 scatter matrix of the real
    and imaginary parts, taken about the origin rather than the mean so that
    the value is informative from two channels. Equal to 1 for one channel.
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


# --- stabilization diagram ---

def stabilisation(est, orders=None, band=BAND, route="shift",
                  tol_f=TOL_F, tol_z=TOL_Z, tol_mac=TOL_MAC):
    """Poles at every model order, graded against the order below.

    Returns a DataFrame with one row per pole per order and the columns
    ``order, f, zeta, emac, mpc, stab_f, stab_z, stab_mac, stable``. Only
    conjugate poles with ``ZETA_MIN < zeta < ZETA_MAX`` inside ``band`` are
    kept. Each is matched to the nearest pole in frequency at the previous
    order and flagged stable if ``|df|/f < tol_f``, ``|dzeta|/zeta < tol_z``
    and ``1 - MAC < tol_mac``.
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

    The distance is
    ``d(p, q) = |f_p - f_q| / max(f_p, f_q) + (1 - MAC(phi_p, phi_q))``
    under average linkage, cut at ``CLUST_CUT``. With one channel the MAC
    term vanishes and the cut is ``CLUST_CUT_1CH`` on frequency alone. A
    cluster is kept if its poles come from at least ``min_order_frac`` of the
    distinct swept orders; this threshold was set on synthetic systems only.
    With ``compact=True`` the median absolute deviations of frequency and
    damping must also be within ``TOL_F`` and ``TOL_Z``. This is off by
    default because real poles on the bridge scatter by about 10 percent in
    damping. The representative frequency and damping are cluster medians.
    ``attrs["n_clusters_raw"]`` counts the clusters that passed persistence.
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


# --- synthetic state-space model with prescribed modal parameters ---

class ModalSystem:
    """Discrete stochastic system with prescribed frequencies and damping.

    Each mode is an exactly discretized oscillator in the state ``[q, q']``,
    driven by white noise and observed as displacement through prescribed
    mode shapes, so its modal parameters are known exactly.
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
        """``R_k`` from the discrete Lyapunov solution, with no data.

        ``Sigma = A Sigma A^T + Q`` with ``Q = B diag(q^2) B^T``, then
        ``R_0 = C Sigma C^T`` and ``R_k = C A^(k-1) G`` with
        ``G = A Sigma C^T``.
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


# --- verification ---

def verify(dur=600.0, verbose=True):
    """Six checks against answers known independently of this code.

    [1] exact correlations return the model's own poles; [2] a well-separated
    pair from a finite record; [3] shift invariance against the Toeplitz
    realization; [4] SSI-COV against SSI-DATA; [5] the bridge away from the
    crossing; [6] decimation to 20 Hz against the raw 100 Hz record.
    """
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
            # a mode not returned counts as a miss, not as a large error
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
        print("\n[3] SHIFT INVARIANCE vs TOEPLITZ REALIZATION, and\n"
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

    # -- [5] the example bridge away from the crossing ----------------------
    if verbose:
        print("\n[5] THE EXAMPLE BRIDGE AWAY FROM THE CROSSING")
    for T, lab in ((1.30e5, "detuned low"), (1.75e5, "detuned high")):
        s5 = RecordSimulator(T, 0.005)
        r5 = s5.record(duration=dur, snr_db=20.0, seed=1)
        ft = np.array([s5.f_lo, s5.f_hi])
        zt = np.array([s5.zj[s5.pair_idx[0]], s5.zj[s5.pair_idx[1]]])
        # stay energy fraction, and mode-shape amplitude at the stay sensor
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
        # same order step in every row; rows differ in rate and highest order
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
    """Largest relative difference of two equal-length vectors, else NaN."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b) or not len(a):
        return np.nan
    return float(np.max(np.abs(a - b) / np.abs(a)))


def _fmt(v):
    if isinstance(v, float):
        return "%.6g" % v
    return str(v)


# --- the isolated-cable inversion, the two laws, and peak picking ---

def eps_from_freq(f_hat, f_iso):
    """Relative tension error of the isolated-cable inversion.

    With ``T = 4 m L^2 f^2 / n^2`` the error of a reading ``f_hat`` is
    ``(f_hat/f_iso)^2 - 1``, the quantity that ``run_merged.picked_error``,
    ``eps_branch_law`` and ``eps_merged_law`` also return.
    """
    return (f_hat / f_iso) ** 2 - 1.0


def eps_branch_law(d, s):
    """Signed stay-dominated branch error, ``sign(d) (hypot(d, s) - |d|)``."""
    sg = 1.0 if d >= 0 else -1.0
    return sg * (np.hypot(d, s) - abs(d))


def peak_pick(y, fs, band=FBAND, nperseg=8192, prom_db=3.0, smooth=3,
               window=None):
    """Frequency read off the stay-channel spectrum as its largest peak.

    Welch spectrum, smoothed over ``smooth`` bins by a Hann kernel so that
    noise wiggles are not counted as peaks, then peaks above ``prom_db`` of
    prominence inside ``band``. Returns ``(f_peak, n_band, f_peaks,
    n_window)``: ``n_band`` counts every peak in the band and ``n_window``
    only those inside ``window``, which the caller sets around the pair.
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
    """Output channels of one instrument configuration.

    ``stay``: one accelerometer on the stay. ``stay+deck``: adds a deck
    channel at the anchorage. ``stay2``: two accelerometers on the same stay,
    at 2 m and mid-chord (the DOFs are set in ``_simulator``).
    """
    if config == "stay":
        return rec["a_stay"][None, :]
    if config == "stay+deck":
        return np.vstack([rec["a_stay"], rec["a_deck"]])
    if config == "stay2":
        return np.vstack([rec["a_stay"], rec["a_deck"]])
    raise ValueError(config)


def _simulator(T, zeta, config, cache={}):
    """Cached simulator for one tension, damping ratio and configuration.

    For ``stay2`` the second output DOF moves from the deck anchorage to
    mid-chord on the stay.
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


# --- study 1: outcome against maximum correlation lag ---

def lag_study(zetas=(0.005, 0.02, 0.03), taus=(2.0, 4.0, 8.0, 12.0, 16.0,
                                              24.0, 32.0),
              seeds=(0, 1, 2), dur=600.0, snr_db=20.0, verbose=True):
    """Whether the pair separates, against the maximum correlation lag.

    Run at the exact crossing. A stay-only instrument is expected to need a
    lag near one beat period ``1/(f_hi - f_lo)``; with a deck channel the two
    hybrid modes differ in shape and can separate at any lag.
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


# --- study 2: through the crossing ---

TENSIONS = (1.430e5, 1.470e5, 1.516e5, 1.560e5, 1.610e5)
ZETAS = (0.002, 0.005, 0.010, 0.020, 0.025, 0.030, 0.040)
CONFIGS = ("stay", "stay2", "stay+deck")
U_SPLIT = np.sqrt(np.sqrt(5.0) - 2.0)      # 0.48587; no dip below this u


def _psd_at(y, fs, freqs, nperseg=8192):
    """Welch density of the stay channel at given frequencies."""
    f, P = welch(y, fs=fs, nperseg=min(nperseg, len(y)))
    return np.array([P[int(np.argmin(np.abs(f - ff)))] for ff in freqs])


def crossing_study(tensions=TENSIONS, zetas=ZETAS, seeds=(0, 1, 2),
                   configs=CONFIGS, methods=("cov", "data"),
                   taus=(4.0, 16.0), dur=600.0, snr_db=20.0, verbose=True):
    """Identification results through the crossing.

    For each tension, damping ratio, seed, instrument configuration, method
    and correlation lag: the poles in the band, whether they match the two
    true branches, and the tension error of three readings, beside the branch
    law and the merged-peak law:

        eps_ssi_amp     SSI pole with the larger spectral density at the stay
        eps_ssi_oracle  SSI pole nearest the true stay-dominated branch
        eps_peak        largest peak of the stay-channel spectrum
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
    """Tabulate the poles found at the exact crossing; return those rows."""
    ok = df[df["failed"] == ""]
    tune = ok[np.abs(ok["d"]) < 0.005]           # the exact crossing
    if verbose:
        print("\n" + "=" * 78)
        print("AT THE CROSSING: poles found in the band")
        print("=" * 78)
        print("  for u = s/2zeta below %.4f the spectrum shows one peak.\n"
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
    """Whether the pair appears in the diagram at all, before selection.

    Returns the number of swept orders at which two distinct poles sit within
    ``tol`` of the two true branches, the first and last such order, and the
    longest run of consecutive such orders. Stability is ignored.
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
    """The merged regime at the exact crossing, twelve seeds per case.

    Damping ratios at which the stay spectrum shows one maximum under the
    3 dB prominence rule (1.0 percent and above on this bridge). Per case:
    whether both branches were returned within 1 percent, whether the pair
    appears in the diagram, and the tension error of the SSI pole with the
    larger spectral density at the stay channel.
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
        print("\nTHE MERGED REGIME: fraction of records returning both "
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

    A property of the mode shapes; no records are simulated. Returns the
    ratio of lower- to upper-branch shape amplitude at each station beside
    the stay kinetic-energy fractions. Near the anchorage the two orderings
    can differ when ``|d| < s``.
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
        print("\n  a ratio above 1 means the lower branch is read larger at that station")
    return df


def stab_diagram_dump(T=T_TUNE, zeta=0.03, seed=0, dur=600.0, snr_db=20.0,
                      taus=(4.0, 16.0), configs=("stay", "stay+deck"),
                      method="cov"):
    """One full stabilization diagram per configuration and lag, for plotting.

    The default damping of 3 percent is inside the merged regime.
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
        print("  checks against known answers; no external OMA package "
              "was available\n  for a cross-check (see the "
              "module docstring).")
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

    # 3600 s records; not part of --all
    if a.longrec:
        print("\n" + "=" * 78)
        print("THE MERGED REGIME ON AN HOUR-LONG RECORD")
        print("=" * 78)
        print("  600 s is a typical ambient stay record length.  If the\n"
              "  pair separates at 3600 s and not at 600 s, the limit is set\n"
              "  by the record length and not by the method.")
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
