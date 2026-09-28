# -*- coding: utf-8 -*-
"""Tension from a measured mode shape, with no assumption at the anchorage.

For each trial tension T, the wavenumbers k and p of the tensioned beam
EI V'''' - T V'' = m omega^2 V follow from the dispersion relation, and the
coefficients of V(x) = A sin(kx) + B cos(kx) + C e^{-px} + D e^{p(x-L)} are a
linear least-squares fit to the sampled shape. The tension that minimizes the
misfit is returned, with the end impedance Z = -T V'(L)/V(L) as a by-product.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def dispersion(T, EI, m, omega):
    """Propagating and evanescent wavenumbers of the tensioned beam."""
    if EI <= 0.0:
        return omega * np.sqrt(m / max(T, 1e-12)), np.inf
    disc = np.sqrt(T * T + 4.0 * EI * m * omega ** 2)
    k = np.sqrt(max((disc - T) / (2.0 * EI), 1e-30))
    p = np.sqrt((disc + T) / (2.0 * EI))
    return k, p


def _basis(x, L, k, p):
    cols = [np.sin(k * x), np.cos(k * x)]
    if np.isfinite(p):
        # evanescent terms decay from each end, so nothing overflows for large p L
        cols += [np.exp(-p * x), np.exp(p * (x - L))]
    return np.column_stack(cols)


def _misfit(T, x, v, L, m, EI, omega):
    k, p = dispersion(T, EI, m, omega)
    A = _basis(x, L, k, p)
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    r = A @ coef - v
    return float(r @ r), coef


def fit_shape(x, v, omega, L, m, EI, n_hint=1, bracket=(0.25, 4.0),
              ngrid=400):
    """Identify tension from sensor positions ``x`` and shape samples ``v``.

    ``omega`` is the measured circular frequency of the same mode and
    ``n_hint`` the apparent mode order, used only to center the search
    bracket via the string formula.  Returns ``(T, info)`` where ``info``
    carries the fitted coefficients, the wavenumber, and the end impedance
    ratio ``Z = -T V'(L)/V(L)`` (np.inf where V(L) is numerically zero,
    which is the isolated-stay limit).
    """
    x = np.asarray(x, float)
    v = np.asarray(v, float)
    vn = v / np.max(np.abs(v))

    f = omega / (2.0 * np.pi)
    T0 = 4.0 * m * L ** 2 * f ** 2 / n_hint ** 2
    Ts = np.geomspace(bracket[0] * T0, bracket[1] * T0, ngrid)
    J = np.array([_misfit(t, x, vn, L, m, EI, omega)[0] for t in Ts])
    i = int(np.argmin(J))
    if i in (0, ngrid - 1):
        # A minimum on the bracket edge means n_hint does not match the
        # supplied mode; the bounded refinement would return a wrong tension.
        raise ValueError(
            "fit_shape: minimum on the bracket edge (n_hint=%d); widen "
            "`bracket` or pass the mode order actually measured" % n_hint)
    lo, hi = Ts[i - 1], Ts[i + 1]
    sol = minimize_scalar(
        lambda t: _misfit(t, x, vn, L, m, EI, omega)[0],
        bounds=(lo, hi), method="bounded",
        options=dict(xatol=1e-8 * Ts[i]))
    T = float(sol.x)

    k, p = dispersion(T, EI, m, omega)
    _, coef = _misfit(T, x, vn, L, m, EI, omega)
    VL = coef[0] * np.sin(k * L) + coef[1] * np.cos(k * L)
    VLp = coef[0] * k * np.cos(k * L) - coef[1] * k * np.sin(k * L)
    if np.isfinite(p):
        VL += coef[2] * np.exp(-p * L) + coef[3]
        VLp += -coef[2] * p * np.exp(-p * L) + coef[3] * p
    Z = -T * VLp / VL if abs(VL) > 1e-9 else np.inf
    return T, dict(k=k, p=p, coef=coef, V_L=float(VL), Z=float(Z),
                   misfit=float(sol.fun))
