# -*- coding: utf-8 -*-
"""Identification of stay tension from a coupled record, four ways.

The finding this file exists to establish is not that one estimator beats
another.  It is that **the bias lives in the physics, not in the estimator**.
Any method that assumes the stay is an isolated element inherits the bias,
however sophisticated the fitting machinery, and a method that carries the
coupled boundary condition removes it, however simple.

Four estimators are compared on identical data:

  1. ``invert_string``          the incumbent, in cablefe.py
  2. ``pinn_identify(coupled=False)``  a physics-informed network whose
     residual is the ISOLATED tensioned beam with ``v(L) = 0``.  This is the
     assumption behind the published PINN work on cable force.
  3. ``fit_closed_form``        least squares on the closed-form coupled
     model derived in this study, using the pattern of perturbation across
     mode orders to separate tension from coupling
  4. ``pinn_identify(coupled=True)``   the same network with the correct
     Robin end condition and an unknown deck impedance

The boundary condition is the whole argument.  Writing the cable mode as
``V(x)`` with the anchorage at ``x = L``, the deck reacts through

    T V'(L) + Z V(L) = 0,        Z = (K - M omega^2) / c^2

an impedance that is infinite for a rigid anchorage, recovering ``V(L) = 0``,
and finite for a real deck.  Estimator 2 asserts the first; estimators 3 and
4 identify the second.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
# 3. closed-form coupled model, fitted by least squares
# ---------------------------------------------------------------------------

def predict_coupled_freqs(orders, T, EI, L, m, f_deck, kappa):
    """Stay-branch frequencies of the coupled system, from the closed form.

    The isolated frequency is displaced away from the deck frequency it
    approaches by half the avoided-crossing gap,

        shift_n = sign(D_n) * 0.5 * (sqrt(D_n^2 + S_n^2) - |D_n|)

    with ``D_n`` the absolute detuning and ``S_n = (2/(n pi)) kappa f_n`` the
    split, ``kappa = cos(theta) sqrt(mu_eff)`` gathering the coupling into one
    unknown.  The ``1/n`` is what makes the set of orders informative: the
    perturbation has a known shape across ``n``, so tension and coupling
    separate.
    """
    orders = np.asarray(orders, dtype=float)
    kn = orders * np.pi / L
    f_iso = np.sqrt(orders ** 2 * (T + EI * kn ** 2) / (4.0 * m * L ** 2))
    D = f_iso - f_deck
    S = (2.0 / (orders * np.pi)) * kappa * f_iso
    shift = np.sign(D) * 0.5 * (np.sqrt(D ** 2 + S ** 2) - np.abs(D))
    return f_iso + shift


def fit_closed_form(f_obs, orders, L, m, EI=0.0, fit_EI=False):
    """Identify tension from observed stay-branch frequencies.

    Unknowns: tension, the nearby deck frequency, and the coupling amplitude
    ``kappa``.  Bending stiffness is optional and off by default, because on
    a real stay it is usually known from the strand schedule and leaving it
    free trades one bias for another.

    Returns ``(T, info)``.
    """
    f_obs = np.asarray(f_obs, dtype=float)
    orders = np.asarray(orders, dtype=float)

    # initial guess: the incumbent inversion on the least-perturbed order,
    # taken as the highest available since the split falls as 1/n
    j = int(np.argmax(orders))
    T0 = 4.0 * m * L ** 2 * f_obs[j] ** 2 / orders[j] ** 2
    fd0 = float(np.median(f_obs))
    p0 = [np.log(T0), fd0, 0.02]
    if fit_EI:
        p0.append(np.log(max(EI, 1.0)))

    def resid(p):
        T = np.exp(p[0])
        fd = p[1]
        kap = abs(p[2])
        ei = np.exp(p[3]) if fit_EI else EI
        return predict_coupled_freqs(orders, T, ei, L, m, fd, kap) - f_obs

    sol = least_squares(resid, p0, method="lm", max_nfev=20000)
    T = float(np.exp(sol.x[0]))
    return T, dict(f_deck=float(sol.x[1]), kappa=float(abs(sol.x[2])),
                   cost=float(sol.cost), success=bool(sol.success))


# ---------------------------------------------------------------------------
# 2 and 4. physics-informed network on the cable, isolated or coupled
# ---------------------------------------------------------------------------

def pinn_identify(x_data, v_data, omega, L, m, EI, coupled=True,
                  T_init=None, epochs=4000, width=32, depth=3, seed=0,
                  lr=5e-3, device="cpu", verbose=False, lbfgs=600):
    """Identify tension from a sampled mode shape by a physics-informed net.

    The network represents the mode shape ``V(x)`` on the stay.  The residual
    is the free-vibration equation of a tensioned beam,

        EI V'''' - T V'' - m omega^2 V = 0

    with ``T`` a trainable parameter.  ``omega`` is the measured circular
    frequency of the mode.  ``x_data``/``v_data`` are the sensor positions
    and the mode shape amplitudes read there.

    ``coupled=False`` clamps ``V(L) = 0``, the isolated assumption every
    published cable-force PINN carries.  ``coupled=True`` replaces it with the
    Robin condition ``T V'(L) + Z V(L) = 0`` and makes ``Z`` trainable, so the
    deck's reaction is identified rather than assumed away.

    Returns ``(T, info)``.
    """
    import torch

    # These networks are tiny. Multi-threaded BLAS spends more time on
    # thread handoff than on arithmetic, and on this machine the parallel
    # build ran roughly an order of magnitude slower than a single thread.
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    dev = torch.device(device)
    dt = torch.float64
    torch.set_default_dtype(dt)

    xd = torch.tensor(np.asarray(x_data, float) / L, dtype=dt, device=dev
                      ).reshape(-1, 1)
    vd = torch.tensor(np.asarray(v_data, float), dtype=dt, device=dev
                      ).reshape(-1, 1)
    # scale the shape so the data term is O(1) regardless of modal scaling
    vscale = float(torch.max(torch.abs(vd)))
    vd = vd / vscale

    if T_init is None:
        T_init = 4.0 * m * L ** 2 * (omega / (2 * np.pi)) ** 2
    logT = torch.tensor(np.log(T_init), dtype=dt, device=dev,
                        requires_grad=True)
    # The impedance Z = (K - M omega^2)/c^2 CHANGES SIGN through a crossing:
    # positive below the deck frequency, negative above it. An exponential
    # parameterisation cannot represent that and pins the fit on the wrong
    # branch, so Z is carried as a signed variable scaled by T/L.
    Zscale = T_init / L
    zpar = torch.tensor(1.0, dtype=dt, device=dev, requires_grad=True)

    layers, d_in = [], 1
    for _ in range(depth):
        layers += [torch.nn.Linear(d_in, width), torch.nn.Tanh()]
        d_in = width
    layers += [torch.nn.Linear(d_in, 1)]
    net = torch.nn.Sequential(*layers).to(dev).to(dt)

    params = list(net.parameters()) + [logT]
    if coupled:
        params.append(zpar)
    opt = torch.optim.Adam(params, lr=lr)

    xc = torch.linspace(0.0, 1.0, 121, dtype=dt, device=dev
                        ).reshape(-1, 1).requires_grad_(True)
    x0 = torch.zeros(1, 1, dtype=dt, device=dev, requires_grad=True)
    xL = torch.ones(1, 1, dtype=dt, device=dev, requires_grad=True)

    def d(y, x, k=1):
        for _ in range(k):
            y = torch.autograd.grad(y, x, torch.ones_like(y),
                                    create_graph=True)[0]
        return y

    hist = []
    for ep in range(epochs):
        opt.zero_grad()
        T = torch.exp(logT)

        V = net(xc)
        # x is normalised by L, so each derivative carries a 1/L
        V2 = d(V, xc, 2) / L ** 2
        V4 = d(V, xc, 4) / L ** 4
        res = EI * V4 - T * V2 - m * omega ** 2 * V
        # normalise the residual by the tension term so the loss is
        # dimensionless and does not simply drive T to zero
        scale = T * torch.mean(torch.abs(V2)) + 1e-30
        loss_pde = torch.mean((res / scale) ** 2)

        V0 = net(x0)
        VL = net(xL)
        VLp = d(VL, xL, 1) / L
        loss_bc = V0[0, 0] ** 2
        if coupled:
            Z = zpar * Zscale
            bc = (T * VLp + Z * VL) / (T / L + 1e-30)
            loss_bc = loss_bc + bc[0, 0] ** 2
        else:
            loss_bc = loss_bc + VL[0, 0] ** 2

        loss_dat = torch.mean((net(xd) - vd) ** 2)
        loss = loss_dat + 1e-2 * loss_pde + 1.0 * loss_bc
        loss.backward()
        opt.step()
        if verbose and ep % 500 == 0:
            hist.append((ep, float(loss), float(torch.exp(logT))))
            print(f"      ep {ep:5d}  loss {float(loss):.3e}  "
                  f"T {float(torch.exp(logT))/1e3:.1f} kN")

    # Adam gets close; a quasi-Newton polish is what actually converges a
    # PINN of this size, and without it the comparison would be measuring
    # optimiser stopping rather than physics.
    if lbfgs:
        optl = torch.optim.LBFGS(params, max_iter=lbfgs, line_search_fn="strong_wolfe",
                                 tolerance_grad=1e-12, tolerance_change=1e-14)

        def closure():
            optl.zero_grad()
            T = torch.exp(logT)
            V = net(xc)
            V2 = d(V, xc, 2) / L ** 2
            V4 = d(V, xc, 4) / L ** 4
            res = EI * V4 - T * V2 - m * omega ** 2 * V
            scale = T * torch.mean(torch.abs(V2)) + 1e-30
            lp = torch.mean((res / scale) ** 2)
            V0 = net(x0); VL = net(xL); VLp = d(VL, xL, 1) / L
            lb = V0[0, 0] ** 2
            if coupled:
                Z = zpar * Zscale
                lb = lb + ((T * VLp + Z * VL) / (T / L + 1e-30))[0, 0] ** 2
            else:
                lb = lb + VL[0, 0] ** 2
            ld = torch.mean((net(xd) - vd) ** 2)
            l = ld + 1e-2 * lp + 1.0 * lb
            l.backward()
            return l

        loss = optl.step(closure)

    return float(np.exp(logT.detach().cpu().numpy())), dict(
        Z=float(zpar.detach().cpu().numpy()) * Zscale if coupled else np.inf,
        loss=float(loss), hist=hist)


# ---------------------------------------------------------------------------
# 5. the PINN done right: hard pylon condition, NOTHING assumed at the
#    anchorage, tension-independent residual scale
# ---------------------------------------------------------------------------

def pinn_identify_free(x_data, v_data, omega, L, m, EI, T_init=None,
                       epochs=6000, width=32, depth=3, seed=0,
                       lr=5e-3, lbfgs=600):
    """The repaired network.  Three changes from ``pinn_identify``:

    1. ``V(0) = 0`` is enforced HARD, by construction ``V = x_hat N(x_hat)``,
       instead of through a penalty.
    2. NOTHING is imposed at the anchorage.  The false clamp ``V(L) = 0`` of
       the isolated variant and the soft Robin penalty of the coupled variant
       are both gone; the data and the interior residual determine the end
       behaviour, and the impedance can be read off the fitted shape
       afterwards.
    3. The residual is normalised by the inertia term ``m omega^2 |V|``,
       which does not contain the trainable tension.  The earlier
       normalisation by ``T |V''|`` put the unknown in its own loss scale.

    With these the network is doing, by gradient descent, what
    ``shapefit.fit_shape`` does in closed form: finding the wavenumber the
    measured shape actually has and reading the tension off the dispersion
    relation.  It exists to show the two agree, so the repair cannot be
    attributed to the machinery.
    """
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    dt = torch.float64
    torch.set_default_dtype(dt)

    xd = torch.tensor(np.asarray(x_data, float) / L, dtype=dt).reshape(-1, 1)
    vd = torch.tensor(np.asarray(v_data, float), dtype=dt).reshape(-1, 1)
    vd = vd / float(torch.max(torch.abs(vd)))

    if T_init is None:
        T_init = 4.0 * m * L ** 2 * (omega / (2 * np.pi)) ** 2
    logT = torch.tensor(np.log(T_init), dtype=dt, requires_grad=True)

    layers, d_in = [], 1
    for _ in range(depth):
        layers += [torch.nn.Linear(d_in, width), torch.nn.Tanh()]
        d_in = width
    layers += [torch.nn.Linear(d_in, 1)]
    net = torch.nn.Sequential(*layers).to(dt)

    def V_of(xh):
        return xh * net(xh)          # V(0) = 0 by construction

    params = list(net.parameters()) + [logT]
    opt = torch.optim.Adam(params, lr=lr)
    xc = torch.linspace(0.0, 1.0, 121, dtype=dt).reshape(-1, 1
                                                         ).requires_grad_(True)

    def d(y, x, kk=1):
        for _ in range(kk):
            y = torch.autograd.grad(y, x, torch.ones_like(y),
                                    create_graph=True)[0]
        return y

    def losses():
        T = torch.exp(logT)
        V = V_of(xc)
        V2 = d(V, xc, 2) / L ** 2
        V4 = d(V, xc, 4) / L ** 4
        res = EI * V4 - T * V2 - m * omega ** 2 * V
        # detached: the scale is a normalisation, not a quantity the
        # optimiser should be able to reduce by inflating the shape
        scale = (m * omega ** 2
                 * torch.mean(torch.abs(V)).detach() + 1e-30)
        loss_pde = torch.mean((res / scale) ** 2)
        loss_dat = torch.mean((V_of(xd) - vd) ** 2)
        return loss_dat + 0.1 * loss_pde

    for _ in range(epochs):
        opt.zero_grad()
        loss = losses()
        loss.backward()
        opt.step()

    if lbfgs:
        optl = torch.optim.LBFGS(params, max_iter=lbfgs,
                                 line_search_fn="strong_wolfe",
                                 tolerance_grad=1e-12,
                                 tolerance_change=1e-14)

        def closure():
            optl.zero_grad()
            l = losses()
            l.backward()
            return l

        loss = optl.step(closure)

    # end impedance read off the trained shape, a by-product not an unknown
    xL = torch.ones(1, 1, dtype=dt, requires_grad=True)
    VL = V_of(xL)
    VLp = d(VL, xL, 1) / L
    T = float(np.exp(logT.detach().numpy()))
    Z = (-T * float(VLp.detach()) / float(VL.detach())
         if abs(float(VL.detach())) > 1e-9 else np.inf)
    return T, dict(Z=Z, loss=float(loss))
