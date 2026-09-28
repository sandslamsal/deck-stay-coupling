# -*- coding: utf-8 -*-
"""Independent cross-check of the coupled model against OpenSees.

``verify_coupling.py`` checks the assembly against an exact solution, but it
checks the model on the model's own terms: reduced planar kinematics, in
which the stay enters the deck as a vertical spring ``(EA/L) sin^2(theta)``
and a transverse tie ``v_bottom = cos(theta) w_deck``.  That reduction could
be self-consistent and still be the wrong idealisation of a real inclined
stay.

This file builds the same bridge in OpenSees with TRUE TWO-DIMENSIONAL
GEOMETRY.  The stay is a chain of corotational truss elements running from
the pylon top down to the deck at its real inclination, sharing a node with
the deck, carrying its own distributed mass.  Nothing is reduced: the axial
and transverse paths both arise from the geometry, and the tension enters
through the corotational tangent rather than through an assumed geometric
stiffness matrix.  If the reduced model is a fair idealisation the two
spectra agree; if the ``sin^2`` and ``cos`` reduction is wrong, they do not.

OpenSees is an independent implementation by an independent group, which is
the property that makes the check worth anything.

Run:  python3 scripts/cross_check_opensees.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from cablefe import CableDeck  # noqa: E402

import openseespy.opensees as ops  # noqa: E402


def opensees_model(Ld, EId, md, Lc, EIc, mc, T, EA, theta,
                   nd=48, nc=48, Ad=1.0, nmodes=14):
    """The same bridge, true 2D geometry, corotational stay.

    Deck axial motion is restrained so the comparison isolates the bending
    and stay dynamics, which is the only part the reduced model represents.
    Everything else is left to the geometry.
    """
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    Ed = EId / 1.0                     # unit Iz, so EI = Ed * 1.0
    Iz = 1.0

    # ---- deck nodes ----------------------------------------------------
    for i in range(nd + 1):
        ops.node(i + 1, i * Ld / nd, 0.0)
    ia = nd // 2                       # anchorage at midspan node index
    anchor_tag = ia + 1

    ops.geomTransf('Linear', 1)
    for e in range(nd):
        ops.element('elasticBeamColumn', e + 1, e + 1, e + 2,
                    Ad, Ed, Iz, 1, '-mass', md, '-cMass')

    # simply supported, and axial motion removed so the reduced model's
    # transverse-only deck is the thing being compared
    ops.fix(1, 1, 1, 0)
    ops.fix(nd + 1, 1, 1, 0)
    for i in range(2, nd + 1):
        ops.fix(i, 1, 0, 0)

    # ---- stay ----------------------------------------------------------
    # from the deck anchorage up to the pylon at the true inclination
    x0, y0 = ia * Ld / nd, 0.0
    xt, yt = x0 - Lc * np.cos(theta), Lc * np.sin(theta)

    cable_first = 1000
    for j in range(nc + 1):
        t = j / nc
        ops.node(cable_first + j, x0 + t * (xt - x0), y0 + t * (yt - y0))
    # node j=0 coincides with the deck anchorage: tie it to the deck node
    ops.equalDOF(anchor_tag, cable_first, 1, 2)
    ops.fix(cable_first + nc, 1, 1, 1)          # pylon top held

    # rotational DOFs of the truss chain carry no stiffness. The pylon node
    # is already fully fixed above, so it is skipped.
    for j in range(nc):
        ops.fix(cable_first + j, 0, 0, 1)

    Ac = 1.0
    Ec = EA / Ac
    eps0 = T / EA                                # initial strain gives T
    ops.uniaxialMaterial('Elastic', 100, Ec)
    ops.uniaxialMaterial('InitStrainMaterial', 101, 100, eps0)

    for j in range(nc):
        ops.element('corotTruss', 2000 + j, cable_first + j,
                    cable_first + j + 1, Ac, 101, '-rho', mc, '-cMass', 1)

    # ---- balance the prestress ----------------------------------------
    # The stay pulls the anchorage up with T sin(theta). On a real bridge
    # dead load reacts that; with nothing to react it the deck simply lifts
    # and the tension bleeds away, which is not the state the reduced model
    # linearises about. The balancing load holds the reference geometry so
    # the tangent is taken at tension T, as intended.
    ops.timeSeries('Constant', 1)
    ops.pattern('Plain', 1, 1)
    ops.load(anchor_tag, 0.0, -T * np.sin(theta), 0.0)

    # ---- equilibrate under the prestress, then take the tangent --------
    ops.system('BandGeneral')
    ops.numberer('RCM')
    ops.constraints('Transformation')
    ops.integrator('LoadControl', 1.0)
    ops.algorithm('Newton')
    ops.analysis('Static')
    ops.test('NormDispIncr', 1e-10, 200, 0)
    okflag = ops.analyze(1)

    achieved = ops.basicForce(2000)[0]
    uplift = ops.nodeDisp(anchor_tag, 2)
    vals = ops.eigen('-fullGenLapack', nmodes)
    f = np.sqrt(np.abs(np.array(vals))) / (2.0 * np.pi)
    return np.sort(f), okflag, achieved, uplift


def main():
    BRIDGE = dict(Ld=80.0, EId=2.0e9, md=1000.0,
                  Lc=25.0, EIc=1.2e4, mc=5.5, EA=1.4e8,
                  theta=np.deg2rad(35.0), nd=48, nc=48)

    print("=" * 74)
    print("Cross-check against OpenSees, true 2D geometry")
    print("=" * 74)
    print("  reduced model : stay as (EA/L)sin^2(theta) spring + cos(theta) tie")
    print("  OpenSees      : inclined corotational truss chain, shared node,")
    print("                  tension through the corotational tangent")
    print()

    for T in (200e3, 400e3, 700e3):
        cd = CableDeck(T=T, **BRIDGE)
        f_mine, _ = cd.modes(14)

        f_ops, okflag, achieved, uplift = opensees_model(
            T=T, nmodes=14, **{**BRIDGE, 'nc': 200})
        if okflag != 0:
            print(f"  T={T/1e3:.0f} kN: OpenSees static step did not converge")
            continue

        n = 8
        a, b = np.array(f_mine[:n]), np.array(f_ops[:n])
        err = 100.0 * (a - b) / b
        print(f"  T = {T/1e3:.0f} kN   (OpenSees achieved "
              f"{achieved/1e3:.1f} kN, deck uplift {uplift:+.2e} m)")
        print(f"    {'mode':>5s} {'reduced':>11s} {'OpenSees':>11s} {'diff':>9s}")
        for i, (x, y, e) in enumerate(zip(a, b, err), start=1):
            print(f"    {i:5d} {x:9.4f} Hz {y:9.4f} Hz {e:+8.3f} %")
        print(f"    worst |diff| over the first {n} modes: "
              f"{np.abs(err).max():.3f} %")
        print()


if __name__ == "__main__":
    main()
