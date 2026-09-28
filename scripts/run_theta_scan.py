# -*- coding: utf-8 -*-
"""Effect of stay inclination theta on the splitting at a crossing.

For a straight stay (g = 0) on the example bridge, holds everything but theta
and compares the finite-element minimum separation s_FE with the closed form
s_law for stay orders 1 and 2. Inclination enters through cos(theta) and
through the anchorage spring (EA / L_c) sin^2(theta), which stiffens the host
and changes phi_a. Writes data/theta_scan.csv.

Run:  python3 scripts/run_theta_scan.py
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from cablefe2d import CableDeck2D, split_one_ended       # noqa: E402

BRIDGE = dict(Ld=80.0, EId=2.0e9, md=1000.0,
              Lc=25.0, EIc=1.2e4, mc=5.5, EA=1.4e8,
              nd=40, nc=60)
THETAS_DEG = (0.0, 10.0, 20.0, 30.0, 35.0, 45.0, 55.0, 65.0)
ORDERS = (1, 2)


def traverse(make, Ts, n):
    """Minimum relative separation of the two lines nearest stay order n
    along a tension traverse; returns (s, T_at, model). Same rule as
    run_sag.traverse."""
    best = (np.inf, None, None)
    for T in Ts:
        cd = make(T)
        f, _ = cd.modes(40)
        fi = cd.stay_alone(n + 1)[n - 1]
        inb = np.where((f > 0.80 * fi) & (f < 1.25 * fi))[0]
        if len(inb) < 2:
            continue
        ff = f[inb]
        k = int(np.argmin(np.diff(ff)))
        gap = (ff[k + 1] - ff[k]) / (0.5 * (ff[k] + ff[k + 1]))
        if gap < best[0]:
            best = (gap, T, cd)
    return best


def host_ordinate(cd, f_target):
    fh, pa, _ = cd.host_alone(16)
    j = int(np.argmin(np.abs(fh - f_target)))
    return fh[j], pa[j]


rows = []
print("=" * 78)
print("STRAIGHT STAY: WHAT INCLINATION CHANGES")
print("=" * 78)
print("Deck, anchorage, stay length, mass and stiffnesses all held; only")
print("theta varies. A straight stay's own frequency does not depend on")
print("theta, so the crossing sits at the same tension throughout.")
print()
for n in ORDERS:
    print("-- stay mode order %d" % n)
    print("  %7s %10s %10s %8s %9s %9s %9s" %
          ("theta", "s_FE(%)", "s_law(%)", "FE/law", "phi_a/0", "f_host", "s/s0 /cos"))
    s0 = None
    for th_deg in THETAS_DEG:
        th = np.deg2rad(th_deg)
        make = lambda T, th=th: CableDeck2D(T=T, g=0.0, theta=th, **BRIDGE)
        # bracket the crossing: the stay's frequency is theta-independent
        probe = CableDeck2D(T=2.0e5, g=0.0, theta=th, **BRIDGE)
        fi = probe.stay_alone(n + 1)[n - 1]
        fh, _ = host_ordinate(probe, fi)
        # T scales as f^2 for a taut stay; center the tension range on the host
        T_c = 2.0e5 * (fh / fi) ** 2
        Ts = np.linspace(0.80 * T_c, 1.25 * T_c, 90)
        s_fe, T_at, cd = traverse(make, Ts, n)
        if cd is None:
            print("   %6.1f  no crossing found in the window" % th_deg)
            continue
        M_s = 0.5 * cd.mc * cd.Lc
        f_iso = cd.stay_alone(n + 1)[n - 1]
        _, phi_a = host_ordinate(cd, f_iso)
        s_law = split_one_ended(M_s, phi_a, th, n)
        if s0 is None:
            s0 = s_fe
            pa0, fh0 = abs(phi_a), 1.0
        fh_at, _ = host_ordinate(cd, f_iso)
        if s0 is None or th_deg == 0.0:
            pa0, fh0 = abs(phi_a), fh_at
        print("  %6.1f deg %9.4f %10.4f %8.4f %9.4f %9.4f %9.4f"
              % (th_deg, 100 * s_fe, 100 * s_law, s_fe / s_law,
                 abs(phi_a) / pa0, fh_at, (s_fe / s0) / np.cos(th)))
        rows.append(("theta_scan", "n=%d theta=%.0f deg" % (n, th_deg),
                     "s_fe_pct", round(100 * float(s_fe), 5), "%",
                     "s_law %.5f %%, s/s0 %.5f, cos %.5f"
                     % (100 * s_law, s_fe / s0, np.cos(th))))
    print()

print("READING")
print("-" * 78)
print("FE/law is the closed form against the finite element: if it stays near")
print("one across the range, the law's whole theta dependence is verified.")
print("phi_a/0 and f_host show the second effect, the host stiffening under")
print("the anchorage spring (EA / L_c) sin^2(theta); it is why the last")
print("column, the width relative to theta = 0 divided by cos(theta), is not")
print("one. Both effects are geometry and a host eigenproblem, so a rig at")
print("theta = 0 tests the dynamics and inclination adds no further")
print("hypothesis. The unmeasured part is the second route, which needs sag.")

out = os.path.join(ROOT, "data", "theta_scan.csv")
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["block", "item", "quantity", "value", "unit", "note"])
    w.writerows(rows)
print("\nwrote %s (%d rows)" % (out, len(rows)))
