# -*- coding: utf-8 -*-
"""Pilot: modal mass ratio by bridge class and inversion error near crossings.

Part A samples the stay-to-deck modal mass ratio mu for footbridges and
long-span vehicular bridges. Part B varies the stay tension so that stay
modes pass through deck modes, and inverts the coupled frequencies after
oracle (MAC-matched), screened and naive (frequency-order) peak picking.
Writes data/pilot_census.csv and data/pilot_detuning.csv.

Run:  python3 scripts/run_pilot.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from cablefe import (CableDeck, invert_multimode, invert_string,  # noqa: E402
                     screened_pick, string_freq, tensioned_beam_freq,
                     xi_param)

DATA = os.path.join(ROOT, "data")


# --- Part A: modal mass ratio by bridge class ---

# Sampled ranges: deck mass md (kg/m), span Ld (m), stay mass mc (kg/m), stay
# chord length Lc (m), stay tension T (N). Footbridge: light steel or
# composite deck; long span: typical highway cable-stayed bridge.
CLASSES = {
    "footbridge": dict(
        md=(800.0, 2500.0), Ld=(40.0, 120.0),
        mc=(3.5, 12.0), Lc=(12.0, 50.0), T=(150e3, 900e3)),
    "long-span vehicular": dict(
        md=(15000.0, 35000.0), Ld=(250.0, 600.0),
        mc=(40.0, 110.0), Lc=(80.0, 280.0), T=(2.5e6, 8.0e6)),
}


def census(nsamp=4000, seed=7):
    """Sample the modal mass ratio mu (stay over deck, both on a half-sine)
    and the stay fundamental for both bridge classes."""
    rng = np.random.default_rng(seed)
    rows = []
    for name, r in CLASSES.items():
        for _ in range(nsamp):
            md = rng.uniform(*r["md"])
            Ld = rng.uniform(*r["Ld"])
            mc = rng.uniform(*r["mc"])
            Lc = rng.uniform(*r["Lc"])
            T = rng.uniform(*r["T"])
            # half-sine modal masses
            Mdeck = md * Ld / 2.0
            Mstay = mc * Lc / 2.0
            rows.append(dict(
                bridge_class=name, md=md, Ld=Ld, mc=mc, Lc=Lc, T=T,
                M_deck=Mdeck, M_stay=Mstay, mu=Mstay / Mdeck,
                f_stay1=string_freq(1, Lc, T, mc)))
    return pd.DataFrame(rows)


def report_census(d):
    print("=" * 74)
    print("PART A  Modal mass ratio by bridge class")
    print("=" * 74)
    print("  mu = stay modal mass / deck modal mass, over the sampled ranges")
    print()
    print(f"  {'class':22s} {'p05':>10s} {'median':>10s} {'p95':>10s}"
          f" {'f_stay1 med':>12s}")
    med = {}
    for name, g in d.groupby("bridge_class", sort=False):
        q = g.mu.quantile([0.05, 0.5, 0.95]).values
        med[name] = q[1]
        print(f"  {name:22s} {q[0]:10.2e} {q[1]:10.2e} {q[2]:10.2e}"
              f" {g.f_stay1.median():10.2f} Hz")
    a = med["footbridge"]
    b = med["long-span vehicular"]
    print()
    print(f"  ratio of medians, footbridge / long-span: {a / b:.2f}x")
    print()
    if a / b < 3.0:
        print("  The two classes have comparable modal mass ratios: the")
        print("  ratio of medians is below 3, well short of an order of")
        print("  magnitude, so mu alone does not single out the footbridge")
        print("  class.")
    else:
        print(f"  The footbridge ratio is {a / b:.1f}x larger,")
        print("  so mu alone singles out the footbridge class.")
    return a / b


# --- Part B: coupling bias of the tension inversion ---

# One representative bridge per class, one stay on a simply supported deck,
# held fixed while the stay tension varies. The long-span deck stiffness
# places the global frequencies where a long-span bridge has them.
BRIDGES = {
    "footbridge": dict(Ld=80.0, EId=2.0e9, md=1000.0,
                       Lc=25.0, EIc=1.2e4, mc=5.5, EA=1.4e8,
                       theta=np.deg2rad(35.0), nd=48, nc=48),
    "long-span vehicular": dict(Ld=500.0, EId=3.2e13, md=20000.0,
                                Lc=150.0, EIc=8.0e6, mc=60.0, EA=2.4e9,
                                theta=np.deg2rad(25.0), nd=48, nc=48),
}
TRANGE = {
    "footbridge": (120e3, 900e3),
    "long-span vehicular": (2.5e6, 8.0e6),
}
BRIDGE = BRIDGES["footbridge"]

NMODE_ID = 5          # stay mode orders the inversion uses
SENSOR_M = 2.0        # sensor distance (m) up the chord from the anchorage


def mac(a, b):
    num = float(a @ b) ** 2
    den = float(a @ a) * float(b @ b)
    return num / den if den > 0 else 0.0


def oracle_pick(cd, f, Phi, nmax):
    """Match each isolated stay mode to its coupled counterpart by MAC;
    returns (frequencies, MAC values)."""
    cdofs = np.array(cd.cable_dofs())
    x = np.linspace(0.0, 1.0, len(cdofs))
    out = []
    for n in range(1, nmax + 1):
        target = np.sin(n * np.pi * x)
        best, bestm = None, -1.0
        for j in range(Phi.shape[1]):
            m = mac(Phi[cdofs, j], target)
            if m > bestm:
                bestm, best = m, j
        out.append((f[best], bestm))
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def naive_pick(cd, f, Phi, nmax, rel_floor=0.10):
    """Peaks in frequency order at the sensor, numbered 1, 2, 3, ..."""
    sd = cd.cable_sensor_dof(SENSOR_M)
    amp = np.abs(Phi[sd, :])
    if amp.max() <= 0:
        return np.array([])
    keep = amp >= rel_floor * amp.max()
    fk = np.sort(f[keep])
    return fk[:nmax]


def detuning_study(nT=360, bridge=None, trange=None):
    global BRIDGE
    BRIDGE = bridge if bridge is not None else BRIDGES['footbridge']
    lo, hi = trange if trange is not None else (120e3, 900e3)
    rows = []
    Ts = np.linspace(lo, hi, nT)
    for T in Ts:
        cd = CableDeck(T=T, **BRIDGE)
        f, Phi = cd.modes(60)

        # detuning: how close does any stay mode sit to any deck mode
        stay = cd.stay_alone(NMODE_ID)
        deck = cd.deck_alone(10)
        beta_min, n_at = np.inf, 0
        for n, fs in enumerate(stay, start=1):
            r = fs / deck
            j = np.argmin(np.abs(r - 1.0))
            if abs(r[j] - 1.0) < abs(beta_min - 1.0):
                beta_min, n_at = r[j], n

        # detuning of stay mode 1, the mode the single-mode inversion uses
        r1 = stay[0] / deck
        j1 = np.argmin(np.abs(r1 - 1.0))
        beta_1 = r1[j1]

        f_or, mac_or = oracle_pick(cd, f, Phi, NMODE_ID)
        f_nv = naive_pick(cd, f, Phi, NMODE_ID)

        rec = dict(T_true=T, beta=beta_min, beta_1=beta_1, n_crossing=n_at,
                   mac_min=float(np.min(mac_or)),
                   xi=xi_param(BRIDGE["Lc"], T, BRIDGE["EIc"]))

        # control: exact isolated-cable frequencies, no coupling; separates
        # the bending-stiffness bias of the formula from the coupling bias
        f_iso = np.array([tensioned_beam_freq(n, BRIDGE["Lc"], T,
                                              BRIDGE["EIc"], BRIDGE["mc"])
                          for n in range(1, NMODE_ID + 1)])
        Ts_ct = invert_string(f_iso, BRIDGE["Lc"], BRIDGE["mc"])
        rec["err_string_n1_control"] = 100.0 * (Ts_ct[0] - T) / T
        Tm_ct, _ = invert_multimode(f_iso, BRIDGE["Lc"], BRIDGE["mc"])
        rec["err_multi_control"] = 100.0 * (Tm_ct - T) / T

        # string and multi-mode inversions, oracle picking
        Ts_or = invert_string(f_or, BRIDGE["Lc"], BRIDGE["mc"])
        rec["err_string_n1_oracle"] = 100.0 * (Ts_or[0] - T) / T
        Tm_or, EIm_or = invert_multimode(f_or, BRIDGE["Lc"], BRIDGE["mc"])
        rec["err_multi_oracle"] = 100.0 * (Tm_or - T) / T
        rec["EI_multi_oracle"] = EIm_or

        # string and multi-mode inversions, screened picking
        sd = cd.cable_sensor_dof(SENSOR_M)
        amp = np.abs(Phi[sd, :])
        cand = f[amp >= 0.05 * amp.max()]
        f_sc, n_sc = screened_pick(cand, None, NMODE_ID)
        if len(f_sc) >= 3:
            Ts_sc = invert_string(f_sc, BRIDGE["Lc"], BRIDGE["mc"], n_sc)
            rec["err_string_n1_screened"] = 100.0 * (Ts_sc[0] - T) / T
            Tm_sc, _ = invert_multimode(f_sc, BRIDGE["Lc"], BRIDGE["mc"], n_sc)
            rec["err_multi_screened"] = 100.0 * (Tm_sc - T) / T
        else:
            rec["err_string_n1_screened"] = np.nan
            rec["err_multi_screened"] = np.nan

        # sensitivity of blind numbering to the amplitude floor
        spread = []
        for fl in (0.02, 0.05, 0.10, 0.20, 0.35):
            fk = np.sort(f[amp >= fl * amp.max()])[:NMODE_ID]
            if len(fk) >= 3:
                Tm, _ = invert_multimode(fk, BRIDGE["Lc"], BRIDGE["mc"])
                spread.append(100.0 * (Tm - T) / T)
        rec["naive_floor_spread"] = (max(spread) - min(spread)) if spread else np.nan

        # string and multi-mode inversions, naive picking
        if len(f_nv) >= 3:
            Ts_nv = invert_string(f_nv, BRIDGE["Lc"], BRIDGE["mc"])
            rec["err_string_n1_naive"] = 100.0 * (Ts_nv[0] - T) / T
            Tm_nv, EIm_nv = invert_multimode(f_nv, BRIDGE["Lc"], BRIDGE["mc"])
            rec["err_multi_naive"] = 100.0 * (Tm_nv - T) / T
            rec["EI_multi_naive"] = EIm_nv
            rec["npeaks"] = len(f_nv)
        else:
            rec["err_string_n1_naive"] = np.nan
            rec["err_multi_naive"] = np.nan
            rec["EI_multi_naive"] = np.nan
            rec["npeaks"] = len(f_nv)

        rows.append(rec)
    return pd.DataFrame(rows)


def report_detuning(d):
    print()
    print("=" * 74)
    print("PART B  Coupling bias of the tension inversion")
    print("=" * 74)
    print(f"  Ld={BRIDGE['Ld']:.0f} m, "
          f"Lc={BRIDGE['Lc']:.0f} m, theta={np.rad2deg(BRIDGE['theta']):.0f} deg")
    print(f"  stay tension varied over {d.T_true.min()/1e3:.0f} to "
          f"{d.T_true.max()/1e3:.0f} kN, {len(d)} steps")
    print(f"  inversion uses stay mode orders 1 to {NMODE_ID}, sensor "
          f"{SENSOR_M:.0f} m above the anchorage")
    print()

    print("  Control, no coupling: worst error of the same formulas on the")
    print("  exact isolated-cable frequencies. This is the")
    print("  bending-stiffness bias; any excess is due to coupling.")
    print(f"    single-mode  {d.err_string_n1_control.abs().max():6.3f} %"
          f"    multi-mode  {d.err_multi_control.abs().max():6.3f} %")
    print()

    # single-mode uses stay mode 1, so it is graded on that mode's detuning
    near1 = d[np.abs(d.beta_1 - 1.0) <= 0.05]
    far1 = d[np.abs(d.beta_1 - 1.0) > 0.25]
    nearm = d[np.abs(d.beta - 1.0) <= 0.05]
    farm = d[np.abs(d.beta - 1.0) > 0.25]

    print(f"  {'quantity':34s} {'near crossing':>16s} {'away':>14s}")
    print(f"  {'':34s} {'|beta-1|<=0.05':>16s} {'>0.25':>14s}")
    print("  " + "-" * 66)
    for col, lab, nr, fr in (
            ("err_string_n1_oracle", "single-mode, oracle picking", near1, far1),
            ("err_multi_oracle", "multi-mode, oracle picking", nearm, farm),
            ("err_string_n1_naive", "single-mode, naive picking", near1, far1),
            ("err_multi_naive", "multi-mode, naive picking", nearm, farm)):
        a = nr[col].abs().max() if len(nr) else np.nan
        b = fr[col].abs().max() if len(fr) else np.nan
        print(f"  {lab:34s} {a:13.2f} %  {b:11.2f} %")
    for col, lab, nr, fr in (
            ("err_string_n1_screened", "single-mode, screened picking", near1, far1),
            ("err_multi_screened", "multi-mode, screened picking", nearm, farm)):
        a = nr[col].abs().max() if len(nr) else np.nan
        b = fr[col].abs().max() if len(fr) else np.nan
        print(f"  {lab:34s} {a:13.2f} %  {b:11.2f} %")
    print(f"  (n near/far: mode-1 detuning {len(near1)}/{len(far1)}, "
          f"any-mode {len(nearm)}/{len(farm)})")

    print()
    print("  Proximity of stay modes to deck modes")
    print(f"    largest |beta-1| reached by any of stay modes 1-{NMODE_ID}, "
          f"over the whole")
    print(f"    tension range: {(d.beta - 1).abs().max():.3f}. Tensions with "
          f"some stay mode")
    print(f"    within 10 % of a deck mode: "
          f"{100 * ((d.beta - 1).abs() <= 0.10).mean():.0f} %.")
    print(f"    Stay mode 1 alone reaches |beta-1| = "
          f"{(d.beta_1 - 1).abs().max():.3f}, and is within 5 % of a deck")
    print(f"    mode for {100 * ((d.beta_1 - 1).abs() <= 0.05).mean():.0f} % "
          f"of tensions.")

    print()
    print("  Sensitivity of blind numbering to the amplitude floor")
    print(f"    spread of the multi-mode error across amplitude floors")
    print(f"    0.02 to 0.35, median over the range: "
          f"{d.naive_floor_spread.median():.1f} pp, worst "
          f"{d.naive_floor_spread.max():.1f} pp.")
    print("    The blind-numbering error depends on the floor chosen,")
    print("    so the screened result is reported instead.")

    print()
    print("  worst tension error anywhere in the range:")
    for col, lab in (("err_string_n1_oracle", "single-mode, oracle"),
                     ("err_multi_oracle", "multi-mode, oracle"),
                     ("err_string_n1_naive", "single-mode, naive"),
                     ("err_multi_naive", "multi-mode, naive")):
        i = d[col].abs().idxmax()
        if np.isnan(d[col].abs().max()):
            continue
        r = d.loc[i]
        print(f"    {lab:22s} {r[col]:+8.2f} %  at T={r.T_true/1e3:6.1f} kN, "
              f"beta={r.beta:.3f}, stay mode {int(r.n_crossing)}")

    print()
    worst_oracle = max(d.err_string_n1_oracle.abs().max(),
                       d.err_multi_oracle.abs().max())
    worst_screened = np.nanmax([d.err_string_n1_screened.abs().max(),
                                d.err_multi_screened.abs().max()])
    worst_naive = np.nanmax([d.err_string_n1_naive.abs().max(),
                             d.err_multi_naive.abs().max()])
    print("=" * 74)
    print("  Worst tension error by picking method")
    print("=" * 74)
    print(f"  frequency-shift bias (oracle picking):        {worst_oracle:.2f} %")
    print(f"  harmonic-comb screening:                      {worst_screened:.2f} %")
    print(f"  blind numbering, threshold-dependent:         {worst_naive:.2f} %")
    print()
    if worst_screened < 2.0:
        print("  The screened-picking error is below 2 %, so the coupling")
        print("  bias is small at these mode orders.")
    elif worst_oracle < 2.0 <= worst_screened:
        print("  The frequency-shift bias alone is below 2 %, but mode")
        print("  mis-assignment raises the screened-picking error to 2 %")
        print("  or more.")
    else:
        print("  Both the frequency-shift bias and the screened-picking")
        print("  error reach 2 % or more.")
    return worst_oracle, worst_naive


def main():
    os.makedirs(DATA, exist_ok=True)

    d = census()
    d.to_csv(os.path.join(DATA, "pilot_census.csv"), index=False)
    report_census(d)

    allframes = []
    for name in BRIDGES:
        print()
        print("#" * 74)
        print(f"#  CLASS: {name}")
        print("#" * 74)
        e = detuning_study(bridge=BRIDGES[name], trange=TRANGE[name])
        e["bridge_class"] = name
        report_detuning(e)
        allframes.append(e)

    out = pd.concat(allframes, ignore_index=True)
    out.to_csv(os.path.join(DATA, "pilot_detuning.csv"), index=False)

    print()
    print("=" * 74)
    print("  Worst tension error by bridge class")
    print("=" * 74)
    print(f"  {'class':22s} {'control':>9s} {'oracle':>9s} {'screened':>10s}"
          f" {'max|beta-1|':>12s}")
    for name, g in out.groupby("bridge_class", sort=False):
        print(f"  {name:22s} "
              f"{g.err_string_n1_control.abs().max():8.2f}% "
              f"{max(g.err_string_n1_oracle.abs().max(), g.err_multi_oracle.abs().max()):8.2f}% "
              f"{np.nanmax([g.err_string_n1_screened.abs().max(), g.err_multi_screened.abs().max()]):9.2f}% "
              f"{(g.beta - 1).abs().max():12.3f}")
    print()
    print(f"  wrote {DATA}/pilot_census.csv and {DATA}/pilot_detuning.csv")


if __name__ == "__main__":
    main()
