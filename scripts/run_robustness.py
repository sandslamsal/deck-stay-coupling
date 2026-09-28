# -*- coding: utf-8 -*-
"""Screening criterion under a wrong nominal tension, and campaign statistics.

Part 1 recomputes the detuning d with the nominal tension off by -10 % and
+10 % and scores the criterion |d| >= (s^2 - tol^2)/(2 tol) against the true
coupling error of each graded design. Part 2a gives R^2 of the closed form with
and without the xi gate. Part 2b replays the sampling of run_campaign.one_case()
and counts each rejection branch. Reads data/campaign.csv; writes data/robustness.csv.

Run:  python3 scripts/run_robustness.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from cablefe import tensioned_beam_freq  # noqa: E402
import run_campaign as rc  # noqa: E402  (import only; main() is guarded)

DATA = os.path.join(ROOT, "data")
TOLS = (0.02, 0.05, 0.10)
N_DRAWN = 12000          # designs drawn by run_campaign.py
N_CENSUS = 2000          # seeds re-run here for the rejection census


# --- Part 1: criterion robustness to a wrong nominal tension ---

def detuning_with_assumed_tension(row, factor):
    """Detuning d with f_iso from the assumed tension factor*T and the campaign f_deck.

    Uses ``row["T"]``, since ``row.T`` on a pandas Series is the transpose."""
    f_iso = tensioned_beam_freq(row["n_stay"], row["Lc"], factor * row["T"],
                                row["EIc"], row["mc"])
    return (f_iso - row.f_deck) / f_iso


def classify(d_abs, s, tol):
    """True where the criterion passes the design (predicts error <= tol)."""
    return d_abs >= (s ** 2 - tol ** 2) / (2.0 * tol)


def part1(g):
    print("=" * 74)
    print("  PART 1  --  CRITERION ROBUSTNESS TO AN UNKNOWN TRUE TENSION")
    print("=" * 74)
    print(f"    graded designs (mac > 0.5, xi > 150): {len(g)}")
    print("    d recomputed with the nominal tension off by -10 % / +10 %;")
    print("    s is tension-independent and does not move.")
    print()

    # detuning under each tension assumption
    d_true = g.d.values                      # d as stored by the campaign
    d_chk = g.apply(lambda r: detuning_with_assumed_tension(r, 1.0), axis=1)
    resid = np.abs(d_chk.values - d_true)
    print(f"    consistency: |d(recomputed, T) - d(stored)| "
          f"max = {resid.max():.2e}")
    d_m10 = g.apply(lambda r: detuning_with_assumed_tension(r, 0.9),
                    axis=1).values
    d_p10 = g.apply(lambda r: detuning_with_assumed_tension(r, 1.1),
                    axis=1).values

    coup = (g.eps - g.eps_control).abs().values   # true coupling error
    s = g.s.values

    straddle = (np.sign(d_m10) != np.sign(d_p10))
    print(f"    designs whose d band straddles an exact crossing "
          f"(sign flip between -10 % and +10 %): {int(straddle.sum())}")
    print()

    out = pd.DataFrame(dict(
        Lc=g.Lc.values, T=g["T"].values, n_stay=g.n_stay.values,
        theta=g.theta.values, mu_eff=g.mu_eff.values, s=s,
        f_deck=g.f_deck.values, f_s=g.f_s.values,
        d_true=d_true, d_m10=d_m10, d_p10=d_p10,
        band_straddles_zero=straddle,
        eps_coupling_abs=coup))

    header = (f"    {'tol':>5s} {'T used':>8s} {'pass':>5s} "
              f"{'false pass':>11s} {'false rej':>10s} {'miscls':>7s} "
              f"{'flipped vs T_true':>18s}")
    for tol in TOLS:
        actual_ok = coup <= tol
        print(f"    tol = {tol*100:.0f} %:  designs truly within tol: "
              f"{int(actual_ok.sum())} of {len(g)}")
        print(header)
        base_pass = classify(np.abs(d_true), s, tol)
        for lab, dd in (("true", d_true), ("-10%", d_m10), ("+10%", d_p10)):
            p = classify(np.abs(dd), s, tol)
            fp = int((p & ~actual_ok).sum())        # unconservative
            fn = int((~p & actual_ok).sum())        # needless rejection
            flip = int((p != base_pass).sum())
            print(f"    {tol*100:4.0f}% {lab:>8s} {int(p.sum()):5d} "
                  f"{fp:11d} {fn:10d} {fp+fn:7d} {flip:18d}")
            out[f"pass_tol{int(tol*100)}_{lab.strip('%').replace('-','m').replace('+','p')}"] = p
        out[f"actual_ok_tol{int(tol*100)}"] = actual_ok
        print()
    return out


# --- Part 2a: R^2 with and without the xi gate ---

def r2_report(sub, label):
    coup = (sub.eps - sub.eps_control).abs()
    res = coup - sub.eps_pred
    ss_res = float((res ** 2).sum())
    ss_tot = float(((coup - coup.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    print(f"    {label:52s} n={len(sub):4d}  R^2 = {r2:8.4f}  "
          f"median|res| = {np.median(np.abs(res))*100:.3f} pp")
    return r2


def part2a(d, g):
    print("=" * 74)
    print("  PART 2a  --  R^2 OF THE CLOSED FORM, WITH AND WITHOUT THE "
          "xi GATE")
    print("=" * 74)
    print("    target in every fit: coupling error |eps - eps_control|")
    print()
    r2_all = r2_report(d, "all usable designs (no mac or xi filter)")
    r2_graded = r2_report(g, "graded subset (mac > 0.5, xi > 150)")

    lo = d[d.xi < 150]
    print()
    print(f"    designs with xi < 150: {len(lo)} of {len(d)}")
    print(f"      their control (no-coupling) bias |eps_control|: "
          f"median {lo.eps_control.abs().median()*100:.2f} %, "
          f"p90 {lo.eps_control.abs().quantile(0.90)*100:.2f} %, "
          f"worst {lo.eps_control.abs().max()*100:.2f} %")
    hi = d[d.xi >= 150]
    print(f"      for comparison, xi >= 150 ({len(hi)} designs): "
          f"median {hi.eps_control.abs().median()*100:.2f} %, "
          f"worst {hi.eps_control.abs().max()*100:.2f} %")
    print(f"    designs with mac <= 0.5: {int((d.mac <= 0.5).sum())}")
    print()
    return r2_all, r2_graded


# --- Part 2b: rejection census ---

def classify_seed(seed):
    """Replay the draws of run_campaign.one_case(seed); name the first rejection.

    Mirrors one_case() in the same rng call order; census() checks the match."""
    rng = np.random.default_rng(seed)

    Ld = float(np.exp(rng.uniform(np.log(40.0), np.log(600.0))))
    md = float(np.exp(rng.uniform(np.log(800.0), np.log(35000.0))))
    fd1 = float(np.exp(rng.uniform(np.log(0.20), np.log(3.5))))
    EId = md * (fd1 * 2 * np.pi * Ld ** 2 / np.pi ** 2) ** 2

    Lc = float(np.exp(rng.uniform(np.log(12.0), np.log(280.0))))
    theta = float(rng.uniform(np.deg2rad(20.0), np.deg2rad(60.0)))
    A = float(np.exp(rng.uniform(np.log(5e-4), np.log(2.0e-2))))
    n_stay = int(rng.integers(1, 6))
    d_target = float(rng.uniform(-0.15, 0.15))

    xfrac = Lc * np.cos(theta) / Ld
    if not (0.10 <= xfrac <= 0.48):
        return "geometry (anchorage outside 0.10-0.48 of span)"
    ia = int(round(xfrac * rc.ND))
    if ia < 2 or ia > rc.ND - 2:
        return "ia bounds (anchorage node too close to a support)"

    mc = rc.RHO_STEEL * A
    EA = rc.E_STEEL * A
    k_ax = EA / Lc * np.sin(theta) ** 2

    fd, phia = rc.deck_alone_modes(Ld, EId, md, rc.ND, ia, k_ax)
    ok = np.abs(phia) > 0.15 * np.abs(phia).max()
    if not ok.any():
        return "no coupled deck mode (node at anchorage in all modes)"
    j = int(rng.choice(np.where(ok)[0]))
    f_deck = float(fd[j])

    f_target = f_deck / (1.0 - d_target)
    sigma = rc.RHO_STEEL * (2.0 * Lc * f_target / n_stay) ** 2
    if not (120e6 <= sigma <= 800e6):
        return ("stress bounds (sigma outside 120-800 MPa; equivalently "
                "T/mc = sigma/rho outside band)")
    return "pass_cheap"


BRANCHES = (
    "geometry (anchorage outside 0.10-0.48 of span)",
    "ia bounds (anchorage node too close to a support)",
    "no coupled deck mode (node at anchorage in all modes)",
    ("stress bounds (sigma outside 120-800 MPa; equivalently "
     "T/mc = sigma/rho outside band)"),
    "fe failure (eigen-solve raised)",
)


def census(d):
    print("=" * 74)
    print("  PART 2b  --  REJECTION CENSUS BY "
          "BRANCH")
    print("=" * 74)
    print(f"    re-running the sampling logic of one_case() for "
          f"{N_CENSUS} seeds ...")

    counts = {b: 0 for b in BRANCHES}
    usable_rows = []
    for seed in range(N_CENSUS):
        reason = classify_seed(seed)
        if reason == "pass_cheap":
            # the remaining branch is the FE eigen-solve; run one_case() itself
            row = rc.one_case(seed)
            if row is None:
                counts["fe failure (eigen-solve raised)"] += 1
            else:
                usable_rows.append(row)
        else:
            counts[reason] += 1
    n_usable = len(usable_rows)

    print()
    print(f"    {'branch':78s} {'count':>6s} {'share':>7s}")
    for k in BRANCHES:
        v = counts[k]
        print(f"    {k:78s} {v:6d} {100*v/N_CENSUS:6.2f}%")
    print(f"    {'usable (survives every branch)':78s} {n_usable:6d} "
          f"{100*n_usable/N_CENSUS:6.2f}%")
    print()
    print(f"    usable rate in census: {100*n_usable/N_CENSUS:.2f} %  "
          f"(full design set: 699/12000 = {699/120:.2f} %)")

    # cross-check: the campaign ran seeds 0..11999 in order and kept the
    # non-None rows, so the census's usable rows for seeds 0..N_CENSUS-1
    # must reproduce the first n_usable rows of campaign.csv exactly
    head = d.head(n_usable)
    cen = pd.DataFrame(usable_rows)
    cols = ["Ld", "Lc", "T", "f_deck", "f_coupled", "eps", "eps_control"]
    match = all(np.allclose(head[c].values, cen[c].values, rtol=1e-12)
                for c in cols)
    print(f"    cross-check: census usable rows reproduce the first "
          f"{n_usable} rows of campaign.csv on {cols}: "
          f"{'PASS' if match else 'FAIL'}")
    if not match:
        raise RuntimeError("census does not reproduce campaign.csv; the "
                           "replicated sampling logic has diverged from "
                           "run_campaign.one_case")
    print("    every rejection branch acts before the coupled model is")
    print("    solved; rejections come from geometric and stress limits,")
    print("    not from the computed error.")
    return counts, n_usable


def main():
    d = pd.read_csv(os.path.join(DATA, "campaign.csv"))
    g = d[(d.mac > 0.5) & (d.xi > 150)].copy()
    n_all, n_graded = len(d), len(g)
    print(f"  design set: {n_all} usable designs of {N_DRAWN} drawn "
          f"({100*(1-n_all/N_DRAWN):.1f} % attrition to usable); "
          f"{n_graded} graded ({100*(1-n_graded/N_DRAWN):.1f} % attrition "
          f"to graded)")
    print()

    out = part1(g)
    part2a(d, g)
    census(d)

    path = os.path.join(DATA, "robustness.csv")
    out.to_csv(path, index=False)
    print(f"  per-design table written to {path}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
