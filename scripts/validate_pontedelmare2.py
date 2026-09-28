# -*- coding: utf-8 -*-
"""Ponte del Mare: measured pulls, sag closure, and mu_eff.

Uses stay frequencies, SISTRAL-measured pulls, stay material and global
modes from Kumar (2011), Tables 4.4, 5.3, 5.5 and 5.6 and Figs. 4.13-4.14.
Prints tension from orders 2-5 and from the fundamental against the pull
(A, B), sag closure of the fundamental anomaly (C), the detuning screen (D),
and predicted veering widths from mu_eff (E).
Run: python3 scripts/validate_pontedelmare2.py
"""
from __future__ import annotations
import os
import sys
import numpy as np, pandas as pd, os

G = 9.80665
RHO_STAY = 8289.0

# STAYS: L, m, five picked f (Table 4.4), SISTRAL pull in kN and string f
# (Table 5.5), and an inclination band in degrees from the plan geometry.
# Values transcribed from Kumar (2011), PhD thesis, University of Trento
# (Ponte del Mare footbridge). They are not redistributed with this code;
# they live in data/external/validate_pontedelmare2_data.py (see README).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "external"))
try:
    from validate_pontedelmare2_data import (  # noqa: E402
        STAYS, CASES, E_STAY)
except ImportError as exc:
    raise SystemExit("scripts/validate_pontedelmare2.py needs values transcribed from "
                     "Kumar (2011), PhD thesis, University of Trento (Ponte del Mare footbridge), which are not redistributed here. "
                     "See README, 'Third-party data'.") from exc
DECK = np.array([0.747,1.065,1.126,1.243,1.394,1.510,1.716,1.791,
                 2.306,2.364,2.512,2.862])

def lam2(L,T,m,th,EA):
    H=T*np.cos(th); Lh=L*np.cos(th)
    d=m*G*Lh**2/(8*H); Le=Lh*(1+8*(d/Lh)**2)
    return (m*G*Lh/H)**2*Lh/(H*Le/EA)

print("="*84)
print("A/B  tension against the SISTRAL-measured pull [kN]")
print("="*84)
print("stay   T_meas   T(orders 2-5)  err     T(f1 alone)  err")
rows=[]
for k,s in STAYS.items():
    L,m=s["L"],s["m"]; f=np.array(s["f"]); n=np.arange(1,6)
    c=np.sum(n[1:]*f[1:])/np.sum(n[1:]**2)        # lsq f_n = c n, orders 2-5
    T_hi=4*m*L**2*c**2/1e3
    T_f1=4*m*L**2*f[0]**2/1e3
    e_hi=100*(T_hi-s["T_meas"])/s["T_meas"]
    e_f1=100*(T_f1-s["T_meas"])/s["T_meas"]
    print("%-5s %7.0f %11.0f %+7.1f%% %11.0f %+7.1f%%"
          % (k,s["T_meas"],T_hi,e_hi,T_f1,e_f1))
    rows.append(dict(stay=k,T_meas=s["T_meas"],T_hi=T_hi,e_hi=e_hi,
                     T_f1=T_f1,e_f1=e_f1))

print()
print("="*84)
print("C  sag closure of the fundamental anomaly (theta bands, E=165 GPa)")
print("="*84)
print("stay   f1_meas  f1_string(T_meas)  anomaly   sag-lift band   residual")
for k,s in STAYS.items():
    L,m=s["L"],s["m"]; A=m/RHO_STAY; EA=E_STAY*A
    anom=100*(s["f"][0]/s["f_string_meas"]-1)
    lifts=[100*lam2(L,s["T_meas"]*1e3,m,np.deg2rad(t),EA)/(4*np.pi**2)
           for t in s["th"]]
    lo,hi=min(lifts),max(lifts)
    res_lo, res_hi = anom-hi, anom-lo
    print("%-5s %8.2f %10.2f %14.1f%%   %5.1f-%4.1f%%   %+5.1f..%+5.1f%%"
          % (k,s["f"][0],s["f_string_meas"],anom,lo,hi,res_lo,res_hi))

print()
print("="*84)
print("D  detuning screen: stay modes within 5% of an identified global mode")
print("="*84)
for k,s in STAYS.items():
    for i,fn in enumerate(s["f"],start=1):
        j=int(np.argmin(np.abs(DECK-fn))); d=(fn-DECK[j])/fn
        if abs(d)<=0.05:
            print("  %-4s mode %d: %5.2f Hz vs global %5.3f Hz, d=%+.4f"
                  % (k,i,fn,DECK[j],d))

print()
print("="*84)
print("E  mu_eff: prediction against observation at the three coincidences")
print("="*84)
# Kumar Fig. 4.14 ordinates (normalized to unit maximum) at each anchorage, with
# read bands; modal masses banded from the deck construction (55 mm slab +
# steel truss + finishes, both decks 148-173 m): M_modal = 40-160 t for
# single-deck modes, 80-250 t when both decks move.
print("%-5s %-6s %10s %14s %16s" % ("stay","d","M_s [kg]","mu_eff band","s band"))
for k,n,fg,(p_lo,p_hi),(M_lo,M_hi),note in CASES:
    s=STAYS[k]; L,m=s["L"],s["m"]
    f1=s["f"][0]; d=(f1-fg)/f1
    Ms=m*L/2
    mu_lo=Ms*(p_lo/np.sqrt(M_hi*1e3))**2
    mu_hi=Ms*(p_hi/np.sqrt(M_lo*1e3))**2
    th_mid=np.deg2rad(np.mean(s["th"]))
    s_lo=2/np.pi/n*np.cos(th_mid)*np.sqrt(mu_lo)
    s_hi=2/np.pi/n*np.cos(th_mid)*np.sqrt(mu_hi)
    print("%-5s %+.4f %9.0f   %8.1e-%7.1e   %5.2f-%5.2f %%"
          % (k,d,Ms,mu_lo,mu_hi,100*s_lo,100*s_hi))
    print("      %s" % note)
    if k=="N8E":
        kap_lo,kap_hi=s_lo/(2*abs(d)),s_hi/(2*abs(d))
        print("      predicted deck-record mixing kappa = s/2|d| = %.3f-%.3f;"
              % (kap_lo,kap_hi))
        print("      predicted peak shift s^2/2|d| = %.3f-%.3f %% (unresolvable)"
              % (100*s_lo**2/(2*abs(d)),100*s_hi**2/(2*abs(d))))
    if k=="N5E":
        print("      predicted upward branch displacement at d~0: s/2 to s "
              "= %.1f-%.1f %%" % (100*s_lo/2,100*s_hi))
print()
print("observed: N8E line visible but small in deck records (both damper")
print("states), no PSD splitting anywhere, N7E shows nothing at exact")
print("tuning, N5E fundamental sits +2.4-3.0 % above its sag-closed")
print("isolated position. One anomaly (N4W, +7-8 % above sag) remains open;")
print("the identified-mode table stops at 2.86 Hz, below where its")
print("neighbouring modes would sit.")
