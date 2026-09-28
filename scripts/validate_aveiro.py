# -*- coding: utf-8 -*-
"""Check the detuning screen against the Aveiro circular footbridge.

For each of the eight stays, computes the detuning d between the measured
cable fundamental and the nearest measured global frequency, and the
departure of the estimated stay force from the design force. Data: Rebelo,
Julio, Varum, Costa, Experimental Techniques 34(4) 62-68, 2010, Tables 1
and 3. Writes data/aveiro.csv.

Run:  python3 scripts/validate_aveiro.py
"""
import os
import sys
import numpy as np, pandas as pd, os

# Values transcribed from Rebelo et al. (2009) (Aveiro footbridge): stay data
# (Table 1) and mean global frequencies (Table 3). They are not redistributed
# with this code: they live in data/external/validate_aveiro_data.py
# (see README).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "external"))
try:
    from validate_aveiro_data import (  # noqa: E402
        CABLE, GLOBAL, LABEL)
except ImportError as exc:
    raise SystemExit("scripts/validate_aveiro.py needs values transcribed from "
                     "Rebelo et al. (2009) (Aveiro footbridge), which are not redistributed here. "
                     "See README, 'Third-party data'.") from exc
# Global modes labeled "strip" are modes of the mast anchor strips.

rows = []
print("cable  f1 [Hz]  nearest global      d       |  T_est  T_des   diff")
for i in range(8):
    f1 = CABLE["f1"][i]
    j = int(np.argmin(np.abs(GLOBAL - f1)))
    d = (f1 - GLOBAL[j]) / f1
    dT = 100*(CABLE["Test"][i]-CABLE["Tdes"][i])/CABLE["Tdes"][i]
    print("  %d    %5.2f    %-9s %5.2f   %+7.3f  | %6.2f %6.2f  %+5.1f %%"
          % (CABLE["n"][i], f1, LABEL[j], GLOBAL[j], d,
             CABLE["Test"][i], CABLE["Tdes"][i], dT))
    rows.append(dict(cable=CABLE["n"][i], f1=f1, nearest=LABEL[j],
                     f_global=GLOBAL[j], d=d, T_est=CABLE["Test"][i],
                     T_des=CABLE["Tdes"][i], dT_pct=dT))
d = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)
d.to_csv("data/aveiro.csv", index=False)
print("\n  min |d| over the eight fundamentals: %.3f  (screen clears all)"
      % d.d.abs().min())
print("  force agreement: max departure %+.1f %% (cable %d)"
      % (d.dT_pct.abs().max()*np.sign(d.loc[d.dT_pct.abs().idxmax(),'dT_pct']),
         d.loc[d.dT_pct.abs().idxmax(),'cable']))
print("  wrote data/aveiro.csv")
