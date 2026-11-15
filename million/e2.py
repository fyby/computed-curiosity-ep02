"""E2, the dials: the same 100,000 test planets (E1 ids 0..99,999) around 8 other giants,
843 laps (about 10,000 yr) each. Only the giant changes."""
import json
import time

import numpy as np

from . import disc, paths, pool, runset, stagelog
from .constants import A_G, E_G, MU_JUPITER, N_DIAL, SMOKE, period

CASES = {
    "e00": dict(m=MU_JUPITER, e=0.0), "e010": dict(m=MU_JUPITER, e=0.10),
    "e023": dict(m=MU_JUPITER, e=0.23), "e04": dict(m=MU_JUPITER, e=0.40),
    "m01": dict(m=0.1 * MU_JUPITER, e=E_G), "m03": dict(m=0.3 * MU_JUPITER, e=E_G),
    "m3": dict(m=3 * MU_JUPITER, e=E_G), "m10": dict(m=10 * MU_JUPITER, e=E_G),
}
N_LAPS = 843 if not SMOKE else 51


def strobe_laps(n_laps=N_LAPS):
    return list(range(16)) + disc.geometric_laps(15, n_laps, (843 / 15) ** (1 / 48.0))


def run(cases=None):
    d = disc.e1_disc()
    ids = np.arange(N_DIAL)
    pl = disc.subset(d, ids)
    for name, c in CASES.items():
        if cases and name not in cases:
            continue
        t0 = time.time()
        g = dict(m=c["m"], a=A_G, e=c["e"])
        tab, metas, wall, jobs = runset.run(f"e2_{name}", ids, pl, g, N_LAPS, strobe_laps=strobe_laps())
        runset.save_table(paths.raw(f"e2_{name}.npz"), tab, giant=g, n_laps=N_LAPS, T_exact_yr=N_LAPS * period(c["m"]))
        laps = runset.merge_strobes(jobs, N_DIAL, lambda i: i, paths.viz("e2", f"{name}_strobe.npy"))
        with open(paths.viz("e2", f"{name}_strobe_meta.json"), "w") as f:
            json.dump(dict(giant=g, period_yr=period(c["m"]), laps=[int(x) for x in laps],
                           t_yr=[float(x * period(c["m"])) for x in laps], shape=[len(laps), N_DIAL, 2],
                           note="strobes at the giant's pericentre passages; encoding as e1_strobe_meta.json"), f, indent=1)
        stagelog.record(f"e2_{name}", time.time() - t0, planet_steps=sum(m["psteps"] for m in metas), n=N_DIAL, n_laps=N_LAPS)
