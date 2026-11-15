"""E1, the main run: 1,000,000 test planets around a copy of Jupiter.

Two phases. Phase 1 runs every chunk to lap 169 (2,005 yr, "the first 2,000
simulated years") and measures the throughput. The code then projects the
wall-clock time of the whole run; if it exceeds 85 minutes the run length is
cut from 50,000 to 40,000 yr (SIM_SPEC section 4). The decision is written to
raw/e1/decision.json and later to run_meta.json. Phase 2 continues every chunk
from its checkpoint to the chosen end.
"""
import json
import math
import os
import time

import numpy as np

from . import disc, paths, pool, runset, stagelog
from .constants import (A_G, E_G, MU_JUPITER, P_G, T_MAIN, T_FALLBACK, FALLBACK_LIMIT_MIN, N_MAIN, N_DIAL,
                        SMOKE, N_PROCESSES)

JUP = dict(m=MU_JUPITER, a=A_G, e=E_G)
PHASE1_LAPS = 169 if not SMOKE else 17          # 169 laps = 2,005 yr
NOMINAL_LAPS = int(round(T_MAIN / P_G))         # 4,214 laps
EXPORT_OVERHEAD_MIN = 4.0                       # merge + strobe/element export (spec section 9)


def laps_of(T):
    return int(round(T / P_G))


def strobe_laps(n_laps):
    """Every lap 0..240, then about 120 log-spaced laps to the end (prefix-stable)."""
    first = min(240, n_laps)
    ratio = (NOMINAL_LAPS / 240.0) ** (1 / 120.0) if not SMOKE else 1.05
    return list(range(first + 1)) + (disc.geometric_laps(first, n_laps, ratio) if n_laps > first else [])


def elem_laps(n_laps):
    """About 64 log-spaced laps (0, 1, ..., end) for the element export."""
    ratio = NOMINAL_LAPS ** (1 / 63.0)
    return [0, 1] + disc.geometric_laps(1, n_laps, ratio)


def jobs_for(n_laps, stop_lap=None):
    d = disc.e1_disc()
    ids = np.arange(N_MAIN)
    kw = dict(strobe_laps=strobe_laps(n_laps), elem_laps=elem_laps(n_laps),
              elem_mask_fn=lambda c: c < N_DIAL,
              checkpoint_laps=list(range(PHASE1_LAPS, n_laps, PHASE1_LAPS)))
    if stop_lap:
        kw["stop_lap"] = stop_lap
    return runset.make_jobs("e1", ids, d, JUP, n_laps, **kw)


def decide():
    path = paths.raw("e1", "decision.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # ---- phase 1 ----
    jobs = jobs_for(NOMINAL_LAPS, stop_lap=PHASE1_LAPS)
    t0 = time.time()
    pool.run_chunks("e1_phase1", jobs)
    w1 = time.time() - t0
    ps, rem = 0, 0
    for j in jobs:
        with np.load(j["out"] + ".ckpt.npz") as z:
            ps += int(z["psteps"])
            n_alive = len(z["alive"])
            k_check = j["steps_per_lap"] // j["checks_per_lap"]
            rem += n_alive * (NOMINAL_LAPS * j["checks_per_lap"] - int(z["check"])) * k_check
    rate = ps / w1
    proj_min = (w1 + rem / rate) / 60 + EXPORT_OVERHEAD_MIN
    T = T_MAIN if proj_min <= FALLBACK_LIMIT_MIN else T_FALLBACK
    dec = dict(phase1_laps=PHASE1_LAPS, phase1_years=PHASE1_LAPS * P_G, phase1_wall_s=w1,
               phase1_planet_steps=ps, throughput_planet_steps_per_s=rate,
               remaining_planet_steps_at_50kyr=rem, projected_total_min_at_50kyr=proj_min,
               limit_min=FALLBACK_LIMIT_MIN, fallback_used=bool(T != T_MAIN), T_nominal_yr=T,
               n_laps=laps_of(T), T_exact_yr=laps_of(T) * P_G, processes=N_PROCESSES,
               decided=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(path, "w") as f:
        json.dump(dec, f, indent=1)
    print(f"[e1] projection {proj_min:.1f} min at 50,000 yr -> run length {T:.0f} yr ({dec['n_laps']} laps)", flush=True)
    return dec


def run():
    t0 = time.time()
    dec = decide()
    jobs = jobs_for(dec["n_laps"])
    metas, w2 = pool.run_chunks("e1", jobs)
    # ---- merge ----
    tab = pool.merge([j["out"] for j in jobs], runset.PLANET_FIELDS)
    runset.save_table(paths.raw("e1_planets.npz"), tab, n_laps=dec["n_laps"], T_exact_yr=dec["T_exact_yr"])
    laps = runset.merge_strobes(jobs, N_MAIN, lambda i: i, paths.viz("e1_strobe.npy"))
    # element export (ids < 100,000)
    parts, pid = [], []
    for j in jobs:
        with np.load(j["out"]) as z:
            if len(z["elem_ids"]):
                parts.append(z["elem"])
                pid.append(z["elem_ids"])
                el_laps = z["elem_laps"]
    pid = np.concatenate(pid)
    el = np.concatenate(parts, axis=1)[:, np.argsort(pid)]
    np.savez_compressed(paths.viz("e1_elements_subset.npz"), ids=np.sort(pid), laps=el_laps,
                        t_yr=el_laps * P_G, a=el[..., 0], e=el[..., 1], varpi=el[..., 2], lam=el[..., 3])
    with open(paths.viz("e1_strobe_meta.json"), "w") as f:
        json.dump(dict(file="e1_strobe.npy", shape=[len(laps), N_MAIN, 2], dtype="uint16",
                       laps=[int(x) for x in laps], t_yr=[float(x * P_G) for x in laps],
                       encoding="x,y in the frame that turns with the giant (heliocentric, giant on +x); "
                                "value = round((x + 12) / 24 * 65534); 65535 = planet no longer followed (lost) or outside +/-12 au",
                       column=" = planet id (0..999,999); loss times in e1_map_points.npz"), f, indent=1)
    w_wall = time.time() - t0
    ps = sum(m["psteps"] for m in metas)
    stagelog.record("e1", w_wall, planet_steps=ps, n=N_MAIN, n_laps=dec["n_laps"],
                    phase1_wall_s=dec["phase1_wall_s"], phase2_wall_s=w2)
