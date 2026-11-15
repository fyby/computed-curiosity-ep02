"""Run a set of test planets around one giant (any experiment) as resumable WHFast chunks."""
import json
import os

import numpy as np

from . import disc, engine, paths, pool
from .constants import R_IN, R_OUT, CHECKS_PER_LAP

PLANET_FIELDS = ("cls", "t_event", "emax", "emin", "qmin", "amin", "amax", "a_mean_last", "afin", "efin",
                 "under", "start_a", "start_e", "start_inc", "start_Omega", "start_omega", "start_M")


def make_jobs(name, ids, pl, giant, n_laps, step_mult=1, checks_per_lap=None, max_cost=10_000, **extra):
    """Chunks of the planets `ids` (pl: dict of arrays aligned with ids).

    checks_per_lap: int, or a function of steps_per_lap (V9 uses one check per step).
    Any extra keyword goes into every job (strobe_laps, segment, hill_factor, ...).
    `elem_mask_fn(ids)` (optional) selects the planets of the element export."""
    ids = np.asarray(ids)
    pos = {int(i): k for k, i in enumerate(ids)}
    elem_mask_fn = extra.pop("elem_mask_fn", None)
    jobs = []
    for ci, (zone, spl, cids) in enumerate(disc.chunks(ids, pl["a"], max_cost=max_cost, step_mult=step_mult)):
        sel = np.array([pos[int(i)] for i in cids])
        cpl = checks_per_lap if checks_per_lap is not None else CHECKS_PER_LAP * step_mult
        if callable(cpl):
            cpl = cpl(spl)
        job = dict(out=paths.raw(name, f"chunk_{ci:03d}.npz"), ids=cids, pl={k: np.asarray(v)[sel] for k, v in pl.items()},
                   giant=giant, steps_per_lap=spl, checks_per_lap=cpl, n_laps=n_laps, r_in=R_IN, r_out=R_OUT,
                   zone=zone)
        job.update(extra)
        if elem_mask_fn is not None:
            job["elem_mask"] = elem_mask_fn(cids)
        jobs.append(job)
    return jobs


def run(name, ids, pl, giant, n_laps, **kw):
    """Run (or resume) and return (merged per-planet table, metas, wall seconds of this call)."""
    jobs = make_jobs(name, ids, pl, giant, n_laps, **kw)
    metas, wall = pool.run_chunks(name, jobs)
    table = pool.merge([j["out"] for j in jobs], PLANET_FIELDS)
    return table, metas, wall, jobs


def merge_strobes(jobs, n_total, id_to_col, out_path):
    """Write the strobe frames of all chunks into one uint16 .npy (frames, n_total, 2),
    columns ordered by `id_to_col` (planet id -> column). Returns the lap list."""
    with np.load(jobs[0]["out"]) as z:
        laps = z["strobe_laps"]
    arr = np.lib.format.open_memmap(out_path + ".tmp.npy", mode="w+", dtype=np.uint16,
                                    shape=(len(laps), n_total, 2))
    arr[:] = engine.STROBE_LOST
    for j in jobs:
        with np.load(j["out"]) as z:
            cols = id_to_col(z["ids"])
            arr[:, cols, :] = z["strobe"]
    arr.flush()
    del arr
    os.replace(out_path + ".tmp.npy", out_path)
    return laps


def save_table(path, table, **meta):
    np.savez_compressed(path, **table)
    if meta:
        with open(path.replace(".npz", ".meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
