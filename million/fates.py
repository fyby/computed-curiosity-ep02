"""E3, the fates of the lost: 1,000 E1 planets of class ENCOUNTER, started again at t = 0
alone with the star and the giant (physical radii), IAS15, up to 200,000 yr."""
import time

import numpy as np

from . import disc, ias15, paths, pool, stagelog
from .constants import ENCOUNTER, SEED_E3, T_FATE, SMOKE

N_FATES = 1000 if not SMOKE else 50


def worker(item):
    r = ias15.run(item)
    np.save(paths.raw("e3_tracks", f"{item['id']:07d}.npy"), r.pop("track"))
    return r


def sample():
    e1 = np.load(paths.raw("e1_planets.npz"))
    enc = np.nonzero(e1["cls"] == ENCOUNTER)[0]
    ids = np.random.default_rng(SEED_E3).choice(e1["ids"][enc], N_FATES, replace=False)
    return ids


def items(ids):
    d = disc.e1_disc()
    return [dict(id=int(i), el={k: float(d[k][i]) for k in disc.KEYS}, mode="fate", T=T_FATE) for i in ids]


def run():
    t0 = time.time()
    ids = sample()
    recs, wall = pool.run_items("e3_fates", items(ids), worker, key="id", every=100)
    stagelog.record("e3", time.time() - t0, n=len(ids), T_yr=T_FATE)
    return recs
