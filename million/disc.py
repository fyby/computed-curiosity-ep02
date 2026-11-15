"""Starting conditions (the random draws) and the cutting of planets into chunks."""
import math

import numpy as np

from .constants import (A_LO, A_HI, E0_MAX, INC_MAX_DEG, SEED_E1, N_MAIN, ZONES, zone_index)

KEYS = ("a", "e", "inc", "Omega", "omega", "M")


def e1_disc(n=N_MAIN):
    """The E1 disc: numpy.random.default_rng(2_000_001), six arrays drawn in this order:
    a, e, inclination, node, argument of pericentre, mean anomaly. Planet id = index.
    Angles in radians. Ids 0..99,999 are the E2 dial set (a random subset by construction)."""
    rng = np.random.default_rng(SEED_E1)
    a = rng.uniform(A_LO, A_HI, n)
    e = rng.uniform(0.0, E0_MAX, n)
    inc = np.radians(rng.uniform(0.0, INC_MAX_DEG, n))
    Om = rng.uniform(0.0, 2 * math.pi, n)
    om = rng.uniform(0.0, 2 * math.pi, n)
    M = rng.uniform(0.0, 2 * math.pi, n)
    return dict(a=a, e=e, inc=inc, Omega=Om, omega=om, M=M)


def subset(disc, ids):
    return {k: np.asarray(disc[k])[ids] for k in KEYS}


def chunks(ids, a, max_cost=10_000, step_mult=1):
    """Sort `ids` by starting a, split by step-size zone, cut each zone into chunks of
    equal cost (at most max_cost zone-D-equivalent planets, so zone A chunks hold 2,500
    planets, zone D chunks 10,000). Returned zone A first: it is the slowest per planet.
    Yields (zone_name, steps_per_lap, ids_of_chunk)."""
    ids = np.asarray(ids)
    a = np.asarray(a)
    order = np.argsort(a, kind="stable")
    ids, a = ids[order], a[order]
    z = zone_index(a)
    out = []
    for zi, (name, lo, hi, spl) in enumerate(ZONES):
        zid = ids[z == zi]
        if len(zid) == 0:
            continue
        size = max(1, int(max_cost * ZONES[-1][3] / spl))
        n_chunks = max(1, math.ceil(len(zid) / size))
        for part in np.array_split(zid, n_chunks):
            out.append((name, spl * step_mult, np.sort(part)))
    return out


def geometric_laps(first, last, ratio):
    """Laps first*ratio^k rounded, unique, up to `last`, with `last` appended.
    The list for a shorter run is a prefix of the list for a longer one."""
    laps, k = [], 0
    while True:
        lap = int(round(first * ratio ** k))
        if lap >= last:
            break
        if lap > first and (not laps or lap > laps[-1]):
            laps.append(lap)
        k += 1
    return laps + [int(last)]
