"""Small statistics helpers (no scipy needed)."""
import math

import numpy as np

Z95 = 1.959964


def wilson(k, n, z=Z95):
    """Proportion k/n in percent with its 95% Wilson score interval."""
    k, n = int(k), int(n)
    if n == 0:
        return dict(value=None, ci95=[None, None], k=k, n=n)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return dict(value=100 * p, ci95=[100 * max(0.0, centre - half), 100 * min(1.0, centre + half)], k=k, n=n)


def paired_diff(x, y, z=Z95):
    """Difference of two paired proportions mean(y) - mean(x) (booleans), in points,
    with a 95% interval from the discordant pairs (Wald on the paired difference)."""
    x = np.asarray(x, bool)
    y = np.asarray(y, bool)
    n = len(x)
    b = int(np.sum(x & ~y))
    c = int(np.sum(~x & y))
    d = (c - b) / n
    se = math.sqrt(max(b + c - (c - b) ** 2 / n, 0.0)) / n
    return dict(value=100 * d, ci95=[100 * (d - z * se), 100 * (d + z * se)], discordant=b + c, n=n)


def bootstrap(fn, n_items, n_boot=200, seed=12345):
    """Percentile 95% interval of fn(index_array) over bootstrap resamples."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        v = fn(rng.integers(0, n_items, n_items))
        if v is not None and np.isfinite(v):
            vals.append(v)
    if len(vals) < n_boot // 2:
        return [None, None]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
