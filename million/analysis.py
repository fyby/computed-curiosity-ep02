"""Measurements on per-planet tables (shared by validation, E1, E2)."""
import math

import numpy as np

from . import stats
from .constants import A_G, E_G, SURVIVED, ENCOUNTER, hill

RESONANCES = {"4:1": (4, 1), "3:1": (3, 1), "5:2": (5, 2), "7:3": (7, 3), "2:1": (2, 1)}


def res_a(p, q, a_g=A_G):
    """Kepler distance of the p:q resonance (planet makes p laps while the giant makes q)."""
    return a_g * (q / p) ** (2.0 / 3.0)


def walk_edge(a0, surv, start, stop, step=0.02, win=0.04, level=0.5, min_n=5, consecutive=1):
    """Start at `start`, walk toward `stop` in `step` increments; return the centre of
    the first window (width `win`) whose survival is below `level` (None if never).

    With `consecutive` = k > 1 the window only counts if the next k - 1 adjacent,
    non-overlapping windows toward `stop` are below `level` as well (k windows side by
    side, k x `win` au in all). One narrow dip, such as a single resonance window that
    happens to fall just under 50%, can then no longer define an edge (fact-check B3)."""
    order = np.argsort(a0)
    a_s = a0[order]
    s_s = surv[order].astype(float)
    cs = np.concatenate([[0.0], np.cumsum(s_s)])
    sign = 1 if stop > start else -1
    n_steps = int(abs(stop - start) / step + 1e-9)

    def below(x):
        lo = np.searchsorted(a_s, x - win / 2)
        hi = np.searchsorted(a_s, x + win / 2)
        if hi - lo < min_n:
            return None
        return (cs[hi] - cs[lo]) / (hi - lo) < level

    for k in range(n_steps + 1):
        x = start + sign * k * step
        if all(below(x + sign * j * win) for j in range(consecutive)):
            return float(x)
    return None


def lane(a0, surv, a_g=A_G, level=0.5, boot=True, consecutive=1):
    """Lane edges for the given planets (usually e0 < 0.05): walk in from 2.5 au on each side.
    `consecutive`: see walk_edge (1 = first window below the level, the rule of the validation
    tests V4 and V5; 3 = the rule of RESULTS.json for E1 and every dial run)."""
    def edges(idx):
        i = walk_edge(a0[idx], surv[idx], a_g - 2.5, a_g, level=level, consecutive=consecutive)
        o = walk_edge(a0[idx], surv[idx], a_g + 2.5, a_g, level=level, consecutive=consecutive)
        return i, o
    i, o = edges(np.arange(len(a0)))
    res = dict(inner=i, outer=o, width=(o - i) if (i is not None and o is not None) else None, n=int(len(a0)),
               rule=f"centre of the first of {consecutive} adjacent 0.04-au window(s) all below {level:g} survival, "
                    "walking in 0.02-au steps from 2.5 au inside / outside the giant toward it",
               inner_at_walk_start=bool(i is not None and abs(i - (a_g - 2.5)) < 1e-9),
               outer_at_walk_start=bool(o is not None and abs(o - (a_g + 2.5)) < 1e-9))
    if boot and len(a0):
        res["inner_ci95"] = stats.bootstrap(lambda ix: edges(ix)[0], len(a0))
        res["outer_ci95"] = stats.bootstrap(lambda ix: edges(ix)[1], len(a0))
        res["width_ci95"] = stats.bootstrap(
            lambda ix: (lambda e: (e[1] - e[0]) if None not in e else None)(edges(ix)), len(a0))
    return res


def reach_mask(a, e, m_g, e_g=E_G, a_g=A_G):
    """Starting ellipse can touch the giant's Hill sphere (tilt ignored)."""
    rh = hill(m_g, a_g)
    q_g, Q_g = a_g * (1 - e_g), a_g * (1 + e_g)
    return (a * (1 + e) >= q_g - rh) & (a * (1 - e) <= Q_g + rh)


ARTIFACT_Q_AU = 20.0


def artifact_mask(cls, afin, efin, amin, emax):
    """Planets with an unphysical kick (fact-check B4 / F12; README "Known numerical artifacts").

    Two signatures, both only among OUTWARD exits:
      (a) an element sample with a <= 0 (hyperbolic with |a| under 1 au: tens to more than a
          thousand km/s) while the planet was still being followed;
      (b) beyond 52 au at the event test although the last element sample, at most one sample
          interval (P_G/3 = 4 yr) earlier, was an ordinary orbit with its far point under 20 au.
    Both come from a deep pass by the point-mass giant that the fixed WHFast step (zone D, 0.2 yr)
    did not resolve. The event test then saw the planet outside 52 au instead of inside the Hill
    sphere, so it is counted as lost either way; only its class and its recorded eccentricity
    are wrong. Returns (mask, mask_a, mask_b)."""
    from .constants import OUTWARD
    out = cls == OUTWARD
    m_a = out & ((afin <= 0) | (amin <= 0) | (emax >= 1))
    with np.errstate(invalid="ignore"):
        m_b = out & ~m_a & (afin * (1 + efin) < ARTIFACT_Q_AU)
    return m_a | m_b, m_a, m_b


def stretch_profile(a0, gain, lo=1.56, hi=4.30, width=0.01, exclude=None):
    """Mean of `gain` per `width`-au bin of a0. `exclude`: boolean mask of planets left out
    (the numerical artifacts of artifact_mask). Returns bin centres, mean, count."""
    if exclude is not None:
        a0, gain = a0[~exclude], gain[~exclude]
    edges = np.arange(lo, hi + width / 2, width)
    idx = np.digitize(a0, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    s = np.bincount(idx[ok], weights=gain[ok], minlength=len(edges) - 1)
    c = np.bincount(idx[ok], minlength=len(edges) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / c
    return 0.5 * (edges[1:] + edges[:-1]), mean, c


def stretch_median(a0, gain, lo=1.56, hi=4.30, width=0.01):
    """Median of `gain` per bin (all planets; the median does not feel single outliers)."""
    edges = np.arange(lo, hi + width / 2, width)
    idx = np.digitize(a0, edges) - 1
    med = np.full(len(edges) - 1, np.nan)
    order = np.argsort(idx, kind="stable")
    si, sg = idx[order], gain[order]
    starts = np.searchsorted(si, np.arange(len(edges) - 1))
    ends = np.searchsorted(si, np.arange(len(edges) - 1), side="right")
    for k in range(len(edges) - 1):
        if ends[k] > starts[k]:
            med[k] = np.median(sg[starts[k]:ends[k]])
    return med


def res_peaks(centres, mean, a_g=A_G):
    """For each resonance: peak bin within 0.03 au of the Kepler distance, its value,
    the background (median of the bins 0.05-0.15 au from the peak), the ratio, and
    whether it is a genuine local peak (highest bin within 0.10 au and >= 1.5 x background)."""
    out = {}
    for name, (p, q) in RESONANCES.items():
        a_r = res_a(p, q, a_g)
        w = np.nonzero(np.abs(centres - a_r) <= 0.03 + 1e-9)[0]
        if len(w) == 0 or not np.any(np.isfinite(mean[w])):
            continue
        k = w[np.nanargmax(mean[w])]
        dist = np.abs(centres - centres[k])
        bg_bins = (dist >= 0.05 - 1e-9) & (dist <= 0.15 + 1e-9)
        bg = float(np.nanmedian(mean[bg_bins]))
        near = dist <= 0.10 + 1e-9
        local_max = bool(mean[k] >= np.nanmax(mean[near]) - 1e-15)
        out[name] = dict(nominal_au=a_r, peak_au=float(centres[k]), peak=float(mean[k]), background=bg,
                         factor=float(mean[k] / bg) if bg > 0 else None, offset_au=float(centres[k] - a_r),
                         local_max=local_max, genuine=bool(local_max and bg > 0 and mean[k] >= 1.5 * bg))
    return out


def phase_bins(a0, lam0, surv, a_g=A_G, half_width=0.25, nb=12):
    """Survival of co-orbital planets (|a0 - a_g| < half_width) by starting phase
    lambda - lambda_giant (giant starts at lambda = 0), 12 bins of 30 degrees from -180."""
    m = np.abs(a0 - a_g) < half_width
    ph = np.degrees(np.angle(np.exp(1j * lam0[m])))
    edges = np.linspace(-180, 180, nb + 1)
    rows = []
    for k in range(nb):
        sel = (ph >= edges[k]) & (ph < edges[k + 1])
        rows.append(dict(lo=float(edges[k]), hi=float(edges[k + 1]), **stats.wilson(surv[m][sel].sum(), sel.sum())))
    absph = np.abs(ph)
    troj = (absph >= 30) & (absph < 90)
    other = (absph < 30) | (absph >= 150)
    between = (absph >= 90) & (absph < 150)
    s = surv[m]
    return dict(bins=rows, trojan=stats.wilson(s[troj].sum(), troj.sum()),
                other=stats.wilson(s[other].sum(), other.sum()),
                between=stats.wilson(s[between].sum(), between.sum()),
                coorb_survivors=int(s.sum()), n=int(m.sum()))


def survival_by_a(a0, surv, lo, hi, width, mask=None):
    edges = np.arange(lo, hi + width / 2, width)
    if mask is not None:
        a0, surv = a0[mask], surv[mask]
    idx = np.digitize(a0, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    n = np.bincount(idx[ok], minlength=len(edges) - 1)
    k = np.bincount(idx[ok], weights=surv[ok].astype(float), minlength=len(edges) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = k / n
    return dict(lo=edges[:-1].tolist(), hi=edges[1:].tolist(), n=n.tolist(), surv=frac.tolist())


def lost_by(t_event, t, n):
    """Wilson percentage of planets lost at or before time t."""
    k = int(np.sum(np.isfinite(t_event) & (t_event <= t + 1e-9)))
    return stats.wilson(k, n)


def libration_period(phase, dt, smooth=15):
    """Libration period of each column of phase (radians, NaN after loss) by
    zero crossings of the smoothed angle about its mean. Returns array (NaN if < 2 crossings)."""
    out = np.full(phase.shape[1], np.nan)
    ker = np.ones(smooth) / smooth
    for j in range(phase.shape[1]):
        x = phase[:, j]
        if np.any(~np.isfinite(x)):
            continue
        x = np.unwrap(x.astype(float))
        xs = np.convolve(x, ker, mode="valid")
        y = xs - xs.mean()
        z = np.nonzero(np.sign(y[1:]) != np.sign(y[:-1]))[0]
        if len(z) >= 3:
            # linear interpolation of the crossing times
            tz = (z + y[z] / (y[z] - y[z + 1])) * dt
            out[j] = 2 * (tz[-1] - tz[0]) / (len(tz) - 1)
    return out
