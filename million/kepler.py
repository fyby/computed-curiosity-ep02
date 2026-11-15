"""Vectorised osculating elements of test planets (heliocentric, GM of the star)."""
import numpy as np

from .constants import G, M_STAR

MU = G * M_STAR


def a_e(r, v, mu=MU):
    """Semimajor axis and eccentricity from heliocentric position r and velocity v (N,3)."""
    rr = np.sqrt((r * r).sum(1))
    v2 = (v * v).sum(1)
    a = 1.0 / (2.0 / rr - v2 / mu)
    h = np.cross(r, v)
    h2 = (h * h).sum(1)
    e = np.sqrt(np.maximum(0.0, 1.0 - h2 / (mu * a)))
    return a, e


def full(r, v, mu=MU):
    """a, e, inc, longitude of pericentre (varpi) and mean longitude (lambda), radians.

    lambda is computed as Omega + u - f + M (u = argument of latitude), which stays
    well defined for nearly circular orbits. Unbound orbits give NaN angles.
    """
    rr = np.sqrt((r * r).sum(1))
    v2 = (v * v).sum(1)
    a = 1.0 / (2.0 / rr - v2 / mu)
    h = np.cross(r, v)
    hn = np.sqrt((h * h).sum(1))
    inc = np.arccos(np.clip(h[:, 2] / hn, -1, 1))
    Om = np.arctan2(h[:, 0], -h[:, 1])
    Om = np.where(np.hypot(h[:, 0], h[:, 1]) < 1e-14 * hn, 0.0, Om)
    evec = np.cross(v, h) / mu - r / rr[:, None]
    e = np.sqrt((evec * evec).sum(1))
    # unit vectors of the node line and of the in-plane normal to it
    n_hat = np.stack([np.cos(Om), np.sin(Om), np.zeros_like(Om)], 1)
    m_hat = np.cross(h / hn[:, None], n_hat)
    omega = np.arctan2((evec * m_hat).sum(1), (evec * n_hat).sum(1))
    u = np.arctan2((r * m_hat).sum(1), (r * n_hat).sum(1))
    f = u - omega
    with np.errstate(invalid="ignore"):
        E = 2.0 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(f / 2.0))
    M = E - e * np.sin(E)
    varpi = np.mod(Om + omega, 2 * np.pi)
    lam = np.mod(Om + u - f + M, 2 * np.pi)
    bad = a <= 0
    varpi[bad] = np.nan
    lam[bad] = np.nan
    return a, e, inc, varpi, lam


def mean_anomaly_to_true(M, e):
    """Solve Kepler's equation (Newton) and return the true anomaly."""
    E = M + e * np.sin(M)
    for _ in range(30):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
