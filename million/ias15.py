"""One test planet at a time with IAS15 (adaptive, 15th order, default settings).

Used for
* validation V3: the same outcome classes as the WHFast runs (Hill sphere,
  inner and outer radius, segment test every P_G/15), no physical radii;
* experiment E3, the fates of lost planets: physical radii for star and giant,
  REBOUND's direct collision search with `halt`, ejection test every 50 yr,
  first Hill-sphere entry recorded (but the run continues), 200,000 yr.

All `integrate` calls use exact_finish_time = 0, so IAS15 chooses every step
on its own and the trajectory does not depend on where we stop to look. A
replay that stops more often (dense output for the films) therefore follows
the identical path; replays.py checks this by comparing fate and time.
"""
import math
import warnings

import numpy as np
import rebound

from . import kepler
from .engine import closest_approach2
from .constants import (G, M_STAR, MU_JUPITER, A_G, E_G, R_STAR, R_GIANT, R_IN, R_OUT, period, hill,
                        SURVIVED, ENCOUNTER, INWARD, OUTWARD)

warnings.simplefilter("ignore", RuntimeWarning)

EJECT_R = 200.0           # au
EJECT_EVERY = 50.0        # yr
TRACK_EVERY = 100.0       # yr, thinned track of every E3 planet


def build(el, radii, giant=None):
    giant = giant or dict(m=MU_JUPITER, a=A_G, e=E_G)
    sim = rebound.Simulation()
    sim.G = G                                # 4 pi^2 exactly (see constants.py)
    sim.add(m=M_STAR, r=R_STAR if radii else 0.0)
    sim.add(m=giant["m"], a=giant["a"], e=giant["e"], M=0.0, primary=sim.particles[0],
            r=R_GIANT if radii else 0.0)
    sim.N_active = 2
    sim.add(a=el["a"], e=el["e"], inc=el["inc"], Omega=el["Omega"], omega=el["omega"], M=el["M"],
            primary=sim.particles[0])
    sim.move_to_com()
    sim.integrator = "ias15"
    if radii:
        sim.collision = "direct"
        sim.collision_resolve = "halt"
    return sim


def _state(sim):
    p = sim.particles
    s = np.array([[q.x, q.y, q.z, q.vx, q.vy, q.vz] for q in (p[0], p[1], p[2])])
    return s


def _rel_rot(s):
    """Planet minus giant (position, velocity) in the frame turning with the giant."""
    from .engine import _rot_pv
    return _rot_pv(s[:, :3], s[:, 3:])


def run(item):
    """item: dict(id, el (dict of starting elements), mode 'events' or 'fate', T,
    optional record_dt (dense output), optional record_until)."""
    el = item["el"]
    mode = item["mode"]
    fate_mode = mode == "fate"
    sim = build(el, radii=fate_mode)
    P = period()
    rh = hill()
    tc = P / 15.0
    T = float(item["T"])
    rec_dt = item.get("record_dt")
    rec_until = item.get("record_until", T)
    a0, e0 = el["a"], el["e"]

    s = _state(sim)
    prev = _rel_rot(s)
    t_prev = 0.0
    next_check, next_eject, next_track = tc, EJECT_EVERY, TRACK_EVERY
    next_rec = 0.0 if rec_dt else math.inf
    t_entry = None
    t_change = None
    changed = None
    cls, t_event = SURVIVED, None
    fate, t_fate = "BOUND", None
    rec_t, rec_s, track = [], [], []
    n_calls = 0

    def elements(s):
        a, e, inc, varpi, lam = kepler.full((s[2, :3] - s[0, :3])[None], (s[2, 3:] - s[0, 3:])[None])
        return a[0], e[0], varpi[0]

    try:
        while sim.t < T:
            if rec_dt and next_rec <= sim.t and sim.t <= rec_until:
                rec_t.append(sim.t)
                rec_s.append(_state(sim)[:, :3])
                next_rec = (math.floor(sim.t / rec_dt) + 1) * rec_dt
            target = min(next_check if (t_entry is None or not fate_mode) else math.inf,
                         next_eject if fate_mode else math.inf,
                         next_track if fate_mode else math.inf,
                         t_change if (t_change is not None and changed is None) else math.inf,
                         next_rec if (rec_dt and next_rec <= rec_until) else math.inf, T)
            sim.integrate(target, exact_finish_time=0)
            n_calls += 1
            t = sim.t
            s = _state(sim)
            if t >= next_check and (t_entry is None or not fate_mode):
                d = _rel_rot(s)
                hit = float(closest_approach2(prev, d, t - t_prev)[0]) < rh * rh
                prev, t_prev = d, t
                next_check = (math.floor(t / tc) + 1) * tc
                if fate_mode:
                    if hit:
                        t_entry = t
                        t_change = t + 100.0
                else:
                    r2 = float(((s[2, :3] - s[0, :3]) ** 2).sum())
                    if hit:
                        cls, t_event = ENCOUNTER, t
                    elif r2 < R_IN ** 2:
                        cls, t_event = INWARD, t
                    elif r2 > R_OUT ** 2:
                        cls, t_event = OUTWARD, t
                    if cls != SURVIVED:
                        break
            if fate_mode and t_change is not None and changed is None and t >= t_change:
                a, e, _ = elements(s)
                changed = bool(abs(a - a0) / a0 > 0.05 or abs(e - e0) > 0.05 or a <= 0)
            if fate_mode and t >= next_track:
                a, e, varpi = elements(s)
                h = s[2, :3] - s[0, :3]
                track.append([t, h[0], h[1], h[2], a, e, varpi])
                next_track = (math.floor(t / TRACK_EVERY) + 1) * TRACK_EVERY
            if fate_mode and t >= next_eject:
                next_eject = (math.floor(t / EJECT_EVERY) + 1) * EJECT_EVERY
                m_tot = sim.particles[0].m + sim.particles[1].m
                r = math.sqrt(float((s[2, :3] ** 2).sum()))          # barycentric (move_to_com)
                v2 = float((s[2, 3:] ** 2).sum())
                if r > EJECT_R and 0.5 * v2 - G * m_tot / r > 0:
                    fate, t_fate = "EJECTED", t
                    break
    except rebound.Collision:
        s = _state(sim)
        p = sim.particles
        gaps = {}
        for i, j in ((0, 2), (1, 2), (0, 1)):
            d = math.sqrt(float(((s[i, :3] - s[j, :3]) ** 2).sum()))
            gaps[(i, j)] = d - (p[i].r + p[j].r)
        pair = min(gaps, key=gaps.get)
        fate = {(0, 2): "HIT_STAR", (1, 2): "HIT_GIANT", (0, 1): "STAR_GIANT"}[pair]
        t_fate = sim.t
    if rec_dt:
        rec_t.append(sim.t)
        rec_s.append(_state(sim)[:, :3])

    out = dict(id=int(item["id"]), n_calls=n_calls, t_end=sim.t, steps=int(sim.steps_done))
    if fate_mode:
        s = _state(sim)
        a, e, _ = elements(s)
        q_g, Q_g = A_G * (1 - E_G), A_G * (1 + E_G)
        crossing = bool(a > 0 and a * (1 + e) >= q_g - hill() and a * (1 - e) <= Q_g + hill())
        if changed is None and fate != "BOUND":
            changed = True                    # orbit ended (impact or ejection) before the 100-yr mark
        out.update(fate=fate, t_fate=t_fate, t_entry=t_entry, entered=t_entry is not None,
                   orbit_changed=changed, a_final=float(a), e_final=float(e), crossing=crossing,
                   track=np.array(track, np.float32).reshape(-1, 7))
    else:
        out.update(cls=int(cls), t_event=t_event)
    if rec_dt:
        out["rec_t"] = np.array(rec_t)
        out["rec_xyz"] = np.array(rec_s)            # (n, 3 bodies, 3), barycentric
    return out
