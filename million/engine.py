"""The core loop: one star, one giant and a chunk of massless test planets.

WHFast (Wisdom-Holman), democratic heliocentric coordinates, no corrector,
safe_mode = 0. Every `k_check` steps the state is synchronised and tested in
numpy (REBOUND's own collision search with a Python callback is far too slow
for thousands of test planets):

  ENCOUNTER  closest approach to the giant between two consecutive checks is below
             hill_factor x Hill radius. The path between the checks is reconstructed in
             the frame turning with the giant: straight segment, refined with the cubic
             Hermite curve through both end states for planets passing within 1.5 au
             (closest_approach2), so a fast pass between checks is not missed
  INWARD     distance from the star below r_in
  OUTWARD    distance from the star above r_out

A test planet stops at its first event and is removed from the simulation.

Determinism and chunk independence: after every check the integrator is told
that the particles were modified (`did_modify_particles = 1`), so the internal
coordinates are rebuilt from the synchronised state at every check, whether or
not a planet was removed. Test planets have zero mass, so removing one cannot
change any sum the others feel; together this makes every planet's history
independent of which chunk it is in and of checkpoint/resume (validation V8).

The function `run_chunk` is the multiprocessing worker. It writes its own
output file (so large arrays never travel through the process pool) and
returns a small dictionary of bookkeeping numbers.
"""
import math
import os
import time

import numpy as np
import rebound

from . import kepler
from .constants import (G, M_STAR, SAMPLES_PER_LAP, UNDER_RESOLVED_STEPS, SURVIVED, ENCOUNTER,
                        INWARD, OUTWARD, period, hill, A_G)

STROBE_HALF = 12.0          # strobe positions are stored as uint16 over +/- 12 au
STROBE_LOST = 65535         # sentinel: planet not in the simulation any more


def quantize(xy):
    """Rotating-frame (x, y) in au -> uint16 over +/- STROBE_HALF; outside -> sentinel."""
    q = np.round((xy + STROBE_HALF) / (2 * STROBE_HALF) * 65534.0)
    bad = ~np.all((q >= 0) & (q <= 65534), axis=1)
    q = np.clip(q, 0, 65534).astype(np.uint16)
    q[bad] = STROBE_LOST
    return q


def dequantize(q):
    """uint16 strobe -> float32 au, NaN for the sentinel."""
    x = q.astype(np.float32) / 65534.0 * (2 * STROBE_HALF) - STROBE_HALF
    x[q == STROBE_LOST] = np.nan
    return x


def build_sim(giant, pl, dt):
    """Star (1 Msun) + giant (starts at pericentre, pericentre on +x) + test planets.

    giant = dict(m, a, e); pl = dict of arrays a, e, inc, Omega, omega, M
    (heliocentric elements about the star, radians)."""
    sim = rebound.Simulation()
    sim.G = G                                # 4 pi^2 exactly (see constants.py)
    sim.add(m=M_STAR)
    sim.add(m=giant["m"], a=giant["a"], e=giant["e"], inc=0.0, Omega=0.0, omega=0.0, M=0.0,
            primary=sim.particles[0])
    sim.N_active = 2
    for k in range(len(pl["a"])):
        sim.add(a=float(pl["a"][k]), e=float(pl["e"][k]), inc=float(pl["inc"][k]),
                Omega=float(pl["Omega"][k]), omega=float(pl["omega"][k]), M=float(pl["M"][k]),
                primary=sim.particles[0])      # re-fetch: the particle array may move
    sim.move_to_com()
    configure(sim, dt)
    return sim


def configure(sim, dt):
    sim.integrator = "whfast"
    sim.integrator.coordinates = "democraticheliocentric"
    sim.integrator.safe_mode = 0
    sim.integrator.corrector = 0
    sim.dt = dt
    sim.N_active = 2
    sim.testparticle_type = 0


def _jacobi(xyz, vxyz, m_g, n_g):
    """Jacobi constant of each test planet (barycentric frame, giant on a circular orbit)."""
    r1 = np.sqrt(((xyz[2:] - xyz[0]) ** 2).sum(1))
    r2 = np.sqrt(((xyz[2:] - xyz[1]) ** 2).sum(1))
    p, v = xyz[2:], vxyz[2:]
    v2 = (v * v).sum(1)
    hz = p[:, 0] * v[:, 1] - p[:, 1] * v[:, 0]
    return 2 * (G * M_STAR / r1 + G * m_g / r2) + 2 * n_g * hz - v2


def run_chunk(job):
    """Integrate one chunk. See module docstring. Keys of `job`:

    out            output .npz path (written at the end; a sibling .ckpt is used for resume)
    ids            int64 planet ids (only stored, never used in the physics)
    pl             dict of starting elements (arrays, radians)
    giant          dict(m, a, e)
    steps_per_lap  integration steps per giant lap (dt = P_giant / steps_per_lap)
    checks_per_lap event tests per lap (must divide steps_per_lap)
    n_laps         end time in giant laps
    segment        True: segment closest-approach test; False: test the check positions only
    hill_factor    event radius in Hill radii (default 1)
    r_in, r_out    INWARD / OUTWARD radii (au)
    strobe_laps    laps at which rotating-frame positions of all planets are stored
    elem_laps      laps at which (a, e, varpi, lambda) of the planets in elem_mask are stored
    elem_mask      bool array, which planets get the element export
    last_window    years at the end over which the mean semimajor axis is averaged
    checkpoint_laps laps after which state is saved for resume
    stop_lap       optional: stop (after checkpointing) at this lap; run again to continue
    fine_events    validation V9 only: "points" or "segment" -> integrate exactly as production
                   but test events after every single step (keep_unsynchronized synchronisation,
                   which leaves the integration untouched); planets are still removed only at the
                   regular checks, so the trajectories are bit-identical to production
    record         optional dict(every=k checks) -> heliocentric positions of all planets
    trojan_phase   optional True -> lambda - lambda_giant of every planet at every check
    jacobi         optional True -> max relative drift of the Jacobi constant (sampled per lap)
    """
    t_wall0 = time.time()
    out = job["out"]
    ckpt = out + ".ckpt"
    ids = np.asarray(job["ids"], dtype=np.int64)
    n = len(ids)
    giant = job["giant"]
    m_g = giant["m"]
    P = period(m_g, giant["a"])
    spl = int(job["steps_per_lap"])
    cpl = int(job["checks_per_lap"])
    assert spl % cpl == 0
    k_check = spl // cpl
    dt = P / spl
    t_check = P / cpl
    n_laps = int(job["n_laps"])
    n_checks = n_laps * cpl
    spc = cpl // SAMPLES_PER_LAP                        # checks per element sample
    assert cpl % SAMPLES_PER_LAP == 0
    rh = job.get("hill_factor", 1.0) * hill(m_g, giant["a"])
    rh2 = rh * rh
    r_in2 = job["r_in"] ** 2
    r_out2 = job["r_out"] ** 2
    segment = job.get("segment", True)
    T_end = n_laps * P
    t_last0 = T_end - job.get("last_window", 1000.0)
    tq_min = UNDER_RESOLVED_STEPS * dt
    stop_check = min(n_checks, int(job.get("stop_lap", n_laps)) * cpl)

    strobe_laps = np.asarray(job.get("strobe_laps", []), dtype=np.int64)
    strobe_pos = {int(l): i for i, l in enumerate(strobe_laps)}
    elem_laps = np.asarray(job.get("elem_laps", []), dtype=np.int64)
    elem_pos = {int(l): i for i, l in enumerate(elem_laps)}
    elem_mask = np.asarray(job.get("elem_mask", np.zeros(n, bool)), bool)
    elem_local = np.nonzero(elem_mask)[0]
    elem_slot = -np.ones(n, np.int64)
    elem_slot[elem_local] = np.arange(len(elem_local))
    ckpt_laps = set(int(x) for x in job.get("checkpoint_laps", []))
    rec = job.get("record")
    want_phase = job.get("trojan_phase", False)
    want_jac = job.get("jacobi", False)
    n_g = 2 * math.pi / P

    # ---- state (indexed by local planet index 0..n-1) ----
    S = dict(
        cls=np.zeros(n, np.int8), t_event=np.full(n, np.nan), emax=np.array(job["pl"]["e"], float),
        emin=np.array(job["pl"]["e"], float),
        qmin=np.array(job["pl"]["a"], float) * (1 - np.array(job["pl"]["e"], float)),
        amin=np.array(job["pl"]["a"], float), amax=np.array(job["pl"]["a"], float),
        a_sum=np.zeros(n), a_cnt=np.zeros(n, np.int64), afin=np.array(job["pl"]["a"], float),
        efin=np.array(job["pl"]["e"], float), under=np.zeros(n, bool),
        strobe=np.full((len(strobe_laps), n, 2), STROBE_LOST, np.uint16),
        elem=np.full((len(elem_laps), len(elem_local), 4), np.nan, np.float32),
        jac_drift=np.zeros(n), jac0=np.zeros(n),
    )
    if rec:
        n_rec = n_checks // rec["every"] + 1
        S["rec_xyz"] = np.full((n_rec, n, 3), np.nan, np.float32)
        S["rec_giant"] = np.zeros((n_rec, 3), np.float32)
        S["rec_t"] = np.zeros(n_rec)
        if rec.get("with_v"):
            S["rec_v"] = np.full((n_rec, n, 3), np.nan, np.float32)
    if want_phase:
        S["phase"] = np.full((n_checks + 1, n), np.nan, np.float32)

    if os.path.exists(ckpt + ".npz"):
        # ---- resume ----
        with np.load(ckpt + ".npz") as z:
            for k in S:
                if k in ("strobe", "elem"):
                    # the lap lists may have changed (E1 fallback to a shorter run): copy by lap
                    old = list(z[k + "_laps_saved"])
                    new = list(strobe_laps if k == "strobe" else elem_laps)
                    for i, lap in enumerate(new):
                        if lap in old:
                            S[k][i] = z[k][old.index(lap)]
                else:
                    S[k] = z[k].copy()
            alive = z["alive"].copy()
            prev = z["prev"].copy()
            check = int(z["check"])
            psteps = int(z["psteps"])
            wall_before = float(z["wall"])
            binf = os.path.join(os.path.dirname(ckpt), str(z["binfile"]))
        sim = rebound.Simulation(binf)
        configure(sim, dt)
        sim.did_modify_particles = 1
    else:
        sim = build_sim(giant, job["pl"], dt)
        alive = np.arange(n)                               # local index of particle j+2
        check = 0
        psteps = 0
        wall_before = 0.0
        prev = None

    N = sim.N
    xyz = np.zeros((N, 3))
    vxyz = np.zeros((N, 3))

    def record_lap(lap, xyz, vxyz, Nn):
        """Strobe / element export / Jacobi at an exact lap boundary."""
        star, gi = xyz[0], xyz[1]
        if lap in strobe_pos:
            th = math.atan2(gi[1] - star[1], gi[0] - star[0])
            c, s = math.cos(-th), math.sin(-th)
            hx = xyz[2:Nn, 0] - star[0]
            hy = xyz[2:Nn, 1] - star[1]
            S["strobe"][strobe_pos[lap], alive] = quantize(np.stack([c * hx - s * hy, s * hx + c * hy], 1))
        if lap in elem_pos:
            sel = elem_mask[alive]
            if sel.any():
                a, e, inc, varpi, lam = kepler.full(xyz[2:Nn][sel] - star, vxyz[2:Nn][sel] - vxyz[0])
                S["elem"][elem_pos[lap], elem_slot[alive[sel]]] = np.stack([a, e, varpi, lam], 1)
        if want_jac:
            C = _jacobi(xyz[:Nn], vxyz[:Nn], m_g, n_g)
            if lap == 0:
                S["jac0"][alive] = C
            else:
                d = np.abs(C - S["jac0"][alive]) / np.abs(S["jac0"][alive])
                S["jac_drift"][alive] = np.maximum(S["jac_drift"][alive], d)

    def record_check(ci, xyz, vxyz, Nn):
        star, gi = xyz[0], xyz[1]
        if rec and ci % rec["every"] == 0:
            ri = ci // rec["every"]
            S["rec_xyz"][ri, alive] = xyz[2:Nn] - star
            S["rec_giant"][ri] = gi - star
            S["rec_t"][ri] = ci * t_check
            if rec.get("with_v"):
                S["rec_v"][ri, alive] = vxyz[2:Nn] - vxyz[0]
        if want_phase:
            _, _, _, _, lam = kepler.full(xyz[2:Nn] - star, vxyz[2:Nn] - vxyz[0])
            _, _, _, _, lamg = kepler.full((gi - star)[None], (vxyz[1] - vxyz[0])[None])
            S["phase"][ci, alive] = np.angle(np.exp(1j * (lam - lamg[0])))

    if check == 0:
        sim.serialize_particle_data(xyz=xyz, vxvyvz=vxyz)
        prev = _rot_pv(xyz[:N], vxyz[:N])
        record_lap(0, xyz, vxyz, N)
        record_check(0, xyz, vxyz, N)

    fine = job.get("fine_events")          # validation V9 only (see docstring of run_chunk)
    if fine:
        flag_t = np.full(n, np.nan)
        flag_c = np.zeros(n, np.int8)
        xs = np.zeros((N, 3))
    t_seg = time.time()
    while check < stop_check:
        if fine:
            # same integration as production (state rebuilt only every k_check steps), but the
            # synchronised positions are read and tested after every single step
            Nn = sim.N
            prev_s = prev[:, :3].copy()
            for sub in range(k_check):
                # sim.steps() synchronises at its end. For the first k_check-1 single steps that
                # synchronisation must not touch the integrator state (keep_unsynchronized = 1);
                # the last one synchronises normally, exactly like production's steps(k_check).
                # Verified bit-identical (survivors_final_a_identical in validation.json).
                sim.integrator.keep_unsynchronized = 1 if sub < k_check - 1 else 0
                sim.steps(1)
                sim.serialize_particle_data(xyz=xs[:Nn])
                ds = _rot(xs[2:Nn] - xs[1], xs[0], xs[1])
                if fine == "segment":
                    dd = ds - prev_s
                    den = (dd * dd).sum(1)
                    den[den == 0] = 1.0
                    u = np.clip(-(prev_s * dd).sum(1) / den, 0.0, 1.0)
                    cp = prev_s + u[:, None] * dd
                    g2 = (cp * cp).sum(1)
                else:
                    g2 = (ds * ds).sum(1)
                hrs = xs[2:Nn] - xs[0]
                rr = (hrs * hrs).sum(1)
                evs = np.where(g2 < rh2, ENCOUNTER, np.where(rr < r_in2, INWARD, np.where(rr > r_out2, OUTWARD, 0)))
                new = (evs > 0) & np.isnan(flag_t[alive])
                flag_t[alive[new]] = (check * k_check + sub + 1) * dt
                flag_c[alive[new]] = evs[new]
                prev_s = ds
            sim.integrator.keep_unsynchronized = 0
        else:
            sim.steps(k_check)
        sim.synchronize()
        check += 1
        Nn = sim.N
        psteps += k_check * (Nn - 2)
        t = check * t_check
        sim.serialize_particle_data(xyz=xyz[:Nn], vxvyvz=vxyz[:Nn])
        star, gi = xyz[0], xyz[1]
        d = _rot_pv(xyz[:Nn], vxyz[:Nn])
        if segment:
            dg2 = closest_approach2(prev, d, t_check)
        else:
            dg2 = (d[:, :3] ** 2).sum(1)
        hr = xyz[2:Nn] - star
        r2 = (hr * hr).sum(1)
        ev = np.where(dg2 < rh2, ENCOUNTER, np.where(r2 < r_in2, INWARD, np.where(r2 > r_out2, OUTWARD, SURVIVED)))
        if fine:
            ev = np.where(np.isnan(flag_t[alive]), SURVIVED, flag_c[alive]).astype(ev.dtype)
        lost = np.nonzero(ev)[0]
        keep = ev == 0

        # element sample of the planets that are still on their orbits
        if check % spc == 0:
            a, e = kepler.a_e(hr[keep], vxyz[2:Nn][keep] - vxyz[0])
            li = alive[keep]
            q = a * (1 - e)
            S["emax"][li] = np.maximum(S["emax"][li], e)
            S["emin"][li] = np.minimum(S["emin"][li], e)
            S["qmin"][li] = np.minimum(S["qmin"][li], q)
            S["amin"][li] = np.minimum(S["amin"][li], a)
            S["amax"][li] = np.maximum(S["amax"][li], a)
            S["afin"][li] = a
            S["efin"][li] = e
            with np.errstate(invalid="ignore"):
                tq = 2 * math.pi * np.sqrt(np.abs(q) ** 3 / (G * M_STAR))
            S["under"][li] |= tq < tq_min
            if t >= t_last0 - 1e-9:
                S["a_sum"][li] += a
                S["a_cnt"][li] += 1
        if len(lost):
            li = alive[lost]
            S["cls"][li] = ev[lost]
            S["t_event"][li] = flag_t[li] if fine else t
        # removal (REBOUND keeps the order; remove from the top down, mirror on `alive`)
        if len(lost):
            for j in lost[::-1]:
                sim.remove(int(j) + 2)
            alive = alive[keep]
            d = d[keep]
            Nn = sim.N
            xyz[2:Nn] = xyz[2:Nn + len(lost)][keep]
            vxyz[2:Nn] = vxyz[2:Nn + len(lost)][keep]
        sim.did_modify_particles = 1     # rebuild internal coordinates at every check (see docstring)
        prev = d
        record_check(check, xyz, vxyz, Nn)
        if check % cpl == 0:
            lap = check // cpl
            record_lap(lap, xyz, vxyz, Nn)
            if lap in ckpt_laps or check == stop_check < n_checks:
                _save_ckpt(ckpt, sim, S, alive, prev, check, psteps, wall_before + time.time() - t_seg,
                           strobe_laps, elem_laps)

    wall = wall_before + time.time() - t_seg
    done = check >= n_checks
    if done:
        a_mean = np.where(S["a_cnt"] > 0, S["a_sum"] / np.maximum(S["a_cnt"], 1), np.nan)
        res = dict(ids=ids, cls=S["cls"], t_event=S["t_event"], emax=S["emax"], emin=S["emin"], qmin=S["qmin"],
                   amin=S["amin"], amax=S["amax"], a_mean_last=a_mean, afin=S["afin"], efin=S["efin"],
                   under=S["under"], strobe=S["strobe"], strobe_laps=strobe_laps, elem=S["elem"],
                   elem_laps=elem_laps, elem_ids=ids[elem_local])
        for k in ("a", "e", "inc", "Omega", "omega", "M"):
            res["start_" + k] = np.asarray(job["pl"][k])
        if want_jac:
            res["jac_drift"] = S["jac_drift"]
        if rec:
            res["rec_xyz"], res["rec_giant"], res["rec_t"] = S["rec_xyz"], S["rec_giant"], S["rec_t"]
            if rec.get("with_v"):
                res["rec_v"] = S["rec_v"]
        if want_phase:
            res["phase"] = S["phase"]
        tmp = out + ".tmp.npz"
        np.savez(tmp, **res)
        os.replace(tmp, out)
        _clear_ckpt(ckpt)
    return dict(out=out, done=done, n=n, alive=int(len(alive)), check=check, n_checks=n_checks,
                psteps=psteps, wall=wall, setup_wall=time.time() - t_wall0 - (wall - wall_before),
                dt=dt, steps_per_lap=spl, remaining_steps=int(len(alive)) * (n_checks - check) * k_check)


def _rot_pv(xyz, vxyz):
    """(N-2, 6): position and velocity of each planet relative to the giant, in the frame
    that turns with the giant (about z, by minus the giant's heliocentric longitude)."""
    star, gi = xyz[0], xyz[1]
    gx, gy = gi[0] - star[0], gi[1] - star[1]
    gvx, gvy = vxyz[1, 0] - vxyz[0, 0], vxyz[1, 1] - vxyz[0, 1]
    w = (gx * gvy - gy * gvx) / (gx * gx + gy * gy)          # giant's angular rate
    th = math.atan2(gy, gx)
    c, s = math.cos(th), math.sin(th)
    d = xyz[2:] - gi
    v = vxyz[2:] - vxyz[1]
    out = np.empty((len(d), 6))
    out[:, 0] = c * d[:, 0] + s * d[:, 1]
    out[:, 1] = -s * d[:, 0] + c * d[:, 1]
    out[:, 2] = d[:, 2]
    out[:, 3] = c * v[:, 0] + s * v[:, 1] + w * out[:, 1]
    out[:, 4] = -s * v[:, 0] + c * v[:, 1] - w * out[:, 0]
    out[:, 5] = v[:, 2]
    return out


def closest_approach2(a, b, h, n_sub=16, near=1.5):
    """Squared closest approach to the giant between two checks.

    a, b: (N, 6) rotating-frame position+velocity at the two checks, h: interval (yr).
    First the straight segment between the end points; for planets that come within
    `near` au of the giant on it, the path is refined with the cubic Hermite curve
    through both ends (positions and velocities), sampled at n_sub points. In the turning
    frame the relative path is smooth, so this catches fast and curved passes alike."""
    p0, p1 = a[:, :3], b[:, :3]
    dd = p1 - p0
    den = (dd * dd).sum(1)
    den[den == 0] = 1.0
    u = np.clip(-(p0 * dd).sum(1) / den, 0.0, 1.0)
    c = p0 + u[:, None] * dd
    d2 = (c * c).sum(1)
    cand = np.nonzero(d2 < near * near)[0]
    if len(cand):
        q0, q1 = p0[cand], p1[cand]
        m0, m1 = a[cand, 3:] * h, b[cand, 3:] * h
        best = np.minimum((q0 * q0).sum(1), (q1 * q1).sum(1))
        for k in range(1, n_sub):
            t = k / n_sub
            t2, t3 = t * t, t * t * t
            p = (2 * t3 - 3 * t2 + 1) * q0 + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * q1 + (t3 - t2) * m1
            best = np.minimum(best, (p * p).sum(1))
        d2[cand] = best
    return d2


def _rot(d, star, gi):
    """Planet-minus-giant vectors turned into the frame that rotates with the giant
    (about the z axis, by minus the giant's heliocentric longitude). The segment test is
    done in this frame, where the relative path between two checks is nearly straight;
    in the inertial frame the whole pattern turns by ~24 degrees per check interval and
    the straight chord would cut inside the true path."""
    th = math.atan2(gi[1] - star[1], gi[0] - star[0])
    c, s = math.cos(th), math.sin(th)
    out = np.empty_like(d)
    out[:, 0] = c * d[:, 0] + s * d[:, 1]
    out[:, 1] = -s * d[:, 0] + c * d[:, 1]
    out[:, 2] = d[:, 2]
    return out


def _save_ckpt(ckpt, sim, S, alive, prev, check, psteps, wall, strobe_laps, elem_laps):
    """Write the REBOUND state under a name that contains the check number, then the
    numpy bookkeeping that points to it (atomic rename), then delete older states."""
    binfile = os.path.basename(ckpt) + f".{check}.bin"
    path = os.path.join(os.path.dirname(ckpt), binfile)
    if os.path.exists(path):
        os.remove(path)
    sim.save_to_file(path)
    np.savez(ckpt + ".tmp.npz", alive=alive, prev=prev, check=check, psteps=psteps, wall=wall,
             binfile=binfile, strobe_laps_saved=strobe_laps, elem_laps_saved=elem_laps, **S)
    os.replace(ckpt + ".tmp.npz", ckpt + ".npz")
    _clear_ckpt(ckpt, keep=binfile)


def _clear_ckpt(ckpt, keep=None):
    d, base = os.path.dirname(ckpt), os.path.basename(ckpt)
    for fn in os.listdir(d):
        if fn.startswith(base + ".") and fn.endswith(".bin") and fn != keep:
            os.remove(os.path.join(d, fn))
    if keep is None and os.path.exists(ckpt + ".npz"):
        os.remove(ckpt + ".npz")
