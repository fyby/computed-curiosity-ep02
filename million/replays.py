"""E4, replays and dense clips for the hero shots (viz/replays/).

The selection rules are written to viz/replays/index.json *before* any clip is
computed; the chosen ids are added after selection. Every clip is a new
computation with the stated settings. Clips that re-run a single planet with
exactly the settings of its original run (res31: WHFast as E1; ejection,
giant_hit, wanderer: IAS15 as E3) reproduce the original history bit for bit;
the file records that check (`reproduces_original`).
"""
import json
import math
import os
import time

import numpy as np

from . import disc, engine, ias15, paths, pool, stagelog
from .constants import (A_G, E_G, MU_JUPITER, P_G, R_IN, R_OUT, SURVIVED, INWARD, ENCOUNTER, SMOKE, period)
from .fates import N_FATES

RULES = {
    "first_laps.npz": "the first 5,000 ids (ascending) with starting a in 4.2-6.4 au; WHFast, dt = P_G/60, "
                      "event test and position output every P_G/60, 11 laps (130.5 yr)",
    "control.npz": "the same 5,000 planets with the giant's mass set to zero, same settings; "
                   "plus pointer to the E0 strobes of 100,000 planets (e0_control_strobe.npy)",
    "trojans.npz": "the first 2,000 ids (ascending) among E1 survivors that started with |a0 - a_G| < 0.25 au "
                   "and 30-90 degrees ahead of or behind the giant; WHFast, dt = P_G/60, output every P_G/60, "
                   "51 laps (605 yr), positions in the frame that turns with the giant",
    "res31.npz": "among E1 INWARD exits that started within 0.03 au of 2.502 au, the one with the median exit "
                 "time (lower median if even); if none, the survivor with the median eccentricity gain in the "
                 "3:1 peak bin. Re-run alone with the E1 settings; a, e, varpi every P_G/3",
    "ejection.npz": "the E3 EJECTED planet with the median ejection time (lower median if even); "
                    "IAS15 re-run, star/giant/planet positions every P_G/120 until ejection",
    "giant_hit.npz": "the E3 HIT_GIANT planet with the median impact time (lower median if even); "
                     "IAS15 re-run, positions every P_G/120 until impact",
    "wanderer.npz": "the E3 BOUND planet with the `crossing` flag and the median final semimajor axis "
                    "(lower median if even); IAS15 re-run, positions every 1 yr for 200,000 yr, "
                    "a and e every 100 yr from the E3 run",
}


def _index_path():
    return paths.viz("replays", "index.json")


def write_rules():
    idx = {}
    if os.path.exists(_index_path()):
        with open(_index_path()) as f:
            idx = json.load(f)
    idx.setdefault("rules_written", time.strftime("%Y-%m-%d %H:%M:%S"))
    idx.setdefault("clips", {})
    for k, v in RULES.items():
        idx["clips"].setdefault(k, {})["rule"] = v
    with open(_index_path(), "w") as f:
        json.dump(idx, f, indent=1)
    return idx


def _note(name, **info):
    with open(_index_path()) as f:
        idx = json.load(f)
    idx["clips"][name].update(info)
    with open(_index_path(), "w") as f:
        json.dump(pool._jsonable(idx), f, indent=1)


def lower_median_index(values):
    order = np.argsort(values, kind="stable")
    return order[(len(values) - 1) // 2]


def _whfast_clip(name, ids, giant, n_laps, cpl, extra=None):
    d = disc.e1_disc()
    pl = disc.subset(d, ids)
    zone_spl = 60
    job = dict(out=paths.raw("replays", name + ".npz"), ids=ids, pl=pl, giant=giant, steps_per_lap=zone_spl,
               checks_per_lap=cpl, n_laps=n_laps, r_in=R_IN, r_out=R_OUT, record=dict(every=1))
    if extra:
        job.update(extra)
    if not os.path.exists(job["out"]):
        engine.run_chunk(job)
    return np.load(job["out"])


def first_laps_and_control():
    d = disc.e1_disc()
    n = 5000 if not SMOKE else 500
    ids = np.nonzero((d["a"] >= 4.2) & (d["a"] <= 6.4))[0][:n]
    for name, m in (("first_laps", MU_JUPITER), ("control", 0.0)):
        z = _whfast_clip(name, ids, dict(m=m, a=A_G, e=E_G), 11, 60)
        np.savez_compressed(paths.viz("replays", name + ".npz"), ids=ids, t_yr=z["rec_t"].astype(np.float32),
                            giant_xyz=z["rec_giant"], xyz=z["rec_xyz"].astype(np.float16), cls=z["cls"],
                            t_event=z["t_event"].astype(np.float32),
                            note="heliocentric positions (au), inertial frame; NaN after a planet is lost; "
                                 "cls 0 still on orbit, 1 entered Hill sphere, 2 inward, 3 outward",
                            control_strobe_file="e0_control_strobe.npy" if name == "control" else "")
        _note(name + ".npz", ids_first_last=[int(ids[0]), int(ids[-1])], n=int(len(ids)),
              lost_in_clip=int(np.sum(z["cls"] != SURVIVED)))


def trojans(e1):
    lam0 = e1["start_Omega"] + e1["start_omega"] + e1["start_M"]
    ph = np.degrees(np.angle(np.exp(1j * lam0)))
    sel = (np.abs(e1["start_a"] - A_G) < 0.25) & (np.abs(ph) >= 30) & (np.abs(ph) < 90) & (e1["cls"] == SURVIVED)
    ids = e1["ids"][sel][: (2000 if not SMOKE else 200)]
    z = _whfast_clip("trojans", ids, dict(m=MU_JUPITER, a=A_G, e=E_G), 51, 60)
    g = z["rec_giant"]
    th = np.arctan2(g[:, 1], g[:, 0])
    c, s = np.cos(-th)[:, None], np.sin(-th)[:, None]
    x, y = z["rec_xyz"][..., 0], z["rec_xyz"][..., 1]
    rot = np.stack([c * x - s * y, s * x + c * y], -1).astype(np.float16)
    np.savez_compressed(paths.viz("replays", "trojans.npz"), ids=ids, t_yr=z["rec_t"].astype(np.float32),
                        xy_rot=rot, giant_r=np.hypot(g[:, 0], g[:, 1]).astype(np.float32), cls=z["cls"],
                        t_event=z["t_event"].astype(np.float32),
                        note="positions in the frame that turns with the giant (giant on +x), heliocentric, au")
    _note("trojans.npz", n=int(len(ids)), n_candidates=int(sel.sum()), lost_in_clip=int(np.sum(z["cls"] != SURVIVED)))


def res31(e1, n_laps):
    a31 = A_G * (1 / 3) ** (2 / 3)
    near = np.abs(e1["start_a"] - a31) <= 0.03
    cand = np.nonzero(near & (e1["cls"] == INWARD))[0]
    if len(cand):
        k = cand[lower_median_index(e1["t_event"][cand])]
        how = "INWARD exit, median exit time"
    else:
        cand = np.nonzero(near & (e1["cls"] == SURVIVED))[0]
        k = cand[lower_median_index((e1["emax"] - e1["start_e"])[cand])]
        how = "no INWARD exit: survivor with median eccentricity gain"
    pid = int(e1["ids"][k])
    d = disc.e1_disc()
    pl = disc.subset(d, np.array([pid]))
    from .constants import zone_index, ZONES
    spl = ZONES[int(zone_index(pl["a"][0]))][3]
    job = dict(out=paths.raw("replays", "res31.npz"), ids=np.array([pid]), pl=pl,
               giant=dict(m=MU_JUPITER, a=A_G, e=E_G), steps_per_lap=spl, checks_per_lap=15, n_laps=n_laps,
               r_in=R_IN, r_out=R_OUT, record=dict(every=5, with_v=True))
    if not os.path.exists(job["out"]):
        engine.run_chunk(job)
    z = np.load(job["out"])
    from . import kepler
    r, v = z["rec_xyz"][:, 0].astype(float), z["rec_v"][:, 0].astype(float)
    ok = np.isfinite(r[:, 0])
    a, e, inc, varpi, lam = kepler.full(r[ok], v[ok])
    same = bool(int(z["cls"][0]) == int(e1["cls"][k]) and
                (np.isnan(e1["t_event"][k]) and np.isnan(z["t_event"][0]) or z["t_event"][0] == e1["t_event"][k]))
    np.savez_compressed(paths.viz("replays", "res31.npz"), id=pid, t_yr=z["rec_t"][ok].astype(np.float32),
                        a=a.astype(np.float32), e=e.astype(np.float32), varpi=varpi.astype(np.float32),
                        cls=int(z["cls"][0]), t_event=z["t_event"][0], start_a=pl["a"][0], start_e=pl["e"][0])
    _note("res31.npz", id=pid, selected_as=how, n_candidates=int(len(cand)), cls=int(z["cls"][0]),
          t_event_yr=None if np.isnan(z["t_event"][0]) else float(z["t_event"][0]), reproduces_original=same,
          e_max=float(np.max(e)), start_a=float(pl["a"][0]), start_e=float(pl["e"][0]))


def e3_clips(fates):
    d = disc.e1_disc()
    picks = {}
    ej = [r for r in fates if r["fate"] == "EJECTED"]
    if ej:
        picks["ejection.npz"] = ej[lower_median_index([r["t_fate"] for r in ej])]
    hg = [r for r in fates if r["fate"] == "HIT_GIANT"]
    if hg:
        picks["giant_hit.npz"] = hg[lower_median_index([r["t_fate"] for r in hg])]
    wd = [r for r in fates if r["fate"] == "BOUND" and r["crossing"]]
    if wd:
        picks["wanderer.npz"] = wd[lower_median_index([r["a_final"] for r in wd])]
    for name, r in picks.items():
        pid = r["id"]
        el = {k: float(d[k][pid]) for k in disc.KEYS}
        from .constants import T_FATE
        if name == "wanderer.npz":
            item = dict(id=pid, el=el, mode="fate", T=T_FATE, record_dt=1.0)
        else:
            item = dict(id=pid, el=el, mode="fate", T=T_FATE, record_dt=P_G / 120)
        out = ias15.run(item)
        same = bool(out["fate"] == r["fate"] and out["t_fate"] == r["t_fate"] and out["t_entry"] == r["t_entry"])
        xyz = out["rec_xyz"]                      # (n, 3 bodies, 3) barycentric
        star = xyz[:, 0:1, :]
        extra = {}
        if name == "wanderer.npz":
            tr = np.load(paths.raw("e3_tracks", f"{pid:07d}.npy"))
            extra = dict(track_t_yr=tr[:, 0], track_a=tr[:, 4], track_e=tr[:, 5])
            helio = (xyz[:, 2, :] - xyz[:, 0, :]).astype(np.float32)
            np.savez_compressed(paths.viz("replays", name), id=pid, t_yr=out["rec_t"].astype(np.float32),
                                planet_helio=helio, fate=r["fate"], **extra)
        else:
            np.savez_compressed(paths.viz("replays", name), id=pid, t_yr=out["rec_t"],
                                xyz_bary=xyz.astype(np.float32), bodies=np.array(["star", "giant", "planet"]),
                                fate=r["fate"], t_fate=r["t_fate"], t_entry=r["t_entry"])
        desc = describe_track(out, r)
        _note(name, id=pid, fate=r["fate"], t_fate_yr=r["t_fate"], t_entry_yr=r["t_entry"],
              a_final=r["a_final"], e_final=r["e_final"], reproduces_original=same, n_frames=int(len(out["rec_t"])),
              n_candidates=len(ej if name == "ejection.npz" else hg if name == "giant_hit.npz" else wd),
              track_facts=desc)


def describe_track(out, r):
    """Plain facts read from a replay track, for the [RD-REPLAY] line (no interpretation)."""
    xyz = out["rec_xyz"]
    t = out["rec_t"]
    pg = xyz[:, 2, :] - xyz[:, 1, :]
    ps = xyz[:, 2, :] - xyz[:, 0, :]
    dg = np.sqrt((pg ** 2).sum(1))
    rs = np.sqrt((ps ** 2).sum(1))
    from .constants import R_HILL
    inside = dg < R_HILL
    # separate Hill-sphere passages (entries)
    entries = int(np.sum(inside[1:] & ~inside[:-1]) + (1 if inside[0] else 0))
    facts = dict(duration_yr=float(t[-1]), hill_entries_sampled=entries,
                 closest_to_giant_au=float(dg.min()), closest_to_giant_km=float(dg.min() * 149_597_870.7),
                 t_closest_yr=float(t[np.argmin(dg)]), max_dist_star_au=float(rs.max()),
                 min_dist_star_au=float(rs.min()), first_hill_entry_yr=r["t_entry"])
    if r["fate"] == "EJECTED":
        # time from the last Hill-sphere exit to the ejection test
        last_in = np.nonzero(inside)[0]
        if len(last_in):
            facts["last_hill_exit_yr"] = float(t[last_in[-1]])
            facts["years_from_last_hill_exit_to_200au"] = float(t[-1] - t[last_in[-1]])
        v = np.diff(ps[-50:], axis=0) / np.diff(t[-50:])[:, None]
        facts["speed_at_end_km_s"] = float(np.sqrt((v[-1] ** 2).sum()) * 149_597_870.7 / 3.15576e7)
    return facts


def run():
    t0 = time.time()
    write_rules()
    e1 = np.load(paths.raw("e1_planets.npz"))
    with open(paths.raw("e1", "decision.json")) as f:
        n_laps = json.load(f)["n_laps"]
    first_laps_and_control()
    trojans(e1)
    res31(e1, n_laps)
    with open(paths.raw("e3_fates.jsonl")) as f:
        fates = [json.loads(line) for line in f if line.strip()]
    e3_clips(fates)
    stagelog.record("replays", time.time() - t0)
