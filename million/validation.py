"""Validation gate V1-V9 (SIM_SPEC section 5). Writes raw/validation.json.

V6 needs the main run; it is evaluated in results.py (and pre-checked here on V4's
20,000 planets, for information only).
"""
import json
import math
import os
import time

import numpy as np
import rebound

from . import analysis, disc, engine, ias15, kepler, paths, pool, runset, stats
from .constants import (A_G, E_G, MU_JUPITER, P_G, ZONES, G, M_STAR, A_LO, A_HI, E0_MAX, INC_MAX_DEG,
                        N_MAIN, SEED_VALID, SMOKE, SURVIVED, ENCOUNTER, period, hill)

JUP = dict(m=MU_JUPITER, a=A_G, e=E_G)
SC = 20 if SMOKE else 1                              # smoke test: 20x fewer planets


def laps_for(T, m=MU_JUPITER):
    return int(math.ceil(T / period(m) - 1e-9))


def random_disc(n, seed, e_max=E0_MAX, inc_max=INC_MAX_DEG, a_lo=A_LO, a_hi=A_HI):
    rng = np.random.default_rng(seed)
    return dict(a=rng.uniform(a_lo, a_hi, n), e=rng.uniform(0, e_max, n),
                inc=np.radians(rng.uniform(0, inc_max, n)), Omega=rng.uniform(0, 2 * np.pi, n),
                omega=rng.uniform(0, 2 * np.pi, n), M=rng.uniform(0, 2 * np.pi, n))


def e1_sample(n, seed):
    """n random ids of the E1 disc (sorted) and their starting elements."""
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(N_MAIN, n, replace=False))
    return ids, disc.subset(disc.e1_disc(), ids)


# ----------------------------------------------------------------------------- V1
def v1():
    """(a) E0 control: giant mass 0, 100,000 E1 planets, >= 1,000 yr: a and e unchanged.
    (b) the giant alone for 50,000 yr at each of the four step sizes: period and phase."""
    n = 100_000 // SC
    ids = np.arange(n)
    pl = disc.subset(disc.e1_disc(), ids)
    g0 = dict(m=0.0, a=A_G, e=E_G)
    n_laps = laps_for(1000.0 if not SMOKE else 200.0, 0.0)
    tab, metas, wall, jobs = runset.run("e0", ids, pl, g0, n_laps, strobe_laps=list(range(n_laps + 1)))
    da = np.maximum(np.abs(tab["amax"] / tab["start_a"] - 1), np.abs(tab["amin"] / tab["start_a"] - 1))
    de = np.maximum(np.abs(tab["emax"] - tab["start_e"]), np.abs(tab["emin"] - tab["start_e"]))
    lap_list = runset.merge_strobes(jobs, n, lambda i: i, paths.viz("e0_control_strobe.npy"))
    a_res = dict(n=n, years=n_laps * period(0.0), laps=n_laps, max_rel_da=float(da.max()),
                 max_de=float(de.max()), n_changed_1e9=int(np.sum((da > 1e-9) | (de > 1e-9))),
                 n_lost=int(np.sum(tab["cls"] != SURVIVED)), wall_s=wall)
    a_res["pass"] = bool(a_res["max_rel_da"] < 1e-9 and a_res["max_de"] < 1e-9 and a_res["n_lost"] == 0)

    rows = []
    T = 50_000.0 if not SMOKE else 5_000.0
    P = period()
    mu_g = G * (M_STAR + MU_JUPITER)
    n_kep = 2 * math.pi / P
    for (_, _, _, spl) in ZONES:
        sim = rebound.Simulation()
        sim.G = G                                # 4 pi^2 exactly (see constants.py)
        sim.add(m=M_STAR)
        sim.add(m=MU_JUPITER, a=A_G, e=E_G, M=0.0, primary=sim.particles[0])
        sim.move_to_com()
        engine.configure(sim, P / spl)
        laps = int(round(T / P))
        ts, lams, n_steps = [], [], 0
        for k in range(laps + 1):
            if k:
                sim.steps(spl)
                sim.synchronize()
                n_steps += spl
            p = sim.particles
            r = np.array([[p[1].x - p[0].x, p[1].y - p[0].y, p[1].z - p[0].z]])
            v = np.array([[p[1].vx - p[0].vx, p[1].vy - p[0].vy, p[1].vz - p[0].vz]])
            _, _, _, _, lam = kepler.full(r, v, mu=mu_g)
            ts.append(n_steps * (P / spl))
            lams.append(lam[0])
        ts = np.array(ts)
        # sampled once per lap: add 2 pi per lap, unwrap only the small remainder
        lam = np.unwrap(np.array(lams)) + 2 * math.pi * np.arange(len(lams))
        n_fit = np.polyfit(ts, lam, 1)[0]
        P_sim = 2 * math.pi / n_fit
        # analytic true longitude at the end (pericentre on +x, starts at pericentre)
        f_end = kepler.mean_anomaly_to_true(np.array([n_kep * ts[-1]]), E_G)[0]
        p = sim.particles
        th = math.atan2(p[1].y - p[0].y, p[1].x - p[0].x)
        dphi = math.degrees(np.angle(np.exp(1j * (th - f_end))))
        rows.append(dict(steps_per_lap=spl, years=float(ts[-1]), period_sim=P_sim, period_kepler=P,
                         rel_err=abs(P_sim / P - 1), phase_err_deg=dphi,
                         **{"pass": bool(abs(P_sim / P - 1) < 1e-6 and abs(dphi) < 0.5)}))
    return dict(control=a_res, giant=rows, period_sim_yr=rows[-1]["period_sim"],
                **{"pass": bool(a_res["pass"] and all(r["pass"] for r in rows))})


# ----------------------------------------------------------------------------- V2
def v2():
    n = 20_000 // SC
    pl = random_disc(n, SEED_VALID + 2)
    g = dict(m=MU_JUPITER, a=A_G, e=0.0)
    n_laps = 1000 if not SMOKE else 100
    tab, metas, wall, _ = runset.run("v2", np.arange(n), pl, g, n_laps, hill_factor=2.0, jacobi=True)
    keep = tab["cls"] == SURVIVED
    drift = pool.merge([m["out"] for m in metas], ("jac_drift",))["jac_drift"][keep]
    med, p99 = float(np.median(drift)), float(np.percentile(drift, 99))
    return dict(n=n, laps=n_laps, n_outside_2rh=int(keep.sum()), median_rel_drift=med, p99_rel_drift=p99,
                max_rel_drift=float(drift.max()), wall_s=wall, **{"pass": bool(med < 1e-6 and p99 < 1e-4)})


# ----------------------------------------------------------------------------- V3
def v3():
    n = 5_000 // SC
    ids, pl = e1_sample(n, SEED_VALID + 3)
    n_laps = laps_for(5000.0 if not SMOKE else 1000.0)
    tab, metas, wall_w, _ = runset.run("v3_whfast", ids, pl, JUP, n_laps)
    items = [dict(id=int(i), el={k: float(pl[k][j]) for k in disc.KEYS}, mode="events", T=n_laps * P_G)
             for j, i in enumerate(ids)]
    recs, wall_i = pool.run_items("v3_ias15", items, ias15.run, key="id")
    cls_i = np.array([r["cls"] for r in recs])
    lost_w = tab["cls"] != SURVIVED
    lost_i = cls_i != SURVIVED
    d = stats.paired_diff(lost_w, lost_i)
    agree = stats.wilson(np.sum(lost_w == lost_i), n)
    agree_cls = stats.wilson(np.sum(tab["cls"] == cls_i), n)
    ok = d["ci95"][0] <= 0 <= d["ci95"][1]
    return dict(n=n, years=n_laps * P_G, lost_whfast=stats.wilson(lost_w.sum(), n),
                lost_ias15=stats.wilson(lost_i.sum(), n), diff_points=d, agreement_lost=agree,
                agreement_class=agree_cls, wall_whfast_s=wall_w, wall_ias15_s=wall_i, **{"pass": bool(ok)})


# ----------------------------------------------------------------------------- V4
def v4():
    n = 20_000 // SC
    ids, pl = e1_sample(n, SEED_VALID + 4)
    n_laps = laps_for(10_000.0 if not SMOKE else 1000.0)
    a, wa, _, _ = runset.run("v4_prod", ids, pl, JUP, n_laps)
    b, wb, _, _ = runset.run("v4_half", ids, pl, JUP, n_laps, step_mult=2)
    la, lb = a["cls"] != SURVIVED, b["cls"] != SURVIVED
    spread = 100 * (lb.mean() - la.mean())
    round_ = a["start_e"] < 0.05
    lane_a = analysis.lane(a["start_a"][round_], ~la[round_], boot=False)
    lane_b = analysis.lane(b["start_a"][round_], ~lb[round_], boot=False)
    d_in = abs(lane_a["inner"] - lane_b["inner"]) if None not in (lane_a["inner"], lane_b["inner"]) else None
    d_out = abs(lane_a["outer"] - lane_b["outer"]) if None not in (lane_a["outer"], lane_b["outer"]) else None
    agree = stats.wilson(np.sum(la == lb), n)
    # spread of other headline numbers (production -> half step), in points
    other = {}
    for name, t in (("lost_1lap", P_G), ("lost_100yr", 100.0), ("lost_1kyr", 1000.0), ("lost_10kyr", n_laps * P_G)):
        other[name] = 100 * (np.mean(b["t_event"] <= t + 1e-9) - np.mean(a["t_event"] <= t + 1e-9))
    reach = analysis.reach_mask(a["start_a"], a["start_e"], MU_JUPITER)
    other["reach_surv"] = 100 * (np.mean(~lb[reach]) - np.mean(~la[reach]))
    other["out_of_reach_lost_share_of_all"] = 100 * (np.mean(lb & ~reach) - np.mean(la & ~reach))
    lam0 = pl["Omega"] + pl["omega"] + pl["M"]
    pa = analysis.phase_bins(a["start_a"], lam0, ~la)
    pb = analysis.phase_bins(b["start_a"], lam0, ~lb)
    other["trojan_surv"] = pb["trojan"]["value"] - pa["trojan"]["value"]
    other["coorb_other_surv"] = pb["other"]["value"] - pa["other"]["value"]
    hil = (a["start_a"] >= 3.93) & (a["start_a"] < 4.01)
    other["hilda_surv"] = 100 * (np.mean(~lb[hil]) - np.mean(~la[hil]))
    # V6 pre-check on 20,000 planets (information only)
    c, m, _ = analysis.stretch_profile(a["start_a"], a["emax"] - a["start_e"])
    pre_v6 = analysis.res_peaks(c, m)
    ok = abs(spread) < 0.5 and d_in is not None and d_out is not None and d_in < 0.04 and d_out < 0.04
    return dict(n=n, years=n_laps * P_G, lost_prod=stats.wilson(la.sum(), n), lost_half=stats.wilson(lb.sum(), n),
                spread_points=spread, agreement=agree, disagree_pct=100 - agree["value"],
                class_agreement=stats.wilson(np.sum(a["cls"] == b["cls"]), n),
                lane_prod=lane_a, lane_half=lane_b, lane_in_diff=d_in, lane_out_diff=d_out,
                headline_spreads=other, v6_precheck=pre_v6, wall_s=wa + wb, **{"pass": bool(ok)})


# ----------------------------------------------------------------------------- V5
V5_MASSES = (1e-4, 3e-4, 1e-3, 3e-3)


def v5():
    n = 5_000 // SC
    rows = []
    for k, mu in enumerate(V5_MASSES):
        rng = np.random.default_rng(SEED_VALID + 50 + k)
        pl = dict(a=rng.uniform(0.55 * A_G, 1.60 * A_G, n), e=np.zeros(n), inc=np.radians(rng.uniform(0, 0.5, n)),
                  Omega=rng.uniform(0, 2 * np.pi, n), omega=rng.uniform(0, 2 * np.pi, n), M=rng.uniform(0, 2 * np.pi, n))
        g = dict(m=mu, a=A_G, e=0.0)
        n_laps = laps_for(10_000.0 if not SMOKE else 2000.0, mu)
        tab, _, wall, _ = runset.run(f"v5_m{k}", np.arange(n), pl, g, n_laps)
        surv = tab["cls"] == SURVIVED
        ln = analysis.lane(tab["start_a"], surv, boot=True)
        rows.append(dict(mu=mu, n=n, laps=n_laps, inner_edge_au=ln["inner"], inner_ci95=ln["inner_ci95"],
                         outer_edge_au=ln["outer"], outer_ci95=ln["outer_ci95"],
                         inner_dist_rel=(A_G - ln["inner"]) / A_G if ln["inner"] else None,
                         outer_dist_rel=(ln["outer"] - A_G) / A_G if ln["outer"] else None, wall_s=wall))
    ok_rows = [r for r in rows if r["inner_dist_rel"]]
    x = np.log([r["mu"] for r in ok_rows])
    y = np.log([r["inner_dist_rel"] for r in ok_rows])
    b, lnC = np.polyfit(x, y, 1)
    # uncertainty of the exponent from the bootstrap intervals of the edges (resample edges)
    rng = np.random.default_rng(55)
    bs = []
    for _ in range(400):
        yy = []
        for r in ok_rows:
            lo, hi = r["inner_ci95"]
            if lo is None:
                yy.append(math.log(r["inner_dist_rel"]))
            else:
                sd = (hi - lo) / 3.92
                yy.append(math.log((A_G - rng.normal(r["inner_edge_au"], sd)) / A_G))
        bs.append(np.polyfit(x, yy, 1))
    bs = np.array(bs)
    out = dict(rows=rows, exponent=float(b), coefficient=float(math.exp(lnC)),
               exponent_ci95=[float(np.percentile(bs[:, 0], 2.5)), float(np.percentile(bs[:, 0], 97.5))],
               coefficient_ci95=[float(math.exp(np.percentile(bs[:, 1], 2.5))), float(math.exp(np.percentile(bs[:, 1], 97.5)))],
               law_exponent=2 / 7, law_coefficient=[1.3, 1.57])
    yo = [r["outer_dist_rel"] for r in rows if r["outer_dist_rel"]]
    if len(yo) == len(rows):
        bo, lnCo = np.polyfit(np.log([r["mu"] for r in rows]), np.log(yo), 1)
        out.update(outer_exponent=float(bo), outer_coefficient=float(math.exp(lnCo)))
    out["pass"] = bool(0.24 <= b <= 0.33 and 1.0 <= math.exp(lnC) <= 1.8)
    return out


# ----------------------------------------------------------------------------- V7
def v7():
    n = 2_000 // SC
    rng = np.random.default_rng(SEED_VALID + 7)
    side = np.where(np.arange(n) < n // 2, 1.0, -1.0)              # +60 (L4) / -60 (L5)
    Om = rng.uniform(0, 2 * np.pi, n)
    varpi = np.radians(60.0) * side
    pl = dict(a=np.full(n, A_G), e=np.full(n, E_G), inc=np.radians(rng.uniform(0, 1.0, n)), Omega=Om,
              omega=varpi - Om, M=np.radians(rng.uniform(-5, 5, n)))
    n_laps = laps_for(5000.0 if not SMOKE else 1000.0)
    tab, metas, wall, jobs = runset.run("v7", np.arange(n), pl, JUP, n_laps, trojan_phase=True)
    surv = tab["cls"] == SURVIVED
    ph = np.concatenate([np.load(j["out"])["phase"] for j in jobs], axis=1)
    idsc = np.concatenate([np.load(j["out"])["ids"] for j in jobs])
    ph = ph[:, np.argsort(idsc)]
    per = analysis.libration_period(ph, P_G / 15)
    per_s = per[surv & np.isfinite(per)]
    med = float(np.median(per_s)) if len(per_s) else None
    amp = np.degrees(np.nanmax(np.abs(np.unwrap(ph[:, surv], axis=0) - np.radians(60) * side[surv]), axis=0))
    return dict(n=n, years=n_laps * P_G, survival=stats.wilson(surv.sum(), n), median_libration_yr=med,
                libration_iqr=[float(np.percentile(per_s, 25)), float(np.percentile(per_s, 75))] if len(per_s) else None,
                n_period_measured=int(len(per_s)), median_max_offset_deg=float(np.median(amp)) if surv.any() else None,
                wall_s=wall, **{"pass": bool(surv.mean() >= 0.95 and med is not None and 135 <= med <= 165)})


# ----------------------------------------------------------------------------- V8
def v8():
    n = 10_000 // SC
    d = disc.e1_disc()
    cand = np.nonzero((d["a"] > 4.0) & (d["a"] < 7.0))[0][:n]
    pl = disc.subset(d, cand)
    n_laps = laps_for(2000.0 if not SMOKE else 500.0)
    one = dict(max_cost=10 * n)                       # one chunk
    r1, _, w1, _ = runset.run("v8_run1", cand, pl, JUP, n_laps, **one)
    r2, _, w2, _ = runset.run("v8_run2", cand, pl, JUP, n_laps, **one)
    sp, _, w3, jobs = runset.run("v8_split", cand, pl, JUP, n_laps, max_cost=n // 2 + 1)
    stop = n_laps // 2
    pool.run_chunks("v8_resume_part1", runset.make_jobs("v8_resume", cand, pl, JUP, n_laps,
                                                        checkpoint_laps=[stop // 2], stop_lap=stop, **one))
    rs, _, w4, _ = runset.run("v8_resume", cand, pl, JUP, n_laps, checkpoint_laps=[stop // 2], **one)

    def same(x, y):
        return bool(np.array_equal(x["cls"], y["cls"]) and np.array_equal(x["t_event"], y["t_event"], equal_nan=True))
    res = dict(n=n, years=n_laps * P_G, n_chunks_split=len(jobs), lost=int(np.sum(r1["cls"] != SURVIVED)),
               repeat_identical=same(r1, r2), split_identical=same(r1, sp), resume_identical=same(r1, rs),
               bit_identical_final_a=bool(np.array_equal(r1["afin"], sp["afin"]) and np.array_equal(r1["afin"], rs["afin"])))
    res["pass"] = bool(res["repeat_identical"] and res["split_identical"] and res["resume_identical"])
    return res


# ----------------------------------------------------------------------------- V9
def v9():
    """Event test. Production (check every P_G/15 with the segment closest-approach test) against
    two references that test after every integration step (P_G/240 ... P_G/60; P_G/120 as in the
    spec is not a whole number of zone-D steps, and changing the step would change the orbits).
    The references integrate bit-identically to production (engine fine_events), so any
    difference comes from the event test alone, not from chaos:
      ref_noseg  every step, positions only (the spec's reference)
      ref_seg    every step, with the segment test
    A disagreement can go two ways: production EARLIER than ref_noseg means the point-sampled
    reference missed a grazing pass that the segment test caught (a weakness of the reference);
    production LATER by more than one check would mean production missed a pass (the failure this
    test looks for). Pass: production later by more than one check interval for < 1% of the
    planets lost in both runs (against both references), and >= 99% agreement within one check
    interval against ref_seg. The spec's literal number (agreement against ref_noseg) is reported."""
    n = 20_000 // SC
    ids, pl = e1_sample(n, SEED_VALID + 9)
    n_laps = laps_for(2000.0 if not SMOKE else 500.0)
    a, _, wa, _ = runset.run("v9_prod", ids, pl, JUP, n_laps)
    tc = P_G / 15 + 1e-9
    out = dict(n=n, years=n_laps * P_G, encounter_prod=int(np.sum(a["cls"] == ENCOUNTER)))
    ok = True
    for name, seg in (("ref_noseg", False), ("ref_seg", True)):
        b, _, wb, _ = runset.run("v9_" + name, ids, pl, JUP, n_laps, fine_events="segment" if seg else "points")
        both = (a["cls"] == ENCOUNTER) & (b["cls"] == ENCOUNTER)
        d = a["t_event"][both] - b["t_event"][both]
        within = stats.wilson(np.sum(np.abs(d) <= tc), both.sum())
        later = stats.wilson(np.sum(d > tc), both.sum())
        earlier = int(np.sum(d < -tc))
        out[name] = dict(encounter_ref=int(np.sum(b["cls"] == ENCOUNTER)), encounter_both=int(both.sum()),
                         within_one_check=within, prod_later_by_more_than_one_check=later,
                         prod_earlier_by_more_than_one_check=earlier,
                         earlier_examples_yr=np.sort(d[d < -tc])[:10].tolist(),
                         survivors_final_a_identical=bool(np.array_equal(a["afin"][(a["cls"] == 0) & (b["cls"] == 0)],
                                                                         b["afin"][(a["cls"] == 0) & (b["cls"] == 0)])),
                         wall_s=wb)
        ok &= later["value"] < 1.0
        if seg:
            ok &= within["value"] >= 99.0
    out["spec_literal_within_one_check_pct"] = out["ref_noseg"]["within_one_check"]["value"]
    out["pass"] = bool(ok)
    return out


TESTS = dict(V1=v1, V2=v2, V3=v3, V4=v4, V5=v5, V7=v7, V8=v8, V9=v9)


def run(only=None):
    path = paths.raw("validation.json")
    res = {}
    if os.path.exists(path):
        with open(path) as f:
            res = json.load(f).get("tests", {})
    for name, fn in TESTS.items():
        if only and name not in only:
            continue
        if name in res and not only:
            print(f"[validation] {name} already done: pass={res[name]['pass']}", flush=True)
            continue
        t0 = time.time()
        r = fn()
        r["wall_total_s"] = time.time() - t0
        res[name] = pool._jsonable(r)
        print(f"[validation] {name}: pass={r['pass']} ({r['wall_total_s']:.0f} s)", flush=True)
        _write(path, res)
    _write(path, res)
    return res


def _write(path, res):
    gate = [k for k in TESTS]
    all_passed = all(k in res and res[k]["pass"] for k in gate)
    with open(path, "w") as f:
        json.dump(dict(all_passed=all_passed, n_passed=sum(1 for k in gate if k in res and res[k]["pass"]),
                       n_tests_gate=len(gate), note="V6 (resonance positions) is evaluated on the main run in results.py",
                       tests=res), f, indent=1)
