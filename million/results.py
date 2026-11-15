"""RESULTS.json (every number the script needs), run_meta.json and the viz/ exports.

Every placeholder: value, unit, 95% interval (Wilson for fractions unless noted),
n, scheme spread where V4 measures one, a definition and the source file.
"""
import glob
import json
import math
import os
import platform
import subprocess
import time

import numpy as np

from . import analysis, paths, stagelog, stats
from .constants import (A_G, E_G, MU_JUPITER, P_G, R_HILL, A_LO, A_HI, SURVIVED, ENCOUNTER, INWARD, OUTWARD,
                        N_MAIN, N_DIAL, SEED_E1, SEED_E3, T_FATE, ZONES, CHECKS_PER_LAP, hill, period)

LANE_CONSECUTIVE = 3   # lane edges need 3 adjacent 0.04-au windows below 50% survival (fact-check B3)

PH = {}          # placeholders
DET = {}         # details


def put(name, value, unit=None, ci95=None, n=None, definition="", source="", spread=None, **extra):
    d = dict(value=value, unit=unit, ci95=ci95, n=n, scheme_spread_points=spread, definition=definition, source=source)
    d.update(extra)
    PH[name] = d


def putw(name, w, definition, source, spread=None, **extra):
    put(name, w["value"], "%", w["ci95"], w["n"], definition, source, spread, k=w["k"], **extra)


def _f(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    return x


# ----------------------------------------------------------------------------- E1
def e1_section(val):
    e1 = np.load(paths.raw("e1_planets.npz"))
    with open(paths.raw("e1", "decision.json")) as f:
        dec = json.load(f)
    T = dec["T_exact_yr"]
    n = len(e1["ids"])
    cls, tev = e1["cls"], e1["t_event"]
    lost = cls != SURVIVED
    surv = ~lost
    a0, e0 = e1["start_a"], e1["start_e"]
    sp = val["V4"]["headline_spreads"]
    spread_main = val["V4"]["spread_points"]
    src = "raw/e1_planets.npz"

    put("T_MAIN_YR", int(dec["T_nominal_yr"]), "yr", None, None,
        f"length of the main run; exactly {dec['n_laps']} laps of the giant = {T:.1f} yr. "
        f"Fallback used: {dec['fallback_used']} (projected {dec['projected_total_min_at_50kyr']:.1f} min at 50,000 yr, limit 85)",
        "raw/e1/decision.json")
    put("LAPS", dec["n_laps"], "laps", None, None, "laps of the giant in the main run (P_G = %.4f yr)" % P_G, "raw/e1/decision.json")
    w = stats.wilson(lost.sum(), n)
    putw("LOST_PCT", w, "E1: share of the 1,000,000 test planets lost (ENCOUNTER, INWARD or OUTWARD) by the end", src,
         spread=spread_main, spread_note="V4 change of the lost fraction at 10,000 yr with half step and half check interval (20,000 planets)")
    putw("SURV_PCT", stats.wilson(surv.sum(), n), "E1: share not lost at the end (never entered the Hill sphere, never left the disk; not the same as still on the starting orbit, see details.e1.survivors_changed)", src, spread=-spread_main)
    put("SURV_COUNT", int(surv.sum()), "planets", [int(round(n * (1 - c / 100))) for c in w["ci95"][::-1]], n,
        "E1: number of test planets not lost at the end: never entered the Hill sphere, never left the disk (interval from LOST_PCT). Not the same as still on the starting orbit, see details.e1.survivors_changed", src)
    for name, t, key in (("LOST_1LAP_PCT", P_G, "lost_1lap"), ("LOST_100YR_PCT", 100.0, "lost_100yr"),
                         ("LOST_1KYR_PCT", 1000.0, "lost_1kyr"), ("LOST_10KYR_PCT", 843 * P_G, "lost_10kyr")):
        putw(name, analysis.lost_by(tev, t, n), f"E1: share lost by t = {t:.1f} yr" +
             (" (843 laps; the dial runs' length)" if key == "lost_10kyr" else ""), src, spread=sp[key])
    tl = tev[lost]
    half = float(np.median(tl))
    boot = stats.bootstrap(lambda ix: float(np.median(tl[ix])), len(tl), n_boot=200)
    put("HALF_LOSS_YR", half, "yr", boot, int(lost.sum()),
        "E1: median loss time of all lost planets (half of all losses of the run had happened by then); "
        "times are on the event-test grid (P_G/15 = 0.79 yr); interval: bootstrap", src)
    late = (tev > 0.8 * T) & lost
    putw("LATE_LOSS_PCT", stats.wilson(late.sum(), n), "E1: share of ALL test planets lost in the last fifth of the run "
         f"(t > {0.8 * T:.0f} yr)", src)
    for c, nm in ((ENCOUNTER, "ENC"), (INWARD, "INWARD"), (OUTWARD, "OUTWARD")):
        k = int(np.sum(cls == c))
        if nm == "ENC":
            putw("ENC_PCT", stats.wilson(k, lost.sum()), "E1: share of the LOST planets whose event was entering the "
                 "giant's Hill sphere (ENCOUNTER)", src, of_all=stats.wilson(k, n))
        else:
            put(f"{nm}_N", k, "planets", None, n, f"E1: number of {nm} exits" +
                (" (includes the ARTIFACT_N planets with an unphysical kick; OUTWARD_GENUINE_N is the rest)"
                 if nm == "OUTWARD" else ""), src)

    # numerical artifacts (fact-check B4 / F12): identified from the saved table, nothing is re-run
    art, art_a, art_b = analysis.artifact_mask(cls, e1["afin"], e1["efin"], e1["amin"], e1["emax"])
    art_rule = ("OUTWARD exits with (a) an element sample with a <= 0 (afin <= 0, amin <= 0 or emax >= 1) or "
                f"(b) a last element sample (at most P_G/3 = {P_G / 3:.2f} yr before the event) whose far point "
                f"a(1+e) was under {analysis.ARTIFACT_Q_AU:g} au although the event test then found the planet "
                "beyond 52 au. Cause: a deep pass by the point-mass giant not resolved by the fixed WHFast step. "
                "They are counted as lost, which they would be anyway (they went deep inside the Hill sphere); "
                "only their class (OUTWARD instead of ENCOUNTER) and their recorded eccentricity are wrong")
    put("ARTIFACT_N", int(art.sum()), "planets", None, n, "E1: planets with an unphysical kick. " + art_rule, src,
        share_of_all_pct=100.0 * art.sum() / n, unbound_sample_n=int(art_a.sum()), jump_n=int(art_b.sum()))
    put("OUTWARD_GENUINE_N", int(np.sum(cls == OUTWARD) - art.sum()), "planets", None, n,
        "E1: OUTWARD exits without the ARTIFACT_N planets", src)

    # reach
    reach = analysis.reach_mask(a0, e0, MU_JUPITER)
    putw("REACH_PCT", stats.wilson(reach.sum(), n), "share of the sampled disc whose starting ellipse can touch the "
         "giant's Hill sphere (Q >= q_G - r_H and q <= Q_G + r_H, tilt ignored); exact count of the starting conditions, "
         "our own arithmetic (draw in --muted)", src)
    putw("REACH_SURV_PCT", stats.wilson(np.sum(reach & surv), reach.sum()), "E1: survival of the planets IN reach", src,
         spread=sp["reach_surv"])
    oor = lost & ~reach
    put("OUT_OF_REACH_LOST_N", int(oor.sum()), "planets", None, n, "E1: lost planets that started out of reach", src,
        share_of_all=stats.wilson(oor.sum(), n), spread_points_of_all=sp["out_of_reach_lost_share_of_all"])
    putw("OUT_OF_REACH_LOST_PCT", stats.wilson(oor.sum(), lost.sum()), "E1: share of the LOST planets that started "
         "out of reach", src)
    q_g, Q_g = A_G * (1 - E_G), A_G * (1 + E_G)
    put("REACH_LANE_WIDTH", (Q_g + R_HILL) - (q_g - R_HILL), "au", None, None,
        f"width of the 'simple guess' lane for round orbits: {q_g - R_HILL:.3f} to {Q_g + R_HILL:.3f} au [CALC]", "constants")

    # the lane (round orbits)
    rnd = e0 < 0.05
    ln = analysis.lane(a0[rnd], surv[rnd], level=0.5, consecutive=LANE_CONSECUTIVE)
    ln_fw = analysis.lane(a0[rnd], surv[rnd], level=0.5, boot=False)
    ln["first_window_rule"] = dict(inner=ln_fw["inner"], outer=ln_fw["outer"])
    ln1 = analysis.lane(a0[rnd], surv[rnd], level=0.01, boot=False)
    put("LANE_IN_AU", ln["inner"], "au", ln["inner_ci95"], ln["n"], "E1, round orbits (e0 < 0.05): inner lane edge = "
        f"centre of the first of {LANE_CONSECUTIVE} adjacent 0.04-au windows that all have survival below 50%, walking "
        "from 2.5 au inside the giant toward it in 0.02-au steps (for E1 the older first-window rule gives the same "
        "edge); interval: bootstrap", src, spread_au=val["V4"]["lane_in_diff"],
        spread_note="V4 used the first-window rule")
    put("LANE_OUT_AU", ln["outer"], "au", ln["outer_ci95"], ln["n"], "E1, round orbits: outer lane edge (same walk "
        "from 2.5 au outside)", src, spread_au=val["V4"]["lane_out_diff"])
    put("LANE_WIDTH", ln["width"], "au", ln["width_ci95"], ln["n"], "LANE_OUT_AU - LANE_IN_AU", src)
    between = (a0 >= ln["inner"]) & (a0 <= ln["outer"])
    putw("LANE_ONLY_LOST_PCT", stats.wilson(np.sum(lost & between), between.sum()),
         "E1: lost share of all planets (any starting shape) starting between LANE_IN_AU and LANE_OUT_AU", src)

    # Trojans (co-orbital)
    lam0 = e1["start_Omega"] + e1["start_omega"] + e1["start_M"]
    pb = analysis.phase_bins(a0, lam0, surv)
    putw("TROJAN_SURV_PCT", pb["trojan"], "E1: survival of planets starting with |a0 - a_G| < 0.25 au and 30-90 deg "
         "ahead of or behind the giant (mean longitude)", src, spread=sp["trojan_surv"])
    putw("COORB_OTHER_SURV_PCT", pb["other"], "E1: same, but within 30 deg of the giant or of the point opposite it",
         src, spread=sp["coorb_other_surv"])
    put("TROJAN_SURV_N", pb["coorb_survivors"], "planets", None, pb["n"],
        "E1: number of co-orbital (|a0 - a_G| < 0.25 au) survivors, all phases", src)

    # Hildas (3:2)
    hil = (a0 >= 3.93) & (a0 < 4.01)
    nb = ((a0 >= 3.80) & (a0 < 3.93)) | ((a0 >= 4.01) & (a0 < 4.14))
    putw("HILDA_SURV_PCT", stats.wilson(np.sum(surv & hil), hil.sum()), "E1: survival for a0 in 3.93-4.01 au (3:2), all e0",
         src, spread=sp["hilda_surv"])
    putw("HILDA_NEIGHBOR_PCT", stats.wilson(np.sum(surv & nb), nb.sum()), "E1: survival for a0 in 3.80-3.93 and "
         "4.01-4.14 au", src)
    nb_in = (a0 >= 3.80) & (a0 < 3.93)
    nb_out = (a0 >= 4.01) & (a0 < 4.14)
    putw("HILDA_INSIDE_PCT", stats.wilson(np.sum(surv & nb_in), nb_in.sum()), "E1: survival for a0 in 3.80-3.93 au "
         "(just inside the 3:2 bin), all e0", src)
    putw("HILDA_OUTSIDE_PCT", stats.wilson(np.sum(surv & nb_out), nb_out.sum()), "E1: survival for a0 in 4.01-4.14 au "
         "(just outside the 3:2 bin), all e0", src)
    bands = {}
    for lo, hi in ((0, 0.05), (0.05, 0.15), (0.15, 0.3)):
        b = (e0 >= lo) & (e0 < hi + (1e-9 if hi == 0.3 else 0))
        bands[f"{lo}-{hi}"] = dict(hilda=stats.wilson(np.sum(surv & hil & b), np.sum(hil & b)),
                                   neighbors=stats.wilson(np.sum(surv & nb & b), np.sum(nb & b)))

    # stretching
    # (fact-check B4) the per-bin MEAN is kept, so STRETCH_31 keeps its definition, but the ARTIFACT_N planets
    # are dropped from it: four of them (emax 85 to 1,922) made the spikes at 3.70, 4.12, 4.20 and 4.24 au.
    # The per-bin median of ALL planets is exported next to it as a cross-check.
    gain = e1["emax"] - e0
    c, m_raw, _ = analysis.stretch_profile(a0, gain)
    c, m, cnt = analysis.stretch_profile(a0, gain, exclude=art)
    gain_med = analysis.stretch_median(a0, gain)
    changed_bins = np.nonzero(np.abs(np.nan_to_num(m_raw) - np.nan_to_num(m)) > 1e-12)[0]
    peaks = analysis.res_peaks(c, m)
    p31 = peaks["3:1"]
    in31 = np.abs(a0 - p31["peak_au"]) <= 0.005 + 1e-9

    def factor_boot(ix):
        cc, mm, _ = analysis.stretch_profile(a0[ix], gain[ix], exclude=art[ix])
        pk = analysis.res_peaks(cc, mm)["3:1"]
        return pk["factor"]
    sub = np.nonzero((a0 > 2.2) & (a0 < 2.8))[0]
    fb = stats.bootstrap(lambda ix: factor_boot(sub[ix]), len(sub), n_boot=100)
    put("STRETCH_31", p31["peak"], "eccentricity", None, int(in31.sum()), "E1: mean of (max e - e0) in the 0.01-au bin "
        f"at the 3:1 peak ({p31['peak_au']:.3f} au)", src)
    put("STRETCH_BG", p31["background"], "eccentricity", None, None, "E1: median of the bins 0.05-0.15 au from the 3:1 peak", src)
    put("STRETCH_31_FACTOR", p31["factor"], "x", fb, None, "STRETCH_31 / STRETCH_BG; interval: bootstrap", src)
    a31 = analysis.res_a(3, 1)
    put("INWARD_31_N", int(np.sum((cls == INWARD) & (np.abs(a0 - a31) <= 0.03))), "planets", None,
        int(np.sum(np.abs(a0 - a31) <= 0.03)), "E1: INWARD exits that started within 0.03 au of 2.502 au", src)
    put("RES_PEAKS_AU", {k: v["peak_au"] for k, v in peaks.items()}, "au", None, None,
        "E1: centre of the 0.01-au bin with the largest mean eccentricity gain within 0.03 au of each resonance", src)
    v6 = dict(peaks=peaks, **{"pass": bool(all(peaks[k]["genuine"] for k in ("3:1", "5:2", "2:1")))},
              rule="each of 3:1, 5:2, 2:1 must have its highest bin within 0.03 au of the Kepler distance; because "
                   "that is true by construction, we also require a genuine peak: highest bin within 0.10 au and "
                   ">= 1.5 x the background")

    # survival by distance
    sva = dict(all=analysis.survival_by_a(a0, surv, A_LO, A_HI, 0.02))
    for lo, hi in ((0, 0.05), (0.05, 0.15), (0.15, 0.3)):
        sva[f"e{lo}-{hi}"] = analysis.survival_by_a(a0, surv, A_LO, A_HI, 0.02, mask=(e0 >= lo) & (e0 <= hi))
    under = int(e1["under"].sum())
    put("UNDER_RESOLVED_N", under, "planets", None, n, "E1: planets whose osculating pericentre time q^1.5 fell below "
        "12 integration steps at any element sample (resolution warning; most of them were lost soon after)", src,
        lost_among=int(np.sum(e1["under"] & lost)))
    inner3 = a0 < 3.0
    region_44_49 = (a0 >= 4.4) & (a0 < 4.9)

    # dense loss curve
    tgrid = np.unique(np.concatenate([np.geomspace(P_G / 15, T, 240), [P_G, 100, 1000, 843 * P_G, T]]))
    st = np.sort(tl)
    curve = dict(t_yr=tgrid.tolist(), lost_pct=(100 * np.searchsorted(st, tgrid + 1e-9, side="right") / n).tolist())

    DET["e1"] = dict(n=n, T_exact_yr=T, class_counts={k: int(np.sum(cls == c)) for c, k in
                     ((SURVIVED, "SURVIVED"), (ENCOUNTER, "ENCOUNTER"), (INWARD, "INWARD"), (OUTWARD, "OUTWARD"))},
                     survivors_stirred=int(np.sum(surv & (gain > 0.1))), survivors_calm=int(np.sum(surv & (gain <= 0.1))),
                     survivors_changed=dict(
                         note="survivors (never entered the Hill sphere, never left the disk) whose LAST element "
                              "sample differs from the start; 'not lost' is therefore not 'on the starting orbit'",
                         n_survivors=int(surv.sum()),
                         a_by_5pct=int(np.sum(surv & (np.abs(e1["afin"] / a0 - 1) > 0.05))),
                         a_by_20pct=int(np.sum(surv & (np.abs(e1["afin"] / a0 - 1) > 0.20))),
                         a_more_than_doubled=int(np.sum(surv & (e1["afin"] > 2 * a0))),
                         a_by_5pct_or_e_by_0p05=int(np.sum(surv & ((np.abs(e1["afin"] / a0 - 1) > 0.05) |
                                                                    (np.abs(e1["efin"] - e0) > 0.05))))),
                     reach_survivors=dict(
                         note="where the in-reach survivors (REACH_SURV_PCT) started",
                         n=int(np.sum(reach & surv)),
                         coorbital_within_0p25au=int(np.sum(reach & surv & (np.abs(a0 - A_G) < 0.25))),
                         by_start_a={f"{lo}-{hi}": int(np.sum(reach & surv & (a0 >= lo) & (a0 < hi)))
                                     for lo, hi in ((3.5, 3.93), (3.93, 4.01), (4.01, 4.36), (4.36, 4.9538),
                                                    (4.9538, 5.4538), (5.4538, 6.18), (6.18, 6.6), (6.6, 7.0), (7.0, 8.5))}),
                     lane_survivors=dict(
                         note="survivors (any starting shape) that started between LANE_IN_AU and LANE_OUT_AU",
                         n=int(np.sum(surv & between)),
                         outside_coorbital_band=stats.wilson(np.sum(surv & between & ~(np.abs(a0 - A_G) < 0.25)),
                                                             np.sum(between & ~(np.abs(a0 - A_G) < 0.25)))),
                     numerical_artifacts=dict(
                         n=int(art.sum()), share_of_all_pct=100.0 * art.sum() / n, rule=art_rule,
                         unbound_sample_n=int(art_a.sum()), jump_n=int(art_b.sum()),
                         all_class_OUTWARD=bool(np.all(cls[art] == OUTWARD)),
                         outward_total=int(np.sum(cls == OUTWARD)), outward_genuine=int(np.sum(cls == OUTWARD) - art.sum()),
                         start_a_range_au=[float(a0[art].min()), float(a0[art].max())],
                         emax_range_unbound=[float(e1["emax"][art_a].min()), float(e1["emax"][art_a].max())],
                         afin_range_unbound_au=[float(e1["afin"][art_a].min()), float(e1["afin"][art_a].max())],
                         last_far_point_max_jump_au=float((e1["afin"] * (1 + e1["efin"]))[art_b].max()),
                         genuine_outward_last_far_point_min_au=float(
                             (e1["afin"] * (1 + e1["efin"]))[(cls == OUTWARD) & ~art].min()),
                         ids_unbound_sample=e1["ids"][art_a].tolist(), ids_jump=e1["ids"][art_b].tolist(),
                         stretch_bins_changed=[dict(a_centre=float(c[i]), mean_with=_f(float(m_raw[i])),
                                                    mean_without=_f(float(m[i])), median_all=_f(float(gain_med[i])))
                                               for i in changed_bins],
                         effect="none on LOST_PCT or any spoken number: they are lost either way. OUTWARD_N and "
                                "ENC_PCT are left as classified; the stretching profile and e_gain exclude them"),
                     lane=ln, lane_1pct=dict(inner=ln1["inner"], outer=ln1["outer"]),
                     phase_bins=pb, hilda_by_e0_band=bands, stretch_peaks=peaks,
                     survival_inside_3au=stats.wilson(np.sum(surv & inner3), inner3.sum()),
                     lost_inside_3au=int(np.sum(lost & inner3)),
                     survival_4p4_4p9=stats.wilson(np.sum(surv & region_44_49), region_44_49.sum()),
                     loss_curve=curve, decision=dec)
    # exports for the renderers
    np.savez_compressed(paths.viz("e1_map_points.npz"), id=e1["ids"].astype(np.int32), a0=a0.astype(np.float32),
                        e0=e0.astype(np.float32), inc0_deg=np.degrees(e1["start_inc"]).astype(np.float16),
                        phase0_deg=np.degrees(np.angle(np.exp(1j * lam0))).astype(np.float16),
                        cls=cls.astype(np.int8), t_event=tev.astype(np.float32), reach=reach,
                        e_gain=np.where(art, np.nan, gain).astype(np.float16), artifact=art,
                        note="per planet: start a (au), e, tilt, phase lambda-lambda_giant (deg); class 0 survived, "
                             "1 Hill sphere, 2 inward, 3 outward; loss time yr (NaN = survived); reach = simple guess; "
                             "artifact = unphysical kick (ARTIFACT_N), e_gain is NaN for those planets")
    na, ne = 850, 300
    ia = np.clip(((a0 - A_LO) / (A_HI - A_LO) * na).astype(int), 0, na - 1)
    ie = np.clip((e0 / 0.3 * ne).astype(int), 0, ne - 1)
    cell = ia * ne + ie
    cnt_c = np.bincount(cell, minlength=na * ne).reshape(na, ne)
    sv_c = np.bincount(cell, weights=surv.astype(float), minlength=na * ne).reshape(na, ne)
    lt = np.where(lost, tev, np.inf)
    med = np.full(na * ne, np.nan)
    order = np.lexsort((lt, cell))
    cs, ltd = cell[order], lt[order]
    starts = np.searchsorted(cs, np.arange(na * ne))
    ends = np.searchsorted(cs, np.arange(na * ne), side="right")
    for k in np.nonzero(ends > starts)[0]:
        v = ltd[starts[k]:ends[k]]
        md = v[(len(v) - 1) // 2]
        med[k] = md if np.isfinite(md) else np.nan
    np.savez_compressed(paths.viz("e1_map_grid.npz"), a_edges=np.linspace(A_LO, A_HI, na + 1),
                        e_edges=np.linspace(0, 0.3, ne + 1), count=cnt_c.astype(np.int16),
                        survivors=sv_c.astype(np.int16), median_loss_time_yr=med.reshape(na, ne).astype(np.float32),
                        note="850 x 300 cells over start a x start e; median loss time over the cell (NaN if the "
                             "median planet survived)")
    with open(paths.viz("e1_curves.json"), "w") as f:
        json.dump(jsonable(dict(loss_curve=curve, survival_by_a_0p02=sva,
                                stretch=dict(a_centre=c, mean_gain=m, median_gain=gain_med, n=cnt, peaks=peaks,
                                             statistic="mean_gain = per-bin mean of (max e - e0) WITHOUT the "
                                                       "ARTIFACT_N planets (unphysical kicks); median_gain = per-bin "
                                                       "median of all planets, as a cross-check",
                                             artifacts_excluded=int(art.sum()),
                                             guides_au={"3:1": analysis.res_a(3, 1), "5:2": analysis.res_a(5, 2)},
                                             not_reproduced_au={"2:1": analysis.res_a(2, 1)}),
                                phase_bins=pb, lane=ln, lane_1pct=dict(inner=ln1["inner"], outer=ln1["outer"]),
                                reach_lane=[q_g - R_HILL, Q_g + R_HILL],
                                resonances_au={k: analysis.res_a(*v) for k, v in analysis.RESONANCES.items()},
                                hilda=dict(bin=[3.93, 4.01], neighbors=[[3.80, 3.93], [4.01, 4.14]]))), f)
    return e1, dict(v6=v6, lost=lost, T=T, dec=dec, peaks=peaks, ln=ln, pb=pb, inner3=inner3, surv=surv,
                    region_44_49=region_44_49, under=under)


# ----------------------------------------------------------------------------- E2
def e2_section(e1, val):
    from .e2 import CASES, N_LAPS
    ids = np.arange(N_DIAL)
    a0, e0 = e1["start_a"][ids], e1["start_e"][ids]
    lam0 = (e1["start_Omega"] + e1["start_omega"] + e1["start_M"])[ids]
    T10 = N_LAPS * P_G
    base_lost = np.isfinite(e1["t_event"][ids]) & (e1["t_event"][ids] <= T10 + 1e-9)
    runs = {"base": dict(lost=base_lost, giant=dict(m=MU_JUPITER, e=E_G), T=T10)}
    for name, c in CASES.items():
        p = paths.raw(f"e2_{name}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        runs[name] = dict(lost=z["cls"] != SURVIVED, giant=c, T=N_LAPS * period(c["m"]),
                          cls=z["cls"], t_event=z["t_event"])
    rows = {}
    rnd = e0 < 0.05
    inner_edge = a0 < A_LO + 0.3
    outer_edge = a0 > A_HI - 0.3
    for name, r in runs.items():
        lost = r["lost"]
        surv = ~lost
        ln = analysis.lane(a0[rnd], surv[rnd], level=0.5, consecutive=LANE_CONSECUTIVE)
        ln_fw = analysis.lane(a0[rnd], surv[rnd], level=0.5, boot=False)
        ln["first_window_rule"] = dict(inner=ln_fw["inner"], outer=ln_fw["outer"],
                                       note="superseded: one resonance window just under 50% could set this edge")
        pb = analysis.phase_bins(a0, lam0, surv)
        rows[name] = dict(giant=r["giant"], T_yr=r["T"], lost=stats.wilson(lost.sum(), N_DIAL), lane=ln,
                          trojan=pb["trojan"], coorb_other=pb["other"],
                          edge_in=stats.wilson(np.sum(surv & inner_edge), inner_edge.sum()),
                          edge_out=stats.wilson(np.sum(surv & outer_edge), outer_edge.sum()),
                          survival_by_a_0p05=analysis.survival_by_a(a0, surv, A_LO, A_HI, 0.05),
                          reach_pct=stats.wilson(analysis.reach_mask(a0, e0, r["giant"]["m"], r["giant"]["e"]).sum(), N_DIAL))
    src = "raw/e2_<case>.npz; baseline from raw/e1_planets.npz (ids 0-99,999, t <= 843 laps)"
    spread = val["V4"]["headline_spreads"]["lost_10kyr"]
    putw("LOST_BASE10K_PCT", rows["base"]["lost"], "E2 baseline: the 100,000 dial planets around the Jupiter copy, lost "
         "by 843 laps (10,002 yr), read from E1", src, spread=spread)
    names = dict(e00="LOST_E00_PCT", e010="LOST_E010_PCT", e023="LOST_E023_PCT", e04="LOST_E04_PCT",
                 m01="LOST_M01_PCT", m03="LOST_M03_PCT", m3="LOST_M3_PCT", m10="LOST_M10_PCT")
    for k, ph in names.items():
        if k in rows:
            g = rows[k]["giant"]
            putw(ph, rows[k]["lost"], f"E2: lost by 843 laps of the giant, giant mass {g['m'] / MU_JUPITER:g} Jupiter, "
                 f"eccentricity {g['e']:g}; the same 100,000 planets", src, spread=spread)
    if "m10" in rows and "m01" in rows:
        l10, l01 = rows["m10"]["lost"], rows["m01"]["lost"]
        ratio = l10["value"] / l01["value"]
        lo = l10["ci95"][0] / l01["ci95"][1]
        hi = l10["ci95"][1] / l01["ci95"][0]
        put("MASS_RATIO", ratio, "x", [lo, hi], N_DIAL, "LOST_M10_PCT / LOST_M01_PCT (interval from the two Wilson "
            "intervals, conservative)", src)
    if "e04" in rows:
        d = stats.paired_diff(runs["base"]["lost"], runs["e04"]["lost"])
        put("E04_MINUS_BASE", d["value"], "points", d["ci95"], N_DIAL, "LOST_E04_PCT - LOST_BASE10K_PCT (paired: same "
            "planets)", src)
        putw("TROJAN_E04_PCT", rows["e04"]["trojan"], "E2 e = 0.4: survival of planets starting 30-90 deg ahead of or "
             "behind the giant with |a0 - a_G| < 0.25 au", src, baseline=rows["base"]["trojan"])
        putw("EDGE_SURV_E04_IN", rows["e04"]["edge_in"], "E2 e = 0.4: survival of planets starting in the innermost "
             "0.3 au of the disc (1.561-1.861 au)", src, baseline=rows["base"]["edge_in"])
        putw("EDGE_SURV_E04_OUT", rows["e04"]["edge_out"], "E2 e = 0.4: survival in the outermost 0.3 au "
             "(10.108-10.408 au)", src, baseline=rows["base"]["edge_out"])
    if "e023" in rows:
        DET["e023_minus_base"] = stats.paired_diff(runs["base"]["lost"], runs["e023"]["lost"])
    DET["e2"] = rows
    DET["e2_lane_edges"] = dict(
        rule=f"round orbits (e0 < 0.05), the same 100,000 test planets, 843 laps (about 10,000 yr): centre of the "
             f"first of {LANE_CONSECUTIVE} adjacent 0.04-au windows (about 76 planets each) that all have survival "
             "below 50%, walking in 0.02-au steps from 2.5 au inside / outside the giant (2.70 and 7.70 au) toward it. "
             "An edge AT 2.70 or 7.70 au means the lane already reaches the start of the walk: a limit, not an edge",
        edges={k: dict(inner=r["lane"]["inner"], outer=r["lane"]["outer"],
                       inner_ci95=r["lane"].get("inner_ci95"), outer_ci95=r["lane"].get("outer_ci95"),
                       inner_is_limit=r["lane"]["inner_at_walk_start"], outer_is_limit=r["lane"]["outer_at_walk_start"],
                       first_window_rule=[r["lane"]["first_window_rule"]["inner"], r["lane"]["first_window_rule"]["outer"]])
               for k, r in rows.items()})
    paired = {k: stats.paired_diff(runs["base"]["lost"], runs[k]["lost"]) for k in runs if k != "base"}
    DET["e2_paired_vs_base"] = paired
    with open(paths.viz("e2_curves.json"), "w") as f:
        json.dump(jsonable(rows), f)
    return rows, runs


# ----------------------------------------------------------------------------- E3
def e3_section():
    path = paths.raw("e3_fates.jsonl")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    n = len(recs)
    src = "raw/e3_fates.jsonl"
    fates = [r["fate"] for r in recs]
    for nm, key in (("FATE_BOUND_PCT", "BOUND"), ("FATE_GIANT_PCT", "HIT_GIANT"), ("FATE_EJECT_PCT", "EJECTED"),
                    ("FATE_STAR_PCT", "HIT_STAR")):
        putw(nm, stats.wilson(fates.count(key), n), f"E3: share of the {n} re-run lost planets with fate {key} by "
             f"{T_FATE:.0f} yr (IAS15, one planet per simulation, physical radii)", src)
    ent = [r for r in recs if r["entered"]]
    putw("E3_ENTERED_PCT", stats.wilson(len(ent), n), "E3: share of the IAS15 re-runs that entered the giant's Hill "
         "sphere at least once (independent check of the WHFast classification)", src)
    ch = [r for r in ent if r["orbit_changed"]]
    putw("ORBIT_CHANGED_PCT", stats.wilson(len(ch), len(ent)), "E3: among re-runs that entered the Hill sphere, share "
         "whose a differs from the start by > 5% or e by > 0.05, 100 yr after the first entry (impact/ejection before "
         "that counts as changed)", src)
    bound = [r for r in recs if r["fate"] == "BOUND"]
    cross = [r for r in bound if r["crossing"]]
    te = sorted(r["t_fate"] for r in recs if r["fate"] == "EJECTED")
    tg = sorted(r["t_fate"] for r in recs if r["fate"] == "HIT_GIANT")
    entry_dt = [r["t_entry"] - None if False else r["t_entry"] for r in ent]
    nc = [r for r in bound if not r["crossing"]]
    lane_in, lane_out = PH["LANE_IN_AU"]["value"], PH["LANE_OUT_AU"]["value"]
    q_nc = np.array([r["a_final"] * (1 - r["e_final"]) for r in nc])
    Q_nc = np.array([r["a_final"] * (1 + r["e_final"]) for r in nc])
    in_lane = (q_nc < lane_out) & ((Q_nc > lane_in) | (Q_nc < 0))
    DET["e3"] = dict(n=n, counts={k: fates.count(k) for k in ("BOUND", "HIT_GIANT", "EJECTED", "HIT_STAR")},
                     bound_crossing=stats.wilson(len(cross), len(bound)),
                     bound_not_crossing=dict(
                         note="still-bound planets whose orbit at year 200,000 does not touch the giant's Hill sphere "
                              "(one snapshot of an osculating orbit, not a prediction): most still pass close by",
                         n=len(nc), pericentre_beyond_giant=int(np.sum(q_nc > A_G)),
                         orbit_still_enters_lane=int(in_lane.sum()), lane_au=[lane_in, lane_out],
                         pericentre_beyond_10au=int(np.sum(q_nc > 10.0)),
                         pericentre_median_au_of_outer=float(np.median(q_nc[q_nc > A_G])) if np.any(q_nc > A_G) else None),
                     ejection_time_pct_10_50_90=np.percentile(te, [10, 50, 90]).tolist() if te else None,
                     giant_hit_time_pct_10_50_90=np.percentile(tg, [10, 50, 90]).tolist() if tg else None,
                     entry_time_median_yr=float(np.median(entry_dt)) if entry_dt else None,
                     bound_final_a_pct_10_50_90=np.percentile([r["a_final"] for r in bound], [10, 50, 90]).tolist() if bound else None,
                     ias15_steps=int(sum(r.get("steps", 0) for r in recs)))
    # export: fates + thinned tracks
    tr, off = [], [0]
    for r in recs:
        p = paths.raw("e3_tracks", f"{r['id']:07d}.npy")
        a = (np.load(p) if os.path.exists(p) else np.zeros((0, 7), np.float32)).reshape(-1, 7)
        tr.append(a)
        off.append(off[-1] + len(a))
    np.savez_compressed(paths.viz("e3_tracks.npz"), ids=np.array([r["id"] for r in recs]),
                        fate=np.array([r["fate"] for r in recs]), t_fate=np.array([r["t_fate"] if r["t_fate"] else np.nan for r in recs]),
                        offsets=np.array(off), track=np.concatenate(tr) if tr else np.zeros((0, 7)),
                        columns=np.array(["t_yr", "x", "y", "z", "a", "e", "varpi"]),
                        note="every 100 yr, heliocentric position (au) and osculating a, e, varpi of each E3 planet; "
                             "rows offsets[k]:offsets[k+1] belong to ids[k]")
    with open(paths.viz("e3_fates.json"), "w") as f:
        json.dump(jsonable([{k: v for k, v in r.items()} for r in recs]), f)
    return recs


# ----------------------------------------------------------------------------- validation placeholders
def validation_section(val, v6):
    t = val
    putw = globals()["putw"]
    put("PERIOD_SIM_YR", t["V1"]["period_sim_yr"], "yr", None, None, "V1: giant's period in the simulation "
        "(fit over 50,000 yr, dt = P/60); Kepler value %.5f yr" % P_G, "raw/validation.json",
        all_steps=[r["period_sim"] for r in t["V1"]["giant"]])
    put("JACOBI_ERR", t["V2"]["median_rel_drift"], "relative", None, t["V2"]["n_outside_2rh"],
        "V2: median over planets of the largest relative change of the Jacobi constant (circular giant, 1,000 laps, "
        "planets that stayed outside 2 Hill radii)", "raw/validation.json", p99=t["V2"]["p99_rel_drift"])
    put("LIBRATION_YR", t["V7"]["median_libration_yr"], "yr", t["V7"]["libration_iqr"], t["V7"]["n_period_measured"],
        "V7: median libration period of 2,000 Trojan test planets (interval = interquartile range, not a CI)",
        "raw/validation.json")
    putw("SCHEME_AGREEMENT_PCT", t["V4"]["agreement"], "V4: share of 20,000 planets with the same lost/not-lost outcome "
         "at production settings and with half step and half check interval (10,000 yr)", "raw/validation.json",
         class_agreement=t["V4"]["class_agreement"])
    put("SCHEME_DISAGREE_PCT", t["V4"]["disagree_pct"], "%", None, t["V4"]["n"], "100 - SCHEME_AGREEMENT_PCT",
        "raw/validation.json")
    n4 = t["V4"]["n"]
    disc4 = n4 - t["V4"]["agreement"]["k"]
    net4 = t["V4"]["lost_half"]["k"] - t["V4"]["lost_prod"]["k"]
    half4 = 100 * stats.Z95 * math.sqrt(max(disc4 - net4 ** 2 / n4, 0.0)) / n4
    put("SCHEME_SPREAD", t["V4"]["spread_points"], "points",
        [t["V4"]["spread_points"] - half4, t["V4"]["spread_points"] + half4], n4, "V4: change of the lost fraction "
        "(half step minus production) at 10,000 yr. The two runs differ in `discordant` single outcomes that nearly "
        "cancel; the interval is the paired 95% interval of the difference, and it, not the value, is the precision "
        "(say 'less than 0.2 points')", "raw/validation.json", discordant=int(disc4), ci95_half_width_points=half4)
    put("ZONE_EXPONENT_OUTER", t["V5"]["outer_exponent"], None, None, 4, "V5: the same fit on the OUTER lane edge "
        "(reported, not part of the test)", "raw/validation.json")
    putw("IAS15_AGREEMENT_PCT", t["V3"]["agreement_lost"], "V3: share of 5,000 planets with the same lost/not-lost "
         "outcome in WHFast (production) and IAS15 over 5,000 yr", "raw/validation.json",
         class_agreement=t["V3"]["agreement_class"], diff_points=t["V3"]["diff_points"])
    put("ZONE_EXPONENT", t["V5"]["exponent"], None, t["V5"]["exponent_ci95"], 4, "V5: fitted exponent of the inner "
        "lane edge distance (a_G - edge)/a_G against giant mass mu (4 masses, circular giant, round test orbits); "
        "published 2/7 = 0.286. Interval: refit with edges drawn from their bootstrap intervals", "raw/validation.json")
    put("ZONE_COEFF", t["V5"]["coefficient"], None, t["V5"]["coefficient_ci95"], 4, "V5: fitted coefficient "
        "(published 1.3 Wisdom 1980; 1.57 Murray & Dermott fit)", "raw/validation.json")
    n_pass = sum(1 for k in t if t[k]["pass"]) + (1 if v6["pass"] else 0)
    put("N_PASSED", n_pass, "tests", None, 9, "validation tests passed of V1-V9 (V6 on the main run)", "raw/validation.json")


# ----------------------------------------------------------------------------- compute cost and meta
def cost_section():
    n_planets, ps = 0, 0
    per = {}
    for mp in glob.glob(os.path.join(paths.RAW, "*", "chunk_*.npz.meta.json")):
        with open(mp) as f:
            m = json.load(f)
        stage = os.path.basename(os.path.dirname(mp))
        per.setdefault(stage, [0, 0])
        per[stage][0] += m["n"]
        per[stage][1] += m["psteps"]
    for k, (nn, s) in per.items():
        n_planets += nn
        ps += s
    ias = 0
    n_ias = 0
    for name in ("v3_ias15", "e3_fates"):
        p = paths.raw(name + ".jsonl")
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        n_ias += 1
                        ias += r.get("steps", 0)
    # replays: v8 run2/split/resume are re-runs of the same planets (counted, they are runs); replays are counted
    rep = 0
    for p in glob.glob(os.path.join(paths.RAW, "replays", "*.npz")):
        with np.load(p) as z:
            rep += len(z["ids"])
    times = stagelog.load()
    wall = sum(v["wall_s"] for v in times.values())
    put("TOTAL_PLANETS", n_planets + n_ias + rep, "test-planet runs", None, None,
        "all test-planet runs of all experiments and validation tests (WHFast chunk planets + IAS15 single-planet runs "
        "+ replays); a planet re-run in another test counts again", "raw/*/chunk_*.meta.json, raw/*.jsonl",
        by_stage={k: v[0] for k, v in per.items()}, ias15_runs=n_ias, replay_planets=rep)
    put("TOTAL_STEPS", ps + ias, "steps", None, None, "WHFast planet-steps (one test planet advanced by one step) of all "
        "WHFast runs + IAS15 steps of all single-planet runs (replays not counted)", "raw/*/chunk_*.meta.json",
        whfast_planet_steps=ps, ias15_steps=ias)
    put("WALL_MIN", wall / 60, "min", None, None, "wall-clock minutes of all stages (sum of raw/stage_times.json), "
        "Mac mini M4, 10 worker processes, machine shared with other jobs", "raw/stage_times.json",
        by_stage={k: round(v["wall_s"] / 60, 2) for k, v in times.items()})
    return times


def run_meta(times, dec):
    import rebound
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=paths.SIM_DIR, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "not a git repository yet"
    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        cpu = platform.processor()
    meta = dict(
        rebound_version=rebound.__version__, numpy=np.__version__, python=platform.python_version(),
        machine=dict(cpu=cpu, cores=os.cpu_count(), os=platform.platform()),
        processes=int(os.environ.get("MILLION_PROCESSES", "10")), git_commit=commit,
        giant=dict(mass_solar=MU_JUPITER, a_au=A_G, e=E_G, period_yr=P_G, hill_au=R_HILL),
        integrator=dict(name="WHFast", coordinates="democratic heliocentric", corrector=0, safe_mode=0,
                        zones=[dict(zone=z[0], a_from=z[1], a_to=z[2], dt_yr=P_G / z[3], dt=f"P_G/{z[3]}") for z in ZONES],
                        event_check=f"every P_G/{CHECKS_PER_LAP} (segment closest-approach test)",
                        element_sample="every P_G/3",
                        note="internal coordinates rebuilt from the synchronised state at every check "
                             "(did_modify_particles), which makes chunking and resume bit-identical (V8)"),
        ias15=dict(settings="REBOUND defaults, exact_finish_time = 0", radii_au=dict(star=0.0046505, giant=4.6733e-4)),
        seeds=dict(E1=SEED_E1, E2="E1 ids 0-99,999", E3=SEED_E3, validation="2_100_000 + test number (V5: +50+k)"),
        e1_decision=dec, stage_wall_s={k: v["wall_s"] for k, v in times.items()},
        wall_time_min=sum(v["wall_s"] for v in times.values()) / 60,
        total_planet_steps=PH["TOTAL_STEPS"]["value"], total_planets=PH["TOTAL_PLANETS"]["value"],
        T_main_used_yr=dec["T_exact_yr"], generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        real_data=(dict(source="https://ssd-api.jpl.nasa.gov/sbdb_query.api", file=DET["real_asteroids"]["file"],
                        retrieved_utc_date=DET["real_asteroids"]["date"], queries=DET["real_asteroids"]["queries"],
                        fields="a,e,i,class") if "real_asteroids" in DET else
                   "not downloaded (python run.py real; https://ssd-api.jpl.nasa.gov/sbdb_query.api)"))
    with open(os.path.join(paths.RESULTS, "run_meta.json"), "w") as f:
        json.dump(jsonable(meta), f, indent=1)


# ----------------------------------------------------------------------------- [RD] checks
def spoken_fraction(w, spread):
    cands = [("about a quarter", "a Quarter", 25.0), ("about a third", "a Third", 100 / 3), ("about half", "Half", 50.0),
             ("about two thirds", "Two Thirds", 200 / 3), ("about three quarters", "Three Quarters", 75.0)]
    v = w["value"]
    best = min(cands, key=lambda c: abs(c[2] - v))
    ok = abs(best[2] - v) <= 2.0 + abs(spread or 0)
    return (best[0], best[1]) if ok else (None, None)


def rd_checks(e1info, e2rows, e3recs, val):
    P = PH
    rd = {}
    rd["S01_p2_knocked_off"] = dict(rule="ORBIT_CHANGED_PCT >= 90", value=P.get("ORBIT_CHANGED_PCT", {}).get("value"))
    rd["S01_p2_knocked_off"]["holds"] = bool(rd["S01_p2_knocked_off"]["value"] is not None and rd["S01_p2_knocked_off"]["value"] >= 90)
    rd["S01_p2_half_loss"] = dict(rule="HALF_LOSS_YR <= 1000", value=P["HALF_LOSS_YR"]["value"],
                                  holds=bool(P["HALF_LOSS_YR"]["value"] <= 1000))
    rd["S01_p2_fifty_thousand"] = dict(rule="no fallback", holds=not e1info["dec"]["fallback_used"])
    rd["S01_p3_trojans"] = dict(rule="TROJAN_SURV_PCT >= 20", value=P["TROJAN_SURV_PCT"]["value"],
                                holds=bool(P["TROJAN_SURV_PCT"]["value"] >= 20))
    rd["S05_p2_still_leaking"] = dict(rule="LATE_LOSS_PCT >= 0.1 points", value=P["LATE_LOSS_PCT"]["value"],
                                      holds=bool(P["LATE_LOSS_PCT"]["value"] >= 0.1))
    diff = P["LOST_PCT"]["value"] - P["REACH_PCT"]["value"]
    rd["S05_p4_same_number"] = dict(rule="|LOST_PCT - REACH_PCT| < 2 points", value=diff, holds=bool(abs(diff) < 2))
    ratio = P["LANE_WIDTH"]["value"] / P["REACH_LANE_WIDTH"]["value"]
    rd["S05_p6_lane_vs_guess"] = dict(rule="LANE_WIDTH >= 1.3 x REACH_LANE_WIDTH", value=ratio, holds=bool(ratio >= 1.3))
    rd["S05_p7_out_of_reach"] = dict(rule="OUT_OF_REACH_LOST_N >= 1% of the disc",
                                     value=100 * P["OUT_OF_REACH_LOST_N"]["value"] / N_MAIN,
                                     holds=bool(P["OUT_OF_REACH_LOST_N"]["value"] >= 0.01 * N_MAIN))
    if e3recs:
        bound = [r for r in e3recs if r["fate"] == "BOUND"]
        cr = 100 * np.mean([r["crossing"] for r in bound]) if bound else None
        rd["S06_p2_crossing"] = dict(rule=">= 80% of BOUND carry the crossing flag", value=cr, holds=bool(cr is not None and cr >= 80))
        rd["S06_p2_star_hits"] = dict(rule="any HIT_STAR", value=sum(r["fate"] == "HIT_STAR" for r in e3recs),
                                      holds=bool(any(r["fate"] == "HIT_STAR" for r in e3recs)))
        rd["S06_p2_ejections"] = dict(rule="any EJECTED", value=sum(r["fate"] == "EJECTED" for r in e3recs),
                                      holds=bool(any(r["fate"] == "EJECTED" for r in e3recs)))
        order = sorted(((P[k]["value"], k) for k in ("FATE_BOUND_PCT", "FATE_GIANT_PCT", "FATE_EJECT_PCT", "FATE_STAR_PCT")), reverse=True)
        rd["S06_p2_order"] = dict(rule="name the fates in order of size", value=[k for _, k in order], holds=True)
    h, nbh = P["HILDA_SURV_PCT"], P["HILDA_NEIGHBOR_PCT"]
    rd["S07_p4_hilda"] = dict(rule="HILDA >= NEIGHBOR + 10 points and intervals do not overlap",
                              value=h["value"] - nbh["value"], holds=bool(h["value"] - nbh["value"] >= 10 and h["ci95"][0] > nbh["ci95"][1]))
    rd["S07_p5_inward31"] = dict(rule="INWARD_31_N >= 20", value=P["INWARD_31_N"]["value"], holds=bool(P["INWARD_31_N"]["value"] >= 20))
    rd["S07_p5_factor"] = dict(rule="STRETCH_31_FACTOR >= 1.5", value=P["STRETCH_31_FACTOR"]["value"],
                               holds=bool(P["STRETCH_31_FACTOR"]["value"] >= 1.5))
    rd["S08_p3_resonances"] = dict(rule="name only resonances with a genuine peak (V6)",
                                   value={k: v["genuine"] for k, v in e1info["peaks"].items()}, holds=True)
    s4449 = DET["e1"]["survival_4p4_4p9"]["value"]
    rd["S08_p4_like_ours"] = dict(rule="E1 survival at 4.4-4.9 au < 5%", value=s4449, holds=bool(s4449 < 5))
    if "MASS_RATIO" in P:
        rd["S09_p2_mass_ratio"] = dict(rule="MASS_RATIO < 6", value=P["MASS_RATIO"]["value"], holds=bool(P["MASS_RATIO"]["value"] < 6))
    if "e04" in e2rows:
        tb = e2rows["base"]["trojan"]["value"]
        rd["S09_p4_trojans_gone"] = dict(rule="TROJAN_E04_PCT < baseline/3", value=[P["TROJAN_E04_PCT"]["value"], tb],
                                         holds=bool(P["TROJAN_E04_PCT"]["value"] < tb / 3))
        ei, eo = e2rows["e04"]["edge_in"]["value"], e2rows["e04"]["edge_out"]["value"]
        bi, bo = e2rows["base"]["edge_in"]["value"], e2rows["base"]["edge_out"]["value"]
        rd["S09_p4_both_edges"] = dict(rule="edges < 98% at e=0.4 and >= 99.5% at baseline", value=dict(e04_in=ei, e04_out=eo, base_in=bi, base_out=bo),
                                       holds=bool(ei < 98 and eo < 98 and bi >= 99.5 and bo >= 99.5))
        if "m10" in e2rows:
            d = P["LOST_E04_PCT"]["value"] - P["LOST_M10_PCT"]["value"]
            branch = "within 10 points: as written" if abs(d) <= 10 else ("E04 higher" if d > 0 else "mass did more (E04 lower by > 10)")
            rd["S09_p5_shape_vs_mass"] = dict(rule="LOST_E04 vs LOST_M10", value=d, holds=True, branch=branch)
        flat = P["LOST_E04_PCT"]["value"] - P["LOST_BASE10K_PCT"]["value"]
        rd["DIAL_FLAT"] = dict(rule="shape dial flat if LOST_E04 - LOST_BASE10K < 5 points", value=flat, holds=bool(flat < 5))
    if "e023" in e2rows:
        d = DET["e023_minus_base"]
        sp = abs(P["SCHEME_SPREAD"]["value"])
        rd["S10_p2_rounder_kept_more"] = dict(rule="LOST_E023 - BASE > scheme spread and paired 95% interval excludes 0",
                                              value=d, holds=bool(d["value"] > sp and d["ci95"][0] > 0))
        b, c = e2rows["base"], e2rows["e023"]
        def beyond(lo_new, hi_old):
            return lo_new is not None and hi_old is not None and lo_new > hi_old
        ins = b["lane"]["inner_ci95"][0] is not None and c["lane"]["inner_ci95"][1] is not None and c["lane"]["inner_ci95"][1] < b["lane"]["inner_ci95"][0]
        outs = beyond(c["lane"]["outer_ci95"][0], b["lane"]["outer_ci95"][1])
        trs = c["trojan"]["ci95"][1] < b["trojan"]["ci95"][0]
        rd["S13_p1_shrinks"] = dict(rule="from e 0.05 to 0.23: inner edge inward, outer edge outward, Trojan survival "
                                    "down, each beyond its interval", value=dict(inner_moved_in=ins, outer_moved_out=bool(outs), trojan_fell=bool(trs),
                                    base_lane=[b["lane"]["inner"], b["lane"]["outer"]], e023_lane=[c["lane"]["inner"], c["lane"]["outer"]]),
                                    holds=bool(ins and outs and trs))
    rd["S12_p3_nothing_lost_inside_3au"] = dict(rule="every planet inside 3.0 au survived",
                                                value=DET["e1"]["lost_inside_3au"], holds=DET["e1"]["lost_inside_3au"] == 0)
    rd["S12_p3_lane_almost_nothing"] = dict(rule="LANE_ONLY_LOST_PCT > 90", value=P["LANE_ONLY_LOST_PCT"]["value"],
                                            holds=bool(P["LANE_ONLY_LOST_PCT"]["value"] > 90))
    rd["S11_p3_validation"] = dict(rule="V5 exponent in 0.24-0.33", value=P["ZONE_EXPONENT"]["value"], holds=bool(val["V5"]["pass"]))
    return rd


# ----------------------------------------------------------------------------- main
def run():
    PH.clear()
    DET.clear()
    with open(paths.raw("validation.json")) as f:
        valf = json.load(f)
    val = valf["tests"]
    e1, info = e1_section(val)
    e2rows, runs = e2_section(e1, val)
    e3recs = e3_section()
    validation_section(val, info["v6"])
    DET["validation_summary"] = {k: v["pass"] for k, v in val.items()}
    DET["validation_summary"]["V6"] = info["v6"]["pass"]
    DET["v6"] = info["v6"]
    # spoken helper fields (formatting only; final spoken forms are written at script lock)
    s, t = spoken_fraction(PH["LOST_PCT"], PH["LOST_PCT"]["scheme_spread_points"])
    put("LOST_FRACTION_SPOKEN", s, None, None, None, "nearest of quarter/third/half/two thirds/three quarters to "
        "LOST_PCT, only if within 2 points + |scheme spread|; null = say the number", "derived from LOST_PCT")
    put("LOST_FRACTION", t, None, None, None, "title form of LOST_FRACTION_SPOKEN", "derived from LOST_PCT")
    if "MASS_RATIO" in PH:
        put("MASS_RATIO_SPOKEN", None, None, None, None, f"write at script lock from MASS_RATIO = "
            f"{PH['MASS_RATIO']['value']:.2f} (interval {PH['MASS_RATIO']['ci95'][0]:.2f}-{PH['MASS_RATIO']['ci95'][1]:.2f})",
            "derived")
    zx = PH["ZONE_EXPONENT"]["value"]
    put("ZONE_EXPONENT_DIGITS", f"{zx:.2f}"[2:], None, None, None, "two digits after 'zero point' of ZONE_EXPONENT", "derived")
    if e3recs:
        ej = [r for r in e3recs if r["fate"] == "EJECTED"]
        idx_path = paths.viz("replays", "index.json")
        rid = None
        facts = None
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                idx = json.load(f)
            c = idx["clips"].get("ejection.npz", {})
            rid, facts = c.get("id"), c.get("track_facts")
        put("EJECT_RUN", rid, None, None, len(ej), "E3/E4: id of the replayed ejection (the EJECTED planet with the "
            "median ejection time)", "viz/replays/index.json")
        put("EJECT_REPLAY_LINE", None, None, None, None, "[RD-REPLAY] to be written by a person after reading the "
            "track; measured facts of the track are in `facts`", "viz/replays/index.json", facts=facts)
    from . import realdata
    real = realdata.derived()
    if real is None:
        for k in ("REAL_N_TOTAL", "REAL_N_BOX", "REAL_LANE_N", "REAL_TROJAN_N", "REAL_HILDA_RANGE_N", "REAL_DATA_DATE"):
            put(k, None, None, None, None, "PENDING: JPL Small-Body Database not downloaded yet "
                "(python run.py real; https://ssd-api.jpl.nasa.gov/sbdb_query.api)", "not computed", status="pending")
    else:
        src = "raw/" + real["file"] + " (NASA/JPL SBDB, measured)"
        put("REAL_N_TOTAL", real["total"], "asteroids", None, None, "all asteroids (sb-kind=a) in the JPL SBDB on REAL_DATA_DATE", src)
        put("REAL_N_BOX", real["box"], "asteroids", None, None, "asteroids with 1.56 <= a <= 10.41 au (our disc's range)", src)
        put("REAL_LANE_N", real["lane_4p2_5p05"], "asteroids", None, None, "asteroids with 4.2 <= a < 5.05 au", src)
        put("REAL_TROJAN_N", real["trojans_tjn"], "asteroids", None, None, "asteroids of orbit class TJN (Jupiter Trojan)", src)
        put("REAL_HILDA_RANGE_N", real["hilda_range_3p7_4p2"], "asteroids", None, None,
            "asteroids with 3.7 <= a < 4.2 au (a distance range, not a resonance test)", src)
        put("REAL_DATA_DATE", real["date"], None, None, None, "UTC retrieval date of the cached download", src)
        DET["real_asteroids"] = real
    for k in ("LAPS_SPOKEN", "SURV_COUNT_SPOKEN", "TOTAL_PLANETS_SPOKEN", "TOTAL_STEPS_SPOKEN", "WALL_SPOKEN"):
        put(k, None, None, None, None, "spoken form, written at script lock from the numeric placeholder", "derived")
    times = cost_section()
    rd = rd_checks(info, e2rows, e3recs, val)
    out = dict(
        _about="Every number in the video comes from this file: value + 95% interval (Wilson for fractions unless "
               "noted) + n + scheme spread (V4: production vs half step, in points) + the raw file. Generated by "
               "sim/run.py results. Percentages are in percent.",
        sampling_box=dict(E1=dict(n=N_MAIN, a_au=[A_LO, A_HI, "uniform"], e=[0, 0.3, "uniform"], inc_deg=[0, 5, "uniform"],
                                  angles="node, argument of pericentre, mean anomaly uniform 0-360 deg",
                                  seed="numpy.random.default_rng(2_000_001), arrays drawn in the order a, e, inc, node, peri, M"),
                          E2="E1 ids 0-99,999, 843 laps of each giant", E3="rng(2_000_003).choice(E1 ENCOUNTER ids, 1000)"),
        placeholders=PH, rd_checks=rd, details=DET)
    with open(os.path.join(paths.RESULTS, "RESULTS.json"), "w") as f:
        json.dump(jsonable(out), f, indent=1)
    run_meta(times, info["dec"])
    with open(paths.viz("validation.json"), "w") as f:
        json.dump(jsonable(dict(v5=val["V5"], v1=val["V1"], v7={k: v for k, v in val["V7"].items()},
                                n_passed=PH["N_PASSED"]["value"])), f, indent=1)
    print(f"[results] {len(PH)} placeholders written to {os.path.join(paths.RESULTS, 'RESULTS.json')}", flush=True)
