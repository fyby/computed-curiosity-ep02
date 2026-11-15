# EP02 "One star, one giant, 1,000,000 test planets": what the simulations found

Run on 2026-11-14 on the Mac mini (Apple M4, 10 worker processes, machine shared with a video render pipeline; every heavy job ran under the machine-wide CPU slot). REBOUND 5.2.2, WHFast (democratic heliocentric) for the disc runs, IAS15 for the fates. G = 4π² exactly.
Every number below is copied from `RESULTS.json` (rounded here for reading; use the JSON for the script). Brackets are 95% intervals (Wilson unless noted). "Spread" = change with half the step and half the check interval (V4).

**Compute:** validation 7.4 min, E1 71.7 min (the code projected 73.9 min at 2,005 yr, under the 85-min limit, so the **full 50,000 yr** was run, no fallback), E2 11.5 min, E3 5.7 min, replays 0.3 min. `WALL_MIN` = 96.7 min (sum of stages; results/previews and slot waiting not included). `TOTAL_PLANETS` = 2,105,001 test-planet runs, `TOTAL_STEPS` = 2.97 × 10¹¹ (WHFast planet-steps + IAS15 steps).

## 1. Validation: 8 of 9 (`N_PASSED` = 8). The production gate (V1–V5, V7–V9) passed; V6 passes for 3:1 and 5:2, not for 2:1

| Test | Result | Pass condition |
|---|---|---|
| V1 control + giant alone | 100,000 planets with a massless giant: largest change of a 8e-14 (relative), of e 1.4e-10. Giant's period **11.86516 yr** vs Kepler 11.86516 (rel. error 5e-9 to 8e-8); phase after 50,000 yr off by 0.008° (P/240) to 0.14° (P/60) | 1e-9; 1e-6; 0.5° |
| V2 Jacobi constant | median drift 8.1e-7, 99th pct 4.4e-6 (13,264 planets outside 2 Hill radii, 1,000 laps) | 1e-6 / 1e-4 |
| V3 WHFast vs IAS15 | lost 29.7% vs 29.6% (5,000 planets, 5,007 yr); difference −0.1 points [−0.49, +0.29]; same lost/not-lost outcome for **98.0%** of planets (`IAS15_AGREEMENT_PCT`) | 0 inside the interval |
| V4 half step & check | lost 31.690% vs 31.685% (**spread −0.005 points**); same outcome for **98.4%** (`SCHEME_AGREEMENT_PCT`); lane edges moved 0.04 au (inner, one grid step; borderline) and 0.00 au (outer) | 0.5 points; 0.04 au |
| V5 **published law** | inner lane edge vs giant mass: **exponent 0.279 [0.272–0.285]**, coefficient **1.25 [1.20–1.31]** (law: 2/7 = 0.286; 1.3 Wisdom, 1.57 Murray & Dermott). The outer edge gives exponent 0.34 (reported, not tested) | 0.24–0.33; 1.0–1.8 |
| V6 resonance peaks | 4:1 at 2.065 (Kepler 2.065), **3:1 at 2.505 (2.502), 5:2 at 2.825 (2.825)**: genuine peaks (×3.2, ×5.5, ×5.3 background). 7:3: bin at 2.955 only ×1.4 (weak). **2:1: no peak at 3.278**; the stretching there is a broad bump peaking at 3.335 au, 0.06 au outside the Kepler distance | within 0.03 au + genuine peak |
| V7 Trojans | 100% of 2,000 survive 5,007 yr; median libration period **146.9 yr** (`LIBRATION_YR`; Wikipedia ~150) | ≥95%; 135–165 yr |
| V8 determinism | one 10,000-planet chunk run twice, split in two, and interrupted/resumed: identical class and event time for all (bit-identical orbits) | identical |
| V9 event test | 20,000 planets, 2,005 yr, against references that test after **every** step on bit-identical trajectories: vs per-step segment test 99.7% within one check; production later by > 1 check for 0 (points-only ref) / 9 of 5,664 (0.16%, segment ref). Spec's literal number (points-only reference): **99.5%** | see README |

Two problems found and fixed before production: (1) REBOUND's `units=("yr","AU","Msun")` uses the Gaussian G (3.8e-5 smaller than 4π²), which failed V1; G is now set to 4π² directly. (2) A straight-chord event test in the inertial frame flagged ~3% of grazing passes at the wrong time (V9 at 97%); the test now reconstructs the path in the frame turning with the giant, with a Hermite curve near the giant.

Deviations from the written spec: V9's reference tests every step (P/240…P/60) rather than P/120, because P/120 is not a whole zone-D step; V6 got an extra "genuine peak" condition because its literal condition is true by construction.

## 2. The main run (E1): 1,000,000 test planets, copy of Jupiter, 50,000 yr (4,214 laps)

- **Lost: 35.37% [35.28–35.46]** (`LOST_PCT`); half-step re-run: −0.005 points, paired 95% interval ±0.18 points (`SCHEME_SPREAD`: 321 of 20,000 single outcomes differ and nearly cancel, so say "less than 0.2 points"). **Not lost: 646,295** [645,357–647,232] (`SURV_COUNT`). "Not lost" = never entered the Hill sphere, never left the disk. It is not "still on the starting orbit": 45,090 of them changed orbit size by more than 5% (25,968 by more than 20%, 7,814 more than doubled it), and 188,580 changed a by more than 5% or e by more than 0.05 (`details.e1.survivors_changed`).
- `LOST_FRACTION_SPOKEN` is **null**: 35.4% is 2.0 points from a third, just outside the rule. "More than a third" is true; "about a third" is not supported by the rule. Title 3 in PACKAGE.md ("…Took a Third") does not hold.
- How: Hill-sphere entry 94.9% of losses (335,626), outward 16,735, inward 1,344.
- **Known numerical artifacts: 72 planets (0.007%)** (`ARTIFACT_N`; `details.e1.numerical_artifacts` lists the ids and the rule). 72 planets received an unphysical kick in an unresolved close pass by the giant; they are counted as lost, which they would be anyway. All 72 are in the OUTWARD class, so `OUTWARD_N` = 16,735 is 16,663 genuine outward exits (`OUTWARD_GENUINE_N`) plus these 72. How they were found, from the saved table only: (a) 31 have an element sample with a ≤ 0 (a between −0.0006 and −0.39 au, e from 8 to 8,120: speeds of tens to more than a thousand km/s); (b) 41 more were beyond 52 au at the event test although their last element sample, at most 4 yr earlier, was an ordinary orbit with its far point under 13 au (rule: under 20 au; every genuine outward exit has it beyond 52 au). All started at 3.70–8.26 au, in the zone with the coarsest step (P/60 = 0.2 yr). No spoken number changes: `LOST_PCT`, `ENC_PCT` (94.9%) and the lane are the same with or without them. No validation test detects them.
- Loss curve: 1 lap **1.79%**, 100 yr **11.3%**, 1,000 yr **26.5%**, 10,002 yr **31.5%**, 50,000 yr 35.4%. **Half of all losses by 235 yr** [233–237] (`HALF_LOSS_YR`). Last fifth of the run: **+0.73%** of all planets (`LATE_LOSS_PCT`): still leaking.
- **Simple guess:** 32.7% in reach (`REACH_PCT`). It fails both ways: **11.6%** of the planets in reach survived (`REACH_SURV_PCT`), and **64,405** lost planets started out of reach (`OUT_OF_REACH_LOST_N`; 18.2% of all losses, 6.4% of the disc).
- **The lane (round orbits, e0 < 0.05):** **4.06 to 6.60 au, 2.54 au wide** (`LANE_WIDTH`, bootstrap [2.52–2.56]) against the guess's 1.22 au: 2.1×. Edge rule since the fact-check: the first of three adjacent 0.04-au windows that are all below 50% survival; for E1 the older first-window rule gives the same two edges. These are 50% edges: inside them round-orbit survival is still 20–47% at 4.06–4.30 au and 15–32% at 6.2–6.5 au. Below 1% survival from 4.36 to 6.18 au. 91.1% of all planets that start between the edges were lost (`LANE_ONLY_LOST_PCT`). Note: lane edges are found on a 0.02-au grid, so their bootstrap intervals are one grid step or zero wide; treat ±0.02–0.04 au as the real precision.
- **Trojans:** **56.1% [55.4–56.8]** survive near 60° ahead/behind (`TROJAN_SURV_PCT`, n = 18,784) vs **4.4% [4.1–4.7]** beside or opposite the giant (`COORB_OTHER_SURV_PCT`). Spread −0.8 / +1.5 points. 14,653 co-orbital survivors (`TROJAN_SURV_N`). By 30° bin: 45–68% at 30–90°, 32–34% at 90–120°, 0.2–9% elsewhere. Of all 56,346 planets that started within 0.25 au of the giant's distance, 26% survived. The co-orbital survivors are 14,653 of the 37,971 in-reach survivors (39%); the rest started at 6.6–7.0 au (9,127), 7.0–8.5 au (3,390), 4.01–4.36 au (3,186), 6.18–6.60 au (3,037), in the 3:2 bin (2,403) and at 3.5–3.93 au (2,096) (`details.e1.reach_survivors`).
- **3:2 (Hildas):** **73.0% [72.1–73.9]** at 3.93–4.01 au vs **43.6% [43.0–44.2]** next to it. The 43.6% pools two unlike neighbors: **60.1% [59.3–60.9]** just inside (3.80–3.93 au, `HILDA_INSIDE_PCT`) and **27.0% [26.3–27.7]** just outside (4.01–4.14 au, `HILDA_OUTSIDE_PCT`); quote the three numbers, not the pooled one. A strong island, in every e0 band (round orbits 98% vs 76%; e0 0.15–0.3: 58% vs 22%).
- **Stretching:** at 3:1 the mean eccentricity gain is 0.198 vs 0.036 background: **5.5× [5.2–5.7]** (`STRETCH_31_FACTOR`). The profile is the per-bin **mean without the 72 artifact planets** (the per-bin median of all planets is exported next to it in `viz/e1_curves.json`). Before that correction four single planets made false spikes at 3.70, 4.12, 4.20 and 4.24 au (bin means 1.90, 0.24, 0.28, 0.17; now 0.16, 0.14, 0.09, 0.09). No bin inside 3.6 au changed, so the 3:1 and 5:2 numbers are identical. The real features beyond 3.5 au are small: 0.17 at 3.575 au (the 7:4 distance), 0.17 at 3.70–3.72 au (5:3), 0.17 near 4.05 au. **352** planets that started within 0.03 au of 2.502 au left through the inner edge (`INWARD_31_N`). Peaks (`RES_PEAKS_AU`): 4:1 2.065, 3:1 2.505, 5:2 2.825, 7:3 2.955 (weak), 2:1 search window max at 3.305 (not a peak, see V6).
- Inside 3.0 au: 99.25% survive, but **1,212 planets were lost there** (inward exits from the resonances).
- `UNDER_RESOLVED_N` = 5,249 planets (0.5%) had their pericentre time fall below 12 steps at some sample; 3,312 of them were later lost. Their exact outcomes are less certain; even if every one of them were wrong, the headline could move by at most 0.5 points.

## 3. The dials (E2): the same 100,000 planets, 843 laps (~10,000 yr)

| Giant | Lost | Lane (round orbits, 50% edges, 3-window rule) | Trojan survival | Edge survival in / out |
|---|---|---|---|---|
| **Jupiter copy (baseline, from E1 at 10,002 yr)** | **31.6%** | 4.30–6.50 au | 62.2% | 100 / 100 |
| e = 0 | 30.1% | 4.30–6.42 | 66.8% | 100 / 100 |
| e = 0.10 | 34.6% | 4.04–6.62 | 54.8% | 100 / 100 |
| **e = 0.23** | **47.4%** | **3.46–7.20** | 29.4% | 98.7 / 99.8 |
| **e = 0.40** | **69.5%** | ≤ 2.70 – ≥ 7.70 (already below 50% where the walk starts: limits, not edges) | **4.1%** | **76.4 / 91.7** |
| 0.1 Jupiter | 19.7% | 4.66–5.78 | 49.8% | 100 / 100 |
| 0.3 Jupiter | 23.9% | 4.50–6.00 | 59.0% | 100 / 100 |
| 3 Jupiters | 46.0% | 3.96–6.82 | 48.5% | 100 / 99.6 |
| **10 Jupiters** | **69.4%** | 3.40– ≥ 7.70 | 22.6% | 100 / **71.6** |

**Lane edges, recomputed after the fact-check (B3)** from the saved tables; nothing was re-run. Rule: round orbits (e0 < 0.05) of the same 100,000 planets at 843 laps; walking in 0.02-au steps from 2.70 au and from 7.70 au toward the giant, the edge is the centre of the first of **three adjacent 0.04-au windows (about 76 planets each) that are all below 50% survival** (`details.e2_lane_edges`, with bootstrap intervals; the old value is kept there as `first_window_rule`). The first version took the first single window below 50%, and one resonance dip could set the edge:
- e = 0.23: the old inner edge 2.82 au was the 5:2 window alone (43 of 87 survive, 49.4%); survival is 72–100% from 2.86 to 3.30 au. New: **3.46 au** [3.46–3.48]. Outer edge unchanged at 7.20 au [7.18–7.24].
- 3 Jupiters: 3.72–6.88 became 3.96–6.82.
- Baseline at 10,000 yr: 4.16 became 4.30 [4.16–4.30]. Between 4.14 and 4.30 au survival sits on a plateau of 42–60%, so this edge is soft; quote it to one decimal ("4.3").
- All other edges are the same under both rules. E1 at 50,000 yr: 4.06–6.60 au under both.

Simple guess for each dial run (`details.e2.<case>.reach_pct`): baseline 32.8% in reach (31.6% lost), e = 0.23 54.8% (47.4% lost), e = 0.4 75.2% (69.5% lost), 10 Jupiters 42.4% (69.4% lost). The shape dial is the case where the simple guess roughly works; the mass dial is the case where it fails.

Intervals are ±0.3 points on the lost fractions (Wilson, n = 100,000); scheme spread −0.005 points.

**The dial is NOT flat — it is the episode's surprise.** Stretching Jupiter's orbit to e = 0.4 raises the losses by **+37.9 points** [37.6–38.2] (`E04_MINUS_BASE`, paired), to **69.5%**, the same as making the giant **ten times heavier (69.4%)**: the two differ by 0.08 points, well inside S09_p5's "within 10 points" branch. At the measured cold-giant median e = 0.23 the losses rise from 31.6% to 47.4% (+15.8 [15.6–16.1] points, paired). Mass: 100× the mass (0.1 → 10 Jupiters) gives **3.5× [3.47–3.59]** the losses (`MASS_RATIO`).

## 4. Fates of the lost (E3): 1,000 ENCOUNTER planets, IAS15, 200,000 yr

- **Still bound 69.8% [66.9–72.6], ejected 22.3% [19.8–25.0], hit the giant 7.9% [6.4–9.7], hit the star 0% [0–0.4].**
- 98.1% of the IAS15 re-runs entered the Hill sphere too (`E3_ENTERED_PCT`); **93.0%** of those had a or e clearly changed 100 yr after the first entry (`ORBIT_CHANGED_PCT`).
- **57.7%** of the still-bound ones have a final orbit that still crosses the giant's Hill sphere (`crossing`); the median still-bound planet ends at a = 52 au (10th–90th pct 7–1,576 au): most are flung onto huge, very elongated orbits. The other 295 (42.3%) are **not** out of the giant's hands: 277 have their pericentre outside the giant's orbit (median 6.2 au), and **224 of the 295 still come inside the lane** (4.06–6.60 au); only 18 have a pericentre beyond 10 au (`details.e3.bound_not_crossing`). The test is one snapshot of an osculating orbit at year 200,000, not a prediction.
- Ejections take long: median 124,500 yr (10th–90th: 58,000–182,000 yr). Giant impacts: median 19,500 yr.

## 5. Replays (E4), selected by the rules in `viz/replays/index.json`

- `res31`: planet 473570 (a0 = 2.478, e0 = 0.255), the median-time inward exit among 352: e rises to 0.58, leaves inward at 21,194 yr. Re-run reproduced E1 exactly.
- `ejection` (`EJECT_RUN` = **54519**), measured facts for `{{EJECT_REPLAY_LINE}}` (to be written by a person after looking): first Hill-sphere entry at 392 yr; 27 separate entries in the sampled track; closest pass 0.046 au (6.9 million km) at 485 yr; last exit from the Hill sphere at 66,325 yr; then 58,000 yr before the ejection test fired at 124,500 yr, about 3.4 km/s at the end. It reproduced the E3 run exactly.
- `giant_hit`: planet 283633, first entry 123 yr, 20 entries, impact at 19,533 yr. `wanderer`: planet 44938, still bound at 200,000 yr on a = 38 au, e = 0.85, 171 sampled Hill entries.
- `first_laps`: 5,000 planets at 4.2–6.4 au, 2,077 lost within 130 yr. `trojans`: 2,000 co-orbital survivors, none lost in 605 yr. `control`: 0 changed.

## 6. How each [RD] line resolves

| Line | Resolution |
|---|---|
| S01_p2 "knocked off their orbits" | **keep** (ORBIT_CHANGED 93% ≥ 90). Fraction: "more than a third" (rule rejects "about a third"). "first {{HALF_LOSS_YR}}" = about two hundred thirty five: **keep**. "Fifty thousand": **keep** |
| S01_p3 | **keep** (Trojans 56% ≥ 20) |
| S01_p4 "Not its mass. The shape" | **keep**: dial not flat |
| S05_p2 still leaking | **keep** (+0.73%) |
| S05_p4 | **change** to "The giant took more than its bubble could ever touch." (35.4 vs 32.7: 2.6 points apart) |
| S05_p6 | **keep** (2.54 vs 1.22 au: 2.1×) |
| S05_p7 | **keep** (64,405 ≈ "about sixty four thousand", 6.4% of the disc) |
| S06_p2 | order: bound 70, **ejected 22, then giant 8**. Only 58% of the bound ones cross → drop "that cross the giant's path", say "on stretched orbits". No star hits → no "fell into the star". Ejections exist → keep S06_p3 |
| S06_p4 | **keep** second sentence (93% ≥ 90) |
| S07_p2/p3 | **keep** (56% vs 4%) |
| S07_p4 Hildas | **keep** (73 vs 44, +29 points, intervals apart) |
| S07_p5 | **keep** (5.5×, 352 exits ≥ 20) |
| S08_p3 | **name 3:1 and 5:2 only** (2:1 fails V6: the stretching there is a broad bump 0.06 au outside). Also: the 2:1 sentence on screen should not show a teal spike at 3.28 |
| S08_p4 "like ours" | **keep** (E1 survival at 4.4–4.9 au: 0.04%) — the real counts are pending the JPL download |
| S09_p2 | **keep** "a hundred times the mass, for {{MASS_RATIO_SPOKEN}} times the losses" (3.5 < 6): "three and a half" |
| S09_p4 | **keep both** (Trojans 4.1% vs 62% baseline; edges 76/92% vs 100/100%) |
| S09_p5 | **branch 1, as written** (0.08 points apart) |
| DIAL flat? | **No** (+37.9 points). S09/S10 stay full length |
| S10_p2 last sentence | **keep** (+15.8 points, interval excludes 0, far above spread) |
| S11_p3 | ZONE_EXPONENT 0.279 → "zero point two eight" (law rounds to 0.29, so not "the same") |
| S12_p1 | **keep** "still creeping up" |
| S12_p3 | "**almost** nothing is lost" (1,212 lost inside 3.0 au); "almost nothing survives" **keep** (91.1% > 90) |
| S13_p1 | **keep all** (Trojans held; from e 0.05 to 0.23 the inner edge moved in 4.30 → 3.46, outer out 6.50 → 7.20, Trojan survival fell 62 → 29%; edges by the 3-window rule, see section 3) |

## 7. Surprises

1. **Shape = ten times the mass.** e = 0.4 costs 69.5%, ten Jupiters 69.4%. The cold-giant median (0.23) already adds 16 points.
2. **"Lost" mostly means "flung far", not "ejected" or "destroyed":** 70% still bound after 200,000 yr, but the median survivor of that group sits at 52 au. 42% of them end on orbits that no longer cross the giant's Hill sphere, though most of those still pass close by (section 4).
3. **The guess got the number roughly right for the wrong reasons** (32.7% vs 35.4%), failing in both directions by tens of thousands of planets.
4. **The 3:2 island is strong** (73% vs 44%), and 2:1 is not a sharp feature at all in our run.
5. Ejections are slow: the replayed one spent 58,000 years on a long excursion after its last close pass.

## 8. Caveats

- One giant, massless planets, 50,000 yr (the real system is ~92,000× older); losses are still growing (+0.73% in the last fifth).
- The headline depends on our box (uniform 1.56–10.4 au, e ≤ 0.3); the map is the finding.
- Chaos: ~2% of individual planets change outcome with a different step (V3, V4); totals do not.
- E2 at e = 0.4 and at 10 Jupiters: losses reach the edge of the box (inner/outer lane edges beyond the 2.5-au walk), so those lane edges and loss fractions are lower limits on the damage.
- The Hill radius is held at a(m/3)^(1/3) for every eccentricity.
- E3 fates: 1,000 planets, 200,000 yr; the bound ones have no final fate. Ejection is tested only every 50 yr beyond 200 au.
- **Numerical artifacts:** 72 planets (0.007%) received an unphysical kick in an unresolved close pass by the giant; they are counted as lost, which they would be anyway (section 2). The stretching profile leaves them out.
- **Lane edges** are 50% survival edges on a 0.02-au grid from windows of about 76 planets (dials) or about 750 (E1). They need three adjacent windows below 50% (section 3). Where survival sits near 50% over a stretch (baseline inner edge at 10,000 yr) the edge is soft.
- **Real asteroids (S08, H9):** downloaded once on 2026-11-14 with `python run.py real` from https://ssd-api.jpl.nasa.gov/sbdb_query.api (a, e, i, class). The `REAL_*` placeholders are filled from the frozen file; `preview/13_real_belt.png` draws it with guides at 3:1 and 5:2 only and 2:1 marked "not reproduced".

## 9. Corrections after the independent fact-check (2026-11-14)

Recomputed from the saved outputs with `python run.py results previews`. No simulation was run again. Every placeholder value in `RESULTS.json` is identical to the version the script was locked on; the only changed intervals are `LANE_IN_AU` (upper end 4.06 → 4.08) and `SCHEME_SPREAD` (now has one).

| Item | What changed |
|---|---|
| B4 stretching profile | per-bin mean without the 72 artifact planets (median exported as cross-check); `e_gain` in `viz/e1_map_points.npz` is NaN for them and the file has an `artifact` flag; `viz/e1_curves.json` re-exported. The spikes at 3.70, 4.12, 4.20 and 4.24 au are gone |
| B4 / F12 artifacts | new keys `ARTIFACT_N` = 72, `OUTWARD_GENUINE_N` = 16,663, `details.e1.numerical_artifacts` (rule, counts, ids) |
| B3 dial lane edges | three-adjacent-window rule; `details.e2_lane_edges`; table in section 3 |
| Previews | `preview/04_stretch.png` and `preview/13_real_belt.png`: guide lines only at 3:1 (2.50 au) and 5:2 (2.82 au); 2:1 (3.28 au) marked "not reproduced" with no guide; 4:1 and 7:3 guides removed |
| New keys for on-screen text | `HILDA_INSIDE_PCT`, `HILDA_OUTSIDE_PCT`, `ZONE_EXPONENT_OUTER`, `SCHEME_SPREAD.ci95`, `details.e1.survivors_changed`, `details.e1.reach_survivors`, `details.e1.lane_survivors`, `details.e3.bound_not_crossing` |
