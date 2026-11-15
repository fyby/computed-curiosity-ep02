# One star, one giant, 1,000,000 test planets — simulation code

Source code for episode 2 of **Computed Curiosity**. Every number and every orbit
shown in the video comes from this code and the result files it writes.

**What "1,000,000 planets" means.** One star. One giant planet. 1,000,000 massless
**test planets**: each feels the star and the giant, none has mass, so they never
pull on each other or on the giant. It is a million one-planet experiments that share
a star and a giant, computed together — the standard "test particle" method.

The question: when a star has a giant planet, where can a smaller world keep its orbit?

| Experiment | What | Size |
|---|---|---|
| **E0** control | the giant's mass set to zero: nothing may change | 100,000 × 1,008 yr |
| **E1** main run | a copy of Jupiter (mass 1/1047.35, a = 5.2038 au, e = 0.0487); test planets with a = 1.561–10.408 au, e = 0–0.3, tilt 0–5°, random angles | 1,000,000 × 50,000 yr (40,000 yr if the code projects > 85 min) |
| **E2** dials | the same 100,000 test planets around 8 other giants: eccentricity 0, 0.10, 0.23, 0.40; mass 0.1, 0.3, 3, 10 Jupiters | 8 × 100,000 × 843 laps (~10,000 yr) |
| **E3** fates | 1,000 planets that E1 counted as lost, re-run alone from t = 0 with IAS15, real radii for star and giant | 1,000 × 200,000 yr |
| **E4** replays | dense trajectories for the hero shots, chosen by rules written before they were computed | 7 clips |
| **R** real asteroids | NASA/JPL Small-Body Database: a, e, i, class of all asteroids with a = 1.56–10.41 au (downloaded once) | ~1.55 million rows |


**Not in this repository:** `results/raw/` and `results/viz/` (about 4.1 GB of per-planet tables, frame data and the JPL download). `make all` regenerates them bit-identically from the seeds; `results/RESULTS.json`, `RESULTS.md`, `run_meta.json`, the logs and the preview charts are included.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make all            # = python run.py all   (validation, E1, E2, E3, replays, results, previews)
python run.py real  # optional one-time download of the JPL asteroid orbits (network), then: python run.py results previews
```

On a 10-core Apple M4 Mac mini `make all` takes about 1.5–2 hours. Single stages:
`python run.py e2`, `python run.py results`, … (`python run.py` lists them).
Every stage is resumable: kill it, start it again, and it continues where it stopped
(E1 chunks write a checkpoint every 169 laps ≈ 2,005 yr; resuming is bit-identical, test V8).
`make smoke` runs the whole pipeline on a tiny problem in `results_smoke/` (meaningless numbers).

Environment variables:

* `MILLION_RESULTS` — output folder. **Default: `results/` next to `run.py`**, so this
  folder works on its own when published.
* `MILLION_PROCESSES` — worker processes (default 10).
* `MILLION_SMOKE=1` — pipeline test only.

Seeds: E1 draws `numpy.random.default_rng(2_000_001)` in the order a, e, inclination,
node, argument of pericentre, mean anomaly; planet id = array index. E2 uses ids
0–99,999 of the same disc. E3 samples with `default_rng(2_000_003)`. Validation tests
use `2_100_000 + test number`.

## The model

* **Physics:** Newtonian gravity of the star (1 solar mass) and the giant on each test
  planet, and of star and giant on each other. G = 4π² exactly (au, yr, solar masses;
  set directly, because REBOUND's `units` option would use the Gaussian constant).
* **Integrator:** [REBOUND](https://rebound.hanno-rein.de/) 5.2.2, WHFast (Wisdom–Holman),
  democratic heliocentric coordinates, no corrector, `safe_mode = 0`. Step by starting
  distance so every orbit gets ≥ 20 steps per pericentre passage:
  1.561–1.72 au P/240, 1.72–2.25 P/180, 2.25–3.60 P/120, 3.60–10.408 P/60 (P = 11.865 yr).
  A test planet keeps its step size for the whole run.
* **Outcomes** (tested every P/15 = 0.79 yr; a planet stops at its first event):
  `ENCOUNTER` entered the giant's Hill sphere (0.3553 au; the closest approach between two
  checks is reconstructed in the frame turning with the giant — straight segment, refined
  by a cubic Hermite curve through both end states near the giant — so fast passes are
  not missed), `INWARD` closer than 1.041 au to the star, `OUTWARD` farther than 52 au,
  else `SURVIVED`. **"Lost" = any of the three. It does not mean destroyed**: the model
  stops following the planet. E3 follows a sample further.
* **Not in the model:** masses of the small planets, a second giant (no Saturn, so none
  of the secular resonances that empty the real Kirkwood gaps), times beyond 50,000 yr,
  gas, collisions between small bodies, tides, relativity, the star's size (except E3),
  formation or migration of the giant.
* **Statistics:** every percentage has n and a 95% Wilson interval; the "scheme spread"
  (V4: production vs half step and half check interval) is listed next to it. The
  headline percentage depends on the sampling box; the map is the finding.

## Known numerical artifacts

72 of the 1,000,000 planets of the main run (0.007%) received an unphysical kick in an
unresolved close pass by the giant; they are counted as lost, which they would be anyway.

* **What happened:** the giant is a point mass and WHFast keeps a fixed step (0.2 yr in the
  outer zone). A planet that passes very deep inside the Hill sphere between two event tests
  can come out with a wildly wrong velocity. The event test then finds it beyond 52 au
  (`OUTWARD`) instead of inside the Hill sphere (`ENCOUNTER`).
* **How they are identified** (`million/analysis.py`, `artifact_mask`; from the saved table,
  nothing is re-run): `OUTWARD` exits with (a) an orbital-element sample with a ≤ 0, i.e. a
  hyperbolic "orbit" with |a| under 1 au (31 planets, eccentricities 8 to 8,120), or (b) a
  last element sample, at most 4 yr before the event, with its far point under 20 au
  (41 planets; the largest is 13 au, while every genuine outward exit has it beyond 52 au).
* **What it changes:** no headline number. `OUTWARD_N` = 16,735 = 16,663 genuine + 72
  (`ARTIFACT_N`, `OUTWARD_GENUINE_N`, ids in `details.e1.numerical_artifacts`). The stretching
  profile (`viz/e1_curves.json`, `e_gain` in `viz/e1_map_points.npz`) leaves these planets
  out: four of them had produced false spikes at 3.70, 4.12, 4.20 and 4.24 au. The profile is
  the per-bin mean without them; the per-bin median of all planets is exported next to it.
* No validation test catches them. A fix in the engine (an adaptive step or a smaller
  check interval near the giant) is not in this version.

## Lane edges

A lane edge is a 50% survival edge for round orbits (e0 < 0.05): walking toward the giant in
0.02-au steps, the centre of the first of **three adjacent 0.04-au windows that are all below
50% survival** (`analysis.lane(..., consecutive=3)`). The first version used the first single
window, and in the dial run with eccentricity 0.23 one resonance window (2.82 au, 49.4%)
set the "edge" 0.6 au too far in. The main run's edges (4.06 and 6.60 au) are the same under
both rules. The validation tests V4 and V5 were run with the first-window rule and were not
re-run.

## Validation (run first; production refuses to start if it fails)

| | Test | Pass condition |
|---|---|---|
| V1 | E0 control (giant mass 0, 100,000 planets) and the giant alone 50,000 yr at all four steps | a, e unchanged to 1e-9; period = Kepler to 1e-6, phase within 0.5° |
| V2 | circular giant, 20,000 planets, 1,000 laps: Jacobi constant | median drift < 1e-6, 99th pct < 1e-4 |
| V3 | WHFast vs IAS15 (one planet per simulation), 5,000 planets, 5,000 yr | lost fractions agree within the 95% interval of the paired difference |
| V4 | half step and half check interval, 20,000 planets, 10,000 yr | lost fraction within 0.5 points, lane edges within 0.04 au |
| V5 | **published law**: lane width against giant mass (4 masses, circular giant, round orbits) | exponent 0.24–0.33 (Wisdom 1980: 2/7), coefficient 1.0–1.8 |
| V6 | resonance positions of the E1 stretching peaks (3:1, 5:2, 2:1) | within 0.03 au, and a genuine local peak ≥ 1.5 × background |
| V7 | 2,000 Trojan test planets, 5,000 yr | ≥ 95% survive, median libration period 135–165 yr |
| V8 | determinism: one chunk twice, split in two, interrupted and resumed | identical class and event time for every planet |
| V9 | event test against references that test after every step (same trajectories, bit-identical) | production never later by > 1 check for ≥ 99%; ≥ 99% within one check of the per-step segment reference |

Deviations from the written spec, and why: V9's reference uses every integration
step (P/240 … P/60) instead of P/120, because P/120 is not a whole number of zone-D
steps and changing the step would change the orbits; the references are integrated
bit-identically to production so only the event test differs. V6's literal condition
is true by construction, so a genuine-peak condition was added.

## Code layout

```
run.py                   command-line entry point (stages)
Makefile                 make all / validate / production / results / smoke
million/constants.py     physical constants, zones, seeds, run lengths
million/paths.py         output folders (MILLION_RESULTS override; default ./results)
million/disc.py          the random draws, chunking by step-size zone
million/engine.py        WHFast chunk worker: events, element samples, strobes, checkpoints
million/ias15.py         one planet with IAS15 (V3 and the E3 fates)
million/runset.py        run a set of planets as resumable chunks, merge tables and strobes
million/pool.py          multiprocessing with resume
million/validation.py    V1-V9
million/e1.py            main run with the 85-minute fallback rule
million/e2.py            the dials
million/fates.py         E3
million/replays.py       E4 clips (rules in viz/replays/index.json)
million/realdata.py      R: JPL Small-Body Database download (once) and counts
million/analysis.py      lane edges, reach, resonance peaks, phase bins, libration, artifact mask
million/results.py       RESULTS.json, run_meta.json, viz exports
million/previews.py      quick-look PNGs
```

## Output (`results/`)

```
RESULTS.json             every number for the script: value, 95% interval, n, scheme spread, source; [RD] checks
RESULTS.md               what we found, in plain words
run_meta.json            versions, machine, settings, seeds, E1 fallback decision, wall time per stage
raw/                     per-planet tables (e1_planets.npz, e2_<case>.npz), chunk files, validation.json,
                         e3_fates.jsonl + e3_tracks/, cached JPL download (real_asteroids_YYYYMMDD.json)
viz/e1_strobe.npy        uint16 (frames, 1,000,000, 2): every planet in the frame turning with the giant,
                         every lap 0-240 then ~120 log-spaced laps (meta: e1_strobe_meta.json)
viz/e1_map_points.npz    per planet: start a, e, tilt, phase, class, loss time, reach flag, e gain (NaN for artifacts), artifact flag
viz/e1_map_grid.npz      850 x 300 survival map and median loss time
viz/e1_curves.json       loss curve, survival vs distance, stretching (mean without artifacts + median), phase bins, lane
viz/e1_elements_subset.npz  a, e, varpi, lambda of planets 0-99,999 at ~60 log-spaced laps
viz/e2/<case>_strobe.npy  64 strobes of each dial run (100,000 planets); viz/e2_curves.json
viz/e3_fates.json, e3_tracks.npz   fates and 100-yr thinned tracks of the 1,000 E3 planets
viz/e0_control_strobe.npy  control run strobes (100,000 planets, every lap)
viz/replays/             first_laps, control, trojans, res31, ejection, giant_hit, wanderer + index.json
viz/real_asteroids_hist.json  JPL counts per 0.01 au (only after `run.py real`)
preview/                 quick matplotlib plots
```

## Data source for the real asteroids

NASA/JPL Small-Body Database Query API, https://ssd-api.jpl.nasa.gov/sbdb_query.api
(documentation https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html). Query: `sb-kind=a`,
fields `a,e,i,class`, semimajor axis 1.56–10.41 au in 0.5-au slices, plus one count
query for the whole catalogue. The exact URLs and the retrieval date are stored with the
data and in `run_meta.json`. Credit: NASA/JPL Solar System Dynamics.

## Credits

REBOUND (Rein & Liu 2012; WHFast: Rein & Tamayo 2015; IAS15: Rein & Spiegel 2015), GPL v3.
Wisdom & Holman 1991 (the symplectic map); Wisdom 1980 (resonance-overlap law, via
Lecar, Franklin, Holman & Murray 2001).
