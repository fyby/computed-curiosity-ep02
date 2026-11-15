#!/usr/bin/env python
"""Command-line entry point.  Usage:  python run.py <stage> [<stage> ...]   (or: python run.py all)

Stages, in the order `all` runs them:
  validate   V1-V9 validation gate (V6 is checked on the main run); production refuses to start if it fails
  e1         main run: 1,000,000 test planets, copy of Jupiter, 50,000 yr (40,000 yr fallback, decided by the code)
  e2         dials: the same 100,000 test planets around 8 other giants, 843 laps (~10,000 yr) each
  e3         fates: 1,000 lost planets re-run alone with IAS15 for 200,000 yr
  replays    E4 dense clips for the hero shots (selection rules written first)
  results    results/RESULTS.json, run_meta.json and the viz/ exports
  previews   quick-look PNGs in results/preview/
Optional:
  e5         20,000 test planets at 2.0-3.6 au for 300,000 yr (only if time allows)
  real       JPL asteroid comparison: NOT IMPLEMENTED YET (needs an approved network download; see README)

Every stage is resumable: re-running it skips work that is already on disk.
Environment: MILLION_RESULTS (output folder, default ./results next to this file),
MILLION_PROCESSES (worker processes, default 10), MILLION_SMOKE=1 (tiny pipeline test, meaningless numbers).
"""
import json
import os
import sys
import time

from million import paths

ORDER = ["validate", "e1", "e2", "e3", "replays", "results", "previews"]


def require_validation():
    path = paths.raw("validation.json")
    if not os.path.exists(path):
        sys.exit("Run `python run.py validate` first.")
    with open(path) as f:
        if not json.load(f)["all_passed"] and "--force" not in sys.argv:
            sys.exit("Validation gate failed: see results/raw/validation.json (use --force to override).")


def main(stages):
    for st in stages:
        t0 = time.time()
        print(f"=== {st} === {time.strftime('%H:%M:%S')}", flush=True)
        if st == "validate":
            from million import validation, stagelog
            only = [a for a in sys.argv if a.startswith("V")]
            validation.run(only=only or None)
            stagelog.record("validation" + ("_" + "_".join(only) if only else ""), time.time() - t0)
            continue
        if st in ("results", "previews", "real"):
            pass
        else:
            require_validation()
        if st == "e1":
            from million import e1; e1.run()
        elif st == "e2":
            from million import e2; e2.run()
        elif st == "e3":
            from million import fates; fates.run()
        elif st == "replays":
            from million import replays; replays.run()
        elif st == "e5":
            from million import e5; e5.run()
        elif st == "results":
            from million import results; results.run()
        elif st == "previews":
            from million import previews; previews.run()
        elif st == "real":
            from million import realdata; realdata.run()
        else:
            sys.exit(f"unknown stage {st}")
        print(f"=== {st} done in {(time.time() - t0) / 60:.1f} min ===", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--") and not (a.startswith("V") and a[1:].isdigit())]
    if not args:
        sys.exit(__doc__)
    main(ORDER if args == ["all"] else args)
