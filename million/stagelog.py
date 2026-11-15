"""Wall-clock time of every stage, collected for run_meta.json."""
import json
import os
import time

from . import paths


def record(stage, wall_s, **info):
    path = paths.raw("stage_times.json")
    d = {}
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
    d[stage] = dict(wall_s=round(float(wall_s), 2), finished=time.strftime("%Y-%m-%d %H:%M:%S"), **info)
    with open(path, "w") as f:
        json.dump(d, f, indent=1)


def load():
    path = paths.raw("stage_times.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
