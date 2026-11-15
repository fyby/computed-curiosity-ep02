"""Where results are written.

Default: ``results/`` next to ``run.py`` (i.e. inside the folder that holds
this package), so the ``sim/`` folder works when it is published on its own.
Override with the environment variable ``MILLION_RESULTS``.
"""
import os

SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # folder of run.py
RESULTS = os.path.abspath(os.environ.get("MILLION_RESULTS", os.path.join(SIM_DIR, "results")))
RAW = os.path.join(RESULTS, "raw")          # per-planet tables, chunk files, validation
VIZ = os.path.join(RESULTS, "viz")          # compact data for the renderers
PREVIEW = os.path.join(RESULTS, "preview")  # quick matplotlib PNGs
LOGS = os.path.join(RESULTS, "logs")        # progress logs

for _d in (RESULTS, RAW, VIZ, PREVIEW, LOGS):
    os.makedirs(_d, exist_ok=True)


def _mk(base, parts):
    p = os.path.join(base, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def raw(*parts):
    return _mk(RAW, parts)


def viz(*parts):
    return _mk(VIZ, parts)


def preview(name):
    return os.path.join(PREVIEW, name)


def log(name):
    return os.path.join(LOGS, name)
