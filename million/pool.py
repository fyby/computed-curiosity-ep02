"""Resumable multiprocessing.

* `run_chunks`: WHFast chunks (engine.run_chunk). A chunk whose output file exists is
  skipped; an interrupted chunk resumes from its last checkpoint.
* `run_items`: any picklable worker over a list of items, one JSON line per finished
  item in a .jsonl file; items already in the file are skipped.
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from . import engine, paths
from .constants import N_PROCESSES


def _log(name, msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(f"[{name}] {line}", flush=True)
    with open(paths.log(name + ".log"), "a") as f:
        f.write(line + "\n")


def run_chunks(name, jobs, processes=N_PROCESSES):
    """Run every job not finished yet. Returns the list of metas of all jobs (new and old)."""
    todo = []
    for j in jobs:
        if os.path.exists(j["out"]) and os.path.exists(j["out"] + ".meta.json"):
            continue
        todo.append(j)
    _log(name, f"{len(jobs)} chunks, {len(jobs) - len(todo)} already done, {len(todo)} to run, {processes} processes")
    t0 = time.time()
    done_ps = 0
    if todo:
        with Pool(processes) as p:
            for k, meta in enumerate(p.imap_unordered(engine.run_chunk, todo, chunksize=1)):
                done_ps += meta["psteps"]
                if meta["done"]:
                    with open(meta["out"] + ".meta.json", "w") as f:
                        json.dump(meta, f)
                el = time.time() - t0
                _log(name, f"chunk {k + 1}/{len(todo)} {os.path.basename(meta['out'])} "
                           f"n={meta['n']} alive={meta['alive']} wall={meta['wall']:.0f}s "
                           f"elapsed={el / 60:.1f} min")
    metas = []
    for j in jobs:
        mp = j["out"] + ".meta.json"
        if os.path.exists(mp):
            with open(mp) as f:
                metas.append(json.load(f))
    return metas, time.time() - t0


def run_items(name, items, worker, key, processes=N_PROCESSES, every=50):
    """worker(item) -> JSON-able dict. Resumable through raw/<name>.jsonl."""
    path = paths.raw(name + ".jsonl")
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r[key]] = r
    todo = [it for it in items if it[key] not in done]
    _log(name, f"{len(items)} items, {len(done)} already done, {len(todo)} to run")
    t0 = time.time()
    if todo:
        with Pool(processes) as p, open(path, "a") as f:
            for k, r in enumerate(p.imap_unordered(worker, todo, chunksize=1)):
                f.write(json.dumps(_jsonable(r)) + "\n")
                f.flush()
                done[r[key]] = json.loads(json.dumps(_jsonable(r)))
                if (k + 1) % every == 0 or k + 1 == len(todo):
                    _log(name, f"{k + 1}/{len(todo)} elapsed={(time.time() - t0) / 60:.1f} min")
    return [done[it[key]] for it in items], time.time() - t0


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def merge(outs, fields, order_by="ids"):
    """Concatenate per-chunk npz files and sort by planet id. Returns dict of arrays."""
    parts = {f: [] for f in fields}
    ids = []
    for o in outs:
        with np.load(o) as z:
            ids.append(z["ids"])
            for f in fields:
                parts[f].append(z[f])
    ids = np.concatenate(ids)
    order = np.argsort(ids)
    res = {f: np.concatenate(parts[f])[order] for f in fields}
    res["ids"] = ids[order]
    return res
