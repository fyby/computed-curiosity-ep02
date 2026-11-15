"""R: real asteroid orbits from the NASA/JPL Small-Body Database (SBDB) Query API.

Source:  https://ssd-api.jpl.nasa.gov/sbdb_query.api
Docs:    https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html

`python run.py real` downloads, ONCE, the minimum needed: semimajor axis, eccentricity,
inclination and orbit class (a, e, i, class) of every asteroid (sb-kind=a) with
1.56 <= a <= 10.41 au, one request per 0.5-au slice, plus one count-only request for
the whole asteroid catalogue. The rows, the exact query URLs and the retrieval time are
stored in results/raw/real_asteroids_YYYYMMDD.json and never fetched again (delete the
file to refresh). Nothing else in this package depends on this stage; without the file
every REAL_* placeholder stays null ("pending").

Derived numbers (FACTS section 6 ranges): REAL_N_TOTAL (all asteroids), REAL_N_BOX
(1.56-10.41 au), REAL_LANE_N (4.2-5.05 au), REAL_TROJAN_N (class TJN), REAL_HILDA_RANGE_N
(3.7-4.2 au, a distance range, not a resonance test), REAL_DATA_DATE; histogram in 0.01-au
bins for shot H9 (viz/real_asteroids_hist.json). The catalogue is a discovery record:
compare positions of features, never heights.
"""
import glob
import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np

from . import paths

SOURCE_URL = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
DOC_URL = "https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html"
A_MIN, A_MAX, SLICE = 1.56, 10.41, 0.5
FIELDS = "a,e,i,class"


def _url(params):
    return SOURCE_URL + "?" + urllib.parse.urlencode(params)


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "computed-curiosity-ep02 (one-time download)"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())


def cached_file():
    files = sorted(glob.glob(os.path.join(paths.RAW, "real_asteroids_*.json")))
    return files[-1] if files else None


def download():
    if cached_file():
        print(f"[real] using cached {cached_file()}", flush=True)
        return cached_file()
    queries, rows = [], []
    edges = [round(x, 2) for x in np.arange(A_MIN, A_MAX, SLICE)] + [A_MAX]
    for lo, hi in zip(edges[:-1], edges[1:]):
        last = hi == A_MAX
        cons = {"AND": [f"a|RG|{lo}|{hi}"]} if last else {"AND": [f"a|GE|{lo}", f"a|LT|{hi}"]}
        url = _url({"fields": FIELDS, "sb-kind": "a", "sb-cdata": json.dumps(cons, separators=(",", ":"))})
        d = _get(url)
        rows += d.get("data", [])
        queries.append(dict(url=url, count=int(d.get("count", len(d.get("data", []))))))
        print(f"[real] a {lo}-{hi}: {queries[-1]['count']} rows", flush=True)
        time.sleep(1.0)
    total_url = _url({"fields": "spkid", "sb-kind": "a", "limit": 1})
    total = int(_get(total_url).get("count"))
    out = dict(source=SOURCE_URL, documentation=DOC_URL, fields=FIELDS.split(","), queries=queries,
               total_count_query=dict(url=total_url, count=total), n_rows=len(rows),
               retrieved_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), rows=rows,
               credit="NASA/JPL Small-Body Database (ssd.jpl.nasa.gov)")
    path = paths.raw(f"real_asteroids_{time.strftime('%Y%m%d', time.gmtime())}.json")
    with open(path, "w") as f:
        json.dump(out, f)
    return path


def load():
    path = cached_file()
    if not path:
        return None
    with open(path) as f:
        d = json.load(f)
    col = lambda k: np.array([float(r[k]) if r[k] not in (None, "") else np.nan for r in d["rows"]])
    a, e, inc = col(0), col(1), col(2)
    cls = np.array([r[3] or "" for r in d["rows"]])
    return d, a, e, inc, cls


def derived():
    """Numbers and the histogram from the cached file, or None if there is no file."""
    got = load()
    if got is None:
        return None
    d, a, e, inc, cls = got

    def cnt(lo, hi):
        return int(np.sum((a >= lo) & (a < hi)))
    edges = np.round(np.arange(A_MIN, A_MAX + 0.005, 0.01), 2)
    h, _ = np.histogram(a[np.isfinite(a)], edges)
    res = dict(date=d["retrieved_utc"][:10], total=d["total_count_query"]["count"],
               box=int(np.sum((a >= A_MIN) & (a <= A_MAX))), lane_4p2_5p05=cnt(4.2, 5.05),
               hilda_range_3p7_4p2=cnt(3.7, 4.2), trojans_tjn=int(np.sum(cls == "TJN")),
               ranges={"2.06-3.28": cnt(2.06, 3.28), "3.28-3.7": cnt(3.28, 3.7), "5.05-5.35": cnt(5.05, 5.35),
                       "5.35-10.41": cnt(5.35, 10.41), "2.44-2.48": cnt(2.44, 2.48), "2.48-2.52": cnt(2.48, 2.52),
                       "2.52-2.56": cnt(2.52, 2.56), "2.80-2.84": cnt(2.80, 2.84), "3.26-3.30": cnt(3.26, 3.30)},
               mba=int(np.sum(cls == "MBA")), file=os.path.basename(cached_file()),
               queries=[q["url"] for q in d["queries"]])
    with open(paths.viz("real_asteroids_hist.json"), "w") as f:
        json.dump(dict(a_edges=edges.tolist(), count=h.tolist(), date=res["date"], source=SOURCE_URL,
                       note="number of catalogued asteroids per 0.01-au bin of semimajor axis (measured, amber). "
                            "A discovery record: compare positions, not heights."), f)
    return res


def run():
    download()
    print(json.dumps({k: v for k, v in derived().items() if k != "queries"}, indent=1), flush=True)
