"""Quick-look PNGs (results/preview/). Not the video graphics: checks for humans."""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import engine, paths
from .constants import A_G, A_LO, A_HI, P_G

plt.rcParams.update({"figure.dpi": 110, "font.size": 9})


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(paths.preview(name))
    plt.close(fig)


def _density(ax, xy, title, lim=11.5):
    ok = np.isfinite(xy[:, 0])
    ax.hist2d(xy[ok, 0], xy[ok, 1], bins=500, range=[[-lim, lim], [-lim, lim]], cmap="magma",
              norm=matplotlib.colors.LogNorm())
    ax.plot([A_G], [0], "o", ms=4, mfc="w", mec="w")
    ax.add_patch(plt.Circle((A_G, 0), 0.3553, fill=False, color="w", lw=0.6))
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("black")


def _guides(ax, st, top, color="orange", vertical=False):
    """Guide lines at 3:1 (2.50 au) and 5:2 (2.82 au) only; 2:1 gets a cross and "not reproduced"."""
    for k, a in st["guides_au"].items():
        ax.axvline(a, color=color, lw=0.7, ls="--")
        if vertical:
            ax.text(a - 0.03, top * 0.97, f"{k} · {a:.3f} au", ha="right", va="top", fontsize=7, rotation=90)
        else:
            ax.text(a, top * 0.93, f"{k}\n{a:.3f} au", ha="center", va="top", fontsize=8)
    for k, a in st["not_reproduced_au"].items():
        ax.plot([a], [top * 0.80], marker="x", color="crimson", ms=7, mew=1.6, ls="none")
        ax.text(a + 0.04, top * 0.80, f"{k} ({a:.2f} au): not reproduced", color="crimson", ha="left", va="center", fontsize=8)


def run():
    R = json.load(open(os.path.join(paths.RESULTS, "RESULTS.json")))
    P = R["placeholders"]
    cur = json.load(open(paths.viz("e1_curves.json")))

    # 01 map
    g = np.load(paths.viz("e1_map_grid.npz"))
    with np.errstate(invalid="ignore"):
        sv = g["survivors"] / g["count"]
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(sv.T, origin="lower", aspect="auto", extent=[A_LO, A_HI, 0, 0.3], cmap="RdYlGn", vmin=0, vmax=1)
    ln = cur["lane"]
    for x in (ln["inner"], ln["outer"]):
        ax.axvline(x, color="k", lw=0.6, ls=":")
    rl = cur["reach_lane"]
    ee = np.linspace(0, 0.3, 50)
    ax.plot(rl[0] / (1 + ee), ee, "k--", lw=0.8)
    ax.plot(rl[1] / (1 - ee), ee, "k--", lw=0.8)
    ax.set_xlim(A_LO, A_HI)
    ax.set_xlabel("starting distance a0 (au)")
    ax.set_ylabel("starting eccentricity e0")
    ax.set_title(f"E1 map: survival to {P['T_MAIN_YR']['value']} yr (green) / lost (red); dashed = simple guess; dotted = lane edges")
    fig.colorbar(im, ax=ax, label="survival fraction per cell")
    _save(fig, "01_map.png")

    # 02 survival vs distance
    fig, ax = plt.subplots(figsize=(11, 3.5))
    for k, lab in (("all", "all e0"), ("e0-0.05", "e0 < 0.05"), ("e0.05-0.15", "0.05-0.15"), ("e0.15-0.3", "0.15-0.3")):
        s = cur["survival_by_a_0p02"][k]
        x = 0.5 * (np.array(s["lo"]) + np.array(s["hi"]))
        ax.plot(x, np.array(s["surv"], float) * 100, lw=1.2 if k == "all" else 0.7, label=lab)
    ax.set_xlabel("a0 (au)")
    ax.set_ylabel("survival %")
    ax.legend()
    ax.set_title("E1 survival against starting distance (0.02-au bins)")
    _save(fig, "02_survival_vs_a.png")

    # 03 loss curve
    lc = cur["loss_curve"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogx(lc["t_yr"], lc["lost_pct"])
    ax.axvline(P["HALF_LOSS_YR"]["value"], color="gray", ls=":", label="half of all losses")
    ax.axvline(P_G, color="c", ls=":", label="1 lap")
    ax.set_xlabel("time (yr)")
    ax.set_ylabel("lost %")
    ax.legend()
    ax.set_title("E1 loss curve")
    _save(fig, "03_loss_curve.png")

    # 04 stretching (fact-check B4: mean WITHOUT the planets with an unphysical kick; guides only at the two
    # resonances our run reproduces; 2:1 is marked "not reproduced", never as a guide)
    st = cur["stretch"]
    n_art = st.get("artifacts_excluded", 0)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(st["a_centre"], st["mean_gain"], lw=0.9, color="teal", label=f"mean, {n_art} artifact planets dropped")
    ax.plot(st["a_centre"], st["median_gain"], lw=0.6, color="gray", label="median, all planets")
    top = float(np.nanmax(np.array(st["mean_gain"], float))) * 1.25
    ax.set_ylim(0, top)
    _guides(ax, st, top)
    ax.set_xlabel("a0 (au)")
    ax.set_ylabel("max e - e0 per 0.01-au bin")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title(f"E1 stretching; 3:1 factor {P['STRETCH_31_FACTOR']['value']:.2f}; guides at 3:1 and 5:2 only")
    _save(fig, "04_stretch.png")

    # 05 trojan phase bins
    pb = cur["phase_bins"]["bins"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar([0.5 * (b["lo"] + b["hi"]) for b in pb], [b["value"] for b in pb], width=28)
    ax.set_xlabel("starting phase relative to the giant (deg)")
    ax.set_ylabel("survival %")
    ax.set_title("E1 co-orbital planets (|a0 - a_G| < 0.25 au)")
    _save(fig, "05_trojan_phase.png")

    # 06 dials
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    mk = [("LOST_M01_PCT", 0.1), ("LOST_M03_PCT", 0.3), ("LOST_BASE10K_PCT", 1), ("LOST_M3_PCT", 3), ("LOST_M10_PCT", 10)]
    ek = [("LOST_E00_PCT", 0.0), ("LOST_BASE10K_PCT", 0.0487), ("LOST_E010_PCT", 0.1), ("LOST_E023_PCT", 0.23), ("LOST_E04_PCT", 0.4)]
    for ax, ks, lab, logx in ((axs[0], mk, "giant mass (Jupiters)", True), (axs[1], ek, "giant eccentricity", False)):
        xs = [x for k, x in ks if k in P and P[k]["value"] is not None]
        ys = [P[k]["value"] for k, x in ks if k in P and P[k]["value"] is not None]
        lo = [max(0, P[k]["value"] - P[k]["ci95"][0]) for k, x in ks if k in P and P[k]["value"] is not None]
        hi = [max(0, P[k]["ci95"][1] - P[k]["value"]) for k, x in ks if k in P and P[k]["value"] is not None]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(lab)
        ax.set_ylabel("lost by 843 laps (%)")
    fig.suptitle("E2 dials: the same 100,000 planets")
    _save(fig, "06_dials.png")

    # 07 dial survival strips
    e2 = json.load(open(paths.viz("e2_curves.json")))
    fig, ax = plt.subplots(figsize=(11, 4))
    for k, r in e2.items():
        s = r["survival_by_a_0p05"]
        x = 0.5 * (np.array(s["lo"]) + np.array(s["hi"]))
        ax.plot(x, np.array(s["surv"], float) * 100, lw=1.6 if k == "base" else 0.8, label=k)
    ax.legend(ncol=3)
    ax.set_xlabel("a0 (au)")
    ax.set_ylabel("survival %")
    ax.set_title("E2 survival against distance for every giant (843 laps)")
    _save(fig, "07_dial_survival.png")

    # 08 fates
    if "FATE_BOUND_PCT" in P:
        fig, ax = plt.subplots(figsize=(6, 3))
        ks = ["FATE_BOUND_PCT", "FATE_GIANT_PCT", "FATE_EJECT_PCT", "FATE_STAR_PCT"]
        ax.barh([k[5:-4] for k in ks], [P[k]["value"] for k in ks],
                xerr=[[max(0, P[k]["value"] - P[k]["ci95"][0]) for k in ks], [max(0, P[k]["ci95"][1] - P[k]["value"]) for k in ks]])
        ax.set_xlabel("% of 1,000 re-run lost planets after 200,000 yr")
        _save(fig, "08_fates.png")

    # 09 validation V5
    v = json.load(open(paths.viz("validation.json")))["v5"]
    rows = [r for r in v["rows"] if r["inner_dist_rel"]]
    mu = np.array([r["mu"] for r in rows])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(mu, [r["inner_dist_rel"] for r in rows], "o", label="ours (inner edge)")
    mm = np.geomspace(5e-5, 5e-3)
    ax.loglog(mm, 1.3 * mm ** (2 / 7), "--", color="orange", label="1.3 mu^(2/7)")
    ax.loglog(mm, 1.57 * mm ** 0.286, ":", color="orange", label="1.57 mu^0.286")
    ax.loglog(mm, v["coefficient"] * mm ** v["exponent"], "-", color="teal", lw=0.8,
              label=f"fit: {v['coefficient']:.2f} mu^{v['exponent']:.3f}")
    ax.set_xlabel("giant mass / star mass")
    ax.set_ylabel("lane half-width / a_G")
    ax.legend(fontsize=7)
    _save(fig, "09_validation_v5.png")

    # 10 strobes
    meta = json.load(open(paths.viz("e1_strobe_meta.json")))
    arr = np.load(paths.viz("e1_strobe.npy"), mmap_mode="r")
    laps = meta["laps"]
    pick = [0, laps.index(min(240, laps[-1])), len(laps) - 1]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, i in zip(axs, pick):
        xy = engine.dequantize(np.asarray(arr[i]))
        _density(ax, xy, f"E1 strobe lap {laps[i]} (t = {meta['t_yr'][i]:.0f} yr)")
    _save(fig, "10_e1_strobes.png")

    # 11 E2 final strobes
    names = [k for k in ("e00", "e010", "e023", "e04", "m01", "m03", "m3", "m10")
             if os.path.exists(paths.viz("e2", f"{k}_strobe.npy"))]
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for ax, k in zip(axs.flat, names):
        a = np.load(paths.viz("e2", f"{k}_strobe.npy"), mmap_mode="r")
        _density(ax, engine.dequantize(np.asarray(a[-1])), f"E2 {k}, last strobe")
    _save(fig, "11_e2_last_strobes.png")

    # 12 replays
    fig, axs = plt.subplots(2, 2, figsize=(11, 10))
    rp = paths.viz("replays")
    if os.path.exists(os.path.join(rp, "trojans.npz")):
        z = np.load(os.path.join(rp, "trojans.npz"))
        xy = z["xy_rot"].astype(float)
        for j in range(0, xy.shape[1], max(1, xy.shape[1] // 40)):
            axs[0, 0].plot(xy[:, j, 0], xy[:, j, 1], lw=0.3)
        axs[0, 0].plot(A_G, 0, "ko")
        axs[0, 0].set_aspect("equal")
        axs[0, 0].set_title("trojans.npz (rotating frame, 40 of them)")
    if os.path.exists(os.path.join(rp, "res31.npz")):
        z = np.load(os.path.join(rp, "res31.npz"))
        axs[0, 1].plot(z["t_yr"], z["e"], lw=0.5)
        axs[0, 1].set_title(f"res31.npz: planet {int(z['id'])}, eccentricity")
        axs[0, 1].set_xlabel("yr")
    if os.path.exists(os.path.join(rp, "ejection.npz")):
        z = np.load(os.path.join(rp, "ejection.npz"))
        x = z["xyz_bary"]
        axs[1, 0].plot(x[:, 2, 0], x[:, 2, 1], lw=0.2)
        axs[1, 0].plot(x[:, 1, 0], x[:, 1, 1], lw=0.3, color="k")
        axs[1, 0].set_aspect("equal")
        axs[1, 0].set_title(f"ejection.npz: planet {int(z['id'])}, ejected at {float(z['t_fate']):.0f} yr")
    if os.path.exists(os.path.join(rp, "first_laps.npz")):
        z = np.load(os.path.join(rp, "first_laps.npz"))
        x = z["xyz"].astype(float)
        for j in range(0, x.shape[1], 50):
            axs[1, 1].plot(x[:, j, 0], x[:, j, 1], lw=0.3, color="r" if z["cls"][j] else "teal")
        axs[1, 1].set_aspect("equal")
        axs[1, 1].set_title("first_laps.npz (every 50th, red = lost)")
    _save(fig, "12_replays.png")

    # 13 real belt (only with the JPL download)
    hp = paths.viz("real_asteroids_hist.json")
    if os.path.exists(hp):
        h = json.load(open(hp))
        e = np.array(h["a_edges"])
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.step(e[:-1], np.maximum(h["count"], 0.8), where="post", color="orange", lw=0.7)
        ax.set_yscale("log")
        ax.set_ylabel("real asteroids per 0.01 au (JPL)", color="orange")
        ax2 = ax.twinx()
        ax2.plot(st["a_centre"], st["mean_gain"], color="teal", lw=0.8)
        ax2.set_ylabel(f"our mean eccentricity gain ({n_art} artifact planets dropped)", color="teal")
        top = float(np.nanmax(np.array(st["mean_gain"], float))) * 1.25
        ax2.set_ylim(0, top)
        _guides(ax2, st, top, color="gray", vertical=True)
        ax.set_xlim(A_LO, A_HI)
        ax.set_title(f"Real belt ({h['date']}) vs our stretching: compare positions, not heights")
        _save(fig, "13_real_belt.png")
    print("[previews] written to", paths.PREVIEW, flush=True)
