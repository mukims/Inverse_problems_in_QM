"""Evaluate Shazam's ensemble lookup (LOOKUP-1, BUILD-27): tests T1-T5, speed, and today's Shazam side by side.

Every test signature is what an experimentalist would supply: energies in eV and the raw T, with no labels.
"""
import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from atlaslib import CloudStore, MultiStore
from atlaslib.energy import on_axis
from atlaslib.lookup import Catalogue, lookup
from tbribbon.materials import make_model

HERE = Path(__file__).resolve().parent
WINDOWS = [(0.0, 0.5), (0.0, 1.0), (1.0, 3.0), (3.0, 8.3)]
BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
TOP = 0.0505            # pilot densities up to 5% are T2; above are T5


def model_of(mid):
    mat, edge, n = mid.split("/")
    return make_model(mat, edge, int(n[1:]))


def signatures(cat, store, mid, density, seed_min, limit):
    """(energies in eV, raw T rows) for the seeds >= seed_min, at most `limit` of them."""
    e_t, _ = store.read_pristine(mid)
    e_ev, _ = on_axis(cat.spec, model_of(mid), e_t)
    cloud, seeds = store.read_cloud(mid, density)
    return e_ev, cloud[np.flatnonzero(seeds >= seed_min)[:limit]]


def run(cat, items, window=None):
    recs = []
    for mid, c, e, rows in items:
        sel = np.ones(e.size, bool) if window is None else (e >= window[0] - 1e-9) & (e <= window[1] + 1e-9)
        for T in rows:
            r = lookup(cat, e[sel], T[sel])
            recs.append({"true": mid, "c": float(c), "dev": r.device, "p": r.probability, "conc": r.concentration,
                         "lo": r.concentration_lo, "hi": r.concentration_hi, "no_match": r.no_match,
                         "top3": [r.device] + [d for d, _ in r.runners_up[:2]]})
    return recs


def summary(recs):
    if not recs:
        return {"n": 0}
    part = lambda s, k: "/".join(s.split("/")[:k])
    right = [r for r in recs if r["dev"] == r["true"]]
    rel = [abs(r["conc"] - r["c"]) / r["c"] for r in right if r["c"] > 0]
    return {"n": len(recs),
            "device_pct": round(100 * len(right) / len(recs), 3),
            "material_pct": round(100 * float(np.mean([part(r["dev"], 1) == part(r["true"], 1) for r in recs])), 3),
            "material_edge_pct": round(100 * float(np.mean([part(r["dev"], 2) == part(r["true"], 2) for r in recs])), 3),
            "top3_pct": round(100 * float(np.mean([r["true"] in r["top3"] for r in recs])), 3),
            "median_rel_err_pct": round(100 * float(np.median(rel)), 2) if rel else None,
            "mae_pp": round(100 * float(np.mean([abs(r["conc"] - r["c"]) for r in right])), 4) if right else None,
            "coverage_pct": round(100 * float(np.mean([r["dev"] == r["true"] and r["lo"] <= r["c"] <= r["hi"]
                                                       for r in recs])), 3),
            "no_match_pct": round(100 * float(np.mean([r["no_match"] for r in recs])), 3),
            "silent_wrong_pct": round(100 * float(np.mean([r["dev"] != r["true"] and not r["no_match"] for r in recs])), 3)}


def by(recs, key):
    groups = {}
    for r in recs:
        groups.setdefault(key(r), []).append(r)
    return {k: summary(v) for k, v in sorted(groups.items())}


def reliability(recs):
    """Stated device probability against observed accuracy, in probability bins."""
    p = np.array([r["p"] for r in recs])
    ok = np.array([r["dev"] == r["true"] for r in recs])
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = (p >= lo) & (p < hi)
        if sel.any():
            out.append({"bin": [lo, round(min(hi, 1.0), 2)], "n": int(sel.sum()),
                        "stated_pct": round(100 * float(p[sel].mean()), 1), "observed_pct": round(100 * float(ok[sel].mean()), 1)})
    return out


def shares(recs, level=1):
    c = Counter("/".join(r["dev"].split("/")[:level]) for r in recs)
    return {k: round(100 * v / len(recs), 2) for k, v in c.most_common()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=str(HERE / "lookup_v2"))
    ap.add_argument("--graphene-store", default="~/atlas_store/engine_v1")
    ap.add_argument("--materials-store", default="~/atlas_store/materials_ev_full")
    ap.add_argument("--pilot-store", default="~/atlas_store/conc_v1")
    ap.add_argument("--novelty-store", default="~/atlas_store/novelty_v1")
    ap.add_argument("--today", default=str(HERE / "atlas_v4m" / "results.json"))
    ap.add_argument("--routing", default=str(HERE / "conc_v1" / "routing_by_density.json"))
    ap.add_argument("--smoke", action="store_true", help="2-3 test seeds per cloud: a quick end-to-end check")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda s: print(f"[eval-lookup] {s} ({time.time() - t0:.0f}s)", flush=True)
    cat = Catalogue.load(a.catalogue)
    store = MultiStore(os.path.expanduser(a.graphene_store), os.path.expanduser(a.materials_store))
    pilot = CloudStore(os.path.expanduser(a.pilot_store))
    n1, n2, n3a, n3b, n4 = (3, 3, 2, 2, 3) if a.smoke else (150, 150, 20, 30, 150)
    res = {"settings": {"catalogue": a.catalogue, "kappa": cat.kappa, "kappa_material": cat.kappa_material,
                        "smoke": a.smoke, "test_seeds": "850-999",
                        "per_cloud": {"t1": n1, "t2": n2, "t3_catalogue": n3a, "t3_pilot": n3b, "t4": n4}}}

    t1 = [(mid, d, *signatures(cat, store, mid, d, 850, n1)) for mid in cat.ids for d in cat.stored[mid]]
    r1 = run(cat, t1)
    res["t1"] = {"all": summary(r1), "per_material": by(r1, lambda r: r["true"].split("/")[0])}
    log(f"T1 {res['t1']['all']}")

    t2 = [(mid, d, *signatures(cat, pilot, mid, d, 850, n2)) for mid in pilot.models() for d in pilot.densities(mid) if d <= TOP]
    r2 = run(cat, t2)
    res["t2"] = {"all": summary(r2), "per_ribbon": by(r2, lambda r: r["true"]),
                 "per_density": {mid: by([r for r in r2 if r["true"] == mid], lambda r: f"{r['c']:.4f}") for mid in pilot.models()}}
    log(f"T2 {res['t2']['all']}")

    res["t3"] = {}
    for w in WINDOWS:
        ra = run(cat, [(m, c, e, rows[:n3a]) for m, c, e, rows in t1], window=w)
        rb = run(cat, [(m, c, e, rows[:n3b]) for m, c, e, rows in t2], window=w)
        res["t3"][f"{w[0]}-{w[1]} eV"] = {"catalogue": summary(ra), "pilot": summary(rb), "reliability": reliability(ra + rb)}
        log(f"T3 {w}: catalogue {summary(ra)['device_pct']}%, pilot {summary(rb)['device_pct']}%")

    res["t4"] = {}
    for M in sorted({m.material for m in cat.models} - {"graphene-ideal"}):
        items = [(mid, d, *signatures(cat, store, mid, d, 850, n4)) for mid in cat.ids if mid.split("/")[0] == M
                 for d in cat.stored[mid]]
        r4 = run(cat.without([M]), items)
        res["t4"][M] = {"n": len(r4), "no_match_pct": summary(r4)["no_match_pct"], "nearest_material": shares(r4),
                        "nearest_device_top3": Counter(r["dev"] for r in r4).most_common(3)}
        log(f"T4 hidden {M}: no match {res['t4'][M]['no_match_pct']}%")
    nov, sq = CloudStore(os.path.expanduser(a.novelty_store)), "square/strip/N10"
    rs = run(cat, [(sq, d, *signatures(cat, nov, sq, d, 0, n4)) for d in nov.densities(sq)])
    res["t4"]["square"] = {"n": len(rs), "no_match_pct": summary(rs)["no_match_pct"], "nearest_material": shares(rs),
                           "nearest_device_top3": Counter(r["dev"] for r in rs).most_common(3)}
    log(f"T4 square: no match {res['t4']['square']['no_match_pct']}%")

    t5 = [(mid, d, *signatures(cat, pilot, mid, d, 850, n2)) for mid in pilot.models() for d in pilot.densities(mid) if d > TOP]
    r5 = run(cat, t5)
    res["t5"] = {mid: dict(summary(v := [r for r in r5 if r["true"] == mid]),
                           median_concentration_pct=round(100 * float(np.median([r["conc"] for r in v])), 3))
                 for mid in pilot.models()}
    log("T5 " + json.dumps({k: v["silent_wrong_pct"] for k, v in res["t5"].items()}))

    e, rows = signatures(cat, store, cat.ids[0], cat.stored[cat.ids[0]][1], 850, 10 if a.smoke else 100)
    ts = []
    for T in rows:
        t = time.perf_counter()
        lookup(cat, e, T)
        ts.append(1000 * (time.perf_counter() - t))
    res["speed_ms"] = {"median": round(float(np.median(ts)), 1), "p90": round(float(np.percentile(ts, 90)), 1), "n": len(ts)}

    today = json.loads(Path(a.today).read_text())
    routing = json.loads(Path(a.routing).read_text())
    res["today"] = {
        "source": {"t1_t4": a.today, "t2_t5": a.routing},
        "t1_per_material": {k: {"width_pct": v["width_accuracy"], "unknown_pct": v["unknown_pct"]}
                            for k, v in today["identification"]["per_material"].items()},
        "t2_routed_pct": {mid: {k: v["routed_pct"] for k, v in dd.items() if float(k) <= TOP} for mid, dd in routing.items()},
        "t4_unknown_pct": dict({k: v["unknown_pct"] for k, v in today["lomo"].items()}, square=today["square"]["unknown_pct"]),
        "t5_silent_misread_pct": {mid: {k: v["silently_misread_pct"] for k, v in dd.items() if float(k) > TOP}
                                  for mid, dd in routing.items()}}
    out = Path(a.catalogue) / ("results_smoke.json" if a.smoke else "results.json")
    out.write_text(json.dumps(res, indent=2))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
