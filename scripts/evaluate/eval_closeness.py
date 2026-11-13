"""Spectrum-closeness scoreboard: how close are the hosts a method proposes to the target's transmission?

    TRANSMISSION_DATA=... PYTHONPATH=src nice -n 10 python scripts/evaluate/eval_closeness.py \\
        results/inverse_design/model_v2_depth_bandtop_600k [more model folders] [--ensemble] --jobs 18

The benchmark is --n seeded rows of the val split (data.seeded_rows, seed BENCHMARK_SEED); with --targets known, of the
val rows whose host is near a known material (the targets the realistic plan cares about; default --out
closeness_known). For each target every method
proposes up to --k hosts, each host is simulated ONCE with a fresh disorder seed, and its spectrum is compared with the
target's in units of seed-to-seed noise (shazam.inverse.closeness: smoothed v4 misfit / m0 of the target's width and
density). The simulations score the methods; no method sees them. Methods:

* truth      the target's own host with a fresh impurity arrangement: the floor no method can beat on average. m0 is
             one arrangement against the median of nine others, so two single arrangements of one host sit above 1.
* lookup-known  the same lookup over the train hosts near a known material only (shazam.inverse.realism): how close
             a realistic host can get.
* lookup     the hosts of the --k train spectra nearest the target (no model). Its `stored` block is the same ratio
             taken straight from those stored spectra, without a new simulation: the nearest spectrum was picked BY this
             misfit, so `stored` flatters it, and the freshly simulated numbers are the fair bar a model has to beat.
* lookup-extra  with --lookup-extra ROOT: the lookup over the train rows of --data and of those roots together (the
             bar for a model trained with that data added).
* <folder>   each model: the top --k candidates of candidates() on --samples posterior draws (--candidates medoids, the
             list design() reports), or the first --k posterior draws (--candidates samples).
* ensemble   with --ensemble: the average of all models given.

Every record keeps the host that was scored, and each method's summary has a `near_known` block: the share of its
hosts near a known material (--rel-tol, --abs-tol) and the closeness of those hosts alone.

Disorder seeds are SEED_BASE + MAX_K * (target position) + rank: the same for every method, and disjoint from the data's
seeds (draw indices), VERIFY / FINAL / OPT (1e9, 2e9), m0 (3e9), the phase-2 pilot (3.5e9) and phase 2 (4e9).
Writes closeness.{json,txt} and one resumable <method>.jsonl per method into --out; model folders are only read.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import paths
from shazam.arrangement import misfit
from shazam.inverse import closeness as cl
from shazam.inverse import data, realism as rl, reward as rw, sim_reward as sr
from shazam.inverse.candidates import candidates
from shazam.inverse.design import model_input
from shazam.inverse.encoding import decode, encode
from shazam.inverse.model import EnsembleInverseModel, InverseModel
from shazam.inverse.runio import RunLog, atomic_write_json, code_version, sha256_of
from tbribbon.parent import TYPES

DEFAULT_DATA = paths.data_root() / "_generated" / "hexagonal_parent_v1"
DEFAULT_M0 = paths.results_dir("inverse_design", "model_v4_simreward") / "m0.json"
DEFAULT_OUT = paths.results_dir("inverse_design", "closeness")
NEEDED = ("index", "x", "on", "lead", "params", "edge", "width", "density")
BENCHMARK_SEED = 5            # data.seeded_rows seed of the benchmark rows (0: model selection, 2: recreation, 1 and 7: train)
SEED_BASE = 3_800_000_000
MAX_K = 16
MODES = ("medoids", "samples")


def candidate_seeds(position, k):
    """The disorder seeds of the k candidates of the target at `position` in the benchmark (see the module docstring)."""
    if not 1 <= k <= MAX_K:
        raise ValueError(f"k must be between 1 and {MAX_K}, got {k}")
    return [SEED_BASE + MAX_K * int(position) + j for j in range(k)]


def truth_proposer(targets):
    """propose(i) -> the target's own host, alone: scored with a fresh disorder seed it is the scoreboard's floor."""
    truths = data.draws(targets, slice(None))
    return lambda i: [dict(draw=truths[i])]


def host_record(draw):
    """The scored host as JSON: site-type pattern, edge, density and the physical quantities in eV (realism.physical_ev),
    enough to judge it against the known materials at any tolerance later."""
    return dict(pattern=rw.pattern_name(draw.params.pattern), edge=draw.edge, density=float(draw.density),
                ev=[float(v) for v in rl.physical_ev(draw)])


def hosts_nearest_material(hosts, rel_tol=rl.REL_TOL, abs_tol=rl.ABS_TOL):
    """(names, distances) of the nearest known material of each host_record."""
    return rl.nearest_material_ev([h["ev"] for h in hosts], [[t in h["pattern"] for t in TYPES] for h in hosts],
                                  rel_tol=rel_tol, abs_tol=abs_tol)


def lookup_proposer(train, targets, k, known_only=False, rel_tol=rl.REL_TOL, abs_tol=rl.ABS_TOL):
    """propose(i) -> the stored hosts of the k train spectra nearest target i, nearest first: dicts with `draw`,
    `source_index` (the train draw index) and `stored_misfit` (smoothed misfit of the stored spectrum to the target).
    known_only: search only the train hosts near a known material (realism.near_known_rows)."""
    if known_only:
        keep = rl.near_known_rows(train, rel_tol=rel_tol, abs_tol=abs_tol)
        train = {key: v[keep] for key, v in train.items()}
    rows, dist = cl.nearest_spectra(train["x"], targets["x"], k)

    def propose(i):
        return [dict(draw=d, source_index=int(train["index"][r]), stored_misfit=float(m))
                for d, r, m in zip(data.draws(train, rows[i]), rows[i], dist[i])]
    return propose


def model_proposer(model, targets, k, n_samples=1000, mode="medoids"):
    """propose(i) -> up to k hosts of `model` for target i, seeded by the target's draw index.

    medoids: the top k candidates of candidates() on n_samples posterior draws, most probable first, with `probability`
    (fewer than k when the samples form fewer clusters). samples: the first k posterior draws themselves."""
    if mode not in MODES:
        raise ValueError(f"unknown candidates mode {mode!r}; use one of {MODES}")
    kind = model.cfg.get("input", "log")

    def propose(i):
        x = model_input(targets["x"][i], kind)[None, :]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(targets["index"][i]))
            states, z = model.sample(x, n_samples)
        states, z = np.asarray(states), np.asarray(z)
        if mode == "samples":
            return [dict(draw=decode(int(s), v)) for s, v in zip(states[:k], z[:k])]
        return [dict(draw=c.draw, probability=float(c.probability)) for c in candidates(states, z, top=k)[0]]
    return propose


def _family(draw):
    return int(rw.FAMILY_OF[encode(draw)[0]])


def score(propose, targets, m0, log, simulate, n_jobs=1, batch=16, tag="closeness"):
    """One record per target, in benchmark order: each proposed host simulated once and scored against the target.

    propose(i) gives target i's candidates (dicts with `draw` and anything else worth keeping). Targets already in `log`
    (a RunLog keyed by draw index) are not proposed or simulated again. `simulate(draws, seeds, n_jobs)` is
    sim_reward.simulate_candidates: one v4 input per draw, None for a failed simulation, which leaves ratio None.
    A record: index, pattern, m0 (the target's), candidates [{ratio, misfit, family_right, width, ...the proposer's
    extras, stored_ratio when it gave stored_misfit, host (host_record)}]."""
    truths = data.draws(targets, slice(None))
    scale = cl.noise_scale(m0, targets["width"], targets["density"])
    todo = [i for i in range(len(truths)) if int(targets["index"][i]) not in log.done]
    t0 = time.time()
    for b in range(0, len(todo), batch):
        rows = todo[b:b + batch]
        cands = [propose(i) for i in rows]
        k = max((len(c) for c in cands), default=0)
        seeds = [s for i, c in zip(rows, cands) for s in candidate_seeds(i, k)[:len(c)]] if k else []
        sims = iter(simulate([c["draw"] for cs in cands for c in cs], seeds, n_jobs) if seeds else [])
        for i, cs in zip(rows, cands):
            target = sr.smooth(targets["x"][i])
            out = []
            for c in cs:
                sx = next(sims)
                d = None if sx is None else misfit(sr.smooth(sx), target)
                rec = dict(ratio=None if d is None else float(d / scale[i]), misfit=d,
                           family_right=_family(c["draw"]) == _family(truths[i]), width=int(c["draw"].width))
                rec.update({key: v for key, v in c.items() if key != "draw"})
                rec["host"] = host_record(c["draw"])
                if "stored_misfit" in c:
                    rec["stored_ratio"] = float(c["stored_misfit"] / scale[i])
                out.append(rec)
            log.add(int(targets["index"][i]), dict(index=int(targets["index"][i]),
                                                   pattern=rw.pattern_name(truths[i].params.pattern),
                                                   m0=float(scale[i]), candidates=out))
        print(f"[{tag}] {min(b + batch, len(todo))}/{len(todo)} targets ({time.time() - t0:.0f} s)", flush=True)
    return [log.done[int(i)] for i in targets["index"]]


def _matrix(records, k, key):
    out = np.full((len(records), k), np.nan)
    for i, r in enumerate(records):
        for j, c in enumerate(r["candidates"][:k]):
            if c.get(key) is not None:
                out[i, j] = c[key]
    return out


def summary(records, k, rel_tol=rl.REL_TOL, abs_tol=rl.ABS_TOL):
    """closeness.summarise of a method's records, by site-type pattern; a lookup's records (they carry stored_ratio) get
    the same summary of the stored spectra under `stored`. Records that keep their hosts also get `near_known`: the
    share of scored hosts near a known material (share_of_hosts), the share of targets with at least one
    (targets_with_one), and the summary of those hosts alone (a target without one is a miss)."""
    ratio = _matrix(records, k, "ratio")
    right = _matrix(records, k, "family_right") == 1
    groups = [r["pattern"] for r in records]
    out = cl.summarise(ratio, right, groups)
    if any("stored_ratio" in c for r in records for c in r["candidates"]):
        out["stored"] = cl.summarise(_matrix(records, k, "stored_ratio"), right, groups)
    where = [(i, j) for i, r in enumerate(records) for j, c in enumerate(r["candidates"][:k]) if "host" in c]
    if where:
        dist = hosts_nearest_material([records[i]["candidates"][j]["host"] for i, j in where], rel_tol, abs_tol)[1]
        near = np.zeros(ratio.shape, bool)
        near[tuple(np.array(where).T)] = dist <= 1.0
        near &= np.isfinite(ratio)
        out["near_known"] = dict(share_of_hosts=float(near.sum() / max(np.isfinite(ratio).sum(), 1)),
                                 targets_with_one=float(near.any(1).mean()),
                                 **cl.summarise(np.where(near, ratio, np.nan), right, groups))
    return out


def _fmt(v, spec="7.2f"):
    return f"{v:{spec}}" if v is not None else f"{'-':>{int(spec.split('.')[0])}}"


def _table(methods):
    rows = {}
    for name, s in methods.items():                          # the lookup's stored-spectrum block as a row of its own
        if "stored" in s:
            rows[f"{name} (stored)"] = s["stored"]
        rows[name] = s
    lines = [f"{'method':40s} {'cands':>5s} {'first':>7s} {'best':>7s} {'all':>7s} {'<=1x':>6s} {'<=2x':>6s} "
             f"{'fam@1':>6s} {'fam@k':>6s} {'famBest':>7s}"]
    for name, s in rows.items():
        lines.append(f"{name:40s} {s['candidates_per_target']:5.2f} {_fmt(s['first']['median'])} "
                     f"{_fmt(s['best']['median'])} {_fmt(s['all']['median'])} {s['best_within_1']:6.3f} "
                     f"{s['best_within_2']:6.3f} {s['first_right_family']:6.3f} {s['any_right_family']:6.3f} "
                     f"{s['closest_is_right_family']:7.3f}")
    lines += ["", "first / best / all: median misfit of the rank-1, the closest and every candidate, in units of the "
              "target's seed noise m0", "<=1x, <=2x: share of targets whose closest candidate is within that many m0; "
              "fam@1 / fam@k / famBest: the rank-1, any, the closest candidate has the true family", ""]
    pats = [p for p in rw.PATTERN_NAMES if any(p in s["by_group"] for s in rows.values())]
    lines.append(f"{'closest candidate, median by pattern':40s} " + " ".join(f"{p:>7s}" for p in pats))
    for name, s in rows.items():
        lines.append(f"{name:40s} " + " ".join(_fmt(s["by_group"].get(p, {}).get("best_median")) for p in pats))
    known = {name: s["near_known"] for name, s in rows.items() if "near_known" in s}
    if known:
        lines += ["", f"{'hosts near a known material':40s} {'hosts':>6s} {'targets':>7s} {'best':>7s} {'<=2x':>6s}"]
        for name, s in known.items():
            lines.append(f"{name:40s} {s['share_of_hosts']:6.3f} {s['targets_with_one']:7.3f} "
                         f"{_fmt(s['best']['median'])} {s['best_within_2']:6.3f}")
        lines.append("hosts: share of scored hosts near a known material; targets: share of targets with one; best: its "
                     "median misfit (targets that have one); <=2x: share of ALL targets with one within 2 m0")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("models", nargs="*", type=Path, help="model folders (each with model.pt); none: the lookup alone")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--m0", default=str(DEFAULT_M0), help="m0.json of scripts/build/estimate_m0.py")
    ap.add_argument("--n", type=int, default=300, help="benchmark targets (seeded val rows)")
    ap.add_argument("--targets", choices=("val", "known"), default="val",
                    help="val: any val row; known: only val rows whose host is near a known material")
    ap.add_argument("--k", type=int, default=5, help="hosts proposed per target")
    ap.add_argument("--samples", type=int, default=1000, help="posterior draws per target behind a model's candidates")
    ap.add_argument("--candidates", choices=MODES, default="medoids")
    ap.add_argument("--ensemble", action="store_true", help="also score the ensemble of all models given")
    ap.add_argument("--lookup-extra", action="append", default=[], type=Path,
                    help="a data root whose train rows the lookup may also return (repeatable): adds the method "
                         "lookup-extra, the lookup over --data and these together")
    ap.add_argument("--rel-tol", type=float, default=rl.REL_TOL, help="near a known material: share of its own value")
    ap.add_argument("--abs-tol", type=float, default=rl.ABS_TOL, help="... and eV for quantities it does not have")
    ap.add_argument("--jobs", type=int, default=18, help="simulation workers")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None, help="default: results/inverse_design/closeness, or "
                                                            "closeness_known with --targets known")
    a = ap.parse_args(argv)
    code_version()
    candidate_seeds(0, a.k)                                   # k within range, before anything is loaded
    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.time()
    m0 = sr.M0Model.from_json(json.loads(Path(a.m0).read_text())["model"])
    if a.out is None:
        a.out = DEFAULT_OUT if a.targets == "val" else DEFAULT_OUT.with_name(DEFAULT_OUT.name + "_known")
    va = data.load_split(a.data, "val", NEEDED)
    if a.targets == "known":
        keep = rl.near_known_rows(va, rel_tol=a.rel_tol, abs_tol=a.abs_tol)
        va = {key: v[keep] for key, v in va.items()}
    rows = data.seeded_rows(len(va["index"]), a.n, seed=BENCHMARK_SEED)
    targets = {key: v[rows] for key, v in va.items()}
    settings = dict(n=len(rows), k=a.k, benchmark_seed=BENCHMARK_SEED, m0_sha256=sha256_of(a.m0),
                    smooth_channels=sr.SMOOTH_CHANNELS)
    if a.targets != "val":                                    # the default benchmark's logs keep their settings
        settings.update(targets=a.targets, rel_tol=a.rel_tol, abs_tol=a.abs_tol)
    a.out.mkdir(parents=True, exist_ok=True)
    methods, meta = {}, {}

    def run(name, propose, cfg):
        with RunLog(a.out / f"{name}.jsonl", {**settings, **cfg}, key_field="index", tag=name) as log:
            records = score(propose, targets, m0, log, sr.simulate_candidates, a.jobs, tag=name)
        methods[name] = summary(records, a.k, a.rel_tol, a.abs_tol)
        print(f"[closeness] {name} done ({time.time() - t0:.0f} s)", flush=True)

    run("truth", truth_proposer(targets), dict(method="truth"))
    train = data.load_split(a.data, "train", NEEDED)
    run("lookup", lookup_proposer(train, targets, a.k), dict(method="lookup", n_train=len(train["index"])))
    run("lookup-known", lookup_proposer(train, targets, a.k, True, a.rel_tol, a.abs_tol),
        dict(method="lookup-known", n_train=len(train["index"]), materials=list(rl.NAMED), rel_tol=a.rel_tol,
             abs_tol=a.abs_tol))
    if a.lookup_extra:
        more = [data.load_split(r, "train", NEEDED) for r in a.lookup_extra]
        train = {key: np.concatenate([train[key]] + [e[key] for e in more]) for key in train}
        run("lookup-extra", lookup_proposer(train, targets, a.k),
            dict(method="lookup-extra", n_train=len(train["index"]), roots=[str(r) for r in a.lookup_extra]))
        del more
    del train
    model_cfg = dict(samples=a.samples, candidates=a.candidates)
    loaded = {}
    for folder in a.models:
        loaded[folder.name] = InverseModel.load(folder / "model.pt")
        meta[folder.name] = dict(path=str(folder), model_sha256=sha256_of(folder / "model.pt"))
        run(folder.name, model_proposer(loaded[folder.name], targets, a.k, a.samples, a.candidates),
            dict(method="model", model_sha256=meta[folder.name]["model_sha256"], **model_cfg))
    if a.ensemble and len(loaded) > 1:
        meta["ensemble"] = dict(models=[meta[n]["path"] for n in loaded],
                                model_sha256=[meta[n]["model_sha256"] for n in loaded])
        run("ensemble", model_proposer(EnsembleInverseModel(list(loaded.values())), targets, a.k, a.samples,
                                       a.candidates),
            dict(method="ensemble", model_sha256=meta["ensemble"]["model_sha256"], **model_cfg))
    text = _table(methods)
    print(text, flush=True)
    atomic_write_json(a.out / "closeness.json", dict(
        methods=methods, models=meta,
        settings={**settings, **model_cfg, **({"lookup_extra": [str(r) for r in a.lookup_extra]} if a.lookup_extra else {})}, targets=[int(i) for i in targets["index"]],
        realism=dict(materials=list(rl.NAMED), rel_tol=a.rel_tol, abs_tol=a.abs_tol),
        m0=str(a.m0), code_version=code_version(), data_provenance=data.data_provenance(a.data)))
    (a.out / "closeness.txt").write_text(text + "\n")
    print(f"[closeness] wrote {a.out}/closeness.{{txt,json}} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
