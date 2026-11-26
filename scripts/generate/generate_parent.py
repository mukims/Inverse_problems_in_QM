#!/usr/bin/env python
"""Parallel, resumable generation of the hexagonal-parent training set (spec section 5).

One draw = one host from the prior of section 4.2 + one impurity count + one disorder seed (the draw index) -> one
disordered spectrum on the v4 grid of that host's scale. Draws go into shards of --shard-size; a shard is written to
shard_NNNNN.npz.tmp and renamed when complete, and the manifest lists the finished shards, so an interrupted run is
resumed by running the same command again.

    PYTHONPATH=src python scripts/generate/generate_parent.py --n-draws 2000 --n-jobs 22

Active learning (shazam.inverse.active): `--proposal cells.json` draws hosts from the tilted prior whose per-cell
acceptance table is the file's "accept" list (each row then also stores its `cell`), and `--index-start` puts the draw
indices (and so the disorder seeds and the index % 20 split) in a range disjoint from every other data set.

Near known materials (shazam.inverse.realism): `--near hbn=12000 graphene-ideal=10000 ...` draws hosts from the prior
restricted to within --near-radius tolerances of each material, one block of draw indices per material in the order
given. Each row then also stores `near` (the material's position in tbribbon.parent.NAMED) and `tries` (prior draws
used, over all attempts of that index); the manifest stores the plan, and a rerun must give the same one.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")             # one BLAS thread per worker; must be set before numpy is imported

import argparse
import json
import multiprocessing as mp
import queue
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import paths
from shazam.inverse.data import EDGE_CODES, LEAD_CODES, PARAM_COLUMNS, load_manifest
from shazam.inverse.runio import atomic_write_bytes, code_version
from shazam.inverse import realism
from tbribbon import parent_data
from tbribbon.parent import NAMED, active_atoms_per_cell
from tbribbon.parent_prior import RANGES, sample_prior

PROPOSAL = None             # per-cell acceptance table of the active-learning proposal (None: the plain prior)
NEAR = None                 # near-material plan (near_spec) plus its index_start; write_shard sets it for its workers

STALL_SECONDS = 1800        # no result for this long with workers alive: stop (a worker may have died holding the queue lock)
MAX_ATTEMPTS = 100          # redraws allowed per index; the prior rejects a small fraction, so reaching this is a bug
REASONS = ("invisible_type", "no_channel", "few_channels")      # in the order they are checked (cheapest first)
N_CHANNELS = 416
DEFAULT_OUT = paths.data_root() / "_generated" / "hexagonal_parent_v1"
# Fixed once, when the process starts: a commit made while a multi-day run is going must not relabel the shards it writes
# afterwards. Stored per shard (manifest entry and the shard file's 0-d `code_version`) and as the manifest's top-level field.
CODE_VERSION = code_version()


def near_spec(counts, radius=2.0, rel_tol=realism.REL_TOL, abs_tol=realism.ABS_TOL):
    """The plan of a near-material data set, as stored in its manifest: `counts` is [(material, draws), ...]; the
    materials take consecutive blocks of draw indices in this order."""
    names = [n for n, _ in counts]
    for n in names:
        if n not in NAMED:
            raise ValueError(f"unknown material {n!r}; choose from {NAMED}")
        if names.count(n) > 1:
            raise ValueError(f"material {n!r} is listed twice")
    if any(int(c) != c or c < 1 for _, c in counts):
        raise ValueError(f"every count must be a positive whole number, got {[c for _, c in counts]}")
    if not radius > 0:
        raise ValueError(f"radius must be positive, got {radius}")
    return dict(materials=names, counts=[int(c) for _, c in counts], radius=float(radius), rel_tol=float(rel_tol),
                abs_tol=float(abs_tol))


def near_material(spec, offset):
    """The material of the draw `offset` places after the plan's first index."""
    stops = np.cumsum(spec["counts"])
    if not 0 <= offset < stops[-1]:
        raise ValueError(f"draw {offset} is outside the plan's {int(stops[-1])} draws")
    return spec["materials"][int(np.searchsorted(stops, offset, side="right"))]


def _host(rng, index):
    """(draw, prior draws used, material or None) of one attempt, from the plain prior, the proposal or the plan."""
    if NEAR is not None:
        name = near_material(NEAR, index - NEAR["index_start"])
        draw, tries = realism.sample_near(rng, name, NEAR["radius"], NEAR["rel_tol"], NEAR["abs_tol"],
                                          sampler=sample_prior)
        return draw, tries, name
    if PROPOSAL is None:
        return sample_prior(rng), 1, None
    from shazam.inverse import active
    return active.sample_proposal(rng, PROPOSAL, sampler=sample_prior), 1, None


def draw_with_resampling(index):
    """Accepted draw for this index: (draw, n_imp, raw, clean, attempts, reasons).

    Attempt j draws from default_rng([index, j]); the disorder seed is the index. Checks run cheapest first: band
    weights (ms), then the clean open channels (seconds), then the disordered transport (up to a minute). If both the
    band-weight and the channel rule would reject a draw it is counted as invisible_type. `attempts` counts the
    attempts up to and including the accepted one; `reasons` counts the rejections by reason for this index.
    """
    return _draw(index)[:6]


def _draw(index):
    """draw_with_resampling plus (prior draws used over all attempts, material of a near-material plan or None)."""
    reasons = {r: 0 for r in REASONS}
    probe = np.ones(N_CHANNELS)          # stands in for a spectrum not yet computed, so the cheap rules can run first
    tries = 0
    for j in range(MAX_ATTEMPTS):
        rng = np.random.default_rng([index, j])
        draw, used, material = _host(rng, index)
        tries += used
        weights = parent_data.band_weights(draw.params)
        why = parent_data.reject_reason(draw, np.ones(1, int), probe, weights)      # only the weight rule can fire
        if why is None:
            clean = parent_data.clean_channels(draw)
            why = parent_data.reject_reason(draw, clean, probe, weights)            # only the channel rule can fire
        if why is None:
            n_cells = parent_data.device_cells_for(draw.params, draw.edge)
            n_imp = parent_data.n_impurities(draw, n_cells)
            raw = parent_data.simulate_raw(draw, n_imp, index, n_cells=n_cells)
            why = parent_data.reject_reason(draw, clean, raw, weights)
            if why is None:
                return draw, n_imp, raw, clean, j + 1, reasons, tries, material
        reasons[why] += 1
    raise RuntimeError(f"draw index {index}: no acceptable host in {MAX_ATTEMPTS} attempts (rejections {reasons})")


def _record(index):
    """Everything stored for one index. `seconds` is the wall time of the whole attempt sequence up to acceptance."""
    t0 = time.perf_counter()
    draw, n_imp, raw, clean, attempts, reasons, tries, material = _draw(index)
    n_cells = parent_data.device_cells_for(draw.params, draw.edge)
    p = draw.params
    return dict(
        index=index, x=parent_data.to_input(raw), raw=raw, clean=clean,
        params=[getattr(p, c) for c in PARAM_COLUMNS], on=[p.on_C, p.on_H, p.on_K], lead=LEAD_CODES[p.lead],
        edge=EDGE_CODES[draw.edge], width=draw.width, density=draw.density, n_imp=n_imp, n_cells=n_cells,
        active_atoms=n_cells * active_atoms_per_cell(p, draw.edge, draw.width),
        seconds=time.perf_counter() - t0, attempts=attempts, reasons=reasons,
        **({} if PROPOSAL is None else dict(cell=_cell(draw))),
        **({} if material is None else dict(near=NAMED.index(material), tries=tries)))


def _cell(draw):
    from shazam.inverse import active
    return active.cell_of(draw)


def _limit_blas():
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1)      # the env vars above only help if numpy was not imported yet
    except ImportError:
        return None


def _worker(tasks, results):
    keep = _limit_blas()
    pid = os.getpid()
    while True:
        index = tasks.get()
        if index is None:
            return
        results.put(("start", pid, index))
        try:
            results.put(("ok", _record(index), pid))
        except BaseException:
            results.put(("error", index, traceback.format_exc()))
            return


def _compute(indices, n_jobs, label="shard"):
    """The records of all indices, in index order. Workers are forked, so they see the parent's (patched) modules.

    Fails fast: a draw that raises, a worker that exits with a nonzero code (killed, out of memory) or no result for
    STALL_SECONDS while workers are alive (a worker may have died holding a queue lock) stops the run with an error
    that names `label` and the draw index in flight.
    """
    indices = list(indices)
    if n_jobs <= 1:
        _limit_blas()
        return [_record(i) for i in indices]
    ctx = mp.get_context("fork")
    tasks, results = ctx.Queue(), ctx.Queue()
    n = min(n_jobs, len(indices))
    procs = [ctx.Process(target=_worker, args=(tasks, results), daemon=True) for _ in range(n)]
    for p in procs:                                  # fork before the first put: the parent has no feeder thread yet
        p.start()
    for i in indices:
        tasks.put(i)
    for _ in range(n):
        tasks.put(None)
    got, running, t0, last = [], {}, time.time(), time.time()
    try:
        while len(got) < len(indices):
            bad = [p for p in procs if p.exitcode not in (None, 0)]        # every iteration: cheap, and a busy
            if bad:                                                        # result stream must not hide a dead worker
                raise RuntimeError(f"{label}: a worker exited with code {bad[0].exitcode} while computing draw index "
                                   f"{running.get(bad[0].pid, 'unknown')}")
            try:
                msg = results.get(timeout=1)
            except queue.Empty:
                if not any(p.is_alive() for p in procs):
                    raise RuntimeError(f"{label}: all workers exited before finishing their draws") from None
                if time.time() - last > STALL_SECONDS:
                    raise RuntimeError(f"{label}: no result for {STALL_SECONDS} s with workers alive; draws in flight: "
                                       f"{sorted(running.values())}") from None
                continue
            last = time.time()
            if msg[0] == "start":
                running[msg[1]] = msg[2]
            elif msg[0] == "error":
                raise RuntimeError(f"{label}: draw index {msg[1]} failed in a worker:\n{msg[2]}")
            else:
                got.append(msg[1])
                running.pop(msg[2], None)
                if len(got) % 100 == 0:
                    print(f"  {len(got)}/{len(indices)} draws, {time.time() - t0:.0f} s", flush=True)
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join()
        for q in (tasks, results):              # undelivered tasks would keep a feeder thread blocked on a full pipe
            q.cancel_join_thread()              # and hang the interpreter at exit
            q.close()
    return sorted(got, key=lambda r: r["index"])


def _update_manifest(root, entry, shard_size, index_start=0, proposal=None, near=None):
    """Add or replace the entry of one finished shard; counts are recomputed from the entries."""
    m = load_manifest(root) or {}
    shards = {e["shard"]: e for e in m.get("shards", [])}
    shards[entry["shard"]] = entry
    shards = [shards[k] for k in sorted(shards)]
    rej = {r: sum(e["rejections"][r] for e in shards) for r in REASONS}
    manifest = dict(
        prior_ranges=RANGES, code_version=CODE_VERSION, shard_size=shard_size, shards=shards,
        counts=dict(n_draws=sum(e["n_draws"] for e in shards), n_attempts=sum(e["n_attempts"] for e in shards),
                    rejections=rej, seconds=sum(e["seconds"] for e in shards)))
    if index_start or proposal is not None:            # active-learning data sets only: v1 manifests are unchanged
        manifest.update(index_start=index_start, proposal=proposal)
    if near is not None:
        manifest.update(index_start=index_start, proposal=proposal, near=near)
    atomic_write_bytes(Path(root) / "manifest.json", lambda f: f.write(json.dumps(manifest, indent=2).encode()))


def write_shard(root, shard, indices, n_jobs, shard_size=None, index_start=0, proposal=None, near=None):
    """Compute the draws of `indices` and write shard_{shard:05d}.npz, its rejection counts and the manifest entry.
    `near` (near_spec) draws the hosts from that near-material plan, which starts at `index_start`."""
    global NEAR
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    indices = list(indices)
    if not indices:
        raise ValueError(f"shard {shard}: no draw indices to compute")
    old = load_manifest(root)
    if old:                                          # an existing data set keeps its shard size
        if shard_size not in (None, old["shard_size"]):
            raise ValueError(f"{root} was started with shard size {old['shard_size']}, not {shard_size}")
        shard_size = old["shard_size"]
        if old.get("index_start", 0) != index_start:
            raise ValueError(f"{root} was started at index {old.get('index_start', 0)}, not {index_start}")
        if old.get("near") != near:
            raise ValueError(f"{root} was started with the near-material plan {old.get('near')}, not {near}")
        first = index_start + shard * shard_size
        if len(indices) > shard_size or indices[0] != first:     # a short final shard is allowed
            raise ValueError(f"shard {shard} of {root} must start at {first} and hold at most "
                             f"{shard_size} draws, got {indices[0]}..{indices[-1]}")
    NEAR = None if near is None else dict(near, index_start=index_start)       # read by the forked workers
    try:
        recs = _compute(indices, n_jobs, label=f"shard {shard}")
    finally:
        NEAR = None
    col = lambda key, dtype: np.array([r[key] for r in recs], dtype=dtype)
    arrays = dict(
        index=col("index", np.int64), x=col("x", np.float32), raw=col("raw", np.float32), clean=col("clean", np.int16),
        params=col("params", np.float64), on=col("on", np.int8), lead=col("lead", np.int8), edge=col("edge", np.int8),
        width=col("width", np.int16), density=col("density", np.float64), n_imp=col("n_imp", np.int32),
        n_cells=col("n_cells", np.int32), active_atoms=col("active_atoms", np.int32),
        seconds=col("seconds", np.float32), attempts=col("attempts", np.int16),
        code_version=np.array(CODE_VERSION))                    # 0-d: per shard, not a column (data.SHARD_META)
    if "cell" in recs[0]:
        arrays["cell"] = col("cell", np.int32)
    if "near" in recs[0]:
        arrays.update(near=col("near", np.int8), tries=col("tries", np.int32))
    rej = {r: sum(rec["reasons"][r] for rec in recs) for r in REASONS}
    name = f"shard_{shard:05d}"
    summary = dict(n_draws=len(recs), n_attempts=int(arrays["attempts"].sum()), rejections=rej)
    atomic_write_bytes(root / f"{name}_rejections.json", lambda f: f.write(json.dumps(summary, indent=2).encode()))
    atomic_write_bytes(root / f"{name}.npz", lambda f: np.savez(f, **arrays))
    entry = dict(shard=shard, file=f"{name}.npz", start=indices[0], stop=indices[-1] + 1, **summary,
                 seconds=float(arrays["seconds"].sum()), code_version=CODE_VERSION)
    _update_manifest(root, entry, shard_size or len(indices), index_start, proposal, near)
    return root / f"{name}.npz"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-draws", type=int, default=300_000)
    ap.add_argument("--shard-size", type=int, default=10_000)
    ap.add_argument("--n-jobs", type=int, default=22)
    ap.add_argument("--index-start", type=int, default=0, help="first draw index (active learning: a disjoint range)")
    ap.add_argument("--proposal", type=Path, default=None,
                    help="active-learning cells json; hosts come from its tilted prior instead of the plain prior")
    ap.add_argument("--near", nargs="+", metavar="NAME=COUNT", default=None,
                    help="hosts near known materials: draws per material, in index order; sets --n-draws to their sum")
    ap.add_argument("--near-radius", type=float, default=2.0, help="... within this many tolerances (realism.py)")
    a = ap.parse_args(argv)
    global PROPOSAL
    proposal, near = None, None
    if a.near is not None:
        if a.proposal is not None:
            sys.exit("--near and --proposal cannot be combined")
        try:
            near = near_spec([(n, int(c)) for n, c in (item.split("=") for item in a.near)], a.near_radius)
        except ValueError as e:
            sys.exit(f"--near takes NAME=COUNT items ({e})")
        a.n_draws = sum(near["counts"])
    if a.proposal is not None:
        from shazam.inverse import active
        spec = json.loads(a.proposal.read_text())
        PROPOSAL = np.asarray(spec["accept"], float)
        if PROPOSAL.shape != (active.N_CELLS,) or not (0 < PROPOSAL.max() <= 1) or PROPOSAL.min() < 0:
            sys.exit(f"{a.proposal}: 'accept' must hold {active.N_CELLS} probabilities with a positive maximum")
        proposal = dict(file=str(a.proposal), accept=PROPOSAL.tolist())
    if a.index_start % 20:
        sys.exit("--index-start must be a multiple of 20 so the index % 20 split is unchanged")
    if a.index_start + a.n_draws >= 2 ** 32:
        sys.exit("draw indices are disorder seeds and must stay below 2**32")
    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for stray in out.glob("*.tmp"):                    # left by an interrupted run
        stray.unlink()
    manifest = load_manifest(out)
    if manifest and manifest["shard_size"] != a.shard_size:
        sys.exit(f"{out} was started with --shard-size {manifest['shard_size']}, not {a.shard_size}")
    if manifest and manifest.get("near") != near:
        sys.exit(f"{out} was started with the near-material plan {manifest.get('near')}, not {near}")
    done = {e["shard"]: e for e in (manifest or {}).get("shards", [])}
    n_shards = -(-a.n_draws // a.shard_size)
    for k in range(n_shards):
        lo, hi = a.index_start + k * a.shard_size, a.index_start + min((k + 1) * a.shard_size, a.n_draws)
        e = done.get(k)
        if e and (out / e["file"]).exists() and (e["start"], e["stop"]) == (lo, hi):
            print(f"shard {k}: done, skipping", flush=True)
            continue
        t0 = time.time()
        print(f"shard {k}: draws {lo}..{hi - 1}", flush=True)
        write_shard(out, k, range(lo, hi), a.n_jobs, shard_size=a.shard_size, index_start=a.index_start,
                    proposal=proposal, near=near)
        print(f"shard {k}: written in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
