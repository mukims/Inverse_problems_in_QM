#!/usr/bin/env python
"""Parallel, resumable generation of the hexagonal-parent training set (spec section 5).

One draw = one host from the prior of section 4.2 + one impurity count + one disorder seed (the draw index) -> one
disordered spectrum on the v4 grid of that host's scale. Draws go into shards of --shard-size; a shard is written to
shard_NNNNN.npz.tmp and renamed when complete, and the manifest lists the finished shards, so an interrupted run is
resumed by running the same command again.

    PYTHONPATH=src python scripts/generate/generate_parent.py --n-draws 2000 --n-jobs 22
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")             # one BLAS thread per worker; must be set before numpy is imported

import argparse
import json
import multiprocessing as mp
import queue
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import paths
from shazam.inverse.data import EDGE_CODES, LEAD_CODES, PARAM_COLUMNS, load_manifest
from tbribbon import parent_data
from tbribbon.parent import active_atoms_per_cell
from tbribbon.parent_prior import RANGES, sample_prior

MAX_ATTEMPTS = 100          # redraws allowed per index; the prior rejects a small fraction, so reaching this is a bug
REASONS = ("invisible_type", "no_channel", "few_channels")      # in the order they are checked (cheapest first)
N_CHANNELS = 416
DEFAULT_OUT = paths.data_root() / "_generated" / "hexagonal_parent_v1"


def draw_with_resampling(index):
    """Accepted draw for this index: (draw, n_imp, raw, clean, attempts, reasons).

    Attempt j draws from default_rng([index, j]); the disorder seed is the index. Checks run cheapest first: band
    weights (ms), then the clean open channels (seconds), then the disordered transport (up to a minute). If both the
    band-weight and the channel rule would reject a draw it is counted as invisible_type. `attempts` counts the
    attempts up to and including the accepted one; `reasons` counts the rejections by reason for this index.
    """
    reasons = {r: 0 for r in REASONS}
    probe = np.ones(N_CHANNELS)          # stands in for a spectrum not yet computed, so the cheap rules can run first
    for j in range(MAX_ATTEMPTS):
        draw = sample_prior(np.random.default_rng([index, j]))
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
                return draw, n_imp, raw, clean, j + 1, reasons
        reasons[why] += 1
    raise RuntimeError(f"draw index {index}: no acceptable host in {MAX_ATTEMPTS} attempts (rejections {reasons})")


def _record(index):
    """Everything stored for one index. `seconds` is the wall time of the whole attempt sequence up to acceptance."""
    t0 = time.perf_counter()
    draw, n_imp, raw, clean, attempts, reasons = draw_with_resampling(index)
    n_cells = parent_data.device_cells_for(draw.params, draw.edge)
    p = draw.params
    return dict(
        index=index, x=parent_data.to_input(raw), raw=raw, clean=clean,
        params=[getattr(p, c) for c in PARAM_COLUMNS], on=[p.on_C, p.on_H, p.on_K], lead=LEAD_CODES[p.lead],
        edge=EDGE_CODES[draw.edge], width=draw.width, density=draw.density, n_imp=n_imp, n_cells=n_cells,
        active_atoms=n_cells * active_atoms_per_cell(p, draw.edge, draw.width),
        seconds=time.perf_counter() - t0, attempts=attempts, reasons=reasons)


def _limit_blas():
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1)      # the env vars above only help if numpy was not imported yet
    except ImportError:
        return None


def _worker(tasks, results):
    keep = _limit_blas()
    while True:
        index = tasks.get()
        if index is None:
            return
        try:
            results.put(("ok", _record(index)))
        except BaseException:
            results.put(("error", index, traceback.format_exc()))
            return


def _compute(indices, n_jobs):
    """The records of all indices, in index order. Workers are forked, so they see the parent's (patched) modules."""
    indices = list(indices)
    if n_jobs <= 1:
        _limit_blas()
        return [_record(i) for i in indices]
    ctx = mp.get_context("fork")
    tasks, results = ctx.Queue(), ctx.Queue()
    for i in indices:
        tasks.put(i)
    n = min(n_jobs, len(indices))
    for _ in range(n):
        tasks.put(None)
    procs = [ctx.Process(target=_worker, args=(tasks, results), daemon=True) for _ in range(n)]
    for p in procs:
        p.start()
    got, t0 = [], time.time()
    try:
        while len(got) < len(indices):
            try:
                msg = results.get(timeout=5)
            except queue.Empty:
                if not any(p.is_alive() for p in procs):
                    raise RuntimeError("all workers exited before finishing their draws") from None
                continue
            if msg[0] == "error":
                raise RuntimeError(f"draw index {msg[1]} failed in a worker:\n{msg[2]}")
            got.append(msg[1])
            if len(got) % 100 == 0:
                print(f"  {len(got)}/{len(indices)} draws, {time.time() - t0:.0f} s", flush=True)
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join()
    return sorted(got, key=lambda r: r["index"])


def _atomic_bytes(path, write):
    """Run write(file) into path.tmp, flush to disk and rename over path."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _code_version():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.repo_root(), capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _update_manifest(root, entry, shard_size):
    """Add or replace the entry of one finished shard; counts are recomputed from the entries."""
    m = load_manifest(root) or {}
    shards = {e["shard"]: e for e in m.get("shards", [])}
    shards[entry["shard"]] = entry
    shards = [shards[k] for k in sorted(shards)]
    rej = {r: sum(e["rejections"][r] for e in shards) for r in REASONS}
    manifest = dict(
        prior_ranges=RANGES, code_version=_code_version(), shard_size=shard_size, shards=shards,
        counts=dict(n_draws=sum(e["n_draws"] for e in shards), n_attempts=sum(e["n_attempts"] for e in shards),
                    rejections=rej, seconds=sum(e["seconds"] for e in shards)))
    _atomic_bytes(Path(root) / "manifest.json", lambda f: f.write(json.dumps(manifest, indent=2).encode()))


def write_shard(root, shard, indices, n_jobs, shard_size=None):
    """Compute the draws of `indices` and write shard_{shard:05d}.npz, its rejection counts and the manifest entry."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    indices = list(indices)
    recs = _compute(indices, n_jobs)
    col = lambda key, dtype: np.array([r[key] for r in recs], dtype=dtype)
    arrays = dict(
        index=col("index", np.int64), x=col("x", np.float32), raw=col("raw", np.float32), clean=col("clean", np.int16),
        params=col("params", np.float64), on=col("on", np.int8), lead=col("lead", np.int8), edge=col("edge", np.int8),
        width=col("width", np.int16), density=col("density", np.float64), n_imp=col("n_imp", np.int32),
        n_cells=col("n_cells", np.int32), active_atoms=col("active_atoms", np.int32),
        seconds=col("seconds", np.float32), attempts=col("attempts", np.int16))
    rej = {r: sum(rec["reasons"][r] for rec in recs) for r in REASONS}
    name = f"shard_{shard:05d}"
    summary = dict(n_draws=len(recs), n_attempts=int(arrays["attempts"].sum()), rejections=rej)
    _atomic_bytes(root / f"{name}_rejections.json", lambda f: f.write(json.dumps(summary, indent=2).encode()))
    _atomic_bytes(root / f"{name}.npz", lambda f: np.savez(f, **arrays))
    entry = dict(shard=shard, file=f"{name}.npz", start=indices[0], stop=indices[-1] + 1, **summary,
                 seconds=float(arrays["seconds"].sum()))
    _update_manifest(root, entry, shard_size or len(indices))
    return root / f"{name}.npz"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-draws", type=int, default=300_000)
    ap.add_argument("--shard-size", type=int, default=10_000)
    ap.add_argument("--n-jobs", type=int, default=22)
    a = ap.parse_args(argv)
    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for stray in out.glob("*.tmp"):                    # left by an interrupted run
        stray.unlink()
    manifest = load_manifest(out)
    if manifest and manifest["shard_size"] != a.shard_size:
        sys.exit(f"{out} was started with --shard-size {manifest['shard_size']}, not {a.shard_size}")
    done = {e["shard"]: e for e in (manifest or {}).get("shards", [])}
    n_shards = -(-a.n_draws // a.shard_size)
    for k in range(n_shards):
        lo, hi = k * a.shard_size, min((k + 1) * a.shard_size, a.n_draws)
        e = done.get(k)
        if e and (out / e["file"]).exists() and (e["start"], e["stop"]) == (lo, hi):
            print(f"shard {k}: done, skipping", flush=True)
            continue
        t0 = time.time()
        print(f"shard {k}: draws {lo}..{hi - 1}", flush=True)
        write_shard(out, k, range(lo, hi), a.n_jobs, shard_size=a.shard_size)
        print(f"shard {k}: written in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
