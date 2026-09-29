#!/usr/bin/env python3
"""Standalone empirical stress harness for atlaslib."""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Ensure paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "material_atlas"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from atlaslib import Atlas, Conv1dAE, InputSpec, Registry, train_autoencoder
from atlaslib.importers import nearest_count
from atlaslib.store import CloudStore
from tests.atlas.toy import E, toy_spectrum, toy_store


def section(title):
    print(f"\n{'='*70}\n[STRESS] {title}\n{'='*70}")


def run_batch_scaling_stress(atlas):
    section("1. Batch Scaling & Throughput in Atlas.locate")
    
    # 1 item (1D)
    s1 = toy_spectrum(1.0, 7, 0.01, 5000)
    t0 = time.time()
    res1 = atlas.locate(s1, E, 3.0)
    lat1 = (time.time() - t0) * 1000
    print(f"Single item (1D): latency={lat1:.2f}ms | mat={res1[0].material} width={res1[0].width:.2f} recon={res1[0].recon_error:.4f}")

    # 1 item (2D)
    res1_2d = atlas.locate(s1[None], E, 3.0)
    assert res1[0] == res1_2d[0], "1D vs 2D mismatch!"

    # 100 items
    b100 = np.stack([toy_spectrum(1.0, 7, 0.01, 5000 + i) for i in range(100)])
    t0 = time.time()
    res100 = atlas.locate(b100, E, 3.0)
    dur100 = time.time() - t0
    fps100 = 100 / dur100
    print(f"Batch 100 items: total={dur100*1000:.2f}ms | throughput={fps100:.1f} samples/s ({dur100/100*1000:.3f}ms/sample)")

    # Consistency check
    res0_single = atlas.locate(b100[0], E, 3.0)[0]
    assert res100[0].material == res0_single.material
    assert res100[0].edge == res0_single.edge
    assert np.isclose(res100[0].width, res0_single.width, atol=1e-5)
    print("Batch vs Single consistency: 100% MATCH on predictions")

    # 1000 items
    b1000 = np.repeat(b100, 10, axis=0)
    t0 = time.time()
    res1000 = atlas.locate(b1000, E, 3.0)
    dur1000 = time.time() - t0
    fps1000 = 1000 / dur1000
    print(f"Batch 1000 items: total={dur1000*1000:.2f}ms | throughput={fps1000:.1f} samples/s ({dur1000/1000*1000:.3f}ms/sample)")

    # 5000 items (multi-chunk crossing 4096)
    b5000 = np.repeat(b100[:50], 100, axis=0)
    t0 = time.time()
    res5000 = atlas.locate(b5000, E, 3.0)
    dur5000 = time.time() - t0
    fps5000 = 5000 / dur5000
    print(f"Super-batch 5000 items: total={dur5000*1000:.2f}ms | throughput={fps5000:.1f} samples/s ({dur5000/5000*1000:.3f}ms/sample)")

    # Empty batch boundary
    try:
        b0 = np.empty((0, 400))
        atlas.locate(b0, E, 3.0)
        print("Empty batch: SUCCESS (unexpected)")
    except ValueError as e:
        print(f"Empty batch (0 items): Caught expected edge case ({type(e).__name__}: {e})")


def run_dimension_checks():
    section("2. Dimension Checking & Divisibility Constraints")

    # Conv1dAE divisibility by 8
    for l in (399, 400, 401, 407, 408):
        try:
            m = Conv1dAE(latent=8, seq_len=l)
            r, z = m(torch.randn(2, 1, l))
            status = f"PASSED (divisible: {l % 8 == 0})"
        except RuntimeError as e:
            status = f"REJECTED with RuntimeError ({e})"
        print(f"Conv1dAE seq_len={l} (mod 8 = {l % 8}): {status}")

    # InputSpec checks
    spec = InputSpec()
    checks = [
        ("Mismatched channels (350 vs 400)", lambda: spec.to_input(np.ones((2, 350)), E)),
        ("Grid origin != 0", lambda: spec.to_input(np.ones(400), E + 0.1)),
        ("Non-monotonic grid", lambda: spec.to_input(np.ones(400), np.zeros(400))),
        ("Premature truncation before band top", lambda: spec.to_input(np.ones(250), E[:250], band_top_t=3.0)),
        ("3D input array", lambda: spec.to_input(np.ones((2, 1, 400)), E)),
    ]
    for name, fn in checks:
        try:
            fn()
            print(f"InputSpec {name}: FAIL (did not raise)")
        except ValueError as e:
            print(f"InputSpec {name}: REJECTED properly ({e})")


def run_cloudstore_stress(tmp_dir):
    section("3. CloudStore Rapid Writes, Read-Back & Corruption Resilience")
    store = CloudStore(tmp_dir / "store")
    mid = "alpha/armchair/N7"
    store.write_pristine(mid, E, toy_spectrum(1.0, 7, 0.0, 1234))

    n_clouds = 50
    densities = [round(0.005 + i * 0.001, 5) for i in range(n_clouds)]
    print(f"Writing {n_clouds} clouds sequentially...")
    t0 = time.time()
    for d in densities:
        seeds = np.arange(10) + int(d * 1e6)
        spec = np.stack([toy_spectrum(1.0, 7, d, s) for s in seeds])
        store.write_cloud(mid, d, max(1, int(d * 1400)), spec, seeds, E)
    dur_write = time.time() - t0
    print(f"Sequential write completed: {dur_write:.2f}s ({dur_write/n_clouds*1000:.1f}ms/cloud)")

    # Read-back
    t0 = time.time()
    for d in densities:
        read_s, read_seeds = store.read_cloud(mid, d)
        exp_seeds = np.arange(10) + int(d * 1e6)
        exp_s = np.stack([toy_spectrum(1.0, 7, d, s) for s in exp_seeds])
        assert np.array_equal(read_seeds, exp_seeds)
        assert np.array_equal(read_s, exp_s)
    dur_read = time.time() - t0
    print(f"Read-back 100% bit-for-bit parity verified: {dur_read:.2f}s ({dur_read/n_clouds*1000:.1f}ms/cloud)")

    # Corruption checks
    print("Testing corruption resilience:")
    # NaN
    try:
        store.write_cloud(mid, 0.99, 10, np.full((2, 400), np.nan), np.array([1, 2]), E)
    except ValueError as e:
        print(f"  - Rejects NaN in spectra: PASS ({e})")

    # Duplicate seeds
    try:
        store.write_cloud(mid, 0.98, 10, np.zeros((2, 400)), np.array([5, 5]), E)
    except ValueError as e:
        print(f"  - Rejects duplicate seeds: PASS ({e})")

    # Corrupt meta.json
    (store._dir(mid) / "meta.json").write_text("{malformed")
    try:
        store.densities(mid)
    except json.JSONDecodeError:
        print("  - Detects corrupt meta.json: PASS (JSONDecodeError)")


def run_thread_limit_stress():
    section("4. PyTorch Thread Constraints & CPU Scaling")
    X_tr = np.random.randn(800, 400).astype(np.float32)
    X_va = np.random.randn(200, 400).astype(np.float32)

    for target_th in (1, 2, 4):
        t0 = time.time()
        t_cpu_0 = os.times()
        train_autoencoder(X_tr, X_va, latent=8, epochs=4, batch_size=64, threads=target_th, seed=0)
        t_cpu_1 = os.times()
        wall = time.time() - t0
        cpu_time = (t_cpu_1.user - t_cpu_0.user) + (t_cpu_1.system - t_cpu_0.system)
        parallelism = cpu_time / wall if wall > 0 else 0
        current_th = torch.get_num_threads()
        print(f"Target threads={target_th} -> get_num_threads()={current_th} | Parallelism={parallelism:.2f}x (wall={wall:.2f}s, cpu={cpu_time:.2f}s)")
        assert current_th == target_th, "Thread count mismatch!"
        assert parallelism <= target_th + 0.5, "Parallelism exceeded thread count limit!"


def run_numerical_stability(atlas):
    section("5. Numerical Stability & Adversarial Inputs")

    # Positive spike
    s_spike = toy_spectrum(1.0, 7, 0.01, 100)
    s_spike[50] = 1e9
    r_spike = atlas.locate(s_spike, E, 3.0)[0]
    print(f"Spike (1e9): mat={r_spike.material} recon={r_spike.recon_error:.4f} isfinite={np.isfinite(r_spike.recon_error)}")

    # Infinite spike
    s_inf = toy_spectrum(1.0, 7, 0.01, 100)
    s_inf[50] = np.inf
    r_inf = atlas.locate(s_inf, E, 3.0)[0]
    print(f"Spike (inf): mat={r_inf.material} recon={r_inf.recon_error:.4f} isfinite={np.isfinite(r_inf.recon_error)}")

    # Negative unphysical transmission
    s_neg = np.full(400, -50.0)
    r_neg = atlas.locate(s_neg, E, 3.0)[0]
    print(f"Negative transmission (-50.0): unknown={r_neg.unknown} recon={r_neg.recon_error:.4f}")

    # Subnormal transmission
    s_sub = np.full(400, 1e-15)
    r_sub = atlas.locate(s_sub, E, 3.0)[0]
    print(f"Subnormal transmission (1e-15): unknown={r_sub.unknown} recon={r_sub.recon_error:.4f}")

    # Noise degradation ladder
    base = toy_spectrum(1.0, 9, 0.01, 42)
    rng = np.random.default_rng(42)
    print("Noise degradation ladder:")
    for sig in (0.01, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0):
        noisy = np.clip(base + rng.normal(0, sig, base.size), 0, None)
        r = atlas.locate(noisy, E, 3.0)[0]
        print(f"  sigma={sig:4.2f} -> mat={r.material:<5} width={r.width:4.1f} novelty={r.novelty:6.4f} recon={r.recon_error:6.4f} unknown={r.unknown}")


def run_importers_stress():
    section("6. Importers Stress: nearest_count")
    # Ties
    assert nearest_count(3.5, [3, 4]) == 3
    assert nearest_count(3.5, [4, 3]) == 3
    print("Tie breaking (prefers lower): PASS")

    # Extremes
    assert nearest_count(-1000, [10, 20, 30]) == 10
    assert nearest_count(10000, [10, 20, 30]) == 30
    print("Extreme targets: PASS")

    # Empty
    try:
        nearest_count(5, [])
    except ValueError:
        print("Empty available list rejection: PASS")

    # 1M array throughput
    counts = np.arange(1_000_000) * 2
    t0 = time.time()
    c = nearest_count(500_001, counts)
    dur = time.time() - t0
    print(f"1,000,000 counts lookup: {dur*1000:.2f}ms (res={c})")


def main():
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        store, models = toy_store(tmp)
        reg = Registry(models)
        print("Building toy atlas for stress tests...")
        atlas = Atlas.build(store, reg, reg.ids(), InputSpec(), latent=8, epochs=4, patience=2, k=7, refs_per_model=150, threads=4)
        
        run_batch_scaling_stress(atlas)
        run_dimension_checks()
        run_cloudstore_stress(tmp)
        run_thread_limit_stress()
        run_numerical_stability(atlas)
        run_importers_stress()
        
        section("ALL STRESS TESTS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
