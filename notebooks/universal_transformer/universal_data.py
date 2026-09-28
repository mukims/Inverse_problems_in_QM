#!/usr/bin/env python
"""
universal_data.py — Unified Dataset Loader for 7-AGNR, 9-AGNR, and Square Lattice (Size 10)

Loads, normalizes, and splits transmission spectra across:
  - 7-AGNR: 34 concentrations (c in {2, 4, ..., 68})
  - 9-AGNR: 49 concentrations (c in {2, 4, ..., 98})
  - Square Lattice (Size 10): 48 concentrations (c in [5, 98])

Each sample contains:
  - x: Normalized transmission spectrum (first L channels, clipped to [0, 1])
  - y_type: 0 for AGNR, 1 for Square Lattice
  - y_width: 0 for width 7, 1 for width 9, 2 for size 10
  - y_conc: Scalar concentration float c
  - raw: Unnormalized raw transmission curve
"""

import os
import sys
import re
import glob
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Ensure project root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

# Default data directories
AGNR_CONSOLIDATED_DIR = "/run/media/shardul/storage1/machine_learning/transmission_data/transmission_results/consolidated_data"
SQUARE_DATA_DIR = "/run/media/shardul/storage1/machine_learning/transmission_data/size_10"

# Target concentration grids
CONCS_7 = np.arange(2, 70, 2)
CONCS_9 = np.arange(2, 100, 2)
CONCS_SQ = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39,
            41, 43, 45, 47, 49, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72,
            74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98]


def _read_sq_file(path_and_len):
    path, length = path_and_len
    try:
        df = pd.read_csv(path)
        return df["G"].values[:length].astype(np.float32)
    except Exception:
        return None


class TargetScaler:
    """Zero-mean, unit-variance scaler for regression targets."""
    def __init__(self, targets):
        arr = np.asarray(targets, dtype=np.float32)
        self.mean = float(arr.mean())
        self.std = float(arr.std()) if arr.std() > 1e-6 else 1.0

    def transform(self, y):
        if torch.is_tensor(y):
            return (y - self.mean) / self.std
        return (np.asarray(y, dtype=np.float32) - self.mean) / self.std

    def inverse_transform(self, y_norm):
        if torch.is_tensor(y_norm):
            return y_norm * self.std + self.mean
        return np.asarray(y_norm, dtype=np.float32) * self.std + self.mean

    def as_dict(self):
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d):
        obj = cls.__new__(cls)
        obj.mean = float(d["mean"])
        obj.std = float(d["std"])
        return obj


def load_universal_data(
    repo_root: Path,
    samples_per_conc: int = 500,
    spectrum_len: int = 150,
    seed: int = 42,
    cache_path: str = None,
    num_workers: int = 16,
):
    """
    Loads and compiles the universal dataset across 7-AGNR, 9-AGNR, and Square-10.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path} ...")
        t0 = time.time()
        try:
            cached = torch.load(cache_path, weights_only=False)
            scaler_val = cached["scaler"]
            scaler = TargetScaler.from_dict(scaler_val) if isinstance(scaler_val, dict) else scaler_val
            print(f"Loaded cache in {time.time() - t0:.2f}s.")
            return cached["train"], cached["val"], cached["test"], scaler, cached["meta"]
        except Exception as e:
            print(f"Cache load failed ({e}), regenerating dataset...")

    print("=" * 70)
    print("ASSEMBLING UNIVERSAL MULTI-SYSTEM DATASET")
    print(f"  7-AGNR (width 7)   | 34 concentrations ({len(CONCS_7)})")
    print(f"  9-AGNR (width 9)   | 49 concentrations ({len(CONCS_9)})")
    print(f"  Square-10 (size 10)| 48 concentrations ({len(CONCS_SQ)})")
    print(f"  Channels: {spectrum_len} | Samples per conc: {samples_per_conc}")
    print("=" * 70)

    # 1. Load Pristine References
    p7_path = repo_root / "7_agnr_pris.npy"
    p9_path = repo_root / "9_agnr_pris.npy"
    psq_path = Path(SQUARE_DATA_DIR) / "size_10_pris.csv"

    p7 = np.round(np.load(str(p7_path))[:spectrum_len], 3).astype(np.float32)
    p9 = np.round(np.load(str(p9_path))[:spectrum_len], 3).astype(np.float32)
    psq = pd.read_csv(str(psq_path), header=None).values.squeeze()[:spectrum_len].astype(np.float32)

    p7_safe = np.where(p7 > 1e-12, p7, 1.0)
    p9_safe = np.where(p9 > 1e-12, p9, 1.0)
    psq_safe = np.where(psq > 1e-12, psq, 1.0)

    X_list, y_type_list, y_width_list, y_conc_list, raw_list = [], [], [], [], []

    # 2. Load 7-AGNR & 9-AGNR from memory-mapped files
    print("\n[1/3] Loading 7-AGNR & 9-AGNR memory maps...")
    s7 = np.load(os.path.join(AGNR_CONSOLIDATED_DIR, "size_7.npy"), mmap_mode="r")
    s9 = np.load(os.path.join(AGNR_CONSOLIDATED_DIR, "size_9.npy"), mmap_mode="r")

    # 7-AGNR: type=0 (AGNR), width=0 (7)
    for idx, c in enumerate(CONCS_7):
        if idx >= s7.shape[0]:
            break
        raw = np.round(np.array(s7[idx, :samples_per_conc, :spectrum_len], dtype=np.float32), 3)
        normed = np.clip(raw / p7_safe, 0.0, 1.0)
        n = len(raw)
        X_list.append(normed)
        raw_list.append(raw)
        y_type_list.append(np.full(n, 0, dtype=np.int64))
        y_width_list.append(np.full(n, 0, dtype=np.int64))  # 0 -> width 7
        y_conc_list.append(np.full(n, c, dtype=np.float32))

    # 9-AGNR: type=0 (AGNR), width=1 (9)
    for idx, c in enumerate(CONCS_9):
        if idx >= s9.shape[0]:
            break
        raw = np.round(np.array(s9[idx, :samples_per_conc, :spectrum_len], dtype=np.float32), 3)
        normed = np.clip(raw / p9_safe, 0.0, 1.0)
        n = len(raw)
        X_list.append(normed)
        raw_list.append(raw)
        y_type_list.append(np.full(n, 0, dtype=np.int64))
        y_width_list.append(np.full(n, 1, dtype=np.int64))  # 1 -> width 9
        y_conc_list.append(np.full(n, c, dtype=np.float32))

    print(f"  Loaded AGNR samples: {sum(len(x) for x in X_list):,}")

    # 3. Load Square Lattice (Size 10): type=1 (Square), width=2 (10)
    print("\n[2/3] Loading Square Lattice (Size 10) CSV files...")
    t0_sq = time.time()
    sq_tasks = []
    sq_concs_tracked = []
    pattern = re.compile(r"lead_size_10_conc_(\d+)_config_(\d+)\.csv")

    for c in CONCS_SQ:
        files = glob.glob(os.path.join(SQUARE_DATA_DIR, f"lead_size_10_conc_{c}_config_*.csv"))
        def cfg_id(p):
            m = pattern.search(p)
            return int(m.group(2)) if m else -1
        files.sort(key=cfg_id)
        selected_files = files[:samples_per_conc]
        for f in selected_files:
            sq_tasks.append((f, spectrum_len))
            sq_concs_tracked.append(c)

    print(f"  Reading {len(sq_tasks):,} square lattice files with {num_workers} processes...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        sq_results = list(executor.map(_read_sq_file, sq_tasks, chunksize=200))

    sq_raw = []
    sq_valid_concs = []
    for raw, c in zip(sq_results, sq_concs_tracked):
        if raw is not None and len(raw) == spectrum_len:
            sq_raw.append(raw)
            sq_valid_concs.append(c)

    sq_raw = np.array(sq_raw, dtype=np.float32)
    sq_normed = np.clip(sq_raw / psq_safe, 0.0, 1.0)
    n_sq = len(sq_raw)

    X_list.append(sq_normed)
    raw_list.append(sq_raw)
    y_type_list.append(np.full(n_sq, 1, dtype=np.int64))     # 1 -> Square Lattice
    y_width_list.append(np.full(n_sq, 2, dtype=np.int64))    # 2 -> Size 10
    y_conc_list.append(np.array(sq_valid_concs, dtype=np.float32))

    print(f"  Loaded Square-10 samples: {n_sq:,} in {time.time() - t0_sq:.2f}s")

    # 4. Concatenate and Split
    print("\n[3/3] Merging and partitioning into 70/15/15 splits...")
    X = np.concatenate(X_list, axis=0)
    raw_all = np.concatenate(raw_list, axis=0)
    y_type = np.concatenate(y_type_list, axis=0)
    y_width = np.concatenate(y_width_list, axis=0)
    y_conc = np.concatenate(y_conc_list, axis=0)

    total_samples = len(X)
    print(f"  Total assembled samples: {total_samples:,}")
    print(f"    - Type 0 (AGNR):           {(y_type == 0).sum():,}  (7-AGNR: {(y_width == 0).sum():,}, 9-AGNR: {(y_width == 1).sum():,})")
    print(f"    - Type 1 (Square Lattice): {(y_type == 1).sum():,}  (Size 10: {(y_width == 2).sum():,})")
    print(f"    - Concentration range:     [{y_conc.min():.0f}, {y_conc.max():.0f}]")

    # Deterministic Split
    rng = np.random.RandomState(seed)
    perm = rng.permutation(total_samples)

    n_tr = int(0.70 * total_samples)
    n_va = int(0.15 * total_samples)
    idx_tr = perm[:n_tr]
    idx_va = perm[n_tr:n_tr + n_va]
    idx_te = perm[n_tr + n_va:]

    print(f"  Split counts: Train {len(idx_tr):,} | Val {len(idx_va):,} | Test {len(idx_te):,}")

    scaler = TargetScaler(y_conc[idx_tr])

    def make_dataset(indices):
        return {
            "x": torch.tensor(X[indices], dtype=torch.float32),
            "y_type": torch.tensor(y_type[indices], dtype=torch.int64),
            "y_width": torch.tensor(y_width[indices], dtype=torch.int64),
            "y_conc": torch.tensor(y_conc[indices], dtype=torch.float32),
            "raw": torch.tensor(raw_all[indices], dtype=torch.float32),
        }

    train_data = make_dataset(idx_tr)
    val_data = make_dataset(idx_va)
    test_data = make_dataset(idx_te)

    meta = {
        "samples_per_conc": samples_per_conc,
        "spectrum_len": spectrum_len,
        "seed": seed,
        "total_samples": total_samples,
        "type_labels": {0: "AGNR", 1: "Square Lattice"},
        "width_labels": {0: "7-AGNR", 1: "9-AGNR", 2: "Square-10"},
        "concs_7": CONCS_7.tolist(),
        "concs_9": CONCS_9.tolist(),
        "concs_sq": CONCS_SQ,
    }

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({
            "train": train_data,
            "val": val_data,
            "test": test_data,
            "scaler": scaler.as_dict(),
            "meta": meta,
        }, cache_path)
        print(f"✓ Saved preprocessed dataset cache to {cache_path}")

    return train_data, val_data, test_data, scaler, meta
