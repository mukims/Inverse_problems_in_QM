#!/usr/bin/env python
"""
universal_data.py — Unified Dataset Loader for 7-AGNR, 9-AGNR, and Square Lattice (Size 10)

Loads, normalizes, and splits transmission spectra across:
  - 7-AGNR: 34 concentrations (c in {2, 4, ..., 68})
  - 9-AGNR: 49 concentrations (c in {2, 4, ..., 98})
  - Square Lattice (Size 10): 18 concentrations (c in {5, 10, ..., 90}), clean ca_sq.py data

Each sample contains:
  - x: Normalized transmission spectrum (first L channels, clipped to [0, 1])
  - y_type: 0 for AGNR, 1 for Square Lattice
  - y_width: 0 for width 7, 1 for width 9, 2 for size 10
  - y_conc: Scalar concentration float c
  - raw: Unnormalized raw transmission curve
"""

import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

# Default data directories
AGNR_CONSOLIDATED_DIR = "/run/media/shardul/storage/machine_learning/transmission_data/transmission_results/consolidated_data"
# Clean square data from ca_sq.py, stacked by combine_sq.py. The old CSVs in
# transmission_data/size_10 came from CA.ipynb's cache-mutation bug and are unusable.
SQUARE_DATA_DIR = os.path.expanduser("~/transmissions_sq/size_10_combined")
SQUARE_PRISTINE = os.path.expanduser("~/transmissions_sq/pristine_10.npy")

# Target concentration grids
CONCS_7 = np.arange(2, 70, 2)
CONCS_9 = np.arange(2, 100, 2)
CONCS_SQ = list(range(5, 91, 5))


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

    Split 70/15/15 by CONFIG SEED (LOGBOOK Bug #7): one seed gives nested impurity
    sets across concentrations, so whole seeds are held out in every system.
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
    print(f"  Square-10 (size 10)| {len(CONCS_SQ)} concentrations (c = {CONCS_SQ[0]}..{CONCS_SQ[-1]})")
    print(f"  Channels: {spectrum_len} | Samples per conc: {samples_per_conc}")
    print("=" * 70)

    # 1. Load Pristine References
    p7_path = repo_root / "7_agnr_pris.npy"
    p9_path = repo_root / "9_agnr_pris.npy"

    p7 = np.round(np.load(str(p7_path))[:spectrum_len], 3).astype(np.float32)
    p9 = np.round(np.load(str(p9_path))[:spectrum_len], 3).astype(np.float32)
    psq = np.round(np.load(SQUARE_PRISTINE)[:spectrum_len], 3).astype(np.float32)

    p7_safe = np.where(p7 > 1e-12, p7, 1.0)
    p9_safe = np.where(p9 > 1e-12, p9, 1.0)
    psq_safe = np.where(psq > 1e-12, psq, 1.0)

    X_list, y_type_list, y_width_list, y_conc_list, raw_list, cfg_list = [], [], [], [], [], []

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
        cfg_list.append(np.arange(n))  # row index == config seed

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
        cfg_list.append(np.arange(n))

    print(f"  Loaded AGNR samples: {sum(len(x) for x in X_list):,}")

    # 3. Load Square Lattice (Size 10): type=1 (Square), width=2 (10)
    print("\n[2/3] Loading Square Lattice (Size 10) stacked arrays...")
    t0_sq = time.time()
    n_sq = 0
    for c in CONCS_SQ:
        rows = np.load(os.path.join(SQUARE_DATA_DIR, f"conc_{c}.npy"), mmap_mode="r")
        cfg = np.loadtxt(os.path.join(SQUARE_DATA_DIR, f"conc_{c}_meta.csv"), delimiter=",", skiprows=1, dtype=int)[:, 1]
        pick = np.where(cfg < samples_per_conc)[0]
        raw = np.round(np.asarray(rows[pick, :spectrum_len], dtype=np.float32), 3)
        n = len(raw); n_sq += n
        X_list.append(np.clip(raw / psq_safe, 0.0, 1.0))
        raw_list.append(raw)
        y_type_list.append(np.full(n, 1, dtype=np.int64))     # 1 -> Square Lattice
        y_width_list.append(np.full(n, 2, dtype=np.int64))    # 2 -> Size 10
        y_conc_list.append(np.full(n, c, dtype=np.float32))
        cfg_list.append(cfg[pick])

    print(f"  Loaded Square-10 samples: {n_sq:,} in {time.time() - t0_sq:.2f}s")

    # 4. Concatenate and Split
    print("\n[3/3] Merging and partitioning into 70/15/15 splits...")
    X = np.concatenate(X_list, axis=0)
    raw_all = np.concatenate(raw_list, axis=0)
    y_type = np.concatenate(y_type_list, axis=0)
    y_width = np.concatenate(y_width_list, axis=0)
    y_conc = np.concatenate(y_conc_list, axis=0)
    cfg = np.concatenate(cfg_list, axis=0)

    total_samples = len(X)
    print(f"  Total assembled samples: {total_samples:,}")
    print(f"    - Type 0 (AGNR):           {(y_type == 0).sum():,}  (7-AGNR: {(y_width == 0).sum():,}, 9-AGNR: {(y_width == 1).sum():,})")
    print(f"    - Type 1 (Square Lattice): {(y_type == 1).sum():,}  (Size 10: {(y_width == 2).sum():,})")
    print(f"    - Concentration range:     [{y_conc.min():.0f}, {y_conc.max():.0f}]")

    # Deterministic split by config seed (Bug #7): seeds [0, 0.7K) train, [0.7K, 0.85K) val, rest test
    n_tr_cfg, n_va_cfg = int(0.70 * samples_per_conc), int(0.85 * samples_per_conc)
    idx_tr = np.where(cfg < n_tr_cfg)[0]
    idx_va = np.where((cfg >= n_tr_cfg) & (cfg < n_va_cfg))[0]
    idx_te = np.where(cfg >= n_va_cfg)[0]

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
        "split": "config-seed",
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
