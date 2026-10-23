#!/usr/bin/env python3
"""
Transmission Data Loader
------------------------
High-performance, memory-efficient Python module to access transmission data from `consolidated_data/`.
Uses NumPy memory-mapping (`mmap_mode='r'`) for sub-millisecond, zero-copy reads.
"""

import os
import json
import numpy as np

import paths

class TransmissionDataLoader:
    def __init__(self, consolidated_dir=None):
        if consolidated_dir is None:
            consolidated_dir = str(paths.agnr_consolidated_dir())
        
        self.consolidated_dir = consolidated_dir
        self._mmap_cache = {}
        
        # Load metadata if exists
        meta_path = os.path.join(self.consolidated_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None

    def get_available_sizes(self):
        """Returns sorted list of available sizes [7, 9]."""
        if self.metadata:
            return sorted([int(k.replace("size_", "")) for k in self.metadata.keys() if k.startswith("size_")])
        return [7, 9]

    def get_available_concentrations(self, size: int):
        """Returns sorted list of available concentrations for a given size."""
        key = f"size_{size}"
        if self.metadata and key in self.metadata:
            return self.metadata[key]["concentrations"]
        
        folder = os.path.join(self.consolidated_dir, f"size_{size}")
        if not os.path.exists(folder):
            return []
        concs = []
        for f in os.listdir(folder):
            if f.endswith('.npy'):
                parts = f.replace('.npy', '').split('_')
                if len(parts) == 3:
                    concs.append(int(parts[2]))
        return sorted(concs)

    def load_master_array(self, size: int, use_mmap: bool = True) -> np.ndarray:
        """
        Loads 3D master array for size_7 or size_9.
        Shapes:
          size_7: (34, 10000, 300)
          size_9: (49, 10000, 300)
        """
        valid_sizes = self.get_available_sizes()
        if size not in valid_sizes:
            raise ValueError(f"Invalid size {size}. Available sizes are: {valid_sizes}")

        filepath = os.path.join(self.consolidated_dir, f"size_{size}.npy")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Master file not found: {filepath}")
        mode = 'r' if use_mmap else None
        return np.load(filepath, mmap_mode=mode)

    def load_concentration_array(self, size: int, conc: int, use_mmap: bool = True) -> np.ndarray:
        """
        Loads 2D array of shape (10000, 300) for size_X_Y.
        """
        valid_sizes = self.get_available_sizes()
        if size not in valid_sizes:
            raise ValueError(f"Invalid size {size}. Available sizes are: {valid_sizes}")

        valid_concs = self.get_available_concentrations(size)
        if conc not in valid_concs:
            raise ValueError(
                f"Concentration {conc} does not exist for size_{size}.\n"
                f"Available concentrations for size_{size} are: {valid_concs}"
            )

        key = (size, conc)
        if use_mmap and key in self._mmap_cache:
            return self._mmap_cache[key]

        filepath = os.path.join(self.consolidated_dir, f"size_{size}", f"size_{size}_{conc}.npy")
        mode = 'r' if use_mmap else None
        arr = np.load(filepath, mmap_mode=mode)
        if use_mmap:
            self._mmap_cache[key] = arr
        return arr

    def get_config(self, size: int, conc: int, cfg_idx: int) -> np.ndarray:
        """
        Retrieves a single configuration array (300,) for a given size, concentration, and cfg_idx.
        Executes in microsecond time using ~2.4 KB RAM.
        """
        arr = self.load_concentration_array(size, conc, use_mmap=True)
        if not (0 <= cfg_idx < arr.shape[0]):
            raise IndexError(f"cfg_idx {cfg_idx} out of bounds. Must be 0 <= cfg_idx < {arr.shape[0]}")
        return arr[cfg_idx]

    def close(self):
        """Clears memory mapping cache."""
        self._mmap_cache.clear()

if __name__ == "__main__":
    loader = TransmissionDataLoader()
    sizes = loader.get_available_sizes()
    print("Available sizes:", sizes)
    for s in sizes:
        concs = loader.get_available_concentrations(s)
        print(f"Size {s}: {len(concs)} concentrations -> {concs[:5]} ... max: {concs[-1]}")
