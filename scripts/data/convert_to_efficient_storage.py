#!/usr/bin/env python3
"""
Convert transmission_results dataset into a memory-mapped consolidated storage folder.

Creates a new directory `consolidated_data/` next to the raw output containing:
  - `consolidated_data/size_7/size_7_{conc}.npy` (2D arrays: 10000 x 300)
  - `consolidated_data/size_9/size_9_{conc}.npy` (2D arrays: 10000 x 300)
  - `consolidated_data/size_7.npy` (3D master array: 34 x 10000 x 300)
  - `consolidated_data/size_9.npy` (3D master array: 49 x 10000 x 300)
  - `consolidated_data/metadata.json` (concentration lists & mapping info)

Original folders are left untouched!
"""

import paths
import os
import sys
import glob
import re
import time
import json
import numpy as np

DATA_DIR = str(paths.raw_root() / "transmission_results")           # the size_<w>_<conc> folders of raw output
OUTPUT_DIR = str(paths.raw_root() / "consolidated_data")            # the product; move it under data/ as legacy/

def get_target_directories():
    pattern = os.path.join(DATA_DIR, "size_*_*")
    dirs = sorted([d for d in glob.glob(pattern) if os.path.isdir(d)])
    return dirs

def parse_dir_name(dir_path):
    basename = os.path.basename(dir_path)
    match = re.match(r'size_(\d+)_(\d+)', basename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def convert_all():
    dirs = get_target_directories()
    print(f"Found {len(dirs)} original directories.")
    
    os.makedirs(os.path.join(OUTPUT_DIR, "size_7"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "size_9"), exist_ok=True)
    
    t_start = time.time()
    
    s7_concs = []
    s9_concs = []
    s7_arrays = []
    s9_arrays = []
    
    total_processed = 0

    for idx, dpath in enumerate(dirs, 1):
        size, conc = parse_dir_name(dpath)
        basename = os.path.basename(dpath)
        
        t0 = time.time()
        files = [os.path.join(dpath, f"{size}_agnr_conc{conc}_cfg{i}.npy") for i in range(10000)]
        
        data_2d = np.zeros((10000, 300), dtype=np.float64)
        for i, fpath in enumerate(files):
            data_2d[i] = np.load(fpath)
            
        out_2d_file = os.path.join(OUTPUT_DIR, f"size_{size}", f"size_{size}_{conc}.npy")
        np.save(out_2d_file, data_2d)
        
        # Verify 2D file
        sample_check = np.load(files[500])
        assert np.array_equal(data_2d[500], sample_check), f"Mismatch in {basename}"
        
        if size == 7:
            s7_concs.append(conc)
            s7_arrays.append(data_2d)
        elif size == 9:
            s9_concs.append(conc)
            s9_arrays.append(data_2d)
            
        elapsed = time.time() - t0
        total_processed += len(files)
        print(f"[{idx}/{len(dirs)}] Saved consolidated_data/size_{size}/size_{size}_{conc}.npy ({data_2d.nbytes/1e6:.2f} MB) [{elapsed:.2f}s]")

    # Create 3D Master Arrays
    print("\nCreating 3D master arrays for size_7 and size_9...")
    
    # Sort by concentration
    s7_indices = np.argsort(s7_concs)
    s7_concs = [s7_concs[i] for i in s7_indices]
    s7_arrays = [s7_arrays[i] for i in s7_indices]
    
    s9_indices = np.argsort(s9_concs)
    s9_concs = [s9_concs[i] for i in s9_indices]
    s9_arrays = [s9_arrays[i] for i in s9_indices]
    
    s7_master = np.stack(s7_arrays, axis=0) # shape (34, 10000, 300)
    s9_master = np.stack(s9_arrays, axis=0) # shape (49, 10000, 300)
    
    s7_master_path = os.path.join(OUTPUT_DIR, "size_7.npy")
    s9_master_path = os.path.join(OUTPUT_DIR, "size_9.npy")
    
    np.save(s7_master_path, s7_master)
    np.save(s9_master_path, s9_master)
    
    print(f"Saved {s7_master_path} - Shape: {s7_master.shape} ({os.path.getsize(s7_master_path)/1e6:.2f} MB)")
    print(f"Saved {s9_master_path} - Shape: {s9_master.shape} ({os.path.getsize(s9_master_path)/1e6:.2f} MB)")
    
    # Metadata JSON
    metadata = {
        "size_7": {
            "concentrations": s7_concs,
            "shape": list(s7_master.shape),
            "master_file": "size_7.npy",
            "folder": "size_7/"
        },
        "size_9": {
            "concentrations": s9_concs,
            "shape": list(s9_master.shape),
            "master_file": "size_9.npy",
            "folder": "size_9/"
        },
        "total_files_consolidated": total_processed
    }
    
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    t_total = time.time() - t_start
    print("\n" + "="*60)
    print("CONSOLIDATION COMPLETE!")
    print("="*60)
    print(f"Total time elapsed: {t_total:.2f} seconds")
    print(f"Files processed: {total_processed} files across {len(dirs)} folders")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    convert_all()
