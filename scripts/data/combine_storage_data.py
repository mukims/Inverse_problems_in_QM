import os
import re
import sys
import argparse
from collections import defaultdict
import numpy as np
from multiprocessing import Pool, cpu_count

# Regex patterns for parsing files
SIZE_10_PATTERN = r'lead_size_\d+_conc_(\d+)_config_(\d+)\.csv'
TRANSMISSIONS_PATTERN = r'size_\d+_conc_(\d+)_config_(\d+)\.csv'
TRANSMISSION_RESULTS_PATTERN = r'(?:7_agnr_conc)?(\d+)_cfg(\d+)\.npy'
LEADS_PATTERN = r'leads_(\d+)_w_([\d\.]+)\.csv'

# Read helper functions for multiprocessing
def read_size_10_file(args):
    config_id, filepath = args
    try:
        with open(filepath, 'r') as f:
            next(f)  # skip header
            # Extract the G column (second column)
            g_vals = [line.split(',')[1].strip() for line in f]
        return config_id, g_vals, None
    except Exception as e:
        return config_id, None, str(e)

def read_transmissions_file(args):
    config_id, filepath = args
    try:
        with open(filepath, 'r') as f:
            g_vals = [line.strip() for line in f]
        return config_id, g_vals, None
    except Exception as e:
        return config_id, None, str(e)

def read_npy_file(args):
    config_id, filepath = args
    try:
        arr = np.load(filepath)
        return config_id, arr, None
    except Exception as e:
        return config_id, None, str(e)

def read_lead_file(args):
    w_val, filepath, size = args
    try:
        matrix = []
        with open(filepath, 'r') as f:
            for line in f:
                # Parse complex numbers in form (a+bj)
                row = [complex(x.strip('() \n\t')) for x in line.split(',') if x.strip()]
                matrix.append(row)
        arr = np.array(matrix, dtype=np.complex128)
        if arr.shape != (size, size):
            return w_val, None, f"Expected shape ({size}, {size}), got {arr.shape}"
        return w_val, arr, None
    except Exception as e:
        return w_val, None, str(e)

def scan_dir(dir_path, file_regex):
    print(f"Scanning directory: {dir_path} ...")
    pattern = re.compile(file_regex)
    grouped_files = defaultdict(list)
    
    count = 0
    with os.scandir(dir_path) as it:
        for entry in it:
            if entry.is_file():
                m = pattern.match(entry.name)
                if m:
                    conc = int(m.group(1))
                    config = int(m.group(2))
                    grouped_files[conc].append((config, entry.path))
                    count += 1
                    
    print(f"Scan complete. Found {count} files matching pattern. {len(grouped_files)} unique concentrations.")
    return grouped_files

def scan_leads_dir(dir_path):
    print(f"Scanning directory: {dir_path} ...")
    pattern = re.compile(LEADS_PATTERN)
    grouped_files = defaultdict(list)
    
    count = 0
    with os.scandir(dir_path) as it:
        for entry in it:
            if entry.is_file():
                m = pattern.match(entry.name)
                if m:
                    size = int(m.group(1))
                    w_val = float(m.group(2))
                    grouped_files[size].append((w_val, entry.path))
                    count += 1
                    
    print(f"Scan complete. Found {count} files matching pattern. {len(grouped_files)} unique sizes.")
    return grouped_files

def get_energy_grid(filepath):
    try:
        with open(filepath, 'r') as f:
            next(f)  # skip header
            w_vals = [line.split(',')[0].strip() for line in f]
        return w_vals
    except Exception as e:
        print(f"Error reading energy grid: {e}")
        return None

def merge_size_10(base_dir, num_workers):
    input_dir = os.path.join(base_dir, 'size_10')
    output_dir = os.path.join(base_dir, 'size_10_combined')
    os.makedirs(output_dir, exist_ok=True)
    
    grouped = scan_dir(input_dir, SIZE_10_PATTERN)
    if not grouped:
        print("No matching files found in size_10.")
        return
        
    # Get energy grid w from one of the files
    first_conc = list(grouped.keys())[0]
    sample_file = grouped[first_conc][0][1]
    w_grid = get_energy_grid(sample_file)
    if not w_grid:
        print("Could not load energy grid. Aborting size_10 merge.")
        return
        
    with Pool(num_workers) as pool:
        for conc, files in sorted(grouped.items()):
            print(f"Processing size_10 | concentration {conc} ({len(files)} files)...")
            # Sort files by config_id
            files = sorted(files, key=lambda x: x[0])
            
            # Map with pool
            results = pool.map(read_size_10_file, files)
            
            # Write to output CSV
            output_file = os.path.join(output_dir, f'conc_{conc}.csv')
            with open(output_file, 'w') as f_out:
                # Write header: config_id, w0, w1, ...
                f_out.write("config_id," + ",".join(w_grid) + "\n")
                
                for config_id, g_vals, err in results:
                    if err:
                        print(f"  Error reading config {config_id}: {err}")
                        continue
                    if g_vals:
                        f_out.write(f"{config_id}," + ",".join(g_vals) + "\n")
            print(f"  Saved to {output_file}")

def merge_transmissions(base_dir, num_workers):
    input_dir = os.path.join(base_dir, 'transmissions')
    output_dir = os.path.join(base_dir, 'transmissions_combined')
    os.makedirs(output_dir, exist_ok=True)
    
    grouped = scan_dir(input_dir, TRANSMISSIONS_PATTERN)
    if not grouped:
        print("No matching files found in transmissions.")
        return
        
    # Get num values from first file
    first_conc = list(grouped.keys())[0]
    sample_file = grouped[first_conc][0][1]
    with open(sample_file, 'r') as f:
        num_vals = len(f.readlines())
        
    val_headers = [f"val_{i}" for i in range(num_vals)]
    
    with Pool(num_workers) as pool:
        for conc, files in sorted(grouped.items()):
            print(f"Processing transmissions | concentration {conc} ({len(files)} files)...")
            files = sorted(files, key=lambda x: x[0])
            
            results = pool.map(read_transmissions_file, files)
            
            output_file = os.path.join(output_dir, f'conc_{conc}.csv')
            with open(output_file, 'w') as f_out:
                f_out.write("config_id," + ",".join(val_headers) + "\n")
                for config_id, g_vals, err in results:
                    if err:
                        print(f"  Error reading config {config_id}: {err}")
                        continue
                    if g_vals:
                        f_out.write(f"{config_id}," + ",".join(g_vals) + "\n")
            print(f"  Saved to {output_file}")

def merge_transmission_results(base_dir, num_workers):
    input_dir = os.path.join(base_dir, 'transmission_results')
    output_dir = os.path.join(base_dir, 'transmission_results_combined')
    os.makedirs(output_dir, exist_ok=True)
    
    grouped = scan_dir(input_dir, TRANSMISSION_RESULTS_PATTERN)
    if not grouped:
        print("No matching files found in transmission_results.")
        return
        
    with Pool(num_workers) as pool:
        for conc, files in sorted(grouped.items()):
            print(f"Processing transmission_results | concentration {conc} ({len(files)} files)...")
            files = sorted(files, key=lambda x: x[0])
            
            results = pool.map(read_npy_file, files)
            
            configs = []
            arrays = []
            for config_id, arr, err in results:
                if err:
                    print(f"  Error reading config {config_id}: {err}")
                    continue
                if arr is not None:
                    configs.append(config_id)
                    arrays.append(arr)
                    
            if not arrays:
                continue
                
            # Stack arrays into 2D: shape (num_configs, 300)
            stacked = np.stack(arrays, axis=0)
            
            # Save stacked array
            output_file = os.path.join(output_dir, f'conc_{conc}.npy')
            np.save(output_file, stacked)
            
            # Save a config mapping metadata (CSV) to know which row corresponds to which config_id
            meta_file = os.path.join(output_dir, f'conc_{conc}_meta.csv')
            with open(meta_file, 'w') as f_meta:
                f_meta.write("row_idx,config_id\n")
                for idx, cfg in enumerate(configs):
                    f_meta.write(f"{idx},{cfg}\n")
                    
            print(f"  Saved {stacked.shape} array to {output_file} and metadata to {meta_file}")

def merge_leads(base_dir, num_workers):
    input_dir = os.path.join(base_dir, 'leads')
    output_dir = os.path.join(base_dir, 'leads_combined')
    os.makedirs(output_dir, exist_ok=True)
    
    grouped = scan_leads_dir(input_dir)
    if not grouped:
        print("No matching files found in leads.")
        return
        
    with Pool(num_workers) as pool:
        for size, files in sorted(grouped.items()):
            print(f"Processing leads | size {size} ({len(files)} files)...")
            # Sort files by w energy value
            files_sorted = sorted(files, key=lambda x: x[0])
            
            # Prepare arguments for parallel pool mapping
            pool_args = [(w, path, size) for w, path in files_sorted]
            
            results = pool.map(read_lead_file, pool_args)
            
            ws = []
            arrays = []
            for w_val, arr, err in results:
                if err:
                    print(f"  Error reading file at w={w_val}: {err}")
                    continue
                if arr is not None:
                    ws.append(w_val)
                    arrays.append(arr)
                    
            if not arrays:
                continue
                
            # Stack arrays into 3D: shape (num_w, size, size)
            stacked = np.stack(arrays, axis=0)
            
            # Save stacked array
            output_file = os.path.join(output_dir, f'leads_{size}.npy')
            np.save(output_file, stacked)
            
            # Save energy mapping metadata (CSV)
            meta_file = os.path.join(output_dir, f'leads_{size}_meta.csv')
            with open(meta_file, 'w') as f_meta:
                f_meta.write("row_idx,w\n")
                for idx, w in enumerate(ws):
                    f_meta.write(f"{idx},{w}\n")
                    
            print(f"  Saved {stacked.shape} complex array to {output_file} and metadata to {meta_file}")

def main():
    parser = argparse.ArgumentParser(description="Combine sensor data files by concentration")
    parser.add_argument('--base_dir', type=str, default='/run/media/shardul/storage/machine_learning/transmission_github/transmissions',
                        help="Base path of the transmissions project")
    parser.add_argument('--parts', nargs='+', default=['size_10', 'transmissions', 'transmission_results', 'leads'],
                        help="Subdirectories to process")
    parser.add_argument('--workers', type=int, default=cpu_count(),
                        help="Number of parallel worker processes")
                        
    args = parser.parse_args()
    
    print(f"Starting merge pipeline in: {args.base_dir}")
    print(f"Workers: {args.workers}")
    print(f"Tasks: {args.parts}")
    
    if 'size_10' in args.parts:
        print("\n=== Processing size_10 ===")
        merge_size_10(args.base_dir, args.workers)
        
    if 'transmissions' in args.parts:
        print("\n=== Processing transmissions ===")
        merge_transmissions(args.base_dir, args.workers)
        
    if 'transmission_results' in args.parts:
        print("\n=== Processing transmission_results ===")
        merge_transmission_results(args.base_dir, args.workers)
        
    if 'leads' in args.parts:
        print("\n=== Processing leads ===")
        merge_leads(args.base_dir, args.workers)
        
    print("\nMerge pipeline complete!")

if __name__ == '__main__':
    main()
