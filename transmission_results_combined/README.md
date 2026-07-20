# Transmission Results (7-AGNR) Combined NumPy Data

This directory contains consolidated 7-AGNR transmission spectra compiled into 2D NumPy binary arrays, grouped by impurity concentration.

## File Contents
For each concentration `C`:
1. `conc_{C}.npy`: Stacked 2D NumPy array containing the transmission spectra.
2. `conc_{C}_meta.csv`: Metadata file documenting the configuration ID mapping.

## Structure

### `conc_{C}.npy`
- **Type**: 2D float64 NumPy array.
- **Dimensions**: `(num_configs, 300)` where each row is a 300-point transmission spectrum.
- **Ordering**: Sorted sequentially by `config_id`.

### `conc_{C}_meta.csv`
- **Columns**: `row_idx`, `config_id`.
- **Purpose**: Maps each row index of the corresponding `.npy` file back to its original physical configuration ID (e.g. `cfg1241`).

## Loading the Data
You can load the NumPy array and metadata in Python:

```python
import numpy as np
import pandas as pd

# 1. Load the 2D stacked transmission matrix
transmissions = np.load("conc_10.npy")
print("Matrix shape:", transmissions.shape)  # e.g., (10000, 300)

# 2. Load the metadata mapping
meta_df = pd.read_csv("conc_10_meta.csv", index_col="row_idx")
print(meta_df.head())

# Look up the config_id of the 5th configuration (row index 4)
config_id = meta_df.loc[4, "config_id"]
print(f"Row 4 represents config: {config_id}")
```
