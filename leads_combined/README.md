# Precomputed Leads Combined NumPy Data

This directory contains consolidated Green's function matrices for the leads compiled into 3D NumPy binary arrays, grouped by system size.

## File Contents
For each system size `S` (5, 10, 15, ..., 60):
1. `leads_{S}.npy`: Stacked 3D NumPy array containing the complex lead self-energies.
2. `leads_{S}_meta.csv`: Metadata file documenting the energy parameter `w` grid mapping.

## Structure

### `leads_{S}.npy`
- **Type**: 3D float64 complex NumPy array (`complex128`).
- **Dimensions**: `(400, S, S)` where each slice `[i, :, :]` is a complex matrix of shape `(S, S)` representing the lead surface Green's function at the `i`-th energy point `w`.
- **Ordering**: Sorted sequentially by energy value `w` (from 0.0 to 3.99).

### `leads_{S}_meta.csv`
- **Columns**: `row_idx`, `w`.
- **Purpose**: Maps each row index of the corresponding `.npy` file to its original energy parameter `w` (frequency).

## Loading the Data
You can load the NumPy array and metadata in Python:

```python
import numpy as np
import pandas as pd

# 1. Load the 3D complex lead array for size 10
leads_matrix = np.load("leads_10.npy")
print("Matrix shape:", leads_matrix.shape)  # (400, 10, 10)
print("Data type:", leads_matrix.dtype)      # complex128

# 2. Get the lead Green's function matrix at index 0 (w = 0.0)
g_w0 = leads_matrix[0]
print("Matrix at w=0.0:\n", g_w0)

# 3. Load the metadata mapping
meta_df = pd.read_csv("leads_10_meta.csv", index_col="row_idx")
print(meta_df.head())
```
