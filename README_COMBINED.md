# Combined Datasets Overview: Systems & Concentrations

This document provides a comprehensive summary of all consolidated datasets generated from individual quantum transport simulation runs across different lattice geometries and system sizes.

For primary project overview, model architectures, and benchmark analysis, see [**`README.md`**](README.md).

---

## 1. Size 10 Square Lattice System (`size_10_combined/`)
- **System Description**: Tight-binding 2D square lattice / graphene system with size parameter 10.
- **Input Data**: 326,557 CSV files containing frequency `w` and conductance `G`.
- **Output Path**: `size_10_combined/conc_{C}.csv`
- **Output Format**: Row-wise CSV where columns are `config_id` followed by the 400 energy grid points `w` (`0.0`, `0.01`, ..., `3.99`).
- **Generation Script**: [`notebooks/square_lattice/combine_sq.py`](notebooks/square_lattice/combine_sq.py) and [`notebooks/square_lattice/pipeline_sq.py`](notebooks/square_lattice/pipeline_sq.py).

### Concentrations & Configuration Counts
There are **48 unique concentrations** in this dataset:

| Concentrations | Configs per Concentration | Total Configs |
|---|---|---|
| **5** | 659 configs | 659 |
| **7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39** (17 concs) | 10,050 configs each | 170,850 |
| **41** | 9,847 configs | 9,847 |
| **43, 45, 47, 49** (4 concs) | 5,050 configs each | 20,200 |
| **50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98** (25 concs) | 5,000 configs each | 125,000 |
| **Total** | | **326,557 configs** |

---

## 2. Size 25 System (`transmissions_combined/`)
- **System Description**: Graphene nanoribbon system with size parameter 25.
- **Input Data**: 17,218 CSV files containing 400-point conductance `G` vectors (no header).
- **Output Path**: `transmissions_combined/conc_{C}.csv`
- **Output Format**: Row-wise CSV where columns are `config_id` followed by `val_0` through `val_399`.

### Concentrations & Configuration Counts
There are **9 unique concentrations** (multiples of 5):

| Concentration | Configs per Concentration |
|---|---|
| **5** | 1,218 configs |
| **10, 15, 20, 25, 30, 35, 40, 45** | 2,000 configs each |
| **Total** | **17,218 configs** |

---

## 3. 7-AGNR System (`transmission_results_combined/`)
- **System Description**: 7-atom wide Armchair Graphene Nanoribbon (7-AGNR) transmission spectra across 100 unit cells ($N=1400$ atomic sites).
- **Input Data**: 740,352 binary `.npy` files containing 1D arrays of shape `(300,)`.
- **Output Path**: `transmission_results_combined/conc_{C}.npy` (along with metadata mapping `conc_{C}_meta.csv`).
- **Output Format**: 2D stacked NumPy array of shape `(num_configs, 300)` where each row is a transmission curve $T(E)$.
- **Generation Script**: [`scripts/combine_storage_data.py`](scripts/combine_storage_data.py) and [`notebooks/agnr/ca_agnr.py`](notebooks/agnr/ca_agnr.py).

### Concentrations & Configuration Counts
There are **74 unique concentrations** (sequential up to 98):
- **Concentrations**: `1` through `98` (excluding 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97).
- **Configuration Counts**: 
  - Concentrations `1` to `9`: 10,352 configs each.
  - Concentrations `10` to `98` (even/odd values present): 10,000 configs each.
- **Total Configurations**: **740,352 spectra**.

---

## 4. Precomputed Leads (`leads_combined/`)
- **System Description**: Precomputed lead self-energy / surface Green's function matrices over a frequency grid using vectorized Dyson iteration.
- **Input Data**: 4,400 CSV files containing `(size, size)` complex matrices (parenthesized complex numbers).
- **Output Path**: `leads_combined/leads_{S}.npy` (along with metadata mapping `leads_{S}_meta.csv`).
- **Output Format**: 3D complex NumPy array of shape `(400, S, S)` and dtype `complex128`.
- **Generation Scripts**: [`notebooks/square_lattice/compute_leads_sq.py`](notebooks/square_lattice/compute_leads_sq.py) and [`notebooks/agnr/build_leads_agnr.py`](notebooks/agnr/build_leads_agnr.py).

### System Sizes & Energy Grid Counts
There are **11 unique system sizes**:
- **Sizes ($S$)**: `5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60`.
- **Energy Grid Size**: 400 energy points `w` (sorted from `0.0` to `3.99` with step `0.01`—excluding `0.04`-step differences for intermediate values).

---

## Quick Loading Reference (Python)

### Loading CSV Combined Files (Size 10 / Size 25)
```python
import pandas as pd

# Load size 10 data for concentration 15
df_size10 = pd.read_csv("size_10_combined/conc_15.csv", index_col="config_id")
print("Size 10 shape:", df_size10.shape)  # (10050, 400)

# Load size 25 data for concentration 30
df_size25 = pd.read_csv("transmissions_combined/conc_30.csv", index_col="config_id")
print("Size 25 shape:", df_size25.shape)  # (2000, 400)
```

### Loading Stacked NumPy Arrays (7-AGNR & Leads)
```python
import numpy as np
import pandas as pd

# Load 2D stacked transmission matrix for concentration 10 (7-AGNR)
spectra = np.load("transmission_results_combined/conc_10.npy")
print("Numpy array shape:", spectra.shape)  # (10000, 300)

# Load 3D complex lead Green's functions for size 10
leads = np.load("leads_combined/leads_10.npy")
print("Leads array shape:", leads.shape)  # (400, 10, 10)
print("Leads array dtype:", leads.dtype)  # complex128
```
