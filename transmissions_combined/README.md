# Transmissions (Size 25) Combined Sensor Data

This directory contains consolidated sensor transmission data for the **size 25** transmissions system, grouped by impurity concentration.

## File Contents
For each concentration `C`, there is a file named `conc_{C}.csv`.

## Structure
The CSV files are structured **row-wise** (configurations as rows, transmission points as columns):
- **Columns**: 
  - `config_id`: The ID of the randomized impurity configuration.
  - `val_0`, `val_1`, ..., `val_399`: The transmission/conductance values.
- **Dimensions**: `(num_configs, 401)` (1 column for config ID, 400 columns for transmission values).

## Loading the Data
You can easily load any concentration file into a Pandas DataFrame or NumPy array:

```python
import pandas as pd
import numpy as np

# Load as Pandas DataFrame
df = pd.read_csv("conc_10.csv", index_col="config_id")
print("Shape:", df.shape)  # Should be (2000, 400)
print(df.head())

# Extract values as NumPy array
transmission_matrix = df.values  # shape: (2000, 400)
```
