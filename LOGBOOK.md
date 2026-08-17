# Quantum Transport ML Experiment Logbook & Build Tracker

Welcome to the project **Logbook**. This document serves as the single source of truth for tracking all model builds, data cleaning iterations, bug fixes, architecture hyperparameters, and benchmark metrics across the 7-AGNR and 9-AGNR quantum transport inverse problem pipelines.

---

## 1. Experiment & Build Registry

| Build ID | Date | Target System | Architectures / Models | Dataset & Samples | Data Cleaning / Normalization | Width Acc (%) | Conc MAE | Conc RMSE | Status & Notes |
|:---|:---|:---|:---|:---|:---|:---:|:---:|:---:|:---|
| **BUILD-01** | *Initial* | 7-AGNR | ConductanceMLP (PINN) | `data/raw/` (2,100 samples) | $T(E) / T_{\text{pris}}(E)$ with CurvatureMisfit | N/A (7 only) | 1.176 | 1.528 | Baseline PINN architecture established. |
| **BUILD-02** | *Initial* | 7-AGNR | Patched Transformer v2 | `data/raw/` (2,100 samples) | $T(E) / T_{\text{pris}}(E)$ + ConvStem | N/A (7 only) | 0.978 | 1.348 | 1D Self-attention with ConvStem and [CLS] token. |
| **BUILD-03** | *Optuna* | 7-AGNR & 9-AGNR | XGBoost Regressor | Consolidated stacks (`xgb.ipynb`) | $E \le 1.50\,\text{eV}$, $\text{clip}(T / T_{\text{pris}}, 0, 1)$ | N/A | 2.072 | 2.895 | Histogram algorithm, tuned with Optuna. |
| **BUILD-04** | 2026-08-17 | 7-AGNR | 4-Way Comparison | Held-out test set (2,100 spectra) | 21 concentrations ($c \in [3, 43]$) | N/A | 0.98 (TF) / 1.18 (MLP) | 1.35 (TF) / 1.53 (MLP) | Verified on freshly generated test spectra. |
| **BUILD-05** | 2026-08-17 | 7-AGNR | MLP + Transformer | `size_7.npy` (170k samples) | `xgb.ipynb` pipeline (150 channels, clip [0, 1]) | N/A | 1.84 (MLP) | 2.51 (MLP) | Consolidated 34 concentrations ($c \le 68$). |
| **BUILD-06 (Completed)** | **2026-08-17** | **7-AGNR & 9-AGNR** | **Multi-Task 4-Way Pipeline** | `size_7.npy` + `size_9.npy` (249k samples) | New base pristine files (`7_agnr_pris.npy`, `9_agnr_pris.npy`), 150 channels, clip [0, 1] | **100.00%** | **1.982 (XGB)<br>2.139 (TF)<br>2.158 (MLP)** | **2.804 (XGB)<br>2.942 (MLP)<br>3.065 (TF)** | **Completed**: Sequential pipeline run (Misfit $\to$ XGB $\to$ MLP $\to$ TF). |

---

## 2. Detailed Final Benchmark Table: BUILD-06 (7-AGNR & 9-AGNR)

Evaluated on **37,350 held-out test configurations** across all 83 concentrations:
- **7-AGNR**: 34 concentrations ($c \in \{2, 4, 6, \ldots, 68\}$)
- **9-AGNR**: 49 concentrations ($c \in \{2, 4, 6, \ldots, 98\}$)

| Model / Method | Width Accuracy (%) | Overall Conc MAE | 7-AGNR Conc MAE | 9-AGNR Conc MAE | Overall Conc RMSE | Max Abs Error | Execution / Eval Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Physical Misfit Baseline** | 99.65% | 2.445 | 2.180 | 2.628 | 3.788 | 34.00 | 18.2s |
| **XGBoost (Hist Gradient Boosting)** | **100.00%** | **1.982** | **1.756** | **2.138** | **2.804** | **18.50** | 27.7s |
| **ConductanceMLP (Multi-Task PINN)** | **100.00%** | 2.158 | 1.917 | 2.324 | 2.942 | 18.95 | 704.6s |
| **Patched Transformer v2 (ConvStem)** | **100.00%** | 2.139 | 1.838 | 2.347 | 3.065 | 21.31 | 3864.8s |

---

## 3. Bug History, Root Causes & Fixes

### Bug #1: Hardcoded Lead Paths in Generation Scripts
* **Symptom**: `FileNotFoundError: /home/shardul/machine_learning/.../leads/agnr_7.npy` when running `generate_test_data.py`.
* **Root Cause**: Scripts contained absolute paths from an older machine configuration.
* **Resolution**: Updated `generate_test_data.py` to resolve paths relative to project root and fallback to Sancho-Rubio decimation if lead files are missing.

### Bug #2: Non-local Operator Matrix Convention Divergence
* **Symptom**: Numerical discrepancy between training data generated via `ca_agnr.py` and test data generated via `agnr.py`.
* **Root Cause**: Training script used $G_{\text{nonlocal}} = g_L \cdot \rho \cdot I_L$ (`"IL"`) with broadening $d = 1\times 10^{-5}$, whereas older test scripts used $G_{\text{nonlocal}} = g_R \cdot T^\dagger \cdot I_L$ (`"IR"`) with $d = 1\times 10^{-4}$.
* **Resolution**: Consolidated all physics functions into [`agnr_lib.py`](file:///run/media/shardul/storage/machine_learning/transmission_github/transmissions/notebooks/agnr/agnr_lib.py) with explicit `nonlocal_mode` parameter, standardized to $d = 1\times 10^{-5}$ and `"IL"`.

### Bug #3: Pristine Calculation Discrepancy (Resolved)
* **Symptom**: Earlier calculated pristine files had conductance scaling mismatch near the band gap edge.
* **Root Cause**: Old pristine generator had an index shift in the lead Green's function surface coupling.
* **Resolution**: User regenerated correct pristine files via [`notebooks/agnr/test.ipynb`](file:///run/media/shardul/storage/machine_learning/transmission_github/transmissions/notebooks/agnr/test.ipynb) and saved in base directory:
  - [**`7_agnr_pris.npy`**](file:///run/media/shardul/storage/machine_learning/transmission_github/transmissions/7_agnr_pris.npy): shape `(300,)`, max = $3.00\,G_0$, onset index = 24 ($E = 0.24\,\text{eV}$).
  - [**`9_agnr_pris.npy`**](file:///run/media/shardul/storage/machine_learning/transmission_github/transmissions/9_agnr_pris.npy): shape `(300,)`, max = $4.00\,G_0$, onset index = 18 ($E = 0.18\,\text{eV}$).
  - Synced into `data/raw/transmission_results/` and linked into all training scripts.

---

## 4. Generated Artifacts & Visualizations

The following figure artifacts and checkpoint files were saved in [`notebooks/agnr/`](file:///run/media/shardul/storage/machine_learning/transmission_github/transmissions/notebooks/agnr):
- **Model Checkpoints**:
  - `multi_width_pinn_mlp.pt` (ConductanceMLP weights)
  - `multi_width_transformer.pt` (PatchedTransformerV2 weights)
  - `multi_width_xgb_width.json` (XGBoost Width Classifier)
  - `multi_width_xgb_conc.json` (XGBoost Concentration Regressor)
  - `multi_width_metrics.json` (Full numerical JSON metrics)
- **Plots**:
  - `multi_width_scatter.png`: 4-way predicted vs true scatter plot for 7-AGNR ($c \le 68$) and 9-AGNR ($c \le 98$).
  - `multi_width_error_dist.png`: Error distribution density comparisons ($\hat{c} - c$).
  - `multi_width_training_curves.png`: Training loss and validation accuracy curves.
