# Quantum Transport ML Experiment Logbook & Build Tracker

Welcome to the project **Logbook**. This document serves as the single source of truth for tracking all model builds, data cleaning iterations, bug fixes, architecture hyperparameters, benchmark metrics, and spectral sequence continuation across the AGNR (7-AGNR & 9-AGNR), ZGNR, and square lattice quantum transport inverse problem pipelines.

---

## 1. Experiment & Build Registry

| Build ID | Date | Target System | Architectures / Models | Dataset & Samples | Data Cleaning / Normalization | Width Acc (%) | Conc MAE | Conc RMSE | Status & Notes |
|:---|:---|:---|:---|:---|:---|:---:|:---:|:---:|:---|
| **BUILD-01** | *Initial* | 7-AGNR | ConductanceMLP (PINN) | `data/raw/` (2,100 samples) | $T(E) / T_{\text{pris}}(E)$ with CurvatureMisfit | N/A (7 only) | 1.176 | 1.528 | Baseline PINN architecture established with physical misfit penalty. |
| **BUILD-02** | *Initial* | 7-AGNR | Patched Transformer v2 | `data/raw/` (2,100 samples) | $T(E) / T_{\text{pris}}(E)$ + ConvStem | N/A (7 only) | 0.978 | 1.348 | 1D Self-attention with ConvStem and [CLS] token. |
| **BUILD-03** | *Optuna* | 7-AGNR & 9-AGNR | XGBoost Regressor | Consolidated stacks (`xgb.ipynb`) | $E \le 1.50\,\text{eV}$, $\text{clip}(T / T_{\text{pris}}, 0, 1)$ | N/A | 2.072 | 2.895 | Histogram gradient boosting algorithm, tuned with Optuna. |
| **BUILD-04** | 2026-08-17 | 7-AGNR | 4-Way Comparison | Held-out test set (2,100 spectra) | 21 concentrations ($c \in [3, 43]$) | N/A | 0.98 (TF) / 1.18 (MLP) | 1.35 (TF) / 1.53 (MLP) | Verified on freshly generated test spectra. |
| **BUILD-05** | 2026-08-17 | 7-AGNR | MLP + Transformer | `size_7.npy` (170k samples) | `xgb.ipynb` pipeline (150 channels, clip [0, 1]) | N/A | 1.84 (MLP) | 2.51 (MLP) | Consolidated 34 concentrations ($c \le 68$). |
| **BUILD-06** | 2026-08-17 | **7-AGNR & 9-AGNR** | **Multi-Task 4-Way Pipeline** | `size_7.npy` + `size_9.npy` (249k samples) | Base pristine files (`7_agnr_pris.npy`, `9_agnr_pris.npy`), 150 channels, clip [0, 1] | **100.00%** | **1.390 (TF)<br>1.982 (XGB)<br>2.018 (MLP)** | **1.977 (TF)<br>2.804 (XGB)<br>2.680 (MLP)** | **Completed**: Multi-width pipeline with Width-Conditioned Patched Transformer v2 achieving state-of-the-art accuracy. |
| **BUILD-07** | 2026-08-20 | 7-AGNR & 9-AGNR | Bayesian Optimization Sweep | `manifest_agnr.csv` / consolidated sets | Physical Curvature Misfit + Loss weighting sweep | **100.00%** | **1.563 (BO-PINN)** | **2.196 (BO-PINN)** | **Completed**: Optuna Bayesian hyperparameter search (misfit weight $\lambda = 0.00286$, lr = $1.98\times 10^{-4}$, dropout = 0.07). |
| **BUILD-08** | 2026-08-25 | 7-AGNR & 9-AGNR | Spectral Sequence Continuation (Time Series NN) | Combined `size_7.npy` & `size_9.npy` (580,000 samples) | Sequence mapping: 150 low-energy channels ($E \le 1.50\,\text{eV}$) $\to$ 20 high-energy channels ($E \in [1.50, 1.70]\,\text{eV}$) normalized to $[0, 1]$ | N/A | **Val MSE: 0.0222** | **Val RMSE: 0.149** | **Active**: MultiOutput LightGBM baseline & 4-layer PyTorch MLP (`mulit_prediction`) with ReduceLROnPlateau, MPS acceleration. |

---

## 2. Detailed Final Benchmark Tables

### A. Multi-Width Inverse Characterization Benchmark (BUILD-06)
Evaluated on **37,350 held-out test configurations** across all 83 concentrations:
- **7-AGNR**: 34 concentrations ($c \in \{2, 4, 6, \ldots, 68\}$)
- **9-AGNR**: 49 concentrations ($c \in \{2, 4, 6, \ldots, 98\}$)

| Model / Method | Width Accuracy (%) | Overall Conc MAE | 7-AGNR Conc MAE | 9-AGNR Conc MAE | Overall Conc RMSE | Max Abs Error | Train / Eval Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Physical Misfit Baseline** | 99.65% | 2.445 | 2.008 | 2.748 | 3.788 | 34.00 | 0.6s eval |
| **XGBoost (Hist Gradient Boosting)** | **100.00%** | 1.982 | 1.756 | 2.138 | 2.804 | 18.50 | 34.3s train |
| **ConductanceMLP (Multi-Task PINN)** | **100.00%** | 2.018 | 1.882 | 2.112 | 2.680 | 17.22 | 902.5s train |
| **Patched Transformer v2 (Width-Conditioned)** | **100.00%** | **1.390** | **1.319** | **1.440** | **1.977** | **16.09** | 8003.9s train |

---

### B. Bayesian Optimization Hyperparameter Sweep (BUILD-07)
Hyperparameter search conducted via Optuna with SQLite backend (`optuna_study.db`):
- **Objective**: Minimize validation Concentration MAE using physics-informed curvature misfit loss.
- **Optimal Hyperparameters**:
  - `lr`: $1.976 \times 10^{-4}$
  - `misfit_weight` ($\lambda$): $0.002859$
  - `temperature`: $1.6738$
  - `dropout`: $0.0697$
  - `noise_std`: $0.0292$
  - `weight_decay`: $2.920 \times 10^{-5}$
  - `hidden_arch`: `standard`

| Evaluation Metric | BO-Tuned PINN Model | Raw Misfit Baseline | Relative Improvement |
|:---|:---:|:---:|:---:|
| **Test Conc MAE** | **1.563** | 2.102 | **+25.6% lower error** |
| **Test Conc RMSE** | **2.196** | 3.469 | **+36.7% lower error** |
| **Best Trial Val MAE** | **1.634** | — | Trial #2 |

---

### C. Spectral Sequence Continuation & Extrapolation (BUILD-08)
Framing quantum transmission spectrum prediction as an autoregressive / sequence extrapolation problem:
- **Input Feature Vector**: 150 low-energy transmission channels ($E \in [0.01, 1.50]\,\text{eV}$).
- **Target Feature Vector**: Next 20 high-energy transmission channels ($E \in [1.51, 1.70]\,\text{eV}$).
- **Total Dataset Size**: 580,000 samples (29 concentrations $\times$ 10,000 configurations $\times$ 2 geometries [7-AGNR & 9-AGNR]).
- **Split**: 80% train (464,000 samples), 20% test (116,000 samples).

| Model / Architecture | Loss Function | Optimizer & Scheduler | Hardware Device | Best Val Loss (MSE) | Key Observations |
|:---|:---|:---|:---|:---:|:---|
| **LightGBM MultiOutput** | Multi-target MSE | Hist Gradient Boosting | CPU (Multi-threaded) | Baseline | Fast multi-channel baseline for high-energy step extrapolation. |
| **Deep MultiOutput MLP (`mulit_prediction`)** | $\text{MSE}(\hat{Y}, Y)$ | Adam (lr=0.01) + `ReduceLROnPlateau(factor=0.5, patience=3)` | Apple Silicon (MPS) / CUDA | **0.0222** | 4-layer fully connected network ($150 \to 256 \to 256 \to 128 \to 20$) with ReLU activations; achieved rapid convergence within 100 epochs. |

---

## 3. Bug History, Architectural Evolutions & Root Cause Fixes

### Bug #1: Hardcoded Lead Paths in Generation Scripts
* **Symptom**: `FileNotFoundError: /home/shardul/machine_learning/.../leads/agnr_7.npy` when running `generate_test_data.py`.
* **Root Cause**: Scripts contained absolute paths from an older machine configuration.
* **Resolution**: Updated `generate_test_data.py` to resolve paths relative to project root and fallback to Sancho-Rubio decimation if lead files are missing.

### Bug #2: Non-local Operator Matrix Convention Divergence
* **Symptom**: Numerical discrepancy between training data generated via `ca_agnr.py` and test data generated via `agnr.py`.
* **Root Cause**: Training script used $G_{\text{nonlocal}} = g_L \cdot \rho \cdot I_L$ (`"IL"`) with broadening $d = 1\times 10^{-5}$, whereas older test scripts used $G_{\text{nonlocal}} = g_R \cdot T^\dagger \cdot I_L$ (`"IR"`) with $d = 1\times 10^{-4}$.
* **Resolution**: Consolidated all physics functions into [`notebooks/agnr/physics/agnr_lib.py`](notebooks/agnr/physics/agnr_lib.py) with explicit `nonlocal_mode` parameter, standardized to $d = 1\times 10^{-5}$ and `"IL"`.

### Bug #3: Pristine Calculation Discrepancy (Resolved)
* **Symptom**: Earlier calculated pristine files had conductance scaling mismatch near the band gap edge.
* **Root Cause**: Old pristine generator had an index shift in the lead Green's function surface coupling.
* **Resolution**: Regenerated correct pristine reference files via [`notebooks/agnr/nb/test.ipynb`](notebooks/agnr/nb/test.ipynb) and saved in the repository root:
  - **`7_agnr_pris.npy`**: shape `(300,)`, max = $3.00\,G_0$, onset index = 24 ($E = 0.24\,\text{eV}$).
  - **`9_agnr_pris.npy`**: shape `(300,)`, max = $4.00\,G_0$, onset index = 18 ($E = 0.18\,\text{eV}$).
  - Synced into data directories and referenced consistently in all model pipelines.

### Bug #4: Cross-Geometry Transmission Bounds & Normalization Mismatch
* **Symptom**: Discontinuity and unbounded variance when training joint 7-AGNR and 9-AGNR sequence extrapolation models.
* **Root Cause**: 7-AGNR has maximum conductance of $3.0\,G_0$ while 9-AGNR has maximum conductance of $4.0\,G_0$. Directly stacking raw transmission matrices skewed gradient updates toward 9-AGNR channels.
* **Resolution**: Standardized normalization in `time_series.ipynb` and `time_series_nn.ipynb` by dividing each ribbon's transmission spectrum by its exact pristine counterpart before clipping:
  $$X_7 = \text{clip}\left(\frac{T_7(E)}{T_{\text{pris},7}(E)}, 0, 1\right), \quad X_9 = \text{clip}\left(\frac{T_9(E)}{T_{\text{pris},9}(E)}, 0, 1\right)$$
  This guarantees unified $[0, 1]$ bounds across all arbitrary nanoribbon widths and energy subbands.

### Bug #5: Hardware-Agnostic Device Dispatch for PyTorch Models
* **Symptom**: `RuntimeError: Expected all tensors to be on the same device` or CPU fallback during training on Apple Silicon.
* **Root Cause**: Hardcoded `cuda` checks failed to detect macOS Metal Performance Shaders (MPS).
* **Resolution**: Standardized device dispatch across all scripts and notebooks:
  ```python
  dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
  ```

---

## 4. Generated Artifacts & Visualizations

### Model Checkpoints & Serialization
- **Multi-Width Suite** ([`notebooks/agnr/multi_width/`](notebooks/agnr/multi_width/)):
  - `mw_xgb_width.json` (XGBoost Width Classifier)
  - `mw_xgb_conc.json` (XGBoost Concentration Regressor)
  - `mw_results/mw_all_metrics.json` (Comprehensive numerical benchmark metrics)
  - `mw_results/xgboost_preds.npz`, `mlp_preds.npz`, `transformer_preds.npz`, `misfit_preds.npz`
- **Bayesian Sweeps** ([`notebooks/agnr/sweeps/bo_sweep_results/`](notebooks/agnr/sweeps/bo_sweep_results/)):
  - `optuna_study.db` (SQLite study database)
  - `bo_summary.json` (Top trials, optimal parameters, and test set evaluations)

### Benchmark Figures & Visual Diagnostics
- **Multi-Width Model Comparison**:
  - `notebooks/agnr/multi_width/mw_scatter.png`: 4-way predicted vs true scatter plot for 7-AGNR ($c \le 68$) and 9-AGNR ($c \le 98$).
  - `notebooks/agnr/multi_width/mw_error_dist.png`: Error distribution density comparisons ($\hat{c} - c$).
  - `notebooks/agnr/multi_width/mw_training_curves.png`: Training loss and validation accuracy trajectories.
- **Bayesian Optimization Visualizations**:
  - `notebooks/agnr/sweeps/bo_sweep_results/bo_optimisation_history.png`: Convergence trajectory over trials.
  - `notebooks/agnr/sweeps/bo_sweep_results/bo_param_importance.png`: Hyperparameter importance ranking.
  - `notebooks/agnr/sweeps/bo_sweep_results/bo_parallel_coordinates.png`: Multi-dimensional parameter space exploration.
  - `notebooks/agnr/sweeps/bo_sweep_results/bo_test_scatter.png`: Scatter comparison of BO-tuned model vs physical misfit.
- **Concentration Analysis**:
  - `notebooks/agnr/concentration/compare_scatter.png`: Single-width 4-way scatter evaluation.
  - `notebooks/agnr/concentration/compare_error_dist.png`: Residual density distribution.
  - `notebooks/agnr/concentration/compare_per_conc.png`: MAE error breakdown per concentration level.
  - `notebooks/agnr/concentration/compare_training.png`: Training and validation loss curves.

---
*Logbook maintained by the Quantum Transport & Inverse Problems Research Group.*
