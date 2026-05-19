# Quantum Transport & Impurity Concentration Prediction

A research project for simulating electron transmission through graphene-based quantum devices and predicting impurity concentrations using a **Physics-Informed MLP (PINN)**.

## Overview

This project solves the **inverse problem** in quantum transport: given an electron transmission spectrum $T(E)$ of a graphene nanoribbon with randomly distributed impurities, predict the impurity concentration.

The forward problem — computing $T(E)$ from a known impurity configuration — is solved via the recursive Green's function method (Landauer–Büttiker formalism). The inverse problem is tackled with a 1D Convolutional Neural Network regularised by a differentiable physics-informed loss.

### Physical Systems

Three lattice geometries are supported:

| System | Description | Unit Cell Size |
|--------|-------------|----------------|
| **Square Lattice** | 2D tight-binding square lattice | $n \times n$ |
| **AGNR** (Armchair GNR) | 7-atom-wide armchair graphene nanoribbon | $2m \times 2m$ with anti-diagonal couplings |
| **ZGNR** (Zigzag GNR) | Zigzag graphene nanoribbon | $n \times n$ with 4-periodic hopping pattern |

---

## Physics-Informed Neural Network (PINN)

The current PINN for the 7-AGNR system uses a `ConductanceMLP` backbone with a curvature-weighted physics regulariser, implemented in [`notebooks/agnr/pinn_agnr_curvature.py`](notebooks/agnr/pinn_agnr_curvature.py).

### Problem Statement

- **Input**: Normalised transmission spectrum $T(E) / T_{\text{pristine}}(E)$, a 200-dimensional vector in $[0, 1]$.
- **Output**: Predicted impurity concentration $\hat{c}$ (scalar, range 3–43).
- **Loss**: $\mathcal{L} = \text{MSE}(\hat{c}, c) + \lambda \cdot \text{CurvatureMisfit}(x, \hat{c})$

### Architecture: `ConductanceMLP`

A fully-connected MLP with LayerNorm replacing the previous ResNet-style 1D CNN:

```
Input [B, 200]
  │
  ▼ Linear(200 → 256) + LayerNorm + ReLU + Dropout(0.2)
  ▼ Linear(256 → 128) + LayerNorm + ReLU + Dropout(0.2)
  ▼ Linear(128 →  64) + LayerNorm + ReLU + Dropout(0.2)
  ▼ Linear( 64 →  32) + LayerNorm + ReLU + Dropout(0.2)
  │
  ▼ Regressor: Linear(32 → 1)
  │
  ▼ Output: predicted concentration ĉ [B, 1]
```

LayerNorm is used instead of BatchNorm throughout to improve training stability and avoid batch-size sensitivity.

### Curvature-Weighted Misfit Regulariser

The `CurvatureMisfit` module (in `pinn_agnr_curvature.py`) enforces **physical consistency** between the input spectrum and the predicted concentration via a curvature-weighted penalty.

**How it works:**

1. Precompute reference spectra $R(c)$ (config-averaged) for $c \in \{3, 5, \ldots, 43\}$, cropped to the spectral region of interest $[20:150]$.
2. For each input $x$, compute the full misfit curve against all references after un-normalising and clipping:
   $$\text{mis}(c_i) = \frac{1}{150} \sum_E \left(x_E - R_{c_i,E}\right)^2$$
3. Estimate the curvature at the misfit minimum using a 3-point finite-difference stencil:
   $$\kappa = \frac{\text{mis}(c_{i-1}) - 2\,\text{mis}(c_i) + \text{mis}(c_{i+1})}{\Delta c^2}$$
4. Use $\kappa$ as an adaptive weight: when the misfit landscape has a sharp, well-defined minimum (high $\kappa$), the physics signal is reliable and the penalty is amplified; when it is flat (low $\kappa$), the penalty is reduced.

This means **gradients flow through both the prediction and the curvature weight**, giving the model a self-calibrating physical prior.

**Temperature parameter** $\tau$: Controls soft-selection sharpness. Default: 2.0.

### Training Pipeline

| Component | Details |
|-----------|---------|
| **Optimizer** | AdamW (lr=1e-3, weight_decay=1e-4) |
| **Scheduler** | Cosine annealing over all epochs |
| **Early stopping** | Patience = 20 epochs on validation loss |
| **Data augmentation** | 2% multiplicative Gaussian noise (training only) |
| **Input normalisation** | $x = \text{clip}(T, 0, T_{\text{pristine}}) / T_{\text{pristine}}$ |
| **Spectral crop** | Energy indices 20–150 (region of significant variation) |
| **Validation split** | 20% of training data |
| **Misfit weight** $\lambda$ | 0.1 (tunable) |

---

## Results: PINN vs. Physical Misfit Baseline

Evaluated on **2100 held-out test spectra** spanning concentrations $c \in \{3, 5, \ldots, 43\}$ (generated via `generate_test_data.py`). Full analysis in [`notebooks/agnr/compare_analysis.ipynb`](notebooks/agnr/compare_analysis.ipynb).

| Method | MAE (impurities) | RMSE (impurities) |
|--------|-----------------|-------------------|
| **Deep Learning MLP (PINN)** | **1.17** | **1.56** |
| Physical Misfit Function | 1.89 | 3.11 |

The PINN achieves a **38% reduction in MAE** and **50% reduction in RMSE** over the purely physics-based misfit baseline.

**Physical Misfit baseline** predicts concentration by argmin of the squared difference between the test spectrum and 21 config-averaged reference spectra — no learned parameters.

---

## Project Structure

```
transmissions/
├── README.md
├── .gitignore
│
├── notebooks/                    # Research notebooks & modules
│   ├── agnr/                     # Armchair GNR (7-AGNR)
│   │   ├── agnr.py               # Core physics: unitcell, Green's functions, transmission
│   │   ├── ca_agnr.py            # Parallel configuration generation & transmission computation
│   │   ├── pinn_agnr.py          # Original PINN module (ResNet CNN)
│   │   ├── pinn_agnr_curvature.py  # Current PINN: ConductanceMLP + CurvatureMisfit
│   │   ├── pinn_agnr_curvature.pt  # Saved model checkpoint
│   │   ├── forward_model.py      # Forward transmission model
│   │   ├── inverse_model.py      # Inverse model utilities
│   │   ├── generate_test_data.py # Test dataset generation script
│   │   ├── sweep_misfit_weight.py# Hyperparameter sweep for misfit weight λ
│   │   ├── compare_analysis.ipynb  # ★ Benchmark: PINN vs. Misfit baseline
│   │   ├── misfit_agnr.ipynb     # Original PINN training notebook
│   │   ├── test_validation_pinn_agnr.ipynb  # PINN validation notebook
│   │   ├── inverse_design.ipynb  # Inverse design experiments
│   │   ├── agnr_leads.ipynb      # Lead surface Green's function computation
│   │   └── manifest_agnr.csv     # Training data manifest
│   │
│   ├── zgnr/                     # Zigzag GNR
│   │   ├── zgnr_leads.py         # Core physics module
│   │   ├── zgnr_pristine.ipynb   # Pristine transmission computation
│   │   ├── zgnr_transmission.ipynb  # Device transmission with impurities
│   │   └── leads_gnr_2.ipynb     # Lead computation
│   │
│   ├── square_lattice/           # 2D square lattice system
│   │   ├── CA.ipynb              # Configuration averaging & ML
│   │   └── device.ipynb          # Lead & device transmission computation
│   │
│   ├── extras/                   # Miscellaneous notebooks
│   │   ├── graph.ipynb
│   │   ├── dist_to_transmission.ipynb
│   │   ├── leads_gnr.ipynb
│   │   ├── model_2.ipynb
│   │   ├── mode_3.ipynb
│   │   └── n_n1.ipynb
│   │
│   └── data/                     # Notebook-level data
│
├── data/
│   ├── raw/transmission_results/ # Training spectra (.npy per config)
│   └── test/transmission_results/# Held-out test spectra (2100 files)
├── leads/                        # Precomputed lead surface Green's functions
│   └── agnr_7.npy                # 7-AGNR leads (300 energy points)
├── models/
│   ├── trained/                  # Saved model weights
│   │   └── pinn_agnr.pth
│   └── checkpoints/              # Training checkpoints
│
├── src/                          # Source library (under development)
│   ├── physics/                  # Tight-binding Hamiltonians, Green's functions
│   ├── models/                   # ML model definitions
│   ├── train/                    # Training loops
│   ├── data/                     # Dataset utilities
│   └── utils/                    # Helpers
│
├── scripts/                      # CLI scripts
└── tests/                        # Unit tests
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch ≥ 2.0
- NumPy, SciPy, Pandas, Matplotlib

### Install

```bash
pip install torch numpy scipy pandas matplotlib tqdm
```

### Training the AGNR PINN

Use the curvature-based PINN module in `notebooks/agnr/`:

```python
import pinn_agnr_curvature as pinn
import numpy as np

# 1. Build dataset (after running ca_agnr.py to generate data)
dataset = pinn.NormalizedTransmissionsDataset(
    manifest_file='manifest_agnr.csv',
    root_dir='~/machine_learning/transmission_github/transmissions/data/raw/transmission_results',
    pristine=pristine,        # pristine transmission array
    spectrum_length=200,
)

# 2. Build curvature misfit module
misfit = pinn.build_misfit_module(
    pristine=pristine,
    conc_range=np.arange(3, 45, 2),   # concentrations 3–43 (step 2)
    n_sample_configs=100,
    data_dir='data/raw/transmission_results',
)

# 3. Train
model, train_losses, val_losses = pinn.train_pinn(
    dataset=dataset,
    misfit_module=misfit,
    num_epochs=200,
    misfit_weight=0.1,        # λ
)

# 4. Test
preds, labels, results = pinn.test_pinn(model, test_dataset, misfit)
```

A saved checkpoint (`pinn_agnr_curvature.pt`) is included and can be loaded directly for inference — see [`compare_analysis.ipynb`](notebooks/agnr/compare_analysis.ipynb) for a full worked example.

---

## Physics Background

### Recursive Green's Function Method

For a device of $N$ unit cells connected to semi-infinite leads:

1. **Lead surface Green's function** $g_L$: Computed by iterating the Dyson equation until convergence:
   $$G^{(n+1)} = \left(I - g \cdot V \cdot G^{(n)} \cdot V^\dagger\right)^{-1} g$$

2. **Device Green's function**: Built by recursively attaching unit cells (some with impurities) to the left lead.

3. **Transmission** (Landauer formula):
   $$T(E) = \left|\text{Tr}\left[\Gamma_L \cdot G^r \cdot \Gamma_R \cdot G^a - V \cdot A_{LR} \cdot V^\dagger \cdot A_{LR}\right]\right|$$

   where $\Gamma_{L,R} = i(G - G^\dagger)$ are the broadening matrices and $A_{LR}$ is the non-local spectral function.

### Impurity Model

Impurities are modelled as on-site energy modifications to the tight-binding Hamiltonian:
$$H_{ii} \rightarrow H_{ii} - 0.5$$

The **concentration** $c$ is the total number of impurity sites distributed across the 100 unit cells of the device, placed at random positions determined by a seeded RNG for reproducibility.

---

## License

See [LICENSE](LICENSE) for details.
