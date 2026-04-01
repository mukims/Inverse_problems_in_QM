# Quantum Transport & Impurity Concentration Prediction

A research project for simulating electron transmission through graphene-based quantum devices and predicting impurity concentrations using **Physics-Informed Neural Networks (PINNs)**.

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

The PINN for the 7-AGNR system is implemented in [`notebooks/agnr/pinn_agnr.py`](notebooks/agnr/pinn_agnr.py).

### Problem Statement

- **Input**: Normalised transmission spectrum $T(E) / T_{\text{pristine}}(E)$, a 200-dimensional vector in $[0, 1]$.
- **Output**: Predicted impurity concentration $\hat{c}$ (scalar, range 1–49).
- **Loss**: $\mathcal{L} = \text{MSE}(\hat{c}, c) + \lambda \cdot \text{Misfit}(x, \hat{c})$

### Architecture: `ImprovedConductanceCNN`

A ResNet-style 1D CNN designed for spectral regression:

```
Input [B, 200]
  │
  ▼ Unsqueeze → [B, 1, 200]
  │
  ▼ Stem: Conv1d(1 → 32, k=7) + BatchNorm + ReLU + MaxPool(2)   → [B, 32, 100]
  │
  ▼ ResBlock1D(32) + MaxPool(2) + Dropout(0.2)                   → [B, 32, 50]
  ▼ ResBlock1D(32) + MaxPool(2) + Dropout(0.2)                   → [B, 32, 25]
  ▼ ResBlock1D(32) + MaxPool(2) + Dropout(0.2)                   → [B, 32, 12]
  │
  ▼ Projection: Conv1d(32 → 64, k=3) + BatchNorm + ReLU         → [B, 64, 12]
  ▼ AdaptiveAvgPool1d(1)                                         → [B, 64]
  │
  ▼ Regressor: Linear(64 → 32) + ReLU + Dropout + Linear(32 → 1)
  │
  ▼ Output: predicted concentration ĉ [B, 1]
```

Each **ResBlock1D** contains:
```
x → Conv1d(k=3) → BatchNorm → ReLU → Conv1d(k=3) → BatchNorm → (+x) → ReLU
```

### Differentiable Misfit Regulariser

The key innovation over the original implementation. The misfit loss enforces **physical consistency** between the input spectrum and the predicted concentration.

**How it works:**

1. Precompute configuration-averaged reference spectra $R(c)$ for each concentration $c \in \{1, 3, 5, \ldots, 49\}$.
2. Given the model's prediction $\hat{c}$, compute Gaussian-kernel soft weights:
   $$w_i = \text{softmax}\left(-\frac{(\hat{c} - c_i)^2}{\tau}\right)$$
3. Soft-select a reference spectrum: $\hat{R} = \sum_i w_i \cdot R(c_i)$
4. Compute misfit: $\text{Misfit} = \text{MSE}(x, \hat{R})$

Since the weights $w_i$ depend on $\hat{c}$, **gradients flow through the prediction** back into the model — this is what makes it "physics-informed". The original implementation used NumPy for this computation, which broke the computational graph entirely.

**Temperature parameter** $\tau$: Controls the sharpness of concentration selection. Lower $\tau$ → sharper (closer to hard argmin), higher $\tau$ → smoother. Default: 2.0.

### Training Pipeline

| Component | Details |
|-----------|---------|
| **Optimizer** | AdamW (lr=1e-3, weight_decay=1e-4) |
| **Scheduler** | Cosine annealing over all epochs |
| **Early stopping** | Patience = 20 epochs on validation loss |
| **Data augmentation** | 2% multiplicative Gaussian noise (training only) |
| **Input normalisation** | $x = \text{clip}(T, 0, T_{\text{pristine}}) / T_{\text{pristine}}$ |
| **Validation split** | 20% of training data |
| **Misfit weight** $\lambda$ | 0.1 (tunable) |

### Key Improvements Over Original

| Aspect | Original (`misfit_agnr.ipynb`) | Improved (`pinn_agnr.py`) |
|--------|-------------------------------|--------------------------|
| Misfit gradient | ❌ Zero (NumPy breaks graph) | ✅ Full gradient via soft indexing |
| Architecture | Sequential Conv → ReLU → Pool | ResNet-1D with skip connections |
| Normalisation | None | BatchNorm in every block |
| Input | Raw transmission values | $T / T_{\text{pristine}} \in [0, 1]$ |
| Augmentation | None | Multiplicative Gaussian noise |
| Optimizer | Adam | AdamW with weight decay |
| LR schedule | ReduceLROnPlateau | Cosine annealing |
| Early stopping | None | Patience-based |

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
│   │   ├── pinn_agnr.py          # Improved PINN module (model, training, testing)
│   │   ├── misfit_agnr.ipynb     # Original PINN training notebook
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
├── data/                         # Raw & processed datasets
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

From a Jupyter notebook in `notebooks/agnr/`:

```python
import pinn_agnr
import numpy as np

# 1. Build dataset (after running ca_agnr.py to generate data)
dataset = pinn_agnr.NormalizedTransmissionsDataset(
    manifest_file='manifest_agnr.csv',
    root_dir='~/machine_learning/transmission_github/transmissions/data/raw/transmission_results',
    pristine=pristine,        # pristine transmission array
    spectrum_length=200,
)

# 2. Build differentiable misfit module
misfit = pinn_agnr.build_misfit_module(
    ca_fn=ca,                 # configuration-averaging function
    pristine=pristine,
    conc_range=np.arange(1, 50, 2),
)

# 3. Train
model, train_losses, val_losses = pinn_agnr.train_pinn(
    dataset=dataset,
    misfit_module=misfit,
    num_epochs=200,
    misfit_weight=0.1,        # λ
)

# 4. Test
preds, labels, results = pinn_agnr.test_pinn(model, test_dataset, misfit)
```

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
