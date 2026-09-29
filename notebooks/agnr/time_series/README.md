# AGNR Quantum Transport — Spectral Sequence Continuation & Extrapolation

This subfolder frames quantum transmission spectrum extrapolation as a **sequence prediction / time-series continuation** problem.

---

## 1. Overview & Formulation

Given a partial low-energy transmission signature $T(E)$ for $E = 0$–$1.49\,\text{eV}$ (channels $0$ to $149$; the grid starts at $E = 0$ in $0.01$ eV steps), the objective is to predict/extrapolate the higher-energy subband transmission behavior $T(E)$ for $E = 1.50$–$1.69\,\text{eV}$ (channels $150$ to $169$).

```
                      ┌──────────────────────────────────────────────┐
                      │ Input: Channels 0–149 (E ≤ 1.50 eV)          │
                      │ T(E) / T_pris(E) ∈ [0, 1]                    │
                      └──────────────────────┬───────────────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       │                                           │
                       ▼                                           ▼
             [LightGBM MultiOutput]                      [Deep MultiOutput MLP]
             MultiOutputRegressor                        PyTorch (150→256→256→128→20)
             `time_series.ipynb`                         `time_series_nn.ipynb`
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │ Output: Channels 150–169 (1.50 < E ≤ 1.70 eV)│
                      │ Extrapolated High-Energy Conductance         │
                      └──────────────────────────────────────────────┘
```

---

## 2. Notebooks

| Notebook | Description | Key Components |
|---|---|---|
| [`time_series.ipynb`](time_series.ipynb) | Exploratory sequence continuation & tree-based baselines | ARIMA analysis, TimeSeriesSplit, MultiOutput LightGBM regressor |
| [`time_series_nn.ipynb`](time_series_nn.ipynb) | Deep Neural Network multi-channel extrapolation | 4-layer PyTorch MLP (`mulit_prediction`), ReduceLROnPlateau, MPS acceleration, Val MSE = 0.0222 (random split) |
| [`build08_baselines.py`](build08_baselines.py) | Comparable baselines on identical normalised inputs, held-out configuration seeds (80/20) | Results in `build08_baselines.json` (below) |

### Comparable results (`build08_baselines.py`, 464,000 train / 116,000 test spectra)

| Model | MSE | RMSE | MAE | Train time |
|---|---|---|---|---|
| LightGBM (one model per channel, 1,000 trees) | **0.0213** | 0.146 | 0.107 | 211 s |
| 4-layer MLP (as in the notebook) | 0.0218 | 0.148 | 0.108 | 604 s |
| Persistence (repeat the last input channel) | 0.0477 | 0.218 | 0.163 | — |

The notebook's LightGBM figure (0.1073) is an MAE printed as "RMSE" and computed on raw transmission, so it is not comparable with the MLP's normalised MSE.

---

## 3. Data Processing & Normalization

1. **Input Data**: Consolidated 7-AGNR and 9-AGNR datasets (`size_7.npy` and `size_9.npy`).
2. **Standardization**:
   $$X_7 = \text{clip}\left(\frac{T_7(E)}{T_{\text{pris},7}(E)}, 0, 1\right), \quad X_9 = \text{clip}\left(\frac{T_9(E)}{T_{\text{pris},9}(E)}, 0, 1\right)$$
3. **Dataset Scale**: 580,000 total spectra (29 concentrations $\times$ 10,000 configurations $\times$ 2 ribbon widths).
