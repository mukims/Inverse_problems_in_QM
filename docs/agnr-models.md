# Earlier 7/9-AGNR models, defect reconstruction and inference agent

This page keeps the detailed descriptions that used to be in the main README. Every number and its history are in [`LOGBOOK.md`](../LOGBOOK.md).

Energy windows quoted here in "eV" (for example 0–1.5 or 0–3 eV) are in units of the graphene hopping t; t = 2.7 eV, so 0–3 t is 0–8.1 eV.

## Inverse Problems & Machine Learning Suite

```
                                    ┌─► ConductanceMLP (PINN) ───────────► Impurity Concentration ĉ
                                    │
Normalized Spectrum T(E) [B, L]   ──┼─► Patched Transformer (Global Attn) ─► Impurity Concentration ĉ
                                    │
                                    ├─► 1D Inverse ResNet CNN ────────────► 10×10 Defect Distance Matrix
                                    │
                                    └─► XGBoost Regressor ────────────────► Impurity Concentration ĉ
```

### 1. Physics-Informed MLP (PINN with Curvature Misfit)
* **Code**: [`notebooks/agnr/concentration/pinn_agnr_curvature.py`](../notebooks/agnr/concentration/pinn_agnr_curvature.py)
* **Checkpoint**: `pinn_agnr_curvature.pt`
* **Architecture**: Fully connected MLP with `LayerNorm`, `ReLU`, and `Dropout(0.2)`:
  $$\text{Linear}(200 \to 256) \to \text{Linear}(256 \to 128) \to \text{Linear}(128 \to 64) \to \text{Linear}(64 \to 32) \to \text{Linear}(32 \to 1)$$
* **Curvature-Weighted Physical Misfit**:
  $$\mathcal{L} = \text{MSE}(\hat{c}, c) + \lambda \cdot \text{CurvatureMisfit}(x, \hat{c})$$
  Evaluates a finite-difference curvature $\kappa$ over the configuration-averaged reference misfit landscape $\text{mis}(c_i) = \frac{1}{N}\sum_E (x_E - R_{c_i,E})^2$:
  $$\kappa = \frac{\text{mis}(c_{i-1}) - 2\,\text{mis}(c_i) + \text{mis}(c_{i+1})}{\Delta c^2}$$
  When the misfit minimum is sharp (high $\kappa$), the physical constraint is amplified; when flat or ambiguous, the penalty is dynamically tempered.

---

### 2. 1D-Patched Vision Transformer (`PatchedTransformerV2`)
* **Code**: [`notebooks/agnr/concentration/patched_transformer_v2.py`](../notebooks/agnr/concentration/patched_transformer_v2.py) & [`notebooks/agnr/defect_reconstruction/patched_transformer_model.py`](../notebooks/agnr/defect_reconstruction/patched_transformer_model.py)
* **Documentation**: [`notebooks/agnr/docs/patched_transformer_explanation.md`](../notebooks/agnr/docs/patched_transformer_explanation.md)
* **Checkpoint**: `patched_transformer_v2.pt`
* **Motivation**: Standard 1D CNNs have local receptive fields and require deep hierarchies to correlate distant spectral features (e.g. Fano resonance dips and band-edge shifts). The Patched Transformer directly models long-range cross-band correlations using global self-attention.
* **Architecture**:
  1. **ConvStem**: `Conv1d(1 → 16, k=7, s=1, p=3) + GELU` extracts fine-scale local slope and edge transitions.
  2. **1D Patch Embedding**: `Conv1d(16 → 64, k=10, s=10)` projects the sequence into $N=20$ non-overlapping 10-point tokens.
  3. **Learned Positional Embeddings**: Injects absolute energy coordinates into each token.
  4. **Transformer Encoder**: 3 Pre-Norm layers with Multi-Head Attention (4 heads, `GELU`, `DropPath`).
  5. **Regression Head**: Operates on the prepend `[CLS]` token $\to$ `Linear(64 → 32) + GELU` $\to$ `Linear(32 → 1)`.

---

### 3. Spatial Defect Reconstruction (Distance Matrix CNN)
* **Code**: [`notebooks/agnr/defect_reconstruction/inverse_model.py`](../notebooks/agnr/defect_reconstruction/inverse_model.py)
* **Documentation**: [`notebooks/agnr/docs/walkthrough.md`](../notebooks/agnr/docs/walkthrough.md) & [`notebooks/agnr/docs/inverse_model_explanation.md`](../notebooks/agnr/docs/inverse_model_explanation.md)
* **Checkpoint**: `distance_matrix_model.pth`
* **Target Output**: $10 \times 10$ matrix representing spatial distribution of up to 10 impurities:
  - **Diagonal $M[i,i]$**: Transverse site index within the unit cell ($0–13$).
  - **Off-Diagonal $M[i,j]$**: Euclidean distance between defect $i$ and defect $j$: $\sqrt{(\Delta \text{cell})^2 + (\Delta \text{site})^2}$.
  - **Sparsity**: Zero-padded rows/columns encode the exact impurity count.
* **Architecture**: 1D ResNet Encoder with strided convolutions $\to 1\times 1$ Bottleneck $\to$ Transposed Convolutional Decoder with learned positional queries.

---

### 4. Tree-Based Baselines (XGBoost)
* **Code**: [`notebooks/agnr/nb/xgb.ipynb`](../notebooks/agnr/nb/xgb.ipynb) & [`notebooks/agnr/concentration/train_conc_models.py`](../notebooks/agnr/concentration/train_conc_models.py)
* **Architecture & Training**:
  - Uses `xgboost.XGBRegressor` with the hardware-accelerated histogram algorithm (`tree_method="hist"`), `n_estimators=500`, `max_depth=6`, `learning_rate=0.03`, stochastic row subsampling (`0.8`), column subsampling (`0.8`), and $L_2$ regularization (`reg_lambda=1.0`).
  - Explored in `xgb.ipynb` with **Optuna Bayesian optimization** for hyperparameter tuning across both raw 1D transmission curves and statistical feature representations (band-edge onset, spectral variance, mean suppression).
  - Integrated into [`train_conc_models.py`](../notebooks/agnr/concentration/train_conc_models.py) as one of the three primary model backends and queryable by [`agnr_agent.py`](../notebooks/agnr/agent/agnr_agent.py).

---

## Autonomous Inference Agent & RL Framework

```
                          ┌──────────────────────────────────────────────────────────┐
                          │         Unknown Transmission Signature T(E)              │
                          └────────────────────────────┬─────────────────────────────┘
                                                       │
                                 ┌─────────────────────▼─────────────────────┐
                                 │       Step 1: Deterministic Physics       │
                                 │  - tool_band_gap: Extract onset & index   │
                                 │  - tool_rank_widths: Pristine comparison  │
                                 │  - Filter impossible widths               │
                                 └─────────────────────┬─────────────────────┘
                                                       │
                                 ┌─────────────────────▼─────────────────────┐
                                 │       Step 2: Validation & Decision       │
                                 │  - LLM Reasoning (Gemma via Ollama)       │
                                 │    or Deterministic Fallback Logic        │
                                 └─────────────────────┬─────────────────────┘
                                                       │
                                 ┌─────────────────────▼─────────────────────┐
                                 │       Step 3: Concentration Model         │
                                 │  - Dispatch Width-Specific Model          │
                                 │    (MLP / Transformer / XGBoost)          │
                                 └─────────────────────┬─────────────────────┘
                                                       │
                                 ┌─────────────────────▼─────────────────────┐
                                 │       Step 4: Feasibility Assessment      │
                                 │  - Disorder suppression check             │
                                 │  - Model agreement & confidence scoring   │
                                 └───────────────────────────────────────────┘
```

### Inference Agent (`agnr_agent.py`)
- **Code**: [`notebooks/agnr/agent/agnr_agent.py`](../notebooks/agnr/agent/agnr_agent.py)
- **Design**: Separates deterministic physical computation from high-level reasoning. The agent extracts the physical band gap, filters out incompatible pristine geometries from a width library ([`width_id.py`](../notebooks/agnr/agent/width_id.py)), queries the appropriate machine learning model, and flags out-of-domain/unfeasible predictions.

### Reinforcement Learning Scheme (`RL_SCHEME.md`)
- **Specification**: [`notebooks/agnr/agent/RL_SCHEME.md`](../notebooks/agnr/agent/RL_SCHEME.md)
- Casts device identification into a sequential Markov Decision Process (MDP) under an execution budget:
  - **State $s_t$**: Signature summary, band gap probes, candidate width posterior belief $b_t$, and remaining compute budget.
  - **Discrete Actions $a_t$**: `PROBE_GAP(\theta)`, `REJECT_IMPOSSIBLE`, `SIMULATE(m, c)` (high-cost RGF forward call), `CALL_MODEL(k, m)`, `COMMIT(m, c)`, `ABSTAIN`.
  - **Reward**: Strongly penalizes wrong width classification and wasted simulations while rewarding accurate concentration estimates and honest abstention under extreme disorder.

---

## Benchmark Results & Comparative Analysis

### Current benchmark: 7- and 9-AGNR, held-out configuration seeds (BUILD-09)
249,000 spectra (3,000 seeds per concentration; 7-AGNR c = 2–68, 9-AGNR c = 2–98), seeds split 70/15/15, 37,350 test spectra, input E = 0–1.49 eV. Results in [`notebooks/agnr/multi_width/seed_split/`](../notebooks/agnr/multi_width/seed_split/).

| Model | Concentration MAE | 7-AGNR | 9-AGNR | RMSE | Published random-split MAE (BUILD-06) |
|---|---|---|---|---|---|
| **Patched Transformer v2** (width-conditioned) | **2.267** | 1.871 | 2.542 | 3.274 | 1.390 |
| **XGBoost** | 2.394 | 2.016 | 2.656 | 3.432 | 1.982 |
| **Physics misfit baseline** (no learning) | 2.442 | 1.985 | 2.760 | 3.834 | 2.445 |
| **Multi-task MLP** | 2.884 | 2.454 | 3.182 | 3.673 | 2.018 |

**With the full 0–3 eV spectrum** ([`energy_window_check.py`](../notebooks/agnr/multi_width/energy_window_check.py), same XGBoost and test set): MAE 2.396 (0–1.5 eV) → **1.977** (0–3 eV); the upper half alone gives 2.769. Training on all 10,000 seeds instead of 3,000 improves XGBoost only from 2.387 to 2.330 on a common test set.

> **Key takeaway**: once near-duplicate spectra are kept out of the test set, all learned models land close together and only the transformer clearly beats the no-learning physics baseline. Giving models more of the spectrum helps more than changing the model.

### Other results (all held-out seeds)
| Study | Result | Where |
|---|---|---|
| Universal multi-task transformer (BUILD-10), 7-AGNR + 9-AGNR + Square-10 | Concentration MAE 1.855 / 2.551 / 3.126 (type and width accuracy is circular, Bug #8) | [`notebooks/universal_transformer/`](../notebooks/universal_transformer/) |
| Square-10 autoencoder | Concentration from the 64-d latent: MAE 3.58 (PCA-64: 3.17, raw spectrum: 3.22) | [`notebooks/square_lattice/sq_ae_results/`](../notebooks/square_lattice/sq_ae_results/) |
| Joint autoencoder, all three systems | Reconstruction R² 0.948 / 0.949 / 0.911; concentration MAE 2.12 / 2.80 / 3.65 (PCA-64: 1.90 / 2.58 / 3.18) | [`notebooks/joint_autoencoder/results/`](../notebooks/joint_autoencoder/results/) |
| Spectral continuation (BUILD-08) | Predict 1.50–1.69 eV from 0–1.49 eV: LightGBM MSE 0.0213, MLP 0.0218, persistence 0.0477 | [`notebooks/agnr/time_series/build08_baselines.json`](../notebooks/agnr/time_series/build08_baselines.json) |

### Historical benchmark (BUILD-04; not comparable)
2,100 freshly generated 7-AGNR test spectra, $c \in \{3, 5, \ldots, 43\}$ ([`generate_test_data.py`](../notebooks/agnr/physics/generate_test_data.py), [`compare_all_models.py`](../notebooks/agnr/concentration/compare_all_models.py)): Patched Transformer v2 MAE 0.98, ConductanceMLP (PINN) 1.18, physics misfit 1.92, XGBoost 2.07. The narrower concentration range and single width make these numbers much lower than the current benchmark.
