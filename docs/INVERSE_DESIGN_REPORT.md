# Inverse Design Knowledge Base

**Last Updated:** November 2026  
**Active Branch:** `feat/inverse-design` (HEAD: `d458e5a0d`, synchronized with `origin/main`)  
**Design Reference:** [`docs/superpowers/specs/2026-10-23-inverse-design-design.md`](superpowers/specs/2026-10-23-inverse-design-design.md)  
**Detailed Review:** [`docs/superpowers/plans/2026-10-28-model-v1-review.md`](superpowers/plans/2026-10-28-model-v1-review.md)  
**Interactive Explainer:** [`docs/inverse_design_explainer.html`](inverse_design_explainer.html)  
**Later Designs:** [`reward fine-tuning`](superpowers/specs/2026-11-04-reward-finetuning-design.md), [`active learning`](superpowers/specs/2026-11-04-active-learning-design.md)  
**Outcomes of every run:** [`LOGBOOK.md`](../LOGBOOK.md) (entries `INV-*`)  

---

## 1. Executive Summary & Purpose

Earlier stages of the project (Shazam, `src/atlaslib/lookup.py`) matched an unknown transmission spectrum $T(E)$ to the nearest *catalogued* device. The catalogue covers graphene, silicene, germanene, hBN, MoS₂, phosphorene, square and triangular ribbons; lookup can only return one of its entries, so a spectrum from an uncatalogued material is necessarily assigned to a wrong one.

**Inverse Design** replaces "nearest catalogued device" with **"most probable tight-binding host"** across a continuous space of tight-binding Hamiltonians:
1. **Input**: A target disordered transmission spectrum $T(E)$ with 416 energy channels ($E \in [0, 8.3]\,\text{eV}$, $\Delta E = 0.02\,\text{eV}$, InputSpec v4).
2. **Generative Model (`InverseModel`)**: Predicts the discrete lattice/edge configuration and continuous tight-binding parameters (hoppings, on-site potentials, ribbon width $w \in [4, 22]$, and defect concentration $c$).
3. **Clustering (`candidates.py`)**: Draws 1,000 posterior samples and aggregates them into up to 5 distinct candidate hosts with calibrated probabilities.
4. **Physical Verification (`verify.py` & `refine.py`)**: Simulates 50 fresh disorder seeds using recursive Green's functions (RGF). Recreates the target spectrum if the misfit falls within the 95th percentile spread of the ensemble; refines host parameters via trust-region optimization if near the threshold.
5. **Bayesian Update (`posterior.py`)**: Combines generative prior belief with simulation likelihood and unassigned background mass.

---

## 2. Codebase Architecture & File Map

### Core Pipeline (`src/shazam/inverse/`)
| Module | Path | Key Functions & Responsibilities |
| :--- | :--- | :--- |
| **`model.py`** | [`src/shazam/inverse/model.py`](file:///run/media/shardul/storage/machine_learning/transmission_github/transmissions/src/shazam/inverse/model.py) | Defines `InverseModel`, `make_encoder`, and `run_encoder`. Supports CNN and MLP backbones, extra stride-1 conv depth, GroupNorm, discrete classifier head (156 states), and continuous normalizing flow head (14 slots). |
| **`encoding.py`** | [`src/shazam/inverse/encoding.py`](file:///run/media/shardul/storage/machine_learning/transmission_github/transmissions/src/shazam/inverse/encoding.py) | Encodes/decodes prior draws to vectors. Handles 156 discrete states (pattern $\times$ lead gauge $\times$ switches), active slot masks, and canonical parameter bounds. |
| **`candidates.py`** | [`src/shazam/inverse/candidates.py`](file:///run/media/shardul/storage/machine_learning/transmission_github/transmissions/src/shazam/inverse/candidates.py) | Clusters 1,000 posterior samples into at most 5 `Candidate` objects. Computes 90% marginal intervals, Mahalanobis distances, and probability shares. |
| **`verify.py`** | [`src/shazam/inverse/verify.py`](file:///run/media/shardul/storage/machine_learning/transmission_github/transmissions/src/shazam/inverse/verify.py) | RGF recreation check across 50 fresh seeds. Computes clean + disordered transmission envelopes, misfit ranks, and simulation log-likelihoods. |
| **`refine.py`** | [`src/shazam/inverse/refine.py`](file:///run/media/shardul/storage/machine_learning/transmission_github/transmissions/src/shazam/inverse/refine.py) | Local trust-region optimization of candidate continuous parameters when initial simulation check marginally misses. |
| **`posterior.py`** | [`src/shazam/inverse/posterior.py`](file:///run/media/shardul/storage/machine_learning/transmission_github/transmissions/src/shazam/inverse/posterior.py) | Bayesian updating of candidate probabilities: combines prior weights, simulation likelihoods, and background mass ($p_0$). |
| **`design.py`** | [`src/shazam/inverse/design.py`](file:///run/media/shardul/storage/machine_learning/transmission_github/transmissions/src/shazam/inverse/design.py) | End-to-end `inverse_design(spectrum)` entry point. Generates final JSON report containing candidates, verification verdicts, and updated probabilities. |
| **`calibration.py`** | [`src/shazam/inverse/calibration.py`](file:///run/media/shardul/storage/machine_learning/transmission_github/transmissions/src/shazam/inverse/calibration.py) | Evaluation metrics: expected calibration error (ECE), 90% coverage gate, Simulation-Based Calibration (SBC) rank tests. |
| **`autoencoder.py`** | [`src/shazam/inverse/autoencoder.py`](file:///run/media/shardul/storage/machine_learning/transmission_github/transmissions/src/shazam/inverse/autoencoder.py) | Standalone `SpectrumAutoencoder` supporting clean pristine reconstruction from disordered spectra with identical encoder interfaces. |
| **`data.py`** | [`src/shazam/inverse/data.py`](../src/shazam/inverse/data.py) | Shard loading (`load_split`), the seed split (`split_mask`), draws from stored arrays, data provenance. |
| **`pattern.py`** | [`src/shazam/inverse/pattern.py`](../src/shazam/inverse/pattern.py) | `PatternModel`: lattice-type-only classifier (the band-top CNN encoder and a 7-way head over C, H, K, C+H, C+K, H+K, C+H+K), optional dropout; `pattern_labels` maps stored `on` rows to the 7 patterns. |
| **`reward.py`** | [`src/shazam/inverse/reward.py`](../src/shazam/inverse/reward.py) | Truth rewards for phase-1 fine-tuning: label and physics cost tables over the 156 states, flow rewards. |
| **`sim_reward.py`** | [`src/shazam/inverse/sim_reward.py`](../src/shazam/inverse/sim_reward.py) | Simulator reward for phase-2 fine-tuning: simulate a candidate with a fresh disorder seed and score how well it recreates the target. |
| **`active.py`** | [`src/shazam/inverse/active.py`](../src/shazam/inverse/active.py) | Active learning: cell assignment, empirical-Bayes shrinkage, expected gain, the tilted proposal sampler and the pooled importance weights. Pure NumPy. |
| **`runio.py`** | [`src/shazam/inverse/runio.py`](../src/shazam/inverse/runio.py) | Atomic checkpoint writes and the run's code version. |

### Supporting Packages & Scripts
- **Physics Engine (`src/tbribbon/`)**:
  - `parent.py`: Hexagonal parent Hamiltonian builder spanning site types C (centre), H (honeycomb A/B), and K (kagome midpoints).
  - `parent_data.py`: Preprocessing transforms (`to_input_linear`, `to_input_bandtop`, `pristine_target_bandtop`).
  - `leads.py` & `transport.py`: Causal decimation with `LeadCache` energy-shift retries (`1j * 1e-9`) preventing Sancho-Rubio NaNs.
- **Execution Scripts (`scripts/`)**:
  - `scripts/build/train_inverse.py`: Inverse model trainer with AdamW, Cosine/Plateau scheduling, and feature flags (`--encoder`, `--input`, `--extra-conv`, `--pool`, `--embed` for the encoder output width, `--seed`, `--max-draws`, `--extra-data` for importance-weighted active-learning rows, `--drop-pattern` to leave a lattice type out of train and val).
  - `scripts/build/train_pattern.py`: Trains the lattice-type-only classifier (`--dropout`, `--weight-decay`, `--label-smoothing`, `--noise`).
  - `scripts/build/finetune_inverse.py`, `finetune_simulator.py`, `reward_tables.py`: Reward fine-tuning, phase 1 (truth reward), phase 2 (simulator reward) and their cost tables.
  - `scripts/build/acquire_round.py`: Active learning, steps 0-1: per-cell diagnosis and the round's tilted proposal (`cells_round{k}.json`). No simulation.
  - `scripts/build/train_autoencoder.py`: Pretrains denoising autoencoder (`ae_cnn`, `ae_mlp`).
  - `scripts/evaluate/compare_inverse.py`: Side-by-side McNemar statistical comparison of models across 15,000 validation draws.
  - `scripts/evaluate/eval_inverse.py`: Benchmarks against hidden catalogue materials and synthetic test draws.
  - `scripts/evaluate/probe_latent.py`: Linear/logistic probing of encoder representations.
  - `scripts/evaluate/eval_pattern.py`: Lattice-type accuracy, top-2, ECE, confusion matrix, 156-state top-1 and McNemar for lattice-type and full models; `--drop-pattern` scores without a lattice type.
  - `scripts/evaluate/eval_active_round.py`: Targets and guards of an active-learning round on the shared val split. No simulation.
  - `scripts/evaluate/compare_reward.py`: The reward-run comparison table for any set of models.
  - `scripts/evaluate/design_from_spectrum.py`: Design a host from one target spectrum (report output).
  - `scripts/generate/generate_parent.py`: Multi-worker dataset generator for the hexagonal parent; `--proposal` draws from an active-learning proposal and `--index-start` gives new rounds disjoint draw indices.
  - `scripts/generate/check_parent.py` & `report_parent.py`: Validation checks and cost/rejection reporting.

---

## 3. Data Generation Pipeline (`hexagonal_parent_v1`)

The training set is generated under `$TRANSMISSION_DATA/_generated/hexagonal_parent_v1/`.

### Prior & Constraints
- **Site Types**: C (triangular), H (honeycomb), K (kagome). 7 discrete patterns: `C`, `H`, `K`, `C+H`, `C+K`, `H+K`, `C+H+K`.
- **Hopping Shells**: Identified strictly by `(type pair, distance)`.
- **Edge & Width**: Zigzag and Armchair edges; ribbon widths $w \in [4, 22]$ cells; device lengths scaled to $\sim 2,000$ atoms.
- **Disorder**: Random on-site shifts on active atoms with $V = 0.5\,s$.
- **Rejection Rules**:
  1. Invisible active type: Active site carries $< 1\%$ bulk band weight in $[0, 8.3]\,\text{eV}$.
  2. Zero clean channels across the entire grid.
  3. Fewer than 10 non-zero disordered channels after rounding.
  - Rejection counts per step are in the `full_step*_report.json` files and `INV-FULL`.

### Milestone Steps
- **Steps 1–4 (Draws 0 to 299,999)**: Completed in 104 hours wall time on 22 workers. Generated baseline dataset for `model_v1`.
- **Steps 5–7 (Draws 300,000 to 524,999)**: Completed and validated (Step 7 report committed in `6e2e6bd52`).
- **Step 8 (Draws 525,000 to 599,999)**: Completed (`full_step8_report.json`).
- **Split Rule**: Strict seed split (`index % 20 < 18` train [540,000 draws], `18` validation [30,000 draws], `19` test [30,000 draws]). **Never tune on test!**

---

## 4. Model Progression

The models and studies in order; what each one scored is recorded in [`LOGBOOK.md`](../LOGBOOK.md) under the entry named.

| Model / Experiment | Encoder Backbone | Input Transformation | LOGBOOK | What changed |
| :--- | :--- | :--- | :--- | :--- |
| **`model_v1`** | CNN | Log-scaled (`v4`) | `INV-TRAIN`, `INV-CAL`, `INV-STEP1`, `INV-PROBE` | Initial model; calibration gate, step-1 baseline and latent probes |
| **`model_v1_linear`** | CNN | Linear $T$ (clipped/rounded) | `INV-LINEAR` | Linear spectrum instead of log |
| **`model_v2_depth`** | CNN + 1 conv | Linear $T$ | `INV-DEPTH` | One extra stride-1 conv block |
| **`model_v2_depth_bandtop`** | **CNN + 1 conv** | **Band-top resampled** | `INV-BANDTOP` | Spectrum resampled to its own band top, plus the band top as an extra value; the reference recipe |
| **MLP Suite** | MLP (base, deep, wide, ...) | Linear, Log, Band-top | `INV-MLP` | Fully connected encoder instead of the CNN |
| **Denoising AE** | Pretrained `ae_cnn` | Band-top | `INV-AE` | Encoder pretrained as a denoising autoencoder, fine-tuned or frozen |
| **Pooling Suite** | Global average vs spectral | Band-top | `INV-POOL` | Pooling before the encoder's Linear layer |
| Other early runs | CNN | Log | `INV-NOGN`, `INV-SCHED` | Encoder without GroupNorm; AdamW weight decay with a plateau schedule |

### Experiments Since the Table Above

Questions and where each run lives; the outcomes are in [`LOGBOOK.md`](../LOGBOOK.md) under the entry named.

| LOGBOOK entry | Question | Runs |
| :--- | :--- | :--- |
| `INV-600K` | The reference recipe retrained on the full 600,000 draws. | `model_v2_depth_bandtop_600k` |
| `INV-REWARD-1` | Does fine-tuning with a truth reward (label or physics cost) improve the classifier and flow? | `model_v3_reward_*`, `model_v3_more_ce`, `reward_sweep/` |
| `INV-AL-0` | Set-up of the training-time active-learning loop: controls (training-seed spread, 450k learning-curve point) and the round-1 proposal. Phase-2 simulator-reward run stopped. | `model_v2_depth_bandtop_600k_seed1`, `_450k`, `active/` |
| `INV-EMBED` | Does a wider encoder output (832 or 256 instead of 128) help both heads? | `..._600k_embed832`, `..._600k_embed256` |
| `INV-PATTERN-1` | Does a model trained on the lattice type alone (7 site-type patterns) identify it better than the full model's summed state probabilities? | `pattern_v1` |
| `INV-PATTERN-2` | Does regularising that model (dropout, weight decay, label smoothing, input noise) change the answer? | `pattern_v2_*` |
| `INV-NOCHK` | Does retraining without the C+H+K lattices help on the other six types, compared with masking C+H+K in the existing model? | `..._600k_nochk`, `nochk/` |

---

## 5. Diagnostic Studies

### A. Candidate Clustering Review
- [`docs/superpowers/plans/2026-10-28-model-v1-review.md`](superpowers/plans/2026-10-28-model-v1-review.md) traces how much
  of the raw posterior samples' information reaches the candidate output of `candidates.py`, and lists the fixes.

### B. Ergodic Hypothesis & Spatial Disorder (`ERGO-1` in `LOGBOOK.md`)
- Correlation length and stationarity of spectrum fluctuations across atlas clouds (germanene, silicene, hBN, kagome),
  and how much of the parent data's variance the defects' spatial arrangement carries. Study files in
  `results/inverse_design/ergodic/`.

---

## 6. Quick-Start & Operations Cheat Sheet

### Environment Setup
```bash
export PYTHONPATH=src
export TRANSMISSION_DATA=/run/media/shardul/storage/machine_learning/data
export OMP_NUM_THREADS=1
```

### Running Test Suite
```bash
# Run all inverse design unit and regression tests (about 200 tests)
TRANSMISSION_DATA=/run/media/shardul/storage/machine_learning/data python -m pytest tests/atlas/test_inverse_*.py -q
```

### Evaluating or Comparing Models
```bash
# Statistical comparison on 15,000 validation draws
python scripts/evaluate/compare_inverse.py \
    results/inverse_design/model_v2_depth \
    results/inverse_design/model_v2_depth_bandtop \
    --threads 8
```

### Lattice-Type Accuracy
```bash
# Lattice type (7 site-type patterns) of full and lattice-type-only models on the val split
python scripts/evaluate/eval_pattern.py --threads 16 --out results/inverse_design/pattern_v1/eval_pattern.json \
    results/inverse_design/model_v2_depth_bandtop_600k results/inverse_design/pattern_v1

# The same without a lattice type (dropped from val, given no probability in every model)
python scripts/evaluate/eval_pattern.py --threads 16 --drop-pattern CHK --out results/inverse_design/nochk/eval_pattern.json \
    results/inverse_design/model_v2_depth_bandtop_600k results/inverse_design/model_v2_depth_bandtop_600k_nochk
```

### Training Variants
```bash
# Reference recipe; add --embed N, --seed N, --drop-pattern CHK or --extra-data <root> as needed
python scripts/build/train_inverse.py --input bandtop --extra-conv 1 --threads 16 --out results/inverse_design/<name>

# Lattice-type-only classifier
python scripts/build/train_pattern.py --threads 16 --out results/inverse_design/<name> [--dropout 0.3 --noise 0.05]
```

### Running End-to-End Inverse Design
```bash
python -m shazam.inverse.design \
    --target /path/to/target_spectrum.npy \
    --model results/inverse_design/model_v2_depth_bandtop/model.pt \
    --out results/inverse_design/my_design_report.json
```

---

## 7. Open Priorities & Next Actions

1. **Post-Processing Fixes (`candidates.py` & `calibration.py`)**:
   - The clustering and coverage-interval changes listed in the model-v1 review.
2. **Dedicated Discrete Width Head**:
   - Extract ribbon width $w \in [4, 22]$ from the continuous flow head into a dedicated 19-class classification head with ordinal smoothing.
3. **Decouple Clean Host Flow from Defect Density**:
   - Remove `log_c` from the Step-1 continuous flow vector, deferring concentration regression to Step 2 defect analysis.
4. **Active Learning, Round 1**:
   - The round-1 proposal is written (`results/inverse_design/active/cells_round1.json`); generating its 50,000 draws is on hold until approved. Run with `results/inverse_design/active/run_round1.sh`.
