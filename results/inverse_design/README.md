# Inverse Design Results & Model Index

This directory holds checkpoints, evaluation metrics, logs, and diagnostic studies for the continuous tight-binding inverse design project (`src/shazam/inverse/`).

For full technical details, refer to:
- [`docs/INVERSE_DESIGN_REPORT.md`](../../docs/INVERSE_DESIGN_REPORT.md): Knowledge base: architecture, data, models and how to run them.
- [`docs/superpowers/specs/2026-10-23-inverse-design-design.md`](../../docs/superpowers/specs/2026-10-23-inverse-design-design.md): System architecture specification.
- [`docs/superpowers/plans/2026-10-28-model-v1-review.md`](../../docs/superpowers/plans/2026-10-28-model-v1-review.md): Detailed Round-1 code review and action plan.
- [`docs/inverse_design_explainer.html`](../../docs/inverse_design_explainer.html): Interactive visual explainer page.
- [`docs/superpowers/specs/2026-11-04-reward-finetuning-design.md`](../../docs/superpowers/specs/2026-11-04-reward-finetuning-design.md): Reward fine-tuning design (phases 1 and 2).
- [`docs/superpowers/specs/2026-11-04-active-learning-design.md`](../../docs/superpowers/specs/2026-11-04-active-learning-design.md): Training-time active-learning loop design.
- [`LOGBOOK.md`](../../LOGBOOK.md): The record of every run's setup and outcome (entries `INV-*`, named in the tables below).

---

## Directory Overview

| Folder / File | Type | Description |
| :--- | :--- | :--- |
| **`model_v2_depth_bandtop/`** | Model | **Reference recipe** (300k draws). CNN + 1 conv, band-top input. Retrained on 600k draws in `model_v2_depth_bandtop_600k/`. |
| `model_v2_depth/` | Model | CNN + 1 conv, linear input (`INV-DEPTH`). |
| `model_v1_linear/` | Model | Baseline CNN, linear input (`INV-LINEAR`). |
| `model_v1/` | Model | Initial baseline CNN, log input (`v4`) (`INV-TRAIN`). Includes `report.md` and `host_capture/`. |
| `ae_cnn/` | Pretraining | Standalone denoising CNN spectrum autoencoder (`INV-AE`). |
| `ae_mlp/` | Pretraining | Standalone denoising MLP spectrum autoencoder. |
| `model_v3_cnn_ae_ft_s0/` | Model | CNN fine-tuned from `ae_cnn` initialization (seed 0). |
| `model_v3_cnn_ae_ft_s1/` | Model | CNN fine-tuned from `ae_cnn` initialization (seed 1). |
| `model_v3_cnn_ae_frozen_s0/` | Model | Diagnostic run with frozen autoencoder encoder. |
| `model_v2_mlp_*/` | Exploration | MLP encoder variations (`base`, `deep`, `wide`, `drop`, `log`, `bandtop`). |
| `ergodic/` | Study | Ergodic hypothesis and correlation length study across the atlas clouds (`ERGO-1`). |
| `model_v2_depth_bandtop_gap/`, `model_v2_depth_bandtop_spectral/` | Exploration | Pooling variants of the band-top CNN (`INV-POOL`). |
| `full_step1_report.json` to `full_step8_report.json` | Dataset Reports | Step reports documenting the growth of `hexagonal_parent_v1` from 75k to 600k draws. |
| `smoke_report.json` | Dataset Report | Preliminary 2,000-draw feasibility report (`INV-SMOKE`). |

## Runs Since November 2026

Folders only; what each run found is recorded in [`LOGBOOK.md`](../../LOGBOOK.md) under the entry named.

| Folder / File | Type | Description | LOGBOOK |
| :--- | :--- | :--- | :--- |
| `model_v2_depth_bandtop_600k/` | Model | The band-top CNN + 1 conv recipe trained on 600,000 draws (540,000 train / 30,000 val). Reference model for the runs below. | `INV-600K` |
| `model_v2_depth_bandtop_600k_seed1/` | Control | Same as `600k`, training seed 1: the training-seed spread. | `INV-AL-0` |
| `model_v2_depth_bandtop_450k/` | Control | Same recipe on 450,000 draws: a point on the passive-data learning curve. | `INV-AL-0` |
| `model_v2_depth_bandtop_600k_embed832/`, `..._embed256/` | Model | Encoder output (embedding read by both heads) widened from 128 to 832 and 256 (`--embed`). | `INV-EMBED` |
| `model_v2_depth_bandtop_600k_nochk/` | Model | Retrained without any C+H+K draw in train or val (`--drop-pattern CHK`); C+H+K states get no probability. | `INV-NOCHK` |
| `model_v3_reward_label/`, `model_v3_reward_physics/` | Model | Truth-reward fine-tuning, phase 1, label-cost and physics-cost arms. | `INV-REWARD-1` |
| `model_v3_more_ce/` | Control | Phase-1 control: the original loss trained as long. | `INV-REWARD-1` |
| `model_v4_simreward_pilot/`, `model_v4_simreward/` | Model | Phase 2, simulator-reward fine-tuning (GRPO style). Stopped with its `STOP` file; `checkpoint.pt` resumes it. | `INV-AL-0` |
| `reward_tables/`, `reward_sweep/`, `reward_compare/` | Study | Band-weight and truth-cost tables, the phase-1 settings sweep, and the reward-run comparison tables. | `INV-REWARD-1` |
| `active/` | Study | Active learning: controls evaluation and the round-1 proposal (`cells_round1.json`, `run_round1.sh`). Round-1 generation not started. | `INV-AL-0` |
| `embed/` | Study | Runner and live loss page for the embedding-width runs. | `INV-EMBED` |
| `pattern_v1/` | Model | Lattice-type-only classifier (`shazam.inverse.pattern`): the band-top CNN with a 7-way head (C, H, K, C+H, C+K, H+K, C+H+K), no state head or flow. | `INV-PATTERN-1` |
| `pattern_v2_drop/`, `pattern_v2_noise/`, `pattern_v2_all/` | Model | `pattern_v1` with regularisation: dropout + weight decay + label smoothing, input noise, and both. | `INV-PATTERN-2` |
| `pattern_v2/` | Study | Runner, evaluation and plotting scripts for the lattice-type models (`plot_loss.py`, `plot_chk_spectra.py`, `plot_pattern_by_width.py`). | `INV-PATTERN-2` |
| `nochk/` | Study | Runner and evaluation of the no-C+H+K model (`eval_pattern.py --drop-pattern CHK`). | `INV-NOCHK` |
