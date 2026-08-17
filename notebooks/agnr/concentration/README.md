# concentration/ — Single-width concentration models (the original lineage)

Predict impurity concentration `c` from a transmission spectrum `T(E)`, **assuming the
ribbon width is already known** (historically 7-AGNR).

This is the older line of work. For new development use [`../multi_width/`](../multi_width/),
which handles 7- and 9-AGNR jointly and carries a set of training fixes these scripts
do not have. This folder is kept because it holds the trained checkpoints, the
published benchmark numbers, and the physics-informed variants.

---

## Model lineage

| File | Model | Note |
|---|---|---|
| `pinn_agnr.py` | Physics-informed **CNN** with a differentiable misfit regulariser | `DifferentiableMisfit`, residual 1D blocks |
| `pinn_agnr_curvature.py` | `ConductanceMLP` + **curvature-weighted** misfit | The "PINN" the README and LOGBOOK refer to. Checkpoint: `pinn_agnr_curvature.pt` |
| `patched_transformer_v2.py` | 1D-patched transformer with ConvStem + `[CLS]` head | Best single-width result (MAE ≈ 0.98). Checkpoint: `patched_transformer_v2.pt` |
| `train_conc_models.py` | Trains/benchmarks XGBoost + MLP + transformer from one entry point | Step 2 of the inference pipeline |
| `retrain_models.py` | Retrains the MLP and transformer on consolidated data, backing up the old checkpoints first | Writes `*_backup.pt` |
| `compare_all_models.py` | Four-way benchmark → the `compare_*.png` plots in this folder | Loads checkpoints from **this** folder |
| `test_validation_pinn_agnr.py` | Test/validation suite for the PINN models | |
| `create_notebook.py` | Generates `compare_analysis.ipynb` programmatically | **No argparse** — running it writes a notebook into the current directory |

---

## About the "physics-informed" part

`pinn_agnr_curvature.py` adds a real physics term to the loss:

```
L = MSE(ĉ, c) + λ · CurvatureMisfit(x, ĉ)
```

It evaluates the curvature κ of the configuration-averaged misfit landscape around the
predicted concentration. Where the misfit minimum is sharp (high κ) the physical
constraint is amplified; where it is flat or ambiguous the penalty is tempered.

**The current multi-width models deliberately drop this.** The working assumption there
is that a sufficiently expressive network learns scattering behaviour from `T(E)`
without a hand-built physics prior. These files remain as the reference implementation
of the other approach — don't build new work on them, but don't delete them either.

---

## Checkpoints in this folder

| File | Produced by |
|---|---|
| `pinn_agnr_curvature.pt` / `_backup.pt` | `pinn_agnr_curvature.py`, `retrain_models.py` |
| `patched_transformer_v2.pt` / `_backup.pt` | `patched_transformer_v2.py`, `retrain_models.py` |

They live here on purpose: `compare_all_models.py` and `retrain_models.py` load and save
them with `script_dir / "<name>.pt"`, so **checkpoint and script must share a folder**.
Moving either one breaks the other.

---

## Typical use

```bash
cd concentration && conda run -n ml python compare_all_models.py
```

## Gotchas

- `manifest_agnr.csv` lives one level up (it is shared with `sweeps/`); these scripts
  reach it via `script_dir.parent / "manifest_agnr.csv"`.
- `create_notebook.py` has no argument parser — invoking it with any argument still
  regenerates the notebook in the current directory.
- `test_validation_pinn_agnr.py` expects data under `<repo>/data/raw/transmission_results/`.
  If that tree is incomplete it fails with `FileNotFoundError` — that is missing data,
  not a broken import.
