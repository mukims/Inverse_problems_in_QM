# sweeps/ — Hyperparameter optimisation

Search jobs. Nothing here defines a new model; these scripts repeatedly train models
defined elsewhere and report what settings worked.

---

## Files

| File | Searches over | Method |
|---|---|---|
| `bayesian_opt_sweep.py` | PINN hyperparameters (`concentration/pinn_agnr_curvature.py`) | **Optuna TPE** — Tree-structured Parzen Estimator. Saves per-trial checkpoints, then re-evaluates the best trial on the test set. |
| `sweep_misfit_weight.py` | The single `misfit_weight` (λ) trade-off between Focal Loss and the physics-informed Misfit Loss, for `defect_reconstruction/inverse_model.py` | Exhaustive 1D sweep with comparison plots |

`bo_sweep_results/` holds the output of a completed Optuna run: optimisation history,
parallel-coordinates and parameter-importance plots, a test scatter, and
`bo_summary.json`.

---

## Typical use

```bash
cd sweeps && conda run -n ml python bayesian_opt_sweep.py --help
```

## Gotchas

- **These are the most cross-cutting scripts in the repo.** `bayesian_opt_sweep.py`
  imports from [`../concentration/`](../concentration/) and `sweep_misfit_weight.py`
  imports from [`../defect_reconstruction/`](../defect_reconstruction/), both via the
  `_bootstrap` shim. If you refactor either of those folders, these break first.
- Both are **long-running** — they train a model per trial. Run them detached and tail
  the output rather than waiting.
- `sweep_misfit_weight.py` writes to `../../../models/trained/sweep/` **relative to your
  current directory**, so run it from inside `sweeps/`.
- `manifest_agnr.csv` lives two levels up at the `agnr/` root (shared with
  `concentration/`); these scripts reach it via `SCRIPT_DIR.parent` or `'../manifest_agnr.csv'`.
- λ (misfit weight) and Optuna's search space overlap conceptually. If you sweep both at
  once, expect them to interact.
