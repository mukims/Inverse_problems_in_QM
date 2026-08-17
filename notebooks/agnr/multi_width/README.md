# multi_width/ — Joint 7- & 9-AGNR models (BUILD-06, current work)

**This is where active development happens.**

Every other model folder assumes you already know the ribbon width. These scripts
drop that assumption and solve both halves at once from a single spectrum:

1. **Width classification** — is this a 7-AGNR or a 9-AGNR?
2. **Concentration regression** — how many impurities, `c`?

Dataset: 249,000 spectra (7-AGNR: 34 concentrations `c ∈ [2,68]`; 9-AGNR: 49
concentrations `c ∈ [2,98]`; 3,000 samples each), split 70/15/15.

---

## The four techniques (`mw_*.py`)

Run them **independently, in any order, at your convenience**. Each is standalone.

| Script | Technique | Rough cost (24-core CPU) |
|---|---|---|
| `mw_misfit.py` | Analytical reference-library baseline. No learning. | seconds |
| `mw_xgboost.py` | Gradient-boosted trees: width classifier → concentration regressor | ~1 min |
| `mw_mlp.py` | Multi-task `ConductanceMLP`, dual heads | ~25 min (100 epochs) |
| `mw_transformer.py` | Multi-task 1D-patched transformer | ~100 min (40 epochs) |
| `mw_compare.py` | Assembles whatever results exist into a table + plots | seconds |

```bash
cd multi_width
conda run -n ml python mw_misfit.py
conda run -n ml python mw_xgboost.py
conda run -n ml python mw_mlp.py
conda run -n ml python mw_transformer.py
conda run -n ml python mw_compare.py
```

`mw_common.py` is the shared library (**not runnable**): data loading, the
train/val/test split, metrics, logging, the target scaler, and the shared trainer
used by both neural scripts.

---

## The one rule that matters

**All four techniques must use the same `--samples-per-conc` and `--spectrum-len`.**

The split is deterministic given those two values plus `seed=42`, which is exactly
what makes the four result sets comparable. Change either and you get a different
test set. `mw_compare.py` checks this and warns loudly if the stored predictions
disagree — trust that warning.

---

## Outputs

Each technique writes to `mw_results/`:

- `<tag>_metrics.json` — the metric block
- `<tag>_preds.npz` — test predictions, ground truth, and (for the nets) training history

Checkpoints (`mw_mlp.pt`, `mw_transformer.pt`, `mw_xgb_*.json`) go to the folder root,
as do `mw_compare.py`'s plots. Every script also writes a tail-able `<tag>.log`.

Use `--out-dir /some/scratch` for experiments so a test run **cannot overwrite real
results**. This flag exists because a careless test run once clobbered a finished
64-minute training run.

---

## Design decisions baked into these scripts

These came out of a post-mortem on the first BUILD-06 run, where XGBoost
unexpectedly beat both networks:

1. **Targets are standardised** (`TargetScaler`). Concentrations span 2–98, so raw-scale
   MSE started near 2500 and the output layer had to emit ~98 from unit-scale features.
   Trees are scale-invariant; networks are not. This was the main defect.
2. **`alpha_width` is 1.0, not 10.** With normalised targets both loss terms are O(1).
   The old α=10 contributed ~0.3% of the loss and width is solved by epoch ~5 anyway.
3. **The concentration head sees the width posterior**, mirroring the extra feature
   XGBoost gets from its classifier.
4. **Less regularisation, more epochs.** Training loss sat *above* validation loss with
   both still falling — underfitting, not overfitting.
5. **Huber loss, and checkpoints selected on validation MAE** — the metric actually
   reported, rather than the MSE-dominated composite loss.
6. **Positional embeddings added before the LayerNorm** at std 0.10 (transformer). The old
   ordering left position at ~2% of token magnitude, which is fatal when band edges
   sit at specific energies.
7. **LR warmup** before cosine decay (transformer).

Every one of these has an ablation flag (`--no-width-condition`, `--pos-after-norm`,
`--alpha-width`, `--snap-grid`, …) so you can measure what actually helps.

**No physics-informed loss term is used.** The premise is that a sufficiently
expressive network learns scattering behaviour from `T(E)` directly.

---

## Legacy files here

| File | Status |
|---|---|
| `run_sequential_pipeline.py` | The original monolith that ran all stages in sequence. Superseded by the `mw_*` split; kept for reference. Its models lack the improvements above. |
| `train_multi_width.py` | An earlier multi-width trainer, superseded. |
| `multi_width_*.pt` / `multi_width_*.json` | Artifacts from the original BUILD-06 run. `multi_width_transformer.pt` is **not** the real 64-minute model — that one was lost and needs regenerating via `mw_transformer.py`. |
