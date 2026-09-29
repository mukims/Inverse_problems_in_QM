# multi_width/ — Joint 7- & 9-AGNR models (BUILD-06, leak-free re-run BUILD-09)

**This is where active development happens.**

Every other model folder assumes you already know the ribbon width. These scripts
drop that assumption and solve both halves at once from a single spectrum:

1. **Width classification** — is this a 7-AGNR or a 9-AGNR?
2. **Concentration regression** — how many impurities, `c`?

Dataset: 249,000 spectra (7-AGNR: 34 concentrations `c ∈ [2,68]`; 9-AGNR: 49
concentrations `c ∈ [2,98]`; 3,000 samples each), split 70/15/15 **by configuration seed**
(`--split seed`, the default since BUILD-09). `--split random` reproduces the original BUILD-06
row split, which leaked near-identical spectra into the test set (LOGBOOK Bug #7).

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

### Current results (held-out seeds, BUILD-09)

| Model | Concentration MAE | 7-AGNR | 9-AGNR | RMSE | Random split (BUILD-06) |
|---|---|---|---|---|---|
| Transformer | **2.267** | 1.871 | 2.542 | 3.274 | 1.390 |
| XGBoost | 2.394 | 2.016 | 2.656 | 3.432 | 1.982 |
| Physics misfit (no learning) | 2.442 | 1.985 | 2.760 | 3.834 | 2.445 |
| MLP | 2.884 | 2.454 | 3.182 | 3.673 | 2.018 |

Files: `seed_split/mw_results/*_metrics.json`. Related checks:
- `energy_window_check.py`: the same XGBoost on 0–1.5 eV gives 2.396, on 1.5–3 eV 2.769, on the **full 0–3 eV spectrum 1.977**.
- `seed_split_10k/`: XGBoost with all 10,000 seeds per concentration; on a common test set it improves only from 2.387 (3k seeds) to 2.330.

**BUILD-06 training settings.** The script defaults have drifted; to reproduce BUILD-06 training use
`mw_transformer.py --epochs 80 --patience 80` and `mw_mlp.py --epochs 120 --lr-schedule plateau --plateau-patience 8 --patience 60`.

**Width accuracy here is circular** (LOGBOOK Bug #8): each spectrum is divided by its *own* width's pristine,
which requires knowing the width. Treat the concentration results as "given the correct width". For label-free
width identification see `notebooks/material_atlas/`.

---

## The one rule that matters

**All four techniques must use the same `--samples-per-conc`, `--spectrum-len` and `--split`.**

The split is deterministic given those values (plus `seed=42` for `--split random`), which is exactly
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
