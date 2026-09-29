# Traps this project has already hit

Each entry: symptom → cause → what to do.

## Data and evaluation

**Seed leakage (Bug #7).** Scores 21–63% better than they should be. A seed's impurity set at c + 2 contains its set at c, so near-duplicate spectra sit on both sides of a random row split. → Split by configuration seed everywhere, and give every new store and script a seed column.

**Circular classification (Bug #8).** 100% material or width accuracy that comes too easily. The input was divided by the material's own pristine spectrum, which encodes the answer. → Keep everything before identification label-free; use pristine normalisation only after the material has been predicted.

**Gap pinning (Bug #6).** Band-gap channels jump between 0 and exactly 1.0 after normalisation. Pristine T inside the gap is about 1e-6, not 0. → Round T and T_pristine to 3 decimals before dividing, and treat T_pristine = 0 as "divide by 1".

**Cache mutation (square generator).** Byte-identical spectra at different concentrations, and a mean spectrum that stops changing with c. `@lru_cache` returned a shared NumPy array that the caller then edited in place. → Build a fresh array per call, or `.copy()` the cached one; `CloudStore` refuses cross-density duplicates to catch this class of bug.

**Formula spikes.** Raw T above the pristine channel count. The legacy trace formula is unbounded near subband edges. → Check `CloudStore.spike_fraction`; with `caroli` it is ≈ 0.

**Mislabelled metrics.** A number that doesn't match a similar result elsewhere, such as the notebook's LightGBM "RMSE" 0.1073, which was a raw-scale MAE. → Recompute any metric you compare, on the same scale and split, before putting it in a table.

**Different test sets.** Comparing MAEs across builds with different concentration ranges or splits. → Compare only on a common test set; state the range and split beside every number.

## Compute

**Thread oversubscription.** Epochs 3–6× slower than expected. This CPU (i7-13700) has 8 performance cores and 8 efficiency cores; PyTorch with more threads than free performance cores, or two trainings sharing cores, stalls on the slowest thread. BLAS inside multiprocessing workers multiplies threads again. → 16 PyTorch threads when alone, ≤ 4 beside another training; `OMP_NUM_THREADS=1` in generator workers; benchmark one epoch before a long run.

**Orphan processes.** A job you "stopped" keeps running, and the next job is mysteriously slow. Killing the `bash -c` wrapper of a background command leaves its `python` child alive. → Find the python PID with `pgrep -fa "python -u <script>"` and stop that PID; confirm with `ps -p <pid>`.

**Self-matching pkill.** `pkill -f "<pattern>"` also matches, and kills, the shell running it (exit code 144). → Stop jobs by PID.

**Schedule drift.** A retrained network scores worse than its record. Script defaults drifted from the settings that produced the recorded result (e.g. the BUILD-06 transformer needs `--epochs 80 --patience 80`; the MLP needs `--epochs 120 --lr-schedule plateau --plateau-patience 8 --patience 60`). → Read the original run's log header and reuse its exact flags.

**Silent early stopping.** A gradient-boosting fit finishes in seconds and underperforms. scikit-learn's `HistGradientBoostingRegressor` enables early stopping automatically above 10,000 samples. → Set `early_stopping=False` (or state the choice) when comparing models.

**Stale caches.** A loader silently reuses preprocessed data built before a fix (e.g. `universal_cache_*.pt` from before rounding). → Name caches by their inputs (length, seeds, spec version) and delete them when preprocessing changes.

**Stdout buffering.** Background logs stay empty for a long time. → Run Python with `-u`, or `flush=True`.

## Physics and grids

**Band top ≠ 3t.** An armchair ribbon's band top is t(1 + 2cos(π/(N+1))), below 3t. → Compute band tops with `band_edges`; never hard-code them.

**Energy grid origin.** Channel *i* is E = 0.01·*i* starting at 0, so the first 150 channels are 0–1.49 and channels 150–169 are 1.50–1.69.

**AGNR legacy leads cover only 0–2.99.** → The engine computes its own leads on the 0–4 t grid.

## Repository

**Repository name.** `github.com/mukims/transmissions` redirects to `github.com/mukims/Inverse_problems_in_QM`, the canonical name used in the README.

**Large files.** Model checkpoints (`*.pt`, `*.pth`) and `*.npy`, `*.csv` are git-ignored; metrics JSON, small `.npz` predictions and plots are committed. XGBoost model JSONs can reach 14 MB; commit them only when the result depends on them.
