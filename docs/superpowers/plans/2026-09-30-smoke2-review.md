# SMOKE-2 review (2026-09-30)

For the implementing agent and the human. This reviews commit `038a2f835` (pushed; 84/84 tests pass). The next step, choosing the width grid and seed counts, belongs to the human. **Start no full-scale generation.**

## Up to mark
- **Option A step 1.**
  - The seed split is correct: test seeds 43–49 are excluded from training, the references and the threshold.
  - `width_vote` is implemented and tested.
  - `identification.json` has pooled and per-line sections.
  - Material and edge accuracy are 100%; zigzag width is 100%.
  - Armchair width is 100 / 100 / 97.6 / 85.7% at d = 0.005 / 0.01 / 0.02 / 0.04.
- **All 31 smoke models pass `check_store.py`**, including the new width-independent check. The reviewer independently checked:
  - armchair N20, N27, N31, N40 and N50 (clean T equals open channels at `nk ≥ 4001`, and nothing exceeds pristine away from edges);
  - zigzag N16, N20, N27 and N40 (Caroli; clean T equals open channels; T ≤ pristine away from edges).
- **The InputSpec cap problem and the shrinking guard coverage** are documented, and v2 is correctly deferred until a wide-grid atlas is trained.

## Fix before the human decides

### 1. The trend test does not support "proves"
Moving from 15 to 35 training seeds changed width accuracy by **1 of 84** test spectra at d = 0.02 (96.4 → 97.6%) and **2 of 84** at d = 0.04 (83.3 → 85.7%). At p ≈ 0.85 and n = 84 the binomial standard error is about 3.9%, so both changes are within noise. Reword the LOGBOOK to "consistent with improvement; not significant at smoke scale".

A test that can settle the question with the existing store:
- Rebuild the smoke atlas with 5, 10, 15, 25 and 35 training seeds, and 3 autoencoder seeds each.
- Report armchair width-vote accuracy at d = 0.04 as mean ± spread for each training-seed count.

If the human wants a firmer answer before the full run, an intermediate run is cheap: 300 seeds for the 21 baseline models gives 210 training and 45 test seeds per line.

### 2. The cost table mixes two measurements
Reviewer's measurement: median time between consecutive density files in `smoke_v1`, so each value covers one 50-spectrum pool including its start-up.

| Model | Table (s per 50) | File timestamps (s per 50) |
|---|---|---|
| armchair N5 | 5.9 | 21.5 |
| armchair N7 | 8.4 | 22.6 |
| armchair N16 | 14.6 | 34.7 |
| armchair N20 | 43.8 | 45.6 |
| armchair N50 | 306.3 | 307.0 |
| zigzag N4 | 2.6 | 19.2 |
| zigzag N50 | 317.6 | 319.9 |

- Every pool carries about **19–20 s of fixed overhead**: spawning 12 workers, the imports, and pickling the leads to each worker. Zigzag N4 takes 19 s for 50 tiny spectra.
- The wide-ribbon rows include that overhead. It is small next to their compute, so those rows are about right.
- The narrow-ribbon rows look like compute-only numbers. That compute-only rate (about 6 spectra/s at N7) also disagrees with SMOKE-1's "14 spectra/s". Nothing is wrong with the data; the projections for narrow ribbons are simply not trustworthy.
- The "single-core CPU time" column is rate × 12 for armchair, but not for wide zigzag (N16: 0.718 × 12 = 8.6, not 2.31).
- The H₀ dimension column lists N instead of 2N for armchair N ≥ 20.

Fix:
- Time the compute of each spectrum inside the worker (`time.perf_counter()` around `_one` / `_one_agnr`). Store the median seconds per spectrum per model in `meta.json`.
- Project each grid as Σ over models of (seeds × 4 × t_spectrum / n_workers) plus the per-pool overhead.

The headline stands either way: the wide ribbons dominate. Armchair and zigzag N50 cost about 5.7 s per spectrum on 12 workers (about 69 s per spectrum per core), so the 5–50 grid at 10,000 seeds per density is months, not days.

### 3. Headline
The SMOKE-2 row in the build table reports "98.6% Width (d ≤ 0.02)". Also put the d = 0.04 figure (91.8% overall, 85.7% armchair) in the headline, since that is the case that fails the revised Gate 5.
