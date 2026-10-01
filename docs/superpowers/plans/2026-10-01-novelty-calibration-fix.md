# Option B calibration fix and a sound false-alarm gate (2026-10-01)

For the implementing agent. This reviews `894b4d4c7`. Two corrections to the reviewer's own directive, `2026-10-01-class-conditional-novelty.md`, are explained below.

## Up to mark
- The implementation matches the design:
  - the score uses only the predicted model's references (a per-model `NearestNeighbors`);
  - calibration filters seeds to 700–849 (`val_seed_min=700`, `max_seed=849`);
  - τ is looked up by the predicted model and the snapped predicted density.
- The existing encoder is loaded, not retrained. **0** identification fields differ from FULL-1 (`1ffaff0db`): material, edge, width, width-continuous and predicted density are all unchanged.
- Option B equalises the flag across classes. The worst line drops from 22.0% (armchair N40, d = 0.04) to 7.3%. Armchair at d = 0.04 is no longer singled out.

## Correction 1: the log-normal 99th percentile is too low (the reviewer's spec error)
The pooled false-alarm rate under B is **2.06%**, against the 1% target, and 47 lines exceed 2%.

The reviewer checked validation seeds 700–849 for six models (armchair N7, N20, N40; zigzag N9, N27, N50; 3,600 spectra in all). log s, standardised per class, is right-skewed: skew +0.61, excess kurtosis +0.60.
- The cut at z = 2.326 lets through **1.86%** of validation spectra.
- The empirical 99th percentile of z is **2.73** (mean/sd standardisation) or 3.10 (median/MAD).

Replace the calibration:
- **Per class** (model, density), on validation seeds: centre c = median(log s) and scale w = 1.4826 · MAD(log s). These are robust to the few large values.
- **Pooled tail:** z = (log s − c)/w over **all** validation spectra (31 × 4 × 150 = 18,600). z* = the empirical 99th percentile of that pooled z.
- **Threshold:** τ(model, density) = exp(c + z* · w). Store c, w and z* with the atlas.
- **Toy test:** two classes with the same skewed score shape but different spreads each get about 1% false alarms.

## Correction 2: the per-line gate "≤ 2% on every line" cannot be met at n = 150
With 150 test spectra per line, even a perfectly calibrated 1% flag shows more than 2% (4 or more of 150) on a line with probability 0.065. That is about **8 of 124 lines by chance alone**. Use this gate instead:
- **pooled** false alarms over all 18,600 test spectra **≤ 1.5%**;
- **no line at or above 8 of 150 (5.33%).** A line whose true rate is 1% reaches that with probability 0.00015, and a true-2% line with probability 0.011.

Report the per-line distribution as well, for example a histogram of flagged counts per line.

## Then
- Rerun `identification.json`. The identification fields must still match FULL-1 exactly.
- Rerun the square-strip and leave-one-out evaluations with the new τ, if `eval_novelty.py` used the old table. A higher threshold flags slightly fewer square spectra, and the ≥ 95% gate still applies.
- Record this in FULL-2. State the original log-normal calibration, its measured 2.06% pooled rate, the cause (skewed log scores), and the fix.
