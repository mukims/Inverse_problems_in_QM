# Class-conditional novelty for the atlas (option B) (2026-10-01)

For the implementing agent. **The human chose option B** from `2026-10-01-full1-review.md`: score novelty against the predicted class's own reference distribution instead of one global reconstruction-error threshold.

The problem being fixed: the global 99th-percentile threshold gives 0.98% false alarms pooled, but 6.3% on armchair at d = 0.04 and up to 22% on armchair N40 at d = 0.04. High-disorder wide ribbons simply have more varied spectra.

The identification results (material, edge, width, density) must not change; only the unknown flag does. Do this on the existing `atlas_v2` (same encoder, same references); no retraining except the leave-one-out test in step 4.

## 1. The score
- **Class at query time.** Use the locate result's majority (material, edge) group, its `width_vote`, and its predicted density snapped to the nearest grid density. That gives a (model, density) class.
- **Score s.** The mean distance, in the standardized latent space, to the k = 15 nearest references **of that model only** (not of all models).
- **Calibration** on validation seeds 700–849. These are not in the references, so the threshold is honest.
  - Embed each model's validation spectra, locate them exactly as at query time, and compute s.
  - Set the threshold τ(model, density) = exp(mean(log s) + 2.326 · sd(log s)): a log-normal 99th percentile, which is steadier than the empirical one from 150 samples.
  - Store the τ table with the atlas (in `refs.npz` or the manifest), and bump the manifest with a `novelty: "class_conditional_v1"` field.
- **Output.** `unknown = s > τ`. Add a `novelty_ratio = s / τ` field. Keep `recon_error` and also report the old global flag as `unknown_recon`, so the two can be compared.

## 2. False alarms on known spectra (test seeds 850–999)
- Rerun `identification.json` with the new flag.
- **Gate:** per-line false alarms ≤ 2% on all 124 lines, with about 1% expected.
- Material, edge, width and density results must be byte-identical to `1ffaff0db`'s `identification.json`, apart from the unknown fields. Check this.

## 3. Unseen material: square strip
BUILD-12 found the reconstruction error was the best detector of an unseen material (AUROC 1.00 against 0.95 for global k-NN distance). Option B must not lose that.
- Generate a square strip N10 novelty set in a **separate store**, `~/atlas_store/novelty_v1`:
  - 150 seeds × the 4 densities;
  - `formula="legacy_trace"`, which is exact for the symmetric square H1 and is the project's square convention;
  - the engine's 0–4 t grid.

  It is a small ribbon, so this takes minutes.
- Query it **label-free**, with `band_top_t=None`: the engine grid covers the whole window, so no band top is needed.
- **Report:**
  - the flag rate under B and under the global reconstruction threshold;
  - the AUROC of each score against the known test spectra.
- **Gate:** B flags ≥ 95% of the square spectra. If it does not, stop and report to the human with both numbers instead of combining scores on your own.

## 4. Unseen width (informational)
- Build one leave-one-out atlas without armchair N13 and zigzag N8, with the same split, InputSpec v2 and the same B calibration.
- Report the flag rates on those two widths' test seeds under B and under the reconstruction threshold. At smoke scale the reconstruction threshold flagged 0% of unseen armchair widths.
- This is not a gate under option A (every queried width is trained), but it says how B behaves on an untrained width.

## 5. Deliverables
- `atlas_v2/novelty.json`: the per-line false alarms, the square result, the leave-one-out result, and the AUROCs.
- LOGBOOK **FULL-2**: the before and after false-alarm table (per line and pooled per edge and density), the square detection, the unseen-width rates, and the corrected FULL-1 false-alarm wording from the review.
- Tests:
  - a toy where a broad class and a tight class each get about 1% false alarms under B;
  - a toy where an out-of-class spectrum is flagged.
- Commit, push, stop for the human.

## Guard rails
- `engine_v1` and `smoke_v1` stay read-only.
- Write novelty data only to `novelty_v1`, and the leave-one-out atlas to a separate directory (for example `atlas_v2_loo/`).
