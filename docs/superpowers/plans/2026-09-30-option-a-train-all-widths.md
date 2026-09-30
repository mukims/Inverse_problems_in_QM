# Option A: train the atlas on every width (2026-09-30)

For the implementing agent. **The human chose option A** from `2026-09-30-smoke-review-and-gate5-diagnosis.md`: train the atlas on every width that will be queried, rather than asking it to generalise to widths it never saw.

Why: at smoke scale the atlas already identifies every trained width perfectly on unseen configurations (100% edge type and exact width at every density). It fails only on untrained armchair widths, because armchair spectra do not interpolate between widths (the 3p / 3p+1 / 3p+2 families).

Work at smoke scale (50 seeds) only. The full run still waits for the human.

## Step 1: Retrain the smoke atlas on all widths, evaluated on held-out seeds
No new clouds are needed; `smoke_v1` already holds every width in the grid.

1. **`build_atlas_v2.py`**
   - Train on every model in `TRAIN`.
   - Replace `HELD_OUT` with an optional flag `--holdout-widths` (default: none). The old width-held-out test stays reproducible as a diagnostic.
2. **Split by configuration seed, 70/15/15, in every model** (the project rule, LOGBOOK Bug #7).
   - Smoke: seeds 0–34 train, 35–42 validation, 43–49 test.
   - The autoencoder, the references and the novelty threshold use only train and validation seeds.
   - Test seeds are used only for evaluation. `Atlas.build` currently takes its validation from the top 15% of all seeds it receives, so pass it only seeds 0–42, for example through a seed filter in `_load_inputs`.
3. **Write `atlas_v2_smoke/identification.json`**: for every model and density, on test seeds:
   - material accuracy;
   - edge accuracy;
   - width accuracy (`round(width) == N`);
   - median |width − N|;
   - share flagged unknown (the false-alarm rate on known materials);
   - median predicted density vs true density.
4. **Revised Gate 5** (the human's option A replaces the width-held-out gate). For every model at every density, on test seeds:
   - edge accuracy ≥ 99%;
   - width accuracy ≥ 99%;
   - material accuracy 100%.

   Report the false-alarm rate. The ≤ 2% target applies at full scale; at smoke it is informational, since the threshold is set from only 8 validation seeds per model.
5. **LOGBOOK.** Record it as SMOKE-2. Keep the SMOKE-1 width-held-out numbers there as a documented limitation: "untrained armchair widths are confidently misidentified; the atlas must be trained on every width it will see."

## Step 2: Measure the cost of a wider grid (for the human's choice of final grid)
The plan's widths include 27 and 50, and neither is in the current grid (armchair 5–16, zigzag 4–12). Before anyone chooses the final grid, measure:
- Generate 50-seed smoke clouds into `smoke_v1` for armchair N = 20, 27, 31, 40, 50 and zigzag N = 16, 20, 27, 40, 50. Run `check_store.py` on them; they must PASS like the others.
- Record the **seconds per spectrum against width** for each edge, and project the full run (10,000 seeds × 4 densities) for three grids:
  - armchair 5–31, zigzag 4–31;
  - armchair 5–50, zigzag 4–50;
  - a sparse grid, the current one plus {20, 27, 31, 40, 50}.
- Facts to rely on:
  - Every armchair width fits `agnr_lib`'s 0–2.99 t grid. The band top is below 3 t for all N (2.9962 t at N = 50), so `InputSpec.to_input` zero-fills exactly. Checked for N = 16, 27, 31, 40 and 50.
  - Stored leads cover m = 5–31 only. For odd m > 31, make `load_leads` compute leads with `leads_sancho_rubio` and cache them under `agnr_cell_v2`, as it already does for even m. The odd-m cell is unchanged, and the recompute reproduces the stored odd leads (m = 5, 9, 11 exactly; m = 7 except at the subband edge E = 1.00 t).
  - Cost grows with width (2m × 2m inversions over 100 cells), so expect wide armchair ribbons to dominate the run time.
- Do **not** retrain the atlas on the wider grid in this step. Report the table and stop.

## Done when
- `identification.json` exists, with every model × density line against the revised Gate 5;
- the cost table and the three projections are in the LOGBOOK SMOKE-2 entry;
- the new clouds pass `check_store.py`;
- the tests pass and everything is committed and pushed.

Then stop for the human, who chooses the final width grid and when to start the full run.
