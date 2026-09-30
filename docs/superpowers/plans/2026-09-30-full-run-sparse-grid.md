# Full run: sparse 31-width grid, 1,000 configurations each (2026-09-30)

For the implementing agent. **The human chose** the sparse grid from SMOKE-2, with **1,000 configuration seeds per (model, density)** for every width. This is the production run into `~/atlas_store/engine_v1`.

## The grid (31 models × 4 densities × 1,000 seeds)
- Armchair (`agnr_lib_IL_1e-5`, corrected cell): N = 5–16, 20, 27, 31, 40, 50.
- Zigzag (`caroli`): N = 4–12, 16, 20, 27, 40, 50.
- Densities: 0.005, 0.01, 0.02, 0.04.
- Seeds: 0–999.
  - Pass `--n-seeds 1000` explicitly. `seeds_for_width` would give only 300 seeds for N ≤ 27 and 100 for N > 27.

## State of `engine_v1` (reviewer check, 12:50)
- **Clean spectra.** 23 of the 31 models already have a `pristine.npy` identical to the verified `smoke_v1` one, including every even armchair width.
- **Missing clean spectra.** Armchair N20, 31, 40, 50 and zigzag N16, 20, 40, 50 have none. `generate()` writes them from the current code.
- **Existing clouds.** Armchair N5 and N7 (4 densities) and N9 at d = 0.005 already hold 1,000-seed clouds tagged `agnr_lib_IL_1e-5`. These are odd widths whose cell never changed, so keep them; the resume logic skips them.

## Order of work
1. **Before generating, commit the following:**
   - **Per-spectrum timing** (from `2026-09-30-smoke2-review.md` §2). Time each spectrum inside the worker and store the median seconds per spectrum per model and density in `meta.json`.
   - **The model list.** `generate_clouds.py`'s `__main__` list (or a `--grid sparse31` option) must hold exactly the 31 models above.
   - **The LOGBOOK SMOKE-2 fixes** (§1 and §3 of the review).
2. **InputSpec v2 and a smoke check of the whole grid.**
   - Add `InputSpec(version="v2")` with `cap` = 1.25 × the largest clean T over the 31 models, rounded up to a multiple of 8. Record the cap in the manifest. Test that no clean spectrum in the grid reaches the cap.
   - Build a v2 atlas from `smoke_v1` on all 31 models into `atlas_v2_smoke31/`: seeds 0–34 train, 35–42 validation, 43–49 test. Write `identification.json`.
   - This proves the pipeline end to end on the wide widths before the long run. Stop and report if any wide width scores below the narrow ones on material or edge accuracy.
3. **Generate `engine_v1`.**
   - Run `OMP_NUM_THREADS=1 generate_clouds.py --store ~/atlas_store/engine_v1 --n-seeds 1000 --n-jobs 16`. Nothing else is using the CPU (i7-13700, 24 threads), and `TRAPS.md` allows 16 when alone.
   - Order the models **narrowest first**, so problems show up early and cheaply. The 10 wide models take about 90% of the time.
   - Run `check_store.py --store ~/atlas_store/engine_v1` after the narrow block (armchair N ≤ 16, zigzag N ≤ 12) and again at the end. Every model must PASS.
   - Expected duration: about **10–12 hours on 16 workers**, or 13–16 hours on 12. That is the sum over models of 4,000 × t_spec divided by the workers, using your measured single-core t_spec, plus ~20% worker contention seen on the widest ribbons and ~0.6 h of pool start-up. The earlier 25–35 h figure came from smoke batches, whose wall time is inflated by chunking (see `2026-09-30-timing-model-note.md`). Put the measured figure from `meta.json` timing in the LOGBOOK.
   - The run resumes by density: if it stops, rerun the same command.
4. **Full-scale atlas.**
   - Build a v2 atlas from `engine_v1` on all 31 models into `atlas_v2/`, split by seed 70/15/15: 0–699 train, 700–849 validation, 850–999 test.
   - Write `identification.json` with per-line and pooled sections, plus `width_vote` and `width_continuous`.
   - With 150 test spectra per line, the revised Gate 5 is measured per line: for every model at every density, material accuracy 100%, edge accuracy ≥ 99%, width accuracy ≥ 99%. Also report the false-alarm rate (target ≤ 2%).
5. **LOGBOOK "FULL-1".** Record the grid, the seeds, the real run time, the `check_store` summary, the gate table, and any line below 99% with its width, density and failure pattern. Commit and push, then stop for the human.

## Guard rails
- Write nothing into `smoke_v1`, and never overwrite an existing `engine_v1` cloud.
- If `check_store` fails on any model, stop generation, keep the files, and report. Do not delete or regenerate without the human.
- Concentration models (Stage 3) are not part of this run.
