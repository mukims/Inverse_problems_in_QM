# SMOKE-1 review and why held-out armchair widths fail (2026-09-30)

For the implementing agent and the human. This reviews commit `a6f3d792e` (pushed; 83/83 tests pass). **Do not start the full-scale run.** The human decides that, and the finding below changes what the full run can be expected to deliver.

## Up to mark
- **Data: all 21 models PASS** in `smoke_v1/report.json`.
  - Armchair uses `agnr_lib_IL_1e-5` with the corrected cell; zigzag uses `caroli`.
  - Away from subband edges, the largest median excess is +0.0006 and at most 0.56% of values spike. Zigzag never exceeds pristine away from edges.
  - Seed nesting is checked with `agnr_lib.chosen_for_config` for AGNR, and there are no cross-density duplicates.
- **The even-width fix, the edge-masked guard and their tests** match the directives.
- **The `bands.py` 1e-7 offset** is a benign fix for sign flutter on exactly degenerate bands.
- **`engine_v1`** is untouched (9 clouds), and generation stopped for the human as instructed.

## Finding: held-out armchair widths are confidently misidentified, and more seeds will not fix it
Diagnosis using `Atlas.load("atlas_v2_smoke")` and `locate` on the smoke store:

| Test | Edge accuracy | Width | Flagged unknown |
|---|---|---|---|
| **Known** widths (armchair 7, 9, 14, 16; zigzag 6, 10) on **unseen** seeds 43–49 (validation seeds, not in the references) | **100%** at every density | exact (e.g. 14.0, 16.0) | 0% (14% for N14 at d = 0.04) |
| Held-out armchair N8 | d = 0.005: **8%** (votes zigzag N4); 100% at d ≥ 0.01 | 8.3–9.1 at d ≥ 0.01 | 0–4% |
| Held-out armchair N12 | 36% / 100% / 98% / **12%** (d = 0.04 votes zigzag N4) | ~10 | 0–42% |
| Held-out armchair N13 | **6%** (votes zigzag N11–12) / 64% / 98% / 96% | 8.4–9.2 (true 13) | 0–18% |
| Held-out zigzag N8 | 100% | 9.0 → 6.0 as density rises | 0% |

What this shows:
1. **Sample size is not the limit.** With 50 seeds the atlas already identifies every trained width perfectly on unseen configurations. The full run will not change this table's pattern.
2. **Armchair width does not interpolate.** Armchair spectra fall into three families (3p, 3p+1, 3p+2), and subband edges jump with N. An unseen armchair width lands in an unfamiliar region of the latent space, and at low density its nearest references are often **zigzag**.
3. **The failure is silent.** At d = 0.005 the wrong-edge votes are *confident* (0.83–0.90), and the reconstruction-error flag marks **0%** of them as unknown. The autoencoder reconstructs them well because they are ordinary graphene-like spectra.

Gate 5 as written (edge accuracy ≥ 99%, width within ±1.5 for ≥ 90% on held-out armchair 8, 12, 13) is therefore unlikely to pass at any seed count. This is a design question for the human, not a bug to fix here. The options are listed at the end.

## Nits for the LOGBOOK (Bug #9)
- "> 0.4 eV" should read "> 0.4 t". The engine works in units of t.
- "coordination 1–4" is inaccurate. The committed cell had coordination 2–3 with 4-rings on the last row. Coordination 1 appeared only in the half-fix that removes the chain bond alone.

## Options for the human (no code until one is chosen)
- **A. Dense width grid.** Train on every width that will be queried (e.g. armchair 5–50). Known-width identification is already perfect, so this makes Gate 5's held-out test irrelevant rather than passing it.
- **B. Family-aware generalisation.** Treat armchair as three families, and hold out widths whose same-family neighbours are trained. For example, hold out 11 (3p+2) with 8 and 14 trained, instead of 8, 12 and 13 together. Then measure interpolation within a family.
- **C. Make unseen widths detectable.** Add a novelty score that catches them, such as k-NN distance per (material, edge) class, or disagreement between density bands. Unseen widths are then flagged rather than confidently misassigned. Test it with the same held-out set.

> **Decision (human, 2026-09-30): option A.** Implement it from `2026-09-30-option-a-train-all-widths.md`.
