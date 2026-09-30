# Option A, step 1 review: predict width by majority vote (2026-09-30)

For the implementing agent. This reviews your uncommitted step 1 (`build_atlas_v2.py`, `Atlas.build(max_seed, val_seed_min)`, `identification.json`). Step 2 is still running.

## Up to mark
- **The seed split is correct.** `_load_inputs(max_seed=42)` drops test seeds 43–49 before training. Validation is seeds 35–42, and the references and the unknown threshold come from seeds 0–34 and 35–42 only. Evaluation uses seeds ≥ 43 only, and the wider widths are not in the atlas.
- **Material and edge accuracy are 100%** on all 84 model × density lines.
- **The median predicted density equals the true density** on all 84 lines.
- **Unknown false alarms** average 0.5% (at most 1 of 7 on three lines at d = 0.04).
- `load_leads` now computes leads for odd m > 31 (`m % 2 == 0 or m > 31`).

## Finding: width accuracy fails on 20 of 84 lines, all armchair
The failures range down to 28.6% (N10, d = 0.04). The median |width − N| is only 0.1–0.4. The cause is that `Atlas.locate` returns an **inverse-distance-weighted average** of the neighbours' widths. That estimator was built for interpolating between widths, which option A no longer asks for. When disorder mixes neighbours from adjacent widths, the average lands between them and `round(width) != N`.

A **majority vote** over the k neighbours, inside the winning (material, edge) group, fixes most of it. The reviewer recomputed on the same saved atlas and the same test seeds 43–49, pooled over widths:

| Armchair density | Weighted average | Majority vote |
|---|---|---|
| 0.005 | 100.0% | 100.0% |
| 0.01 | 91.7% | **100.0%** |
| 0.02 | 88.1% | **97.6%** |
| 0.04 | 73.8% | **85.7%** |
| Zigzag, all densities | 100% | 100% |

Lines below 99% fall from **20 to 9** of 84, and all nine are armchair at d ≥ 0.02.

Correction to `2026-09-30-smoke-review-and-gate5-diagnosis.md`: "known widths are identified exactly" was measured on the **median** width. Per spectrum, armchair accuracy at d = 0.04 was lower, as this table shows.

## Do
1. **Change the width estimate.** In `Atlas.locate`, add a `width_vote` field (the modal width among the majority group's neighbours), and use it for option A's width accuracy. Keep the weighted average as `width` (or `width_continuous`) for diagnostics. Add a test: a toy query whose neighbours are 9 × width 6 and 6 × width 8 votes 6.
2. **Judge the smoke gate pooled per density.** Each smoke line has only 7 test spectra, so one miss reads as 85.7%. Report both the pooled and the per-line numbers; the per-line ≥ 99% gate applies at full scale, where there are about 1,500 test seeds per line.
3. **Check whether more reference seeds help** before the human commits to the 16.7-hour run. This only needs the existing smoke store:
   - Rebuild the smoke atlas with training seeds 0–14 (validation 35–42, test 43–49 unchanged). Compare armchair width-vote accuracy at d = 0.02 and 0.04 against the 0–34 build.
   - If accuracy rises with training seeds, the full run should close the gap. If it is flat, high-density armchair width needs a different fix, such as a wider input or a width head on the latent. Record the result in SMOKE-2.
4. Rerun `identification.json` with `width_vote`, update SMOKE-2, then continue step 2.
