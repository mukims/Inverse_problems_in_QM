# FULL-1 review (2026-10-01)

For the implementing agent and the human. This reviews commit `1ffaff0db` (pushed): the production run and `atlas_v2`.

## Verified
- **Data.** `engine_v1` holds 124 clouds (31 models × 4 densities × 1,000 seeds).
  - `check_store`: 31/31 PASS.
  - The reviewer's independent sweep of all 124 clouds found 0 failures on: n and seeds; formula tag; finiteness; pristine identical to the verified `smoke_v1` spectra; no repeated rows across densities.
  - Physics checks also pass. Armchair: the median exceeds pristine by at most +0.0004 away from subband edges, and at most 0.55% of values spike. Zigzag: never above pristine away from edges.
- **Split.** `Atlas.build` gets `max_seed=849` and `val_seed_min=700`, so test seeds 850–999 never reach training, the references or the threshold. The manifest records InputSpec v2 (cap 64) and 31 models. Every one of the 124 lines has 150 test spectra.
- **Results reproduce.** The reviewer recomputed three lines from the saved atlas, and they match `identification.json` exactly:

  | Line | Width (vote) | Flagged unknown |
  |---|---|---|
  | armchair N8, d = 0.04 | 98.00% | 1.33% |
  | armchair N40, d = 0.04 | 100% | 22.00% |
  | zigzag N50, d = 0.04 | 100% | 0% |

- **Headline.**
  - Material and edge accuracy are 100% on all 18,600 test spectra.
  - Width accuracy is 99.98%, with 123/124 lines at or above 99%.
  - The only width line below 99% is armchair N8 at d = 0.04: 98.0%, 3/150, all predicted as N5, which is the same 3p+2 family.
  - The smoke-scale gap at d = 0.04 (armchair ~86–92%) closed with 700 training seeds: pooled armchair width at d = 0.04 is 99.84%.

## Correct in FULL-1: the false-alarm rate
"The overall false alarm rate is 0.979%, easily satisfying the Gate 5 target of ≤ 2%" is true but tests nothing. The unknown threshold is the 99th percentile of validation reconstruction error pooled over all models (`atlas.py:82`), so a pooled rate near 1% holds by construction. The flags concentrate on the hardest class:

| Group | Flagged unknown |
|---|---|
| zigzag, every density | 0.0% |
| armchair d = 0.005 / 0.01 / 0.02 | 0.0% / 0.04% / 0.82% |
| **armchair d = 0.04 (pooled)** | **6.27%** |
| armchair N40, d = 0.04 | **22.0%** |
| armchair N50, d = 0.04 | 19.3% |
| armchair N31, d = 0.04 | 14.0% |
| armchair N16 and N27, d = 0.04 | 10.0% each |

13 lines exceed 2%: armchair N10, N11, N12, N13, N14, N16, N20, N27, N31, N40, N50 at d = 0.04, and N12 and N40 at d = 0.02.

In use, a known wide armchair ribbon at 4% disorder would be reported as "unknown material" in up to 1 case in 5. Replace the sentence with the per-line picture above.

## For the human (no code until decided)
How to make the unknown flag useful across classes:
- **A.** Per-class thresholds: calibrate the 99th percentile separately for each (edge, density band) on validation seeds 700–849. The density estimate is already 100% correct, so the band is available at query time.
- **B.** Score novelty against the predicted class's own reference distribution, for example k-NN distance normalised by that class's spread, instead of one global reconstruction threshold.
- **C.** Keep the flag as it is and document that it is conservative for high-disorder, wide armchair ribbons.

None of these changes the identification results above.
