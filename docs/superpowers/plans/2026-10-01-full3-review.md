# FULL-3 review (2026-10-01)

For the implementing agent and the human. This reviews commit `24ab499a2` (pushed, 93/93 tests pass).

## Verified
- **No test tuning reached the atlas.** The reviewer recomputed the rule from validation seeds alone (calibrate on 700–774, dispersion on 775–849, then recalibrate on 700–849), and every value matches the manifest exactly:

  | n0 | Dispersion index |
  |---|---|
  | **0** | **3.337** |
  | 25 | 4.113 |
  | 50 | 4.248 |
  | 100 | 4.388 |
  | 150 | 4.340 |
  | 300 | 4.612 |

  - The rule picks **n0 = 0**.
  - z\*\_armchair = **2.9854** and z\*\_zigzag = **3.3498**.
  - w\_edge is 0.1326 for armchair and 0.1129 for zigzag.

  The test-set scans are archived under `diagnostics/` with a disclosure README, and FULL-3 states this.
- **Identification is unchanged.** 0 fields differ from FULL-1.
- **The two edges are balanced.** Pooled false alarms are armchair **0.87%** and zigzag **1.16%** (FULL-2: 0.61% and 1.48%). Both are within ≤ 1.5%.
- **Detection holds:**
  - the unseen square strip is flagged 100% (AUROC 0.9999);
  - in the leave-one-out atlas, untrained armchair N13 is flagged 99.5% and untrained zigzag N8 100%.
- **Honest reporting.** FULL-3 marks the per-line gate as **FAIL (narrow)**: 3 lines at 8/150 or more, with a maximum of 10/150 (armchair N9, d = 0.005).

## Correct in FULL-3
The "FULL-1 (Global Recon)" column of the comparison table is wrong. The reconstruction threshold has not changed since FULL-1 (0.005190 in both manifests). The correct FULL-1 figures, from `1ffaff0db`'s `identification.json`:

| Row | Correct FULL-1 value | FULL-3 table says |
|---|---|---|
| Overall pooled false alarms | 0.979% (182 / 18,600) | 1.84% |
| Armchair pooled | 1.785% | 2.05% |
| Zigzag pooled | 0.000% | 1.59% |
| Max line | 33/150 | 33/150 (correct) |
| Lines at 8/150 or more | **8** | 11 |

## What FULL-3 shows
- **Shrinking the class scale does not help.** The dispersion rises with n0 at every step. So the over-dispersion of per-line false alarms is not estimation noise: a few classes have heavier-tailed scores than the rest.
- **The remaining excess is narrow armchair at low density:**

  | Line | Flagged |
  |---|---|
  | N9, d = 0.005 | 10/150 |
  | N6, d = 0.01 | 9/150 |
  | N6, d = 0.005 | 8/150 |

  On the validation half, one class reaches 13/75.

## For the human
- **Accept.** Pooled 1.0% with both edges balanced, and the worst line at 6.7% (against 22% under FULL-1). The flag catches unseen materials and untrained widths.
- **Or one more step:** give the few heavy-tailed classes their own empirical quantile from a larger sample, for example by drawing extra calibration seeds from the training range for those classes only. This adds complexity for a handful of lines.

> **Decision (human, 2026-10-01): accept FULL-3 and move to Stage 3.** See `2026-10-01-stage3-concentration.md`, whose Part 0 is the table fix.
