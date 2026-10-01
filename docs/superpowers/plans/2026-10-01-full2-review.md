# FULL-2 review (2026-10-01)

For the implementing agent and the human. This reviews commit `5a6529848` (pushed), option B with the robust calibration.

## Verified
- **Calibration matches the fix.** Per (model, density): median and 1.4826 · MAD of log s on validation seeds 700–849. The pooled empirical 99th percentile is z\* = 3.144 for `atlas_v2` and 3.133 for the leave-one-out atlas.
- **Identification is unchanged.** 0 fields differ from FULL-1 (`1ffaff0db`): material 100%, edge 100%, width 99.98%.
- **Pooled false alarms are 1.001%** (186 / 18,600), which passes the ≤ 1.5% gate.
- **The class concentration is gone.**

  | Line | FULL-1 (global reconstruction flag) | FULL-2 (option B) |
  |---|---|---|
  | armchair N40, d = 0.04 | 22.0% | 0% |
  | armchair pooled, d = 0.04 | 6.3% | 0.8% |

- **Detection, recomputed independently from the saved atlases:**
  - The unseen square strip is **100%** flagged when queried label-free; the lowest novelty ratio is 1.24.
  - With the leave-one-out atlas (29 models), the untrained armchair N13 is flagged **99.0%** (reconstruction flag: 9.5%), and the untrained zigzag N8 **100%** (reconstruction flag: 0%).

## Not fully met: the per-line criterion
The gate's second half, "no line at or above 8/150", fails narrowly. **Armchair N9 at d = 0.005** and **zigzag N9 at d = 0.01** sit at exactly 8/150 (5.33%). FULL-2 lists both lines but never states that this criterion fails. Add one line saying so.

The per-line counts are more spread than chance. A binomial with a 1% rate over 124 lines would give about 27 lines at 0, 31 at 2, about 2 at 5 and about 0.5 at 6 or more. FULL-2 has 45 at 0, 10 at 2, 10 at 5 and 5 at 6 or more. There are two causes:
- Each class's centre and scale come from only 150 validation spectra, so some thresholds land too low and others too high.
- One pooled z\* is shared by both edges. Zigzag runs at 1.3–1.9% and armchair at 0.2–0.9%.

## Optional refinement (the human's call)
- A separate z\* for each edge, so each edge's rate is about 1%.
- Shrink each class's scale w toward its edge's pooled w, to reduce the estimation noise. For example, w_shrunk = (150 · w + n0 · w_edge) / (150 + n0), with n0 chosen on validation.

This would most likely clear the two 8/150 lines and tighten the spread. Identification would not change, and detection would not change in any material way, since the minimum square ratio is 1.24. Without the refinement, the flag is usable as it stands: pooled 1.0%, worst line 5.3% (against 22% before), and it now catches untrained widths as well as unseen materials.
