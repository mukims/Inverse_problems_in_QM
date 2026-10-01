# Guard rail: no tuning on the test seeds (2026-10-01, 20:10)

For the implementing agent, before you commit FULL-3.

**Up to mark.** The uncommitted `atlaslib/atlas.py` change implements the refinement as specified:
- n0 is chosen on validation only (calibrate on 700–774, score the dispersion on 775–849);
- the per-edge z\* is the empirical 99th percentile on validation;
- there are no hard-coded values.

**Not up to mark: the exploratory scripts score candidates on the test seeds.**
- `scan_z_arm.py` says "Scan z_arm values with z_zig=3.350 to see **test** line behavior". It sweeps hand-picked z_arm values (2.985 to 3.35) on seeds ≥ 850, with zigzag fixed at 3.350.
- `eval_n0_exact.py` and `study_n0.py` evaluate each n0 on seeds 850–999.

If any z or n0 that reaches the atlas was chosen, or even nudged, by looking at test results, the FULL-3 test numbers are optimistically biased and the gate means nothing.

## Rules
1. The **only** values that may reach `atlas_v2` are those `calibrate_novelty` computes from seeds 700–849 with its validation rule. No z\*, n0 or τ may be set by hand or overridden.
2. Report the test results for that validation-selected configuration **as they come out**, even if a gate fails. A failed gate is a result for the human, not a reason to keep tuning.
3. Do not commit `scan_z_arm.py`, `eval_n0_exact.py`, `study_n0.py`, `scan_n0.py` or `test_refinement.py` as part of the method. Either delete them, or move them to a `notebooks/material_atlas/diagnostics/` folder with a header saying they are post-hoc test-set diagnostics, not used for selection.
4. In FULL-3, state plainly that test-set scans were run during development, and that the reported configuration is the one the validation-only rule picked. Name its n0 and its z\*_edge.
