# Post-Hoc Diagnostics & Exploratory Scripts

The scripts in this directory are post-hoc test-set diagnostics and exploratory scripts developed during the FULL-3 novelty refinement study.

**Integrity Note**:
These scripts were **NOT** used to select or tune the model parameters.
All model parameters ($n_0$, $w_{\text{edge}}$, $z^*_{\text{edge}}$, and per-class thresholds $\tau$) in `atlas_v2` and `atlas_v2_loo` were strictly computed and calibrated on validation seeds 700–849 using the automated validation-only rule implemented in `atlaslib/atlas.py`.
