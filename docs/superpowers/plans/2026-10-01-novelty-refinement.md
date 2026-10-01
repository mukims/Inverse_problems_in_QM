# Novelty refinement: per-edge tail and shrunk class scale (2026-10-01)

For the implementing agent. **The human chose to apply the optional refinement** from `2026-10-01-full2-review.md`.

The goal is to remove the two lines at 8/150 (armchair N9 at d = 0.005, zigzag N9 at d = 0.01) and the over-dispersion of per-line false alarms. That over-dispersion has two sources:
- noisy per-class scale estimates (150 validation spectra each);
- one shared z\* for two edges whose rates differ (zigzag about 1.6%, armchair about 0.6%).

Identification must not change. Only τ changes; the encoder and references stay the same.

## Method (validation seeds 700–849 only)
1. **Per class** (model, density), as now: c = median(log s), w = 1.4826 · MAD(log s).
2. **Edge scale:** w_edge = the median of w over that edge's classes.
3. **Shrink:** w′ = (150 · w + n0 · w_edge) / (150 + n0).
4. **Per-edge tail:** z = (log s − c) / w′ over all validation spectra of that edge; z\*_edge is the empirical 99th percentile.
5. **Threshold:** τ = exp(c + z\*_edge · w′).
6. **Choose n0 on validation only.** Try n0 ∈ {0, 25, 50, 100, 150, 300}:
   - Calibrate on seeds 700–774 and count the false alarms per class on seeds 775–849.
   - Pick the n0 that minimises the dispersion index of those per-class counts (variance / mean; 1 is pure binomial noise).
   - Then recalibrate on the full 700–849 with that n0.
   - Report the dispersion index for every n0 tried.

Store n0, the w_edge values, the z\*_edge values and the per-class c and w′ in the manifest, under `novelty: "class_conditional_v2"`.

## Gates (test seeds 850–999)
- Pooled false alarms ≤ 1.5% for **each edge separately**, about 1% expected.
- No line at or above 8/150.
- Report the per-line count histogram next to the Binomial(150, 0.01) expectation, and the test-set dispersion index. This is reported, not gated.
- Identification fields must stay byte-identical to FULL-1.
- Square strip, label-free: ≥ 95% flagged. Leave-one-out atlas, recalibrated the same way: report the armchair N13 and zigzag N8 flag rates.

## Deliverables
- **Tests:**
  - a toy with two edges whose score tails differ, where per-edge z\* gives each edge about 1%;
  - a toy with small per-class n, where shrinkage lowers the spread of per-class false-alarm rates.
- Update `identification.json` and `novelty.json`.
- **LOGBOOK FULL-3:** the n0 scan, the before/after table against FULL-2 (pooled per edge, worst line, histogram, dispersion), square and LOO detection.
- Commit, push, stop for the human.
