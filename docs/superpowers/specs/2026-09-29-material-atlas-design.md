# Material Atlas — Design (settled 2026-09-29)

A reusable map of transport fingerprints. A transmission signature T(E) is placed in a learned
latent space, matched to the closest known material (type → edge → width), flagged if it matches
nothing known, and handed to material-specific models that estimate disorder.

## Purpose
1. **Coverage:** the sensor pipeline recognises many ribbon types.
2. **Generalisation:** the map places structures it was never trained on (held-out widths) at a
   sensible position; width is read off the map, not looked up.

## Scope and order
1. **7/9-AGNR reference solution** (must be the best possible before expanding), containing:
   label-free identification (stages 1–2), full-spectrum concentration (stage 3), calibrated
   uncertainty, and end-to-end scoring with the *predicted* width.
   - Success criteria: label-free width accuracy ≥ 99.5%; end-to-end concentration MAE ≤ 1.98;
     90% intervals with coverage within 90 ± 2%.
2. **Width expansion (generalisation test):** idealised AGNR N = 5–16, holding out 8, 12, 13;
   ZGNR N = 4–12, holding out 8.
3. **Real materials in tight binding**, in this order: realistic graphene ribbons → hBN → MoS₂ →
   phosphorene. Parameters from published TB models (lookup in progress).
4. Impurity strength, device length, and the concentration scaling law are later pipeline stages
   (the scaling law is designed by the project owner).

## Models
- Ribbons of width **N = 7, 9, 14, 27, 50** (N = atomic rows across: dimer lines for armchair,
  zigzag chains for zigzag, Mo rows for MoS₂). **Both armchair and zigzag** edges for every material.
- 100 unit cells, leads of the same clean material.
- Impurities: on-site shift **V = 0.5 t** at sites drawn with `RandomState(seed).choice(..., replace=False)`.
- Energy unit: t = largest nearest-neighbour hopping magnitude. E = 0 at charge neutrality.

## Input specification (v1)
- Window **0 to 4 t**, step 0.01 t, **400 channels, positive energies only**. Honeycomb bands end
  at or below 3 t (N-AGNR: t(1 + 2cos(π/(N+1)))), so data above each model's computed band top is
  exactly zero and zero-padding is exact.
- Label-free transform, identical for every spectrum: `log1p(clip(round(T, 3), 0, 20)) / log1p(20)`.
- Principle: capture the basic yet as complete as possible electronic structure of each material.

## Disorder clouds (atlas scale)
- Densities (impurities per site) **0.5%, 1%, 2%, 4%**, plus the clean (pristine) spectrum as an anchor.
- Seeds per density: **1,000 for N ≤ 14, 300 for N = 27, 100 for N = 50**.
- Disorder unit: density per site is canonical; impurities per cell stored alongside.

## Labels
Hierarchy material → edge → width, with **width a continuous coordinate** within (material, edge).

## Evaluation rules
- Always split by configuration seed (LOGBOOK Bug #7).
- Nothing applied before identification may use the material's identity (LOGBOOK Bug #8).

## Presentation
1. (b) Clean fingerprints of every model on the shared axis → new tab in the report doc.
2. (a) Disorder clouds on the atlas map → same tab.
3. (c) Interactive atlas page → standalone artifact.

## Open decisions (to settle before the tasks that need them)
- **D1 — transmission formula for new data.** The legacy generators evaluate an unbounded trace
  formula whose subband-edge spikes are clipped. A generic multi-orbital engine can use the standard
  bounded Caroli form, but then legacy and new clouds differ in formula. Recommendation: Caroli for all
  atlas clouds (regenerate 7/9/square atlas clouds with the engine); keep legacy data for stage 3.
- **D2 — energy unit for unknown samples.** An axis in units of t presumes t is known; a measured
  spectrum of an unknown material is in eV. Recommendation: units of t for the simulated model space
  now; add an eV-axis variant before real measurements are located.
- **D3 — parameter choices**: lookup done (see plan); choose graphene NN + edge factor vs longer-range sets,
  hBN Galvani (GW gap) vs Ribeiro–Peres (GGA gap), MoS₂ GGA vs LDA, phosphorene 5- vs 10-hopping.
- **D4 — energy convention for MoS₂/phosphorene**: in units of the largest NN hopping MoS₂ spans about
  −1.1 t to +6.9 t with E = 0 near the valence-band top; 0–4 t misses most of its conduction band.
- **D5 — positive-only energies** drop the valence band of particle-hole asymmetric materials (MoS₂, phosphorene).
