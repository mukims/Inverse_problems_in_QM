# Triangular lattice (`triangular`, toy model)

An idealised single-orbital triangular lattice. It is a toy lattice for testing whether Shazam separates lattice geometry, not a real material. It shares its lattice with MoS₂'s Mo sublattice. Shared conventions (device, leads, store, Shazam input) are in the [materials README](../README.md).

## Physical model

| Item | Value |
|---|---|
| Orbitals | one per site |
| Hopping | nearest neighbour **t = 1** (six neighbours), enters H as −t; on-site 0 |
| Bulk dispersion | E(k) = −2t [cos k_x + 2 cos(k_x/2) cos(√3 k_y/2)], spanning **[−6t, +3t]** |
| `t_ev` | **1.0, arbitrary**: there is no physical material behind it, so its position on the eV axis is a convention |
| E = 0 | inside the band, so ribbons conduct from E = 0 |
| Window | E ≥ 0, so only the top part of the band (0 to 3t) is in the data |

**Ribbon numbers** (computed from the code):

| Edge | N | Sites per cell | Band top | Max channels |
|---|---|---|---|---|
| zigzag | 7 / 9 / 14 / 27 | 7 / 9 / 14 / 27 | 2.85 / 2.90 / 2.96 / 2.99 t | N |
| armchair | 7 / 9 / 14 / 27 | 14 / 18 / 28 / 54 | 2.90 / 2.93 / 2.97 / 2.99 t | N |

**Verified:** 0.0% of ribbon states fall outside the projected bulk bands for zigzag N30 and armchair N15 (`test_triangular_ribbon_bands_lie_in_bulk_projection`). Clean T equals the open-channel count to < 4e-4 on flat plateaus, and T < 1e-6 above the band top (BUILD-20).

## Ribbon geometry (`tbribbon.lattices.triangular_ribbon`)
- **Zigzag** (also accepted as `strip`): N rows across y, spaced √3/2 apart, period 1 along x, one site per row (**N sites per cell**). Each site bonds to its own image in the next cell, and rows alternate which diagonal neighbour lies in the next cell.
- **Armchair:** N columns across x, period √3 along y. Column j holds A_j at (j, 0) and B_j at (j + ½, √3/2) (**2N sites per cell**).

## Transport
- **Caroli** (the H1 is not symmetric), device η = 1e-6, `meta.json` formula `caroli`.
- **Leads:** Sancho–Rubio, η = 1e-4.

## Energy grid
- **Clean fingerprints** (`materials_v1`): 0–3.99 t.
- **Clouds:**
  - stored grid: 0–8.30 (t_ev = 1) in 0.02 steps (416 points);
  - computed up to `band_top + 0.01`, about 3, so 150 of 416 channels;
  - exact zeros above.

## Disorder

| Item | Value |
|---|---|
| Impurity | on-site **+V = 0.5 t** (0.5 on the eV axis) on a random site |
| Count | `n_imp = round(d × n_sites)` with n_sites = 100 × (N for zigzag, 2N for armchair). Zigzag N14: 7 / 14 / 28 / 56; armchair N14: 14 / 28 / 56 / 112. |
| Placement | `impurity_shifts`, `RandomState(seed).choice` without replacement, nested |
| Densities | 0.5%, 1%, 2%, 4% |
| Seeds | full run 0–999 (split 0–699 / 700–849 / 850–999); smoke 0–49; probe 0–1 |

## What exists

| Store | Contents | Build |
|---|---|---|
| `~/atlas_store/materials_v1/triangular/` | clean spectra only, N4/6/8/10/12 × both edges | BUILD-20 |
| `~/atlas_store/materials_ev_v1/triangular/` | N7/9/14/27 × both edges × 4 densities × 50 seeds; ALL PASS | SMOKE-3 |
| `~/atlas_store/materials_ev_probe/triangular/` | N50 × both edges × 4 × 2 seeds | SMOKE-3 probe |
| `~/atlas_store/materials_ev_full/triangular/` | **N7/9/14/27 × both edges × 4 densities × 1,000 seeds** | FULL-4 |

**Cost:** single-core seconds per spectrum (SMOKE-3):

| Edge | N7 | N9 | N14 | N27 | N50 |
|---|---|---|---|---|---|
| armchair | 0.23 | 0.31 | 0.68 | 3.89 | 11.8 |
| zigzag | 0.15 | 0.17 | 0.23 | 0.63 | 1.98 |

## Reproduce

```bash
export PYTHONPATH=notebooks/material_atlas:notebooks OMP_NUM_THREADS=1
~/miniconda3/envs/ml/bin/python -u notebooks/tbribbon/generate_clouds.py --store ~/atlas_store/materials_ev_full --grid materials \
    --materials triangular --widths 7,9,14,27 --spec-version v3 --n-seeds 1000 --n-jobs 16
```

## Caveats
- **Arbitrary energy scale.** With t_ev = 1, the triangular band ends below 3 on the eV axis, like MoS₂'s (2.72 eV). When Shazam compares materials on the eV axis, similarity between the two may come from the shared energy scale rather than the shared lattice. The closest-lattice analysis (BUILD-23/24) therefore also reports a band-top-normalised distance (`shape_profile`) that removes the scale.
- **Toy model:** no spin, no further-neighbour hopping, and impurities are simple on-site shifts.
