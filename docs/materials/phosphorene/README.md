# Phosphorene (`phosphorene`)

Monolayer black phosphorus nanoribbons in the five-hopping tight-binding model. Shared conventions (device, leads, store, Shazam input) are in the [materials README](../README.md).

## Physical model

| Item | Value |
|---|---|
| Orbitals | one effective orbital per P atom |
| Source | Rudenko & Katsnelson, PRB 89, 201408(R) (2014), a GW fit accurate within about 0.3 eV of the band edges; mapped onto a honeycomb grid as in Ezawa (2014) |
| Hoppings (eV) | **t1 = −1.220, t2 = +3.665, t3 = −0.205, t4 = −0.105, t5 = −0.055** |
| Code units | energies in units of **t_ref = \|t2\| = 3.665 eV** (`t_ev = 3.665`) |
| On-site | −4·t4 / t_ref (= +0.42 eV) on every site, which moves mid-gap from −0.42 eV to **E = 0** |
| E = 0 | mid-gap (decision D4) |
| Bulk gap | 1.52 eV in the model (4t1 + 2t2 + 4t3 + 2t5), direct at Γ (GW: 1.60 eV) |
| Window | E ≥ 0 only, so **the valence band is not in the data** (decision D5) |

**Honeycomb mapping** (`tbribbon.lattices.phosphorene_ribbon`): the puckered lattice is projected onto a honeycomb grid with bond length 1, and each hopping is assigned by the in-plane displacement (|dx|, |dy|) between two sites:

| Hopping | \|dx\| | \|dy\| |
|---|---|---|
| t1 | 0.5 | √3/2 |
| t2 | 1.0 | 0 |
| t4 | 1.5 | √3/2 |
| t5 | 2.0 | 0 |
| t3 | 2.5 | √3/2 |

Hoppings within a cell go to `H0`; those to the next cell, along the ribbon axis, go to `H1`.

**Ribbon numbers** (computed from the code):

| Edge | N | Atoms per cell | Conduction onset (above mid-gap) | Band top | Max channels |
|---|---|---|---|---|---|
| armchair | 7 / 9 / 14 / 27 | 14 / 18 / 28 / 54 | 1.01 / 0.92 / 0.84 / 0.78 eV | 7.05 / 7.14 / 7.23 / 7.28 eV | 4 / 5 / 7 / 14 |
| zigzag | 7 / 9 / 14 / 27 | 14 / 18 / 28 / 54 | 0.00 / 0.01 / 0.09 / 0.11 eV (in-gap edge band) | 7.08 / 7.17 / 7.24 / 7.28 eV | 4 / 5 / 8 / 15 |

- **Armchair** ribbons have no in-gap edge states. Their gap is about 2 × onset, so 2.0 eV at N7 falls towards the bulk 1.52 eV plus confinement: armchair N30 gives 1.549 eV, checked by `test_phosphorene_wide_armchair_gap_approaches_bulk`.
- **Zigzag** ribbons carry the known quasi-flat edge band inside the gap.

**Atoms per cell.** In this honeycomb mapping the cell has **2N atoms** (as in graphene). The physical rectangular cell of phosphorene holds 4 atoms, but the code's ribbon of width N has 2N sites per period, and impurity counts use 2N × 100 atoms. The SMOKE-3 LOGBOOK table says "4N"; the code and the stored impurity counts say 2N.

## Ribbon geometry
- **Armchair:** periodic along x, period 3.
- **Zigzag:** periodic along y, period √3.

The cuts are the same as graphene's on the mapped grid, and the naming matches physical phosphorene: armchair is gapped with no edge band, and zigzag has the edge band.

## Transport
- **Caroli**, device η = 1e-6, `meta.json` formula `caroli`.
- **Leads:** Sancho–Rubio, η = 1e-4.
- **Clean check:** T equals the open-channel count, error < 6e-4 (BUILD-19) and ≤ 1.7e-4 (SMOKE-3).

## Energy grid
- **Clean fingerprints** (`materials_v1`): 0–3.99 t.
- **Clouds:**
  - stored grid: `E_t = 0.02·i / 3.665` for i = 0 … 415, i.e. 0–2.265 t in steps of 0.00546 t (**0–8.30 eV in 0.02 eV steps**);
  - computed up to `band_top + 0.01 t`;
  - exact zeros above.

## Disorder

| Item | Value |
|---|---|
| Impurity | on-site **+V = 0.5 t = 1.83 eV** on a random P atom |
| Count | `n_imp = round(d × 2N × 100)`; N14: 14 / 28 / 56 / 112 |
| Placement | `impurity_shifts`, `RandomState(seed).choice` without replacement, nested |
| Densities | 0.5%, 1%, 2%, 4% |
| Seeds | full run 0–999 (split 0–699 / 700–849 / 850–999); smoke 0–49; probe 0–1 |

## What exists

| Store | Contents | Build |
|---|---|---|
| `~/atlas_store/materials_v1/phosphorene/` | clean spectra only, N7/9/14/27 × both edges | BUILD-19 |
| `~/atlas_store/materials_ev_v1/phosphorene/` | N7/9/14/27 × both edges × 4 densities × 50 seeds; ALL PASS | SMOKE-3 |
| `~/atlas_store/materials_ev_probe/phosphorene/` | N50 × both edges × 4 × 2 seeds | SMOKE-3 probe |
| `~/atlas_store/materials_ev_full/phosphorene/` | **N7/9/14/27 × both edges × 4 densities × 1,000 seeds** | FULL-4 |

**Cost:** single-core seconds per spectrum (SMOKE-3):

| Edge | N7 | N9 | N14 | N27 | N50 |
|---|---|---|---|---|---|
| armchair | 0.57 | 0.77 | 1.63 | 9.5 | 28.5 |
| zigzag | 0.57 | 0.78 | 1.62 | 9.5 | 28.5 |

## Reproduce

```bash
export PYTHONPATH=notebooks/material_atlas:notebooks OMP_NUM_THREADS=1
~/miniconda3/envs/ml/bin/python -u notebooks/tbribbon/generate_clouds.py --store ~/atlas_store/materials_ev_full --grid materials \
    --materials phosphorene --widths 7,9,14,27 --spec-version v3 --n-seeds 1000 --n-jobs 16
```

## Caveats and choices
- **Parameter set (D3):** five hoppings (2014). The ten-hopping GW0 set (Rudenko, Yuan, Katsnelson 2015; gap 1.84 eV, valid over several eV) is not used.
- **Anisotropy:** the mapping keeps phosphorene's strong anisotropy (electron masses 0.164 vs 0.873 m₀).
- **V is large:** V = 0.5 × 3.665 = 1.83 eV is the largest impurity strength in eV of any material, because the project scales V with the largest hopping.
- **Valence band:** absent (D5).
