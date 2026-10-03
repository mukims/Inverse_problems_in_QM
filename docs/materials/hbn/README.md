# Hexagonal boron nitride (`hbn`)

Single-layer hBN nanoribbons: the graphene honeycomb lattice with boron and nitrogen on the two sublattices. Shared conventions (device, leads, store, Shazam input) are in the [materials README](../README.md).

## Physical model

| Item | Value |
|---|---|
| Orbitals | one p_z per atom |
| Hopping | nearest neighbour, **t = 2.30 eV**; enters H as −t |
| On-site | **+Δ on boron** (sublattice A), **−Δ on nitrogen** (sublattice B), **Δ = 3.625 eV** |
| Source | Galvani et al., PRB 94, 125303 (2016), fitted to the GW gap 2Δ = 7.25 eV |
| Code units | energies in units of t: `hbn_ribbon` calls `honeycomb_ribbon(N, edge, t=1, onsite_a=+Δ/t, onsite_b=−Δ/t)` with Δ/t = 1.5761 |
| `t_ev` | 2.30 |
| E = 0 | mid-gap (charge neutrality); the gap 2Δ = 7.25 eV is centred on it |
| Bulk bands | ±Δ to ±√(Δ² + 9t²) = ±3.625 to ±7.79 eV (±1.58 t to ±3.39 t) |
| Window | E ≥ 0 only, so **only the conduction band is in the data**. hBN is particle–hole symmetric, so the valence band is its mirror image. |

**Ribbon numbers** (computed from the code):

| Edge | N | Atoms per cell | Conduction onset | Band top | Max channels |
|---|---|---|---|---|---|
| armchair | 7 / 9 / 14 / 27 | 14 / 18 / 28 / 54 | 3.67 / 3.65 / 3.63 / 3.63 eV | 7.49 / 7.59 / 7.70 / 7.77 eV | 3 / 4 / 7 / 13 |
| zigzag | 7 / 9 / 14 / 27 | 14 / 18 / 28 / 54 | 3.63 eV (edge states at +Δ) | 7.68 / 7.72 / 7.76 / 7.79 eV | N |

In this plain model, zigzag edge states sit exactly at ±Δ rather than inside the gap (`PHYSICS.md`). Real hBN ribbons have edge-modified on-site energies, which this model does not include.

## Ribbon geometry
- **Builder:** `tbribbon.lattices.hbn_ribbon(N, edge)` → `honeycomb_ribbon`, the same geometry as graphene.
  - **Armchair:** N dimer lines, period 3.
  - **Zigzag:** N zigzag chains, period √3.
  - Both have **2N atoms per cell**.
- **Device:** 100 cells; n_atoms = 2N × 100.

## Transport
- **Caroli** recursive Green's function (`meta.json` formula `caroli`), device η = 1e-6. The H1 is not symmetric, so the generator routes it to Caroli automatically.
- **Leads:** Sancho–Rubio (`LeadCache`), η = 1e-4.
- **Clean check:** T equals the open-channel count, error ≤ 1.7e-4 (SMOKE-3 store report).

## Energy grid
- **Clean fingerprints** (`materials_v1`): 0–3.99 t, step 0.01 t (400 points).
- **Clouds** (eV stores):
  - stored grid: `E_t = 0.02·i / 2.30` for i = 0 … 415, i.e. 0–3.609 t in steps of 0.0087 t (**0–8.30 eV in 0.02 eV steps**);
  - only channels up to `band_top + 0.01 t` are computed;
  - the rest are exact zeros.

## Disorder

| Item | Value |
|---|---|
| Impurity | on-site **+V = 0.5 t = 1.15 eV** on a random atom (boron or nitrogen) |
| Count | `n_imp = round(d × 2N × 100)`; N14: 14 / 28 / 56 / 112 |
| Placement | `impurity_shifts(..., orbitals_per_site=1)`, `RandomState(seed).choice` without replacement, nested across densities |
| Densities | 0.5%, 1%, 2%, 4% |
| Seeds | full run 0–999 (split 0–699 / 700–849 / 850–999); smoke 0–49; probe 0–1 |

## What exists

| Store | Contents | Build |
|---|---|---|
| `~/atlas_store/materials_v1/hbn/` | clean spectra only, N7/9/14/27 × both edges, units-of-t grid | BUILD-19 |
| `~/atlas_store/materials_ev_v1/hbn/` | N7/9/14/27 × both edges × 4 densities × 50 seeds; ALL PASS | SMOKE-3 |
| `~/atlas_store/materials_ev_probe/hbn/` | N50 × both edges × 4 densities × 2 seeds; ALL PASS | SMOKE-3 timing probe |
| `~/atlas_store/materials_ev_full/hbn/` | **N7/9/14/27 × both edges × 4 densities × 1,000 seeds** | FULL-4 (running from 2026-10-03) |

**Cost:** single-core seconds per spectrum (SMOKE-3), for N7 / 9 / 14 / 27 / 50:

| Edge | N7 | N9 | N14 | N27 | N50 |
|---|---|---|---|---|---|
| armchair | 0.72 | 0.86 | 1.77 | 10.4 | 30.5 |
| zigzag | 0.66 | 0.88 | 1.89 | 10.3 | 32.3 |

In FULL-4 on 16 workers, the batch wall time was 0.09 s per spectrum (N7), 0.14 s (N9) and 0.32 s (N14).

## Reproduce

```bash
export PYTHONPATH=notebooks/material_atlas:notebooks OMP_NUM_THREADS=1
~/miniconda3/envs/ml/bin/python -u notebooks/tbribbon/generate_clouds.py --store ~/atlas_store/materials_ev_full --grid materials \
    --materials hbn --widths 7,9,14,27 --spec-version v3 --n-seeds 1000 --n-jobs 16
~/miniconda3/envs/ml/bin/python notebooks/tbribbon/check_store.py --store ~/atlas_store/materials_ev_full --out <report>.json
```

## Caveats and choices
- **Parameter set (D3):** Galvani 2016 (GW gap) was chosen over Ribeiro–Peres 2011 (t = 2.33 eV, Δ = 1.96 eV, which fits the GGA valence band). There is no second-neighbour hopping (Galvani's optional t₂ = 0.096 eV is not used).
- **V scaling:** the impurity strength follows the project convention V = 0.5 t, so in eV it differs between materials (graphene 1.35 eV, hBN 1.15 eV).
