# Graphene (`graphene-ideal`)

Idealised nearest-neighbour graphene nanoribbons, both armchair (AGNR) and zigzag (ZGNR). These are Shazam's original reference library, and the legacy 7/9-AGNR concentration dataset that Stage 3 uses. Shared conventions (device, leads, store, Shazam input) are in the [materials README](../README.md).

## Physical model

| Item | Value |
|---|---|
| Orbitals | one p_z per carbon atom |
| Hopping | nearest neighbour only, **t = 2.7 eV** (Castro Neto et al., RMP 2009); enters H as −t |
| On-site | 0 (clean) |
| Edge relaxation | **none** (`edge_bond_factor = 1.0`). Son–Cohen–Louie's 12% stronger armchair edge bonds are *not* applied, hence "ideal". |
| E = 0 | Dirac point (charge neutrality) |
| Bands | within ±3t (±8.1 eV) |
| Window | E ≥ 0 only. The clean nearest-neighbour model is particle–hole symmetric, so nothing essential is lost. |
| `t_ev` | 2.7. It was introduced with the shared eV axis (BUILD-21); before that, everything was in units of t. |

**Known physics, checked by the tests and the store checks:**
- **AGNR families:** N = 3p and 3p+1 are gapped, and N = **3p+2 is metallic** in this model (5, 8, 11, 14, 20, 50). Measured onsets: N7 0.64 eV, N9 0.48 eV, N27 0.18 eV, N14 0.
- **N-AGNR band top:** t(1 + 2cos(π/(N+1))): N7 2.848 t (7.69 eV), N9 2.902 t (7.84 eV), never above 3t.
- **ZGNR:** a flat edge band at E = 0, so zigzag ribbons conduct from 0 eV. Band tops are 2.944–2.996 t (7.95–8.09 eV) for N7–N27.
- **Maximum channel count in the window:** N7 armchair 3, N9 armchair 4; zigzag N.

## Ribbon geometry

| Edge | Builder | Period | Atoms per cell | Width N |
|---|---|---|---|---|
| Armchair | `agnr_lib.unitcell` / `T1_matrix` / `rho_matrix` (legacy code path); `tbribbon.lattices.honeycomb_ribbon(N, "armchair")` gives the same lattice | 3 (bond length 1) | **2N** | N dimer lines across |
| Zigzag | `tbribbon.lattices.honeycomb_ribbon(N, "zigzag")` | √3 | **2N** | N zigzag chains |

`honeycomb_ribbon` builds the cell from geometry: sites of a honeycomb lattice with bond length 1, kept inside one period, with bonds wherever two sites are at distance 1.

**Bug #9.** The old `agnr_lib` cell was wrong for even armchair widths: a chain bond made a 4-ring on the last row, and the inter-cell hopping missed a row. It was fixed with a corrected "cell v2". Even widths (and widths > 31) now use recomputed leads in `~/atlas_store/leads/agnr_cell_v2/`. Odd widths are byte-identical to the historical code. Clouds made before the fix are quarantined in `~/atlas_store/quarantine_b4/` and must never be used.

## Transport

| | Armchair | Zigzag |
|---|---|---|
| Code | `agnr_lib.device_transmission` (via `generate_clouds._one_agnr`) | `tbribbon.transport.spectrum` |
| Formula | **legacy trace**, `nonlocal_mode="IL"`: `T = \|Tr[G̃_dd ρ G̃_rr ρ − ρ G̃nl ρ G̃nl]\|` | **Caroli**, `T = Tr[Γ_R G_N1 Γ_L G_N1†]` |
| Broadening | d = 1e-5 (device and leads) | η = 1e-6 (device), η = 1e-4 (leads) |
| Leads | precomputed Sancho–Rubio surface GFs, shape (300, 2N, 2N): odd N ≤ 31 from `~/Desktop/backup/agnr/size_N/leads_N.npy`; even N and N > 31 from `~/atlas_store/leads/agnr_cell_v2/` | computed per ribbon (`LeadCache`) |
| Stored energy grid | **0–2.99 t, step 0.01 t (300 points) = 0–8.07 eV** | **0–3.99 t, step 0.01 t (400 points) = 0–10.77 eV** |
| `meta.json` formula | `agnr_lib_IL_1e-5` | `caroli` |
| Clean-spectrum check | T = open channels (error 0.00) | T = open channels (error ≤ 4.3e-5) |

**Armchair spikes (legacy trace formula).** The formula is not bounded and spikes at subband openings and band edges:
- inside the band: up to 13.5 G₀ (7-AGNR) and 21.9 G₀ (9-AGNR), about 0.36% of points (Bug #6);
- above the band top: up to T ≈ 449 in some legacy configurations (BUILD-21).

The stored data keeps them. Downstream:
- Stage 3 clips `T / T_pristine` to [0, 1];
- Shazam's InputSpec v4 removes isolated spikes label-free (`T > 2m + 2`; BUILD-22);
- the store's median check masks ±4–5 channels around clean-spectrum steps.

Zigzag uses Caroli and has no spikes.

## Disorder

| Item | Value |
|---|---|
| Impurity | on-site **+V = 0.5 t = 1.35 eV** on a random carbon atom. In agnr_lib the impurity site's diagonal becomes `(E + i d − V)`. |
| Count | `n_imp = round(d × 2N × 100)`. N10: 10 / 20 / 40 / 80 at 0.5 / 1 / 2 / 4%. |
| Placement | armchair: `agnr_lib.chosen_for_config`, `RandomState(seed).choice` over the (cell, site) grid of 100 × 2N, no replacement. Zigzag: `tbribbon.disorder.impurity_shifts`, the same rule. Both are **nested** across densities. |
| Densities | 0.5%, 1%, 2%, 4% |
| Seeds | 0–999 per (ribbon, density). Split 0–699 / 700–849 / 850–999. |

## What exists

| Store or dataset | Contents | Build |
|---|---|---|
| `~/atlas_store/engine_v1/graphene-ideal/` | **31 ribbons:** armchair N 5–16, 20, 27, 31, 40, 50 (17); zigzag N 4–12, 16, 20, 27, 40, 50 (14). × 4 densities × 1,000 seeds = **124,000 spectra**. `check_store.py` ALL PASS (`engine_v1/report.json`). Generated on 16 workers in about 24.5 h. | FULL-1 / BUILD-15 |
| `~/atlas_store/smoke_v1/graphene-ideal/` | the same 31 ribbons × 4 densities × 50 seeds | SMOKE-2 |
| `~/atlas_store/reference_v1/graphene-ideal/armchair/N7, N9` | legacy 7/9-AGNR clouds imported into the atlas format | Phase 1 |
| `.../transmission_results/consolidated_data/size_7.npy` | **legacy 7-AGNR:** shape (34, 10000, 300). Concentrations c = 2, 4 … 68 impurities (index i ↔ c = 2(i+1)); row = configuration seed; E = 0–2.99 t. Density = c / 1,400. | `ca_agnr.py` (agnr_lib "IL", d = 1e-5) |
| `.../consolidated_data/size_9.npy` | **legacy 9-AGNR:** shape (49, 10000, 300); c = 2 … 98; density = c / 1,800 | same |
| repo root `7_agnr_pris.npy`, `9_agnr_pris.npy` | clean references, 300 points (git-ignored; copy in `data/raw/transmission_results/`) | Bug #3 regeneration |

Stage 3 uses seeds 0–2099 to train, 2100–2549 to calibrate and 2550–2999 to test, at every legacy concentration.

## Shazam
- Shazam uses `t_ev = 2.7` and puts the stored grids on the eV axis.
- Each ribbon's band top comes from `make_model` (`band_edges` of the clean ribbon) × 2.7.
- `atlas_v4` (BUILD-22) holds 2,000 references per ribbon from training seeds.

## Reproduce

```bash
export PYTHONPATH=notebooks/material_atlas:notebooks OMP_NUM_THREADS=1
~/miniconda3/envs/ml/bin/python -u notebooks/tbribbon/generate_clouds.py --store ~/atlas_store/engine_v1 \
    --grid sparse31 --spec-version v2 --n-seeds 1000 --n-jobs 16
~/miniconda3/envs/ml/bin/python notebooks/tbribbon/check_store.py --store ~/atlas_store/engine_v1
```

The generator refuses to put graphene armchair on an eV spec on purpose. Armchair data reaches the eV axis by rescaling, never by regeneration.

## History
- **Bug #2:** the train/test non-local convention and broadening were standardised to "IL", d = 1e-5.
- **Bug #3:** the clean references were regenerated.
- **Bug #6:** the legacy dataset is rounded to 3 decimals before dividing by pristine.
- **Bug #7:** splits are by seed.
- **Bug #9:** the even-width armchair cell was fixed.
- **BUILD-21/22:** the eV axis, and despiking of the legacy spikes.
