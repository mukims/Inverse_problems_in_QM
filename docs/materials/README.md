# Materials: how the transmission data is generated

This folder documents every material in the dataset: its physical model, how its ribbons are built, how transmission is computed, how disorder is added, and where the data lives. There is one README per material:

| Material | Model id | Lattice | Energy unit t_ev | README |
|---|---|---|---|---|
| Graphene (idealised) | `graphene-ideal` | honeycomb, 1 orbital | 2.7 eV | [graphene/](graphene/README.md) |
| Hexagonal boron nitride | `hbn` | honeycomb, ±Δ on-site | 2.30 eV | [hbn/](hbn/README.md) |
| Phosphorene (black phosphorus) | `phosphorene` | puckered honeycomb, 5 hoppings | 3.665 eV | [phosphorene/](phosphorene/README.md) |
| MoS₂ | `mos2` | triangular Mo lattice, 3 d-orbitals | 1.0 eV (energies in eV) | [mos2/](mos2/README.md) |
| Triangular lattice (toy) | `triangular` | triangular, 1 orbital | 1.0 (arbitrary) | [triangular/](triangular/README.md) |
| Square strip (toy) | `square` | square, 1 orbital | 1.0 (arbitrary) | [square/](square/README.md) |

Sources of truth:
- **Code:** `notebooks/tbribbon/` (lattices, transport, leads, disorder, generator) and `notebooks/agnr/physics/agnr_lib.py` (graphene armchair).
- **Results:** `LOGBOOK.md` (builds, bugs).
- **Literature parameters:** `docs/superpowers/handoff/material-atlas/MATERIALS.md` and `PHYSICS.md`.

When the code and a document disagree, the code wins. The numbers here were checked against the code on 2026-10-03.

Paths under `docs/superpowers/` are local planning and handoff notes. They are deliberately kept out of the repository.

---

## The pipeline, shared by every material

```
material parameters ──► ribbon unit cell (H0, H1) ──► clean leads (surface Green's functions)
        │                                                      │
        └──► 100-cell device + random on-site impurities ──────┴──► T(E) on an energy grid ──► store
                                                                                                 │
                                         Shazam reads it: E × t_ev → eV axis, label-free transform ◄┘
```

### 1. Ribbon and device
- **Ribbon.** A ribbon is a 1-D periodic strip of a 2-D lattice. One period (the unit cell) has Hamiltonian `H0`, and `H1` couples a cell to the next one. The block convention is `H[n, n] = H0`, `H[n, n+1] = H1`, `H[n+1, n] = H1†`.
- **Device.** The device is **100 unit cells** of the ribbon, sitting between two semi-infinite **leads made of the same clean ribbon**.
- **Width N.** N counts atomic rows across the ribbon: dimer lines (armchair), zigzag chains (zigzag), Mo rows (MoS₂). Atoms per unit cell are listed in each material's README.
- **Hopping and energy units.** Hopping enters as −t on the bond. Energies are stored in **units of the material's t** (`t_ev` converts to eV), except MoS₂ (t_ev = 1, so its stored energies are already eV).
- **E = 0** is charge neutrality, which is mid-gap for gapped materials (decision D4). **Only E ≥ 0 is computed** (decision D5), so the valence bands of MoS₂ and phosphorene are not in the data.

### 2. Leads (`notebooks/tbribbon/leads.py`)
- **Method.** Sancho–Rubio decimation gives the surface Green's function of each semi-infinite clean lead, with broadening **η_lead = 1e-4**, tolerance 1e-10 and at most 300 iterations.
- **Right lead** (extends to +∞ through `H1`): `g_R = surface_gf(E, H0, H1)`.
- **Left lead** (through `H1†`): `g_L = surface_gf(E, H0, H1†)`.
- **Self-energies:** `Σ_R = H1 g_R H1†` and `Σ_L = H1† g_L H1`.
- **Broadening:** `Γ = i(Σ − Σ†)`.
- **Caching.** Leads are computed once per ribbon and energy grid (`LeadCache`) and shared by all disorder configurations.

### 3. Transmission formulas (`notebooks/tbribbon/transport.py`)

| Formula | Used for | Device broadening η | Notes |
|---|---|---|---|
| **Caroli** `T = Tr[Γ_R G_N1 Γ_L G_N1†]` | all new materials, graphene zigzag | **1e-6** | Recursive Green's function over the 100 cells. Bounded by the number of open channels. Clean T equals the channel count except within about half a grid step of a subband edge. |
| **Legacy trace** `T = \|Tr[G̃_LL H1 G̃_RR H1 − H1 G̃nl H1 G̃nl]\|`, with `G̃ = G − G†` | square strip | **1e-3** | Reproduces `notebooks/square_lattice/ca_sq.py`. It requires a symmetric `H1`. It is not bounded: it spikes above the channel count at subband edges. |
| **agnr_lib legacy trace** (`nonlocal_mode="IL"`) | graphene armchair | **1e-5** (device and leads) | `notebooks/agnr/physics/agnr_lib.py`, its own precomputed leads and 0–2.99 t grid. Same spike behaviour. See the [graphene README](graphene/README.md). |

The generator routes automatically (`generate_clouds.py`):
- graphene armchair goes to agnr_lib;
- any other ribbon uses the legacy trace only when it is requested *and* `H1` is symmetric (the square strip);
- everything else uses Caroli.

The formula is recorded per ribbon in `meta.json` (`pristine_formula` and each cloud's `formula`), and the store refuses to mix formulas within a ribbon.

### 4. Energy grids
There are two generations of stored grid:

| Grid | Who | Points | Range |
|---|---|---|---|
| **Units of t** (InputSpec v2) | graphene zigzag, square strip, `materials_v1` fingerprints | 400 | 0–3.99 t, step 0.01 t |
| | graphene armchair (agnr_lib) | 300 | 0–2.99 t, step 0.01 t |
| **Shared eV axis** (InputSpec v3/v4, generation since SMOKE-3) | hBN, phosphorene, MoS₂, triangular | 416 | 0–8.30 eV, step 0.02 eV. Stored in units of t as `E_eV / t_ev`, so the stored step is 0.02 / t_ev. |

- **Band-top truncation.** On the eV grid only channels up to `band_top + 0.01 t` are computed; the rest are written as **exact zeros**.
- **Why zeros are exact.** Above the top of the clean band there are no propagating lead states, so T = 0. This saves 3× on MoS₂ and the triangular lattice, whose bands end below 3 eV.

### 5. Disorder (`notebooks/tbribbon/disorder.py`; `agnr_lib` for graphene armchair)
- **Impurity:** an on-site energy shift **+V** on a randomly chosen atom. V = 0.5 t for every material except MoS₂ (V = 0.5 × t₂ = 0.2535 eV). The values in eV are in each README.
- **Multi-orbital atoms** (MoS₂): the impurity shifts **all orbitals of the chosen atom** (Bug #11).
- **Density d:** impurities per atom. The count is `n_imp = max(1, round(d × n_atoms))` with `n_atoms = 100 cells × atoms per cell`. The stored density key is the actual `n_imp / n_atoms`.
- **Placement:** `np.random.RandomState(seed).choice(n_atoms, n_imp, replace=False)`. This takes a prefix of one fixed permutation, so **a seed's impurity set at a higher density contains its set at a lower density** (nested). The same seed at different densities is therefore a near-twin, which is why data must always be split **by seed** (Bug #7).
- **Densities:** 0.5%, 1%, 2%, 4% (decision F3; a 6% level may be appended later).
- **Configurations:** seeds 0–999 per (ribbon, density) for training-grade data. The split is **0–699 train, 700–849 validation, 850–999 test** (AGENTS.md).

### 6. Store layout (`notebooks/material_atlas/atlaslib/store.py`)

```
~/atlas_store/<store>/<material>/<edge>/N<width>/
    energies_t.npy              stored energy grid (units of the material's t)
    pristine.npy                clean T(E)
    cloud_d0.0050.npy           (n_seeds, n_energies) disordered T(E), one row per seed
    cloud_d0.0050_seeds.npy     the seed of each row
    ...                         one pair per density
    meta.json                   pristine_formula; per cloud: density, n_impurities, n, formula, t_spectrum_sec
```

`CloudStore.write_cloud` refuses:
- non-finite values;
- duplicate seeds;
- a disordered median above the clean spectrum by more than 0.05 away from subband steps;
- identical spectra at two densities (a generator bug);
- a formula that differs from the ribbon's clean spectrum.

`notebooks/tbribbon/check_store.py` validates a whole store (report written next to the logs):
- clean T equals the open-channel count from the band structure;
- medians stay at or below pristine away from steps;
- spike-robust per-seed means;
- no cross-density duplicates;
- nested seeds.

### 7. How Shazam reads the data (label-free)
- `atlaslib.energy.on_axis` multiplies stored energies and the band top by `t_ev` to put them in eV.
- `InputSpec(version="v4").to_input` then, for every spectrum alike:
  1. rounds T to 3 decimals;
  2. despikes on the native grid (a channel with T > 2 × its 5-channel median + 2 is replaced by that median);
  3. resamples onto the 416 eV channels (0–8.30 eV);
  4. zeros everything above the band top;
  5. clips at 64;
  6. applies `log1p(T) / log1p(64)`.
- Nothing material-specific is applied before identification.

## The stores

| Store | What | Status |
|---|---|---|
| `engine_v1` | Graphene: 31 ribbons × 4 densities × 1,000 seeds (124,000 spectra), FULL-1. Plus a clean-only square N10. | Production; Shazam's graphene references |
| `smoke_v1` | Graphene: the same 31 ribbons × 4 densities × 50 seeds (SMOKE-2) | Superseded by engine_v1 |
| `reference_v1` | Legacy 7/9-AGNR clouds imported from the consolidated dataset (Phase 1) | Reference only |
| `quarantine_b4` | Even-width armchair (N6, N8) generated before the Bug #9 fix | **Never use** |
| `novelty_v1` | Square strip N10: 4 densities × 150 seeds | The unseen-lattice test |
| `materials_v1` | Clean fingerprints only (no clouds) of hBN, phosphorene, MoS₂ (N7/9/14/27) and triangular (N4–12), on the units-of-t grid | Superseded by the eV stores |
| `materials_ev_v1` | SMOKE-3: hBN, phosphorene, MoS₂, triangular × armchair/zigzag × N7/9/14/27 × 4 densities × 50 seeds, on the eV grid | Smoke; all 32 pass |
| `materials_ev_probe` | N50 timing probe: the 4 materials × 2 edges × 4 densities × 2 seeds | Cost estimate only |
| `materials_ev_full` | **FULL-4 (running since 2026-10-03 04:18):** the 4 materials × 2 edges × N7/9/14, then hBN, phosphorene and triangular × N27, all × 4 densities × 1,000 seeds | Production for new materials |
| `leads/agnr_cell_v2/` | Recomputed armchair leads for even widths and widths > 31 (Bug #9) | Used by agnr_lib |

## Decisions on record

| Decision | Choice | Where |
|---|---|---|
| D2: energy unit | shared eV axis | `docs/superpowers/plans/2026-10-02-shared-ev-axis.md` |
| D3: parameter sets | graphene NN t = 2.7 eV without edge relaxation; hBN Galvani 2016; MoS₂ Liu 2013 GGA, NN, no spin–orbit; phosphorene Rudenko 2014, 5 hoppings | `MATERIALS.md`, LOGBOOK BUILD-19 |
| D4: E = 0 | mid-gap or charge neutrality for all | `2026-10-02-full-run-decisions.md`, F4 |
| D5: window | positive energies only | same, F5 |
| F1–F3: widths, configurations, densities | N7/9/14 for all + N27 for hBN, phosphorene, triangular; 1,000 configurations; 0.5–4% | same |

## Reproducing a store
Set `OMP_NUM_THREADS=1`, use `--n-jobs 16` when the machine is otherwise idle, and run from the repo root with `PYTHONPATH=notebooks/material_atlas:notebooks`. The exact command for each material is in its README. Always validate afterwards with `check_store.py`.
