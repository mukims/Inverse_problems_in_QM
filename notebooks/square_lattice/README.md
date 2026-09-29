# square_lattice/ — Square-lattice strip transport and models

| File | Role |
|---|---|
| `compute_leads_sq.py` | Lead surface Green's functions (`~/transmissions_sq/leads/leads_{l}.npy`) |
| `ca_sq.py` | Generates disordered spectra, one `.npy` per configuration (**use this generator**) |
| `combine_sq.py` | Stacks them into `conc_{c}.npy` + `conc_{c}_meta.csv` (row → config seed) |
| `pipeline_sq.py` | Leads → spectra → stacking in one command |
| `square_autoencoder.py` | Square-10 autoencoder and concentration probes (results in `sq_ae_results/`) |
| `CA.ipynb`, `device.ipynb` | Original notebooks (**`CA.ipynb` has a data-corrupting bug; see below**) |

## Data-corrupting bug in `CA.ipynb`
`unitcell_leads` is wrapped in `@lru_cache`, which returns the same NumPy array on every call, and
`unidevice` writes the impurity on-site energy into that array (`mat[i, i] = …`). Each impurity therefore
persists into every later unit cell. In the width-10 CSVs it produced, 57% of spectra are byte-identical
across concentrations and the concentration signal vanishes above c ≈ 25. `ca_sq.py` builds a fresh unit
cell per call and is correct (its output matches the pristine channel count exactly). The fix for the notebook
would be `mat = unitcell_leads(...).copy()`.

## Clean data
- Width 10: `~/transmissions_sq/size_10_combined/`, c = 5, 10 … 90, 10,000 configurations each (400 energies,
  E = 0–3.99 eV). Pristine: `~/transmissions_sq/pristine_10.npy`.
- Widths 15 and 20 (c = 5 … 60) exist for a separate project.

Generation runs ~7 configurations/s on all cores of an i7-13700 (400 energies each). Set
`OMP_NUM_THREADS=1` per worker to avoid BLAS oversubscription.

## Square-10 autoencoder (held-out seeds, `sq_ae_results/sq_ae_metrics.json`)
Concentration from the 64-d latent: MAE 3.58 (R² 0.963); PCA-64: 3.17; raw spectrum: 3.22.
Spikes of the transmission formula above the pristine channel count are clipped; they grow from 1% of
inputs at c = 5 to 29% at c = 90, but removing them changes the spectrum probe only from MAE 3.21 to 3.37.
