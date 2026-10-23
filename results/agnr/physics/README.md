# physics/ — Forward simulation & data generation

Everything that **computes** a transmission spectrum from first principles, rather
than learning one. This is the ground truth every model in the other folders is
trained against.

The method is tight-binding + NEGF: build the ribbon Hamiltonian, attach semi-infinite
leads via their surface Green's functions, then get `T(E)` from the
Landauer–Büttiker formula using a recursive Green's function sweep.

---

## Files

| File | Runnable | What it does |
|---|---|---|
| **`agnr_lib.py`** | no (library) | **The physics core — start here.** Self-contained, consolidated implementation: energy grids, unit cells, `beta`/`T1`/`rho` matrices, Sancho–Rubio lead decimation, lead caching, transmission. |
| `agnr.py` | no (library) | Older physics implementation. Kept because `defect_reconstruction/` still imports it. Prefer `agnr_lib.py` for new work. |
| `ca_agnr.py` | yes | Configuration-averaged variant used to produce training sets. **Has no argparse** — running it executes real work immediately. |
| `build_leads_agnr.py` | yes | Precomputes lead surface Green's functions by Sancho–Rubio decimation and caches them as `leads_<m>.npy`. Not run automatically; this is the "I need more lead widths" job. |
| `build_pristine_library.py` | yes | Generates impurity-free reference spectra per width. These are the denominators for normalisation and the basis for band-gap/width identification. |
| `generate_test_data.py` | yes | Produces held-out test spectra for chosen concentrations and random impurity configurations. |

---

## Why there are three physics files

`agnr.py` and `ca_agnr.py` were written first and drifted apart — they used
different non-local operator conventions (`g_L·ρ·I_L` vs `g_R·T†·I_L`) and different
broadening (`1e-5` vs `1e-4`), which made training and test data subtly inconsistent.

`agnr_lib.py` is the consolidation: one implementation with an explicit
`nonlocal_mode` parameter, standardised on `d = 1e-5` and `"IL"`. **New code should
import `agnr_lib`.** The other two remain only so existing scripts keep working.

---

## Typical use

```bash
cd physics
conda run -n ml python build_pristine_library.py --help
```
```bash
cd physics && conda run -n ml python generate_test_data.py --size 7 --nconfigs 100
```

## Gotchas

- **Leads are expensive.** `build_leads_agnr.py` exists so you compute them once and
  cache. If a script is mysteriously slow, it is probably recomputing leads.
- Pristine spectra (`7_agnr_pris.npy`, `9_agnr_pris.npy`) live at the **repository
  root**, reached via `Path(__file__).resolve().parents[2]`.
- `ca_agnr.py` ignores `--help` and starts computing. Read it before running it.
