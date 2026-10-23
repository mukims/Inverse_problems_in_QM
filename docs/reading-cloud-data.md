# Reading the cloud data with NumPy

Every ribbon in the data tree is a folder of plain `.npy`, `.npz` and `.json` files, so NumPy is all you need. This page
shows how to read them without any of the project's code. (`atlaslib.CloudStore` wraps the same files; see the end.)

A **cloud** is the set of transmission spectra of one ribbon at one impurity density: one spectrum per random impurity
configuration ("seed").

## Where the files are

```
data/<material>/<edge>/N<width>/      e.g. data/hbn/armchair/N10/
```

`data/INDEX.csv` lists all 342 ribbons with their device length, number of densities and seeds per cloud.
`TRANSMISSION_DATA` is the usual name for the path to `data/`; the examples below use it.

| File | Shape and type | Content |
|---|---|---|
| `energies_t.npy` | `(n_energies,)` float64 | energy of each channel, in units of the material's hopping `t` (see below) |
| `pristine.npy` | `(n_energies,)` float64 | clean transmission, no impurities |
| `cloud_d<density>.npy` | `(n_seeds, n_energies)` float64 | disordered spectra; row `i` belongs to seed `seeds[i]` |
| `cloud_d<density>_seeds.npy` | `(n_seeds,)` int64 | the configuration seed of each row |
| `cloud_d<density>_config.npz` | see below | where the impurities sit, row by row (most clouds) |
| `meta.json` | JSON | device length, lead broadening, formula, and one entry per density |

Transmission `T(E)` is dimensionless: the number of open conduction channels at that energy, in units of the
conductance quantum. A clean spectrum is a staircase of integers; impurities pull it down.

`<density>` in a file name is the density rounded to 4 decimals (`d0.0100` is 1%). The exact value is in `meta.json`.

## Load one ribbon

```python
import json
import os
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("TRANSMISSION_DATA", "data"))
ribbon = DATA / "hbn" / "armchair" / "N10"

E_t = np.load(ribbon / "energies_t.npy")            # (416,)
clean = np.load(ribbon / "pristine.npy")            # (416,)
meta = json.loads((ribbon / "meta.json").read_text())

print(meta["pristine_n_cells"], "unit cells,", len(meta["clouds"]), "densities")
for key, info in sorted(meta["clouds"].items()):
    print(key, info["density"], info["n_impurities"], "impurities,", info["n"], "seeds")

spectra = np.load(ribbon / "cloud_d0.0100.npy", mmap_mode="r")   # (1000, 416), read lazily
seeds = np.load(ribbon / "cloud_d0.0100_seeds.npy")              # (1000,)
print(spectra.shape, seeds[:5])
```

`mmap_mode="r"` maps the file instead of reading it, which keeps memory low when you only need some rows. Wrap the
result in `np.asarray(...)` to get an ordinary array.

To read every cloud of a ribbon, take the keys from `meta.json`:

```python
clouds = {info["density"]: np.load(ribbon / f"cloud_{key}.npy") for key, info in meta["clouds"].items()}
```

## Energies: units of t, and how to get eV

`energies_t.npy` is in units of each material's nearest-neighbour hopping `t`. Multiply by `t_ev` for electron-volts:

| Material | `t_ev` (eV) | Channels | Energy range |
|---|---|---|---|
| graphene-ideal | 2.7 | 300 (armchair), 400 (zigzag) | 0–2.99 t and 0–3.99 t |
| hbn | 2.30 | 416 | 0–8.3 eV |
| phosphorene | 3.665 | 416 | 0–8.3 eV |
| silicene | 1.067 | 416 | 0–8.3 eV |
| germanene | 0.991 | 416 | 0–8.3 eV |
| mos2, ws2, mose2, wse2 | 1.0 (already eV) | 416 | 0–8.3 eV |
| kagome, lieb, checkerboard | 1.0 | 416 | 0–8.3 t |
| triangular | 1.0 | 416 (N4: 400) | 0–8.3 t (N4: 0–3.99 t) |
| square | 1.0 | 400 | 0–3.99 t |

```python
T_EV = {"graphene-ideal": 2.7, "hbn": 2.30, "phosphorene": 3.665, "silicene": 1.067, "germanene": 0.991}
E_ev = E_t * T_EV.get("hbn", 1.0)                   # 0 ... 8.3 eV
```

The 416-channel ribbons share one axis in eV (0 to 8.3 eV in 0.02 eV steps), so their spectra can be compared channel by
channel. Graphene and square use their own grids: put them on a common axis before comparing (`np.interp`).

## Seeds: always split by seed

Row `i` of a cloud is configuration seed `seeds[i]`, 0 to 999 (0 to 149 for the square strip). The same seed gives
**nested** impurity sets across densities: seed 7 at 2% contains the impurities of seed 7 at 1%. Rows of the same seed
are therefore related, so split train, validation and test by seed, never by random rows:

```python
train = spectra[seeds <= 699]
val = spectra[(seeds >= 700) & (seeds <= 849)]
test = spectra[seeds >= 850]
```

Round `T` to 3 decimals before dividing by the clean spectrum, so channels in a band gap become exact zeros:

```python
ratio = np.round(spectra, 3) / np.where(np.round(clean, 3) > 0, np.round(clean, 3), 1.0)
```

## Extra densities on five ribbons

`hbn/armchair/N9`, `mos2/zigzag/N9`, `phosphorene/armchair/N9`, `triangular/zigzag/N9` and
`graphene-ideal/armchair/N13` hold 24 densities: the 4 of the catalogue plus 20 from a later density sweep. The extra ones
say so in `meta.json`:

```python
ribbon9 = DATA / "mos2" / "zigzag" / "N9"
meta9 = json.loads((ribbon9 / "meta.json").read_text())
catalogue = {k: v for k, v in meta9["clouds"].items() if "source" not in v}     # the 4 the lookup and atlas use
extra = {k: v for k, v in meta9["clouds"].items() if v.get("source") == "conc_v2"}
print(len(catalogue), "catalogue densities,", len(extra), "extra")
```

Keep only the catalogue densities if you want to reproduce the lookup and autoencoder results in `LOGBOOK.md`.

## Where the impurities are: `cloud_d<density>_config.npz`

Most clouds come with the exact impurity positions behind every spectrum (the 100 extra-density clouds above do not).
Rows line up with the cloud's seeds.

```python
with np.load(ribbon / "cloud_d0.0100_config.npz") as z:
    print(z.files)
    xy = z["coords"]                  # (n_seeds, n_impurities, 2): x along the ribbon, y across
    assert np.array_equal(z["seeds"], seeds)
```

| Key | Shape | Meaning |
|---|---|---|
| `coords` | `(n_seeds, n_imp, 2)` float32 | impurity positions, in lattice units (bond length 1) |
| `atom_indices` | `(n_seeds, n_imp)` | atom index in the device |
| `cell_indices`, `site_indices` | `(n_seeds, n_imp)` | unit cell along the ribbon, and site inside the cell |
| `sublattice` | `(n_seeds, n_imp)` int8 | sublattice of each impurity |
| `host_onsite_t`, `host_onsite_ev` | `(n_seeds, n_imp)` | on-site energy of the clean host atom, in t and in eV |
| `impurity_onsite_t`, `impurity_onsite_ev` | `(n_seeds, n_imp)` | on-site energy with the impurity (host + shift) |
| `defect_v_t`, `defect_v_ev` | scalar | the impurity shift |
| `seeds` | `(n_seeds,)` | the same seeds as `cloud_d<density>_seeds.npy` |
| `n_cells`, `n_impurities`, `density`, `model_id` | scalar | the device this describes |

## Loop over all ribbons

```python
import csv

with open(DATA / "INDEX.csv") as fh:
    for row in csv.DictReader(fh):
        if row["n_densities"] == "0":
            continue                          # triangular N4: a clean spectrum only
        folder = DATA / row["model_id"]
        E_t = np.load(folder / "energies_t.npy")
        # ... read clouds as above
```

## Older formats (`legacy/` folders)

**7- and 9-AGNR, consolidated** (`data/graphene-ideal/armchair/N7/legacy/size_7.npy`, `N9/legacy/size_9.npy`): one 3-D
array per width, shape `(concentrations, 10000 seeds, 300 energies)`. Row `i` of axis 1 is seed `i`. The impurity counts
of axis 0 are in `legacy_consolidated_metadata.json`; energies are 0 to 2.99 t in steps of 0.01 t.

```python
agnr = DATA / "graphene-ideal" / "armchair"
a7 = np.load(agnr / "N7" / "legacy" / "size_7.npy", mmap_mode="r")
counts = json.loads((agnr / "legacy_consolidated_metadata.json").read_text())["size_7"]["concentrations"]
print(a7.shape, counts[:3])                   # (34, 10000, 300) [2, 4, 6]
```

**Square lattice, widths 10, 15, 20** (`data/square/strip/N10/legacy/size_10_combined/`): `conc_<c>.npy` of shape
`(10000, 400)` for `c` impurities, and `conc_<c>_meta.csv` with columns `row_idx, config` giving each row's seed. The clean
spectrum of width 10 is `N10/legacy/pristine_10.npy`.

```python
sq = DATA / "square" / "strip" / "N10" / "legacy" / "size_10_combined"
rows = np.load(sq / "conc_10.npy", mmap_mode="r")
row_seed = np.loadtxt(sq / "conc_10_meta.csv", delimiter=",", skiprows=1, dtype=int)[:, 1]
```

## The same with the project's code

`atlaslib.CloudStore` reads exactly these files (and hides the extra densities unless `extra=True`):

```python
import sys
sys.path.insert(0, "src")                   # or: export PYTHONPATH=src
from atlaslib import CloudStore

store = CloudStore(DATA)
E_t, clean = store.read_pristine("hbn/armchair/N10")
spectra, seeds = store.read_cloud("hbn/armchair/N10", 0.01)
```
