# Start here

This project simulates how electrons transmit through nanoribbons of 2-D materials with random impurities, then solves
the inverse problem: from one transmission spectrum T(E), read the ribbon's material, edge, width and impurity density.

## Where things are

| You want | Go to |
|---|---|
| The data | `data/<material>/<edge>/N<width>/`, described in `data/README.md`; `data/INDEX.csv` lists all 342 ribbons |
| To read the data with plain NumPy | [`reading-cloud-data.md`](reading-cloud-data.md) |
| To simulate a ribbon | `src/tbribbon/` (engine), run with `scripts/generate/generate_clouds.py` |
| To identify a spectrum | `src/atlaslib/` (`Atlas`, ensemble `lookup`), built by `scripts/build/`, scored by `scripts/evaluate/` |
| The older 7/9-AGNR models | `scripts/experiments/agnr/` and `results/agnr/` |
| A number from the logbook | `results/<same path the logbook gives under notebooks/>`; `docs/path_map.csv` maps every moved file |
| One material's model and data | `docs/materials/<material>/README.md` |
| Old runs, raw simulation output | `archives/` (next to `data/`), with a record in `data/MOVES.csv` |

## Folders

```
src/         importable code: tbribbon, atlaslib, shazam, agnr, paths
scripts/     generate/ (simulate, leads), build/ (atlas, lookup, autoencoder), evaluate/, experiments/ (studies),
             report/ (figures), data/ (organise, index)
notebooks/   Jupyter notebooks only
results/     what the scripts write: metrics, logs, figures, checkpoints
tests/       pytest (about 400 tests, 3 minutes)
```

## Run something

```bash
export PYTHONPATH=src
export TRANSMISSION_DATA=/path/to/data      # or a symlink called data/ in the repo root
python -m pytest -q
python scripts/generate/generate_clouds.py --help
```

Every data location comes from `src/paths.py`: `data_root()`, `ribbon_dir(material, edge, width)`, `archive_run(name)`,
`results_dir(...)`. Set `TRANSMISSION_DATA`, `TRANSMISSION_ARCHIVES` or `TRANSMISSION_RAW` to move them.

Load one ribbon:

```python
from atlaslib import CloudStore
store = CloudStore("data")                         # or paths.data_root()
E, clean = store.read_pristine("hbn/armchair/N10")
spectra, seeds = store.read_cloud("hbn/armchair/N10", 0.01)     # density 1%
```

## Rules every analysis follows (LOGBOOK bugs 6-8)

1. Split by configuration seed, never by random rows.
2. Everything before identification must be label-free.
3. Round T to 3 decimals before dividing by a clean spectrum.
4. Square-lattice data comes from `ca_sq.py` only; the `CA.ipynb` CSVs are corrupt (kept in `archives/corrupt_square_CA/`).

## Good to know

* Five ribbons (`hbn/armchair/N9`, `mos2/zigzag/N9`, `phosphorene/armchair/N9`, `triangular/zigzag/N9`,
  `graphene-ideal/armchair/N13`) carry 20 extra densities from the `conc_v2` sweep. `CloudStore` hides them by default so a
  catalogue still has the 4 densities it was defined with; `CloudStore(root, extra=True)` shows all 24.
* `triangular/armchair/N4` and `triangular/zigzag/N4` have a clean spectrum only.
* `src/agnr/hamiltonian.py` (formerly `agnr.py`) refers to an undefined `g21` in one function. It predates the reorganisation
  and nothing in the tests calls it.
