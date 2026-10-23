# AGENTS.md

- Python: `~/miniconda3/envs/ml/bin/python` (CPU PyTorch, NumPy, scikit-learn, XGBoost, LightGBM, pytest). Run tests from the repo root: `~/miniconda3/envs/ml/bin/python -m pytest -q` (`pytest.ini` puts `src/` and the script folders on the path). Scripts need `PYTHONPATH=src`.
- `LOGBOOK.md` is the source of truth for builds, results and bugs; record every new result there.
- Data validity rules (LOGBOOK Bugs #6–#8), for every model and evaluation:
  - **Seed split**: split train/validation/test by configuration seed (row *i* of each concentration is seed *i*).
  - **Label-free**: anything applied before identifying a material or width is computable without knowing it.
  - **Round first**: round T to 3 decimals before dividing by a pristine spectrum.
  - Square-lattice spectra come from `scripts/generate/ca_sq.py`; the CSVs from `CA.ipynb` are corrupt.
- Material atlas, `atlaslib`, `tbribbon`, new materials: start with `docs/superpowers/handoff/material-atlas/BRIEF.md`.
- Layout: library code in `src/` (`tbribbon`, `atlaslib`, `shazam`, `agnr`, `paths`), runnable scripts in `scripts/<job>/`, results in `results/`, data outside git under `TRANSMISSION_DATA` (see `docs/START_HERE.md`). Never hard-code a data path: use `src/paths.py`.
- `CloudStore` hides densities merged in from another run (`source` in the cloud's meta) unless `extra=True`, so catalogue builds keep the densities they were defined with.
