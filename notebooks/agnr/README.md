# AGNR Quantum Transport — Code Map

Everything here concerns **armchair graphene nanoribbons (AGNRs)**: simulating the
transmission spectrum `T(E)` of a disordered ribbon (the *forward* problem), and
recovering device properties from that spectrum (the *inverse* problem).

The code was previously one flat directory of ~50 files. It is now grouped by
**what job the script does**. Start here, then read the README inside the folder
you need.

---

## Which folder do I want?

| Folder | Job | Start with |
|---|---|---|
| [`physics/`](physics/) | Forward simulation: build ribbons, leads, and generate spectra | `agnr_lib.py` |
| [`multi_width/`](multi_width/) | **Current work.** Joint 7- & 9-AGNR width + concentration models | `mw_common.py` |
| [`concentration/`](concentration/) | Older single-width concentration models (the CNN/PINN/transformer lineage) | `pinn_agnr_curvature.py` |
| [`defect_reconstruction/`](defect_reconstruction/) | Recover *where* impurities sit (10×10 distance matrix) | `inverse_model.py` |
| [`agent/`](agent/) | Inference agent that diagnoses an unknown spectrum end-to-end | `agnr_agent.py` |
| [`sweeps/`](sweeps/) | Hyperparameter optimisation (Optuna, misfit-weight sweeps) | `bayesian_opt_sweep.py` |
| [`time_series/`](time_series/) | Time-series & multi-output spectral sequence continuation | `time_series_nn.ipynb` |
| [`nb/`](nb/) | Exploratory Jupyter notebooks | — |
| [`docs/`](docs/) | Long-form explanations of the models | `walkthrough.md` |

**New here?** Read `docs/walkthrough.md`, then `physics/README.md`, then
`multi_width/README.md` (that is where active development happens).

---

## The problem in one picture

```
     FORWARD  (physics/)
     impurity configuration ──► NEGF / recursive Green's function ──► T(E)

     INVERSE  (everything else)
                                     ┌──► ribbon width  (7 vs 9)      multi_width/, agent/
     T(E) [150 energy channels] ─────┼──► impurity concentration ĉ    multi_width/, concentration/
                                     ├──► high-energy continuation    time_series/
                                     └──► impurity positions          defect_reconstruction/
```

---

## Running anything

All Python runs through the **`ml` conda environment** (Python 3.12):

```bash
conda run -n ml python <script>.py --help
```

Scripts are meant to be run **from inside their own folder**, e.g.:

```bash
cd multi_width && conda run -n ml python mw_mlp.py
```

---

## How imports work after the reorganisation

Scripts used to sit in one directory and import each other by bare name
(`import agnr_lib`). Now that they are split across folders, Python would no
longer find those siblings, so any script importing across folders does:

```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401
```

[`_bootstrap.py`](_bootstrap.py) puts every topic folder on `sys.path` and exposes
`PROJECT_ROOT`. If you add a script that imports across folders, copy those two lines.

**Repo-root paths**: scripts reach the repository root with
`Path(__file__).resolve().parents[2]` (they are two levels below `transmissions/`).
CWD-relative hops are `../../../`. If you move a script between folders these stay
valid; if you change its *depth*, they do not.

---

## Shared files at this level

| File | Why it lives here |
|---|---|
| `manifest_agnr.csv` | Dataset index read by scripts in `concentration/` **and** `sweeps/`, so it cannot live inside either. They reach it via `script_dir.parent / "manifest_agnr.csv"`. |
| `_bootstrap.py` | The import shim described above. |

Pristine reference spectra (`7_agnr_pris.npy`, `9_agnr_pris.npy`) live at the
**repository root**, not here.
