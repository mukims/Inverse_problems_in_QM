# Data, leads and stores

Verified on 2026-09-29. Energies are in units of t, starting at E = 0 in 0.01 steps, so channel *i* is E = 0.01·*i*.

## Spectra

| Dataset | Path | Shape | Content |
|---|---|---|---|
| 7-AGNR | `/run/media/shardul/storage/machine_learning/transmission_data/transmission_results/consolidated_data/size_7.npy` | `(34, 10000, 300)` | c = 2, 4 … 68 (index *i* ↔ c = 2(*i*+1)); row = configuration seed; E = 0–2.99 |
| 9-AGNR | `…/consolidated_data/size_9.npy` | `(49, 10000, 300)` | c = 2, 4 … 98; same layout |
| Square-10 (clean) | `~/transmissions_sq/size_10_combined/conc_{c}.npy` + `conc_{c}_meta.csv` | `(10000, 400)` each | c = 5, 10 … 90; the meta file maps row → seed (column `config`); E = 0–3.99 |
| Square 15 and 20 | `~/transmissions_sq/size_{15,20}_combined/` | — | Reserved for a separate project; not part of this work |
| Square-10 (corrupt) | `…/transmission_data/size_10/lead_size_10_conc_*_config_*.csv` | — | From `CA.ipynb` (cache-mutation bug); never use |

Load the big arrays with `np.load(path, mmap_mode="r")` and slice; they are 0.8–1.2 GB each.

`c` is an impurity **count**. The number of sites is 100 cells × sites per cell (7-AGNR 1,400; 9-AGNR 1,800; square-10 1,000), so density = c / sites.

## Clean (pristine) references

| System | Path | Length |
|---|---|---|
| 7-AGNR | repo root `7_agnr_pris.npy` (git-ignored; copy in `data/raw/transmission_results/`) | 300 |
| 9-AGNR | repo root `9_agnr_pris.npy` | 300 |
| Square-10 | `~/transmissions_sq/pristine_10.npy` (computed with `ca_sq.py`; the old `size_10_pris.csv` is wrong at E = 0) | 400 |

## Leads (precomputed surface Green's functions)

| System | Path | Shape |
|---|---|---|
| AGNR width m, 5–31 | `~/Desktop/backup/agnr/size_{m}/leads_{m}.npy` (load via `agnr_lib.load_leads(m)`) | `(300, 2m, 2m)`, E = 0–2.99 only |
| Square width l | `~/transmissions_sq/leads/leads_{l}.npy`, l ∈ {5, 10 … 50, 60} | `(400, l, l)` |

The new engine (`tbribbon`) computes its own leads over the full 0–4 t grid, so it does not need these; they are used only for the legacy cross-check test.

## Stores this work creates
- `~/atlas_store/reference_v1`: legacy 7/9-AGNR atlas clouds (Phase 1)
- `~/atlas_store/engine_v1`: engine-generated clean spectra and clouds (Phases 3a–5)

Each model lives under `<store>/<material>/<edge>/N<width>/`.

## Reference numbers to reproduce (sanity checks)

| Check | Expected | Source |
|---|---|---|
| XGBoost, BUILD-09 split, 0–1.5 eV | MAE 2.394–2.396 | `notebooks/agnr/multi_width/seed_split/`, `energy_window_check.py` |
| Same, full 0–3 eV | MAE 1.977 | `energy_window_check.py` |
| Transformer, BUILD-09 split | MAE 2.267 (7-AGNR 1.871, 9-AGNR 2.542) | `seed_split/mw_results/transformer_metrics.json` |
| Physics misfit, raw spectra, label-free | width accuracy 99.57% | same folder |
| Relative error, full spectrum | 4.2–4.8% of c in every concentration band | LOGBOOK BUILD-11 |

A result far from these on the same setup points to a data or split problem before a model problem.

## Preflight
```bash
~/miniconda3/envs/ml/bin/python - <<'EOF'
import os, numpy as np
D = "/run/media/shardul/storage/machine_learning/transmission_data/transmission_results/consolidated_data"
checks = {
    "size_7": (f"{D}/size_7.npy", (34, 10000, 300)),
    "size_9": (f"{D}/size_9.npy", (49, 10000, 300)),
    "pris_7": ("7_agnr_pris.npy", None), "pris_9": ("9_agnr_pris.npy", None),
    "square_40": (os.path.expanduser("~/transmissions_sq/size_10_combined/conc_40.npy"), (10000, 400)),
    "pris_sq": (os.path.expanduser("~/transmissions_sq/pristine_10.npy"), (400,)),
    "leads_sq10": (os.path.expanduser("~/transmissions_sq/leads/leads_10.npy"), (400, 10, 10)),
    "leads_agnr7": (os.path.expanduser("~/Desktop/backup/agnr/size_7/leads_7.npy"), (300, 14, 14)),
}
for k, (p, shape) in checks.items():
    ok = os.path.exists(p) and (shape is None or np.load(p, mmap_mode="r").shape == shape)
    print(f"{'OK ' if ok else 'MISSING/BAD'} {k}: {p}")
EOF
```
Run it from the repo root.
