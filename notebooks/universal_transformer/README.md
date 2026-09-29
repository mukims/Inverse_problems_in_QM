# universal_transformer/ — One model for type, width and concentration (BUILD-10)

A multi-task 1D-patched transformer that reads one transmission spectrum and predicts
the system type (AGNR or square lattice), the width (7-AGNR, 9-AGNR, Square-10) and the
impurity concentration. The concentration head also sees the model's own softmax type and
width predictions.

| File | Role |
|---|---|
| `universal_data.py` | Loads 7-AGNR, 9-AGNR (`consolidated_data/size_{7,9}.npy`) and clean Square-10 (`~/transmissions_sq/size_10_combined`), normalises each by its pristine spectrum (rounded to 3 decimals, clipped to [0, 1]), splits 70/15/15 **by configuration seed** |
| `universal_transformer.py` | ConvStem → patch embedding (10 channels per token) + `[CLS]` → 3 pre-norm transformer blocks → type, width and conditioned concentration heads |
| `train_and_evaluate.py` | Trains to convergence (warmup, then halve the LR on validation plateaus, early stopping) and writes metrics and plots to `--out-dir` |

```bash
# BUILD-10: 3,000 seeds per concentration, 0-1.49 eV
python train_and_evaluate.py --samples-per-conc 3000 --threads 16
# All data, full 0-3 eV spectrum (30 tokens)
python train_and_evaluate.py --samples-per-conc 10000 --spectrum-len 300 --threads 16 --out-dir full_data_300ch
```

## Results (BUILD-10, held-out seeds, `universal_metrics.json`)

| System | Concentration MAE | RMSE | Max error |
|---|---|---|---|
| 7-AGNR | 1.855 | 2.604 | 15.37 |
| 9-AGNR | 2.551 | 3.655 | 19.30 |
| Square-10 | 3.126 | 4.392 | 21.13 |

303,000 spectra, 658,374 parameters, 29 epochs (45 min on CPU); best validation epoch 17.

**Caveat (LOGBOOK Bug #8):** each spectrum is divided by its *own* system's pristine spectrum, which
requires knowing the system. The reported type accuracy (100%) and width accuracy (≥ 99.99%) are therefore
circular. Read the concentration results as "given the correct system"; label-free identification is in
`notebooks/material_atlas/`.

The first universal run (500 seeds per concentration, corrupt square data, random split; MAE 1.81 / 2.27 / 14.5)
is superseded.
