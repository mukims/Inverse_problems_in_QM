# joint_autoencoder/ — One autoencoder over 7-AGNR, 9-AGNR and Square-10

`joint_autoencoder.py` trains a single `Conv1dAutoencoder` (64-d latent, the same architecture as
the 9-AGNR and Square-10 studies) on all three systems, then probes the shared latent for the
system and for concentration. Inputs: first 150 channels (E = 0–1.49 eV), each spectrum divided by
its own system's pristine (rounded to 3 decimals, clipped to [0, 1]); 3,000 configuration seeds per
concentration; split by seed 80/20.

```bash
python joint_autoencoder.py            # writes results/ (checkpoint is git-ignored)
```

## Results (held-out seeds, `results/joint_ae_metrics.json`)

| | 7-AGNR | 9-AGNR | Square-10 |
|---|---|---|---|
| Reconstruction R² | 0.948 | 0.949 | 0.911 |
| Concentration MAE, AE latent (gradient boosting) | 2.12 | 2.80 | 3.65 |
| Concentration MAE, PCA-64 | **1.90** | **2.58** | **3.18** |
| Concentration MAE, raw spectrum | 2.18 | 2.86 | 3.33 |

The latent separates the three systems into distinct clusters with concentration ordered inside each,
but it never beats PCA-64 for concentration.

**Caveat (LOGBOOK Bug #8):** the reported 100% system accuracy is circular, because the input was normalised
with each system's own pristine spectrum. The label-free version is `notebooks/material_atlas/`.
