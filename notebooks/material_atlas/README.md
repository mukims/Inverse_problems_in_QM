# material_atlas/ — Label-free material identification (sensor pipeline, stages 1–2)

The atlas places a transmission signature in an autoencoder latent space and finds the closest known
material. Stage 3 (material-specific concentration models) then takes over.

```
T(E) ──► label-free input ──► encoder ──► point z in the atlas
                                              ├─► k-NN vote over reference points → material (type + width) + confidence
                                              ├─► distance to nearest references  → "unknown material" flag
                                              └─► neighbours' concentrations      → rough first estimate of c
```

**Label-free input** (identical for every spectrum, no per-material pristine division; see LOGBOOK Bug #8):
`X = log1p(clip(round(T, 3), 0, 20)) / log1p(20)` over E = 0–2.99 eV (300 channels).

**Split:** configuration seeds 70/15/15 in every material. The validation part sets the novelty threshold
(99th percentile of known-material distances).

**Baselines on the same input:** physics descriptors (band onset, plateau level) + shallow decision tree,
library matching against mean spectra, logistic regression, and k-NN on PCA (what the autoencoder adds).

**Novelty test (`--loo`):** retrain without 9-AGNR, and separately without Square-10, and check whether the
unseen material is flagged.

```bash
python material_atlas.py --threads 4 --loo        # writes results/
```
```python
from material_atlas import MaterialAtlas
atlas = MaterialAtlas.load("results")
atlas.locate(T)   # T: raw transmission from E = 0 at 0.01 eV steps, >= 300 channels
# -> [{"material", "type", "confidence", "novelty_score", "unknown", "rough_concentration", "reconstruction_error"}]
```

## Status
First full run in progress (2026-09-29); results will be recorded here and in the LOGBOOK.
A 1%-data smoke test identified an unseen square lattice 100% of the time, but an unseen 9-AGNR only
weakly (AUROC 0.71): a similar material may reconstruct well and escape the novelty flag.
