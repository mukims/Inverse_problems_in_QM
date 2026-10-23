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

## Results (BUILD-12, held-out seeds, `results/atlas_metrics.json`)

| | Result |
|---|---|
| Material and width, atlas (AE + k-NN) | **100.00%** on 45,450 test spectra, every concentration band |
| Baselines on the same input | logistic 99.97%, PCA-32 k-NN 99.96%, onset + plateau tree 99.45%, library matching 99.37% |
| False alarms on known materials | 1.05% |
| Unseen Square-10 (trained without it) | AUROC 1.00 by reconstruction error, 0.95 by k-NN distance; the k-NN threshold flags only 0.6% |
| Unseen 9-AGNR (trained without it) | AUROC 0.79 / 0.51: a new width of a known material looks familiar |
| Rough concentration from neighbours | MAE 3.15 / 4.57 / 4.91 (only 4 reference densities) |

Consequences: flag unknown *materials* by reconstruction error (this script's `unknown` still uses k-NN
distance; the plan's `atlaslib.Atlas` uses reconstruction error); place new *widths* on the continuous width
axis rather than expecting a novelty flag.
