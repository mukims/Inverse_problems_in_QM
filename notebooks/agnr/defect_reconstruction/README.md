# defect_reconstruction/ — Where are the impurities?

The hardest inverse problem in this repo. The other folders ask *how many* impurities
there are; these scripts ask **where they sit**.

Output target is a **10×10 matrix** encoding up to 10 scattering centres:

- **Diagonal `M[i,i]`** — transverse site index of impurity `i` within the unit cell (0–13)
- **Off-diagonal `M[i,j]`** — Euclidean distance between impurities `i` and `j`,
  `sqrt(Δcell² + Δsite²)`
- **Zero-padded rows/columns** — encode the actual impurity count, so the matrix is
  sparse when there are fewer than 10

---

## Files

| File | Direction | What it does |
|---|---|---|
| `forward_model.py` | forward | Learns the *surrogate* forward map: distance matrix → `T(E)`. Useful as a fast stand-in for the NEGF solver. |
| `inverse_model.py` | inverse | **The main model.** 1D ResNet encoder → bottleneck → transposed-conv decoder with learned positional queries. Also defines `InverseGNRDataset`, `FocalLoss`, `MisfitLoss`, reused by the others. |
| `transformer_model.py` | inverse | `SpectrumTransformer` variant on the same task. |
| `patched_transformer_model.py` | inverse | 1D-patched transformer (`PatchedInverseModel`) plus attention-weight extraction for interpretability. Imports from `inverse_model.py`. |

`inverse_model.py` is the hub: the other two inverse scripts import its dataset and
loss classes, so read it first.

---

## Typical use

```bash
cd defect_reconstruction && conda run -n ml python inverse_model.py
```

## Gotchas

- These scripts save into **`../../../models/trained/`** (i.e. `<repo>/models/trained/`),
  not into this folder — and those paths are **relative to your current directory**, so
  run them from inside `defect_reconstruction/`.
- They import `agnr` from [`../physics/`](../physics/) via the `_bootstrap` shim. The
  import is wrapped in `try/except ImportError`, so a failure degrades silently rather
  than crashing — if physics-dependent behaviour seems to be missing, check that import.
- `distance_matrix_model.pth` (the trained checkpoint referenced by the top-level README)
  lives at the **repository root**, not here.
- Documentation for these models is in [`../docs/`](../docs/):
  `inverse_model_explanation.md`, `patched_transformer_explanation.md`, `walkthrough.md`.
