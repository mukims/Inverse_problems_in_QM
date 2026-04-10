# Understanding `patched_transformer_model.py`

This document explains the **1D-Patched Transformer** architecture introduced in `patched_transformer_model.py` and why it was created as a second, standalone model alongside the original CNN-based `inverse_model.py`.

---

## Motivation: Why a Patched Transformer?

The original `InverseModel` is a 1D convolutional encoder-decoder. While effective, it has an inherent limitation: convolutions are **local** operators. A `kernel_size=3` filter can only ever see three adjacent energy bins at once. To correlate distant parts of the spectrum — for example, a Fano resonance dip at $E = -1.5$ eV that is physically coupled to a dip at $E = +1.5$ eV — the network must rely on *many* stacked layers and pooling operations to gradually widen its receptive field. Information about long-range correlations arrives at the bottleneck only indirectly, after passing through a deep chain of lossy compressions.

A **Transformer** solves this with global self-attention: every patch of the spectrum can directly attend to every other patch in a single layer, without needing to route information through an intermediate hierarchy. This makes it a natural fit for quantum transport spectra, where interference effects create non-local correlations between distant energy bands.

### Why patching?

If you feed a 200-point spectrum directly into a standard Transformer token-by-token, the network has to calculate a $200 \times 200 = 40{,}000$-element attention matrix at every layer. This is computationally expensive and, worse, individual energy bins in a discretised spectrum are often too noisy or physically meaningless on their own to form good attention keys and queries.

**1D patching** solves both problems:

1. **The Chop.** Divide the 200-point sequence into $N = 20$ non-overlapping windows of size $P = 10$.
2. **The Projection.** Each 10-element window is multiplied by a learned weight matrix that maps it into a $D = 64$-dimensional embedding. This projection acts as a micro-feature extractor, automatically learning to recognise local shapes like resonance dips, plateaux, and slope changes.
3. **The Global Interaction.** The Transformer now sees only 20 tokens instead of 200, shrinking the attention matrix 100-fold to $20 \times 20 = 400$ elements. Each token represents a coarse-grained energy *band*, and attention directly learns which bands are correlated.

---

## Architecture Overview

```
Input T(E): [B, 200]
       │
       ▼
┌──────────────────────────────┐
│  1. PatchEmbedding1D         │
│     Conv1d(1 → 64, k=10, s=10) │  ← non-overlapping projection
│     + Learned Positional Enc │
│     Output: [B, 20, 64]     │
└──────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│  2. TransformerEncoder       │
│     3 × Pre-Norm Encoder     │  ← global self-attention (GELU, 4 heads)
│     Output: [B, 20, 64]     │
└──────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│  3. 1×1 Conv Bottleneck      │
│     64 → 128 → 32 channels  │  ← channel mixing & compression
│     Output: [B, 32, 20]     │
└──────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│  4. FC Decoder               │
│     Flatten (640)            │
│     → 256 (BatchNorm, ReLU) │
│     → 100 (10×10 matrix)    │
│     Output: [B, 100]        │
└──────────────────────────────┘
```

### 1. Patch Embedding (`PatchEmbedding1D`)

The embedding is implemented as a single `nn.Conv1d(in_channels=1, out_channels=64, kernel_size=10, stride=10)`. Because the kernel size equals the stride, the convolution slides without overlap — each kernel application sees exactly one 10-point window. Mathematically, this is equivalent to chopping the sequence into patches and multiplying each by a learned $10 \times 64$ weight matrix, but it runs as a single fused GPU kernel rather than a manual reshape + linear layer.

After projection, a **learned absolute positional encoding** is added. The resulting tensor has shape `[B, 20, 64]`, which the Transformer interprets as a sequence of 20 tokens, each 64-dimensional.

### 2. Transformer Encoder

The encoder consists of 3 stacked `nn.TransformerEncoderLayer` blocks using the **pre-norm** (norm-first) configuration:

```
  x → LayerNorm → MultiHeadAttention(4 heads) → Residual Add
    → LayerNorm → FFN(64 → 256 → 64, GELU)    → Residual Add
```

Pre-norm is more stable than the original post-norm design because the residual stream stays unnormalised, preventing gradient explosion during early training. GELU activation is used instead of ReLU for smoother gradients.

Each self-attention layer computes a $20 \times 20$ attention matrix, where entry $(i, j)$ represents how much energy band $i$ attends to energy band $j$. These weights are the basis for interpretability (see §5).

### 3. 1×1 Conv Bottleneck

After the Transformer, a small 1D convolution block with `kernel_size=1` performs channel mixing:

```
64 channels → 128 (ReLU, Dropout=0.3) → 32
```

This compresses the per-patch feature representation from 64 dimensions down to 32 before the fully-connected decoder, acting as a learned dimensionality reduction. The 1×1 convolution is equivalent to applying the same linear transformation independently to each of the 20 patches.

### 4. FC Decoder 

The decoder flattens the `[B, 32, 20]` tensor to a 640-element vector, passes it through a 256-neuron hidden layer with batch normalisation and ReLU, and finally projects to 100 outputs representing the flattened $10 \times 10$ impurity distance matrix. No output activation is applied — the model produces raw values for MSE regression.

---

## What Changed vs. `inverse_model.py`

| Aspect | `inverse_model.py` (CNN) | `patched_transformer_model.py` (Transformer) |
|--------|--------------------------|-----------------------------------------------|
| **File** | Existing module | **New** standalone module |
| **Encoder** | Conv1d + ResBlocks + MaxPool | PatchEmbedding + Transformer Encoder |
| **Core operation** | Local convolution (kernel=3–7) | Global self-attention over 20 patches |
| **Receptive field** | Grows gradually with depth | Global from layer 1 |
| **Decoder** | ConvTranspose1d upsampling | Fully-connected layers |
| **Interpretability** | Black box | Attention heatmaps (§5) |
| **Reuses from `inverse_model.py`** | — | `InverseGNRDataset`, `MisfitLoss`, `get_reference_spectra` |
| **Loss function** | MSE + MisfitLoss | MSE + MisfitLoss (identical) |
| **Target** | 10×10 distance matrix | 10×10 distance matrix (identical) |
| **Impact on existing code** | None — `inverse_model.py` is unchanged | N/A |

**Key design decision:** The patched transformer lives in its own file and **does not modify `inverse_model.py`**. It imports `InverseGNRDataset` and `MisfitLoss` directly from the existing module, so both architectures train on the exact same data with the same physics constraint. This modularity means you can switch between models by simply changing one import line.

---

## Loss Function: MSE + Misfit Loss

The training objective is the same for both models:

$$\mathcal{L} = \text{MSE}(\hat{D}, D) + \lambda \cdot \text{MisfitLoss}(\hat{D}, T(E))$$

where:

- $\hat{D}$ is the predicted $10 \times 10$ distance matrix (flattened to 100 values)
- $D$ is the true distance matrix
- $T(E)$ is the input transmission spectrum
- $\lambda$ is the misfit weight (default: 5.0)

**MisfitLoss** provides a physics-informed constraint. It estimates the impurity concentration from the predicted matrix by computing row-wise L2 norms (impurity rows have large norms, zero-padded rows have norms near zero), soft-thresholds them with a sigmoid, and sums to get a differentiable concentration estimate $\hat{c}$. It then looks up the expected average spectrum $\bar{T}_{\hat{c}}(E)$ for that concentration and penalises the model if the input spectrum $T(E)$ doesn't match. This prevents the model from predicting distance matrices that are physically inconsistent with the input spectrum.

---

## Interpretability: Attention Weight Extraction

The function `extract_attention_weights(model, x_input)` is unique to the patched transformer and provides a direct window into *what* the model is learning.

### How it works

Instead of using the standard `model.forward()` which discards attention weights, this function manually walks through each `TransformerEncoderLayer` and calls `self_attn(..., need_weights=True, average_attn_weights=False)`. This returns the raw per-head attention tensors of shape `[B, num_heads, 20, 20]`.

### What to look for

- **Diagonal dominance** means each energy band mostly attends to itself — the model hasn't learned cross-band correlations yet (undertrained or the data doesn't require it).
- **Off-diagonal hot spots** at symmetric positions (e.g., patch 3 attends to patch 17) suggest the model has discovered Fano resonance pairs or other quantum interference signatures coupling distant energy ranges.
- **Head specialisation:** Different attention heads may learn different types of correlations. In quantum transport, you might expect one head to track resonance locations and another to track background suppression patterns.

### Visualisation

The `visualize_prediction()` function displays a 4-panel figure:

1. **Normalised Transmission $T(E)$** — the input spectrum
2. **True Distance Matrix** — the ground-truth $10 \times 10$ matrix
3. **Predicted Distance Matrix** — the model's output reshaped to $10 \times 10$
4. **Attention Heatmap** — the last encoder layer's attention averaged over all 4 heads, showing energy band correlations as a $20 \times 20$ matrix

---

## Usage in `inverse_design.ipynb`

The notebook is organised so that **both models** can be trained and compared in a single session:

- **Sections 1–3** use the original CNN model from `inverse_model.py` (data loading, training, visualisation).
- **Section 4** (newly added) imports from `patched_transformer_model.py` and trains the Patched Transformer on the **same dataset**, then plots loss curves and predictions with attention heatmaps.

The key cells added to Section 4 are:

```python
# Import
from patched_transformer_model import PatchedInverseModel, train_patched_model, visualize_prediction as pt_visualize

# Train (reuses the same dataset object from Section 1)
pt_model, pt_train_losses, pt_val_losses = train_patched_model(
    dataset=dataset,
    num_epochs=50, batch_size=16, lr=7e-4,
    misfit_weight=5, patch_size=10, embed_dim=64,
    num_heads=4, depth=3
)

# Visualise with attention heatmap
pt_visualize(pt_model, dataset, idx=random_idx)
```

---

## Hyperparameters Worth Tuning

| Parameter | Default | Notes |
|-----------|---------|-------|
| `patch_size` | 10 | Smaller patches → more tokens → richer attention but higher cost. Must divide 200 evenly. |
| `embed_dim` | 64 | The dimensionality of each patch token. Larger → more capacity but more parameters. |
| `num_heads` | 4 | Must divide `embed_dim`. More heads → more parallel attention patterns per layer. |
| `depth` | 3 | Number of encoder layers. Diminishing returns past 4–6 for this problem size. |
| `misfit_weight` | 5.0 | How strongly the physics constraint is enforced relative to MSE. |
| `lr` | 7e-4 | Learning rate for AdamW. The cosine annealing scheduler decays this to 0. |
