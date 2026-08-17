# Understanding `inverse_model.py`

The `inverse_model.py` module solves the "Inverse Design" problem in quantum transport. Instead of calculating transmission from a known device structure (forward problem), it maps a given transmission spectrum $T(E)$ back into a spatial physical structure (the defect map).

This module has four main components: Data Loading, The CNN Architecture, the Training Loop, and Visualization.

---

## 1. Data Pipeline (`InverseGNRDataset`)

Instead of wasting weeks of CPU time artificially simulating new data, we **recycle the existing datasets** generated for the original PINN.

### How it works:
1. **Filtering:** Reads `manifest_agnr.csv` and filters out anything above a concentration of 10. The model needs to learn interference, but `c=49` is too chaotic for initial training.
2. **Spectrum Loading:** Loads the 200-point `.npy` transmission spectrum and normalizes it against the pristine ribbon, producing an input tensor $x \in [0, 1]^{200}$.
3. **On-the-fly Labels:** Uses the deterministic random seed `agnr.chosen_for_config(conc, config_id)` to recompute exactly where the impurities were placed in the 100-cell nanoribbon.
4. **Target Map:** Constructs a 1D tensor $y \in \{0, 1\}^{100}$ where a `1` represents an impurity in that unit cell.

## 2. The Model Architecture (`InverseModel`)

The network is an **Encoder-Decoder 1D Convolutional Neural Network**, conceptually similar to a U-Net or a Variational Autoencoder (VAE) architecture.

### The Encoder (Compression)
*   **Goal:** Read the $T(E)$ spectrum, recognize critical resonance dips, and compress that physical information into a dense "latent state".
*   **Structure:** Starts with an initial `Conv1d` followed by three `ResBlock1D` layers. Each ResBlock contains skip-connections (adding the input to the output) which drastically improves gradient flow. Between blocks, `MaxPool1d(2)` halves the resolution.
*   **Output:** Compresses the $[200]$ vector into 32 channels of length 12 ($32 \times 12 = 384$ features).

### The Bottleneck (Capacity)
*   **Goal:** Act as the main "brain" translating physical spectrums into spatial maps.
*   **Structure:** Flattens the 384 features, expands to a dense layer of **512 neurons**, applies **Dropout (30%)** to prevent overfitting on specific noise patterns, and narrows back to 384.

### The Decoder (Spatial Expansion)
*   **Goal:** Expand the dense latent state into a 100-length sequence representing the physical device.
*   **Structure:** Uses `ConvTranspose1d` (Transposed Convolutions or "Deconvolutions"). These specifically upscale the tensor size.
    *   12 $\to$ 25 $\to$ 50 $\to$ 100.
*   **Output:** A single channel sequence of length 100 containing **raw logits**. We specifically *do not* apply a `Sigmoid()` layer here (more on why below).

## 3. The Training Loop

Training an inverse design model on sparse physical systems presents a massive challenge: **Class Imbalance.**

In a device of 100 unit cells with an average of 5 impurities, 95% of the array is `0` and only 5% is `1`. If you use standard loss functions, the network learns that it can achieve **95% accuracy instantly by simply guessing `0` everywhere!**

### Weighted Binary Cross-Entropy
To counteract this lazy guessing, we use PyTorch's `BCEWithLogitsLoss(pos_weight=15.0)`:
1.  **Why Logits?:** Passing raw model outputs (logits) directly into this loss function utilizes the "log-sum-exp" trick. This is far more numerically stable than manually applying a Sigmoid and running `BCELoss()`, which can lead to `NaN` or wildly spiking validation losses.
2.  **Why Pos-Weight?:** By setting `pos_weight=15.0`, we tell the loss function that missing an impurity (predicting a `0` where a `1` should be) is **15 times more painful** than falsely placing an impurity. This forces the model to actually look for the defects.

**Optimizer & Scheduler:** 
Uses `AdamW` (Adam with strict weight decay to prevent overfitting) combined with a `CosineAnnealingLR` scheduler, which smoothly drops the learning rate following a cosine curve as epochs progress, helping the model settle into local minima gently.

## 4. Visualization

The `visualize_prediction` function runs a validation sample through the trained network. 
- It wraps the model's raw logic output in a `torch.sigmoid()` to convert the numbers into readable probabilities $[0, 1]$.
- It plots the input transmission $T(E)$ on the left.
- On the right, it maps the Model's Predicted Probabilities (Red bars) directly over the True Impurity Locations (Black bars).

If the model is trained perfectly, every black bar will have a red bar covering it!
