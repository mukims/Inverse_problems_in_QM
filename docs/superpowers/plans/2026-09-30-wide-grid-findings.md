# Wide-grid findings from step 2 (2026-09-30)

For the implementing agent and the human, who chooses the final width grid. These come from the reviewer's checks on the step 2 smoke clouds. Put both points in SMOKE-2 next to the cost table.

## Up to mark
Armchair N20, N27, N31 and N40 are correct:

| Width | Clean T vs open channels | Away-from-edge median − pristine | Away-from-edge spike share |
|---|---|---|---|
| N20 | 6e-8 | ≤ −0.007 | ≤ 0.25% |
| N27 | 5e-8 | ≤ +0.0003 | 0% |
| N31 | 5e-8 | −0.04 to −0.44 | 0% |
| N40 | 8e-8 | −2.5 to −8.0 | 0% |

N40 has 40–320 impurities per configuration, so its large deficit is strong scattering, not an error. Even-width leads for N40 and N50 are computed and cached in `agnr_cell_v2`.

## 1. The input cap of 20 saturates wide ribbons (blocking for any atlas on the wide grid)
`InputSpec.cap = 20.0` clips T at 20 before the log. It was sized for 7/9-AGNR and square-10, whose clean spectra stay below 10. Wide ribbons carry more channels than that:

| Model | Max clean T | Share of window where clean T ≥ 20 | Input channels pinned at 1.0 |
|---|---|---|---|
| armchair N31 | 15 | 0% | 0 |
| armchair N40 | 20 | 0% | 8 |
| armchair N50 | 25 | 23% | 76 of 400 |
| zigzag N-ZGNR | ≈ N | — | expected heavily for N ≥ 20 |

For those energies every spectrum maps to the same input value, so the atlas cannot tell width or disorder apart there.

Do, before training any atlas that includes widths above 31 (armchair) or 16 (zigzag):
- Add an `InputSpec` version 2 whose cap is set from the grid, for example 1.25 × the largest clean T of any model in the grid, rounded up. The cap stays a fixed constant, so the input remains label-free.
- Record the cap in the manifest.
- The saved v1 atlases (the 7/9 reference and `atlas_v2_smoke`) keep v1. Never mix spec versions inside one atlas.
- Test that no clean spectrum in the grid reaches the cap.

## 2. The edge-masked guard checks less and less as width grows
The ±5-channel mask around each clean-spectrum step is justified: median excess shows up 3–4 channels from a step (N12: +1.16 at 3 channels, +0.65 at 4; N40: +4.4 at 3, +1.8 at 4). But the unmasked region shrinks with the number of subbands:

| Width | Steps in 0–2.99 t | Unmasked channels at ±5 / ±3 / ±2 |
|---|---|---|
| N12 | 12 | 177 / 221 / 243 |
| N31 | 28 | 63 / 124 / 165 |
| N40 | 39 | 27 / 88 / 134 |
| N50 | 45 | **11** / 64 / 111 |

At N50 the store guard checks 11 of 300 channels, and one channel 1 step from an edge has a median of +213 above clean. That is the trace formula's edge divergence, not a lattice error. The band test and the clean-channel test remain the real correctness checks for wide ribbons.

Do: in `check_store.py`, add a width-independent check. For every seed, mean T over unmasked channels ≤ mean pristine over those channels + 0.05. Report the number of unmasked channels per model, so a reader can see how much the guard covered.
