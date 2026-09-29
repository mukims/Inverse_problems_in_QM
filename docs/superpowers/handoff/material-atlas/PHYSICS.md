# Physics conventions and checks

## Model conventions (all materials)
- Single-particle tight binding on a ribbon of 100 unit cells between semi-infinite leads of the same clean ribbon.
- Energy unit **t** = largest nearest-neighbour hopping magnitude. Idealised models set t = 1. **E = 0 at charge neutrality** (for MoS₂ and phosphorene see D4 in `MATERIALS.md`).
- Hopping enters H as −t. For bipartite lattices (honeycomb, square) the sign of t does not change T(E), even with on-site impurities, because a sublattice gauge flip maps +t to −t.
- **Impurity:** on-site energy +V with **V = 0.5 t** on randomly chosen sites. Site index = `cell * sites_per_cell + orbital`. Sites are drawn with `np.random.RandomState(seed).choice(n_sites, n, replace=False)`, which takes a prefix of one fixed permutation: a seed's impurity set at a higher count **contains** its set at a lower count (the nested seeds behind Bug #7).
- Ribbon width N = atomic rows across: dimer lines (armchair), zigzag chains (zigzag), Mo rows (MoS₂). A honeycomb ribbon has 2N sites per cell.

## Transport
Block convention: `H_{n,n}` = H0 (+ impurity shifts), `H_{n,n+1}` = H1, `H_{n+1,n}` = H1†.

**Leads** (Sancho–Rubio, lead broadening η = 1e-4):
- right lead extends to +∞ through H1 → surface GF `g_R = surface_gf(E, H0, H1)`, `Σ_R = H1 g_R H1†` on the last device cell;
- left lead extends to −∞ through H1† → `g_L = surface_gf(E, H0, H1†)`, `Σ_L = H1† g_L H1` on the first device cell;
- `Γ = i(Σ − Σ†)`.

**Caroli (bounded, standard; recommended for new data):** device broadening η = 1e-3; recursive `g_i = (z − H_i − H1† g_{i−1} H1 − Σ_R[i = last])⁻¹` with `Σ_L` in the first cell; `G_{N1} = g_N H1† G_{N−1,1}`; `T = Tr[Γ_R G_{N1} Γ_L G_{N1}†]`. Bounded by the open-channel count.

**Legacy trace (existing data):** `ca_sq.device()` and `agnr_lib.device_transmission()` evaluate
`T = |Tr[G̃_LL τ G̃_RR τ − τ G̃nl τ G̃nl]|` with `G̃ = G − G†`, using the *same* surface GF for both leads. It is not bounded: near subband edges it gives values above the channel count. Existing pipelines clip `T / T_pristine` to [0, 1]. Spikes are 0.36% of in-band AGNR points; for square-10 they rise from 1% of inputs at c = 5 to 29% at c = 90. The plan's `_legacy_trace` reproduces `ca_sq.device` exactly; with H1 = −I the sign flips cancel in both terms.

AGNR legacy data was standardised to `nonlocal_mode="IL"` with device broadening 1e-5 (LOGBOOK Bug #2); the legacy square generator uses 1e-3.

## Analytic checks (use them in tests)

| Quantity | Value |
|---|---|
| Open channels at E | number of k in (−π, π] with E_n(k) = E, divided by 2; clean Caroli T equals this away from subband edges |
| Square strip, width l (hopping −t) | transverse modes ε_p = −2t cos(pπ/(l+1)); channel p open when \|E − ε_p\| < 2t; band top 2t + 2t cos(π/(l+1)) < 4t |
| N-AGNR band top | t(1 + 2cos(π/(N+1))): 2.848 (N = 7), 2.902 (N = 9); never above 3t |
| Honeycomb (any edge) band limits | within ±3t |
| AGNR families (nearest-neighbour model) | 3p and 3p+1 gapped; **3p+2 (5, 8, 11, 14 …) metallic**. Real graphene edge relaxation (bond factor 1.12) opens a small 3p+2 gap |
| ZGNR (nearest-neighbour) | flat edge band at E = 0 |
| hBN (±Δ on-site) | gap 2Δ centred on E = 0; plain nearest-neighbour zigzag edge states sit at ±Δ, not inside the gap |
| Clean T above the band top | exactly 0 (no propagating lead states), which makes zero-padding the input window exact |

## Why the atlas input is label-free
Dividing a spectrum by *its own* material's pristine spectrum needs the material's identity, so any classifier fed that input is circular (Bug #8). The atlas input is `log1p(clip(round(T, 3), 0, 20)) / log1p(20)` on the shared 0–4 t grid for every spectrum. Per-material pristine normalisation is used only in stage 3, after the material has been *predicted*.

## Physics findings the design builds on
- Configuration-to-configuration fluctuations dominate: at 9-AGNR c = 40 vs 42, different seeds are 2.7× further apart than the same seed with two impurities added. That spread sets the accuracy floor on concentration.
- The full 0–3 t spectrum lowers concentration error by 17.5% compared with 0–1.5 t. Both halves have the same channel count, so the gain comes from sampling the same disorder at more energies.
- Impurity arrangement is not recoverable from one spectrum (rank correlation ≈ 0 at c = 20; the test was crude).
