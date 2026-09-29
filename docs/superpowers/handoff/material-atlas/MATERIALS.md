# Real materials in tight binding (Phase 3b)

Collected 2026-09-29 from primary sources, all opened. **[calc]** means derived from the published formulas, not stated in the paper. **[unverified]** means not confirmed from an opened source. Order of work: realistic graphene → hBN → MoS₂ → phosphorene; armchair and zigzag, widths N = 7, 9, 14, 27, 50.

## Graphene ribbons (one pz orbital per site)

| Set | ε | t1 | t2 | t3 | Notes |
|---|---|---|---|---|---|
| **Nearest neighbour (usual default)**, Son–Cohen–Louie 2006 | 0 | 2.7 eV | – | – | armchair edge dimer bonds (1 + δ)t with **δ = 0.12** |
| Reich 2002 (full-BZ fit) | −0.28 | −2.97 | −0.073 | −0.33 | non-orthogonal: overlaps 0.073/0.018/0.026 (engine needs ES − H) |
| Tran 2017 (fit to GNR DFT) | −0.187 | 2.756 | 0.071 | 0.38 | overlaps 0.093/0.079/0.070 |
| Gunlycke–White 2008 | 0 | 3.2 | – | 0.3 | edge Δt1 = 0.0625 t1 (secondary source) |

- Son's edge factor: the outermost armchair bonds parallel to the dimer lines are 3.3–3.5% shorter, which gives about 12% larger hopping. In the plan's builder this is `honeycomb_ribbon(..., edge_bond_factor=1.12)`.
- Castro Neto (RMP 2009): t ≈ 2.8 eV, 0.02t ≲ t′ ≲ 0.2t. A second-neighbour t′ shifts the Dirac point to 3t′ [calc], which moves E = 0.
- Nearest-neighbour armchair 3p+2 is metallic; δ opens a gap. Zigzag has a flat edge band at E = 0 (a gap there needs Hubbard U or a staggered potential).
- Hancock 2010 3NN + Hubbard values 2.70/0.20/0.18 eV are **[unverified]**.

## hBN (one pz orbital per site; +Δ on B, −Δ on N)

| Set | t | Δ | Δ/t | Fitted to |
|---|---|---|---|---|
| **Galvani 2016** | 2.30 eV | 3.625 eV | 1.58 | GW gap 2Δ = 7.25 eV |
| Ribeiro–Peres 2011 | 2.33 eV | 1.96 eV | 0.84 | GGA valence π band |

- Optional t2 = 0.096 eV (Galvani). Particle-hole symmetric, so it fits the conduction band less well.
- E = 0 is mid-gap (charge neutrality). Band limits ±√(Δ² + 9t²): ±3.39t (Galvani) or ±3.12t (Ribeiro) [calc]; fits the 0–4t window.
- Ribbons (Park & Louie, LDA): zigzag gaps converge about 0.7 eV below the sheet gap because of edge states (valence-band top on the N edge, conduction-band bottom on the B edge). Armchair conduction-band bottom is a B-edge state that needs modified edge on-site energies to reproduce. In the plain model, zigzag edge states sit at ±Δ [calc].

## MoS₂: three-band model (Liu, Shan, Yao, Yao, Xiao, PRB 88, 085433, 2013)

Basis: Mo dz², dxy, dx²−y² on a triangular lattice (one Mo per cell); hoppings along R1, the rest by D3h symmetry. t0 is dz²–dz²; t1 dz²–dxy; t2 dz²–dx²−y²; t11, t12, t22 the dxy/dx²−y² pairs.

| Fit | a (Å) | ε1 | ε2 | t0 | t1 | t2 | t11 | t12 | t22 |
|---|---|---|---|---|---|---|---|---|---|
| GGA | 3.190 | 1.046 | 2.104 | −0.184 | 0.401 | 0.507 | 0.218 | 0.338 | 0.057 |
| LDA | 3.129 | 1.238 | 2.366 | −0.218 | 0.444 | 0.533 | 0.250 | 0.360 | 0.047 |

- Spin-orbit λ = 0.073 eV (H0 ± (λ/2)Lz per spin). A third-nearest-neighbour version (19 parameters, Table III) fits the whole Brillouin zone; the NN model is accurate near ±K only.
- Direct gap at K: 1.66 eV (GGA NN) or 1.84 eV (LDA NN) [calc].
- **Energy convention problem (D4):** E = 0 lies near the valence-band top, and the Fermi level of the undoped layer sits in the gap (0 to about 1.6 eV). With t = largest NN hopping (t2 = 0.507 eV, GGA), the band spans about −1.1t to +6.9t and the conduction band starts at 3.15t [calc, Γ/K/M]. The 0–4t window holds only part of it.
- Ribbons: Liu Appendix A (zigzag W = 8) reproduces the two Mo-d edge bands but misses dyz and S-p edge bands; Chu et al. PRB 89, 155317 (zigzag edge bands); Ridolfi et al. PRB 95, 035430 (RGF transport on both edges, but with an 11-band model). No armchair RGF study with the Liu model was found.

## Phosphorene (one orbital per site, 4 sites per cell)

**Rudenko & Katsnelson, PRB 89, 201408(R) (2014)** (usual default; GW fit, accurate within about 0.3 eV of the band edges):

| Hopping | eV | Distance (Å) | Neighbours |
|---|---|---|---|
| t1 | −1.220 | 2.22 | 2 |
| t2 | 3.665 | 2.24 | 1 |
| t3 | −0.205 | 3.34 | 2 |
| t4 | −0.105 | 3.47 | 4 |
| t5 | −0.055 | 4.23 | 1 |

- Gap: GW 1.60 eV; model 4t1 + 2t2 + 4t3 + 2t5 = 1.52 eV, direct at Γ (Ezawa 2014).
- Alternative (Rudenko, Yuan, Katsnelson, PRB 92, 085419, 2015; GW0): 10 hoppings −1.486, 3.729, −0.252, −0.071, −0.019, 0.186, −0.063, 0.101, −0.042, 0.073 eV (2.22–5.49 Å); gap 1.84 eV; valid over several eV. Erratum PRB 93, 199906 **[unverified]**.
- Anisotropy: electron masses 0.164 m0 (armchair, Γ–X) vs 0.873 m0 (zigzag, Γ–Y); holes 0.179 vs 1.175 m0.
- **Energy convention (D4):** mid-gap at 4t4 = −0.42 eV (conduction band +0.34, valence band −1.18 eV at Γ) [calc]. Band about −6.0 to +6.9 eV = −1.65t2 to +1.9t2, inside ±4t.
- Ribbons: zigzag has a quasi-flat edge band inside the gap (clear for widths ≳ 3 nm); armchair has no in-gap edge states, gap ≈ 20.4 eV / Na^1.92 + 1.52 eV.

## Decisions to put to the human before Phase 3b
- **D3:** graphene nearest-neighbour + edge factor (default) vs a longer-range set; hBN Galvani vs Ribeiro–Peres; MoS₂ GGA vs LDA (and whether to include spin-orbit); phosphorene 5- vs 10-hopping.
- **D4:** how the shared energy window and E = 0 apply to MoS₂ (bands extend to about 6.9t) and phosphorene (mid-gap at −0.42 eV).
- **D5:** "positive energies only" drops the valence bands of MoS₂ and phosphorene, which conflicts with "as complete as possible electronic structure".

## Sources opened
Son, Cohen, Louie PRL 97, 216803 (2006) arXiv:cond-mat/0611602 · Reich et al. PRB 66, 035412 (2002) · Castro Neto et al. RMP 81, 109 (2009) arXiv:0709.1163 · Tran et al. AIP Adv. 7, 075212 (2017) arXiv:1702.02606 · Hancock et al. PRB 81, 245402 (abstract) · Ribeiro & Peres PRB 83, 235312 (2011) arXiv:1101.3950 · Galvani et al. PRB 94, 125303 (2016) arXiv:1605.09581 · Park & Louie Nano Lett. 8, 2200 (2008) arXiv:0808.1833 · Liu et al. PRB 88, 085433 (2013) arXiv:1305.6089 · Chu et al. PRB 89, 155317 (2014) arXiv:1308.2032 · Ridolfi et al. PRB 95, 035430 (2017) arXiv:1610.00734 · Rudenko & Katsnelson PRB 89, 201408(R) (2014) arXiv:1404.0618 · Rudenko, Yuan, Katsnelson PRB 92, 085419 (2015) arXiv:1506.01954 · Ezawa NJP 16, 115004 (2014) arXiv:1404.5788 · Taghizadeh Sisakht et al. PRB 91, 085409 (2015) arXiv:1408.6249
