# Smoke build at 50 configurations, and the even-width AGNR correction (2026-09-30)

For the implementing agent. The decisions below come from the human; the root cause and the checks come from the reviewer session. This file supersedes the "halted pending human decision" state of B4 in `2026-09-29-review-notes-for-implementing-agent.md`.

> **Correction, 09:35.** The even-m fix needs **two** changes, not one. The first version of this file named only the chain bond. The reviewer's band test had been run on your working tree, which already had your `m // 2` change to `T1_matrix`/`rho_matrix`. Keep that change and also remove the chain bond; the table below compares all four combinations against the committed code.

## Settled by the human
1. **Build the whole plan end to end at smoke scale first**: Phase 4 clouds, then Phase 5 atlas v2 and its generalisation test, with **50 configuration seeds per (model, density)**. This run proves that the data is correct and that every script works. The full run (10,000 seeds per density) comes later, with the same code and only the seed count and store changed.
2. **ZGNR uses the Caroli formula.** No trace-formula implementation exists for zigzag ribbons.
3. **Even-width AGNR keeps the trace formula** (`agnr_lib`, `nonlocal_mode="IL"`, d = 1e-5), with the geometry correction below.

## Status of your working tree at 09:35 (read before running anything)
- `agnr_lib.py` has change (a) but not (b), and `load_leads` still reads the stale even-m files. Nothing even-width is correct yet.
- `fingerprints.py` reran at 09:17 with that state. Every **even-width** `pristine.npy` in `engine_v1/graphene-ideal/armchair/` is off by 2–3 channels (max |pristine − open channels| at stable energies: N6 = 2.0, N8 = 3.0; N7 = 0.0). Regenerate them after steps 1–2.
- `generate_clouds.py` has already dropped both `NotImplementedError` halts, so a run now would write even-width clouds from the wrong lattice. The store's median-vs-pristine guard might not catch that, because a wrong lattice can still look physical. The band test (step 3a) is the check that does. Run no generation until 3a–3d pass.
- `generate_clouds.py`'s `__main__` still defaults to `engine_v1` and `seeds_for_width` (1,000 seeds). The smoke build needs `--store ~/atlas_store/smoke_v1 --n-seeds 50`.

## B4 root cause: `agnr_lib`'s unit cell is not honeycomb for even m
`unitcell` and `beta_matrix` connect every pair of consecutive sites 0…2m−1 as one chain, including the bond (m−1, m). That bond is a rung between the two columns at the last row, m−1.

In a honeycomb cell, rungs sit on even rows (0, 2, …) and the inter-cell bonds (`T1_matrix`, with the matching `rho_matrix` sites) sit on odd rows:
- For **odd m**, the last row is even. The chain bond is the correct rung there, and `T1` covers odd rows 1…m−2. Both range limits in the committed code, `(m − 1) // 2` and `m // 2`, give the same result.
- For **even m**, the last row m − 1 is odd, so it needs an inter-cell bond and no rung. The committed code gets both wrong. The chain puts a rung on it, which makes a 4-ring with the rung on row m − 2. `T1`/`rho` stop at `(m − 1) // 2` and leave the row without its inter-cell bond.

Evidence: Bloch bands compared with `tbribbon.lattices.honeycomb_ribbon(m, "armchair")` at 7 k-points, starting from the **committed** `agnr_lib`. "Coordination" is the range of bonds per atom; honeycomb is 2–3.

| Variant | m = 6 | m = 7 | m = 8 |
|---|---|---|---|
| Committed code | 0.48, coordination 2–3 | 4e-15 | 0.43, coordination 2–3 |
| Remove chain bond (m−1, m) only | 0.33, coordination **1**–3 | 4e-15 | 0.29, coordination **1**–3 |
| `T1`/`rho` range `m // 2` only (your current working tree) | 0.22 | 4e-15 | 0.17 |
| **Both changes** | **3e-15** | 4e-15 | **3e-15** |

With both changes, m = 10 and m = 16 also match to 4e-15.

Transport with the corrected cell (both changes), with leads recomputed from it by `leads_sancho_rubio(energy_grid(), 1e-5, 1.0, 0.0, m)`. Disorder columns use 12 seeds and every 3rd channel.

| m | Clean T vs open channels (stable energies) | max(median − pristine) at n_imp = 1 / 3 / 2m | Share of samples above pristine + 0.5 |
|---|---|---|---|
| 6 | 4e-8 | 0.0000 / 0.0000 / 0.0000 | 0 / 0.08% / 1.2% |
| 8 | 6e-8 | 0.0000 / 0.0000 / 0.0000 | 0 / 0.08% / 1.0% |

These match the validated odd widths (see B3's calibration on `size_7`/`size_9`).

**Stale files.** Four sets of files were built with the wrong cell:
- the even-m lead files `~/Desktop/backup/agnr/size_{m}/leads_{m}.npy` (they differ from corrected-cell leads by up to ~2e4);
- every even-width `pristine.npy` in `~/atlas_store/engine_v1/graphene-ideal/armchair/`;
- the quarantined N6/N8 clouds in `~/atlas_store/quarantine_b4/`;
- the N6 finding in the review notes, which says the two codebases likely define width differently. That is wrong: `honeycomb_ribbon` is correct, and the fault was `agnr_lib`'s even-m cell.

**The recompute procedure is sound.** `leads_sancho_rubio` reproduces the stored odd-width leads:
- m = 5, 9, 11: exact at all 300 energies.
- m = 7: exact everywhere except E = 1.00 t. That energy is an exact subband edge (cos(4π/8) = 0), where decimation is ill-conditioned (39 iterations, overflow warnings).

### Do
1. **Fix the cell (two changes, both needed).**
   - (a) In `T1_matrix` and `rho_matrix`, change the range to `m // 2`. Your working tree already has this.
   - (b) In `unitcell` and `beta_matrix`, zero entries (m−1, m) and (m, m−1) when m is even.

   Either change alone still gives wrong bands (see the table). Odd m must stay byte-identical: `test_agnr_lib_reproduces_stored_rows` must pass unchanged.
2. **Stop reading stale leads.** `load_leads(m)` must never return the stale even-m files:
   - For even m, compute leads with `leads_sancho_rubio(energy_grid(), 1e-5, 1.0, 0.0, m)`.
   - Cache them in a new directory whose name carries the cell version (for example `~/atlas_store/leads/agnr_cell_v2/size_{m}/leads_{m}.npy`).
   - Leave `~/Desktop/backup/agnr` untouched.
   - Odd m keeps the stored files, which the regression test validates.
3. **Add tests.**
   - (a) Bands of `unitcell` + `T1_matrix` equal the bands of `honeycomb_ribbon(m, "armchair")` for m = 5…16 (atol 1e-10).
   - (b) Clean T equals `open_channels` at stable energies for m = 6 and 8 (atol 1e-6).
   - (c) With one impurity, max(T − pristine) ≤ 0.05 for m = 6 and 8, over the first 60 channels and seeds 0–4. This mirrors `test_agnr_disorder_physical_invariant_on_odd_widths`.
   - (d) Add a test that computed leads are finite at all 300 energies for every width in the model list.
4. **Lift the halt.** Remove the even-width `NotImplementedError` in `generate_clouds.py` only after steps 1–3 pass. Regenerate every even-width `pristine.npy` with the corrected cell before any even-width cloud is written.
5. **Log it.** Add a LOGBOOK bug entry for B4 (root cause, fix, evidence), and mark the review notes' N6 hypothesis as superseded. Keep `quarantine_b4/` until the human says to delete it.

## ZGNR with Caroli
- Remove the zigzag `NotImplementedError`. Route zigzag models through `tbribbon.transport.spectrum(..., formula="caroli")` for both pristine and clouds, on the engine grid (0–4 t, 400 channels, `LeadCache`). `meta.json` records `caroli`, and the store already refuses a formula mismatch.
- Checks for each zigzag width:
  - Clean T equals `open_channels` at stable energies. For 6-ZGNR, Caroli already gave 1, 5, 5, 4, 3 at E = 0.35–2.2 t, matching the channel counts.
  - Caroli is bounded, so every sample must satisfy T ≤ pristine + 1e-6 at energies where the clean ribbon has no subband edge within ±0.02 t.
  - Clouds at different densities differ.

## The smoke build
- **Separate store and outputs.** Write only to `~/atlas_store/smoke_v1` and `notebooks/material_atlas/atlas_v2_smoke/`, never to `engine_v1` or `atlas_v2/`.
- **Seeds.** Add `--n-seeds` to `generate_clouds.py` (the default keeps `seeds_for_width`), and add `--store`/`--out` to every script that lacks them. Smoke run: seeds 0–49.
- **Models.** Use the plan's Task 13 list: armchair N = 5–16 (odd and even, after the fix) and zigzag N = 4–12 (Caroli), at densities 0.005, 0.01, 0.02 and 0.04.
- **Workers.** The machine is free now that the universal-transformer run has finished. Use `--n-jobs 12` to `16` with `OMP_NUM_THREADS=1` (see `TRAPS.md`).
- **Correctness report.** This is the point of the smoke run. A script such as `notebooks/tbribbon/check_store.py` writes `smoke_v1/report.json`.
  - Per model:
    - formula;
    - max |clean T − open channels| at stable energies;
    - per density: max(median − pristine), the share of samples above pristine + 0.5, and the maximum T;
    - no identical rows across densities;
    - **seed nesting**: for each seed, its impurity set at a lower density is contained in its set at a higher density.
  - The report ends with one pass/fail line per model.
- **Atlas.** Run `build_atlas_v2.py --store ~/atlas_store/smoke_v1 --out notebooks/material_atlas/atlas_v2_smoke` with the held-out widths unchanged (armchair 8, 12, 13; zigzag 8). With 50 seeds the Gate 5 numbers are noisy. The smoke run must show the pipeline runs and the numbers are sane; it does not have to pass Gate 5. Label the numbers **SMOKE** in the LOGBOOK, not as a result.
- **Done when:**
  - every listed model has 4 densities × 50 seeds in `smoke_v1`;
  - every line of `report.json` passes;
  - the full test suite passes;
  - `atlas_v2_smoke/generalisation.json` exists;
  - a LOGBOOK entry "SMOKE-1" summarises the report;
  - everything is committed and pushed.

  Then **stop and report to the human** before any full-scale generation.
- **Full run, later.** Same commands, with `--store ~/atlas_store/engine_v1` and the seed count the human chooses (10,000 per density was mentioned). Put a time estimate in the SMOKE-1 entry, measured from the smoke run's own spectra-per-second for each width. At 1,000 seeds, N7–N8 took about 7 minutes per density on 4 workers.
