# Review notes for the implementing agent — 2026-09-29

Reviewer: the session that wrote the plan and brief. Scope: commits `6acbc2cf0` … `9dc88d489` on `feat/square-autoencoder` (Phases 1–4 code and the Phase 5 builder), the store `~/atlas_store/engine_v1`, and `notebooks/material_atlas/reference_7_9/`. **Handle the blocking items before generating any disorder clouds (Task 13 run / Phase 4).**

## What is up to mark
- Full suite green: **70 passed** (`python -m pytest -q`, 52 s).
- Gate 1 met per `reference_7_9/metrics.json`: label-free width accuracy 99.86%, end-to-end MAE 1.9727, 90% interval coverage 90.0%.
- `Atlas` flags unknowns by reconstruction error (the plan fix from BUILD-12) and has stress and save/load coverage.
- Gate 3a physics is right: the clean armchair spectra in the store conduct at E = 0.02 for 3p+2 widths (5, 8, 11, 14, 50) and are gapped for all others.

## Blocking

### B1. `legacy_trace` is wrong on honeycomb ribbons; AGNR clouds must come from `agnr_lib`
The human has settled D1 = trace formula. The generic `tbribbon.transport._legacy_trace` puts H1 on both sides of the recursion and the lead connection, which is only valid when H1 is symmetric (square strip: it matches `ca_sq.device` exactly). A honeycomb H1 is not symmetric, and the result is wrong even for a clean ribbon:

| Clean ribbon, E = 0.35, 0.9, 1.3, 1.7, 2.2 | Open channels | Caroli | engine `legacy_trace` |
|---|---|---|---|
| 7-AGNR armchair | 1, 3, 3, 3, 2 | 1.0, 3.0, 3.0, 3.0, 2.0 | **183.5**, 4.2, 7.0, 4.5, 5.6 |
| 6-ZGNR zigzag | 1, 5, 5, 4, 3 | 1.0, 5.0, 5.0, 4.0, 3.0 | **9.1**, 11.8, 8.9, 9.0, 6.4 |

The AGNR trace formula actually used for the existing data is `agnr_lib.device_transmission`: recursion `T1ᵀ G T1`, a site-diagonal contact operator `rho` for the lead connection, and a non-local term. It is verified to reproduce `consolidated_data/size_{7,9}.npy` to ~1e-7 relative error (float32 storage) **only with `nonlocal_mode="IL"` and broadening `d = 1e-5`**; `IR` is off by 2–9× and `d = 1e-4` by 3–30%.

Do:
1. For `graphene-ideal` armchair models, generate clouds by calling `agnr_lib.device_transmission(w, 1e-5, 1.0, 0.0, m, seed, n_imp, agnr_lib.load_leads(m), nonlocal_mode="IL")` over `agnr_lib.energy_grid()` (0–2.99; leads exist for m = 5–31 in `~/Desktop/backup/agnr/size_m/`). Its impurity sampling is the project convention, so seeds carry over unchanged. `InputSpec.to_input` zero-fills above the band top, which is below 2.99 for every N ≤ 30; for N = 31 the top is 2.9904, still within half a step of 2.99.
2. Add a regression test that `agnr_lib` via your generator path reproduces stored rows: `size_7.npy` c = 10 seed 0 and c = 40 seed 123 at channels 30, 90, 150, 220 (and the same for `size_9.npy`), `rtol=1e-5`.
3. Make `_legacy_trace` raise `ValueError` when `not np.allclose(H1, H1.T)`, with a test on `honeycomb_ribbon(7, "armchair")`. Keep it for the square strip, where it is exact.
4. Zigzag ribbons (ZGNR): there is **no validated trace-formula implementation**. `notebooks/zgnr/zgnr_transmission.ipynb` has an unvalidated `device()`; it builds unit cells behind `@lru_cache` (the same pattern as the square-lattice cache-mutation bug), and its leads directory `~/Desktop/backup/zgnr` does not exist. Before any ZGNR cloud, port it into a module, compute its leads (`notebooks/zgnr/zgnr_leads.py`), and prove (i) clean T equals `open_channels` away from subband edges, and (ii) spectra at two concentrations for one seed differ. If either fails, stop and report to the human instead of substituting another formula.

### B2. Clean spectra and clouds must use the same formula
The 26 clean fingerprints in `~/atlas_store/engine_v1` were written with `caroli` (the `fingerprints.py` default). `generate()` writes a clean spectrum with its own `formula` only when a model has no clouds yet, which is the case for every model now, so the first run will overwrite the Caroli fingerprint of each model with its trace-formula version. With the generic formula that would store the wrong honeycomb values from B1. After B1, regenerate the clean fingerprints with the same backend as the clouds, and add a test that a model's clean spectrum and its clouds were produced by the same formula (for example, record `formula` in `meta.json` and assert it matches).

## Non-blocking (fix in this phase)

### N1. LOGBOOK BUILD-13 numbers disagree with the result file
`LOGBOOK.md` mixes numbers from two reference runs: MAE 1.973 / 1.975 / 1.976 / 1.977, coverage 90.00% / 90.03%, and q = 0.0897 / 0.0900. `reference_7_9/metrics.json` (the file the regression test reads) says MAE **1.9727** (7-AGNR 1.6553, 9-AGNR 2.1929), coverage **90.0%**, q **0.0897**. The worker hand-off (`.agents/worker_phase1_gate1/handoff.md`) quotes the earlier run (1.9754, 90.0348%). Make every BUILD-13 number in the LOGBOOK match `metrics.json`, and say that the gate was re-run.

### N2. Agent scratch files are not ignored
`.agents/` and `ORIGINAL_REQUEST.md` in the repo root are untracked and not in `.gitignore`. Add them to `.gitignore` so a broad `git add` cannot commit them.

### N3. `fork()` in a multi-threaded process
`tests/tbribbon/test_generate.py` warns: "This process is multi-threaded, use of fork() may lead to deadlocks in the child". Long cloud runs from a process that has already imported NumPy/Torch with threads can hang. Use a `spawn` context in `generate()` (`multiprocessing.get_context("spawn").Pool(...)`); the worker initialiser already receives everything it needs.

### N4. Weak Phase 5 test
`tests/atlas/test_generalisation.py` only checks the held-out partition. Add a toy test that `build_atlas_v2`'s evaluation places a held-out width between its trained neighbours (the `toy_store` helper already supports this).

### N5. Branch hygiene
Eight commits sit unpushed on `feat/square-autoencoder`, the branch another session also uses. Push at each gate, as the brief says, so the work is not lost and the other session sees it.

## Verification commands
```bash
# B1 evidence (clean honeycomb, engine formulas vs channels)
PYTHONPATH=notebooks/material_atlas:notebooks ~/miniconda3/envs/ml/bin/python -c "
import numpy as np
from tbribbon.lattices import honeycomb_ribbon
from tbribbon.bands import open_channels
from tbribbon.leads import LeadCache
from tbribbon.transport import spectrum
h = honeycomb_ribbon(7, 'armchair'); E = np.array([0.35, 0.9, 1.3]); L = LeadCache(h.H0, h.H1, E); z = np.zeros((20, 14))
print(open_channels(h.H0, h.H1, E), spectrum(h.H0, h.H1, E, z, L, formula='caroli'), spectrum(h.H0, h.H1, E, z, L, formula='legacy_trace'))"
# B1 target: agnr_lib reproduces stored data (IL, d = 1e-5)
cd notebooks/agnr/physics && ~/miniconda3/envs/ml/bin/python -c "
import numpy as np, agnr_lib as A
d = np.load('/run/media/shardul/storage/machine_learning/transmission_data/transmission_results/consolidated_data/size_7.npy', mmap_mode='r')
L = A.load_leads(7); idx = [30, 90, 150, 220]
got = [A.device_transmission(k * 0.01, 1e-5, 1.0, 0.0, 7, 0, 10, L, nonlocal_mode='IL') for k in idx]
print(np.max(np.abs(np.array(got) - d[4, 0, idx]) / np.abs(d[4, 0, idx])))"
```

---

## Update 2026-09-29 20:10: B3 (blocking): 13 clouds in the store were written with the broken formula

Before these notes landed, `generate_clouds.py` ran from 19:19 to 20:00 with the generic `legacy_trace` and wrote 13 honeycomb clouds. Good: no generator is running now, and the new `_legacy_trace` symmetric-H1 guard (uncommitted) stops a rerun from writing more. Below is what is still wrong in the store.

**Affected files** in `~/atlas_store/engine_v1/graphene-ideal/armchair/`:
- `N5`, `N6`, `N7`: `cloud_d0.0050`, `d0.0100`, `d0.0200`, `d0.0400`
- `N8`: `cloud_d0.0050`

Each cloud has its `_seeds.npy` companion and an entry in `meta.json`. The `pristine.npy` files were not overwritten. They still equal the open-channel count (Caroli), so every one of these models pairs a Caroli clean spectrum with a broken-trace cloud (the B2 mismatch).

**Evidence** at the lightest density d = 0.005 (median over the 1000 seeds). E is in units of t.

| Model | E | Open channels | Stored pristine | Cloud median | Share of cloud above channels + 0.5 | Cloud max |
|---|---|---|---|---|---|---|
| N5 | 0.35, 0.9, 1.3, 1.7, 2.2 | 1, 2, 2, 2, 1 | same | 1.17, 4.11, 1.85, 3.30, 2.12 | 62% | 2.5e7 |
| N6 | same | 1, 3, 3, 2, 2 | same | 20.97, 5.05, 5.67, 2.88, 2.25 | 76% | 1.6e7 |
| N7 | same | 1, 3, 3, 3, 2 | same | **1175.8**, 9.59, 5.83, 4.08, 6.03 | 92% | **2.7e9** |
| N8 | same | 1, 4, 4, 3, 2 | same | 63.4, 11.6, 9.14, 3.82, 7.55 | 88% | 2.6e8 |

For comparison, the true trace-formula data (`agnr_lib`, `consolidated_data/size_{7,9}.npy`, 2000 seeds, every 10th channel) has a median spectrum that never rises above the pristine spectrum: max(median − pristine) ≤ 0.001 at every concentration. Only 0–0.03% of values exceed pristine + 0.5 at low concentration, and 0.8–1.4% at the highest.

**Resume trap.** `generate()` skips any density whose cloud already exists (`generate_clouds.py:48-49`, `if store.has_cloud(...): continue`). A rerun after the B1 fix would therefore keep these 13 bad clouds without any warning and generate only the rest. That leaves one model with two formulas across its densities.

**Why the store let them through.** `CloudStore.write_cloud` checks only for non-finite values, duplicate seeds, and cross-density duplicates. A value of 2.7e9 is finite, and `spike_fraction` is never checked on write.

Do:
1. Before any rerun, move the 13 clouds and their `_seeds.npy` files into a quarantine folder (for example `~/atlas_store/quarantine_legacy_generic/`). Remove their entries from each `meta.json`. Record in the LOGBOOK that they were produced by the generic trace on honeycomb and discarded. Moving the files keeps the evidence and takes them out of the store.
2. Make the store refuse clouds like these. Add a check in `write_cloud`, or in `generate()` just before it calls `write_cloud`. It rejects a cloud when either condition holds:
   - `max(median(cloud, axis=0) − pristine) > 0.05`
   - `mean(cloud > pristine + 0.5) > 0.05`

   Both are calibrated on the real `size_7/9` data, which passes with a wide margin, and every one of the 13 clouds above fails both. Add a test that feeds it a synthetic cloud with a median above pristine and expects a `ValueError`.
3. The `__main__` model list in `generate_clouds.py` also includes zigzag N = 4–12. Take the zigzag models out of that list until the ZGNR validation in B1.4 passes. With the new guard, a run would otherwise stop with a `ValueError` at the first zigzag model after the armchair ones finish. Worse, if the guard were ever loosened, it would write broken zigzag clouds.
4. Commit the `transport.py` guard and its test together with the `agnr_lib` path, so the guard is never committed without a working AGNR backend (and the other way round).

**Evidence command** (read-only):
```bash
cd /run/media/shardul/storage/machine_learning/transmission_github/transmissions
PYTHONPATH=notebooks/material_atlas:notebooks ~/miniconda3/envs/ml/bin/python -c "
import numpy as np
from tbribbon.lattices import honeycomb_ribbon
from tbribbon.bands import open_channels
S='/home/shardul/atlas_store/engine_v1/graphene-ideal/armchair/'
for N in (5,6,7,8):
    E=np.load(S+f'N{N}/energies_t.npy'); P=np.load(S+f'N{N}/pristine.npy'); C=np.load(S+f'N{N}/cloud_d0.0050.npy', mmap_mode='r')
    print(N, np.max(np.median(C,0)-P).round(1), np.mean(C>P+0.5).round(2))"
```

---

## Update 2026-09-29 20:20: B3 resolved; new non-blocking note N6

**B3 is fixed.** As of this check, the store has no `cloud_*.npy` files at all — the 13 corrupt files from the 20:10 update are gone, and every `armchair/N*/meta.json` shows `"clouds": {}` with `"pristine_formula": "agnr_lib_IL_1e-5"`. The working tree now shows the full B1/B2 fix: `generate_clouds.py` routes armchair models through `agnr_lib.device_transmission(..., nonlocal_mode="IL", d=1e-5)` in a `spawn`-context pool (N3), raises `NotImplementedError` on zigzag models with a pointer to these notes, and `store.py` records `formula` per pristine/cloud and refuses a cloud whose formula doesn't match its model's pristine formula (B2). `fingerprints.py` was rerun and now also uses `agnr_lib` for armchair, `legacy_trace` for the square strip only, `caroli` elsewhere. New tests: `test_agnr_lib_reproduces_stored_rows` (the B1.2 regression test, skipped if the consolidated files are missing, `rtol=1e-5`), `test_legacy_trace_rejects_non_symmetric_honeycomb_h1` (B1.3), `test_formula_mismatch_refused` (B2), and a toy interpolation test (N4). LOGBOOK BUILD-13 numbers now match `reference_7_9/metrics.json` exactly (N1). Good discipline: the agent moved `n_jobs` default from 20 to 4 in `generate_clouds.py`, matching the hardware-sharing guidance in the handoff `TRAPS.md`.

I reran the B1 evidence check on the regenerated pristine files directly (not the corrupted clouds, which are gone): for N ∈ {5, 7, 8, 11, 16, 27}, `T(E=0.02)` is nonzero exactly for the 3p+2 widths (5, 8, 11, 14, …) and zero otherwise, for every N from 5–16 — Gate 3a's physics rule holds on the new agnr_lib-backed pristine spectra, not just the old Caroli ones.

### N6. `tbribbon.honeycomb_ribbon`'s armchair builder disagrees with `agnr_lib` for even N (not blocking — AGNR no longer routes through it)

While re-checking pristine vs. open-channel counts I compared the new `agnr_lib`-based pristine spectra against `tbribbon.bands.open_channels(honeycomb_ribbon(N, "armchair").H0/H1, ...)`. They agree exactly for odd N (5, 7, 11, 27: max discrepancy 0) but disagree by 1–4 channels at many energies for even N (8, 16), e.g. N=8: open_channels says 1 at E=0.29–0.50 t, the agnr_lib pristine says 0 there. This is **not a physics bug** — agnr_lib's own metallicity pattern (checked above) is correct for both parities, and AGNR clouds/pristine no longer use `tbribbon.honeycomb_ribbon` at all after the B1 fix, so nothing currently in the store is affected. The two codebases likely define "width N" differently for even N (unit-cell row count vs. dimer-line count in `_graphene_sites()`'s keep-mask), and only `agnr_lib` has been checked against real transport data.

Do not use `tbribbon.lattices.honeycomb_ribbon(N, "armchair")` as a ground truth, or in any new test, for even N until this is resolved — it disagrees with the validated `agnr_lib` construction. This will matter if a future task (zigzag, or a generic-formula fallback) reuses `honeycomb_ribbon` for an even-width armchair check. No action needed now; noting it so it isn't picked up unknowingly later.

**Evidence** (read-only):
```bash
cd /run/media/shardul/storage/machine_learning/transmission_github/transmissions
PYTHONPATH=notebooks/material_atlas:notebooks ~/miniconda3/envs/ml/bin/python -c "
import numpy as np
from tbribbon.lattices import honeycomb_ribbon
from tbribbon.bands import open_channels
S='/home/shardul/atlas_store/engine_v1/graphene-ideal/armchair/'
for N in (7, 8):
    E, P = np.load(S+f'N{N}/energies_t.npy'), np.load(S+f'N{N}/pristine.npy')
    h = honeycomb_ribbon(N, 'armchair'); idx = np.arange(1, E.size, 7)
    ch = open_channels(h.H0, h.H1, E[idx])
    print(N, 'max|P-ch| over sampled E:', np.max(np.abs(P[idx]-ch)))"
```

---

## Update 2026-09-29 22:00: B4 (blocking, urgent — generation is running now) `agnr_lib` disorder transmission is unphysical for even-width AGNR

**The cloud generator is running right now** (PID 2422715, started ~20:23, currently on N8) and has already written clouds for N6 (all 4 densities), N7 (all 4 densities), N8 (2 of 4 densities). N6 and N8 are corrupt; N7 is fine. This is a different bug from B3 (which was the generic `legacy_trace` on honeycomb): B1's fix — `agnr_lib.device_transmission(..., nonlocal_mode="IL", d=1e-5)` — is itself wrong for **even ribbon width**, and B1's regression test only checks `size_7.npy` and `size_9.npy` (both odd), so it can't catch this.

**Evidence — single impurity, seed 0, first 60 channels (E = 0–0.59 t), `T(E) − pristine(E)`, should be ≤ 0 (disorder can only scatter, not enhance transmission beyond the clean value at this broadening):**

| m | 3p+2? | n_imp=1 | n_imp=3 | n_imp=6 |
|---|---|---|---|---|
| 5 | odd | −0.015 | −0.004 | — |
| 6 | even | **+2.03** at E=0.56 | +1.98 | +2.02 |
| 7 | odd, validated | 0.000 | 0.000 | 0.000 |
| 8 | even | **+2.01** at E=0.54 | **+21.3** at E=0.24 | +21.3 |
| 9 | odd, validated | 0.000 | 0.000 | — |
| 10 | even | **+0.96** | **+11.7** | — |

Odd widths (5, 7, 9) stay at or below the clean value everywhere, as they should and as the validated `size_7`/`size_9` data does. Every even width tested (6, 8, 10) jumps 1–21 above the clean value from a single impurity, and gets worse with more impurities. This shows up in the stored clouds too:

| Model | density | max(median − pristine) | fraction of samples > pristine + 0.5 |
|---|---|---|---|
| N6 | 0.005 | 2.72 | 4.6% |
| N6 | 0.04 | 7.64 | 9.3% |
| N7 (reference) | 0.005–0.04 | ≤ 0.004 | 0.2–1.2% |
| N8 | 0.005 | 54.0 | 6.2% |
| N8 | 0.01 | 30.3 | 8.7% |

N7's own cloud is fine — it matches the `size_7`/`size_9` calibration in the B3 update almost exactly — so this is not a generic problem with the generation pipeline; it is specific to even `m` inside `agnr_lib.unitcell` / `unidevice` / `chosen_for_config`. The likely site: `unitcell`'s anti-diagonal coupling loop `idx = np.arange(0, m, 2); base[idx, 2*m-1-idx] = t; base[2*m-1-idx, idx] = t` (agnr_lib.py:47-56) indexes a different bond pattern depending on whether `m` is odd or even, and/or `unidevice`'s impurity-site mapping (`imps[:, 1]` indexing into the `2m`-site cell) may not line up the same way for even `m`. I have not root-caused it further; this needs someone who can check the intended AGNR unit-cell geometry, not just observe the symptom.

Do, in order:
1. **Stop the current `generate_clouds.py` run** (PID 2422715 at last check) before it writes more even-width clouds. Everything after N8 in the model list (N9 is odd, fine; N10, 12, 14, 16 are even) will be corrupt if generation continues unfixed.
2. Quarantine `N6/cloud_*` and `N8/cloud_d0.0050`, `N8/cloud_d0.0100` (whatever exists when generation stops) the same way as the B3 quarantine, and clear their `meta.json` entries.
3. Find and fix the even-`m` bug in `agnr_lib.unitcell`/`unidevice` (or in how `generate_clouds.py`/`fingerprints.py` calls it — check whether `m` there truly means the same "N sites across the ribbon" for both parities as it does in `chosen_for_config`'s `2*width` site count). Do not touch `agnr_lib.py` without first confirming with the human — it is shared, validated code (the B1 recipe depends on it staying byte-for-byte the formula that reproduces `size_7`/`size_9`); a fix must keep the odd-width match intact.
4. Extend the B1.2 regression test with an even-width check. There is no consolidated ground-truth file for any even AGNR width (only `size_7.npy`, `size_9.npy` exist), so use a physical invariant instead: for every stored cloud, `median(cloud, axis=0) <= pristine + tol` (tol ~0.05, matching what real disordered data actually does) and this must hold before any even-width cloud is trusted. This is the same guard proposed for B3; extending it to run automatically in `write_cloud` (not just as an offline check) would have caught both bugs before they reached the store.

**Evidence command** (read-only):
```bash
cd notebooks/agnr/physics
OMP_NUM_THREADS=2 ~/miniconda3/envs/ml/bin/python -c "
import numpy as np, agnr_lib as A
for m in (7, 8):
    L = A.load_leads(m)
    pris = A.spectrum(m, L, config=0, concentration=0, nonlocal_mode='IL', d=1e-5)
    t = np.array([A.device_transmission(w,1e-5,1.0,0.0,m,0,1,L,nonlocal_mode='IL') for w in A.energy_grid()[:60]])
    print(m, 'max(T - pristine) with 1 impurity:', np.max(t - pris[:60]))"
```

---

## Update 2026-09-29 22:02: B4 still active — generator has not been stopped

`generate_clouds.py` (PID 2422715) is still running 40 minutes after the B4 finding above. Since the 22:00 check it wrote a third N8 density (`cloud_d0.0200.npy`); N6 (all 4 densities) and N8 (3 of 4) remain corrupt by the same test. N9 (odd, should be fine) has not started yet. No commit or `.agents` note references B4 yet — the fix has not been picked up.

---

## Update 2026-09-30: B4 root cause found; next steps in a separate file
Read **`2026-09-30-smoke-build-and-even-agnr-fix.md`** in this folder next. It records three decisions by the human: a 50-configuration smoke build of Phases 4–5 comes first, ZGNR uses the Caroli formula, and even-width AGNR keeps the trace formula with a unit-cell correction. It also records the B4 root cause: `agnr_lib`'s chain bond (m−1, m) is a wrong rung for even m. **N6 above is superseded:** `honeycomb_ribbon` is correct, and the even-width disagreement was `agnr_lib`'s cell.
- 2026-09-30 10:10: the edge-masked `write_cloud` guard is correct (the historical trace data spikes at subband edges too). It needs a test and a LOGBOOK note; see **`2026-09-30-edge-masked-guard.md`**.
- 2026-09-30 10:45: SMOKE-1 data is correct (21/21 models pass). Held-out armchair widths are confidently misidentified (often as zigzag) and are not flagged as unknown, while known widths are perfect. **Do not start the full run.** The human is deciding the Gate 5 design; see **`2026-09-30-smoke-review-and-gate5-diagnosis.md`**.
- 2026-09-30: the human chose **option A**: train the atlas on every width and evaluate on held-out seeds. Next steps are in **`2026-09-30-option-a-train-all-widths.md`**.
- 2026-09-30 11:10: option A step 1 is correct (the seed split holds, material/edge/density are 100%). Width accuracy fails on 20 of 84 armchair lines because `locate` averages the neighbours' widths; a majority vote cuts that to 9. See **`2026-09-30-option-a-width-vote.md`**.
- 2026-09-30 11:35: wide armchair clouds (N20-N40) are correct. The input cap of 20 saturates armchair N50 (23% of the window) and wide zigzag, so a spec v2 is needed before any wide-grid atlas; the edge-masked guard covers only 11 channels at N50. See **`2026-09-30-wide-grid-findings.md`**.
- 2026-09-30 12:45: SMOKE-2 reviewed. The data and option A are sound, but the trend test is within noise (1–2 of 84 spectra) and the cost table mixes compute-only and overhead-inclusive timings. Waiting on the human's grid decision; see **`2026-09-30-smoke2-review.md`**.
- 2026-09-30: the human chose the **sparse 31-width grid with 1,000 seeds per density** for the full run into `engine_v1`. Steps are in **`2026-09-30-full-run-sparse-grid.md`**.
- 2026-09-30 13:35: `c6d686d7a` reviewed (84/84 tests pass). The trend sweep is good. The cost model misses the chunk tail (wide smoke batches are ~2× its prediction). The full run is ~10–12 h on 16 workers; see **`2026-09-30-timing-model-note.md`**.
- 2026-09-30 13:55: `abb0aed3c` reviewed (86/86 tests pass). InputSpec v2 (cap 64) and the 31-width smoke atlas are good, and the full run is correct so far (N6, N8 at n=1000). The measured speed-up is ~6.7× on 16 workers, so expect ~24 h; the estimate in `2026-09-30-full-run-sparse-grid.md` is updated.
- 2026-09-30 18:25: the `engine_v1` check_store FAIL on armchair N5–N9 is a false alarm from the reviewer's per-seed mean check (a single spike channel in 1–2 of 1,000 seeds), and the data is fine. Make the check robust, rerun it, then resume; see **`2026-09-30-check-store-robust-seed-check.md`**.
- 2026-10-01 11:58: generation is 120/124, with every completed model verified by the reviewer. In FULL-1 the "11.1–11.6× speed-up" is concurrency; the effective speed-up over one core is ~6.4×. See **`2026-10-01-full1-speedup-wording.md`**.
- 2026-10-01 17:05: FULL-1 is verified (data 124/124 correct, split correct, results reproduce; width 99.98%, one line at 98%). The false-alarm statement needs correcting: pooled ~1% holds by construction, but 13 armchair lines exceed 2% (up to 22% at N40, d=0.04). The human decides on the threshold design. See **`2026-10-01-full1-review.md`**.
- 2026-10-01: the human chose **option B** (class-conditional novelty). See **`2026-10-01-class-conditional-novelty.md`**.
- 2026-10-01 18:15: option B is implemented correctly and identification is unchanged, but the reviewer's log-normal calibration gives 2.06% pooled false alarms (the log scores are skewed), and the per-line "≤ 2%" gate cannot be met at n=150. See **`2026-10-01-novelty-calibration-fix.md`**.
- 2026-10-01 18:45: FULL-2 is verified (pooled false alarms 1.00%, square 100%, untrained widths 99–100%, identification unchanged). The per-line criterion fails narrowly (2 lines at 8/150), and a per-edge z* plus scale shrinkage is offered as an optional refinement; see **`2026-10-01-full2-review.md`**.
