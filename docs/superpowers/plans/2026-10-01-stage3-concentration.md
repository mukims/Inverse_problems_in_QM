# Stage 3: concentration models (2026-10-01, revised 21:10)

For the implementing agent.

**The human accepted FULL-3** and moved to Stage 3. The human is unwell and unavailable, and has asked for the work to **stay centred on the original plan and proceed with caution**. There is **no time pressure**; correctness comes first. The reviewer session monitors and leaves notes in this folder.

**Revision 21:10.** This replaces the earlier version of this file:
- the "tonight" time box is dropped;
- Part B is deferred;
- an explicit stop is added after Part A.

## Part 0: done
The FULL-1 column of the FULL-3 table was fixed in `7fadfc410`.

## Part A: Stage 3 as the original plan defines it (7- and 9-AGNR)
This is the plan's stage 3: "one XGBoost per width on the full 0–3 t spectrum normalised by the **predicted** width's pristine; relative conformal intervals calibrated on held-out seeds". It uses the legacy dense data the spec keeps for Stage 3: `size_7.npy` and `size_9.npy`, 34 and 49 concentrations, 10,000 seeds each, the same `agnr_lib_IL_1e-5` formula as the engine.

The only change from BUILD-13 is the front end, which becomes the production atlas.

- **Pipeline.**
  - Copy `reference_7_9/run_reference_7_9.py` into a new `notebooks/material_atlas/stage3_7_9/`. Leave BUILD-13 and its outputs untouched.
  - Use **`atlas_v2`** (InputSpec v2, class-conditional novelty) to locate each legacy spectrum label-free. The predicted width selects the regressor and its pristine.
  - Spectra flagged unknown get no estimate. Count them separately.
- **Back end, as BUILD-13:**
  - one XGBoost per width (n_estimators 800, max_depth 8);
  - input normalised by the predicted width's pristine, rounded to 3 decimals, over the full 0–2.99 t window.
- **Split by seed, as BUILD-13:** 0–2099 train, 2100–2549 conformal calibration, 2550–2999 test.
- **Gates (from the spec):**
  - label-free width accuracy ≥ 99.5%;
  - end-to-end MAE ≤ 1.98 impurities (BUILD-13 got 1.973);
  - 90% interval coverage within 88–92%.
- **Report:**
  - MAE per width and per concentration band;
  - the unknown-flag rate against density;
  - MAE on unflagged against all spectra.

  Legacy concentrations extend outside the atlas's 0.5–4% training range (0.14% for c = 2 in 7-AGNR, up to 5.4% for c = 98 in 9-AGNR). Say where the flag or the width accuracy degrades.
- **If a gate fails, report it as it comes out.** Do not tune on the test seeds (`2026-10-01-no-test-tuning.md` applies here too).

## Deliverables, then stop
- `stage3_7_9/metrics.json`, plus the saved test predictions so the reviewer can recompute.
- **LOGBOOK STAGE3-1.**
- Tests:
  - toy interval coverage;
  - unknown-flagged spectra get no estimate.
- Commit and push.

**Then stop and wait for the human.** Do not start any of the following without the human:
- Phase 3b (real materials, which needs decisions D3–D5);
- Phase 6 (the interactive page, which needs D2);
- any new data generation;
- the deferred Part B.

Keep the stores read-only: `engine_v1`, `smoke_v1`, `novelty_v1`, `reference_v1`.

## Deferred: Part B (coarse density for all 31 widths)
This was proposed in the first version of this file. It is outside the original plan, and it generates new data, so it waits for the human. The human has said they have their own approach for extending concentration estimation beyond 7/9-AGNR.
