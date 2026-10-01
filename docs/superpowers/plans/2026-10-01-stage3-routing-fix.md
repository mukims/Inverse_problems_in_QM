# STAGE3-1 review: route by the atlas's width vote (2026-10-01, 21:55)

For the implementing agent. This reviews commit `d1f226465`. It is a small correction inside Part A's scope; do nothing else.

## Verified
The reviewer recomputed every reported number from `predictions_test.npz`, and they match:

| Quantity | Value |
|---|---|
| MAE (unflagged) | 1.9814 |
| MAE (all) | 1.9812 |
| 90% interval coverage | 90.02% |
| q | 0.0896 |
| 7-AGNR MAE | 1.615 |
| 9-AGNR MAE | 2.230 |
| Unknown flag rate | 5.56% |

Also confirmed:
- The seed split is 0–2099 train, 2100–2549 calibration, 2550–2999 test, and q is fitted on unflagged calibration spectra only.
- `band_top_t=3.0` is label-free and exact, since every armchair band top is below 3 t.
- BUILD-13 is untouched, and no new data or stores were created.

## Fix: the routing is closed-world
`w_hat = 7 if |r.width − 7| < |r.width − 9| else 9` snaps the continuous width to the nearer of 7 and 9. That uses the knowledge that the answer is 7 or 9. The production atlas knows 31 widths; its decision is `width_vote`, and the directive says the **predicted width** selects the regressor.

- The vote is the better router: 99.87% correct, against 99.82% for the snapped width.
- 38 test spectra (0.10%) get a vote outside {7, 9}: 33 as N6, 3 as N16, 1 as N11, 1 as N15. Nine of them are unflagged.
- On spectra where the vote and the snapped width agree (35,241 of the unflagged), the MAE is **1.957**. Routing by the vote should therefore pass the ≤ 1.98 gate without a closed-world assumption.

Do:
1. Route by `width_vote`, in both calibration and test:
   - vote 7 → the 7-AGNR regressor and pristine;
   - vote 9 → the 9-AGNR regressor and pristine;
   - **any other vote → no estimate**, counted as `no_stage3_model`, the same way unknown-flagged spectra are counted.
2. Refit q on calibration spectra routed the same way. Report:
   - MAE and coverage on spectra that received an estimate;
   - the counts of flagged and `no_stage3_model` spectra;
   - the vote width accuracy.

   Keep the snapped-routing numbers in STAGE3-1 as "superseded".
3. Update `metrics.json`, `predictions_test.npz`, the LOGBOOK STAGE3-1 entry, and a test that a vote outside {7, 9} gets no estimate. Commit, push, and **stop**, as the Stage 3 directive says.

## For the human, later (no action now)
The unknown flag removes most spectra outside the atlas's 0.5–4% training densities. 9-AGNR at c = 2 (0.11%) is 97.6% flagged, and 7-AGNR at the highest concentrations (up to 4.9%) is up to 15% flagged. If Stage 3 must cover 0.1–5.5%, the atlas needs reference densities over that range. That needs new generation, which is the human's decision.
