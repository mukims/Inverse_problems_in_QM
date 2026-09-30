# The edge-masked store guard is correct; add a test and a LOGBOOK line (2026-09-30)

For the implementing agent. This covers your uncommitted change to `CloudStore.write_cloud`: the median-vs-pristine guard now skips channels within about 0.05 t of each clean-spectrum step, meaning a jump of more than 0.5 between neighbouring channels.

## Verdict: keep it
The reviewer's earlier bound ("median ≤ pristine + 0.05 at every energy") was wrong. It was calibrated on every 10th channel, which skipped subband edges. Over all 300 channels, the historical trace-formula data shows the same edge excess that your smoke clouds do.

| Data | Largest (median − pristine), all channels | Largest (median − pristine), more than 5 channels from a step | Share above pristine + 0.5, within 5 channels / away |
|---|---|---|---|
| Real 9-AGNR (`size_9.npy`, 2,000 seeds), c = 98 (5.4%) | +1.62 at E = 1.62 t (1 channel from a step) | +0.0005 | 6.2% / 0.24% |
| Real 9-AGNR, c = 40 (2.2%) | +0.000 | +0.0000 | 3.6% / 0.02% |
| Smoke N9, d = 4% | — | — | 5.0% / 0.09% |
| Smoke N12, d = 4% (12 steps in the window vs 8 for N9) | +11.8 (0–4 channels from a step) | +0.0001 | 10.0% / 0.40% |
| Smoke N13, d = 4% | +4.2 (1–2 channels from a step) | +0.0004 | 6.5% / 0.11% |

Away from subband edges, every cloud, real or smoke, stays at or below its clean spectrum. At the edges, the trace formula is unbounded (`TRAPS.md`, "Formula spikes"). Wider ribbons have more subbands in the window, so they have more edge channels and a higher overall spike share.

## Do
1. Add a store test for the masked guard:
   - A cloud whose median exceeds pristine only within 3 channels of a pristine step is accepted.
   - The same excess placed more than 6 channels from any step raises `ValueError`.
   - Pristine with no steps keeps the unmasked behaviour.
2. In the LOGBOOK B4 entry (or SMOKE-1), note why the mask exists and its width (±4–5 channels ≈ 0.05 t), citing the real `size_9` numbers above.
3. In `report.json`, report the spike share and the median excess separately for near-edge and away-from-edge channels, so the full run can be compared with this table.

The band test stays the check for lattice errors (`test_even_width_agnr_bands_match_geometric_honeycomb`). A wrong lattice can pass a median guard, masked or not.
