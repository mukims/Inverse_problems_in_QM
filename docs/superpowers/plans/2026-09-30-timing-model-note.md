# Timing model: add the chunk tail (2026-09-30)

For the implementing agent. This covers commit `c6d686d7a` (84/84 tests pass). The trend sweep and the per-spectrum timing inside the workers are good.

**One claim is wrong.** The cost-table note says 50 × t_spec / 12 + 19.5 s matches the reviewer's file timestamps. It does for narrow ribbons, but not wide ones: at armchair N50 it gives 140.7 s against 307.0 s measured, and at zigzag N50 149.5 s against 319.9 s.

**The cause is the chunk tail.** `p.map(..., chunksize=4)` with 50 seeds makes 13 chunks for 12 workers, so the busiest worker runs 2 chunks, 8 spectra back to back. The busiest worker's serial work plus start-up reproduces the timestamps:

| Model | Measured | 50 t / 12 + 19.5 | **8 t + 19.5** |
|---|---|---|---|
| armchair N20 | 45.6 | 33.2 | **45.9** |
| armchair N27 | 74.4 | 45.7 | **69.7** |
| armchair N50 | 307.0 | 140.7 | **252.1** |
| zigzag N20 | 45.8 | 32.4 | **44.3** |
| zigzag N50 | 319.9 | 149.5 | **269.0** |

The remaining gap of up to ~20% on the widest ribbons is worker contention on this hybrid CPU (8 P-cores + 8 E-cores).

**Do:**
- Model wall time as ⌈⌈n / chunk⌉ / W⌉ × chunk × t_spec + overhead.
- At 1,000 seeds the tail is negligible (250 chunks on 16 workers means 64 serial spectra against an ideal 62.5), so the full-run projection is Σ 4,000 t_spec / W × ~1.2.
- For the chosen run (sparse grid, 1,000 seeds, the existing N5, N7 and N9 clouds reused) that is **about 10–12 h on 16 workers**. Record the run's measured `meta.json` timings against this estimate in FULL-1.
- Correct the "accurately matching" sentence in SMOKE-2.
