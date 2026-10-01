# FULL-1 wording: report the effective speed-up, not only the concurrency (2026-10-01)

For the implementing agent. This concerns `2b3fb0e79` (LOGBOOK only). The per-model timings are correct; they match the reviewer's file-timestamp checks.

"Measured parallel speedup … 11.1× to 11.6×" is the ratio of the median **loaded** worker time per spectrum to the wall time per spectrum. At armchair N50 that is 51.92 / 4.57 = 11.4. It measures how many workers run concurrently, not how much faster the run is than one core: under load each spectrum takes ~1.8× its single-core time (N50: 51.9 s against 29.1 s single-core in SMOKE-2). The effective speed-up over a single core is 29.1 / 4.57 ≈ **6.4×**, which is the figure behind the ~24 h run time.

In FULL-1, report both:
- the concurrency, ~11.4×;
- the effective speed-up over a single core, ~6.4×, with the per-spectrum slowdown under load of ~1.8× (8 P-cores with hyper-threading + 8 E-cores).
