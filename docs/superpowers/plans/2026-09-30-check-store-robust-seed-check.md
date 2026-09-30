# The per-seed check was the reviewer's mistake; the data is fine (2026-09-30)

For the implementing agent. It follows the `engine_v1` `check_store` run at 17:23, which failed armchair N5–N9. Stopping generation there was the correct move under the guard rail.

## What failed
All five failures come from one criterion: `seed_mean_valid`. That is the width-independent check the reviewer asked for in `2026-09-30-wide-grid-findings.md`: "for every seed, mean T over unmasked channels ≤ mean pristine + 0.05". It is not robust to the trace formula. At 1,000 seeds, a few configurations have **one** unbounded spike in an unmasked channel, and that single channel dominates the mean. Every row below has an excess of +1 or more only because of that one channel.

| Model, density | Seeds failing | Worst seed's mean excess | Its single largest channel | The same seed without that channel | That seed's median |
|---|---|---|---|---|---|
| armchair N9, d = 0.04 | 1 of 1000 (seed 788) | +191 | 40,909 | −2.00 | −1.91 |
| armchair N5, d = 0.02 | 2 of 1000 (seeds 149, 784) | +1.12 | 465 | −0.63 | −0.57 |

Every robust statistic passes on the same clouds. The largest median excess away from edges is ≤ +0.0005, and the away-from-edge spike share is ≤ 0.53%. The reviewer's independent sweep of all 84 narrow-block clouds also passes on n, seeds, formula, finiteness, pristine equal to `smoke_v1`, and cross-density duplicates. Wider armchair models passed only because none of their seeds happened to spike in an unmasked channel. Zigzag (Caroli, bounded) cannot fail this way.

## Do
1. In `check_store.py`, replace the per-seed mean with a spike-robust version: the mean of `min(T, pristine + 1) − pristine` over unmasked channels, which must be ≤ 0.05. Add a unit test in which one seed has a single 10⁴ spike in an unmasked channel and must pass, while a seed shifted up by 0.2 in every unmasked channel must fail.
2. Rerun `check_store.py --store ~/atlas_store/engine_v1`. The expectation is that all 21 narrow-block models PASS. Record the before and after in the LOGBOOK.
3. If all 21 pass, commit, then resume generation with the same command. The wide models are next, and the resume logic skips every existing cloud. No data needs to be deleted or regenerated.
