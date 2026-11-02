# Inverse design, round 1: `model_v1` report

Date: 2026-10-28. Branch `feat/inverse-design`. Logbook entries: INV-SMOKE, INV-FULL, INV-TRAIN, INV-CAL, INV-STEP1.
Design: `docs/superpowers/specs/2026-10-23-inverse-design-design.md`.

## 1. The idea

**Goal.** Shazam today returns the closest *catalogued* device to a target transmission spectrum. If a material isn't
in the catalogue, it gets the nearest wrong answer. Inverse design replaces "nearest catalogued device" with "most
probable tight-binding host": given one spectrum $T(E)$, return the hosts that could have produced it and how many
impurities each would need. Every answer comes with a probability, and every realistic answer is checked by simulating
it again.

**The pieces.**

1. **A continuous space of hosts.** A hexagonal parent lattice with three site types:
   - C: hexagon centres;
   - H: honeycomb A/B;
   - K: bond midpoints.

   It has an energy scale $s$, on-site energies ($\varepsilon_C$, $\bar\varepsilon_H \pm \Delta$, $\varepsilon_K$) and
   hoppings within and between types. Graphene, silicene, germanene, hBN, the triangular lattice and kagome are single
   points in this space.
2. **Training data.** 300,000 simulated disordered ribbons, drawn across the space. Each draw sets:
   - the host;
   - the edge (armchair/zigzag);
   - the width $w$ (4–22);
   - the impurity density $c$ (0.25–5%).
3. **A conditional generative model**, $p(\text{host}, \text{edge}, w, c \mid \text{spectrum})$:
   - a 1D-conv encoder;
   - a classifier over the 156 discrete states (edge × site types × lead × switches);
   - a flow-matching field over the 14 continuous values.
4. **Candidates.** From 1,000 samples, at most 5 clusters, each with a probability and a representative host (the
   medoid).
5. **The simulator check.** Simulate 50 fresh disordered ribbons of each candidate. The candidate *recreates* the target
   when the target's misfit to the ensemble median is within the ensemble's own seed-to-seed spread (95th percentile).
   If it fails, refine the host's continuous values by Bayesian optimisation and check again.

**Step 1 / step 2 split** (the direction after the calibration gate, following the STAGE3-1 / BUILD-13 pipeline):

- **Step 1** identifies the host: structure and host values, plus width and edge.
- **Step 2**, a separate model, estimates the impurity concentration given step 1's host, e.g. from the spectrum
  divided by the predicted pristine spectrum.

Step 1 succeeds when the true host is among the top 5 and a candidate passes the recreation check, with concentration
profiled out (fixed at its true value).

## 2. What went right

**Data generation.**
- All 300,000 draws were generated in four 75k steps (about 104 h on 22 workers), exactly as the smoke run projected.
- Only 2.0% of attempts were rejected. Pattern and edge balance matched the prior.
- Every validity check passed: finite, no duplicates, nothing above the clean channel count.
- The one numerical bug found was a lead Green's-function overflow at single energies. Its root cause was found and
  fixed; the fix gives bit-identical results wherever the old code was finite. The two affected draws were regenerated
  and patched.

**Training.**
- Training is cheap: 12.6 min on CPU, 34 epochs, 512,810 weights.

**The discrete part, at the level of site types.**

| Measure (2,000 test draws) | Result | Target |
|---|---|---|
| True edge + site types in the top 5 | **0.972** | ≥ 0.95 |
| True edge + site types as the top candidate | 0.772 | |
| Candidate-probability calibration (ECE) | **0.023** | ≤ 0.05 |

The candidate probabilities are honest: the reliability curve follows the diagonal.

**When the structure is right, the parameter cloud is right.**
- Of the 1,236 draws with the full true structure in the top 5, the true host values lie inside that candidate's 99%
  region in **99.4%** of draws.
- With width included, the figure is **99.1%**.

**Most structural "misses" are near-degenerate hosts, not wild guesses.**
- The wrong-structure answer's clean spectrum is closer to the truth than a random host of the same type in **93%** of
  misses: distance 0.40 against 0.85.
- Within one site-type pattern, misses are as close as right answers (H: 0.166 against 0.167; HK: 0.41 against 0.39).
- Wrong switches sit near the boundary they cross:
  - $\Delta = 0$ misses have a small true $\Delta$ (median 0.18);
  - $t_{AA}$ misses have a small true $t_{AA}$ (median 0.073, against a range of 0.03–0.3);
  - $\varepsilon = 0$ misses are spectrally closer to the truth than typical right answers (0.13 against 0.18).

**The recreation check itself works.**
- The true host recreates its own spectrum in **96.7%** of draws (58 of 60).
- By construction this should be about 94–95%.

So the yardstick is sound, and a failure under it is a real failure.

## 3. What went wrong

### 3.1 The calibration gate failed (INV-CAL)

| Check | Result | Verdict |
|---|---|---|
| SBC, unconditional | log_c p = 3e-24, w p = 2e-12 | FAIL |
| SBC, state fixed | log_c 7e-31, w 1.5e-7, t_HK 1.3e-4, t_CH 7.6e-3 | FAIL |
| 90% coverage | delta 0.960, t_AA 0.948, t_AB 0.930 (too wide) | FAIL |
| Top-5 recall | 0.972 | ok |
| ECE | 0.023 | ok |

- The model **underestimates density and width** (mean rank of the truth 0.575 and 0.548; 0.5 means calibrated).
  The two errors are coupled: their correlation is +0.46.
- The likely cause is in training. Early stopping on the summed loss ended the run at epoch 23. The classifier was
  already overfitting from epoch 19, but the flow part was still improving at epoch 31. One stopping rule served two
  heads that wanted different stopping times.

### 3.2 The full structure is often wrong

"Site types right" (97%) is not "host right". Requiring the full discrete state (lead, $\Delta = 0$, $\varepsilon = 0$,
$t_{AA}$ on, cross bonds) drops the top-5 rate to **0.618**.

By site types, full structure plus host plus width in the top 5:

| Site types | C | K | CK | H | HK | CH | CHK |
|---|---|---|---|---|---|---|---|
| Found | 0.91 | 0.94 | 0.79 | 0.72 | 0.43 | 0.39 | **0.03** |

Hosts mixing honeycomb with other types are the weak spot. All-three-type hosts are almost never resolved. Among
misses, the wrong field is:

| Wrong field | Share of misses |
|---|---|
| lead | 44% |
| $\Delta = 0$ | 40% |
| $t_{AA}$ on | 36% |
| cross bonds | 25% |

These shares overlap, because several fields can be wrong at once.

### 3.3 Width is right only 41% of the time

- With the right structure, the medoid's width is exact in **41%** of draws, within ±1 in 78%, and within ±2 in 90%.
  It leans narrow by 0.27 cells.
- Width sets the number of open channels, so ±1 is not "close" for a transport spectrum. Section 3.4 shows that a
  wrong width roughly quadruples the misfit.

### 3.4 No candidate recreates the target (step 1 baseline, 60 draws)

The decisive test: is a top-5 candidate's **medoid** (with concentration fixed at the true value) within the
seed-to-seed noise of the target? The run was stopped early at 60 of a planned 150 draws.

| | Result |
|---|---|
| True host recreated (ceiling) | 0.967 |
| A top-5 candidate recreated | **0.000** (top 1: 0.000) |

How far the candidates miss (misfit / threshold; ≤ 1 passes, and the true host's median is 0.65):

| Candidates checked | n | Median | Closest |
|---|---|---|---|
| Right edge + site types | 60 | 10.9 | 1.52 |
| + full structure | 28 | 7.4 | 1.52 |
| + full structure, exact width | 13 | 2.9 | 1.52 |
| + full structure, wrong width | 15 | 12.6 | 2.68 |

Best candidate per draw, by site types (median): C 3.5, K 3.9, H 8.0, HK 13, CH 17, CHK 33, CK 36.

**Why this happens.** Two yardsticks that look similar are very different:

- *Inside the 99% region* asks whether the truth lies within the candidate's parameter cloud. The answer is yes in
  99% of draws once the structure is right.
- *Recreated* asks whether one representative point reproduces the spectrum within disorder noise. The seed-to-seed
  spread of a 50-ribbon ensemble is far smaller than the gap between a medoid and the truth: clean-spectrum distance
  around 0.16 still fails.

A medoid is a typical member of a broad cloud, not a fit to this spectrum.

**What the baseline did not include.** The design plans for this. Section 8, step 3 refines a failing candidate by
trust-region Bayesian optimisation (200 simulations), and the refinement exists (`src/shazam/inverse/refine.py`). The
baseline measured only the unrefined medoid. So the 0% is "the model's answer as it comes out", not the pipeline's
final answer. However, refinement keeps the discrete state and the width fixed, so it can't rescue the 59% of
right-structure candidates with the wrong width, or any candidate with the wrong structure.

### 3.5 Smaller things that went wrong along the way

- **Training launch.** A relaunch while the first run was still alive started two training runs at once. Both were
  killed and the run restarted once. Lesson: check the real python process, not `$!` from `setsid`.
- **Killing the plot viewer.** A `pkill -f` matched its own shell and also closed the live plot viewer without asking.
- **Wrong wall-time estimate.** The baseline wall time was first estimated at 0.3 min per draw; it was really
  1.4 min. The first estimate came from a two-draw smoke test on narrow ribbons.
- **Weak proxy.** The clean-spectrum distance is only a weak stand-in for the recreation check (Spearman +0.42). It
  can't replace simulation for judging "found" on all 2,000 draws.

## 4. Where this leaves step 1

| Level | Top 5 |
|---|---|
| Edge + site types | 0.972 |
| + full structure | 0.618 |
| + host values + width inside the 99% region | 0.612 |
| Medoid recreates the spectrum (60 draws) | **0.000** |

What a step 1 redesign has to address, in order of how much it costs today:

1. **Precision, not only coverage.** The answer must be a host that recreates the spectrum, so step 1 needs a fitting
   stage after the model (as in the misfit-curve analysis: minimise the misfit over the host, the way concentration is
   minimised today). Refinement as built is the starting point, but it must be measured.
2. **Exact width.** Width needs to be exact, or searched over (e.g. the medoid's width ± 2) inside the fitting stage.
3. **Near-degenerate switches** ($\Delta \approx 0$, small $t_{AA}$, near-tied lead). Either merge them in the
   candidate (treat "nearly off" as one host) or let the fitting stage cross them.
4. **The continuous calibration** (log_c, w bias; wide $\Delta$/$t_{AA}$ intervals). Dropping log_c from step 1 (it
   belongs to step 2) removes one of the two biased slots and the width–density coupling.

Per the plan, nothing was tuned on the test split. The next step is a design decision.

## 5. Files

`results/inverse_design/model_v1/`:

| File | Contents |
|---|---|
| `history.json` | training curves |
| `calibration.json` | gate verdict and numbers |
| `calibration.jsonl` | per-draw gate records, 2,000 draws |
| `model_report.png` | training curves, coverage, SBC p-values, reliability |
| `host_capture/host_capture.{py,jsonl}`, `host_capture_summary.py` | top-5 rate by level, 2,000 draws |
| `host_capture/near_identical.{py,jsonl}`, `near_identical_summary.py` | how close the wrong structures are |
| `host_capture/recreation_baseline.{py,jsonl}`, `recreation_baseline_summary.{py,txt}` | step 1 baseline, 60 draws |
| `model.pt` | the model (git-ignored, sha256 starts 8a4ba393d2aa) |

The scripts in `host_capture/` were run from the session scratchpad. They use absolute paths to the repo and the data
folder, and each has a summary script that reproduces the numbers above from its `.jsonl`.
