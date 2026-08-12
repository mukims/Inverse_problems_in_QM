# An RL scheme for AGNR inverse inference

Design for casting "identify (width, concentration) from a transmission signature"
as a reinforcement-learning problem, rather than the fixed pipeline in
`agnr_agent.py`. Written to be implementable against the tools already in this
directory (`agnr_lib`, `width_id`, `train_conc_models`).

---

## 1. Why RL at all?

The current agent runs a **fixed** policy: extract gap → rank widths → validate →
call model → report. That is fine when the signature is clean. It is wasteful or
wrong when it is not:

- If the gap is ambiguous (heavy disorder), the right move is to **gather more
  evidence** (e.g. re-measure the gap with a different threshold, or run a
  forward simulation to test a hypothesis) — not to push on to the concentration
  model and emit a confident number.
- Forward simulation is **expensive** (~1.5 s/spectrum for 7-AGNR, far more for
  wide ribbons). Deciding *when* it is worth paying that cost is exactly a
  sequential decision problem under a budget.
- The pipeline currently cannot **backtrack**: if the concentration model reports
  an out-of-distribution input, that is evidence the width was wrong, but nothing
  feeds it back.

RL learns *when to stop, when to verify, and when to revise* — the parts a fixed
pipeline hard-codes.

---

## 2. MDP formulation

### State `s_t`
A fixed-length vector, all components produced by deterministic tools:

| block | contents | dim |
|---|---|---|
| signature summary | downsampled log(1+T) spectrum (e.g. 30 bins) | 30 |
| gap evidence | observed gap, gap at 3 thresholds (robustness probe), gap index | 5 |
| candidate posterior | current belief `b_t(m)` over widths in the library | \|M\| |
| ranking features | top-1 misfit score, margin to 2nd, #viable, #rejected | 4 |
| disorder | suppression = 1 − transmitted/pristine fraction | 1 |
| model evidence | per-model concentration predictions + spread (0 if not yet called) | 4 |
| budget | remaining simulation budget, steps taken | 2 |

Belief `b_t(m)` starts as a softmax over `−score(m)` from `width_id.misfit_widths`
and is updated by verification actions (below).

### Actions `a_t`
Discrete, each mapping to a real tool call:

| action | cost | effect |
|---|---|---|
| `PROBE_GAP(θ)` | ~0 | recompute band gap at threshold θ; reveals whether the gap is stable or an artifact |
| `REJECT_IMPOSSIBLE` | ~0 | apply the physical-possibility filter; zeroes `b_t` on widths whose pristine gap exceeds the observed onset |
| `SIMULATE(m, c)` | **high** | run the forward model at a hypothesis and compare to the signature; sharpens `b_t` by a likelihood update |
| `CALL_MODEL(k, m)` | low | query concentration model `k` ∈ {xgb, mlp, transformer} for width `m` |
| `COMMIT(m, c)` | ends episode | emit the final answer |
| `ABSTAIN` | ends episode | declare the signature un-identifiable |

`SIMULATE` is the interesting one: it is the only action that can *create* new
physical evidence, and it is the one with real cost.

### Transition
Deterministic given the tool outputs (the environment is the physics code). The
only stochasticity is which signature starts the episode.

### Reward
Terminal, shaped to encode what "a good answer" means here:

```
COMMIT(m, c):
    R = R_width + R_conc − λ_cost · (accumulated cost)
        R_width = +10 if m correct, −10 if wrong
        R_conc  = +5 · exp(−|c − c*| / σ)          σ ≈ 2 impurities
ABSTAIN:
    R = −1 − λ_cost · (accumulated cost)
```

Key properties:
- **Abstaining beats a confident wrong answer** (−1 vs −10). This is the whole
  point: for heavily disordered signatures the honest output is "cannot tell".
- Width error is punished harder than concentration error, because a wrong width
  invalidates the concentration entirely (the model is width-specific).
- The cost term forces the policy to learn that `SIMULATE` is only worth it when
  the belief is genuinely ambiguous.

Optional shaping: `+β · (H(b_t) − H(b_{t+1}))` rewards information gain per step,
which speeds up early learning considerably.

---

## 3. Training

**Environment**: free supervised data — every training spectrum in
`transmission_results_combined/` has known `(width, concentration)`, so ground
truth for the reward is available at zero labelling cost. Episodes are sampled
with a curriculum: low concentrations (clean gaps) first, then high
concentrations and near-metallic widths (3p+2 family) where the gap signal is
weakest.

**Algorithm**: PPO. The action space is small and discrete, the episodes are
short (≤10 steps), and PPO tolerates the non-stationary reward shaping better
than DQN here. An actor-critic MLP over the state vector above suffices —
this does not need a large network.

**Cheap surrogate for `SIMULATE`**: during training, replace forward simulation
with a lookup into the precomputed configuration-averaged references (already
built by the CA step). This makes episodes ~1000× cheaper; at deployment the
real simulator is used. The cost term in the reward keeps the *learned* policy
calibrated to the real cost, not the surrogate's.

**Baselines to beat**:
1. the current fixed pipeline (always: rank → validate → call model → commit);
2. "always COMMIT top-1 misfit" (no verification);
3. an oracle that knows when it is wrong (upper bound on the abstention gain).

---

## 4. What success looks like

The policy is worth having only if it does something the fixed pipeline cannot:

- **Selective abstention**: on the hardest slice (high concentration, 3p+2
  widths), accuracy-on-answered should rise materially while abstaining on a
  minority of cases.
- **Budget efficiency**: match the fixed pipeline's accuracy using fewer
  `SIMULATE` calls on easy signatures, and spend more on hard ones — i.e. learn
  a non-uniform allocation the fixed pipeline cannot express.
- **Backtracking**: measurably recover cases where the top-1 width is wrong but a
  simulation check flips it to the runner-up.

If none of these appear, the honest conclusion is that the fixed pipeline is
sufficient for this problem and RL is unnecessary complexity — the gap signal is
strong enough that width ID is nearly solved (currently 58/60), and the value of
RL is concentrated in the ambiguous tail.

---

## 5. Risks

- **Reward hacking via ABSTAIN**: if `λ_cost` is too high or `R_conc` too low, the
  policy abstains everywhere. Monitor the abstention rate; require coverage above
  a floor.
- **Surrogate/real mismatch**: a policy trained purely on CA-reference lookups may
  over-trust `SIMULATE`. Fine-tune the last phase against the real simulator.
- **Distribution shift in width**: models exist only for width 7 today. Until
  concentration models are trained per width, `CALL_MODEL` for other widths is
  out-of-domain and the reward will be misleading — train the width-7 case first
  (this is the current state of the code).
