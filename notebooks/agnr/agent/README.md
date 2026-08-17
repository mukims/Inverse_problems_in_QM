# agent/ — Autonomous inference agent

Given an **unknown** transmission spectrum, work out what device produced it —
without being told the width in advance.

The design principle is a strict split: **deterministic physics decides what is
possible, and only then does a model estimate a number.** The LLM (or a
deterministic fallback) is used for control flow and validation, never for physics.

---

## Files

| File | What it does |
|---|---|
| `width_id.py` | **Step 1.** Identifies ribbon width by comparing the spectrum against the pristine reference library: extracts the band gap, ranks candidate widths by misfit, validates the gap, and notes degeneracies (widths that are genuinely hard to tell apart). |
| `agnr_agent.py` | **Steps 1–4.** The full agent loop: width identification → validation/decision → dispatch a width-specific concentration model → feasibility assessment. Reasoning via Gemma through Ollama, with a deterministic fallback when no LLM is available. |
| `RL_SCHEME.md` | Specification for casting the whole diagnosis as an MDP under a compute budget. **Design document — not implemented in code yet.** |

---

## The pipeline

```
   unknown T(E)
        │
        ▼
  1. WIDTH        tool_band_gap, tool_rank_widths
                  filter physically impossible widths
        │
        ▼
  2. VALIDATE     LLM reasoning, or deterministic fallback
        │
        ▼
  3. CONCENTRATION  dispatch the width-specific model (MLP / transformer / XGBoost)
        │
        ▼
  4. FEASIBILITY  disorder-suppression check, model agreement, confidence score
                  → may ABSTAIN under extreme disorder
```

Step 4 matters: under heavy disorder the spectrum stops being informative, and the
agent is designed to say so rather than emit a confident wrong number.

---

## The RL framing (`RL_SCHEME.md`)

Treats diagnosis as sequential decision-making under an execution budget:

- **State** — signature summary, band-gap probes, candidate-width posterior, remaining budget
- **Actions** — `PROBE_GAP(θ)`, `REJECT_IMPOSSIBLE`, `SIMULATE(m,c)` (expensive RGF call),
  `CALL_MODEL(k,m)`, `COMMIT(m,c)`, `ABSTAIN`
- **Reward** — heavily penalises wrong width and wasted simulations; rewards accurate
  concentration and *honest abstention*

---

## Typical use

```bash
cd agent && conda run -n ml python agnr_agent.py <spectrum>.npy
```
```bash
cd agent && conda run -n ml python agnr_agent.py <spectrum>.npy --true-width 7 --true-conc 21
```

## Gotchas

- Both scripts import `agnr_lib` from [`../physics/`](../physics/) through the
  `_bootstrap` shim at the top of the file. Keep those lines if you edit the imports.
- `agnr_agent.py` looks for concentration models named `mlp_model.pt` /
  `transformer_model.pt`. The current checkpoints live in
  [`../concentration/`](../concentration/) and [`../multi_width/`](../multi_width/)
  under different names — **check the paths before expecting step 3 to work.**
- The width library comes from `physics/build_pristine_library.py`; regenerate it there
  if you add widths.
- No Ollama running? The agent falls back to deterministic logic rather than failing.
