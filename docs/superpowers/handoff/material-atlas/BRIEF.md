# Material Atlas — Implementation Brief

You are implementing a reusable **atlas**: a label-free map that places a transmission spectrum T(E) of a disordered nanoribbon next to the closest known material (type → edge → continuous width), flags spectra that match nothing known, and hands off to material-specific concentration models. Build it test-first from the plan, one phase at a time, stopping at every **gate** and every **decision point**.

## Sources of truth

| Read | When |
|---|---|
| `docs/superpowers/specs/2026-09-29-material-atlas-design.md` | Before anything: every settled decision and the open decisions D1–D5 |
| `docs/superpowers/plans/2026-09-29-material-atlas-expansion.md` | The tasks, with interfaces, tests and code; work through it in order |
| `LOGBOOK.md` | History, benchmark numbers, Bugs #1–#8 |
| `DATA.md` (this folder) | Before touching any dataset, lead file or store path |
| `PHYSICS.md` (this folder) | Writing or debugging Hamiltonians, leads, transmission, disorder, or any physics test |
| `TRAPS.md` (this folder) | Before launching any long run, and whenever a result looks too good, too bad or too slow |
| `MATERIALS.md` (this folder) | Phase 3b only: published tight-binding parameters for real materials |

The spec and plan win over this brief if they disagree; tell the human about any disagreement you find.

## State at hand-off (2026-09-29)

- Working branch: `feat/square-autoencoder` (pushed to `origin`). Create your branch from it, e.g. `feat/material-atlas`.
- Already in the repo: the experimental atlas `notebooks/material_atlas/material_atlas.py` (BUILD-12, label-free input, 0–3 eV); the leak-free benchmark (BUILD-09); the energy-window finding (BUILD-11); the universal transformer (BUILD-10).
- Possibly still running when you start (check with `pgrep -fa "python -u"`): the BUILD-12 atlas run (log `notebooks/material_atlas/atlas_run.log`) and a 300-channel universal transformer (log `notebooks/universal_transformer/full_data_300ch_run.log`). Leave them running; when they finish, record their results in the LOGBOOK (BUILD-12 row and a new row for the transformer).
- A human-facing report exists as a Claude Doc ("Reading Disorder from Transmission Spectra"). Update it only if you have the docs connector; otherwise list the report changes for the human in your phase summary.

## Steps

### Step 0 — Orient
1. Read the spec and plan in full, then `DATA.md` and `TRAPS.md`.
2. Run the preflight in `DATA.md`; every path it checks must exist.
3. Run `~/miniconda3/envs/ml/bin/python -m pytest -q` to record the starting state.

Done when: preflight is all-OK, and you can name the three reference success criteria and what D1–D5 decide.

### Step 1 — Phase 1: atlas core and the 7/9-AGNR reference (plan Tasks 1–9)
Work each task as a red → green → commit cycle, exactly as the plan's steps lay out. Then run the reference evaluation (Task 8): smoke run first, then the full run in the background.

**Gate 1** — all of:
- full test suite green;
- `notebooks/material_atlas/reference_7_9/metrics.json` exists and `criteria_met` is reported for width accuracy ≥ 99.5%, end-to-end MAE ≤ 1.98, and 90% interval coverage within 90 ± 2%;
- LOGBOOK row BUILD-13 written.

If any criterion is missed, stop and report the gap to the human: Phase 5 reuses this map.

### Step 2 — Phase 2: the tight-binding engine (Tasks 10–11)
**Gate 2**: the engine's clean T equals the analytic open-channel count for honeycomb (armchair, zigzag) and square ribbons; `legacy_trace` reproduces `ca_sq.device` to 1e-8; the engine's idealised 7-AGNR clean spectrum matches `7_agnr_pris.npy` away from subband edges.

### Step 3 — Phase 3a: clean fingerprints of idealised ribbons (Task 12)
**Gate 3a**: clean spectra for every listed model are in the store; `fingerprints_ideal.png` shows the 3p+2 armchair widths (5, 8, 11, 14) conducting near E = 0 and the other armchair widths gapped.

### Decision point D1 — before Task 13
Ask the human which transmission formula new clouds use (`caroli` recommended, or `legacy_trace`); set `DEFAULT_FORMULA` accordingly.

### Step 4 — Phase 4: disorder clouds (Task 13)
Launch generation in the background (thread rules below).

**Gate 4**: every (model, density) of the width expansion is in the store, and `CloudStore.spike_fraction` is ≈ 0 for `caroli` clouds, or matches the legacy level for `legacy_trace`.

### Step 5 — Phase 5: atlas v2 and the generalisation test (Task 14)
**Gate 5**: `atlas_v2/generalisation.json` written; the held-out widths (armchair 8, 12, 13; zigzag 8) meet edge accuracy ≥ 99% and ≥ 90% of estimates within ±1.5 of the true width. Record the result either way; report misses to the human.

### Decision points D3, D4, D5 — before Phase 3b
Ask the human to choose the parameter sets and the energy convention for real materials, using `MATERIALS.md`; then write the Phase 3b plan (new builders, each validated with the channel-count test before its fingerprints). **D2** (energy unit for measured spectra) comes before Phase 6.

## Working rules

- **Tests are the contract.** The plan's code is a strong starting point, not gospel. When plan code fails its own test, fix the code. Change a test only when it encodes wrong physics, prove it with an analytic value, and say so in the commit message and in the plan file.
- **Seed split, label-free, round first** (see `AGENTS.md`) hold for every script you write.
- **One cycle per task:** failing test → implementation → green suite → one commit with a conventional prefix (`feat`, `fix`, `results`, `docs`). Push the working branch at each gate.
- **Compute:** heavy jobs run in the background with logs (`python -u … > file.log 2>&1`). PyTorch uses ≤ 4 threads while another training runs, 16 when alone; generators set `OMP_NUM_THREADS=1` per worker and use ≤ 20 workers. Stop a job by the PID of its `python` process. The why is in `TRAPS.md`.
- **Runs from the repo root** with `PYTHONPATH=notebooks/material_atlas:notebooks`.
- **Every number you report** traces to a file you produced (metrics JSON, log, store); quote it from there.

## Reporting at each gate
1. LOGBOOK: a build row (date, systems, data, split, results, status) and, for any new data or evaluation problem, a Bug entry (symptom, root cause, resolution).
2. README "Current Status" table when a headline number changes.
3. A short summary to the human: what passed, what missed and by how much, what decision is needed next.

## Done
All of Gates 1–5 passed or explicitly accepted by the human; D1 settled; the Phase 3b plan written after D3–D5; every result in the LOGBOOK; working branch pushed.
