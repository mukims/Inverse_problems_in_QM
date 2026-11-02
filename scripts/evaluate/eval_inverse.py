"""Fit the update, the hidden-material tests and the end-to-end run of the inverse design (spec sections 8.4 and 9).

    PYTHONPATH=src python scripts/evaluate/eval_inverse.py fit-update  --model results/inverse_design/model_v1/model.pt --n 500
    PYTHONPATH=src python scripts/evaluate/eval_inverse.py hidden      --material hbn --model results/inverse_design/hidden_hbn/model.pt
    PYTHONPATH=src python scripts/evaluate/eval_inverse.py end-to-end  --model results/inverse_design/model_v1/model.pt --n 200

fit-update  runs `design` (no update) on the first --n VALIDATION draws of the generated data, builds per draw the model
            probabilities p of the realistic candidates, their log-likelihoods, the background mass p0 (`background_mass`,
            ruling R25) and the capture hits (`captures`, ruling R18), and fits kappa and log_L0 (`fit_update`). Writes
            update.json next to the model.
hidden      gives a model trained with --exclude <material> the catalogued material's own spectra (test seeds 850 to 999, all
            four densities: 600 targets) and scores the four marks of spec section 9: (1) the top candidate has the right
            (edge, active-type pattern) for >= 90%; (2) its continuous parameters lie within 0.15 (units of s) of the truth
            and s within 10%; (3) its 90% impurity-count interval holds the true count for >= 85%; (4) its simulator check
            recreates the target for >= 80%. Writes hidden_<material>.json next to the model. A failing mark is a result to
            report, not a prompt to retune.
end-to-end  on the first --n TEST draws: the share whose true host is captured by one of the candidates, the share
            recreated (the top candidate's check, on fresh seeds), and the ECE of the model and of the updated
            probabilities (update.json next to the model). Writes end_to_end.json.

Cost (ruling R26). One simulation is a ribbon of the candidate host, on average as costly as a generated draw (manifest of
the data). Every realistic candidate is checked on 50 seeds and a failed check is refined (200 simulations + 50 for the
final check), so a draw costs a few hundred simulations. --dry-run prints the projected simulations and wall time and exits.
--limit N (first N targets), --check top (only the rank-1 candidate is simulated: enough for `hidden`, which scores only the
top candidate; not for the update fit) and --no-refine (a failed check stays failed) trade fidelity for time; the defaults
follow the spec. The flags used are recorded in every output JSON.

Every finished target is appended to <output>.jsonl, so a crash does not lose hours; a restart skips the targets already
there (records written with other settings, another model.pt (sha256) or another data size are refused: delete the file).
A target whose design raises is recorded in the .jsonl as an error and skipped; the final JSON lists it under
`failed_targets`, and a restart does not retry it unless --retry-errors is passed. The final JSON is written atomically
and records the code version of this run, the model's sha256 and the data's provenance.
`--n-cells` and `--seeds` shorten the devices and the checks and exist for the tests only.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import paths
from atlaslib.store import CloudStore
from shazam.inverse import candidates as candidates_mod
from shazam.inverse import data
from shazam.inverse.calibration import captures, expected_calibration_error
from shazam.inverse.data import PARAM_COLUMNS
from shazam.inverse.design import UPDATE_SETTINGS, background_mass, design, target_input
from shazam.inverse.design import update_mismatch as _update_mismatch
from shazam.inverse.model import InverseModel
from shazam.inverse.posterior import fit_update
from shazam.inverse.runio import RunLog, atomic_write_json, code_version, sha256_of
from shazam.inverse.verify import OPT_SEEDS, VERIFY_SEEDS
from tbribbon.parent import named_material
from tbribbon.parent_prior import Draw

DEFAULT_MODEL = paths.results_dir("inverse_design", "model_v1") / "model.pt"
DEFAULT_DATA = paths.data_root() / "_generated" / "hexagonal_parent_v1"
NEEDED = ("index", "x", "on", "lead", "params", "edge", "width", "density")

CATALOGUE = {"hbn": "hbn/armchair/N10", "silicene": "silicene/armchair/N10", "kagome": "kagome/zigzag/N10"}
TEST_SEEDS = range(850, 1000)
PARAM_NAMES = PARAM_COLUMNS[1:]                    # every continuous host parameter except s
PARAM_TOL = 0.15                                   # units of s (the canonical parameters are in units of s)
S_REL_TOL = 0.10
# Pass marks of spec section 9. The spec gives no share for mark 2 (the parameters): 0.90, like mark 1, is my call.
MARKS = {"key": 0.90, "params": 0.90, "count_interval": 0.85, "recreated": 0.80}
PARAMS_THRESHOLD_NOTE = "assumed: the spec states no share for the parameter mark; 0.90 as for the (edge, pattern) mark"
DRY_DEFAULTS = {"expected_candidates": 3.0, "p_refine": 0.5}


# --------------------------------------------------------------------------------------------------------- catalogue targets

def truth_of(material, density=0.01):
    """The true host of a catalogued material: its spec 4.3 point at width 10 and the catalogue ribbon's edge (hbn and
    silicene armchair, kagome zigzag). The density defaults to 0.01; the catalogue has four, so score each target against
    `truth_of(material, its_density)`. ValueError for a material that is not in the hidden-material tests."""
    if material not in CATALOGUE:
        raise ValueError(f"unknown material {material!r}; choose from {tuple(CATALOGUE)}")
    _, edge, width = CATALOGUE[material].split("/")
    return Draw(named_material(material), edge, int(width[1:]), float(density))


def catalogue_targets(model_id, seeds=TEST_SEEDS, store=None):
    """[(v4 input (416,) float32, true impurity count, density)] of the catalogue ribbon `model_id` (e.g. "hbn/armchair/N10"):
    every disorder spectrum whose seed is in `seeds` (by seed value), over the catalogue's densities (CloudStore, extra=False),
    densities ascending and seeds ascending within one.

    The catalogue energies are in units of the material's own hopping t; the host's s of spec 4.3 (hbn 2.30, silicene 1.067,
    kagome 1.0) is that hopping in eV, so E_eV = energies_t x s, which for these ribbons lands exactly on the v4 grid.
    `target_input` then builds the input; it equals the lookup's on_axis + window_inputs route (a test checks it)."""
    store = CloudStore(paths.data_root()) if store is None else store
    s = named_material(model_id.split("/")[0]).s
    # CloudStore has no public accessor for a density's impurity count; _shown is the extra=False-aware view of meta.json
    counts = {v["density"]: int(v["n_impurities"]) for v in store._shown(model_id).values()}
    e_ev = np.asarray(store.read_pristine(model_id)[0], float) * s
    seeds = np.asarray(list(seeds))
    out = []
    for d in sorted(counts):
        cloud, cseeds = store.read_cloud(model_id, d)
        rows = np.flatnonzero(np.isin(cseeds, seeds))
        for r in rows[np.argsort(cseeds[rows], kind="stable")]:
            out.append((target_input(e_ev, cloud[r]), counts[d], d))
    return out


# ----------------------------------------------------------------------------------------------------------------- helpers

def _clean(x):
    """JSON-ready: numpy scalars and arrays as Python, non-finite floats as null."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, np.generic):
        return _clean(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, _clean(obj), indent=1)


class _Recorder:
    """Passes `sample` through to the model and keeps the samples, so the candidates behind a report are the very ones `design`
    saw (the report holds no cluster means or covariances, which `captures` needs)."""

    def __init__(self, model):
        self.model, self.states, self.z = model, None, None

    def sample(self, x, n, *args, **kw):
        self.states, self.z = self.model.sample(x, n, *args, **kw)
        return self.states, self.z


def _seed(base, index):
    """A per-target seed that depends on (base seed, target index) only, so a target's record does not depend on the others."""
    return int(np.random.default_rng([int(base), int(index)]).integers(2 ** 31))


def design_record(model, x, truth, seed, a, update, provenance=None):
    """{"report", "hits"} of one target: the `design` report and, per returned candidate, whether it captures `truth`
    (None when the truth is unknown). `provenance` goes into the report's provenance (model, update file, ...)."""
    rec = _Recorder(model)
    report = design(x, rec, n_samples=a.n_samples, update=update, refine_budget=0 if a.no_refine else a.refine_budget,
                    n_jobs=a.n_jobs, seed=seed, n_cells=a.n_cells, seeds_override=a.seeds, check=a.check,
                    provenance=provenance)
    cands, _ = candidates_mod.candidates(np.asarray(rec.states), np.asarray(rec.z))
    rows = report["candidates"]
    if len(cands) != len(rows) or any(abs(c.probability - r["model_probability"]) > 1e-12 for c, r in zip(cands, rows)):
        raise RuntimeError("the candidates rebuilt from the recorded samples differ from the report's")
    hits = None if truth is None else [bool(captures(c, truth, cands)) for c in cands]
    return {"report": report, "hits": hits}


def run_targets(items, log, work, tag, retry_errors=False):
    """(records, seconds, new, failed) of the (key, payload) items: the records, in order, of the finished ones from the log
    and of the rest from `work(payload, position)`; `failed` maps key -> repr of the error of each target that raised.

    A target whose `work` raises is recorded in the log as an error and skipped (the run goes on); a target recorded as an
    error before is skipped too, unless retry_errors. Failed targets are not in the records."""
    records, t0, new, failed = [], time.time(), 0, {}
    for j, (key, payload) in enumerate(items):
        if key in log.done:
            records.append(log.done[key])
            continue
        if key in log.errors and not retry_errors:
            failed[key] = log.errors[key]
            continue
        t1 = time.time()
        try:
            rec = work(payload, j)
        except Exception as err:                       # recorded and skipped; KeyboardInterrupt still stops the run
            log.add_error(key, err)
            failed[key] = repr(err)
            print(f"[{tag}] {j + 1}/{len(items)} FAILED after {time.time() - t1:.0f} s: {err!r}", flush=True)
            continue
        log.add(key, rec)
        records.append(rec)
        new += 1
        print(f"[{tag}] {j + 1}/{len(items)} done in {time.time() - t1:.0f} s "
              f"({(time.time() - t0) / new:.0f} s per new target)", flush=True)
    return records, time.time() - t0, new, failed


def failed_summary(failed):
    """The final JSON's `failed_targets`: how many targets raised, their keys (in run order) and errors."""
    return dict(n=len(failed), keys=list(failed), errors=dict(failed))


def _provenance(a, model_sha256, records=()):
    """What every output JSON records: this run's code version, the model's sha256, the data's provenance (not for
    `hidden`, which reads no generated data) and the code versions behind the records (a resumed run may span several)."""
    out = dict(code_version=code_version(), model_sha256=model_sha256,
               record_code_versions=sorted({r["report"].get("provenance", {}).get("code_version", "unknown")
                                            for r in records}))
    if a.cmd != "hidden":
        out["data_provenance"] = data.data_provenance(a.data)
    return out


def _resume_cfg(a, cfg, model_sha256):
    """The run log's cfg: the settings, the model's sha256 and (for the generated-data commands) the data's draw count, so
    a retrained model or an extended data set is refused on resume instead of mixed in."""
    cfg = cfg | dict(model_sha256=model_sha256)
    if a.cmd != "hidden":
        cfg["data_draws"] = data.data_provenance(a.data)["n_draws"]
    return cfg


# ---------------------------------------------------------------------------------------------------------------- dry run

def seconds_per_simulation(root):
    """(seconds, source): the mean seconds per generated draw from the data manifest (the shards' summed `seconds` over their
    draws), else the mean of the shards' own `seconds` arrays. One check simulation costs about as much as one generated draw."""
    manifest = data.load_manifest(root)
    shards = (manifest or {}).get("shards") or []
    n = sum(s.get("n_draws", 0) for s in shards)
    if n and all("seconds" in s for s in shards):
        return sum(s["seconds"] for s in shards) / n, f"manifest of {root}"
    sec = []
    for path in data.shard_paths(root):
        with np.load(path) as z:
            if "seconds" in z.files:
                sec.append(np.asarray(z["seconds"], float))
    if sec:
        return float(np.concatenate(sec).mean()), f"`seconds` of the shards under {root}"
    raise SystemExit(f"no timing found under {root} (no manifest.json with shard seconds, no shard `seconds`): "
                     "pass --seconds-per-sim")


def project(n_targets, check, no_refine, expected_candidates, p_refine, refine_budget, n_seeds, n_opt, seconds, n_jobs):
    """Simulations and wall time of `n_targets` designs.

    Per candidate: a check of n_seeds simulations plus, with probability p_refine, a refinement (its whole budget, a conservative
    figure: budget // n_opt evaluations of n_opt simulations each) and the final check of n_seeds. Candidates: the expected
    number, or 1 for check=top. Each call to the simulator runs its seeds on n_jobs workers, so it takes
    ceil(seeds / n_jobs) rounds of one simulation time (50 seeds on 22 jobs = 3 rounds, 20 seeds = 1 round); wall time is the
    sum of those rounds, not simulations / n_jobs."""
    n_candidates = 1.0 if check == "top" else float(expected_candidates)
    rounds = lambda k: -(-k // n_jobs)
    evals = refine_budget // n_opt
    refine = 0.0 if no_refine else p_refine
    per_sims = n_seeds + refine * (evals * n_opt + n_seeds)
    per_rounds = rounds(n_seeds) + refine * (evals * rounds(n_opt) + rounds(n_seeds))
    sims = n_targets * n_candidates * per_sims
    n_rounds = n_targets * n_candidates * per_rounds
    return dict(n_targets=int(n_targets), check=check, no_refine=bool(no_refine), candidates_per_target=n_candidates,
                simulations_per_candidate=per_sims, rounds_per_candidate=per_rounds, simulations=sims, rounds=n_rounds,
                seconds_per_simulation=seconds, n_jobs=n_jobs, cpu_hours=sims * seconds / 3600,
                wall_hours=n_rounds * seconds / 3600)


def dry_run(a, tag, n_targets):
    seconds, source = (a.seconds_per_sim, "--seconds-per-sim") if a.seconds_per_sim else seconds_per_simulation(a.data)
    args = (a.expected_candidates, a.expected_refine_rate, a.refine_budget, a.seeds or len(VERIFY_SEEDS),
            a.seeds or len(OPT_SEEDS), seconds, a.n_jobs)
    proj = project(n_targets, a.check, a.no_refine, *args)
    proj["source"] = source
    print(f"[dry run] {tag}: {proj['n_targets']} targets x {proj['candidates_per_target']:g} candidates x "
          f"{proj['simulations_per_candidate']:g} simulations = {proj['simulations']:,.0f} simulations", flush=True)
    print(f"[dry run] {seconds:.1f} s per simulation ({source}); every simulator call runs its seeds in rounds of {a.n_jobs} "
          f"jobs ({proj['rounds_per_candidate']:g} rounds per candidate) = {proj['wall_hours']:.1f} wall hours "
          f"({proj['cpu_hours']:.0f} CPU hours); assumes {a.expected_candidates:g} candidates per target and "
          f"{a.expected_refine_rate:.0%} of checks refined (whole budget); a busy machine is slower. "
          "Nothing was run or written.", flush=True)
    alts = []
    for check, no_refine in (("all", False), ("all", True), ("top", False), ("top", True)):
        alt = project(n_targets, check, no_refine, *args)
        alts.append(alt)
        print(f"[dry run]   --check {check}{' --no-refine' if no_refine else ''}: {alt['simulations']:,.0f} simulations, "
              f"{alt['wall_hours']:.1f} wall hours", flush=True)
    proj["alternatives"] = alts
    return proj


# ------------------------------------------------------------------------------------------------------------------ hidden

def _scores(material, rec):
    """What one hidden-material target says: the right (edge, pattern), per-parameter errors of the top candidate, the interval,
    the check. A target without a (checked) top candidate misses every mark."""
    truth = truth_of(material, rec["density"])
    rows = rec["report"]["candidates"]
    top = rows[0] if rows else None
    out = dict(key=False, params=False, count_interval=False, recreated=False, abs_err={}, s_rel_err=None, width=False)
    if top is None:
        return out
    out["key"] = bool(top["edge"] == truth.edge and tuple(bool(v) for v in top["discrete_state"]["pattern"]) == truth.params.pattern)
    canon = top["parameters"]["canonical"]
    out["abs_err"] = {n: abs(float(canon[n]) - float(getattr(truth.params, n))) for n in PARAM_NAMES}
    out["s_rel_err"] = abs(float(canon["s"]) / truth.params.s - 1.0)
    out["params"] = bool(out["key"] and all(e <= PARAM_TOL for e in out["abs_err"].values()) and out["s_rel_err"] <= S_REL_TOL)
    lo, hi = top["impurity_count_90"]
    out["count_interval"] = bool(lo <= rec["true_count"] <= hi)
    out["recreated"] = bool(top["check"] is not None and top["check"]["recreated"])
    out["width"] = bool(top["width"] == truth.width)
    return out


def _share(flags):
    return float(np.mean(flags)) if len(flags) else None


def score_hidden(material, records):
    """The four marks of spec section 9 from the hidden-material records ({"density", "true_count", "report"} each).

    key             the top candidate has the true (edge, active-type pattern);
    params          ... AND every continuous parameter of the top candidate is within 0.15 of the truth (canonical units,
                    which are units of s; a parameter off in the candidate counts as 0) AND s within 10%;
    count_interval  the top candidate's 90% impurity-count interval holds the true count;
    recreated       the top candidate's simulator check recreates the target.
    Each mark is a share of ALL targets: no candidate, a wrong key, or an unchecked top candidate is a miss. Per-parameter errors
    are over the targets with the right key (the comparison means nothing otherwise)."""
    sc = [_scores(material, r) for r in records]
    n = len(sc)
    marks = {}
    for name, threshold in MARKS.items():
        value = _share([s[name] for s in sc])
        marks[name] = dict(value=value, threshold=threshold, **{"pass": bool(value is not None and value >= threshold)})
    marks["params"]["threshold_note"] = PARAMS_THRESHOLD_NOTE
    right = [s for s in sc if s["key"]]
    marks["params"]["among_right_key"] = _share([s["params"] for s in right])
    per = {}
    for name in PARAM_NAMES:
        err = np.array([s["abs_err"][name] for s in right])
        per[name] = dict(within_tolerance=_share(err <= PARAM_TOL), tolerance=PARAM_TOL, n=len(right),
                         mean_abs_error=float(err.mean()) if len(err) else None,
                         median_abs_error=float(np.median(err)) if len(err) else None)
    rel = np.array([s["s_rel_err"] for s in right])
    per["s"] = dict(within_tolerance=_share(rel <= S_REL_TOL), tolerance=S_REL_TOL, relative=True, n=len(right),
                    mean_abs_error=float(rel.mean()) if len(rel) else None,
                    median_abs_error=float(np.median(rel)) if len(rel) else None)
    densities = {}
    for d in sorted({r["density"] for r in records}):
        sel = [s for s, r in zip(sc, records) if r["density"] == d]
        densities[str(d)] = dict(n=len(sel), **{k: _share([s[k] for s in sel]) for k in MARKS})
    return dict(material=material, catalogue=CATALOGUE[material], n=n, marks=marks, per_parameter=per, per_density=densities,
                diagnostics=dict(width_exact=_share([s["width"] for s in sc])),
                note="a target with no candidate, a wrong key or an unchecked top candidate misses every mark it needs",
                **{"pass": bool(marks and all(m["pass"] for m in marks.values()))})


def trained_without(model_path, material):
    """The history.json next to the model, after checking that the model was trained with --exclude <material>
    (SystemExit otherwise: scoring a model that saw the material would be meaningless)."""
    path = Path(model_path).with_name("history.json")
    if not path.exists():
        raise SystemExit(f"{path} not found: `hidden` needs the history.json that train_inverse.py --exclude {material} "
                         "writes next to the model")
    history = json.loads(path.read_text())
    excluded = history.get("excluded") or {}
    if excluded.get("hide") != material:
        raise SystemExit(f"{model_path} was trained with --exclude {excluded.get('hide')}, not --exclude {material} "
                         f"({path}): train it with scripts/build/train_inverse.py --exclude {material}")
    return history


def cmd_hidden(a, cfg):
    model_id = CATALOGUE[a.material]
    targets = catalogue_targets(model_id)
    if a.limit:
        targets = targets[:a.limit]
    if a.dry_run:
        return dry_run(a, f"hidden {a.material}", len(targets))
    history = trained_without(a.model, a.material)
    out = Path(a.out) if a.out else Path(a.model).with_name(f"hidden_{a.material}.json")
    sha = sha256_of(a.model)
    log = RunLog(out.with_suffix(".jsonl"), _resume_cfg(a, cfg, sha) | dict(material=a.material))
    model = InverseModel.load(a.model)
    print(f"[hidden] {a.material}: {len(targets)} catalogue targets of {model_id}", flush=True)
    prov = dict(model=str(a.model), model_sha256=sha)

    def work(target, i):
        x, count, density = target
        rec = design_record(model, x, None, _seed(a.seed, i), a, None, prov)
        return dict(density=density, true_count=count, report=rec["report"])

    try:
        records, seconds, new, failed = run_targets([(str(i), t) for i, t in enumerate(targets)], log, work, "hidden",
                                                    a.retry_errors)
    finally:
        log.close()
    result = score_hidden(a.material, records) | dict(model=str(a.model), model_excluded=history["excluded"],
                                                      model_training={k: history.get(k) for k in ("code_version", "data",
                                                                                                  "data_provenance")},
                                                      flags=_flags(a),
                                                      seconds=seconds, resumed_targets=len(records) - new,
                                                      failed_targets=failed_summary(failed)) | _provenance(a, sha, records)
    _write_json(out, result)
    print(f"[hidden] {a.material} " + ", ".join(f"{k} {m['value']:.3f} (>= {m['threshold']}) {'ok' if m['pass'] else 'FAIL'}"
                                                 for k, m in result["marks"].items()) + f"; wrote {out}", flush=True)
    return result


# ------------------------------------------------------------------------------------------------- generated-data commands

def _load_draws(a, split):
    """(xs, draws, indices) of the first n draws of the split (then the first --limit of those)."""
    arrays = data.load_split(a.data, split, NEEDED)
    n_rows = len(arrays["index"])
    rows = np.arange(min(a.n, n_rows))
    if a.n > n_rows:
        print(f"[{a.cmd}] only {n_rows} {split} draws available, wanted {a.n}", flush=True)
    if a.limit:
        rows = rows[:a.limit]
    return arrays["x"][rows], data.draws(arrays, rows), [int(i) for i in arrays["index"][rows]]


def _designs(model, a, split, update, tag, out, cfg, provenance):
    """Records of design() on the draws of the split, resumable. Returns (records, seconds, new, failed)."""
    xs, draws, indices = _load_draws(a, split)
    print(f"[{tag}] {len(draws)} {split} draws", flush=True)
    log = RunLog(out.with_suffix(".jsonl"), cfg)
    try:
        return run_targets([(str(i), (x, d, i)) for x, d, i in zip(xs, draws, indices)], log,
                           lambda p, j: design_record(model, p[0], p[1], _seed(a.seed, p[2]), a, update, provenance), tag,
                           a.retry_errors)
    finally:
        log.close()


def fit_records(records):
    """The records `fit_update` takes, from the design records: the checked candidates' model probabilities, log-likelihoods and
    capture hits, and the background mass p0 = background_mass(p, n_samples) (ruling R25)."""
    out = []
    for rec in records:
        rows = rec["report"]["candidates"]
        sel = [i for i, r in enumerate(rows) if r["check"] is not None]
        p = [rows[i]["model_probability"] for i in sel]
        out.append(dict(p=p, log_L=[rows[i]["log_likelihood"] for i in sel], hits=[rec["hits"][i] for i in sel],
                        p0=background_mass(p, rec["report"]["n_samples"])))
    return out


def cmd_fit_update(a, cfg):
    if a.check != "all":
        raise SystemExit("fit-update needs every realistic candidate checked (--check all): the update combines them all")
    if a.dry_run:
        return dry_run(a, "fit-update", min(a.n, a.limit) if a.limit else a.n)
    out = Path(a.out) if a.out else Path(a.model).with_name("update.json")
    sha = sha256_of(a.model)
    model = InverseModel.load(a.model)
    records, seconds, new, failed = _designs(model, a, "val", None, "fit-update", out, _resume_cfg(a, cfg, sha),
                                             dict(model=str(a.model), model_sha256=sha))
    recs = fit_records(records)
    try:
        fit = fit_update(recs)
    except ValueError as err:
        raise SystemExit(f"fit-update: cannot fit ({err}); {len(records)} draws, "
                         f"{sum(len(r['p']) for r in recs)} checked candidates")
    result = fit | dict(n=len(records), n_candidates=sum(len(r["p"]) for r in recs), n_hits=int(sum(sum(r["hits"]) for r in recs)),
                        model=str(a.model), data=str(a.data), split="val", flags=_flags(a), seconds=seconds,
                        resumed_draws=len(records) - new, failed_targets=failed_summary(failed)) | _provenance(a, sha, records)
    _write_json(out, result)
    print(f"[fit-update] kappa {fit['kappa']:.4g}, log_L0 {fit['log_L0']:.4g}: ECE {fit['ece']:.4f} updated vs "
          f"{fit['ece_model']:.4f} model" + (" (kappa at the edge of its grid)" if fit["kappa_at_edge"] else "")
          + f"; wrote {out}", flush=True)
    return result


def summarize_end_to_end(records):
    """Top-5 capture, recreation of the top candidate, and the ECE of the model and updated probabilities."""
    top5 = [any(r["hits"]) for r in records]
    recreated = []
    for r in records:
        rows = r["report"]["candidates"]
        recreated.append(bool(rows and rows[0]["check"] is not None and rows[0]["check"]["recreated"]))
    cands = [(row, hit) for r in records for row, hit in zip(r["report"]["candidates"], r["hits"])]
    checked = [(row, hit) for row, hit in cands if row["updated_probability"] is not None]

    def ece(pairs, key):
        return expected_calibration_error([row[key] for row, _ in pairs], [hit for _, hit in pairs]) if pairs else None

    return dict(n=len(records), top5_capture=dict(value=_share(top5), n=len(top5)),
                recreated=dict(value=_share(recreated), n=len(recreated),
                               note="the top candidate's simulator check on fresh seeds (rank 1, none = not recreated)"),
                ece_model=dict(value=ece(cands, "model_probability"), n_candidates=len(cands),
                               note="all returned candidates"),
                ece_model_on_checked=dict(value=ece(checked, "model_probability"), n_candidates=len(checked),
                                          note="the candidates the updated probabilities cover: the like-for-like baseline"),
                ece_updated=dict(value=ece(checked, "updated_probability"), n_candidates=len(checked)))


def update_mismatch(fitted, a, model_sha256):
    """{name: (update.json value, this run's value)} for the check settings (design.UPDATE_SETTINGS: what changes the checks'
    log-likelihoods) and the model hash that differ between the fitted update and this run; an update without recorded
    flags or model hash is a mismatch of its own."""
    return _update_mismatch(fitted, {k: getattr(a, k) for k in UPDATE_SETTINGS}, model_sha256)


def cmd_end_to_end(a, cfg):
    upath = Path(a.update) if a.update else Path(a.model).with_name("update.json")
    if a.dry_run:
        return dry_run(a, "end-to-end", min(a.n, a.limit) if a.limit else a.n)
    if not upath.exists():
        raise SystemExit(f"{upath} not found: run `fit-update` first (or pass --update)")
    fitted = json.loads(upath.read_text())
    update = {k: fitted[k] for k in ("kappa", "log_L0")}
    sha = sha256_of(a.model)
    mismatch = update_mismatch(fitted, a, sha)
    if mismatch and not a.allow_update_mismatch:
        raise SystemExit(f"{upath} was fitted with other settings than this run ({mismatch}): kappa and log_L0 depend on how the "
                         "checks were made. Re-run fit-update with these settings, or pass --allow-update-mismatch")
    if mismatch:
        print(f"[end-to-end] WARNING: update fitted with other settings ({mismatch}); allowed by --allow-update-mismatch", flush=True)
    out = Path(a.out) if a.out else Path(a.model).with_name("end_to_end.json")
    model = InverseModel.load(a.model)
    prov = dict(model=str(a.model), model_sha256=sha, update_file=str(upath), update_flags=fitted.get("flags"),
                update_mismatch=mismatch)
    records, seconds, new, failed = _designs(model, a, "test", update, "end-to-end", out,
                                             _resume_cfg(a, cfg, sha) | dict(update=update), prov)
    result = summarize_end_to_end(records) | dict(update=update, update_file=str(upath), update_sha256=sha256_of(upath),
                                                  update_mismatch=mismatch, model=str(a.model), data=str(a.data),
                                                  split="test", flags=_flags(a), seconds=seconds,
                                                  resumed_draws=len(records) - new,
                                                  failed_targets=failed_summary(failed)) | _provenance(a, sha, records)
    if a.check != "all":
        result["warning"] = "check=top: no updated probabilities, ece_updated is null"
    _write_json(out, result)
    f = lambda m: "n/a" if m["value"] is None else f"{m['value']:.3f}"
    print(f"[end-to-end] top-5 capture {f(result['top5_capture'])}, recreated {f(result['recreated'])}, ECE model "
          f"{f(result['ece_model'])}, updated {f(result['ece_updated'])}; wrote {out}", flush=True)
    return result


# ------------------------------------------------------------------------------------------------------------------- main

def _flags(a):
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items() if k != "run"}


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=None, help=f"default for fit-update and end-to-end: {DEFAULT_MODEL}")
    common.add_argument("--data", default=str(DEFAULT_DATA), help="the generated parent-lattice data (draws and timing)")
    common.add_argument("--out", default=None, help="the output JSON (default: next to the model); the .jsonl log sits beside it")
    common.add_argument("--limit", type=int, default=None, help="run only the first N targets")
    common.add_argument("--check", choices=("all", "top"), default="all",
                        help="simulate every realistic candidate (spec) or only the rank-1 candidate")
    common.add_argument("--no-refine", action="store_true", help="never refine a failed check")
    common.add_argument("--n-jobs", type=int, default=22, help="worker processes per simulation batch")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--dry-run", action="store_true", help="print the projected simulations and wall time and exit")
    common.add_argument("--retry-errors", action="store_true",
                        help="retry the targets the .jsonl records as failed (default: they stay skipped)")
    common.add_argument("--n-samples", type=int, default=1000)
    common.add_argument("--refine-budget", type=int, default=200)
    common.add_argument("--threads", type=int, default=None, help="torch intra-op threads")
    common.add_argument("--seconds-per-sim", type=float, default=None, help="dry run: override the manifest's mean seconds per draw")
    common.add_argument("--expected-candidates", type=float, default=DRY_DEFAULTS["expected_candidates"],
                        help="dry run: realistic candidates per target (at most 5 are returned)")
    common.add_argument("--expected-refine-rate", type=float, default=DRY_DEFAULTS["p_refine"],
                        help="dry run: share of checks that fail and are refined")
    common.add_argument("--n-cells", type=int, default=None, help=argparse.SUPPRESS)       # tests only
    common.add_argument("--seeds", type=int, default=None, help=argparse.SUPPRESS)         # tests only
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit-update", parents=[common], help="fit kappa and log_L0 on validation draws")
    p.add_argument("--n", type=int, default=500)
    p.set_defaults(run=cmd_fit_update)
    p = sub.add_parser("hidden", parents=[common], help="the four hidden-material marks on a catalogue ribbon")
    p.add_argument("--material", required=True, choices=tuple(CATALOGUE))
    p.set_defaults(run=cmd_hidden)
    p = sub.add_parser("end-to-end", parents=[common], help="top-5 capture, recreation and ECE on test draws")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--update", default=None, help="default: update.json next to the model")
    p.add_argument("--allow-update-mismatch", action="store_true",
                   help="run even though update.json was fitted with other check settings (no_refine, refine_budget, n_samples, ...)")
    p.set_defaults(run=cmd_end_to_end)
    return ap


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.model is None:
        if a.cmd == "hidden":
            ap.error("hidden needs --model: the model trained with --exclude <material>")
        a.model = str(DEFAULT_MODEL)
    code_version()                                  # this run's code version: fixed now, when it starts
    if a.threads:
        torch.set_num_threads(a.threads)
    cfg = dict(cmd=a.cmd, model=str(a.model), n_samples=a.n_samples, seed=a.seed, check=a.check, no_refine=a.no_refine,
               refine_budget=a.refine_budget, n_cells=a.n_cells, seeds=a.seeds)
    if a.cmd != "hidden":
        cfg["data"] = str(a.data)
    return a.run(a, cfg)


if __name__ == "__main__":
    main()
