"""scripts/evaluate/eval_inverse.py: catalogue targets, the update fit, the hidden-material marks and the end-to-end run.

Everything that simulates uses a stub model (jittered encodings of fixed hosts), 10-cell devices and 3 seeds per check
(`--n-cells` / `--seeds`, which exist for these tests only)."""
import json

import numpy as np
import pytest

import eval_inverse as ev
from atlaslib.energy import on_axis
from atlaslib.lookup import window_inputs
from atlaslib.store import CloudStore
from shazam.inverse import design as design_mod
from shazam.inverse.model import InverseModel
from shazam.inverse.runio import code_version, sha256_of
from tbribbon.materials import make_model
from tbribbon.parent import TYPES, named_material
from tbribbon.parent_data import V4
from tbribbon.parent_prior import EDGES, Draw
from test_inverse_design import HOST, OTHER, StubModel, sim_target
import paths

pytestmark = pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")

FAST = ["--n-cells", "10", "--seeds", "3", "--refine-budget", "6", "--n-samples", "100", "--n-jobs", "1"]
HBN10 = Draw(named_material("hbn"), "armchair", 10, 0.01)          # the catalogue hBN ribbon (so the stub's top candidate is right)


# ---------------------------------------------------------------------------------------------------------------- catalogue

def test_catalogue_targets_and_truth():
    targets = ev.catalogue_targets("hbn/armchair/N10")
    assert len(targets) == 600
    meta = json.loads((paths.model_dir("hbn/armchair/N10") / "meta.json").read_text())["clouds"]
    counts = {c["density"]: c["n_impurities"] for c in meta.values()}
    assert {d for _, _, d in targets} == set(counts) and len(counts) == 4
    for x, n, d in targets:
        assert x.shape == (416,) and x.dtype == np.float32 and x.any()
        assert n == counts[d]
    assert [sum(d == dd for _, _, d in targets) for dd in sorted(counts)] == [150] * 4
    truth = ev.truth_of("hbn")
    assert truth.params == named_material("hbn")
    assert (truth.edge, truth.width, truth.density) == ("armchair", 10, 0.01)
    assert ev.truth_of("kagome", 0.02).edge == "zigzag" and ev.truth_of("silicene").edge == "armchair"
    with pytest.raises(ValueError):
        ev.truth_of("graphene-ideal")


@pytest.mark.parametrize("material", ["hbn", "silicene", "kagome"])
def test_catalogue_input_agrees_with_the_lookup_route(material):
    """energies_t x s (the 4.3 scale) + target_input = the existing lookup's on_axis + window_inputs, exactly."""
    model_id = ev.CATALOGUE[material]
    _, edge, width = model_id.split("/")
    targets = ev.catalogue_targets(model_id, seeds=[850])
    assert len(targets) == 4                                            # one seed over the four densities
    x, _, density = targets[0]
    store = CloudStore(paths.data_root())
    e_t, _ = store.read_pristine(model_id)
    cloud, seeds = store.read_cloud(model_id, density)
    e_ev, _ = on_axis(V4, make_model(material, edge, int(width[1:])), e_t)
    ref, mask = window_inputs(V4, e_ev, cloud[seeds == 850])
    assert mask.all() and np.array_equal(x, ref[0])


def test_catalogue_seeds_are_selected_by_value_not_position():
    targets = ev.catalogue_targets("kagome/zigzag/N10", seeds=range(990, 1000))
    assert len(targets) == 40


# ---------------------------------------------------------------------------------------------------------- hidden-mark scoring

def _row(host, count=(20, 30), recreated=True, **over):
    canon = {k: float(getattr(host.params, k)) for k in ev.PARAM_NAMES} | {"s": float(host.params.s)}
    canon.update(over)
    return {"rank": 1, "edge": host.edge, "discrete_state": {"pattern": list(host.params.pattern)},
            "parameters": {"canonical": canon}, "impurity_count_90": list(count), "width": host.width,
            "check": None if recreated is None else {"recreated": recreated}}


def _rec(row, true_count=25, density=0.01):
    return {"density": density, "true_count": true_count, "report": {"candidates": [row] if row else []}}


def test_score_hidden_marks_and_tolerances():
    good = _rec(_row(HBN10))
    edge_ok = _rec(_row(HBN10, delta=HBN10.params.delta + 0.149, s=2.30 * 1.099))
    wrong_param = _rec(_row(HBN10, delta=HBN10.params.delta + 0.16))
    wrong_s = _rec(_row(HBN10, s=2.30 * 1.11))
    wrong_edge = _rec(_row(HBN10._replace(edge="zigzag")))
    wrong_pattern = _rec(_row(Draw(named_material("kagome"), "armchair", 10, 0.01)))
    count_miss = _rec(_row(HBN10, count=(26, 30)))
    not_recreated = _rec(_row(HBN10, recreated=False))
    unchecked = _rec(_row(HBN10, recreated=None))
    nothing = _rec(None)
    out = ev.score_hidden("hbn", [good, edge_ok, wrong_param, wrong_s, wrong_edge, wrong_pattern, count_miss,
                                  not_recreated, unchecked, nothing])
    m = out["marks"]
    assert out["n"] == 10
    # right (edge, pattern): good, edge_ok, wrong_param, wrong_s, count_miss, not_recreated, unchecked = 7 of 10
    assert m["key"]["value"] == pytest.approx(0.7)
    # right key AND every parameter within 0.15 AND s within 10%: good, edge_ok, count_miss, not_recreated, unchecked
    assert m["params"]["value"] == pytest.approx(0.5)
    assert m["count_interval"]["value"] == pytest.approx(0.8)             # misses: count_miss and 'nothing' (no candidate)
    # recreated: good, edge_ok, wrong_param, wrong_s, wrong_edge, wrong_pattern, count_miss; the unchecked top does not
    assert m["recreated"]["value"] == pytest.approx(0.7)
    assert out["per_parameter"]["delta"]["within_tolerance"] == pytest.approx(6 / 7)      # among the 7 right-key targets
    assert out["per_parameter"]["s"]["within_tolerance"] == pytest.approx(6 / 7)
    assert out["marks"]["key"]["threshold"] == 0.90 and out["marks"]["recreated"]["threshold"] == 0.80
    assert out["marks"]["count_interval"]["threshold"] == 0.85
    assert out["pass"] is False and not any(v["pass"] for v in m.values())


def test_score_hidden_passes_when_every_mark_clears():
    recs = [_rec(_row(HBN10)) for _ in range(20)]
    out = ev.score_hidden("hbn", recs)
    assert all(v["value"] == 1.0 and v["pass"] for v in out["marks"].values()) and out["pass"] is True
    assert out["per_density"]["0.01"]["n"] == 20


# ------------------------------------------------------------------------------------------------------------------ design.py

def test_design_check_top_and_no_refinement():
    model = StubModel([(HOST, 70), (OTHER, 30)])
    kw = dict(n_samples=100, refine_budget=6, n_jobs=1, n_cells=10, seeds_override=3)
    report = design_mod.design(sim_target(HOST), model, check="top", **kw)
    assert report["candidates"][0]["check"] is not None
    assert report["candidates"][1]["check"] is None and report["candidates"][1]["realistic"] is True
    assert all(c["updated_probability"] is None for c in report["candidates"])
    assert report["update_fitted"] is False and "only the top candidate" in report["warning"]
    json.dumps(report, allow_nan=False)
    assert report["updated_probabilities_computed"] is False
    assert design_mod.design(sim_target(HOST), model, **kw)["updated_probabilities_computed"] is True
    # R24/R25: the background mass counts the realistic candidates whether or not they were checked
    report = design_mod.design(sim_target(HOST), model, check="top", update={"kappa": 1.0, "log_L0": -1e4}, **kw)
    assert report["update_fitted"] is True and report["update"] == {"kappa": 1.0, "log_L0": -1e4}
    assert report["updated_probabilities_computed"] is False and "only the top candidate" in report["warning"]
    assert [c["model_probability"] for c in report["candidates"]] == pytest.approx([0.7, 0.3])
    assert report["p_background"] == pytest.approx(1 / 101)                     # the floor 1 / (n_samples + 1), not 0.3
    assert all(c["updated_probability"] is None for c in report["candidates"])
    # a target that is not this host fails the check: refine_budget 0 keeps the verify verdict and runs no refinement
    report = design_mod.design(sim_target(OTHER), StubModel([(HOST, 100)]), **(kw | {"refine_budget": 0}))
    top = report["candidates"][0]
    assert top["check"]["seeds"] == "verify" and top["refined_parameters"] is None and report["n_simulations"] == 3
    with pytest.raises(ValueError, match="check"):
        design_mod.design(sim_target(HOST), model, check="some", **kw)


# ------------------------------------------------------------------------------------------------------------ subcommands

def make_data(root):
    """A 6-draw generated set: validation draws (index % 20 == 18) and test draws (== 19), three of each."""
    hosts = [HOST, OTHER, HOST]
    rows = []
    for split_res in (18, 19):
        for k, host in enumerate(hosts):
            rows.append((20 * k + split_res, host))
    on = np.array([h.params.pattern for _, h in rows])
    arrays = dict(
        index=np.array([i for i, _ in rows]),
        x=np.array([sim_target(h, seed=i) for i, h in rows], np.float32),
        on=on, lead=np.array([TYPES.index(h.params.lead) for _, h in rows]),
        params=np.array([[getattr(h.params, c) for c in ev.PARAM_COLUMNS] for _, h in rows]),
        edge=np.array([EDGES.index(h.edge) for _, h in rows]), width=np.array([h.width for _, h in rows]),
        density=np.array([h.density for _, h in rows]))
    root.mkdir(parents=True)
    np.savez(root / "shard_00000.npz", **arrays)
    (root / "manifest.json").write_text(json.dumps({"shards": [{"n_draws": 6, "n_attempts": 6, "seconds": 60.0}],
                                                    "counts": {"n_draws": 6}}))
    return root


@pytest.fixture
def stub(monkeypatch):
    model = StubModel([(HOST, 70), (OTHER, 30)])
    monkeypatch.setattr(InverseModel, "load", classmethod(lambda cls, path: model))
    return model


@pytest.fixture
def data(tmp_path):
    return make_data(tmp_path / "data")


def model_file(folder, content=b"stub weights", hide=None):
    """A model.pt for the stub to stand in for (its bytes are hashed), and the history.json of a --exclude <hide> run."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.pt").write_bytes(content)
    if hide is not None:
        (folder / "history.json").write_text(json.dumps({"excluded": {"hide": hide, "train": 1, "val": 0}}))
    return folder / "model.pt"


def test_fit_update_writes_update_json_and_resumes(tmp_path, stub, data):
    out = tmp_path / "m"
    model = model_file(out)
    argv = ["fit-update", "--model", str(model), "--data", str(data), *FAST]
    rep = ev.main(argv + ["--n", "3", "--limit", "2"])
    update = json.loads((out / "update.json").read_text())
    assert {"kappa", "log_L0", "ece", "ece_model", "kappa_at_edge", "n"} <= set(update)
    assert update["n"] == 2 and rep["n"] == 2 and np.isfinite(update["kappa"]) and np.isfinite(update["log_L0"])
    assert update["flags"]["limit"] == 2 and update["flags"]["check"] == "all" and update["flags"]["no_refine"] is False
    assert update["model_sha256"] == sha256_of(model) and update["code_version"] == code_version()
    assert update["data_provenance"]["n_draws"] == 6 and update["failed_targets"] == {"n": 0, "keys": [], "errors": {}}
    assert isinstance(update["log_L0_at_edge"], bool)
    lines = (out / "update.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["cfg"]["model_sha256"] == sha256_of(model) and json.loads(lines[0])["cfg"]["data_draws"] == 6
    report = json.loads(lines[0])["record"]["report"]
    assert report["provenance"]["model_sha256"] == sha256_of(model) and report["provenance"]["model"] == str(model)
    assert len(lines) == 2 and len(stub.calls) == 2
    first = json.loads(lines[0])
    hits = [json.loads(line)["record"]["hits"] for line in lines]
    assert [r["model_probability"] for r in first["record"]["report"]["candidates"]] == pytest.approx([0.7, 0.3])
    assert hits == [[True, False], [False, True]]      # draw 0's truth is HOST (rank 1), draw 1's is OTHER (rank 2)
    assert first["record"]["report"]["update_fitted"] is False and len(first["record"]["hits"]) == len(
        first["record"]["report"]["candidates"])
    # resume: the two finished draws are not designed again, only the third is
    ev.main(argv + ["--n", "3"])
    assert len(stub.calls) == 3 and len((out / "update.jsonl").read_text().splitlines()) == 3
    assert json.loads((out / "update.json").read_text())["n"] == 3
    assert not list(out.glob("*.tmp"))
    # changed settings are refused rather than mixed: other samples, a retrained model, an extended data set
    with pytest.raises(SystemExit):
        ev.main(argv + ["--n", "3", "--n-samples", "50"])
    model.write_bytes(b"retrained weights")
    with pytest.raises(SystemExit, match="model_sha256"):
        ev.main(argv + ["--n", "3"])
    model.write_bytes(b"stub weights")
    m = json.loads((data / "manifest.json").read_text())
    (data / "manifest.json").write_text(json.dumps(m | {"counts": {"n_draws": 12}}))
    with pytest.raises(SystemExit, match="data_draws"):
        ev.main(argv + ["--n", "3"])


def test_fit_records_keep_only_checked_candidates():
    def row(p, checked, log_L=-5.0):
        return {"model_probability": p, "check": {"recreated": True} if checked else None,
                "log_likelihood": log_L if checked else None}

    rec = {"report": {"n_samples": 100, "candidates": [row(0.5, True, -3.0), row(0.2, False), row(0.1, True, -7.0)]},
           "hits": [True, True, False]}
    empty = {"report": {"n_samples": 100, "candidates": [row(0.4, False)]}, "hits": [False]}
    out = ev.fit_records([rec, empty])
    assert out[0]["p"] == [0.5, 0.1] and out[0]["log_L"] == [-3.0, -7.0] and out[0]["hits"] == [True, False]
    assert out[0]["p0"] == pytest.approx(ev.background_mass([0.5, 0.1], 100)) and out[0]["p0"] == pytest.approx(0.4)
    assert out[1]["p"] == [] and out[1]["p0"] == pytest.approx(1.0)
    floor = ev.fit_records([{"report": {"n_samples": 100, "candidates": [row(1.0, True)]}, "hits": [True]}])
    assert floor[0]["p0"] == pytest.approx(1 / 101)


def test_fit_update_refuses_top_only_checks(tmp_path, stub, data):
    with pytest.raises(SystemExit, match="every realistic candidate"):
        ev.main(["fit-update", "--model", str(model_file(tmp_path)), "--data", str(data), "--check", "top", *FAST])


def test_a_failing_target_is_recorded_and_the_run_goes_on(tmp_path, stub, data, monkeypatch):
    real, bad = ev.design_record, ev._seed(0, 38)                     # the second validation draw (index 38)
    tried = []

    def flaky(model, x, truth, seed, a, update, provenance=None):
        tried.append(seed)
        if seed == bad:
            raise RuntimeError("worker died")
        return real(model, x, truth, seed, a, update, provenance)

    monkeypatch.setattr(ev, "design_record", flaky)
    out = tmp_path / "m"
    argv = ["fit-update", "--model", str(model_file(out)), "--data", str(data), *FAST, "--n", "3"]
    rep = ev.main(argv)
    assert rep["n"] == 2 and rep["failed_targets"] == {"n": 1, "keys": ["38"], "errors": {"38": "RuntimeError('worker died')"}}
    lines = [json.loads(line) for line in (out / "update.jsonl").read_text().splitlines()]
    assert [line["key"] for line in lines] == ["18", "38", "58"] and lines[1]["error"] == "RuntimeError('worker died')"
    assert "record" not in lines[1]
    tried.clear()
    assert ev.main(argv)["failed_targets"]["keys"] == ["38"] and tried == []        # a rerun does not retry recorded errors
    monkeypatch.setattr(ev, "design_record", real)
    again = ev.main(argv + ["--retry-errors"])
    assert again["n"] == 3 and again["failed_targets"]["n"] == 0
    assert json.loads((out / "update.json").read_text())["n"] == 3


def test_hidden_scores_the_catalogue_and_resumes(tmp_path, monkeypatch):
    model = StubModel([(HBN10, 100)])
    monkeypatch.setattr(InverseModel, "load", classmethod(lambda cls, path: model))
    path = model_file(tmp_path, hide="hbn")
    argv = ["hidden", "--material", "hbn", "--model", str(path), "--check", "top", "--no-refine", *FAST]
    rep = ev.main(argv + ["--limit", "2"])
    out = json.loads((tmp_path / "hidden_hbn.json").read_text())
    assert out["model_sha256"] == sha256_of(path) and out["code_version"] == code_version()
    assert out["model_excluded"]["hide"] == "hbn" and out["failed_targets"]["n"] == 0
    assert set(out["model_training"]) == {"code_version", "data", "data_provenance"}      # from the model's history.json
    assert json.loads((tmp_path / "hidden_hbn.jsonl").read_text().splitlines()[0])["cfg"]["model_sha256"] == sha256_of(path)
    assert out["material"] == "hbn" and out["n"] == 2 and rep["n"] == 2 and len(model.calls) == 2
    assert set(out["marks"]) == {"key", "params", "count_interval", "recreated"}
    for mark in out["marks"].values():
        assert {"value", "threshold", "pass"} <= set(mark)
    assert out["marks"]["key"]["value"] == 1.0                     # the stub offers the catalogue's own (edge, pattern)
    assert {"s", "delta", "t_AA"} <= set(out["per_parameter"])
    assert out["flags"]["check"] == "top" and out["flags"]["no_refine"] is True and out["flags"]["limit"] == 2
    assert out["pass"] in (True, False) and out["catalogue"] == "hbn/armchair/N10"
    ev.main(argv + ["--limit", "3"])
    assert len(model.calls) == 3
    assert json.loads((tmp_path / "hidden_hbn.json").read_text())["n"] == 3
    with pytest.raises(SystemExit):
        ev.main(["hidden", "--material", "graphene", "--model", str(tmp_path / "model.pt")])


@pytest.mark.parametrize("hide", ["none", "kagome", None])
def test_hidden_refuses_a_model_that_saw_the_material(tmp_path, monkeypatch, hide):
    def boom(cls, path):
        raise AssertionError("the model must not be loaded")

    monkeypatch.setattr(InverseModel, "load", classmethod(boom))
    path = model_file(tmp_path, hide=hide)
    with pytest.raises(SystemExit, match="exclude"):
        ev.main(["hidden", "--material", "hbn", "--model", str(path), *FAST])
    assert not (tmp_path / "hidden_hbn.jsonl").exists()


def test_end_to_end_reports_both_calibrations(tmp_path, stub, data):
    out = tmp_path / "m"
    model = model_file(out)
    sha = sha256_of(model)
    flags = {"no_refine": False, "refine_budget": 6, "n_samples": 100, "seeds": 3, "n_cells": 10}
    (out / "update.json").write_text(json.dumps({"kappa": 100.0, "log_L0": -5000.0, "flags": flags, "model_sha256": sha}))
    argv = ["end-to-end", "--model", str(model), "--data", str(data), *FAST]
    rep = ev.main(argv + ["--n", "3", "--limit", "2"])
    res = json.loads((out / "end_to_end.json").read_text())
    assert {"top5_capture", "recreated", "ece_model", "ece_updated", "n", "flags", "update"} <= set(res)
    assert res["n"] == 2 and rep["n"] == 2 and res["update"]["kappa"] == 100.0
    assert 0.0 <= res["top5_capture"]["value"] <= 1.0 and 0.0 <= res["recreated"]["value"] <= 1.0
    assert res["ece_model"]["value"] is None or 0.0 <= res["ece_model"]["value"] <= 1.0
    assert res["model_sha256"] == sha and res["update_sha256"] == sha256_of(out / "update.json")
    assert res["code_version"] == code_version() and res["data_provenance"]["n_draws"] == 6
    lines = (out / "end_to_end.jsonl").read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["record"]["report"]["update_fitted"] is True
    prov = json.loads(lines[0])["record"]["report"]["provenance"]
    assert prov["update_file"] == str(out / "update.json") and prov["update_flags"] == flags and prov["model_sha256"] == sha
    ev.main(argv + ["--n", "3"])
    assert len(stub.calls) == 3
    # the update was fitted under other check settings: refused, unless explicitly allowed
    (out / "update.json").write_text(json.dumps({"kappa": 100.0, "log_L0": -5000.0, "flags": flags | {"refine_budget": 200},
                                                 "model_sha256": sha}))
    with pytest.raises(SystemExit, match=r"other settings.*refine_budget"):
        ev.main(argv + ["--n", "3", "--out", str(out / "other.json")])
    (out / "update.json").write_text(json.dumps({"kappa": 100.0, "log_L0": -5000.0, "flags": flags, "model_sha256": "0" * 64}))
    with pytest.raises(SystemExit, match=r"other settings.*model_sha256"):           # fitted on another model
        ev.main(argv + ["--n", "3", "--out", str(out / "other.json")])
    (out / "update.json").write_text(json.dumps({"kappa": 100.0, "log_L0": -5000.0}))             # no flags, no model hash
    with pytest.raises(SystemExit, match="other settings"):
        ev.main(argv + ["--n", "3", "--out", str(out / "other.json")])
    ev.main(argv + ["--n", "3", "--out", str(out / "other.json"), "--allow-update-mismatch", "--limit", "1"])
    assert (out / "other.json").exists()
    (out / "update.json").unlink()
    with pytest.raises(SystemExit, match="fit-update"):
        ev.main(argv + ["--n", "3"])


def test_dry_run_projects_cost_and_touches_nothing(tmp_path, monkeypatch, capsys, data):
    def boom(cls, path):
        raise AssertionError("the model must not be loaded")

    monkeypatch.setattr(InverseModel, "load", classmethod(boom))
    out = tmp_path / "m"
    out.mkdir()
    proj = ev.main(["fit-update", "--model", str(out / "model.pt"), "--data", str(data), "--n", "500", "--dry-run",
                    "--n-jobs", "10", "--expected-candidates", "3"])
    assert not list(out.iterdir())
    text = capsys.readouterr().out
    assert "dry run" in text and "simulations" in text and "hours" in text
    assert proj["seconds_per_simulation"] == pytest.approx(10.0)          # 60 s over 6 draws in the manifest
    # per candidate on 10 jobs: 50 verify seeds = 5 rounds; P(refine) 0.5 x (10 evaluations of 20 seeds = 2 rounds each + final 5)
    assert proj["simulations"] == pytest.approx(500 * 3 * 175)            # 50 + 0.5 x (200 + 50) simulations
    assert proj["rounds_per_candidate"] == pytest.approx(5 + 0.5 * (10 * 2 + 5))
    assert proj["wall_hours"] == pytest.approx(500 * 3 * 17.5 * 10.0 / 3600)
    # rounds, not simulations / jobs: 50 seeds on 22 jobs take 3 rounds, 20 seeds 1
    p22 = ev.project(1, "all", False, 1.0, 1.0, 200, 50, 20, 1.0, 22)
    assert p22["rounds_per_candidate"] == 3 + (10 * 1 + 3) and p22["wall_hours"] == pytest.approx(16 / 3600)
    assert ev.project(1, "all", True, 1.0, 1.0, 200, 50, 20, 1.0, 22)["rounds_per_candidate"] == 3
    cheap = ev.main(["hidden", "--material", "kagome", "--model", str(out / "model.pt"), "--dry-run", "--check", "top",
                     "--no-refine", "--limit", "10", "--data", str(data)])
    assert cheap["n_targets"] == 10 and cheap["simulations"] == pytest.approx(10 * 50)
    assert cheap["rounds"] == pytest.approx(10 * 3)                      # 50 seeds on the default 22 jobs
    assert not list(out.iterdir())
