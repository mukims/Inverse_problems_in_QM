"""Design pipeline and report (spec sections 7 and 8, output)."""
import json

import numpy as np
import pytest
import torch

from shazam.inverse import candidates as candidates_mod
from shazam.inverse import design as design_mod
from shazam.inverse import refine as refine_mod
from shazam.inverse.design import background_mass, design, model_input, target_input
from shazam.inverse.refine import RefineResult
from shazam.inverse.verify import FINAL_SEEDS, OPT_SEEDS, VERIFY_SEEDS, log_likelihood
from shazam.inverse.encoding import encode
from tbribbon.parent import HexParams, named_material
from tbribbon.parent_data import V4, simulate_raw, to_input, to_input_bandtop, to_input_linear
from tbribbon.parent_prior import Draw

pytestmark = pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")

N_CELLS = 10              # short devices, and 3 seeds per check, keep the tests fast (seeds_override is for tests only)
SEEDS = 3
HOST = Draw(named_material("hbn"), "zigzag", 5, 0.02)
OTHER = Draw(HexParams(False, True, False, "H", 1.3, eps_H_mean=-0.4, delta=1.0, t_AB=1.0, t_AA=0.2), "armchair", 12, 0.004)


class StubModel:
    """Stands in for InverseModel.sample: jittered encodings of fixed hosts, as in the candidates tests."""

    def __init__(self, hosts, sigma=0.02, seed=0):
        self.hosts, self.sigma, self.seed = hosts, sigma, seed
        self.calls = []

    def sample(self, x, n, steps=64, state=None):
        self.calls.append((tuple(np.shape(x)), n))
        rng = np.random.default_rng(self.seed)
        states, zs = [], []
        total = sum(c for _, c in self.hosts)
        for draw, count in self.hosts:
            k, z, mask = encode(draw)
            m = round(n * count / total)
            states += [k] * m
            zs.append(z + rng.normal(0, self.sigma, (m, z.size)) * mask)
        states, z = np.array(states), np.concatenate(zs)
        order = rng.permutation(len(states))
        return torch.as_tensor(states[order]), torch.as_tensor(z[order], dtype=torch.float32)


def sim_target(host, seed=7):
    from tbribbon.parent_data import n_impurities
    return to_input(simulate_raw(host, n_impurities(host, N_CELLS), seed, n_cells=N_CELLS))


def run(target, model, **kw):
    kw = dict(n_samples=100, refine_budget=6, n_jobs=1, n_cells=N_CELLS, seeds_override=SEEDS) | kw
    return design(target, model, **kw)


CANDIDATE_KEYS = {"rank", "model_probability", "updated_probability", "realistic", "box_edge_warning", "discrete_state",
                  "parameters", "edge", "width", "impurity_count", "impurity_count_90", "density", "density_90",
                  "nearest_material", "check", "refined_parameters"}


def test_report_has_every_key():
    model = StubModel([(HOST, 70), (OTHER, 30)])
    report = run(sim_target(HOST), model)
    json.dumps(report, allow_nan=False)                                # plain JSON types only
    assert {"p_unassigned", "candidates", "update_fitted", "warning", "n_simulations", "seconds"} <= set(report)
    assert report["update_fitted"] is False and "no fitted update" in report["warning"]
    assert model.calls == [((1, 416), 100)]
    assert len(report["candidates"]) == 2
    top = report["candidates"][0]
    assert set(top) >= CANDIDATE_KEYS and top["rank"] == 1
    assert top["model_probability"] == pytest.approx(0.7) and report["candidates"][1]["model_probability"] == pytest.approx(0.3)
    assert top["edge"] == "zigzag" and top["width"] == 5 and top["nearest_material"] == "hbn"
    assert top["realistic"] is True and top["box_edge_warning"] == []
    assert set(top["discrete_state"]) == {"edge", "pattern", "lead", "delta_zero", "eps_zero", "t_AA_on", "cross"}
    assert set(top["parameters"]) == {"canonical", "ev"}
    assert top["parameters"]["canonical"]["s"] == pytest.approx(HOST.params.s, rel=0.05)
    assert top["parameters"]["ev"]["delta"] == pytest.approx(top["parameters"]["canonical"]["delta"]
                                                             * top["parameters"]["canonical"]["s"])
    lo, hi = top["density_90"]
    assert lo <= top["density"] * 1.2 and hi >= top["density"] * 0.8 and len(top["impurity_count_90"]) == 2
    assert top["check"].keys() >= {"p_value", "recreated", "misfit", "seeds"} and top["check"]["seeds"] in ("verify", "final")
    assert 0.0 <= top["updated_probability"] <= 1.0
    # no fitted update: the updated probabilities are renormalised over the returned candidates
    assert sum(c["updated_probability"] for c in report["candidates"]) == pytest.approx(1.0)
    assert report["n_simulations"] >= 2 * SEEDS


def test_fitted_update_keeps_background_mass():
    stray = Draw(named_material("kagome"), "zigzag", 8, 0.01)
    model = StubModel([(HOST, 60), (OTHER, 30), (stray, 10)])                           # 10 samples: below the cluster minimum
    report = run(sim_target(HOST), model, update={"kappa": 1.0, "log_L0": 1e9})        # background wins: all fall to ~0
    assert report["update_fitted"] is True and "warning" not in report
    assert report["p_unassigned"] == pytest.approx(0.1) and report["p_background"] == pytest.approx(0.1)
    assert all(c["updated_probability"] < 1e-6 for c in report["candidates"])


def test_unrealistic_candidates_are_reported_but_not_simulated(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("simulator called")

    monkeypatch.setattr(candidates_mod, "is_realistic", lambda draw: False)
    monkeypatch.setattr(design_mod, "simulate_inputs", boom)
    monkeypatch.setattr(refine_mod, "simulate_inputs", boom)
    report = run(sim_target(HOST), StubModel([(HOST, 70), (OTHER, 30)]))
    assert len(report["candidates"]) == 2 and report["n_simulations"] == 0
    for c in report["candidates"]:
        assert c["realistic"] is False and c["check"] is None
        assert c["updated_probability"] is None and c["refined_parameters"] is None and c["model_probability"] > 0
    json.dumps(report, allow_nan=False)


def test_malformed_targets_are_rejected():
    e, T = V4.energies_t(), np.linspace(0.0, 3.0, 416)
    with pytest.raises(ValueError, match="non-finite"):
        target_input(e, np.where(np.arange(416) == 5, np.nan, T))
    with pytest.raises(ValueError, match="same length"):
        target_input(e, T[:415])
    with pytest.raises(ValueError, match="full 0-8.3 eV window required"):
        target_input(np.linspace(0.0, 4.0, 416), T)
    with pytest.raises(ValueError, match="full 0-8.3 eV window required"):
        target_input(e[1:] + 0.5, T[1:])
    with pytest.raises(ValueError, match="target has no transmission"):
        target_input(e, np.zeros(416))
    x = target_input(np.linspace(0.0, 9.0, 600), np.linspace(0.0, 3.0, 600))         # a finer, wider grid is interpolated
    assert x.shape == (416,) and x.dtype == np.float32 and x.any()
    with pytest.raises(ValueError, match="target_x"):
        design(np.zeros(415, np.float32), StubModel([(HOST, 1)]))


def test_target_input_equals_the_simulated_input_on_the_v4_grid():
    raw = simulate_raw(HOST, 3, 1, n_cells=N_CELLS)
    assert np.array_equal(target_input(V4.energies_t(), raw), to_input(raw))


def test_model_input_rebuilds_the_training_inputs_from_the_v4_input():
    raw = simulate_raw(OTHER, 6, 2, n_cells=N_CELLS)
    x = target_input(V4.energies_t(), raw)
    assert np.array_equal(model_input(x, "log"), x)
    assert np.allclose(model_input(x, "linear"), to_input_linear(raw)[0], atol=1e-6)
    bt = model_input(x, "bandtop")
    assert bt.shape == (417,) and bt.dtype == np.float32
    assert np.allclose(bt, to_input_bandtop(raw)[0], atol=1e-6)
    with pytest.raises(ValueError, match="input"):
        model_input(x, "spectral")


def test_design_feeds_the_model_the_input_it_was_trained_on():
    model = StubModel([(HOST, 1)])
    model.cfg = {"input": "bandtop"}
    run(sim_target(HOST), model, n_samples=20, refine_budget=0)
    assert model.calls[0][0] == (1, 417)


def test_no_candidates_still_writes_a_report():
    rng = np.random.default_rng(0)

    class Scattered:
        def sample(self, x, n, steps=64, state=None):
            from shazam.inverse.encoding import STATES, active_slots
            states = rng.integers(0, len(STATES), n)
            z = np.stack([rng.normal(0, 3, 14) * active_slots(int(k)) for k in states])
            return torch.as_tensor(states), torch.as_tensor(z, dtype=torch.float32)

    report = run(sim_target(HOST), Scattered(), n_samples=40)
    assert report["candidates"] == [] and report["p_unassigned"] == 1.0 and report["n_simulations"] == 0
    json.dumps(report, allow_nan=False)


def test_failed_refinement_still_reports_the_candidate():
    unreachable = np.ones(416, np.float32)                                   # transmission at the cap everywhere
    report = run(unreachable, StubModel([(HOST, 100)]))
    (c,) = report["candidates"]
    assert c["check"]["recreated"] is False and c["check"]["seeds"] == "verify"
    r = c["refined_parameters"]
    assert r is not None and set(r) >= {"canonical", "ev"} and r["stopped_early"] is False
    assert report["n_simulations"] == SEEDS + 6                              # verify seeds + a 6-simulation budget
    assert c["updated_probability"] == pytest.approx(1.0)                    # no fitted update: renormalised


def test_cli_writes_the_report_atomically_next_to_the_target(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "design_from_spectrum", Path(__file__).resolve().parents[2] / "scripts/evaluate/design_from_spectrum.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli.InverseModel, "load", lambda path: StubModel([(HOST, 100)]))
    monkeypatch.setattr(cli, "design", lambda *a, **k: design(*a, **(k | dict(n_cells=N_CELLS, seeds_override=SEEDS, refine_budget=6))))
    np.save(tmp_path / "T.npy", simulate_raw(HOST, 3, 1, n_cells=N_CELLS))
    np.save(tmp_path / "E.npy", V4.energies_t())
    from shazam.inverse.runio import sha256_of
    (tmp_path / "model.pt").write_text("stub")
    sha = sha256_of(tmp_path / "model.pt")
    flags = {"no_refine": False, "refine_budget": 200, "n_samples": 100, "seeds": None, "n_cells": None}
    (tmp_path / "update.json").write_text(json.dumps({"kappa": 2.0, "log_L0": -1e9, "flags": flags, "model_sha256": sha}))
    threads = []
    monkeypatch.setattr(cli.torch, "set_num_threads", threads.append)
    base = ["--target", str(tmp_path / "T.npy"), "--energies", str(tmp_path / "E.npy"), "--model", str(tmp_path / "model.pt"),
            "--n-samples", "100", "--n-jobs", "1"]
    cli.main(base + ["--threads", "3"])
    assert threads == [3]
    report = json.loads((tmp_path / "T_design.json").read_text())
    assert report["update_fitted"] is True and report["update"] == {"kappa": 2.0, "log_L0": -1e9}
    prov = report["provenance"]
    assert prov["model"] == str(tmp_path / "model.pt") and prov["model_sha256"] == sha
    assert prov["update_file"] == str(tmp_path / "update.json") and prov["update_flags"] == flags
    assert prov["update_mismatch"] == {}
    assert not list(tmp_path.glob("*.tmp"))
    # M5: an update fitted under other settings, or on another model, is refused unless explicitly allowed
    for other in ({"flags": flags | {"n_samples": 1000}, "model_sha256": sha}, {"flags": flags, "model_sha256": "0" * 64}, {}):
        (tmp_path / "update.json").write_text(json.dumps({"kappa": 2.0, "log_L0": -1e9} | other))
        with pytest.raises(SystemExit, match="other settings"):
            cli.main(base)
    cli.main(base + ["--allow-update-mismatch"])
    prov = json.loads((tmp_path / "T_design.json").read_text())["provenance"]
    assert set(prov["update_mismatch"]) == {"flags", "model_sha256"}


def test_target_input_crops_a_grid_that_starts_below_zero():
    reference = target_input(np.linspace(0.0, 9.0, 451), np.linspace(0.0, 3.0, 451))
    e = np.arange(-1.01, 9.0, 0.02)                                  # no grid point at 0: the value there is interpolated
    got = target_input(e, np.interp(e, np.linspace(0.0, 9.0, 451), np.linspace(0.0, 3.0, 451), left=0.0))
    assert got.shape == (416,) and np.allclose(got, reference, atol=0.02)
    with pytest.raises(ValueError, match="full 0-8.3 eV window required"):
        target_input(np.linspace(-1.0, 6.0, 400), np.linspace(0.0, 3.0, 400))
    with pytest.raises(ValueError, match="increasing"):
        target_input(np.array([0.0, 2.0, 1.0, 9.0]), np.ones(4))


def test_non_finite_update_is_rejected():
    for bad in ({"kappa": 1.0, "log_L0": -np.inf}, {"kappa": np.nan, "log_L0": 0.0}, {"kappa": 0.0, "log_L0": 0.0}):
        with pytest.raises(ValueError, match="finite"):
            run(sim_target(HOST), StubModel([(HOST, 100)]), update=bad)


def test_background_mass_has_an_add_one_floor():
    assert background_mass([0.4, 0.5], 1000) == pytest.approx(0.1)
    assert background_mass([0.7, 0.3], 1000) == pytest.approx(1 / 1001)
    assert background_mass([], 1000) == 1.0


def test_a_badly_fitting_lone_candidate_is_not_renormalised_to_one_under_a_fitted_update():
    report = run(np.ones(416, np.float32), StubModel([(HOST, 100)]), update={"kappa": 1.0, "log_L0": 0.0})
    (c,) = report["candidates"]
    assert c["model_probability"] == pytest.approx(1.0) and report["p_background"] == pytest.approx(1 / 101)
    assert c["updated_probability"] < 0.01


def _refined_host(s):
    import dataclasses
    return HOST._replace(params=dataclasses.replace(HOST.params, s=s))


def _spied(monkeypatch, refined, stopped_early):
    """Real simulations through a spy; `refine` replaced by one that returns `refined`."""
    calls, real = [], design_mod.simulate_inputs

    def spy(draw, n_imp, seeds, n_jobs=1, n_cells=None):
        out = real(draw, n_imp, seeds, n_jobs=n_jobs, n_cells=n_cells)
        calls.append((draw, n_imp, tuple(seeds), n_cells, out))
        return out

    monkeypatch.setattr(design_mod, "simulate_inputs", spy)
    monkeypatch.setattr(design_mod, "refine", lambda *a, **k: RefineResult(refined, 0.0, 6, stopped_early))
    return calls


def test_an_early_stopped_refinement_is_rechecked_on_the_final_seeds(monkeypatch):
    refined = _refined_host(2.5)
    calls = _spied(monkeypatch, refined, True)
    target = np.ones(416, np.float32)
    report = run(target, StubModel([(HOST, 100)]))
    (c,) = report["candidates"]
    (d0, n0, s0, nc0, _), (d1, n1, s1, nc1, final) = calls
    assert s0 == VERIFY_SEEDS[:SEEDS] and s1 == FINAL_SEEDS[:SEEDS]
    assert d1 == refined and d0 != refined and n1 == n0 == c["impurity_count"] and nc0 == nc1 == N_CELLS
    assert c["check"]["seeds"] == "final" and c["refined_parameters"]["stopped_early"] is True
    assert c["log_likelihood"] == pytest.approx(log_likelihood(target, final))
    assert report["n_simulations"] == SEEDS + 6 + SEEDS


def test_a_refined_host_at_the_box_edge_is_warned_about(monkeypatch):
    for stopped in (True, False):
        _spied(monkeypatch, _refined_host(3.99), stopped)
        (c,) = run(np.ones(416, np.float32), StubModel([(HOST, 100)]))["candidates"]
        assert "s" in c["refined_parameters"]["box_edge_warning"]
        if stopped:                                                   # the refined host is the verdict: warning at candidate level
            assert c["check"]["seeds"] == "final" and "s" in c["box_edge_warning"]
        else:                                                         # the original verify check stands: its host has no warning
            assert c["check"]["seeds"] == "verify" and c["box_edge_warning"] == []


def test_cli_fails_fast_on_a_bad_out_or_update(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "design_from_spectrum2", Path(__file__).resolve().parents[2] / "scripts/evaluate/design_from_spectrum.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    ran = []
    monkeypatch.setattr(cli.InverseModel, "load", lambda path: StubModel([(HOST, 100)]))
    monkeypatch.setattr(cli, "design", lambda *a, **k: ran.append(1) or {"candidates": [], "p_unassigned": 1.0,
                                                                         "n_simulations": 0, "seconds": 0.0})
    np.save(tmp_path / "T.npy", simulate_raw(HOST, 3, 1, n_cells=N_CELLS))
    np.save(tmp_path / "E.npy", V4.energies_t())
    (tmp_path / "model.pt").write_text("stub")
    base = ["--target", str(tmp_path / "T.npy"), "--energies", str(tmp_path / "E.npy"), "--model", str(tmp_path / "model.pt")]
    cli.main(base + ["--out", str(tmp_path / "new" / "dir" / "o.json")])          # a missing parent is created before design
    assert ran and (tmp_path / "new" / "dir" / "o.json").exists()
    ran.clear()
    (tmp_path / "update.json").write_text(json.dumps({"kappa": 2.0}))
    with pytest.raises(SystemExit, match="log_L0"):
        cli.main(base)
    (tmp_path / "update.json").write_text('{"kappa": 2.0, "log_L0": -Infinity}')
    with pytest.raises(SystemExit, match="finite"):
        cli.main(base)
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(OSError):
        cli.main(["--target", base[1], "--energies", base[3], "--model", base[5], "--out", str(blocker / "o.json")])
    assert not ran                                                   # nothing was simulated


def test_report_records_its_provenance_and_the_intervals():
    from shazam.inverse.runio import code_version
    report = run(sim_target(HOST), StubModel([(HOST, 70), (OTHER, 30)]), seed=4,
                 provenance={"model": "m/model.pt", "model_sha256": "ab" * 32})
    prov = report["provenance"]
    assert prov["code_version"] == code_version() and prov["model"] == "m/model.pt" and prov["model_sha256"] == "ab" * 32
    assert (prov["n_samples"], prov["seed"], prov["check"], prov["refine_budget"]) == (100, 4, "all", 6)
    assert prov["seeds"] == {"verify": {"first": VERIFY_SEEDS[0], "n": SEEDS}, "final": {"first": FINAL_SEEDS[0], "n": SEEDS},
                             "optimisation": {"first": OPT_SEEDS[0], "n": SEEDS}}
    assert prov["update_file"] is None and prov["update_flags"] is None and prov["n_cells"] == N_CELLS
    bare = run(sim_target(HOST), StubModel([(HOST, 100)]))["provenance"]
    assert bare["model"] is None and bare["model_sha256"] is None
    for row in report["candidates"]:
        iv = row["intervals_90"]
        assert {"s", "c", "w"} <= set(iv) and all(len(v) == 2 and v[0] <= v[1] for v in iv.values())
        assert iv["c"] == row["density_90"]
    json.dumps(report, allow_nan=False)


def test_the_check_names_the_host_it_simulated(monkeypatch):
    (c,) = run(sim_target(HOST), StubModel([(HOST, 100)]), refine_budget=0)["candidates"]
    assert c["check"]["seeds"] == "verify" and c["check"]["host"] == "parameters"
    _spied(monkeypatch, _refined_host(2.5), True)
    (c,) = run(np.ones(416, np.float32), StubModel([(HOST, 100)]))["candidates"]
    assert c["check"]["seeds"] == "final" and c["check"]["host"] == "refined_parameters"


def test_update_mismatch_compares_the_settings_and_the_model():
    flags = {"no_refine": False, "refine_budget": 200, "n_samples": 1000, "seeds": None, "n_cells": None}
    fitted = {"kappa": 1.0, "log_L0": 0.0, "flags": flags, "model_sha256": "aa"}
    assert design_mod.update_mismatch(fitted, flags, "aa") == {}
    assert design_mod.update_mismatch(fitted, flags | {"refine_budget": 6}, "aa") == {"refine_budget": (200, 6)}
    assert design_mod.update_mismatch(fitted, flags, "bb") == {"model_sha256": ("aa", "bb")}
    assert design_mod.update_mismatch({"kappa": 1.0}, flags, "aa") == {"flags": ("none recorded", "needed"),
                                                                      "model_sha256": ("none recorded", "aa")}


def test_design_supports_drop_patterns():
    model = StubModel([(HOST, 100)])
    target = np.ones(416, np.float32)
    rep = design(target, model, n_samples=100, seeds_override=SEEDS, refine_budget=0,
                 n_cells=N_CELLS, drop_patterns=("CHK",))
    assert "candidates" in rep and len(rep["candidates"]) >= 1

