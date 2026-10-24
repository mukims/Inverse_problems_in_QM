"""Design pipeline and report (spec sections 7 and 8, output)."""
import json

import numpy as np
import pytest
import torch

from shazam.inverse import candidates as candidates_mod
from shazam.inverse import design as design_mod
from shazam.inverse import refine as refine_mod
from shazam.inverse.design import design, target_input
from shazam.inverse.encoding import encode
from tbribbon.parent import HexParams, named_material
from tbribbon.parent_data import V4, simulate_raw, to_input
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
    (tmp_path / "update.json").write_text(json.dumps({"kappa": 2.0, "log_L0": -1e9}))
    (tmp_path / "model.pt").write_text("stub")
    cli.main(["--target", str(tmp_path / "T.npy"), "--energies", str(tmp_path / "E.npy"), "--model", str(tmp_path / "model.pt"),
              "--n-samples", "100", "--n-jobs", "1"])
    report = json.loads((tmp_path / "T_design.json").read_text())
    assert report["update_fitted"] is True and report["update"] == {"kappa": 2.0, "log_L0": -1e9}
    assert not list(tmp_path.glob("*.tmp"))
