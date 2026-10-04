import dataclasses

# tests/atlas/test_lookup.py
import numpy as np
import pytest

from atlaslib import InputSpec
from atlaslib.lookup import Catalogue, P_NO_MATCH, calibrate, candidates, lookup, window_input, window_inputs
from atlaslib.registry import RibbonModel
from atlaslib.store import CloudStore
from toy import E, toy_spectrum


SPEC = InputSpec(version="v4")
DENSITIES = (0.01, 0.02, 0.04)
SPLIT = dict(train_max=99, val=(100, 169))   # toy seeds 0-199: train 0-99, validation 100-169, test 170-199
TEST = np.arange(170, 200)
LEVEL = {"alpha": 1.0, "beta": 4.0}   # beta at 4% must not look like alpha at 0%: 4 e^-1.2 > 1


def _spectrum(material, width, density, seed):
    return toy_spectrum(LEVEL[material], width, density, int(seed + density * 1e5))


def _toy_lookup_store(root):
    store, models = CloudStore(root), []
    for name in LEVEL:
        for w in (7, 9, 14):
            m = RibbonModel(name, "armchair", w, 1.0, 2 * w, 3.0)
            models.append(m)
            store.write_pristine(m.model_id, E, toy_spectrum(LEVEL[name], w, 0.0, 10**6))
            for d in DENSITIES:
                seeds = np.arange(200)
                store.write_cloud(m.model_id, d, m.impurities_for_density(d),
                                  np.stack([_spectrum(name, w, d, s) for s in seeds]), seeds, E)
    return store, models


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    store, models = _toy_lookup_store(tmp_path_factory.mktemp("lookup") / "store")
    return store, models, Catalogue.build(store, models, SPEC, **SPLIT)


def _items(models, densities=DENSITIES, seeds=TEST[:10]):
    for m in models:
        for d in densities:
            for s in seeds:
                yield m, d, _spectrum(m.material, m.width, d, s)


# ---------- Task 1: window-aware transform ----------

def test_window_input_matches_spec_transform_on_covered_channels():
    T = toy_spectrum(1.0, 9, 0.02, 7)
    x, mask = window_input(SPEC, E, T)
    assert mask.sum() == 200 and mask[:200].all() and not mask[200:].any()
    assert np.allclose(x[mask], SPEC.to_input(T, E, 4.0)[0][mask], atol=1e-7)
    assert np.all(x[~mask] == 0)


def test_window_inputs_batch_equals_single():
    T = np.stack([toy_spectrum(1.0, 9, 0.02, s) for s in range(3)])
    X, _ = window_inputs(SPEC, E, T)
    for i in range(3):
        assert np.array_equal(X[i], window_input(SPEC, E, T[i])[0])


def test_energies_beyond_the_axis_are_ignored():
    e = np.round(np.arange(0, 12.0001, 0.01), 6)
    T = np.interp(e, E, toy_spectrum(1.0, 9, 0.02, 7), right=0.0)
    _, mask = window_input(SPEC, e, T)
    assert mask.all()


def test_a_single_spike_is_removed():
    T = toy_spectrum(1.0, 9, 0.02, 7)
    spiked = T.copy()
    spiked[150] = 400.0
    assert np.max(np.abs(window_input(SPEC, E, spiked)[0] - window_input(SPEC, E, T)[0])) < 0.05


@pytest.mark.parametrize("energies, T, message", [
    (E - 0.01, toy_spectrum(1.0, 9, 0.02, 7), "0 eV"),
    (E[::-1], toy_spectrum(1.0, 9, 0.02, 7), "increase"),
    (E, np.full(E.size, np.nan), "finite"),
    (E[:5], np.ones(5), "at least"),
    (E, np.ones(E.size - 1), "values per spectrum"),
])
def test_bad_signatures_are_rejected(energies, T, message):
    with pytest.raises(ValueError, match=message):
        window_input(SPEC, energies, T)


def test_a_units_of_t_spec_is_refused():
    with pytest.raises(ValueError, match="eV-axis"):
        window_input(InputSpec(version="v2"), E, toy_spectrum(1.0, 9, 0.02, 7))


# ---------- Task 2: catalogue ----------

def test_catalogue_reproduces_stored_ensembles_and_clean_anchor(toy):
    store, models, cat = toy
    assert cat.ids == sorted(m.model_id for m in models)
    for k, m in enumerate(cat.models):
        e, pris = store.read_pristine(m.model_id)
        assert np.allclose(cat.mu[k, 0], window_input(SPEC, e, pris)[0], atol=1e-6)
        assert np.all(cat.sd[k, 0] == 0)
        for d in DENSITIES:
            c, s = store.read_cloud(m.model_id, d)
            tr = window_inputs(SPEC, e, c[s <= 99])[0]
            j = int(np.argmin(np.abs(cat.grid - d)))
            assert np.allclose(cat.mu[k, j], np.median(tr, axis=0), atol=1e-6)
            assert np.allclose(cat.sd[k, j], tr.std(axis=0), atol=1e-6)


def test_catalogue_grid_and_validation_split(toy):
    _, _, cat = toy
    assert cat.grid.size == 101 and cat.grid[0] == 0.0 and abs(cat.grid[-1] - 0.05) < 1e-12
    assert len(cat.val_x) == 6 * 3 * 70
    assert set(np.unique(cat.val_conc)) == set(DENSITIES)


def test_without_drops_a_material_and_its_validation_spectra(toy):
    _, _, cat = toy
    sub = cat.without(["beta"])
    assert [m.material for m in sub.models] == ["alpha"] * 3
    assert len(sub.val_x) == 3 * 3 * 70 and sub.val_dev.max() == 2
    with pytest.raises(ValueError, match="no catalogued device"):
        cat.without(["gamma"])


def test_catalogue_save_load_round_trip(toy, tmp_path):
    _, _, cat = toy
    cat.save(tmp_path / "cat")
    back = Catalogue.load(tmp_path / "cat")
    assert back.ids == cat.ids and back.kappa == cat.kappa and back.spec == cat.spec
    for name in ("grid", "mu", "sd", "val_x", "val_dev", "val_conc"):
        assert np.array_equal(getattr(back, name), getattr(cat, name))


def test_build_refuses_a_units_of_t_spec(toy):
    store, models, _ = toy
    with pytest.raises(ValueError, match="eV-axis"):
        Catalogue.build(store, models, InputSpec(version="v2"), **SPLIT)


# ---------- Task 3: lookup ----------

def test_lookup_names_device_and_concentration(toy):
    _, models, cat = toy
    items = list(_items(models))
    right, rel = 0, []
    for m, d, T in items:
        r = lookup(cat, E, T)
        if r.device == m.model_id:
            right += 1
            rel.append(abs(r.concentration - d) / d)
    assert right / len(items) >= 0.95
    assert np.median(rel) <= 0.15


def test_match_fields(toy):
    _, _, cat = toy
    r = lookup(cat, E, _spectrum("alpha", 9, 0.02, 170))
    assert r.device == f"{r.material}/{r.edge}/N{r.width}"
    assert 0 < r.probability <= 1 and len(r.runners_up) == 4
    assert abs(r.probability + sum(p for _, p in r.runners_up) - 1) < 1e-6
    assert r.concentration_lo <= r.concentration <= r.concentration_hi
    assert r.window == (0.0, 3.98, 200) and r.impurity_layout is None
    assert r.no_match == (r.p_value < P_NO_MATCH)


def test_prescreen_keeps_the_true_device(toy):
    _, models, cat = toy
    for m, d, T in _items(models, seeds=TEST[:5]):
        x, mask = window_input(SPEC, E, T)
        devs, _ = candidates(cat, x, mask, top_k=2)
        assert cat.ids.index(m.model_id) in devs


def test_channels_outside_the_window_do_not_matter(toy):
    _, _, cat = toy
    T = _spectrum("alpha", 9, 0.02, 171)
    w = E <= 2.0
    r = lookup(cat, E[w], T[w])
    _, mask = window_input(SPEC, E[w], T[w])
    mu = cat.mu.copy()
    mu[:, :, ~mask] = np.random.default_rng(0).random(mu[:, :, ~mask].shape)
    assert lookup(dataclasses.replace(cat, mu=mu), E[w], T[w]) == r
    assert r.window[2] == int(mask.sum())


def test_left_out_material_is_no_match(toy):
    _, models, cat = toy
    sub = cat.without(["beta"])
    flags = [lookup(sub, E, T).no_match for m, d, T in _items([m for m in models if m.material == "beta"])]
    assert np.mean(flags) >= 0.9


def test_own_device_p_values_are_not_small(toy):
    _, models, cat = toy
    p = np.array([lookup(cat, E, T).p_value for m, d, T in _items(models)])
    assert np.mean(p < P_NO_MATCH) <= 0.05
    assert np.mean(p < 0.1) <= 0.3


def test_an_empty_window_is_not_a_confident_match(toy):
    _, _, cat = toy
    e = np.round(np.arange(3.5, 3.985, 0.01), 6)
    assert lookup(cat, e, np.zeros(e.size)).probability < 0.6


def test_a_coarse_grid_signature_is_still_identified(toy):
    _, _, cat = toy
    right = sum(lookup(cat, E[::10], _spectrum("alpha", 9, 0.02, s)[::10]).device == "alpha/armchair/N9" for s in TEST[:10])
    assert right >= 8


def test_a_spiked_signature_gives_the_same_answer(toy):
    _, _, cat = toy
    T = _spectrum("alpha", 9, 0.02, 172)
    spiked = T.copy()
    spiked[150] = 400.0
    r, r2 = lookup(cat, E, T), lookup(cat, E, spiked)
    assert r2.device == r.device and abs(r2.concentration - r.concentration) <= 0.0005 + 1e-12


def test_beyond_the_grid_reports_the_top_of_the_range(toy):
    _, _, cat = toy
    r = lookup(cat, E, toy_spectrum(1.0, 9, 0.08, 12345))
    assert np.isfinite(r.probability) and 0.04 <= r.concentration <= cat.grid[-1]


def test_saved_catalogue_gives_the_same_match(toy, tmp_path):
    _, _, cat = toy
    cat.save(tmp_path / "cat")
    T = _spectrum("alpha", 9, 0.02, 170)
    assert lookup(Catalogue.load(tmp_path / "cat"), E, T) == lookup(cat, E, T)


# ---------- Task 4: calibration ----------

def test_calibrate_reaches_target_coverage_on_validation(toy):
    _, _, cat = toy
    c = dataclasses.replace(cat)
    res = calibrate(c, per_device=40)
    assert c.kappa == res["kappa"] >= 1.0
    assert 0.9 <= res["coverage"] <= 1.0 and res["n"] == 6 * 40


def test_larger_kappa_widens_the_interval(toy):
    _, _, cat = toy
    T = _spectrum("alpha", 9, 0.02, 173)
    narrow = lookup(dataclasses.replace(cat, kappa=1.0), E, T)
    wide = lookup(dataclasses.replace(cat, kappa=8.0), E, T)
    assert wide.concentration_hi - wide.concentration_lo >= narrow.concentration_hi - narrow.concentration_lo


