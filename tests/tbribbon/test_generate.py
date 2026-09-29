from atlaslib import CloudStore, InputSpec, RibbonModel
from tbribbon.generate_clouds import generate, seeds_for_width


def test_seed_budget_by_width():
    assert [seeds_for_width(n) for n in (7, 14, 27, 50)] == [1000, 1000, 300, 100]


def test_generate_writes_then_resumes(tmp_path):
    m = RibbonModel("square", "strip", 3, 1.0, 3, 3.41, n_cells=5)
    spec = InputSpec(e_max_t=4.0, step_t=0.5)
    store = CloudStore(tmp_path)
    wrote = generate(store, [m], [0.2], spec, n_jobs=2, seeds=range(4))
    assert wrote == [(m.model_id, 3 / 15)]
    spectra, seeds = store.read_cloud(m.model_id, 3 / 15)
    assert spectra.shape == (4, 8) and list(seeds) == [0, 1, 2, 3]
    assert generate(store, [m], [0.2], spec, n_jobs=2, seeds=range(4)) == []
