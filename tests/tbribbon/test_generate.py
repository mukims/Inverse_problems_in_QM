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


def test_agnr_lib_reproduces_stored_rows():
    import numpy as np
    import pytest
    from pathlib import Path
    import sys

    REPO = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO / "notebooks" / "agnr" / "physics"))
    import agnr_lib as A

    d7_path = Path("/run/media/shardul/storage/machine_learning/transmission_data/transmission_results/consolidated_data/size_7.npy")
    d9_path = Path("/run/media/shardul/storage/machine_learning/transmission_data/transmission_results/consolidated_data/size_9.npy")
    if not (d7_path.exists() and d9_path.exists()):
        pytest.skip("Consolidated AGNR datasets not found")

    d7 = np.load(d7_path, mmap_mode="r")
    d9 = np.load(d9_path, mmap_mode="r")
    L7 = A.load_leads(7)
    L9 = A.load_leads(9)
    idx = [30, 90, 150, 220]

    for m, d, L, c_val, c_idx, s in [
        (7, d7, L7, 10, 4, 0),
        (7, d7, L7, 40, 19, 123),
        (9, d9, L9, 10, 4, 0),
        (9, d9, L9, 40, 19, 123),
    ]:
        got = [A.device_transmission(k * 0.01, 1e-5, 1.0, 0.0, m, s, c_val, L, nonlocal_mode="IL") for k in idx]
        ref = d[c_idx, s, idx]
        rel_err = np.max(np.abs(np.array(got) - ref) / np.abs(ref))
        assert rel_err < 1e-5, f"size_{m} c={c_val} seed={s} rel_err={rel_err}"

