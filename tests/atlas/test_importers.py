import numpy as np
from atlaslib.importers import import_consolidated_agnr, import_square_combined, nearest_count
from atlaslib.registry import RibbonModel
from atlaslib.store import CloudStore


def test_nearest_count_prefers_lower_on_tie():
    assert nearest_count(7, [2, 4, 6, 8]) == 6
    assert nearest_count(15, [2, 4, 6, 14, 16]) == 14


def test_agnr_import_records_actual_density(tmp_path, rng):
    arr = rng.random((34, 50, 300))                       # c = 2, 4 ... 68
    np.save(tmp_path / "size_7.npy", arr)
    np.save(tmp_path / "pris.npy", np.ones(300))
    m = RibbonModel("graphene-ideal", "armchair", 7, 1.0, 14, 3.0)
    s = CloudStore(tmp_path / "store")
    got = import_consolidated_agnr(s, m, tmp_path / "size_7.npy", tmp_path / "pris.npy", [0.005, 0.01], np.arange(20))
    assert got[0.005] == 6 / 1400 and got[0.01] == 14 / 1400
    spectra, seeds = s.read_cloud(m.model_id, 6 / 1400)
    assert np.array_equal(spectra, arr[2, :20]) and list(seeds) == list(range(20))


def test_square_import_uses_meta_seeds(tmp_path, rng):
    d = tmp_path / "sq"; d.mkdir()
    for c in (5, 10, 20, 40):
        np.save(d / f"conc_{c}.npy", rng.random((30, 400)))
        np.savetxt(d / f"conc_{c}_meta.csv", np.column_stack([np.arange(30), np.arange(30)[::-1]]),
                   delimiter=",", header="row_idx,config", comments="", fmt="%d")
    np.save(tmp_path / "p.npy", np.ones(400))
    m = RibbonModel("square", "strip", 10, 1.0, 10, 3.92)
    s = CloudStore(tmp_path / "store")
    import_square_combined(s, m, d, tmp_path / "p.npy", [0.01], np.arange(5))
    _, seeds = s.read_cloud(m.model_id, 0.01)
    assert sorted(seeds) == [0, 1, 2, 3, 4]
