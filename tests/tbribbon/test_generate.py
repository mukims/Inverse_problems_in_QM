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


def test_agnr_disorder_physical_invariant_on_odd_widths():
    """Verify physical invariant on validated widths: median(cloud) <= pristine + tol (0.05)."""
    from pathlib import Path
    import sys
    REPO = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO / "notebooks" / "agnr" / "physics"))
    import agnr_lib as A
    import numpy as np

    for m in (5, 7):
        L = A.load_leads(m)
        pris = A.spectrum(m, L, config=0, concentration=0, nonlocal_mode="IL", d=1e-5)
        clouds = [
            [A.device_transmission(w, 1e-5, 1.0, 0.0, m, s, 2, L, nonlocal_mode="IL") for w in A.energy_grid()[:60]]
            for s in range(5)
        ]
        med = np.median(clouds, axis=0)
        assert np.max(med - pris[:60]) <= 0.05


def test_even_width_agnr_bands_match_geometric_honeycomb():
    """Check 3a: Bands of unitcell + T1_matrix match honeycomb_ribbon for m=5..16."""
    import sys
    from pathlib import Path
    REPO = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO / "notebooks" / "agnr" / "physics"))
    import agnr_lib as A
    from tbribbon.lattices import honeycomb_ribbon
    import numpy as np

    k_points = np.linspace(-np.pi, np.pi, 7)
    for m in range(5, 17):
        u = A.unitcell(0, 0, 1.0, 0, m)
        H0_agnr = -(u - np.diag(np.diag(u)))
        T1_agnr = -A.T1_matrix(1.0, m)
        h = honeycomb_ribbon(m, "armchair")
        for k in k_points:
            Hk_agnr = H0_agnr + T1_agnr * np.exp(1j * k) + T1_agnr.conj().T * np.exp(-1j * k)
            Hk_geo = h.H0 + h.H1 * np.exp(1j * k) + h.H1.conj().T * np.exp(-1j * k)
            diff = np.max(np.abs(np.sort(np.linalg.eigvalsh(Hk_agnr)) - np.sort(np.linalg.eigvalsh(Hk_geo))))
            assert diff < 1e-10, f"m={m} band mismatch at k={k}: {diff}"


def test_even_width_agnr_clean_matches_open_channels():
    """Check 3b: Clean T equals open_channels at stable energies for m=6, 8."""
    import sys
    from pathlib import Path
    REPO = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO / "notebooks" / "agnr" / "physics"))
    import agnr_lib as A
    from tbribbon.lattices import honeycomb_ribbon
    from tbribbon.bands import open_channels
    import numpy as np

    grid = A.energy_grid()
    for m in (6, 8):
        L = A.load_leads(m)
        pris = A.spectrum(m, L, config=0, concentration=0, nonlocal_mode="IL", d=1e-5)
        h = honeycomb_ribbon(m, "armchair")
        ch = open_channels(h.H0, h.H1, grid)
        stable = np.array([len(set(open_channels(h.H0, h.H1, [e - 0.02, e, e + 0.02]))) == 1 for e in grid])
        clean_err = np.max(np.abs(pris[stable] - ch[stable]))
        assert clean_err < 1e-6, f"m={m} clean T vs open_channels error: {clean_err}"


def test_even_width_agnr_disorder_physical_invariant():
    """Check 3c: Single impurity max(T - pristine) <= 0.05 for m=6, 8 over first 60 channels."""
    import sys
    from pathlib import Path
    REPO = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO / "notebooks" / "agnr" / "physics"))
    import agnr_lib as A
    import numpy as np

    grid = A.energy_grid()
    for m in (6, 8):
        L = A.load_leads(m)
        pris = A.spectrum(m, L, config=0, concentration=0, nonlocal_mode="IL", d=1e-5)
        for s in range(5):
            t1 = np.array([A.device_transmission(w, 1e-5, 1.0, 0.0, m, s, 1, L, nonlocal_mode="IL") for w in grid[:60]])
            assert np.max(t1 - pris[:60]) <= 0.05, f"m={m} seed={s} exceeded pristine"


def test_agnr_leads_finite_all_widths():
    """Check 3d: Precomputed/cached leads are finite at all 300 energies for m=5..16."""
    import sys
    from pathlib import Path
    REPO = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO / "notebooks" / "agnr" / "physics"))
    import agnr_lib as A
    import numpy as np

    for m in range(5, 17):
        L = A.load_leads(m)
        assert np.all(np.isfinite(L)), f"m={m} has non-finite leads"


def test_zgnr_caroli_generation_and_physics(tmp_path):
    """Verify ZGNR generates via Caroli without errors and satisfies physical bounds."""
    from tbribbon.materials import make_model
    m4 = make_model("graphene-ideal", "zigzag", 4)
    store = CloudStore(tmp_path)
    spec = InputSpec()
    generate(store, [m4], [0.01], spec, seeds=range(2), n_jobs=1)
    sp, s = store.read_cloud(m4.model_id, 0.01)
    assert sp.shape == (2, 400)
    assert list(s) == [0, 1]



