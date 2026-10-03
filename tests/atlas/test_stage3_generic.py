# tests/atlas/test_stage3_generic.py
import numpy as np
from atlaslib.store import CloudStore
from stage3_generic import evaluate_ribbon, fit_ribbon, split_masks

E = np.arange(200) * 0.02


def _store(tmp_path, n=100):
    st, mid = CloudStore(tmp_path / "s"), "toy/armchair/N9"
    pris = 1.0 + np.floor(E)
    st.write_pristine(mid, E, pris)
    for k in range(1, 13):
        d = 0.005 * k
        rng = np.random.default_rng(k)
        c = pris[None] * np.exp(-d * 30 * (1 + 0.05 * rng.standard_normal((n, 1)))) * (1 + 0.01 * rng.standard_normal((n, E.size)))
        st.write_cloud(mid, d, k, np.clip(c, 0, None), np.arange(n), E)
    return st, mid


def test_split_masks_do_not_overlap():
    s = np.arange(100)
    tr, ca, te = split_masks(s, train_max=69, val_max=84)
    assert tr.sum() == 70 and ca.sum() == 15 and te.sum() == 15
    assert not (tr & ca).any() and not (ca & te).any() and not (tr & te).any()


def test_oracle_estimates_density_with_calibrated_intervals(tmp_path):
    st, mid = _store(tmp_path)
    fitted = fit_ribbon(st, mid, train_max=69, val_max=84, n_estimators=200, threads=2)
    res = evaluate_ribbon(fitted, st, mid, train_max=69, val_max=84)
    assert res["oracle"]["median_rel_err"] < 0.15
    assert 0.75 <= res["oracle"]["coverage"] <= 1.0
    assert len(res["oracle"]["per_density"]) == 12 and "shazam" not in res
