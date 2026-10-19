# tests/atlas/test_ae_eval.py
import numpy as np

from eval_ae import identify, knn, novelty, standardise, vote


def test_knn_returns_the_nearest_references_and_their_distances():
    refs = np.array([[0, 0], [1, 0], [5, 5], [6, 5]], dtype=np.float32)
    idx, dist = knn(refs, np.array([[0.9, 0.1], [5.4, 5.0]], dtype=np.float32), k=2)
    assert set(idx[0]) == {0, 1} and set(idx[1]) == {2, 3} and dist[0].max() < 1.0


def test_vote_takes_the_most_common_label_and_the_lowest_on_a_tie():
    assert list(vote(np.array([[3, 3, 7], [2, 5, 5], [4, 9, 9], [1, 2, 3]]))) == [3, 5, 9, 1]


def test_standardise_uses_the_reference_statistics_only():
    R, Q = standardise(np.array([[0.0], [2.0]]), np.array([[10.0]]))
    assert np.allclose(R.mean(), 0) and np.allclose(Q, 9.0 / 1.0)


def test_identify_separates_two_clusters_and_reports_per_material():
    rng = np.random.default_rng(0)
    refs_x = np.r_[rng.normal(0, 0.1, (40, 2)), rng.normal(5, 0.1, (40, 2))]
    meta = {"dev": np.r_[np.zeros(40, int), np.ones(40, int)], "dens": np.r_[np.full(40, 0.01), np.full(40, 0.02)],
            "devices": np.array(["hbn/zigzag/N5", "mos2/zigzag/N5"])}
    q = {"dev": np.array([0, 1]), "dens": np.array([0.01, 0.02])}
    out = identify({"s": (refs_x, np.array([[0.0, 0.1], [5.0, 4.9]]))}, meta, q)["s"]
    assert out["device_pct"] == 100 and out["material_pct"] == 100 and out["median_rel_density_err_pct"] == 0
    assert out["by_material_device_pct"] == {"hbn": 100.0, "mos2": 100.0}


def test_novelty_flags_a_cluster_far_from_the_references():
    rng = np.random.default_rng(1)
    R = rng.normal(0, 1, (500, 3))
    res = novelty(R, rng.normal(0, 1, (400, 3)), rng.normal(0, 1, (400, 3)), rng.normal(8, 1, (100, 3)))
    assert res["auroc"] > 0.99 and res["square_flagged_pct"] > 95 and res["known_flagged_pct"] < 5
