# tests/meaning/test_metrics.py
import numpy as np
import pytest

from meaning.metrics import evaluate_embedding


def _cloud(rng, centre, n=50, s=0.05):
    return (np.asarray(centre, float) + s * rng.standard_normal((n, 2))).astype(np.float32)


def _toy(rng):
    centres = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    refs = [_cloud(rng, c) for c in centres]
    known = [_cloud(rng, c) for c in centres]
    untrained = {"near": _cloud(rng, (0.4, 0.0))}              # between a and b, closer to a; no ties
    unseen = {"far": _cloud(rng, (5.0, 4.0))}                   # distances to a, b, c: 6.40, 5.66, 5.83
    clean = {"near": np.array([0.4, 0.6, 1.1]), "far": np.array([6.4, 5.7, 5.8])}
    return refs, known, untrained, unseen, clean


def test_toy_identification_ordering_placement_and_grading(rng):
    refs, known, untrained, unseen, clean = _toy(rng)
    res = evaluate_embedding(lambda X: X, refs, known, untrained, unseen, clean, ["a", "b", "c"], k=5)
    assert res["known_identification_pct"] == 100.0
    r = res["median_nearest_distance_relative_to_known"]
    assert r["known"] == 1.0 and 1.0 < r["near"] < r["far"]
    assert res["auroc_untrained_vs_known"] > 0.95
    assert res["auroc_unseen_vs_untrained"] == 1.0
    assert {name for name, _ in res["placement"]["near"]} <= {"a", "b"}
    assert res["spearman_embedding_vs_clean_distance"]["near"] == pytest.approx(1.0)
    assert res["spearman_embedding_vs_clean_distance"]["far"] == pytest.approx(1.0)


def test_non_finite_embedding_is_refused(rng):
    refs, known, untrained, unseen, clean = _toy(rng)
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_embedding(lambda X: np.full_like(X, np.nan), refs, known, untrained, unseen, clean, ["a", "b", "c"])


def test_missing_clean_distance_is_refused(rng):
    refs, known, untrained, unseen, clean = _toy(rng)
    del clean["far"]
    with pytest.raises(ValueError, match="far"):
        evaluate_embedding(lambda X: X, refs, known, untrained, unseen, clean, ["a", "b", "c"])
