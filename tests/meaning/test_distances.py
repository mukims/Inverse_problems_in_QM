import numpy as np
import pytest

from meaning.distances import clean_distance_matrix, distances_to, kernel_sigma, rms_distance


def test_matrix_is_symmetric_with_zero_diagonal_and_matches_rms():
    clean = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.0, 2.0, 0.0, 2.0]])
    D = clean_distance_matrix(clean)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0.0)
    assert D[0, 1] == pytest.approx(1.0)
    assert D[0, 2] == pytest.approx(np.sqrt(2.0))
    assert D[1, 2] == pytest.approx(rms_distance(clean[1], clean[2]))


def test_distances_to_matches_matrix_row(rng):
    clean = rng.random((5, 16))
    assert np.allclose(distances_to(clean[2], clean), clean_distance_matrix(clean)[2])


def test_kernel_sigma_is_median_nearest_other_distance():
    D = np.array([[0.0, 1.0, 4.0], [1.0, 0.0, 2.0], [4.0, 2.0, 0.0]])
    assert kernel_sigma(D) == pytest.approx(1.0)


def test_mismatched_channel_counts_are_refused():
    with pytest.raises(ValueError):
        rms_distance(np.zeros(400), np.zeros(300))
    with pytest.raises(ValueError):
        distances_to(np.zeros(300), np.zeros((3, 400)))
