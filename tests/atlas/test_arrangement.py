import numpy as np
import pytest

from shazam.arrangement import across_axis, edge_fraction, misfit


def _strip(rows=4, length=10):
    """A toy device: `rows` lines across (y = 0 .. rows-1), `length` atoms along x."""
    x, y = np.meshgrid(np.arange(length, dtype=float), np.arange(rows, dtype=float))
    return np.column_stack([x.ravel(), y.ravel()])


@pytest.mark.parametrize("model_id, across", [("graphene-ideal/armchair/N10", 1), ("graphene-ideal/zigzag/N10", 0),
                                              ("hbn/zigzag/N10", 0), ("mos2/armchair/N10", 0), ("square/strip/N10", 1),
                                              ("kagome/zigzag/N10", 1)])
def test_the_across_axis_comes_from_the_ribbon_period(model_id, across):
    assert across_axis(model_id) == across


def test_edge_fraction_counts_impurities_on_the_two_outermost_lines():
    dev = _strip(rows=4)
    seeds = np.array([[[2, 0], [5, 3]],          # both on an edge
                      [[2, 1], [5, 2]],          # both inside
                      [[2, 0], [5, 2]]],         # one of each
                     dtype=np.float32)
    assert np.allclose(edge_fraction(seeds, dev, across=1), [1.0, 0.0, 0.5])


def test_edge_fraction_works_whichever_column_runs_across():
    dev = _strip(rows=4)
    seeds = np.array([[[2, 0], [5, 3]], [[2, 1], [5, 2]], [[2, 0], [5, 2]]], dtype=np.float32)
    assert np.allclose(edge_fraction(seeds[..., ::-1], dev[:, ::-1], across=0), [1.0, 0.0, 0.5])


@pytest.mark.parametrize("model_id, outer_share", [("hbn/armchair/N10", 2 / 10), ("graphene-ideal/zigzag/N10", 2 / 20)])
def test_on_a_real_ribbon_every_atom_as_an_impurity_gives_the_share_of_edge_atoms(model_id, outer_share):
    from tbribbon.configuration import device_atom_coordinates
    dev = device_atom_coordinates(model_id, n_cells=3)
    assert edge_fraction(dev[None].astype(np.float32), dev, across_axis(model_id))[0] == pytest.approx(outer_share)


def test_misfit_is_the_mean_squared_difference():
    assert misfit([0.1, 0.3], [0.1, 0.3]) == 0.0
    assert misfit([0.0, 0.0], [0.1, 0.3]) == pytest.approx((0.01 + 0.09) / 2)


def test_predicting_the_mean_gives_a_misfit_equal_to_the_variance():
    y = np.random.default_rng(0).uniform(0, 0.5, 200)
    assert misfit(np.full_like(y, y.mean()), y) == pytest.approx(np.var(y))


# ---------------------------------------------------------------- the other arrangement descriptors

from shazam.arrangement import (centre_distance, close_pairs, descriptors, edge_graph_entropy, laplacian_entropy,
                                near_contact, nearest_neighbour_distance, sublattice_fraction)


def test_centre_distance_is_zero_on_the_centre_line_and_half_on_an_edge():
    dev = _strip(rows=5)                                    # lines across at y = 0..4, centre line y = 2
    seeds = np.array([[[3, 2]], [[3, 0]], [[3, 4]]], dtype=np.float32)
    assert np.allclose(centre_distance(seeds, dev, across=1), [0.0, 0.5, 0.5])


def test_sublattice_fraction_is_the_share_on_the_chosen_sublattice():
    assert np.allclose(sublattice_fraction(np.array([[0, 0, 1, 1], [0, 1, 1, 1]])), [0.5, 0.25])


def test_nearest_neighbour_distance_averages_each_impurity_s_closest_partner():
    seeds = np.array([[[0, 0], [1, 0], [3, 0]]], dtype=np.float32)    # closest partners: 1, 1, 2
    assert nearest_neighbour_distance(seeds)[0] == pytest.approx(4 / 3)


def test_close_pairs_counts_pairs_within_one_bond():
    seeds = np.array([[[0, 0], [1, 0], [5, 0]], [[0, 0], [3, 0], [6, 0]]], dtype=np.float32)
    assert list(close_pairs(seeds)) == [1, 0]


def test_near_contact_counts_impurities_close_to_either_lead():
    dev = _strip(rows=3, length=100)                        # x = 0..99 along the ribbon
    seeds = np.array([[[5, 1], [50, 1], [95, 1]]], dtype=np.float32)
    assert near_contact(seeds, dev, across=1)[0] == pytest.approx(2 / 3)


def test_laplacian_entropy_of_one_linked_pair_is_zero():
    assert laplacian_entropy(np.array([[[0, 0], [1, 0]]], dtype=np.float32), length=1.0)[0] == pytest.approx(0.0, abs=1e-12)


def test_laplacian_entropy_of_n_equally_linked_impurities_is_log_n_minus_one():
    s = np.sqrt(3) / 2
    triangle = np.array([[[0, 0], [1, 0], [0.5, s]]])       # all three pairs at the same distance
    assert laplacian_entropy(triangle, length=1.0)[0] == pytest.approx(np.log(2))


def test_two_separate_pairs_have_less_entropy_than_four_impurities_all_linked():
    pairs = np.array([[[0, 0], [1, 0], [100, 0], [101, 0]]], dtype=np.float32)   # two far-apart pairs
    square = np.array([[[0, 0], [1, 0], [0, 1], [1, 1]]], dtype=np.float32)      # four close together
    assert laplacian_entropy(pairs, length=1.0)[0] < laplacian_entropy(square, length=10.0)[0]


def test_laplacian_entropy_ignores_where_the_pattern_sits_and_its_mirror_image():
    rng = np.random.default_rng(0)
    c = rng.uniform(0, 30, size=(1, 12, 2))
    moved = c + np.array([7.0, -3.0])
    mirrored = c * np.array([-1.0, 1.0])
    s = laplacian_entropy(c, length=5.0)[0]
    assert laplacian_entropy(moved, length=5.0)[0] == pytest.approx(s)
    assert laplacian_entropy(mirrored, length=5.0)[0] == pytest.approx(s)


def test_laplacian_entropy_is_zero_when_no_impurities_are_linked():
    far = np.array([[[0, 0], [1000, 0], [2000, 0]]], dtype=np.float32)
    assert laplacian_entropy(far, length=1.0)[0] == 0.0


def test_descriptors_returns_every_label_per_seed():
    dev = _strip(rows=4, length=50)
    coords = np.array([[[2, 0], [5, 3], [30, 1]], [[10, 1], [11, 1], [40, 2]]], dtype=np.float32)
    d = descriptors(coords, np.array([[0, 1, 0], [1, 1, 0]]), dev, across=1, lengths=(2.0, 5.0))
    assert set(d) == {"edge_frac", "centre_dist", "sub0_frac", "nn_dist", "close_pairs", "near_contact",
                      "graph_entropy_l2", "graph_entropy_l5", "edge_graph_entropy_l2", "edge_graph_entropy_l5"}
    assert all(v.shape == (2,) for v in d.values())


# ---------------------------------------------------------------- graph with the two ribbon edges as nodes

def test_one_impurity_midway_between_the_edges_makes_a_three_node_path():
    dev = _strip(rows=3)                                     # edges at y = 0 and y = 2
    one = np.array([[[4, 1]]], dtype=np.float32)            # one bond from each edge: links edge - impurity - edge
    # path of 3 equal links: Laplacian eigenvalues 0, w, 3w, so rho has 1/4 and 3/4
    expected = -(0.25 * np.log(0.25) + 0.75 * np.log(0.75))
    assert edge_graph_entropy(one, dev, across=1, length=1.0)[0] == pytest.approx(expected)


def test_one_impurity_on_an_edge_of_a_wide_ribbon_makes_a_single_link():
    dev = _strip(rows=50)
    one = np.array([[[4, 0]]], dtype=np.float32)            # on one edge, 49 bonds from the other
    assert edge_graph_entropy(one, dev, across=1, length=1.0)[0] == pytest.approx(0.0, abs=1e-9)


def test_far_from_both_edges_the_edge_graph_reduces_to_the_impurity_graph():
    dev = _strip(rows=200, length=400)
    c = np.random.default_rng(3).uniform([100, 90], [300, 110], size=(4, 15, 2))     # all near the centre line
    assert np.allclose(edge_graph_entropy(c, dev, across=1, length=2.0), laplacian_entropy(c, length=2.0))


def test_edge_graph_entropy_ignores_mirroring_across_the_width_and_moving_along_the_ribbon():
    dev = _strip(rows=20, length=200)
    c = np.random.default_rng(4).uniform([10, 0], [190, 19], size=(3, 10, 2))
    s = edge_graph_entropy(c, dev, across=1, length=3.0)
    mirrored = c * np.array([1.0, -1.0]) + np.array([0.0, 19.0])
    moved = c + np.array([5.0, 0.0])
    assert np.allclose(edge_graph_entropy(mirrored, dev, across=1, length=3.0), s)
    assert np.allclose(edge_graph_entropy(moved, dev, across=1, length=3.0), s)


def test_moving_impurities_towards_an_edge_changes_the_edge_graph_entropy():
    dev = _strip(rows=20, length=200)
    centre = np.array([[[50, 10], [60, 10], [70, 10]]], dtype=np.float64)
    edge = centre.copy(); edge[..., 1] = 0.0
    assert edge_graph_entropy(centre, dev, across=1, length=3.0)[0] != pytest.approx(edge_graph_entropy(edge, dev, across=1, length=3.0)[0])
