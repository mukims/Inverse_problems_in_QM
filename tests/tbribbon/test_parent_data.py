import numpy as np, pytest
from tbribbon.parent import named_material, HexParams
from tbribbon.parent_prior import Draw
from tbribbon.parent_data import band_weights, clean_channels, simulate_raw, to_input, reject_reason, n_impurities

G = Draw(named_material("graphene-ideal"), "armchair", 6, 0.02)

def test_band_weights_sum_to_one_and_single_type_is_all_weight():
    assert band_weights(named_material("kagome")) == {"K": pytest.approx(1.0)}
    w = band_weights(HexParams(on_C=True, on_H=True, on_K=False, lead="H", s=2.0, t_AB=1, t_CC=0.5, t_CH=0.4))
    assert set(w) == {"C", "H"} and sum(w.values()) == pytest.approx(1.0)

def test_simulated_spectrum_never_exceeds_the_clean_channels_much():
    raw = simulate_raw(G, n_imp=6, seed=1, n_cells=20)
    assert raw.shape == (416,) and np.all(np.isfinite(raw))
    assert np.mean(np.minimum(raw, clean_channels(G) + 1) - clean_channels(G)) <= 0.05

def test_reject_reasons():
    zeros, some = np.zeros(416), np.r_[np.ones(50), np.zeros(366)]
    assert reject_reason(G, zeros, zeros, {"H": 1.0}) == "no_channel"
    assert reject_reason(G, some, some, {"H": 0.995, "C": 0.005}) == "invisible_type"
    assert reject_reason(G, some, np.r_[np.ones(5), np.zeros(411)], {"H": 1.0}) == "few_channels"
    assert reject_reason(G, some, some, {"H": 1.0}) is None

def test_input_is_the_v4_transform():
    x = to_input(simulate_raw(G, 6, 1, n_cells=20))
    assert x.shape == (416,) and x.dtype == np.float32 and 0 <= x.min() and x.max() <= 1


def test_band_weights_put_a_decoupled_flat_band_on_its_own_type():
    # one flat C state at 1.5 s against one in-window honeycomb state per k: equal weights; a swapped row map fails
    w = band_weights(HexParams(on_C=True, on_H=True, on_K=False, lead="H", s=2.0, eps_C=1.5, t_AB=1, t_CC=1e-9, t_CH=1e-4))
    assert w == {"C": pytest.approx(0.5, abs=0.01), "H": pytest.approx(0.5, abs=0.01)}


def test_impurity_count_follows_the_density_over_the_whole_device():
    assert n_impurities(G, 125) == round(0.02 * 125 * 12) and n_impurities(Draw(G.params, "armchair", 6, 1e-6), 10) == 1


def test_passing_the_ribbon_and_leads_gives_the_same_spectrum_bit_for_bit():
    from tbribbon.parent_data import host_leads
    rb, lc = host_leads(G)
    for seed in (1, 5):
        assert np.array_equal(simulate_raw(G, 6, seed, n_cells=20, rb=rb, lc=lc), simulate_raw(G, 6, seed, n_cells=20))
