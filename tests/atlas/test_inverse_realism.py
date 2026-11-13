import dataclasses

import numpy as np
import pytest

from shazam.inverse import realism as rl
from shazam.inverse.data import EDGE_CODES, LEAD_CODES
from tbribbon.parent import NAMED, HexParams, named_material
from tbribbon.parent_prior import FIELDS, Draw


def _draw(params, edge="armchair", width=9, density=0.01):
    return Draw(params, edge, width, density)


def _scaled(name, factor, **changes):
    """The named material with its energy scale multiplied by `factor` and model-unit fields replaced."""
    p = named_material(name)
    return dataclasses.replace(p, s=p.s * factor, **changes)


def test_every_named_material_is_at_distance_zero_from_itself_whatever_the_ribbon():
    for name in NAMED:
        for edge, width, density in (("armchair", 4, 0.0025), ("zigzag", 22, 0.05)):
            got, d = rl.nearest_material(_draw(named_material(name), edge, width, density))
            assert got == name and d == 0.0


def test_distance_is_the_largest_deviation_in_units_of_its_tolerance():
    # graphene's hopping is 2.7 eV: at 10% the tolerance is 0.27 eV, so 5% off is half a tolerance
    assert rl.material_distance(_draw(_scaled("graphene-ideal", 1.05)), "graphene-ideal") == pytest.approx(0.5)
    # a quantity that is zero in the material is judged against the absolute tolerance (0.1 eV): t_AA = 0.05 eV
    d = rl.material_distance(_draw(_scaled("graphene-ideal", 1.0, t_AA=0.05 / 2.7)), "graphene-ideal")
    assert d == pytest.approx(0.5)
    # the worst quantity decides: hopping half a tolerance off, on-site energy 0.15 eV off
    both = _scaled("graphene-ideal", 1.05, eps_H_mean=0.15 / (2.7 * 1.05))
    assert rl.material_distance(_draw(both), "graphene-ideal") == pytest.approx(1.5)
    assert rl.material_distance(_draw(both), "graphene-ideal", rel_tol=0.2, abs_tol=0.3) == pytest.approx(0.5)


def test_a_host_of_another_lattice_type_is_infinitely_far():
    kag = _draw(named_material("kagome"))
    assert rl.material_distance(kag, "graphene-ideal") == np.inf
    two_types = HexParams(True, True, False, "H", 2.7, t_AB=1.0, t_CC=0.5, t_CH=0.3)
    assert rl.nearest_material(_draw(two_types)) == (None, np.inf) and not rl.is_near_known(_draw(two_types))


def test_near_known_is_distance_at_most_one_and_names_the_nearest_of_two_close_materials():
    assert rl.is_near_known(_draw(_scaled("hbn", 1.08)))                    # every quantity 8% off: inside 10%
    assert not rl.is_near_known(_draw(_scaled("hbn", 1.12)))
    assert rl.is_near_known(_draw(_scaled("hbn", 1.12)), rel_tol=0.15)
    assert rl.nearest_material(_draw(_scaled("silicene", 0.99)))[0] == "silicene"      # 1.056 eV: silicene 1.067
    assert rl.nearest_material(_draw(_scaled("silicene", 0.94)))[0] == "germanene"     # 1.003 eV: germanene 0.991
    only = rl.nearest_material(_draw(_scaled("silicene", 0.94)), materials=("graphene-ideal", "silicene"))
    assert only[0] == "silicene" and only[1] == pytest.approx(0.6)


def test_the_unit_type_and_the_ribbon_do_not_enter():
    # the same physical host written with either unit type: t_CC = 1.0 eV, t_AB = 2.0 eV, t_CH = 0.5 eV
    a = HexParams(True, True, False, "H", 2.0, t_AB=1.0, t_CC=0.5, t_CH=0.25)
    b = HexParams(True, True, False, "C", 1.0, t_CC=1.0, t_AB=2.0, t_CH=0.5)
    assert np.allclose(rl.physical_ev(_draw(a)), rl.physical_ev(_draw(b)))
    g = named_material("graphene-ideal")
    assert np.array_equal(rl.physical_ev(_draw(g, "armchair", 5, 0.003)), rl.physical_ev(_draw(g, "zigzag", 21, 0.04)))


def test_rows_of_stored_arrays_give_the_same_distances_as_their_draws():
    hosts = [named_material("graphene-ideal"), _scaled("hbn", 1.05), named_material("kagome"),
             HexParams(True, True, False, "H", 2.7, t_AB=1.0, t_CC=0.5, t_CH=0.3), _scaled("germanene", 1.2)]
    arrays = dict(on=np.array([h.pattern for h in hosts], np.int8),
                  lead=np.array([LEAD_CODES[h.lead] for h in hosts], np.int8),
                  params=np.array([[getattr(h, f) for f in FIELDS] for h in hosts]),
                  edge=np.array([EDGE_CODES["armchair"]] * len(hosts), np.int8))
    names, dist = rl.nearest_material_rows(arrays)
    want = [rl.nearest_material(_draw(h)) for h in hosts]
    assert list(names) == [w[0] for w in want] and dist == pytest.approx([w[1] for w in want])
    assert list(rl.near_known_rows(arrays)) == [True, True, True, False, False]


def test_unknown_material_names_and_bad_tolerances_are_refused():
    with pytest.raises(ValueError):
        rl.material_distance(_draw(named_material("hbn")), "unobtainium")
    with pytest.raises(ValueError):
        rl.nearest_material(_draw(named_material("hbn")), abs_tol=0.0)


# ---------------------------------------------------------------- prior draws restricted to a material's neighbourhood

from tbribbon.parent_prior import in_prior, sample_prior


def test_draws_near_a_material_are_prior_draws_inside_the_radius_and_reproducible():
    for name, radius in (("hbn", 2.0), ("kagome", 1.0), ("graphene-ideal", 2.0)):
        got = [rl.sample_near(np.random.default_rng([3, j]), name, radius) for j in range(12)]
        again = [rl.sample_near(np.random.default_rng([3, j]), name, radius) for j in range(12)]
        assert got == again
        for draw, tries in got:
            assert in_prior(draw) and tries >= 1
            assert rl.material_distance(draw, name) <= radius
        if radius == 1.0:
            assert all(rl.is_near_known(d) for d, _ in got)
    wide = [rl.sample_near(np.random.default_rng([4, j]), "silicene", 2.0)[0] for j in range(60)]
    assert any(rl.material_distance(d, "silicene") > 1.0 for d in wide)        # the halo beyond one tolerance is there


def test_tries_count_the_prior_draws_used_and_the_draw_is_the_first_one_inside():
    rng = np.random.default_rng(11)
    draw, tries = rl.sample_near(rng, "triangular", 2.0)
    replay = np.random.default_rng(11)
    seen = [sample_prior(replay) for _ in range(tries)]
    assert seen[-1] == draw
    assert all(rl.material_distance(d, "triangular") > 2.0 for d in seen[:-1])


def test_sample_near_refuses_unknown_materials_bad_radii_and_gives_up_after_max_tries():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="unknown material"):
        rl.sample_near(rng, "diamond")
    with pytest.raises(ValueError, match="radius"):
        rl.sample_near(rng, "hbn", radius=0.0)
    far = lambda r: _draw(named_material("kagome"))
    with pytest.raises(RuntimeError, match="50 prior draws"):
        rl.sample_near(rng, "hbn", sampler=far, max_tries=50)
