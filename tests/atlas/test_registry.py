import pytest
from atlaslib.registry import Registry, RibbonModel


def agnr(n):
    return RibbonModel("graphene-ideal", "armchair", n, 1.0, 2 * n, 3.0, source="legacy")


def test_model_id_and_sites():
    m = agnr(7)
    assert m.model_id == "graphene-ideal/armchair/N7"
    assert m.n_sites == 1400
    assert m.impurities_for_density(0.01) == 14


def test_invalid_edge_or_width_rejected():
    with pytest.raises(ValueError):
        RibbonModel("x", "diagonal", 7, 1.0, 14, 3.0)
    with pytest.raises(ValueError):
        RibbonModel("x", "armchair", 0, 1.0, 14, 3.0)


def test_duplicate_refused_and_json_round_trip(tmp_path):
    r = Registry([agnr(7), agnr(9)])
    with pytest.raises(ValueError):
        r.add(agnr(7))
    r.save(tmp_path / "reg.json")
    r2 = Registry.load(tmp_path / "reg.json")
    assert r2.ids() == r.ids() and r2.get("graphene-ideal/armchair/N9") == agnr(9)


def test_hierarchy_groups_widths():
    r = Registry([agnr(9), agnr(7), RibbonModel("square", "strip", 10, 1.0, 10, 3.92)])
    assert r.hierarchy() == {"graphene-ideal": {"armchair": [7, 9]}, "square": {"strip": [10]}}
