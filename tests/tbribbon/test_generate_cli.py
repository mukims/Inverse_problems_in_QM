# tests/tbribbon/test_generate_cli.py
import pytest
from tbribbon.generate_clouds import _parse_densities, _parse_models


def test_densities_are_sorted_and_validated():
    assert _parse_densities("0.01, 0.005,0.04") == [0.005, 0.01, 0.04]
    with pytest.raises(ValueError, match="duplicate"):
        _parse_densities("0.01,0.010")
    with pytest.raises(ValueError, match="between 0 and 1"):
        _parse_densities("0.0,0.01")


def test_pilot_grid_gives_distinct_impurity_counts_for_n9():
    from tbribbon.materials import make_model
    grid = _parse_densities(",".join(f"{0.0025 * k:.4f}" for k in range(1, 25)))
    for mid in ("hbn/armchair/N9", "phosphorene/armchair/N9", "mos2/zigzag/N9", "triangular/zigzag/N9"):
        m = _parse_models(mid)[0]
        counts = [m.impurities_for_density(d) for d in grid]
        assert len(set(counts)) == len(counts), mid


def test_models_parse_and_bad_ids_fail_early():
    ms = _parse_models("hbn/armchair/N9, mos2/zigzag/N9")
    assert [m.model_id for m in ms] == ["hbn/armchair/N9", "mos2/zigzag/N9"]
    with pytest.raises(ValueError, match="material/edge/N"):
        _parse_models("hbn/armchair/9")
