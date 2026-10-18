# tests/atlas/test_eval_lookup.py
from eval_lookup import width_errors


def test_width_errors_counts_only_the_right_material_and_edge():
    recs = [{"true": "mos2/armchair/N12", "dev": "mos2/armchair/N13"},
            {"true": "mos2/armchair/N12", "dev": "mos2/armchair/N13"},
            {"true": "mos2/armchair/N12", "dev": "mos2/zigzag/N12"},
            {"true": "mos2/armchair/N12", "dev": "ws2/armchair/N12"},
            {"true": "hbn/zigzag/N9", "dev": "hbn/zigzag/N9"}]
    assert width_errors(recs) == {"mos2/armchair/N12": {"mos2/armchair/N13": 2}}
