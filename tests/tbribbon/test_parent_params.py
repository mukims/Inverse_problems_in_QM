import pytest
from tbribbon.parent import HexParams, named_material, NAMED

def test_named_materials_are_the_points_of_section_4_3():
    g = named_material("graphene-ideal")
    assert g.pattern == (False, True, False) and g.t_AB == 1 and g.s == 2.7 and g.delta == 0
    h = named_material("hbn")
    assert h.delta == pytest.approx(3.625 / 2.30) and h.eps_A == pytest.approx(-h.eps_B) and h.s == 2.30
    assert named_material("silicene").s == 1.067 and named_material("germanene").s == 0.991
    assert named_material("triangular").pattern == (True, False, False) and named_material("triangular").t_CC == 1
    assert named_material("kagome").pattern == (False, False, True) and named_material("kagome").t_KK == 1
    for n in NAMED:
        named_material(n).validate()

@pytest.mark.parametrize("bad", [
    dict(on_C=False, on_H=False, on_K=False, lead="H", s=1.0),                       # nothing active
    dict(on_C=False, on_H=True, on_K=False, lead="K", s=1.0, t_AB=1),                # lead not active
    dict(on_C=False, on_H=True, on_K=False, lead="H", s=1.0, t_AB=0.8),              # lead hopping not 1
    dict(on_C=False, on_H=True, on_K=False, lead="H", s=1.0, t_AB=1, eps_K=0.3),     # parameter of an inactive type
    dict(on_C=False, on_H=True, on_K=False, lead="H", s=1.0, t_AB=1, delta=-0.1),    # delta < 0
    dict(on_C=True, on_H=True, on_K=False, lead="H", s=1.0, t_AB=1, t_CC=0.5),       # two types, not connected
    dict(on_C=True, on_H=True, on_K=True, lead="H", s=1.0, t_AB=1, t_CC=0.5, t_KK=0.5, t_CH=0.3),  # three types, one bond
])
def test_invalid_hosts_are_rejected(bad):
    with pytest.raises(ValueError):
        HexParams(**bad).validate()
