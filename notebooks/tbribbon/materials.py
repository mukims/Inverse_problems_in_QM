"""Material catalogue. Real materials (D3) are appended here as new entries."""
from atlaslib.registry import RibbonModel

from .bands import band_edges
from .lattices import honeycomb_ribbon, square_strip

MATERIALS = {
    "graphene-ideal": {"builder": "honeycomb", "params": {"t": 1.0}, "t_ev": 1.0, "source": "idealised nearest-neighbour"},
    "square": {"builder": "square", "params": {"t": 1.0}, "t_ev": 1.0, "source": "idealised square strip"},
}


def _build(material, edge, width):
    spec = MATERIALS[material]
    if spec["builder"] == "honeycomb":
        return honeycomb_ribbon(width, edge, **spec["params"])
    if spec["builder"] == "square":
        return square_strip(width, **spec["params"])
    raise KeyError(spec["builder"])


def hamiltonian_for(model):
    return _build(model.material, model.edge, model.width)


def make_model(material, edge, width):
    spec = MATERIALS[material]
    h = _build(material, edge, width)
    return RibbonModel(material, edge, width, spec["t_ev"], h.H0.shape[0],
                       round(band_edges(h.H0, h.H1)[1], 3), source=spec["source"])
