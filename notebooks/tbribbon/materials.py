"""Material catalogue. Real materials (D3) are appended here as new entries."""
from atlaslib.registry import RibbonModel

from .bands import band_edges
from .lattices import honeycomb_ribbon, square_strip, hbn_ribbon, phosphorene_ribbon, mos2_ribbon, triangular_ribbon

MATERIALS = {
    "graphene-ideal": {"builder": "honeycomb", "params": {"t": 1.0}, "t_ev": 2.7,
                       "source": "idealised nearest-neighbour, t = 2.7 eV (Castro Neto 2009)"},
    "square": {"builder": "square", "params": {"t": 1.0}, "t_ev": 1.0, "source": "idealised square strip"},
    "triangular": {"builder": "triangular", "params": {"t": 1.0, "onsite": 0.0}, "t_ev": 1.0, "source": "idealised triangular lattice"},
    "hbn": {"builder": "hbn", "params": {"t": 2.30, "delta": 3.625}, "t_ev": 2.30, "source": "Galvani 2016 GW"},
    "phosphorene": {"builder": "phosphorene", "params": {}, "t_ev": 3.665, "source": "Rudenko 2014 5-hopping"},
    "mos2": {"builder": "mos2", "params": {}, "t_ev": 1.0, "orbitals_per_site": 3, "impurity_v_t": 0.2535,
             "source": "Liu 2013 3-band GGA; V = 0.5 x t2 (0.507 eV)"},
}


def _build(material, edge, width):
    spec = MATERIALS[material]
    if spec["builder"] == "honeycomb":
        return honeycomb_ribbon(width, edge, **spec["params"])
    if spec["builder"] == "square":
        return square_strip(width, **spec["params"])
    if spec["builder"] == "triangular":
        return triangular_ribbon(width, edge, **spec["params"])
    if spec["builder"] == "hbn":
        return hbn_ribbon(width, edge, **spec["params"])
    if spec["builder"] == "phosphorene":
        return phosphorene_ribbon(width, edge, **spec["params"])
    if spec["builder"] == "mos2":
        return mos2_ribbon(width, edge, **spec["params"])
    raise KeyError(spec["builder"])


def hamiltonian_for(model):
    return _build(model.material, model.edge, model.width)


def make_model(material, edge, width):
    spec = MATERIALS[material]
    h = _build(material, edge, width)
    return RibbonModel(material, edge, width, spec["t_ev"], h.H0.shape[0],
                       round(band_edges(h.H0, h.H1)[1], 3), impurity_v_t=spec.get("impurity_v_t", 0.5),
                       orbitals_per_site=spec.get("orbitals_per_site", 1), source=spec["source"])
