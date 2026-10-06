"""Material catalogue. Real materials (D3) are appended here as new entries."""
from functools import lru_cache
from atlaslib.registry import RibbonModel

from .bands import band_edges
from .lattice2d import LATTICES, lattice_ribbon
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
    # MATERIALS-2 (2026-10-04): Liu et al. PRB 88, 085433 (2013) Table II, GGA, nearest neighbour; V = 0.5 x t2
    "ws2": {"builder": "mos2", "params": {"eps1": 1.130, "eps2": 2.275, "t0": -0.206, "t1": 0.567, "t2": 0.536,
                                          "t11": 0.286, "t12": 0.384, "t22": -0.061},
            "t_ev": 1.0, "orbitals_per_site": 3, "impurity_v_t": 0.268, "source": "Liu 2013 3-band GGA; V = 0.5 x t2"},
    "mose2": {"builder": "mos2", "params": {"eps1": 0.919, "eps2": 2.065, "t0": -0.188, "t1": 0.317, "t2": 0.456,
                                            "t11": 0.211, "t12": 0.290, "t22": 0.130},
              "t_ev": 1.0, "orbitals_per_site": 3, "impurity_v_t": 0.228, "source": "Liu 2013 3-band GGA; V = 0.5 x t2"},
    "wse2": {"builder": "mos2", "params": {"eps1": 0.943, "eps2": 2.179, "t0": -0.207, "t1": 0.457, "t2": 0.486,
                                           "t11": 0.263, "t12": 0.329, "t22": 0.034},
             "t_ev": 1.0, "orbitals_per_site": 3, "impurity_v_t": 0.243, "source": "Liu 2013 3-band GGA; V = 0.5 x t2"},
    # Liu, Jiang, Yao PRB 84, 195430 (2011): t = 2 hbar v_F / (sqrt3 a) with first-principles v_F (Table I, eq. 43)
    "silicene": {"builder": "honeycomb", "params": {"t": 1.0}, "t_ev": 1.067,
                 "source": "NN pz, no SOC or buckling; t from v_F = 5.42e5 m/s, a = 3.86 A (Liu 2011)"},
    "germanene": {"builder": "honeycomb", "params": {"t": 1.0}, "t_ev": 0.991,
                  "source": "NN pz, no SOC or buckling; t from v_F = 5.24e5 m/s, a = 4.02 A (Liu 2011)"},
    # toy lattices with flat bands, built from lattice data (tbribbon.lattice2d)
    "kagome": {"builder": "lattice", "params": {}, "t_ev": 1.0, "source": "idealised kagome, flat band at +2t"},
    "lieb": {"builder": "lattice", "params": {}, "t_ev": 1.0, "source": "idealised Lieb, flat band at 0"},
    "checkerboard": {"builder": "lattice", "params": {}, "t_ev": 1.0,
                     "source": "idealised checkerboard (planar pyrochlore), flat band at +2t"},
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
    if spec["builder"] == "lattice":
        return lattice_ribbon(LATTICES[material], width, edge, **spec["params"])
    raise KeyError(spec["builder"])


def hamiltonian_for(model):
    return _build(model.material, model.edge, model.width)


def make_model(material, edge, width):
    spec = MATERIALS[material]
    h = _build(material, edge, width)
    return RibbonModel(material, edge, width, spec["t_ev"], h.H0.shape[0],
                       round(band_edges(h.H0, h.H1)[1], 3), impurity_v_t=spec.get("impurity_v_t", 0.5),
                       orbitals_per_site=spec.get("orbitals_per_site", 1), source=spec["source"])


# The three-band TMD model keeps only the metal atoms; each carries two chalcogens in the real crystal.
CHALCOGENS_PER_METAL = {"mos2": 2, "ws2": 2, "mose2": 2, "wse2": 2}


def atoms_per_cell(model):
    """Real atoms in one unit cell of the ribbon, chalcogens included."""
    return model.sites_per_cell // model.orbitals_per_site * (1 + CHALCOGENS_PER_METAL.get(model.material, 0))


@lru_cache(maxsize=None)
def device_cells(material, edge, target_atoms, ref_widths=(7, 9)):
    """One device length per material and edge: the cells that put its N = 7-9 devices near target_atoms."""
    per_cell = sum(atoms_per_cell(make_model(material, edge, n)) for n in ref_widths) / len(ref_widths)
    return max(1, round(target_atoms / per_cell))

