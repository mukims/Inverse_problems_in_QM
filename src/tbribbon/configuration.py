"""Microscopic configuration and disorder geometry engine.

Pairs quantum transport spectra with exact physical configuration data:
- Exact 2D spatial coordinates (x, y) of each impurity in the device
- Discrete indices: atom index, unit cell index, site index within unit cell
- Host on-site potential at each impurity site (in units of t and eV)
- Perturbed on-site potential with impurity (V_imp = V_host + v in t and eV)
- Full spatial potential profiles across the device
"""
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from .lattice2d import LATTICES, _vec
from .lattices import RibbonHamiltonian
from atlaslib.registry import RibbonModel
from .materials import MATERIALS, _build

S = np.sqrt(3) / 2

# Approximate lattice constants / bond lengths (in Angstroms) for physical conversions
LATTICE_CONSTANTS_A = {
    "graphene-ideal": 2.46,   # a = 2.46 A, C-C bond = 1.42 A
    "square": 1.0,
    "triangular": 1.0,
    "hbn": 2.50,              # a = 2.50 A, B-N bond = 1.45 A
    "phosphorene": 3.31,      # a1 = 3.31 A, a2 = 4.38 A
    "mos2": 3.16,             # a = 3.16 A
    "ws2": 3.15,              # a = 3.15 A
    "mose2": 3.29,            # a = 3.29 A
    "wse2": 3.28,             # a = 3.28 A
    "silicene": 3.86,         # a = 3.86 A
    "germanene": 4.02,        # a = 4.02 A
    "kagome": 1.0,
    "lieb": 1.0,
    "checkerboard": 1.0,
}


def period_vector(material: str, edge: str) -> np.ndarray:
    """Translation vector R along the ribbon transport axis for one unit cell."""
    spec = MATERIALS[material]
    builder = spec["builder"]
    if builder in ("honeycomb", "hbn", "phosphorene"):
        if edge == "armchair":
            return np.array([3.0, 0.0])
        elif edge == "zigzag":
            return np.array([0.0, 2 * S])
        else:
            raise ValueError(f"Unknown edge {edge!r} for honeycomb builder")
    elif builder == "square":
        return np.array([1.0, 0.0])
    elif builder == "triangular":
        if edge in ("zigzag", "strip"):
            return np.array([1.0, 0.0])
        elif edge == "armchair":
            return np.array([0.0, 2 * S])
        else:
            raise ValueError(f"Unknown edge {edge!r} for triangular ribbon")
    elif builder == "mos2":
        if edge == "zigzag":
            return np.array([1.0, 0.0])
        elif edge == "armchair":
            return np.array([0.0, 2 * S])
        else:
            raise ValueError(f"Unknown edge {edge!r} for TMD ribbon")
    elif builder == "lattice":
        lat = LATTICES[material]
        e = lat.edge(edge)
        return _vec(lat, e.T)
    raise KeyError(f"Unknown builder {builder!r} for material {material!r}")


from functools import lru_cache


@lru_cache(maxsize=1024)
def _resolve_model_by_id(model_id: str):
    parts = model_id.split("/")
    if len(parts) != 3 or not parts[2].startswith("N") or not parts[2][1:].isdigit():
        raise ValueError(f"model_id must look like material/edge/Nwidth, got {model_id!r}")
    mat, edge, width = parts[0], parts[1], int(parts[2][1:])
    spec = MATERIALS[mat]
    h = _build(mat, edge, width)
    return RibbonModel(
        material=mat,
        edge=edge,
        width=width,
        t_ev=spec["t_ev"],
        sites_per_cell=h.H0.shape[0],
        band_top_t=0.0,
        impurity_v_t=spec.get("impurity_v_t", 0.5),
        orbitals_per_site=spec.get("orbitals_per_site", 1),
        source=spec.get("source", ""),
    )


def _resolve_model(model_or_id):
    if isinstance(model_or_id, str):
        return _resolve_model_by_id(model_or_id)
    return model_or_id


@lru_cache(maxsize=1024)
def get_cached_hamiltonian(material: str, edge: str, width: int):
    return _build(material, edge, width)


def device_atom_coordinates(model_or_id, n_cells: Optional[int] = None, unit: str = "lattice") -> np.ndarray:
    """Exact (x, y) coordinates of every atom in the device.

    Returns:
        np.ndarray of shape (n_atoms, 2), ordered by [cell * atoms_per_cell + site_in_cell].
    """
    model = _resolve_model(model_or_id)
    n_cells = model.n_cells if n_cells is None else int(n_cells)
    h = get_cached_hamiltonian(model.material, model.edge, model.width)
    period = period_vector(model.material, model.edge)

    atoms_per_cell = h.positions.shape[0]
    cell_idx = np.arange(n_cells)[:, None, None]  # (n_cells, 1, 1)
    coords = h.positions[None, :, :] + cell_idx * period[None, None, :]
    flat_coords = coords.reshape(-1, 2)

    if unit == "angstrom":
        scale = LATTICE_CONSTANTS_A.get(model.material, 1.0)
        return flat_coords * scale
    return flat_coords


def device_host_onsite(model_or_id, n_cells: Optional[int] = None, unit: str = "t") -> np.ndarray:
    """Unperturbed host on-site potential across all sites/atoms of the device.

    Returns:
        np.ndarray of shape (n_cells * sites_per_cell,) or (n_cells, sites_per_cell).
    """
    model = _resolve_model(model_or_id)
    n_cells = model.n_cells if n_cells is None else int(n_cells)
    h = get_cached_hamiltonian(model.material, model.edge, model.width)
    h0_diag = np.diag(h.H0.real)
    onsite = np.tile(h0_diag, n_cells)
    if unit == "eV":
        return onsite * model.t_ev
    return onsite


@dataclass(frozen=True)
class ImpurityConfiguration:
    """Exact microscopic configuration of impurities in a device."""
    model_id: str
    material: str
    edge: str
    width: int
    seed: int
    n_impurities: int
    density: float
    n_cells: int
    n_atoms: int
    orbitals_per_site: int
    t_ev: float
    atom_indices: np.ndarray        # (n_imp,) global atom index in device: 0..n_atoms-1
    cell_indices: np.ndarray        # (n_imp,) cell index: 0..n_cells-1
    site_indices: np.ndarray        # (n_imp,) site index within unit cell
    coordinates: np.ndarray         # (n_imp, 2) Cartesian (x, y) in device geometry
    host_onsite_t: np.ndarray       # (n_imp,) or (n_imp, orb) host on-site potential in units of t
    host_onsite_ev: np.ndarray      # (n_imp,) or (n_imp, orb) host on-site potential in eV
    impurity_onsite_t: np.ndarray   # (n_imp,) or (n_imp, orb) perturbed potential V_host + v in units of t
    impurity_onsite_ev: np.ndarray  # (n_imp,) or (n_imp, orb) perturbed potential V_host + v in eV
    defect_v_t: float               # perturbation shift v in units of t
    defect_v_ev: float              # perturbation shift v in eV
    sublattice: np.ndarray          # (n_imp,) sublattice / site species


def reconstruct_configuration(
    model_or_id,
    seed: int,
    n_impurities: Optional[int] = None,
    density: Optional[float] = None,
    n_cells: Optional[int] = None,
    unit: str = "lattice",
) -> ImpurityConfiguration:
    """Reconstruct the exact microscopic impurity configuration for any device and seed.

    Determinism follows the project's nested seed convention:
        atoms = RandomState(seed).choice(n_atoms, n_impurities, replace=False)
    """
    model = _resolve_model(model_or_id)
    n_cells = model.n_cells if n_cells is None else int(n_cells)
    orb = model.orbitals_per_site
    atoms_per_cell = model.sites_per_cell // orb
    n_atoms = n_cells * atoms_per_cell

    if n_impurities is None:
        if density is None:
            raise ValueError("Either n_impurities or density must be provided")
        n_impurities = max(1, int(round(density * n_atoms)))
    else:
        n_impurities = int(n_impurities)

    actual_density = n_impurities / n_atoms if density is None else float(density)

    # Seed-deterministic atom selection
    atoms = np.random.RandomState(int(seed)).choice(n_atoms, n_impurities, replace=False)
    cell_idx = atoms // atoms_per_cell
    site_idx = atoms % atoms_per_cell

    # Geometry & Hamiltonians
    h = get_cached_hamiltonian(model.material, model.edge, model.width)
    period = period_vector(model.material, model.edge)
    coords = h.positions[site_idx] + cell_idx[:, None] * period[None, :]
    if unit == "angstrom":
        coords = coords * LATTICE_CONSTANTS_A.get(model.material, 1.0)

    sub = h.sublattice[site_idx]

    # Host on-site potential at each impurity atom
    h0_diag = np.diag(h.H0.real)
    v_t = model.impurity_v_t
    v_ev = v_t * model.t_ev

    if orb == 1:
        v_host_t = h0_diag[site_idx]
        v_host_ev = v_host_t * model.t_ev
        v_imp_t = v_host_t + v_t
        v_imp_ev = v_host_ev + v_ev
    else:
        # Multi-orbital (e.g. TMD 3-band)
        v_host_t = np.array([h0_diag[s * orb : (s + 1) * orb] for s in site_idx])
        v_host_ev = v_host_t * model.t_ev
        v_imp_t = v_host_t + v_t
        v_imp_ev = v_host_ev + v_ev

    return ImpurityConfiguration(
        model_id=model.model_id,
        material=model.material,
        edge=model.edge,
        width=model.width,
        seed=int(seed),
        n_impurities=n_impurities,
        density=actual_density,
        n_cells=n_cells,
        n_atoms=n_atoms,
        orbitals_per_site=orb,
        t_ev=model.t_ev,
        atom_indices=atoms,
        cell_indices=cell_idx,
        site_indices=site_idx,
        coordinates=coords,
        host_onsite_t=v_host_t,
        host_onsite_ev=v_host_ev,
        impurity_onsite_t=v_imp_t,
        impurity_onsite_ev=v_imp_ev,
        defect_v_t=v_t,
        defect_v_ev=v_ev,
        sublattice=sub,
    )


def device_onsite_profile(
    model_or_id,
    seed: int,
    n_impurities: Optional[int] = None,
    density: Optional[float] = None,
    n_cells: Optional[int] = None,
    unit: str = "t",
) -> np.ndarray:
    """Full on-site potential profile across all orbitals/sites of the device.

    Returns:
        np.ndarray of shape (n_cells, sites_per_cell) with host on-site + impurity shifts.
    """
    model = _resolve_model(model_or_id)
    n_cells = model.n_cells if n_cells is None else int(n_cells)
    h = get_cached_hamiltonian(model.material, model.edge, model.width)
    spc = model.sites_per_cell
    orb = model.orbitals_per_site
    n_atoms = n_cells * (spc // orb)

    if n_impurities is None:
        if density is None:
            raise ValueError("Either n_impurities or density must be provided")
        n_impurities = max(1, int(round(density * n_atoms)))
    else:
        n_impurities = int(n_impurities)

    from .disorder import impurity_shifts

    shifts = impurity_shifts(n_cells, spc, n_impurities, int(seed), model.impurity_v_t, orb)
    h0_diag = np.diag(h.H0.real)
    host_profile = np.tile(h0_diag, (n_cells, 1))
    profile = host_profile + shifts

    if unit == "eV":
        return profile * model.t_ev
    return profile


def generate_cloud_configurations(
    model_or_id,
    density: float,
    seeds: np.ndarray,
    n_impurities: Optional[int] = None,
    n_cells: Optional[int] = None,
    unit: str = "lattice",
) -> dict:
    """Vectorized generation of microscopic configuration arrays for a whole cloud of seeds."""
    model = _resolve_model(model_or_id)
    n_cells = model.n_cells if n_cells is None else int(n_cells)
    orb = model.orbitals_per_site
    atoms_per_cell = model.sites_per_cell // orb
    n_atoms = n_cells * atoms_per_cell

    if n_impurities is None:
        n_impurities = max(1, int(round(density * n_atoms)))
    else:
        n_impurities = int(n_impurities)

    seeds = np.asarray(seeds, dtype=np.int32)
    atoms = np.array([
        np.random.RandomState(int(s)).choice(n_atoms, n_impurities, replace=False)
        for s in seeds
    ], dtype=np.int32)

    cell_idx = atoms // atoms_per_cell
    site_idx = atoms % atoms_per_cell

    h = get_cached_hamiltonian(model.material, model.edge, model.width)
    period = period_vector(model.material, model.edge)
    coords = (h.positions[site_idx] + cell_idx[:, :, None] * period[None, None, :]).astype(np.float32)
    if unit == "angstrom":
        coords = coords * np.float32(LATTICE_CONSTANTS_A.get(model.material, 1.0))

    sub = h.sublattice[site_idx].astype(np.int8)

    h0_diag = np.diag(h.H0.real).astype(np.float32)
    v_t = np.float32(model.impurity_v_t)
    v_ev = v_t * np.float32(model.t_ev)

    if orb == 1:
        v_host_t = h0_diag[site_idx]
    else:
        v_host_t = np.stack([h0_diag[site_idx * orb + o] for o in range(orb)], axis=-1)

    v_host_ev = v_host_t * np.float32(model.t_ev)
    v_imp_t = v_host_t + v_t
    v_imp_ev = v_host_ev + v_ev

    return {
        "coords": coords,
        "atom_indices": atoms,
        "cell_indices": cell_idx,
        "site_indices": site_idx,
        "host_onsite_t": v_host_t,
        "host_onsite_ev": v_host_ev,
        "impurity_onsite_t": v_imp_t,
        "impurity_onsite_ev": v_imp_ev,
        "defect_v_t": v_t,
        "defect_v_ev": v_ev,
        "sublattice": sub,
        "seeds": seeds,
        "n_cells": np.int32(n_cells),
        "n_impurities": np.int32(n_impurities),
        "density": np.float32(density),
        "model_id": model.model_id,
    }
