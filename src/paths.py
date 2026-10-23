"""Where the data, archives and results live. Every script and test asks here instead of hard-coding a path.

    TRANSMISSION_DATA       the data tree (default: <repo>/data, usually a symlink to the real folder)
    TRANSMISSION_ARCHIVES   superseded runs and raw output (default: the folder called archives next to the data tree)

Data tree: data/<material>/<edge>/N<width>/ (see data/README.md). Results written by scripts go to <repo>/results,
which mirrors the folder layout the scripts used to write into (results/material_atlas/lookup_v4, ...).
"""
import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    env = os.environ.get("TRANSMISSION_DATA")
    return Path(env).expanduser() if env else repo_root() / "data"


def archives_root() -> Path:
    env = os.environ.get("TRANSMISSION_ARCHIVES")
    if env:
        return Path(env).expanduser()
    return data_root().resolve().parent / "archives"


def derived_dir() -> Path:
    return data_root().resolve().parent / "derived"


def archive_run(name: str) -> Path:
    """An old atlas_store run kept for the record, e.g. archive_run("smoke_v1")."""
    return archives_root() / "atlas_store_runs" / name


def ribbon_dir(material: str, edge: str, width: int) -> Path:
    return data_root() / material / edge / f"N{width}"


def model_dir(model_id: str) -> Path:
    """A ribbon by the id the CloudStore uses, e.g. "mos2/zigzag/N9"."""
    return data_root() / model_id


def leads_dir(material: str) -> Path:
    return data_root() / material / "leads"


def agnr_consolidated(width: int) -> Path:
    """The original consolidated 7/9-AGNR array: shape (concentrations, 10000 seeds, 300 channels)."""
    return ribbon_dir("graphene-ideal", "armchair", width) / "legacy" / f"size_{width}.npy"


def square_combined(width: int) -> Path:
    """The clean square-lattice set from ca_sq.py: conc_<c>.npy + conc_<c>_meta.csv per concentration."""
    return ribbon_dir("square", "strip", width) / "legacy" / f"size_{width}_combined"


def results_dir(*parts, create: bool = False) -> Path:
    d = repo_root().joinpath("results", *parts)
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def require(path) -> Path:
    """The path, or an error that says which setting to change when it is missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Data lives under TRANSMISSION_DATA (now {data_root()}); "
                                f"archives under TRANSMISSION_ARCHIVES (now {archives_root()}).")
    return p
