"""Data for the all-ribbons autoencoder (AE-1): every device, pilot cloud and legacy 7/9-AGNR concentration on one
label-free input (InputSpec v4, 416 channels). Labels (device, density, source, seed) are kept for evaluation only."""
import os
from pathlib import Path

import numpy as np

import paths
from atlaslib import CloudStore, InputSpec
from atlaslib.energy import on_axis
from atlaslib.lookup import window_inputs
from tbribbon.materials import MATERIALS

SPEC = InputSpec(version="v4")
SPLITS = {"train": (0, 699), "val": (700, 849), "test": (850, 999)}      # inclusive configuration seeds
PILOT_STORES = ("conc_v2",)                                                # archived run: 24 densities, 0.25-6%
LEGACY_DIR = "default"                                                     # the 7/9-AGNR arrays under data/graphene-ideal
LEGACY = {7: ("size_7.npy", 1400), 9: ("size_9.npy", 1800)}               # c = 2, 4, ... impurities on `sites` sites
SOURCES = ("catalogue", "pilot", "legacy")


class _Axis:
    """What on_axis needs of a model: its id and hopping t_ev (make_model would recompute band edges, 0.7 s per ribbon)."""
    def __init__(self, mid):
        self.model_id, self.t_ev = mid, MATERIALS[mid.split("/")[0]]["t_ev"]


def _pick(n, per_cloud):
    """Indices of at most `per_cloud` evenly spaced rows out of n."""
    return np.arange(n) if per_cloud is None or n <= per_cloud else np.unique(np.linspace(0, n - 1, per_cloud).astype(int))


class _Rows:
    def __init__(self):
        self.x, self.dev, self.dens, self.src, self.seed = [], [], [], [], []
        self.devices = {}

    def add(self, X, mid, density, source, seeds):
        k = self.devices.setdefault(mid, len(self.devices))
        self.x.append(X)
        self.dev.append(np.full(len(X), k, np.int32))
        self.dens.append(np.full(len(X), density, np.float32))
        self.src.append(np.full(len(X), SOURCES.index(source), np.int8))
        self.seed.append(np.asarray(seeds, np.int32))

    def done(self):
        names = [m for m, _ in sorted(self.devices.items(), key=lambda kv: kv[1])]
        cat = lambda a, dt: np.concatenate(a) if a else np.empty(0, dt)
        return {"x": cat(self.x, np.float32).reshape(-1, SPEC.n_channels), "dev": cat(self.dev, np.int32),
                "dens": cat(self.dens, np.float32), "src": cat(self.src, np.int8), "seed": cat(self.seed, np.int32),
                "devices": np.array(names)}


def _stores(root, stores, pilots):
    """(source, CloudStore) pairs. Default: the data tree is the catalogue (the densities it was built with: clouds
    merged in from another run stay hidden) and the archived conc_v2 run is the pilot. With `root`, `stores` and
    `pilots` are folder names under it."""
    if root is None:
        return [("catalogue", CloudStore(paths.data_root()))] + [
            ("pilot", CloudStore(paths.archive_run(n))) for n in (PILOT_STORES if pilots is None else pilots)]
    root = Path(root).expanduser()
    return [("catalogue", CloudStore(root / n)) for n in stores or ()] + [("pilot", CloudStore(root / n)) for n in pilots or ()]


def collect(split, root=None, stores=None, pilots=None, legacy_dir=LEGACY_DIR,
            per_cloud=None, include=SOURCES, with_configurations=False, config_unit="lattice"):
    """All spectra of one split as one array X (n, 416) with labels. The clean spectrum of every catalogue device joins
    the train split once, at density 0. `per_cloud` thins every cloud to evenly spaced seeds.
    If `with_configurations=True`, attaches exact microscopic impurity configurations to the returned dict."""
    lo, hi = SPLITS[split]
    rows = _Rows()
    for source, store in _stores(root, stores, pilots):
        if source not in include:
            continue
        for mid in sorted(store.models()):
            if mid.startswith("square/") or not store.densities(mid):
                continue                       # the unseen lattice, and ribbons with only a clean spectrum, stay out
            m = _Axis(mid)
            e_t, pris = store.read_pristine(mid)
            e_ev, _ = on_axis(SPEC, m, e_t)
            if split == "train" and source == "catalogue":
                rows.add(window_inputs(SPEC, e_ev, pris)[0], mid, 0.0, source, [-1])
            for d in store.densities(mid):
                cloud, seeds = store.read_cloud(mid, d)
                keep = np.flatnonzero((seeds >= lo) & (seeds <= hi))
                keep = keep[_pick(len(keep), per_cloud)]
                if len(keep):
                    rows.add(window_inputs(SPEC, e_ev, cloud[keep])[0], mid, d, source, seeds[keep])
    if "legacy" in include and legacy_dir:
        for width, (fname, sites) in LEGACY.items():
            path = paths.agnr_consolidated(width) if legacy_dir == LEGACY_DIR else Path(legacy_dir) / fname
            mm = np.load(path, mmap_mode="r")                          # (n_conc, n_seeds, 300), row i = seed i
            m = _Axis(f"graphene-ideal/armchair/N{width}")
            e_ev, _ = on_axis(SPEC, m, np.arange(mm.shape[2]) * 0.01)
            seeds = np.arange(lo, min(hi, mm.shape[1] - 1) + 1)
            seeds = seeds[_pick(len(seeds), per_cloud)]
            for i in range(mm.shape[0]):
                rows.add(window_inputs(SPEC, e_ev, np.asarray(mm[i, seeds]))[0], m.model_id, 2 * (i + 1) / sites, "legacy", seeds)
    out = rows.done()
    if with_configurations:
        attach_configurations(out, unit=config_unit)
    return out


def attach_configurations(data, unit="lattice"):
    """Attach microscopic impurity configurations to dataset dictionary produced by collect()."""
    from tbribbon.configuration import reconstruct_configuration
    devices = data["devices"]
    configs = []
    for i in range(len(data["x"])):
        mid = devices[data["dev"][i]]
        d = float(data["dens"][i])
        s = int(data["seed"][i])
        if s < 0 or d == 0:
            # Pristine / clean device: 0 impurities
            configs.append(reconstruct_configuration(mid, seed=0, n_impurities=0, density=0.0, unit=unit))
        else:
            configs.append(reconstruct_configuration(mid, seed=s, density=d, unit=unit))
    data["configs"] = configs
    return data


def save(data, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for k, v in data.items():
        np.save(directory / f"{k}.npy", v)


def load(directory):
    return {p.stem: np.load(p) for p in Path(directory).glob("*.npy")}
