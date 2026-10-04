"""Shazam's ensemble lookup (LOOKUP-1): a signature in, the closest catalogued device out.

Each catalogued device is described by its disordered ensemble: the median and spread of the label-free input at
every energy, interpolated in impurity concentration from the clean spectrum (0%) and the stored densities.
A signature is scored by how probable it is under each device's ensemble, over the energies it covers.
"""
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from .energy import on_axis
from .registry import RibbonModel
from .spec import InputSpec, despike

S0 = 0.01              # spread floor in input units (about 0.04 in T): absorbs interpolation error, keeps the 0% anchor finite
MIN_CHANNELS = 10      # a window must cover at least this many channels
P_NO_MATCH = 0.01      # below this p-value no catalogued device matches
P_VALUE_SPECTRA = 200  # validation spectra per device used for one p-value


def window_inputs(spec, energies, T):
    """Label-free input over the energies supplied: (X, mask), one row of X per spectrum.

    Same steps as InputSpec.to_input (round to 3 decimals, despike, resample, round, clip, log), but channels outside
    [energies[0], energies[-1]] are absent: mask False, value 0. Nothing is zero-filled from a band top (label-free).
    """
    if spec.unit != "eV":
        raise ValueError("the lookup needs an eV-axis InputSpec (v3 or v4)")
    e = np.asarray(energies, dtype=np.float64)
    T = np.atleast_2d(np.asarray(T, dtype=np.float64))
    if e.ndim != 1 or T.shape[1] != e.size:
        raise ValueError(f"T has {T.shape[1]} values per spectrum but there are {e.size} energies")
    if not (np.all(np.isfinite(e)) and np.all(np.isfinite(T))):
        raise ValueError("energies and T must be finite")
    if np.any(e < 0):
        raise ValueError("energies must be >= 0 eV, measured from charge neutrality")
    if np.any(np.diff(e) <= 0):
        raise ValueError("energies must increase")
    grid = spec.energies_t()
    mask = (grid >= e[0] - 1e-9) & (grid <= e[-1] + 1e-9)
    if mask.sum() < MIN_CHANNELS:
        raise ValueError(f"the window {e[0]:.3f}-{e[-1]:.3f} eV covers {int(mask.sum())} channels of {spec.step_t} eV; "
                         f"at least {MIN_CHANNELS} are needed")
    T = np.round(T, 3)
    if spec.despike:
        T = despike(T)
    out = np.zeros((T.shape[0], grid.size))
    for i in range(T.shape[0]):
        out[i, mask] = np.interp(grid[mask], e, T[i])
    out = np.clip(np.round(out, 3), 0.0, spec.cap)
    return (np.log1p(out) / np.log1p(spec.cap)).astype(np.float32), mask


def window_input(spec, energies, T):
    """One signature: (x, mask)."""
    T = np.asarray(T, dtype=np.float64)
    if T.ndim != 1:
        raise ValueError("a signature is one spectrum: T must be 1D")
    X, mask = window_inputs(spec, energies, T)
    return X[0], mask


@dataclass
class Catalogue:
    spec: InputSpec
    models: list            # RibbonModel per device, sorted by model_id
    grid: np.ndarray        # concentrations, 0 to grid_max
    mu: np.ndarray          # (devices, grid, channels): ensemble median
    sd: np.ndarray          # (devices, grid, channels): ensemble spread
    val_x: np.ndarray       # validation inputs (seeds in `val`), for kappa and p-values
    val_dev: np.ndarray     # device index of each validation input
    val_conc: np.ndarray    # true concentration of each validation input
    stored: dict            # model_id -> stored densities used
    kappa: float = 1.0
    settings: dict = None

    def __post_init__(self):
        self._s = np.sqrt(np.asarray(self.sd, np.float64) ** 2 + S0 ** 2).astype(np.float32)
        self._logs = np.log(self._s)

    @property
    def ids(self):
        return [m.model_id for m in self.models]

    @classmethod
    def build(cls, store, models, spec, train_max=699, val=(700, 849), grid_max=0.05, grid_step=0.0005):
        if spec.unit != "eV":
            raise ValueError("the lookup needs an eV-axis InputSpec (v3 or v4)")
        models = sorted(models, key=lambda m: m.model_id)
        grid = np.round(np.arange(0.0, grid_max + grid_step / 2, grid_step), 6)
        n_ch = spec.n_channels
        mu = np.zeros((len(models), grid.size, n_ch), np.float32)
        sd = np.zeros_like(mu)
        vx, vdev, vconc, stored = [], [], [], {}
        for k, m in enumerate(models):
            e_t, pris = store.read_pristine(m.model_id)
            e_ev, _ = on_axis(spec, m, e_t)
            meds, sds = [window_input(spec, e_ev, pris)[0]], [np.zeros(n_ch, np.float32)]
            ds = sorted(store.densities(m.model_id))
            if not ds:
                raise ValueError(f"{m.model_id} has no disorder clouds")
            for d in ds:
                cloud, seeds = store.read_cloud(m.model_id, d)
                X, _ = window_inputs(spec, e_ev, cloud)
                tr = X[seeds <= train_max]
                if len(tr) < 2:
                    raise ValueError(f"{m.model_id} at density {d} has {len(tr)} training spectra (seeds <= {train_max})")
                meds.append(np.median(tr, axis=0))
                sds.append(tr.std(axis=0))
                va = X[(seeds >= val[0]) & (seeds <= val[1])]
                vx.append(va)
                vdev.append(np.full(len(va), k))
                vconc.append(np.full(len(va), d))
            nodes = np.array([0.0] + ds)
            mu[k] = np.clip(PchipInterpolator(nodes, np.array(meds), axis=0, extrapolate=True)(grid), 0.0, 1.0)
            sd[k] = np.clip(PchipInterpolator(nodes, np.array(sds), axis=0, extrapolate=True)(grid), 0.0, None)
            stored[m.model_id] = [float(d) for d in ds]
        return cls(spec, models, grid, mu, sd, np.concatenate(vx).astype(np.float32), np.concatenate(vdev),
                   np.concatenate(vconc), stored,
                   settings={"train_max": train_max, "val": list(val), "grid_max": grid_max, "grid_step": grid_step,
                             "s0": S0})

    def without(self, materials):
        drop = set(materials)
        keep = [k for k, m in enumerate(self.models) if m.material not in drop]
        if len(keep) == len(self.models):
            raise ValueError(f"no catalogued device has material in {sorted(drop)}")
        remap = {old: new for new, old in enumerate(keep)}
        vk = np.isin(self.val_dev, keep)
        return Catalogue(self.spec, [self.models[k] for k in keep], self.grid, self.mu[keep], self.sd[keep],
                         self.val_x[vk], np.array([remap[int(i)] for i in self.val_dev[vk]]), self.val_conc[vk],
                         {self.models[k].model_id: self.stored[self.models[k].model_id] for k in keep},
                         kappa=self.kappa, settings=self.settings)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "catalogue.npz", grid=self.grid, mu=self.mu, sd=self.sd, val_x=self.val_x,
                 val_dev=self.val_dev, val_conc=self.val_conc)
        (path / "manifest.json").write_text(json.dumps({
            "spec": self.spec.as_dict(), "models": [asdict(m) for m in self.models], "stored": self.stored,
            "kappa": self.kappa, "settings": self.settings, "n_devices": len(self.models),
            "created": date.today().isoformat()}, indent=2))

    @classmethod
    def load(cls, path):
        path = Path(path)
        man = json.loads((path / "manifest.json").read_text())
        a = np.load(path / "catalogue.npz")
        return cls(InputSpec(**man["spec"]), [RibbonModel(**m) for m in man["models"]], a["grid"], a["mu"], a["sd"],
                   a["val_x"], a["val_dev"], a["val_conc"], man["stored"], kappa=man["kappa"],
                   settings=man["settings"])

