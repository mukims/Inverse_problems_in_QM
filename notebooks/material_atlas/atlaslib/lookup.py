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
    kappa_material: dict = None   # material -> kappa for the concentration interval only; device choice keeps kappa

    def __post_init__(self):
        self.kappa_material = dict(self.kappa_material or {})
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
                         kappa=self.kappa, settings=self.settings, kappa_material=self.kappa_material)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "catalogue.npz", grid=self.grid, mu=self.mu, sd=self.sd, val_x=self.val_x,
                 val_dev=self.val_dev, val_conc=self.val_conc)
        (path / "manifest.json").write_text(json.dumps({
            "spec": self.spec.as_dict(), "models": [asdict(m) for m in self.models], "stored": self.stored,
            "kappa": self.kappa, "kappa_material": self.kappa_material, "settings": self.settings,
            "n_devices": len(self.models),
            "created": date.today().isoformat()}, indent=2))

    @classmethod
    def load(cls, path):
        path = Path(path)
        man = json.loads((path / "manifest.json").read_text())
        a = np.load(path / "catalogue.npz")
        return cls(InputSpec(**man["spec"]), [RibbonModel(**m) for m in man["models"]], a["grid"], a["mu"], a["sd"],
                   a["val_x"], a["val_dev"], a["val_conc"], man["stored"], kappa=man["kappa"],
                   settings=man["settings"], kappa_material=man.get("kappa_material"))


def misfit(cat, x, mask, devices=None):
    """Mean squared deviation in units of the spread over the window: (devices, grid)."""
    mu = cat.mu if devices is None else cat.mu[np.asarray(devices)]
    s = cat._s if devices is None else cat._s[np.asarray(devices)]
    z = (x[mask] - mu[:, :, mask]) / s[:, :, mask]
    return (z ** 2).mean(axis=-1)


def log_likelihood(cat, x, mask, devices):
    """Gaussian log-likelihood per channel, summed over the window: (devices, grid)."""
    idx = np.asarray(devices)
    s = cat._s[idx][:, :, mask]
    z = (x[mask] - cat.mu[idx][:, :, mask]) / s
    return -0.5 * (z ** 2).sum(axis=-1) - cat._logs[idx][:, :, mask].sum(axis=-1)


def candidates(cat, x, mask, top_k=5):
    """Misfit pre-screen: the top_k devices by best-concentration misfit, best first, and their misfits."""
    best = misfit(cat, x, mask).min(axis=1)
    order = np.argsort(best, kind="stable")[:top_k]
    return order, best[order]


def p_value(cat, device, mask, stat):
    """Share of the device's own validation spectra whose best misfit over the same window is at least `stat`."""
    V = cat.val_x[cat.val_dev == device]
    if len(V) == 0:
        raise ValueError(f"{cat.models[device].model_id} has no validation spectra")
    if len(V) > P_VALUE_SPECTRA:
        V = V[np.linspace(0, len(V) - 1, P_VALUE_SPECTRA).astype(int)]
    z = (V[:, mask][:, None, :] - cat.mu[device][:, mask][None]) / cat._s[device][:, mask][None]
    stats = (z ** 2).mean(axis=-1).min(axis=1)
    return float((np.sum(stats >= stat) + 1) / (len(stats) + 1))


def posterior(cat, x, mask, devices, kappa=None):
    """p(device, concentration | signature) over the candidates, uniform prior: (devices, grid), sums to 1."""
    ll = log_likelihood(cat, x, mask, devices) / (cat.kappa if kappa is None else kappa)
    p = np.exp(ll - ll.max())
    return p / p.sum()


def _interval(grid, w):
    """Posterior median and 90% interval of a weight vector on the grid, padded by half a grid step."""
    c = np.cumsum(w) / np.sum(w)
    q = lambda p: float(grid[min(int(np.searchsorted(c, p)), grid.size - 1)])
    half = float(grid[1] - grid[0]) / 2 if grid.size > 1 else 0.0
    return q(0.5), max(float(grid[0]), q(0.05) - half), min(float(grid[-1]), q(0.95) + half)


@dataclass(frozen=True)
class Match:
    device: str
    material: str
    edge: str
    width: int
    probability: float
    concentration: float
    concentration_lo: float
    concentration_hi: float
    runners_up: tuple          # ((device, probability), ...) for the other candidates, most probable first
    p_value: float
    no_match: bool
    window: tuple              # (first channel eV, last channel eV, channels)
    impurity_layout: object = None   # reserved for the spatial-distribution layer


def lookup(cat, energies, T, top_k=5):
    """The closest catalogued device for one signature: energies in eV from charge neutrality, T over any window."""
    x, mask = window_input(cat.spec, energies, T)
    devs, _ = candidates(cat, x, mask, top_k)
    post = posterior(cat, x, mask, devs)
    p_dev = post.sum(axis=1)
    order = np.argsort(-p_dev, kind="stable")
    b = int(order[0])
    dev = int(devs[b])
    m = cat.models[dev]
    k_m = cat.kappa_material.get(m.material)
    if k_m is None:
        w = post[b]
    else:                         # the chosen device's concentration interval uses its material's kappa
        ll = log_likelihood(cat, x, mask, [dev])[0] / k_m
        w = np.exp(ll - ll.max())
    conc, lo, hi = _interval(cat.grid, w)
    pv = p_value(cat, dev, mask, float(misfit(cat, x, mask, [dev])[0].min()))
    e = cat.spec.energies_t()[mask]
    return Match(device=m.model_id, material=m.material, edge=m.edge, width=int(m.width),
                 probability=float(p_dev[b]), concentration=conc, concentration_lo=lo, concentration_hi=hi,
                 runners_up=tuple((cat.models[int(devs[i])].model_id, float(p_dev[i])) for i in order[1:]),
                 p_value=pv, no_match=pv < P_NO_MATCH,
                 window=(round(float(e[0]), 6), round(float(e[-1]), 6), int(mask.sum())))


KAPPAS = np.round(np.geomspace(1.0, 1000.0, 61), 3)


def _calibration_rows(cat, devices, per_device, top_k):
    """(log-likelihood over the candidates, candidates, true device, true concentration) for validation spectra."""
    full = np.ones(cat.spec.n_channels, dtype=bool)
    rows = []
    for dev in devices:
        ii = np.flatnonzero(cat.val_dev == dev)
        ii = ii[np.linspace(0, len(ii) - 1, min(per_device, len(ii))).astype(int)]
        for i in ii:
            devs, _ = candidates(cat, cat.val_x[i], full, top_k)
            rows.append((log_likelihood(cat, cat.val_x[i], full, devs), devs, dev, float(cat.val_conc[i])))
    return rows


def _coverage(cat, rows, kappa_device, kappa_conc):
    """Share of rows whose device is right and whose 90% interval contains the true concentration."""
    hit = 0
    for ll, devs, true_dev, c in rows:
        p = np.exp(ll / kappa_device - (ll / kappa_device).max())
        b = int(np.argmax(p.sum(axis=1)))
        if devs[b] != true_dev:
            continue
        w = p[b] if kappa_conc == kappa_device else np.exp(ll[b] / kappa_conc - (ll[b] / kappa_conc).max())
        _, lo, hi = _interval(cat.grid, w)
        hit += int(lo <= c <= hi)
    return hit / len(rows)


def _first_reaching(cover, target):
    ok = np.flatnonzero(cover >= target)
    return int(ok[0]) if ok.size else int(np.argmax(cover))


def calibrate(cat, target=0.90, kappas=None, per_device=50, top_k=5):
    """Smallest kappa >= 1 whose 90% intervals cover the true concentration on validation spectra (sets cat.kappa)."""
    kappas = KAPPAS if kappas is None else np.asarray(kappas, dtype=float)
    rows = _calibration_rows(cat, range(len(cat.models)), per_device, top_k)
    cover = np.array([_coverage(cat, rows, k, k) for k in kappas])
    i = _first_reaching(cover, target)
    cat.kappa = float(kappas[i])
    return {"kappa": cat.kappa, "coverage": float(cover[i]), "n": len(rows),
            "scan": [[float(k), float(c)] for k, c in zip(kappas, cover)]}


def calibrate_per_material(cat, target=0.90, kappas=None, per_device=50, top_k=5):
    """Per material, the smallest kappa >= 1 whose concentration intervals reach the target coverage on that
    material's validation spectra. Device choice keeps the global cat.kappa. Sets cat.kappa_material."""
    kappas = KAPPAS if kappas is None else np.asarray(kappas, dtype=float)
    out = {}
    for mat in sorted({m.material for m in cat.models}):
        rows = _calibration_rows(cat, [k for k, m in enumerate(cat.models) if m.material == mat], per_device, top_k)
        cover = np.array([_coverage(cat, rows, cat.kappa, k) for k in kappas])
        i = _first_reaching(cover, target)
        cat.kappa_material[mat] = float(kappas[i])
        out[mat] = {"kappa": float(kappas[i]), "coverage": float(cover[i]), "n": len(rows)}
    return out




