"""Append-only on-disk store of spectra per ribbon model, with the validity checks
that caught this project's data bugs (non-finite values, cross-density duplicates)."""
import hashlib
import json
from pathlib import Path

import numpy as np


def _key(density: float) -> str:
    return f"d{density:.4f}"


class CloudStore:
    def __init__(self, root):
        self.root = Path(root).expanduser()

    def _dir(self, model_id):
        return self.root / model_id

    def _meta(self, model_id):
        p = self._dir(model_id) / "meta.json"
        return json.loads(p.read_text()) if p.exists() else {"clouds": {}}

    def _save_meta(self, model_id, meta):
        (self._dir(model_id) / "meta.json").write_text(json.dumps(meta, indent=2))

    def _check_energies(self, model_id, energies_t):
        d = self._dir(model_id)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "energies_t.npy"
        energies_t = np.asarray(energies_t, dtype=np.float64)
        if p.exists():
            if not np.allclose(np.load(p), energies_t):
                raise ValueError(f"energy grid differs from the one stored for {model_id}")
        else:
            np.save(p, energies_t)

    def write_pristine(self, model_id, energies_t, T, formula=None):
        energies_t = np.asarray(energies_t, dtype=np.float64)
        self._check_energies(model_id, energies_t)
        T = np.asarray(T, dtype=np.float64)
        if T.ndim != 1 or T.size != energies_t.size:
            raise ValueError(f"pristine spectrum must be 1D with size {energies_t.size}, got shape {T.shape}")
        if not np.all(np.isfinite(T)):
            raise ValueError("pristine spectrum is not finite")
        np.save(self._dir(model_id) / "pristine.npy", T)
        if formula is not None:
            meta = self._meta(model_id)
            meta["pristine_formula"] = str(formula)
            self._save_meta(model_id, meta)

    def read_pristine(self, model_id):
        d = self._dir(model_id)
        return np.load(d / "energies_t.npy"), np.load(d / "pristine.npy")

    def write_cloud(self, model_id, density, n_impurities, spectra, seeds, energies_t, formula=None, max_excess_tol=0.05):
        spectra = np.asarray(spectra, dtype=np.float64)
        seeds = np.asarray(seeds, dtype=np.int64)
        energies_t = np.asarray(energies_t, dtype=np.float64)
        if spectra.ndim != 2 or spectra.shape[0] != seeds.size:
            raise ValueError("spectra must be (n_seeds, n_energies) with one seed per row")
        if spectra.shape[1] != energies_t.size:
            raise ValueError(f"spectra channels ({spectra.shape[1]}) does not match energies_t ({energies_t.size})")
        if not np.all(np.isfinite(spectra)):
            raise ValueError("spectra contain non-finite values")
        if np.unique(seeds).size != seeds.size:
            raise ValueError("duplicate seed in cloud")
        self._check_energies(model_id, energies_t)
        pristine_file = self._dir(model_id) / "pristine.npy"
        if max_excess_tol is not None and pristine_file.exists():
            pris = np.load(pristine_file)
            med = np.median(spectra, axis=0)
            max_excess = float(np.max(med - pris))
            if max_excess > max_excess_tol:
                raise ValueError(
                    f"unphysical cloud for {model_id} (density {density}): median exceeds pristine by {max_excess:.4f} (tol={max_excess_tol}). "
                    "Disorder cannot systematically enhance transmission above clean conductance."
                )
        new = {hashlib.md5(r.tobytes()).hexdigest() for r in spectra}
        for other in self.densities(model_id):
            if abs(other - density) < 1e-12:
                continue
            old, _ = self.read_cloud(model_id, other)
            if new & {hashlib.md5(r.tobytes()).hexdigest() for r in old}:
                raise ValueError(f"identical spectra at densities {density} and {other}: generator bug?")
        d, k = self._dir(model_id), _key(density)
        np.save(d / f"cloud_{k}.npy", spectra)
        np.save(d / f"cloud_{k}_seeds.npy", seeds)
        meta = self._meta(model_id)
        if formula is not None:
            pris_f = meta.get("pristine_formula")
            if pris_f is not None and pris_f != str(formula):
                raise ValueError(f"cloud formula {formula!r} does not match pristine formula {pris_f!r} for {model_id}")
            meta["clouds"][k] = {"density": float(density), "n_impurities": int(n_impurities), "n": int(seeds.size), "formula": str(formula)}
        else:
            meta["clouds"][k] = {"density": float(density), "n_impurities": int(n_impurities), "n": int(seeds.size)}
        self._save_meta(model_id, meta)

    def read_cloud(self, model_id, density):
        d, k = self._dir(model_id), _key(density)
        return np.load(d / f"cloud_{k}.npy"), np.load(d / f"cloud_{k}_seeds.npy")

    def has_cloud(self, model_id, density) -> bool:
        return (self._dir(model_id) / f"cloud_{_key(density)}.npy").exists()

    def densities(self, model_id):
        return sorted(v["density"] for v in self._meta(model_id)["clouds"].values())

    def models(self):
        return sorted(str(p.parent.relative_to(self.root)) for p in self.root.rglob("energies_t.npy"))

    def spike_fraction(self, model_id, density) -> float:
        _, pris = self.read_pristine(model_id)
        c, _ = self.read_cloud(model_id, density)
        return float(np.mean(c > pris[None, :] + 1e-6))
