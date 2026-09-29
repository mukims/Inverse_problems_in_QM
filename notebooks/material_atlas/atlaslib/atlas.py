"""The reusable map: embed spectra with a frozen encoder, answer locate() by k-NN."""
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from .encoder import Conv1dAE, embed, train_autoencoder
from .registry import RibbonModel
from .spec import InputSpec


@dataclass(frozen=True)
class Located:
    material: str
    edge: str
    width: float
    width_extrapolated: bool
    density: float
    confidence: float
    novelty: float
    unknown: bool
    recon_error: float


class Atlas:
    def __init__(self, spec, encoder, mu, sd, refs, ref_model, ref_density, models, threshold, k):
        self.spec, self.encoder, self.mu, self.sd = spec, encoder, mu, sd
        self.refs, self.ref_model, self.ref_density = refs, ref_model, ref_density
        self.models, self.threshold, self.k = list(models), float(threshold), int(k)
        self._nn = NearestNeighbors(n_neighbors=self.k).fit(self.refs)

    # ---------- building ----------
    @staticmethod
    def _load_inputs(store, registry, model_ids, spec):
        X, midx, dens, seeds = [], [], [], []
        for i, mid in enumerate(model_ids):
            m = registry.get(mid)
            e_t, pris = store.read_pristine(mid)
            X.append(spec.to_input(pris[None], e_t, m.band_top_t))
            midx.append([i])
            dens.append([0.0])
            seeds.append([-1])
            for d in store.densities(mid):
                c, s = store.read_cloud(mid, d)
                X.append(spec.to_input(c, e_t, m.band_top_t))
                midx.append(np.full(len(s), i))
                dens.append(np.full(len(s), d))
                seeds.append(s)
        return (np.concatenate(X), np.concatenate(midx), np.concatenate(dens).astype(float), np.concatenate(seeds))

    @classmethod
    def build(cls, store, registry, model_ids, spec, latent=32, epochs=60, patience=8, k=15,
              refs_per_model=2000, seed=2, threads=4):
        X, midx, dens, seeds = cls._load_inputs(store, registry, model_ids, spec)
        val = np.zeros(len(X), bool)
        for i in np.unique(midx):                       # validation = top 15% of seeds per model
            s = seeds[(midx == i) & (seeds >= 0)]
            if s.size:
                val |= (midx == i) & (seeds >= np.quantile(s, 0.85))
        enc, _ = train_autoencoder(X[~val], X[val], latent=latent, epochs=epochs, patience=patience,
                                   threads=threads, seed=seed)
        Z, rec = embed(enc, X)
        mu, sd = Z[~val].mean(0), Z[~val].std(0) + 1e-8
        Zs = (Z - mu) / sd
        rng = np.random.default_rng(seed)
        ref = np.concatenate([rng.permutation(np.where((midx == i) & ~val)[0])[:refs_per_model] for i in np.unique(midx)])
        models = [registry.get(mid) for mid in model_ids]
        atlas = cls(spec, enc, mu, sd, Zs[ref], midx[ref], dens[ref], models, np.inf, k)
        # Unknown = reconstruction error above the 99th percentile of known validation spectra.
        # BUILD-12: reconstruction error separated an unseen material with AUROC 1.00, k-NN distance only 0.95.
        atlas.threshold = float(np.percentile(rec[val], 99)) if np.any(val) else float(np.percentile(rec, 99))
        return atlas

    # ---------- querying ----------
    def _novelty(self, Zs):
        dist, _ = self._nn.kneighbors(Zs)
        return dist.mean(1)

    def locate(self, T, e_t, band_top_t=None):
        X = self.spec.to_input(T, e_t, band_top_t)
        if len(X) == 0:
            return []
        Z, rec = embed(self.encoder, X)
        Zs = (Z - self.mu) / self.sd
        dist, idx = self._nn.kneighbors(Zs)
        out = []
        for r in range(len(X)):
            nb = self.ref_model[idx[r]]
            groups = {}
            for j, mi in enumerate(nb):
                key = (self.models[mi].material, self.models[mi].edge)
                groups.setdefault(key, []).append(j)
            (mat, edge), members = max(groups.items(), key=lambda kv: len(kv[1]))
            w = np.array([self.models[nb[j]].width for j in members], float)
            wt = 1.0 / (dist[r, members] + 1e-9)
            width = float(np.sum(w * wt) / np.sum(wt))
            trained = sorted(m.width for m in self.models if (m.material, m.edge) == (mat, edge))
            width = float(np.clip(width, trained[0], trained[-1]))
            extrap = bool(np.all(w == trained[0]) or np.all(w == trained[-1]))
            nov = float(dist[r].mean())
            out.append(Located(mat, edge, width, extrap, float(np.median(self.ref_density[idx[r][members]])),
                               len(members) / self.k, nov, bool(rec[r] > self.threshold), float(rec[r])))
        return out

    def add_models(self, store, registry, model_ids):
        """Embed new models with the frozen encoder and append them as references."""
        report = {}
        for mid in model_ids:
            X, _, dens, _ = self._load_inputs(store, registry, [mid], self.spec)
            Z, rec = embed(self.encoder, X)
            Zs = (Z - self.mu) / self.sd
            report[mid] = float(np.mean(rec > self.threshold))
            self.models.append(registry.get(mid))
            self.refs = np.vstack([self.refs, Zs])
            self.ref_model = np.concatenate([self.ref_model, np.full(len(Zs), len(self.models) - 1)])
            self.ref_density = np.concatenate([self.ref_density, dens])
            self._nn = NearestNeighbors(n_neighbors=self.k).fit(self.refs)
        return report

    # ---------- persistence ----------
    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.encoder.state_dict(), "latent": self.encoder.to_latent.out_features,
                    "seq_len": self.encoder.seq_len}, path / "encoder.pt")
        np.savez(path / "refs.npz", refs=self.refs, ref_model=self.ref_model, ref_density=self.ref_density,
                 mu=self.mu, sd=self.sd)
        (path / "manifest.json").write_text(json.dumps({
            "spec": self.spec.as_dict(), "models": [asdict(m) for m in self.models], "threshold": self.threshold,
            "k": self.k, "created": date.today().isoformat()}, indent=2))

    @classmethod
    def load(cls, path):
        path = Path(path)
        man = json.loads((path / "manifest.json").read_text())
        ck = torch.load(path / "encoder.pt", weights_only=True)
        enc = Conv1dAE(ck["latent"], ck["seq_len"])
        enc.load_state_dict(ck["state"])
        enc.eval()
        r = np.load(path / "refs.npz")
        return cls(InputSpec(**man["spec"]), enc, r["mu"], r["sd"], r["refs"], r["ref_model"], r["ref_density"],
                   [RibbonModel(**m) for m in man["models"]], man["threshold"], man["k"])
