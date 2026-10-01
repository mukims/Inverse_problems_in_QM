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
    width_vote: float = 0.0
    unknown_recon: bool = False
    novelty_ratio: float = 0.0
    novelty_s: float = 0.0


class Atlas:
    def __init__(self, spec, encoder, mu, sd, refs, ref_model, ref_density, models, threshold, k,
                 threshold_table=None, novelty="reconstruction_v1"):
        self.spec, self.encoder, self.mu, self.sd = spec, encoder, mu, sd
        self.refs, self.ref_model, self.ref_density = refs, ref_model, ref_density
        self.models, self.threshold, self.k = list(models), float(threshold), int(k)
        self.threshold_table = threshold_table or {}
        self.novelty = novelty
        self._nn = NearestNeighbors(n_neighbors=self.k).fit(self.refs)
        self._build_model_nns()

    def _build_model_nns(self):
        self._model_nns = {}
        for i in range(len(self.models)):
            mask = (self.ref_model == i)
            if np.any(mask):
                k_m = min(self.k, int(np.sum(mask)))
                self._model_nns[i] = NearestNeighbors(n_neighbors=k_m).fit(self.refs[mask])

    # ---------- building ----------
    @staticmethod
    def _load_inputs(store, registry, model_ids, spec, max_seed=None):
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
                if max_seed is not None:
                    mask = s <= max_seed
                    c, s = c[mask], s[mask]
                X.append(spec.to_input(c, e_t, m.band_top_t))
                midx.append(np.full(len(s), i))
                dens.append(np.full(len(s), d))
                seeds.append(s)
        return (np.concatenate(X), np.concatenate(midx), np.concatenate(dens).astype(float), np.concatenate(seeds))

    @classmethod
    def build(cls, store, registry, model_ids, spec, latent=32, epochs=60, patience=8, k=15,
              refs_per_model=2000, seed=2, threads=4, max_seed=None, val_seed_min=None):
        X, midx, dens, seeds = cls._load_inputs(store, registry, model_ids, spec, max_seed=max_seed)
        val = np.zeros(len(X), bool)
        for i in np.unique(midx):                       # validation split
            s = seeds[(midx == i) & (seeds >= 0)]
            if s.size:
                if val_seed_min is not None:
                    val |= (midx == i) & (seeds >= val_seed_min)
                else:
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

    def calibrate_novelty(self, store, registry, model_ids, val_seed_min=None, max_seed=None):
        """Calibrate class-conditional thresholds tau(model, density) on validation seeds.
        Uses log-normal 99th percentile: exp(mean(log s) + 2.326 * sd(log s))."""
        table = {}
        for mid in model_ids:
            m = registry.get(mid)
            e_t, _ = store.read_pristine(mid)
            table[mid] = {}
            for d in store.densities(mid):
                c, s = store.read_cloud(mid, d)
                if val_seed_min is not None:
                    mask = (s >= val_seed_min)
                else:
                    mask = (s >= np.quantile(s, 0.85))
                if max_seed is not None:
                    mask &= (s <= max_seed)
                c_val = c[mask]
                if len(c_val) == 0:
                    continue
                loc = self.locate(c_val, e_t, m.band_top_t)
                s_vals = np.array([r.novelty_s for r in loc])
                log_s = np.log(np.maximum(s_vals, 1e-12))
                mean_log = float(np.mean(log_s))
                sd_log = float(np.std(log_s, ddof=1)) if len(log_s) > 1 else 0.0
                tau = float(np.exp(mean_log + 2.326 * sd_log))
                table[mid][f"{d:.4f}"] = tau
        self.threshold_table = table
        self.novelty = "class_conditional_v1"
        return table

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
            vals, counts = np.unique(w, return_counts=True)
            width_vote = float(vals[np.argmax(counts)])
            pred_dens = float(np.median(self.ref_density[idx[r][members]]))

            # Target model identification for class-conditional score s
            target_idx = None
            for i, m in enumerate(self.models):
                if (m.material, m.edge) == (mat, edge) and round(m.width) == round(width_vote):
                    target_idx = i
                    break
            if target_idx is None:
                m_vals, m_counts = np.unique(nb[members], return_counts=True)
                target_idx = int(m_vals[np.argmax(m_counts)])

            # Score s: mean distance to k=15 nearest references of that model only
            if target_idx in self._model_nns:
                m_dist, _ = self._model_nns[target_idx].kneighbors(Zs[r:r+1])
                s = float(m_dist.mean())
            else:
                s = nov

            target_mid = self.models[target_idx].model_id
            unk_recon = bool(rec[r] > self.threshold)

            # Class-conditional threshold lookup
            if self.threshold_table and target_mid in self.threshold_table:
                d_map = self.threshold_table[target_mid]
                avail_d = [float(k_d) for k_d in d_map.keys()]
                snapped_d = min(avail_d, key=lambda x: abs(x - pred_dens))
                snapped_key = f"{snapped_d:.4f}"
                tau = float(d_map.get(snapped_key, d_map[min(d_map.keys(), key=lambda k_d: abs(float(k_d) - pred_dens))]))
                is_unk = bool(s > tau)
                ratio = float(s / tau)
            else:
                is_unk = unk_recon
                ratio = float(rec[r] / self.threshold) if self.threshold > 0 else 0.0

            out.append(Located(mat, edge, width, extrap, pred_dens,
                               len(members) / self.k, nov, is_unk, float(rec[r]),
                               width_vote=width_vote, unknown_recon=unk_recon,
                               novelty_ratio=ratio, novelty_s=s))
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
        self._build_model_nns()
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
            "threshold_table": self.threshold_table, "novelty": self.novelty,
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
        threshold_table = man.get("threshold_table", {})
        novelty = man.get("novelty", "reconstruction_v1")
        return cls(InputSpec(**man["spec"]), enc, r["mu"], r["sd"], r["refs"], r["ref_model"], r["ref_density"],
                   [RibbonModel(**m) for m in man["models"]], man["threshold"], man["k"],
                   threshold_table=threshold_table, novelty=novelty)
