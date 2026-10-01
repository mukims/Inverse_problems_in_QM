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
                 threshold_table=None, novelty="reconstruction_v1", threshold_params=None, z_star=None,
                 w_edge=None, n0=None):
        self.spec, self.encoder, self.mu, self.sd = spec, encoder, mu, sd
        self.refs, self.ref_model, self.ref_density = refs, ref_model, ref_density
        self.models, self.threshold, self.k = list(models), float(threshold), int(k)
        self.threshold_table = threshold_table or {}
        self.threshold_params = threshold_params or {}
        if isinstance(z_star, dict):
            self.z_star = {k_e: float(v_z) for k_e, v_z in z_star.items()}
        elif z_star is not None:
            self.z_star = float(z_star)
        else:
            self.z_star = None
        self.w_edge = {k_e: float(v_w) for k_e, v_w in w_edge.items()} if w_edge else {}
        self.n0 = float(n0) if n0 is not None else None
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

    def calibrate_novelty(self, store, registry, model_ids, val_seed_min=None, max_seed=None, n0=None):
        """Calibrate class-conditional thresholds tau(model, density) on validation seeds (FULL-3).
        1. Per class (model, density): c = median(log s), w = 1.4826 * MAD(log s).
        2. Edge scale: w_edge = median of w over that edge's classes.
        3. Shrink: w' = (N * w + n0 * w_edge) / (N + n0).
        4. Per-edge tail: z = (log s - c) / w' over all validation spectra of that edge;
           z*_edge is the empirical 99th percentile.
        5. Threshold: tau = exp(c + z*_edge * w').
        6. Choose n0 on validation: if n0 is None, scan n0 in {0, 25, 50, 100, 150, 300}
           by calibrating on the first half of validation seeds and measuring per-class
           false-alarm dispersion index (var / mean) on the second half."""
        class_samples = {}
        for mid in model_ids:
            m = registry.get(mid)
            e_t, _ = store.read_pristine(mid)
            for d in store.densities(mid):
                c, s = store.read_cloud(mid, d)
                if val_seed_min is not None:
                    mask = (s >= val_seed_min)
                else:
                    mask = (s >= np.quantile(s, 0.85))
                if max_seed is not None:
                    mask &= (s <= max_seed)
                c_val = c[mask]
                s_val = s[mask]
                if len(c_val) == 0:
                    continue
                loc = self.locate(c_val, e_t, m.band_top_t)
                s_vals = np.array([r.novelty_s for r in loc])
                pred_dens = np.array([r.density for r in loc])
                class_samples[(mid, f"{d:.4f}", m.edge)] = {
                    "scores": s_vals,
                    "seeds": s_val,
                    "pred_dens": pred_dens
                }

        # Step 6: Choose n0 on validation if n0 is None
        n0_candidates = [0, 25, 50, 100, 150, 300]
        if n0 is None and val_seed_min is not None and max_seed is not None and (max_seed - val_seed_min) >= 20:
            mid_seed = val_seed_min + (max_seed - val_seed_min + 1) // 2
            best_n0 = 0
            best_disp = np.inf
            disp_reports = {}
            for cand_n0 in n0_candidates:
                # Calibrate on split 1 (seeds < mid_seed)
                s1_stats = {}
                s1_edge_w = {}
                for (mid, d_str, edge), data in class_samples.items():
                    v1_mask = (data["seeds"] < mid_seed)
                    s1 = data["scores"][v1_mask]
                    if len(s1) == 0:
                        continue
                    log_s = np.log(np.maximum(s1, 1e-12))
                    c = float(np.median(log_s))
                    mad = float(np.median(np.abs(log_s - c)))
                    w = 1.4826 * mad
                    if w < 1e-8:
                        w = 1.0
                    s1_stats[(mid, d_str, edge)] = (c, w, len(s1))
                    s1_edge_w.setdefault(edge, []).append(w)
                w_edge1 = {e: float(np.median(ws)) for e, ws in s1_edge_w.items()}

                w_prime1 = {}
                for (mid, d_str, edge), (c, w, n_s1) in s1_stats.items():
                    wp = (float(n_s1) * w + cand_n0 * w_edge1[edge]) / (float(n_s1) + cand_n0)
                    w_prime1[(mid, d_str, edge)] = wp

                edge_z1 = {}
                for (mid, d_str, edge), data in class_samples.items():
                    v1_mask = (data["seeds"] < mid_seed)
                    s1 = data["scores"][v1_mask]
                    log_s = np.log(np.maximum(s1, 1e-12))
                    c, _, _ = s1_stats[(mid, d_str, edge)]
                    wp = w_prime1[(mid, d_str, edge)]
                    z = (log_s - c) / wp
                    edge_z1.setdefault(edge, []).extend(z)

                z_star1 = {e: float(np.percentile(zs, 99)) for e, zs in edge_z1.items()}

                table1 = {}
                for (mid, d_str, edge), (c, _, _) in s1_stats.items():
                    wp = w_prime1[(mid, d_str, edge)]
                    tau = float(np.exp(c + z_star1[edge] * wp))
                    table1.setdefault(mid, {})[d_str] = tau

                # Count false alarms per class on split 2 (seeds >= mid_seed)
                counts = []
                for (mid, d_str, edge), data in class_samples.items():
                    v2_mask = (data["seeds"] >= mid_seed)
                    s2 = data["scores"][v2_mask]
                    p_d2 = data["pred_dens"][v2_mask]
                    d_map = table1.get(mid, {})
                    avail_d = [float(k) for k in d_map.keys()]
                    if not avail_d:
                        continue
                    fa = 0
                    for s_val, pred_d in zip(s2, p_d2):
                        snapped_d = min(avail_d, key=lambda x: abs(x - pred_d))
                        tau = d_map[f"{snapped_d:.4f}"]
                        if s_val > tau:
                            fa += 1
                    counts.append(fa)

                counts = np.array(counts)
                mean_c = np.mean(counts)
                var_c = np.var(counts, ddof=1) if len(counts) > 1 else 0.0
                disp = var_c / (mean_c + 1e-12)
                disp_reports[cand_n0] = {
                    "dispersion": float(disp),
                    "mean": float(mean_c),
                    "var": float(var_c),
                    "max": int(np.max(counts)) if len(counts) > 0 else 0,
                    "total_fa": int(np.sum(counts))
                }
                if disp < best_disp:
                    best_disp = disp
                    best_n0 = cand_n0

            chosen_n0 = best_n0
            self._n0_disp_reports = disp_reports
        else:
            chosen_n0 = int(n0) if n0 is not None else 0
            self._n0_disp_reports = {}

        # Full calibration with chosen_n0
        class_stats = {}
        edge_w_list = {}
        for (mid, d_str, edge), data in class_samples.items():
            s_vals = data["scores"]
            log_s = np.log(np.maximum(s_vals, 1e-12))
            c_val = float(np.median(log_s))
            mad = float(np.median(np.abs(log_s - c_val)))
            w_val = 1.4826 * mad
            if w_val < 1e-8:
                w_val = 1.0
            class_stats[(mid, d_str, edge)] = (c_val, w_val, len(s_vals))
            edge_w_list.setdefault(edge, []).append(w_val)

        w_edge = {e: float(np.median(ws)) for e, ws in edge_w_list.items()}

        w_prime = {}
        for (mid, d_str, edge), (c_val, w_val, n_cal) in class_stats.items():
            wp = (float(n_cal) * w_val + chosen_n0 * w_edge[edge]) / (float(n_cal) + chosen_n0)
            w_prime[(mid, d_str, edge)] = wp

        edge_z = {}
        for (mid, d_str, edge), data in class_samples.items():
            s_vals = data["scores"]
            log_s = np.log(np.maximum(s_vals, 1e-12))
            c_val, _, _ = class_stats[(mid, d_str, edge)]
            wp = w_prime[(mid, d_str, edge)]
            z = (log_s - c_val) / wp
            edge_z.setdefault(edge, []).extend(z)

        z_star = {e: float(np.percentile(zs, 99)) for e, zs in edge_z.items() if len(zs) > 0}

        table = {}
        params = {}
        for (mid, d_str, edge), (c_val, _, _) in class_stats.items():
            wp = w_prime[(mid, d_str, edge)]
            z_star_val = z_star.get(edge, 2.326)
            tau = float(np.exp(c_val + z_star_val * wp))
            table.setdefault(mid, {})[d_str] = tau
            params.setdefault(mid, {})[d_str] = {"c": c_val, "w": wp}

        self.threshold_table = table
        self.threshold_params = params
        self.z_star = z_star
        self.w_edge = w_edge
        self.n0 = chosen_n0
        self.novelty = "class_conditional_v2"
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
            "threshold_table": self.threshold_table, "threshold_params": self.threshold_params,
            "z_star": self.z_star, "w_edge": self.w_edge, "n0": self.n0, "novelty": self.novelty,
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
        threshold_params = man.get("threshold_params", {})
        z_star = man.get("z_star", None)
        w_edge = man.get("w_edge", {})
        n0 = man.get("n0", None)
        novelty = man.get("novelty", "reconstruction_v1")
        return cls(InputSpec(**man["spec"]), enc, r["mu"], r["sd"], r["refs"], r["ref_model"], r["ref_density"],
                   [RibbonModel(**m) for m in man["models"]], man["threshold"], man["k"],
                   threshold_table=threshold_table, novelty=novelty,
                   threshold_params=threshold_params, z_star=z_star,
                   w_edge=w_edge, n0=n0)
