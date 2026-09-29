#!/usr/bin/env python
"""
material_atlas.py — Label-free material atlas (stages 1-2 of the sensor pipeline)
================================================================================

Places a transmission signature T(E) in an autoencoder latent space and finds
the closest known material (type + width), flags signatures that match nothing
known, and gives a first rough concentration estimate. Stage 3 (material-
specific concentration models) then takes over.

Label-free by construction: every spectrum gets the SAME input transform,
computable without knowing what it is (no per-material pristine division):

    X = log1p(clip(round(T, 3), 0, CAP)) / log1p(CAP)      over E = 0-2.99 eV

Split by configuration seed in every material (LOGBOOK Bug #7): seeds [0, 0.7K)
train, [0.7K, 0.85K) validation (novelty threshold), [0.85K, K) test.

Compared against baselines on the same input: physics descriptors + decision
tree, library matching, logistic regression, and k-NN on PCA (what the AE adds).

Usage
    python material_atlas.py --per-conc 200 --epochs 2 --out-dir /tmp/atlas_smoke   # smoke test
    python material_atlas.py --loo                                                   # full run + leave-one-out novelty
Then, from Python:
    from material_atlas import MaterialAtlas
    atlas = MaterialAtlas.load("results")
    atlas.locate(T)   # T: raw transmission, >= 300 channels from E = 0 at 0.01 eV
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "notebooks" / "square_lattice"))
from square_autoencoder import Conv1dAutoencoder  # noqa: E402

AGNR_DIR = "/run/media/shardul/storage/machine_learning/transmission_data/transmission_results/consolidated_data"
SQUARE_DIR = "~/transmissions_sq/size_10_combined"
CLASSES = ["7-AGNR", "9-AGNR", "Square-10"]
TYPES = ["AGNR", "AGNR", "Square"]
CAP = 20.0          # G0; above every channel count in the atlas, so plateaus survive and spikes are squashed
E_STEP = 0.01


# =============================================================================
# 1. Label-free input and data
# =============================================================================
def to_input(T):
    """Identical for every spectrum: needs no knowledge of the material."""
    T = np.clip(np.round(np.asarray(T, dtype=np.float64), 3), 0.0, CAP)
    return (np.log1p(T) / np.log1p(CAP)).astype(np.float32)


def load(args):
    L, K = args.spectrum_len, args.per_conc
    X, cls, conc, seed = [], [], [], []
    for k, f in enumerate(("size_7.npy", "size_9.npy")):
        mm = np.load(os.path.join(args.agnr_dir, f), mmap_mode="r")
        for i, c in enumerate(range(2, 2 * mm.shape[0] + 1, 2)):
            X.append(to_input(mm[i, :K, :L])); cls.append(np.full(K, k)); conc.append(np.full(K, c, float))
            seed.append(np.arange(K))                      # row index == configuration seed
    sq = os.path.expanduser(args.square_dir)
    for c in range(5, 91, 5):
        rows = np.load(os.path.join(sq, f"conc_{c}.npy"), mmap_mode="r")
        cfg = np.loadtxt(os.path.join(sq, f"conc_{c}_meta.csv"), delimiter=",", skiprows=1, dtype=int)[:, 1]
        pick = np.where(cfg < K)[0]
        X.append(to_input(rows[pick, :L])); cls.append(np.full(len(pick), 2)); conc.append(np.full(len(pick), c, float))
        seed.append(cfg[pick])
    X, cls, conc, seed = (np.concatenate(a) for a in (X, cls, conc, seed))
    split = np.where(seed < int(0.70 * K), 0, np.where(seed < int(0.85 * K), 1, 2))   # 0 train, 1 val, 2 test
    return X, cls, conc, split


# =============================================================================
# 2. Autoencoder
# =============================================================================
def padded_len(L):
    return int(np.ceil(L / 8) * 8)


def to_tensor(X, Lp):
    left = (Lp - X.shape[1]) // 2
    return nn.functional.pad(torch.from_numpy(X).unsqueeze(1), (left, Lp - X.shape[1] - left))


def train_ae(X_tr, X_va, args, tag="atlas"):
    Lp = padded_len(X_tr.shape[1])
    model = Conv1dAutoencoder(latent_dim=args.latent, seq_len=Lp)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(to_tensor(X_tr, Lp)),
                                         batch_size=args.batch_size, shuffle=True)
    xv = to_tensor(X_va, Lp)
    best, best_state, wait, hist = float("inf"), None, 0, []
    epochs = args.loo_epochs if tag != "atlas" else args.epochs
    for ep in range(1, epochs + 1):
        t0 = time.time(); model.train()
        for (xb,) in loader:
            loss = nn.functional.mse_loss(model(xb)[0], xb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val = sum(nn.functional.mse_loss(model(xv[i:i + 4096])[0], xv[i:i + 4096], reduction="sum").item()
                      for i in range(0, len(xv), 4096)) / xv.numel()
        hist.append(val); sched.step(val)
        mark = ""
        if best - val > 1e-7:
            best, best_state, wait, mark = val, {k: v.clone() for k, v in model.state_dict().items()}, 0, " *"
        else:
            wait += 1
        print(f"[{tag}] epoch {ep:3d}  held-out mse {val:.6f}  lr {opt.param_groups[0]['lr']:.1e}  {time.time() - t0:.0f}s{mark}", flush=True)
        if wait >= args.patience:
            print(f"[{tag}] early stop at epoch {ep}", flush=True); break
    model.load_state_dict(best_state); model.eval()
    return model, hist


@torch.no_grad()
def embed(model, X):
    Lp = padded_len(X.shape[1]); left = (Lp - X.shape[1]) // 2
    Z, err = [], []
    for i in range(0, len(X), 4096):
        xb = to_tensor(X[i:i + 4096], Lp)
        r, z = model(xb)
        Z.append(z.numpy()); err.append(((r - xb)[:, 0, left:left + X.shape[1]] ** 2).mean(1).numpy())
    return np.concatenate(Z), np.concatenate(err)


# =============================================================================
# 3. Atlas: k-NN retrieval + novelty
# =============================================================================
def balanced_refs(cls, mask, n_per_class, rng):
    idx = []
    for k in np.unique(cls[mask]):
        pool = np.where(mask & (cls == k))[0]
        idx.append(rng.choice(pool, min(n_per_class, len(pool)), replace=False))
    return np.concatenate(idx)


def retrieve(F_ref, y_ref, c_ref, F_q, k, n_jobs):
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=n_jobs).fit(F_ref, y_ref)
    dist, nn_idx = knn.kneighbors(F_q)
    proba = knn.predict_proba(F_q); pred = knn.classes_[proba.argmax(1)]
    same = y_ref[nn_idx] == pred[:, None]
    c_est = np.array([np.median(c_ref[nn_idx[i]][same[i]]) for i in range(len(F_q))])
    return pred, proba.max(1), dist.mean(1), c_est


# =============================================================================
# 4. Baselines on the same label-free input
# =============================================================================
def descriptors(X):
    """Physics fingerprints: band-onset energy, plateau level, mean transmission, low-energy weight."""
    thr = np.log1p(0.05) / np.log1p(CAP)
    above = X > thr
    onset = np.where(above.any(1), above.argmax(1), X.shape[1]) * E_STEP
    return np.column_stack([onset, np.percentile(X, 95, axis=1), X.mean(1), X[:, :50].mean(1)])


def library_match(X_ref, y_ref, c_ref, X_q):
    keys = sorted({(int(a), float(b)) for a, b in zip(y_ref, c_ref)})
    lib = np.stack([X_ref[(y_ref == a) & (c_ref == b)].mean(0) for a, b in keys])
    lab = np.array([a for a, _ in keys])
    pred = np.empty(len(X_q), int)
    for i in range(0, len(X_q), 4096):
        d = ((X_q[i:i + 4096, None, :] - lib[None]) ** 2).mean(2)
        pred[i:i + 4096] = lab[d.argmin(1)]
    return pred


def acc(pred, y):
    return float((pred == y).mean() * 100)


# =============================================================================
# 5. Reusable tool
# =============================================================================
class MaterialAtlas:
    """Load a saved atlas and locate new signatures: material, confidence, novelty, rough c."""

    def __init__(self, model, scaler_mean, scaler_scale, refs, ref_cls, ref_conc, threshold, k, spectrum_len):
        self.model, self.mu, self.sd = model, scaler_mean, scaler_scale
        self.refs, self.ref_cls, self.ref_conc = refs, ref_cls, ref_conc
        self.threshold, self.k, self.L = threshold, k, spectrum_len

    @classmethod
    def load(cls, out_dir):
        out_dir = Path(out_dir)
        ck = torch.load(out_dir / "material_atlas.pt", weights_only=True)
        model = Conv1dAutoencoder(latent_dim=ck["latent"], seq_len=padded_len(ck["spectrum_len"]))
        model.load_state_dict(ck["model_state"]); model.eval()
        a = np.load(out_dir / "atlas_refs.npz")
        return cls(model, a["mu"], a["sd"], a["refs"], a["cls"], a["conc"], float(a["threshold"]), int(a["k"]), ck["spectrum_len"])

    def locate(self, T):
        T = np.atleast_2d(np.asarray(T))[:, :self.L]
        Z, recon_err = embed(self.model, to_input(T))
        pred, conf, novelty, c_est = retrieve(self.refs, self.ref_cls, self.ref_conc, (Z - self.mu) / self.sd, self.k, 1)
        return [{"material": CLASSES[p], "type": TYPES[p], "confidence": float(cf), "novelty_score": float(nv),
                 "unknown": bool(nv > self.threshold), "rough_concentration": float(ce), "reconstruction_error": float(re)}
                for p, cf, nv, ce, re in zip(pred, conf, novelty, c_est, recon_err)]


# =============================================================================
# 6. Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agnr-dir", default=AGNR_DIR)
    ap.add_argument("--square-dir", default=SQUARE_DIR)
    ap.add_argument("--per-conc", type=int, default=3000, help="configuration seeds per concentration per material")
    ap.add_argument("--spectrum-len", type=int, default=300, help="channels from E = 0 (300 = 0-2.99 eV)")
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--loo-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k", type=int, default=15, help="neighbours for retrieval")
    ap.add_argument("--refs-per-class", type=int, default=20000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--loo", action="store_true", help="also run leave-one-material-out novelty tests")
    ap.add_argument("--out-dir", default=str(SCRIPT_DIR / "results"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(0); rng = np.random.default_rng(0)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    X, y, c, split = load(args)
    tr, va, te = split == 0, split == 1, split == 2
    print(f"Loaded {len(X):,} spectra ({', '.join(f'{n} {(y == k).sum():,}' for k, n in enumerate(CLASSES))}) | "
          f"train {tr.sum():,} / val {va.sum():,} / test {te.sum():,} (held-out seeds)", flush=True)
    y_type = np.array([0, 0, 1])[y]

    # ---- autoencoder + atlas ----
    t0 = time.time()
    model, hist = train_ae(X[tr], X[va], args)
    t_train = time.time() - t0
    Z, rec = embed(model, X)
    scaler = StandardScaler().fit(Z[tr]); Zs = scaler.transform(Z)
    ref = balanced_refs(y, tr, args.refs_per_class, rng)
    pred_va, _, nov_va, _ = retrieve(Zs[ref], y[ref], c[ref], Zs[va], args.k, args.threads)
    threshold = float(np.percentile(nov_va, 99))
    pred, conf, nov, c_est = retrieve(Zs[ref], y[ref], c[ref], Zs[te], args.k, args.threads)
    yt, ct = y[te], c[te]

    res = {"n": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())}, "train_time_s": round(t_train),
           "epochs_run": len(hist), "input": f"log1p(clip(round(T,3),0,{CAP}))/log1p({CAP}), {args.spectrum_len} channels"}
    res["reconstruction_R2"] = {n: float(1 - rec[te & (y == k)].mean() / X[te & (y == k)].var()) for k, n in enumerate(CLASSES)}
    res["atlas_AE_kNN"] = {"material_acc": acc(pred, yt), "type_acc": acc(np.array([0, 0, 1])[pred], y_type[te]),
                           "per_class_acc": {n: acc(pred[yt == k], yt[yt == k]) for k, n in enumerate(CLASSES)},
                           "confusion": [[int(((yt == a) & (pred == b)).sum()) for b in range(3)] for a in range(3)],
                           "rough_conc_MAE": {n: float(np.abs(c_est[yt == k] - ct[yt == k]).mean()) for k, n in enumerate(CLASSES)},
                           "novelty_threshold": threshold, "false_alarm_rate_test": float((nov > threshold).mean() * 100)}
    bands = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    res["atlas_AE_kNN"]["acc_by_conc_band"] = {n: {f"{lo + 1}-{hi}": acc(pred[(yt == k) & (ct > lo) & (ct <= hi)], k)
                                                   for lo, hi in bands if ((yt == k) & (ct > lo) & (ct <= hi)).any()}
                                               for k, n in enumerate(CLASSES)}
    print("\nAtlas (AE + k-NN):", json.dumps(res["atlas_AE_kNN"], indent=1), flush=True)

    # ---- baselines on the same label-free input ----
    base = {}
    sub = balanced_refs(y, tr, args.refs_per_class, rng)
    D = descriptors(X)
    tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(D[sub], y[sub])
    base["descriptors_tree"] = acc(tree.predict(D[te]), yt)
    base["library_matching"] = acc(library_match(X[tr], y[tr], c[tr], X[te]), yt)
    sx = StandardScaler().fit(X[sub])
    lr = LogisticRegression(max_iter=1000).fit(sx.transform(X[sub]), y[sub])
    base["logistic_regression"] = acc(lr.predict(sx.transform(X[te])), yt)
    pca = PCA(args.latent, random_state=0).fit(X[tr]); P = StandardScaler().fit(pca.transform(X[tr]))
    Ps = P.transform(pca.transform(X))
    base[f"PCA{args.latent}_kNN"] = acc(retrieve(Ps[ref], y[ref], c[ref], Ps[te], args.k, args.threads)[0], yt)
    base["tree_rules"] = [f"depth {tree.get_depth()}", "features: onset energy, 95th-pct level, mean level, mean below 0.5 eV",
                          f"importances {np.round(tree.feature_importances_, 3).tolist()}"]
    res["baselines_material_acc"] = base
    print("Baselines (material accuracy on test):", json.dumps(base, indent=1), flush=True)

    # ---- leave-one-material-out novelty ----
    if args.loo:
        res["leave_one_out"] = {}
        for h, hname in ((1, "9-AGNR"), (2, "Square-10")):
            keep = y != h
            m, _ = train_ae(X[tr & keep], X[va & keep], args, tag=f"loo-{hname}")
            Zh, rh = embed(m, X)
            sh = StandardScaler().fit(Zh[tr & keep]); Zhs = sh.transform(Zh)
            rf = balanced_refs(y, tr & keep, args.refs_per_class, rng)
            _, _, nv_va, _ = retrieve(Zhs[rf], y[rf], c[rf], Zhs[va & keep], args.k, args.threads)
            thr = float(np.percentile(nv_va, 99))
            _, _, nv_te, _ = retrieve(Zhs[rf], y[rf], c[rf], Zhs[te], args.k, args.threads)
            unseen = y[te] == h
            res["leave_one_out"][hname] = {
                "AUROC_knn_distance": float(roc_auc_score(unseen, nv_te)),
                "AUROC_reconstruction": float(roc_auc_score(unseen, rh[te])),
                "unseen_flagged_pct": float((nv_te[unseen] > thr).mean() * 100),
                "known_false_alarm_pct": float((nv_te[~unseen] > thr).mean() * 100)}
            print(f"LOO {hname}:", res["leave_one_out"][hname], flush=True)
            np.save(out / f"loo_novelty_{hname}.npy", np.column_stack([nv_te, unseen, [thr] * len(nv_te)]))

    # ---- save atlas + tool ----
    torch.save({"model_state": model.state_dict(), "latent": args.latent, "spectrum_len": args.spectrum_len,
                "classes": CLASSES, "cap": CAP, "args": vars(args)}, out / "material_atlas.pt")
    np.savez(out / "atlas_refs.npz", refs=Zs[ref], cls=y[ref], conc=c[ref], mu=scaler.mean_, sd=scaler.scale_,
             threshold=threshold, k=args.k)
    json.dump(res, open(out / "atlas_metrics.json", "w"), indent=2)

    # ---- plots ----
    s = rng.choice(np.where(te)[0], min(30000, te.sum()), replace=False)
    pc = PCA(2, random_state=0).fit(Zs[tr]).transform(Zs[s])
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
    for k, n in enumerate(CLASSES):
        m = y[s] == k; ax[0].scatter(pc[m, 0], pc[m, 1], s=2, alpha=0.5, label=n)
    ax[0].legend(markerscale=6); ax[0].set_title("Material atlas (PCA-2 of latent), coloured by material")
    sc = ax[1].scatter(pc[:, 0], pc[:, 1], c=c[s], s=2, cmap="viridis")
    plt.colorbar(sc, ax=ax[1], label="c (impurity count)"); ax[1].set_title("Same map, coloured by concentration")
    for k, n in enumerate(CLASSES):
        d = res["atlas_AE_kNN"]["acc_by_conc_band"][n]
        ax[2].plot(list(d.keys()), list(d.values()), "o-", label=n)
    ax[2].set_ylabel("material accuracy (%)"); ax[2].set_xlabel("concentration band"); ax[2].legend()
    ax[2].set_title("Retrieval accuracy by concentration (held-out seeds)")
    plt.tight_layout(); plt.savefig(out / "atlas_map.png", dpi=130); plt.close()
    print(f"\nSaved atlas, metrics and plots to {out}")


if __name__ == "__main__":
    main()
