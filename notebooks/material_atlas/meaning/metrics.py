# notebooks/material_atlas/meaning/metrics.py
"""How well an embedding identifies known ribbons, and how physically it places spectra it never saw."""
from collections import Counter

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


def _finite(Z, what):
    if not np.all(np.isfinite(Z)):
        raise ValueError(f"embedding of {what} contains non-finite values")
    return Z


def _auroc(low, high):
    """AUROC that `high` lies farther than `low` (1.0 = always farther)."""
    return round(float(roc_auc_score(np.r_[np.zeros(len(low)), np.ones(len(high))], np.r_[low, high])), 4)


def evaluate_embedding(emb, refs, known, untrained, unseen, clean_dist, names, k=15):
    if not untrained or not unseen:
        raise ValueError("need at least one untrained and one unseen group")
    missing = sorted((set(untrained) | set(unseen)) - set(clean_dist))
    if missing:
        raise ValueError(f"clean distances missing for {missing}")

    R = [_finite(emb(X), f"references of {names[c]}") for c, X in enumerate(refs)]
    Ry = np.concatenate([np.full(len(Z), c) for c, Z in enumerate(R)])
    R = np.concatenate(R)
    index = NearestNeighbors(n_neighbors=min(k, len(R))).fit(R)

    def query(Z):
        dist, idx = index.kneighbors(Z)
        votes = np.array([Counter(row.tolist()).most_common(1)[0][0] for row in Ry[idx]])
        return dist[:, 0], votes

    d_known, correct = [], []
    for c, X in enumerate(known):
        d1, votes = query(_finite(emb(X), f"test spectra of {names[c]}"))
        d_known.append(d1)
        correct.append(votes == c)
    d_known, correct = np.concatenate(d_known), np.concatenate(correct)
    scale = float(np.median(d_known))
    if scale <= 0:
        raise ValueError("known test spectra coincide with references; distances cannot be normalised")

    centroids = np.stack([R[Ry == c].mean(axis=0) for c in range(len(names))])
    nearest, placement, graded = {}, {}, {}
    for name, X in {**untrained, **unseen}.items():
        Z = _finite(emb(X), name)
        d1, votes = query(Z)
        nearest[name] = d1
        placement[name] = [(names[c], int(n)) for c, n in Counter(votes.tolist()).most_common(4)]
        to_centroids = np.linalg.norm(centroids - Z.mean(axis=0), axis=1)
        graded[name] = round(float(spearmanr(to_centroids, clean_dist[name]).correlation), 3)

    d_untrained = np.concatenate([nearest[n] for n in untrained])
    d_unseen = np.concatenate([nearest[n] for n in unseen])
    return {
        "known_identification_pct": round(100 * float(correct.mean()), 2),
        "median_nearest_distance_relative_to_known": {
            "known": 1.0, **{n: round(float(np.median(d) / scale), 3) for n, d in nearest.items()}},
        "auroc_untrained_vs_known": _auroc(d_known, d_untrained),
        "auroc_unseen_vs_untrained": _auroc(d_untrained, d_unseen),
        "placement": placement,
        "spearman_embedding_vs_clean_distance": graded,
    }
