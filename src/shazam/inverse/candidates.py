"""From posterior samples to candidate hosts with probabilities (spec section 7, steps 1 to 6).

The samples (a discrete state index and a 14-vector z each) are decoded once, written as full canonical vectors,
grouped by (edge, site-type pattern) only, clustered within each group, and every cluster of at least `min_size`
samples becomes a candidate represented by its medoid.
"""
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import linkage

from shazam.inverse.encoding import CONT, SLOT, STATES, active_slots, decode
from tbribbon.parent import NAMED, device_cells_for, named_material
from tbribbon.parent_data import n_impurities
from tbribbon.parent_prior import Draw, box_edge_warnings

CUT = 3.0                  # clusters whose halves have centres more than this many standard units apart are split
COV_JITTER = 1e-4
STD_FLOOR = 1e-6           # a column whose standard deviation is below this is numerically constant: scale 1
INTERVAL = (5, 95)         # percentiles of the 90% marginal intervals
HOST_SLOTS = slice(0, SLOT["log_c"])          # the host's own slots of full_vector: everything before log_c and w
_KEY = {"log_s": "s", "log_c": "c"}           # CONT slot -> interval / canonical key


def full_vector(draw):
    """The 14 CONT slots of a draw in canonical form: log_s and log_c in place of s and c, the width w, and every
    inactive or switched-off parameter (and a delta of 0) as 0; the lead's within-type hopping is 1."""
    p = draw.params
    value = {"log_s": np.log(p.s), "log_c": np.log(draw.density), "w": float(draw.width)}
    return np.array([value[n] if n in value else getattr(p, n) for n in CONT], float)


def is_realistic(draw):
    """Physical plausibility of a host (spec section 7, step 6). Placeholder: every host passes until it is defined."""
    return True


def nearest_named(draw):
    """(name, distance) of the closest named material of spec 4.3: Euclidean distance over the host slots of
    full_vector (log_s included, log_c and w excluded); ties go to the first in NAMED."""
    v = full_vector(draw)[HOST_SLOTS]
    d = [float(np.linalg.norm(v - full_vector(Draw(named_material(n), "armchair", 7, 0.01))[HOST_SLOTS]))
         for n in NAMED]
    i = int(np.argmin(d))
    return NAMED[i], d[i]


@dataclass
class Candidate:
    edge: str
    pattern: tuple
    state: int                                   # discrete state of the medoid
    draw: Draw                                   # the medoid's host
    probability: float                           # cluster size / total samples
    members: np.ndarray                          # indices into the input samples
    intervals: dict                              # name -> (5th, 95th percentile): "c", "w" and every active parameter
    n_imp_interval: tuple                        # (5th, 95th percentile) of the members' impurity counts
    mean: np.ndarray                             # over the members' full_vectors, shape (14,)
    cov: np.ndarray                              # regularised by 1e-4 I, shape (14, 14)
    nearest_material: str
    material_distance: float
    box_edge: list
    realistic: bool


def _standardise(x):
    scale = x.std(axis=0)
    scale[scale < STD_FLOOR] = 1.0
    return (x - x.mean(axis=0)) / scale


def _cluster(x):
    """Labels of a deterministic clustering of the rows of x (already standardised): Ward linkage, then from the root
    downwards a node is split while the centres of its two halves are more than CUT apart (in the Ward tree the
    merge height is sqrt(2 n1 n2 / (n1 + n2)) times that centre distance)."""
    n = len(x)
    if n == 1:
        return np.zeros(1, int)
    z = linkage(x, "ward")
    children = z[:, :2].astype(int)
    size = np.concatenate([np.ones(n), z[:, 3]])
    centre = z[:, 2] * np.sqrt((size[children[:, 0]] + size[children[:, 1]])
                               / (2 * size[children[:, 0]] * size[children[:, 1]]))

    def leaves(node):
        out, stack = [], [node]
        while stack:
            k = stack.pop()
            if k < n:
                out.append(k)
            else:
                stack.extend(children[k - n])
        return out

    labels = np.empty(n, int)
    label, todo = 0, [2 * n - 2]
    while todo:
        k = todo.pop()
        if k >= n and centre[k - n] > CUT:
            todo.extend(children[k - n][::-1])
        else:
            labels[leaves(k)] = label
            label += 1
    return labels


def candidates(states, z, min_size=20, top=5):
    """The candidate hosts of a batch of samples, most probable first, and p_unassigned.

    Samples are grouped by (edge, pattern); within a group they are written as full_vectors, standardised per column
    (columns with zero spread keep scale 1) and clustered. A cluster with at least `min_size` samples is a candidate,
    its probability its share of all samples; the medoid is the member nearest the cluster mean (standardised space).
    p_unassigned is the share of samples in clusters smaller than `min_size`. At most `top` candidates are returned;
    clusters of at least `min_size` that rank beyond `top` are NOT counted in p_unassigned, they are simply not
    returned (the update of spec section 8 treats everything not returned as the prior share p0).
    """
    states = np.asarray(states, int)
    z = np.asarray(z, float)
    total = len(states)
    draws = [decode(int(k), v) for k, v in zip(states, z)]                      # each sample decoded once
    vec = np.array([full_vector(d) for d in draws])
    groups = {}
    for i, k in enumerate(states):
        groups.setdefault((STATES[k].edge, STATES[k].pattern), []).append(i)
    clusters, unassigned = [], 0
    for rows in groups.values():
        rows = np.array(rows)
        x = _standardise(vec[rows])
        labels = _cluster(x)
        for lab in range(labels.max() + 1):
            sel = labels == lab
            if sel.sum() < min_size:
                unassigned += int(sel.sum())
                continue
            centre = x[sel].mean(axis=0)
            medoid = rows[sel][int(np.argmin(((x[sel] - centre) ** 2).sum(axis=1)))]
            clusters.append((rows[sel], int(medoid)))
    clusters.sort(key=lambda c: (-len(c[0]), c[0][0]))
    atoms = {}                                                                  # (pattern, edge, width) -> atoms in the device
    return [_candidate(c, m, states, draws, vec, total, atoms) for c, m in clusters[:top]], unassigned / total


def _n_imp(draw, atoms):
    """n_impurities of a draw, with the (slow) device atom count cached per (pattern, edge, width)."""
    key = (draw.params.pattern, draw.edge, draw.width)
    if key not in atoms:
        atoms[key] = n_impurities(draw._replace(density=1.0), device_cells_for(draw.params, draw.edge))
    return max(1, round(draw.density * atoms[key]))


def _candidate(members, medoid, states, draws, vec, total, atoms):
    draw, state = draws[medoid], int(states[medoid])
    v = vec[members]
    canonical = v.copy()
    canonical[:, [SLOT["log_s"], SLOT["log_c"]]] = np.exp(canonical[:, [SLOT["log_s"], SLOT["log_c"]]])
    mask = active_slots(state)
    intervals = {_KEY.get(n, n): tuple(float(q) for q in np.percentile(canonical[:, i], INTERVAL))
                 for i, n in enumerate(CONT) if mask[i]}
    counts = [_n_imp(draws[i], atoms) for i in members]
    n_imp = tuple(int(round(q)) for q in np.percentile(counts, INTERVAL))
    cov = np.cov(v, rowvar=False) if len(members) > 1 else np.zeros((len(CONT),) * 2)
    material, distance = nearest_named(draw)
    return Candidate(STATES[state].edge, STATES[state].pattern, state, draw, len(members) / total, members, intervals,
                     n_imp, v.mean(axis=0), cov + COV_JITTER * np.eye(len(CONT)), material, distance,
                     box_edge_warnings(draw), is_realistic(draw))
