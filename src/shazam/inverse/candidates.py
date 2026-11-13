"""From posterior samples to candidate hosts with probabilities (spec section 7, steps 1 to 6).

The samples (a discrete state index and a 14-vector z each) are decoded once, written as full canonical vectors,
grouped by (edge, site-type pattern) only, clustered within each group, and every cluster of at least `min_size`
samples becomes a candidate represented by its medoid.
"""
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import linkage

from shazam.inverse.encoding import CONT, SLOT, STATES, active_slots, decode, state_index, state_of
from shazam.inverse.reward import FAMILY_OF
from tbribbon.parent import NAMED, device_cells_for, named_material
from tbribbon.parent_data import n_impurities
from tbribbon.parent_prior import RANGES, Draw, box_edge_warnings

# Clustering happens in prior units: every slot of full_vector is divided by the standard deviation of that slot's prior
# continuous distribution, so a distance of 1 is "one prior standard deviation of one slot", whatever the group holds.
# A Ward-tree node is split into its two children only when all three hold (ruling R17):
#   the children's centres are more than SPLIT_DISTANCE apart (smallest difference worth a candidate: one slot alone,
#       Delta >= 0.28, s ratio >= 1.29, eps >= 0.58, within-type t >= 0.115, t_AA >= 0.039, cross t >= 0.14,
#       c ratio >= 1.54, w >= 3 steps),
#   that distance exceeds SPLIT_RATIO times the pooled RMS spread of the two children along the line joining the
#       centres (the children are separated, not two halves of one blob),
#   both children hold at least MIN_CHILD samples.
SPLIT_DISTANCE = 0.5
SPLIT_RATIO = 4.0
MIN_CHILD = 5
COV_JITTER = 1e-4
INTERVAL = (5, 95)         # percentiles of the 90% marginal intervals
HOST_SLOTS = slice(0, SLOT["log_c"])          # the host's own slots of full_vector: everything before log_c and w
_KEY = {"log_s": "s", "log_c": "c"}           # CONT slot -> interval / canonical key


def _prior_scale():
    """Standard deviation of each CONT slot's prior continuous distribution (interval width / sqrt(12); the integer width
    w has (n^2 - 1) / 12 for n values)."""
    width = {"log_s": np.ptp(np.log(RANGES["s"])), "log_c": np.ptp(np.log(RANGES["c"])),
             "delta": RANGES["delta_min"][1] - RANGES["delta_min"][0], "t_AA": np.ptp(RANGES["t_AA"])}
    width.update({n: np.ptp(RANGES["eps"]) for n in ("eps_C", "eps_H_mean", "eps_K")})
    width.update({n: np.ptp(RANGES["t_within"]) for n in ("t_CC", "t_AB", "t_KK")})
    width.update({n: np.ptp(RANGES["t_cross"]) for n in ("t_CH", "t_CK", "t_HK")})
    scale = np.array([width[n] / np.sqrt(12) if n != "w" else 0.0 for n in CONT])
    n_w = RANGES["w"][1] - RANGES["w"][0] + 1
    scale[SLOT["w"]] = np.sqrt((n_w ** 2 - 1) / 12)
    return scale


PRIOR_SCALE = _prior_scale()


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


def mahalanobis(x, mean, cov):
    """Mahalanobis distance of x (a vector, or the rows of a matrix) from `mean` under `cov`."""
    d = np.atleast_2d(np.asarray(x, float) - mean)
    out = np.sqrt(np.maximum(np.einsum("ij,ij->i", d, np.linalg.solve(cov, d.T).T), 0.0))
    return out if np.ndim(x) > 1 else float(out[0])


@dataclass
class Candidate:
    """One candidate host. `state` is the discrete state of the decoded medoid host (state_of(draw)), which can differ from
    the state the medoid was sampled in when decoding switches a value off (ruling R11: an empty Delta interval gives
    Delta = 0). `intervals` hold only the slots active in that state (members whose state switches a slot off contribute
    0 to it), so the medoid lying inside its intervals is likely but not guaranteed in general."""
    edge: str
    pattern: tuple
    state: int                                   # discrete state of the medoid's decoded host
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
    mahal_99: float                              # 99th percentile of the members' Mahalanobis distances from `mean` under `cov`
    #                                              (unscaled full_vector units, ruling R18; calibration.captures uses it)
    z_members: np.ndarray = None                 # raw logit vectors of cluster members, shape (len(members), 14)
    member_widths: np.ndarray = None             # integer widths of cluster members, shape (len(members),)


def _cluster(x):
    """Labels of a deterministic clustering of the rows of x (in prior units): Ward linkage, then from the root downwards a
    node is split when its two children pass the separation test described at SPLIT_DISTANCE."""
    n = len(x)
    if n == 1:
        return np.zeros(1, int)
    z = linkage(x, "ward")
    children = z[:, :2].astype(int)

    def leaves(node):
        out, stack = [], [node]
        while stack:
            k = stack.pop()
            if k < n:
                out.append(k)
            else:
                stack.extend(children[k - n])
        return out

    def separated(node):
        a, b = (leaves(k) for k in children[node - n])
        if min(len(a), len(b)) < MIN_CHILD:
            return False
        ca, cb = x[a].mean(axis=0), x[b].mean(axis=0)
        d = float(np.linalg.norm(cb - ca))
        if d <= SPLIT_DISTANCE:
            return False
        u = (cb - ca) / d
        pa, pb = (x[a] - ca) @ u, (x[b] - cb) @ u
        spread = np.sqrt((pa @ pa + pb @ pb) / (len(a) + len(b)))
        return d > SPLIT_RATIO * spread

    labels = np.empty(n, int)
    label, todo = 0, [2 * n - 2]
    while todo:
        k = todo.pop()
        if k >= n and separated(k):
            todo.extend(children[k - n][::-1])
        else:
            labels[leaves(k)] = label
            label += 1
    return labels


def candidates(states, z, min_size=20, top=5):
    """The candidate hosts of a batch of samples, most probable first, and p_unassigned.

    Samples are grouped by (edge, pattern); within a group they are written as full_vectors, divided slot by slot by
    the prior standard deviation (PRIOR_SCALE) and clustered. A cluster with at least `min_size` samples is a candidate,
    its probability its share of all samples; the medoid is the member nearest the cluster mean (prior units).
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
        x = vec[rows] / PRIOR_SCALE
        labels = _cluster(x)
        for lab in range(labels.max() + 1):
            sel = labels == lab
            if sel.sum() < min_size:
                unassigned += int(sel.sum())
                continue
            fams = FAMILY_OF[states[rows[sel]]]
            modal_fam = int(np.bincount(fams).argmax())
            widths = np.array([draws[r].width for r in rows[sel]])
            modal_w = int(np.bincount(widths).argmax())
            cand_pool = np.flatnonzero((fams == modal_fam) & (widths == modal_w))
            if len(cand_pool) == 0:
                cand_pool = np.flatnonzero(fams == modal_fam)
            x_sel = x[sel]
            centre = np.median(x_sel[cand_pool], axis=0)
            best_idx = cand_pool[int(np.argmin(((x_sel[cand_pool] - centre) ** 2).sum(axis=1)))]
            medoid = rows[sel][best_idx]
            clusters.append((rows[sel], int(medoid)))
    clusters.sort(key=lambda c: (-len(c[0]), c[0][0]))
    atoms = {}                                                                  # (pattern, edge, width) -> atoms in the device
    return [_candidate(c, m, states, z, draws, vec, total, atoms) for c, m in clusters[:top]], unassigned / total


def _n_imp(draw, atoms):
    """n_impurities of a draw, with the (slow) device atom count cached per (pattern, edge, width)."""
    key = (draw.params.pattern, draw.edge, draw.width)
    if key not in atoms:
        atoms[key] = n_impurities(draw._replace(density=1.0), device_cells_for(draw.params, draw.edge))
    return max(1, round(draw.density * atoms[key]))


def _candidate(members, medoid, states, z, draws, vec, total, atoms):
    draw = draws[medoid]
    state = state_index(state_of(draw))                  # agrees with the parameters reported (R11)
    v = vec[members]
    canonical = v.copy()
    canonical[:, [SLOT["log_s"], SLOT["log_c"]]] = np.exp(canonical[:, [SLOT["log_s"], SLOT["log_c"]]])
    mask = active_slots(state)
    active_mat = np.array([active_slots(int(states[m])) for m in members])
    intervals = {}
    for i, n in enumerate(CONT):
        if not mask[i]:
            continue
        has_active = active_mat[:, i]
        col = canonical[has_active, i] if has_active.sum() >= MIN_CHILD else canonical[:, i]
        intervals[_KEY.get(n, n)] = tuple(float(q) for q in np.percentile(col, INTERVAL))
    counts = [_n_imp(draws[i], atoms) for i in members]
    n_imp = tuple(int(round(q)) for q in np.percentile(counts, INTERVAL))
    cov = np.cov(v, rowvar=False) if len(members) > 1 else np.zeros((len(CONT),) * 2)
    cov = cov + COV_JITTER * np.eye(len(CONT))
    mahal_99 = float(np.percentile(mahalanobis(v, v.mean(axis=0), cov), 99))
    material, distance = nearest_named(draw)
    z_members = z[members]
    member_widths = np.array([draws[m].width for m in members], dtype=int)
    return Candidate(STATES[state].edge, STATES[state].pattern, state, draw, len(members) / total, members, intervals,
                     n_imp, v.mean(axis=0), cov, material, distance,
                     box_edge_warnings(draw), is_realistic(draw), mahal_99, z_members, member_widths)
