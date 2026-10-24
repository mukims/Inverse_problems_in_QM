"""Calibration of the inverse model (spec section 9, "Model calibration"): metrics, the per-draw evaluation and the gates.

Per test draw the model gives `n_samples` posterior samples (state and z); they are decoded to full_vectors (the 14
canonical slots of `candidates.full_vector`, with log_s and log_c) and compared with the true host:

* simulation-based calibration (SBC): the rank of the truth among the samples is uniform. Ties are real (the width w is an
  integer; slots a state switches off are 0), so the rank is the number of samples below the truth plus a uniform random
  integer in 0..#ties, and the KS test is applied to (rank + U(0,1)) / (n_samples + 1). Unconditional SBC covers log_s,
  log_c and w (every host has them); conditional SBC samples with the state fixed to the true one and covers every slot
  active in it, pooled per slot name over the draws;
* marginal coverage of the top candidate's 90% intervals, on draws whose true (edge, pattern) is the top candidate's;
* top-5 recall of the true (edge, pattern);
* candidate-probability calibration (ECE) with the capture rule of `captures` (ruling R18).

Nothing here is tuned on the test split: the thresholds are fixed by the spec.
"""
import time

import numpy as np
import torch
from scipy import stats

from shazam.inverse.candidates import PRIOR_SCALE, candidates, full_vector, mahalanobis
from shazam.inverse.encoding import CONT, SLOT, decode, encode

__all__ = ["sbc_rank", "ks_uniform_p", "coverage", "expected_calibration_error", "mahalanobis", "host_key", "captures",
           "evaluate_draw", "summarize", "run", "KS_MIN_P", "COVERAGE_RANGE", "RECALL_MIN", "ECE_MAX"]

KS_MIN_P = 0.01
COVERAGE_RANGE = (0.87, 0.93)
RECALL_MIN = 0.95
ECE_MAX = 0.05
UNCONDITIONAL = ("log_s", "log_c", "w")
_UNGATED_COVERAGE = ("w",)       # an integer: its 5th..95th percentile interval over-covers (the ends are whole widths)


def sbc_rank(truth, samples, rng=None):
    """Rank of `truth` among `samples`: #(samples < truth) + a uniform random integer in 0..#(samples == truth)."""
    rng = np.random.default_rng(0) if rng is None else rng
    samples = np.asarray(samples)
    return int((samples < truth).sum() + rng.integers(0, int((samples == truth).sum()) + 1))


def ks_uniform_p(ranks, n_samples, rng=None):
    """KS p-value of (rank + U(0,1)) / (n_samples + 1) against U(0,1); the rng (default seed 0) draws the jitter."""
    rng = np.random.default_rng(0) if rng is None else rng
    ranks = np.asarray(ranks, float)
    return float(stats.kstest((ranks + rng.uniform(size=len(ranks))) / (n_samples + 1), "uniform").pvalue)


def coverage(lo, hi, truth):
    """Share of cases with lo <= truth <= hi."""
    lo, hi, truth = (np.asarray(a, float) for a in (lo, hi, truth))
    return float(np.mean((lo <= truth) & (truth <= hi)))


def expected_calibration_error(probs, hits, n_bins=10):
    """sum over equal-width bins of [0, 1] (the last closed) of (n_bin / N) |mean(hit) - mean(prob)|; nan if N == 0."""
    probs, hits = np.asarray(probs, float), np.asarray(hits, float)
    if len(probs) == 0:
        return float("nan")
    b = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    return float(sum((b == k).sum() / len(probs) * abs(hits[b == k].mean() - probs[b == k].mean())
                     for k in range(n_bins) if (b == k).any()))


def host_key(draw):
    """(edge, site-type pattern): what a candidate's `edge` and `pattern` must equal."""
    return draw.edge, tuple(draw.params.pattern)


def captures(cand, truth, all_cands, key=None):
    """Does `cand` capture the true host (spec section 9, ruling R18)? All three must hold:

    1. `cand` has the true (edge, pattern);
    2. `cand` is the nearest of `all_cands` to the truth, measured in prior-scaled space (full_vector / PRIOR_SCALE,
       Euclidean to each candidate's mean; ties go to the first);
    3. Mahalanobis(truth; cand.mean, cand.cov) <= cand.mahal_99 (unscaled full_vector units).

    `truth` is the true host's unscaled full_vector, and `key` its (edge, pattern) (see `host_key`). Alternatively pass
    the true `Draw` as `truth` and leave `key` out: both are then taken from it.
    """
    if key is None:
        if not hasattr(truth, "params"):
            raise ValueError("pass key=(edge, pattern) when truth is a full_vector")
        truth, key = full_vector(truth), host_key(truth)
    truth = np.asarray(truth, float)
    if (cand.edge, tuple(cand.pattern)) != (key[0], tuple(key[1])):
        return False
    dist = [np.linalg.norm((truth - c.mean) / PRIOR_SCALE) for c in all_cands]
    if all_cands[int(np.argmin(dist))] is not cand:
        return False
    return bool(mahalanobis(truth, cand.mean, cand.cov) <= cand.mahal_99)


def _vectors(states, z):
    return np.array([full_vector(decode(int(k), v)) for k, v in zip(states, z)])


def _canonical(vec):
    """The truth in the units of Candidate.intervals: s and c (not their logs), w, every other slot as it is."""
    out = {n: float(vec[SLOT[n]]) for n in CONT}
    out["s"], out["c"] = float(np.exp(out.pop("log_s"))), float(np.exp(out.pop("log_c")))
    return out


def evaluate_draw(model, x, draw, n_samples=1000, steps=64, seed=0, rng=None):
    """Everything the gates need from one test draw (x: its spectrum, shape (416,) or (1, 416)).

    Returns a dict: `uncond` / `cond` ({slot name: SBC rank}), `covered` ({interval name: bool}, or None when the true
    (edge, pattern) is not the top candidate's), `recall` (bool), `probs` and `hits` (one entry per returned candidate)."""
    rng = np.random.default_rng(seed) if rng is None else rng
    x = torch.as_tensor(np.asarray(x, np.float32)).reshape(1, -1)
    true_vec = full_vector(draw)
    state, _, mask = encode(draw)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        states, z = model.sample(x, n_samples, steps)
        cstates, cz = model.sample(x, n_samples, steps, state=state)
    vec, cvec = _vectors(states, z), _vectors(cstates, cz)
    uncond = {n: sbc_rank(true_vec[SLOT[n]], vec[:, SLOT[n]], rng) for n in UNCONDITIONAL}
    cond = {n: sbc_rank(true_vec[i], cvec[:, i], rng) for i, n in enumerate(CONT) if mask[i]}
    cands, _ = candidates(states, z)
    key = host_key(draw)
    covered = None
    if cands and (cands[0].edge, tuple(cands[0].pattern)) == key:
        truth = _canonical(true_vec)
        covered = {n: bool(lo <= truth[n] <= hi) for n, (lo, hi) in cands[0].intervals.items()}
    return dict(uncond=uncond, cond=cond, covered=covered,
                recall=any((c.edge, tuple(c.pattern)) == key for c in cands),
                probs=[c.probability for c in cands], hits=[captures(c, true_vec, cands, key) for c in cands])


def _finite(v):
    return float(v) if v is not None and np.isfinite(v) else None


def summarize(records, n_samples, seed=0):
    """The gates of spec section 9 from a list of `evaluate_draw` records.

    Coverage is gated per parameter (c, s and every active continuous parameter: each must lie in 0.87 to 0.93 over the
    draws where it is reported); the integer width w is reported but not gated, because its interval ends are whole
    widths and over-cover. The pooled coverage over the gated parameters is reported as well, not gated."""
    rng = np.random.default_rng(seed)

    def sbc(key):
        names = sorted({n for r in records for n in r[key]}, key=CONT.index)
        out = {}
        for n in names:
            ranks = np.array([r[key][n] for r in records if n in r[key]])
            p = ks_uniform_p(ranks, n_samples, rng)
            out[n] = dict(ks_p=p, n=len(ranks), **{"pass": bool(p > KS_MIN_P)})
        return out

    uncond, cond = sbc("uncond"), sbc("cond")
    cov_rows = [r["covered"] for r in records if r["covered"] is not None]
    per = {}
    for n in sorted({k for c in cov_rows for k in c}, key=lambda k: (k == "c", k)):
        v = [c[n] for c in cov_rows if n in c]
        gated = n not in _UNGATED_COVERAGE
        per[n] = dict(coverage=float(np.mean(v)), n=len(v), gated=gated,
                      **{"pass": bool(COVERAGE_RANGE[0] <= np.mean(v) <= COVERAGE_RANGE[1]) if gated else None})
    pooled_v = [x for c in cov_rows for k, x in c.items() if k not in _UNGATED_COVERAGE]
    recall = float(np.mean([r["recall"] for r in records])) if records else float("nan")
    probs = np.array([p for r in records for p in r["probs"]], float)
    hits = np.array([h for r in records for h in r["hits"]], bool)
    ece = expected_calibration_error(probs, hits)
    b = np.minimum((probs * 10).astype(int), 9)
    reliability = [dict(lo=k / 10, hi=(k + 1) / 10, n=int((b == k).sum()), mean_prob=float(probs[b == k].mean()),
                        hit_rate=float(hits[b == k].mean())) for k in range(10) if (b == k).any()]
    gates = dict(
        sbc_unconditional=bool(uncond) and all(g["pass"] for g in uncond.values()),
        sbc_conditional=bool(cond) and all(g["pass"] for g in cond.values()),
        coverage=bool(per) and all(g["pass"] for g in per.values() if g["gated"]),
        top5_recall=bool(np.isfinite(recall) and recall >= RECALL_MIN),
        ece=bool(np.isfinite(ece) and ece <= ECE_MAX))
    return dict(
        n_test=len(records), n_samples=n_samples,
        sbc_unconditional=uncond, sbc_conditional=cond,
        coverage=dict(per_parameter=per, pooled=dict(coverage=_finite(np.mean(pooled_v)) if pooled_v else None,
                                                     n=len(pooled_v)), n_draws=len(cov_rows),
                      range=list(COVERAGE_RANGE), gate="per parameter"),
        top5_recall=dict(value=_finite(recall), threshold=RECALL_MIN, **{"pass": gates["top5_recall"]}),
        ece=dict(value=_finite(ece), threshold=ECE_MAX, n_candidates=len(probs), reliability=reliability,
                 **{"pass": gates["ece"]}),
        gates=gates, **{"pass": all(gates.values())})


def run(model, xs, draws, n_samples=1000, steps=64, seed=0, progress=None):
    """Evaluate every (spectrum, draw) pair; draw i is sampled with torch seed `seed + i`. Returns (summary, seconds)."""
    rng = np.random.default_rng(seed)
    t0, records = time.time(), []
    for i, (x, d) in enumerate(zip(xs, draws)):
        records.append(evaluate_draw(model, x, d, n_samples, steps, seed + i, rng))
        if progress:
            progress(i + 1, len(draws), time.time() - t0)
    return summarize(records, n_samples, seed), time.time() - t0
