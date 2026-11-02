"""Calibration of the inverse model (spec section 9, "Model calibration"): metrics, the per-draw evaluation and the gates.

Per test draw the model gives `n_samples` posterior samples (state and z); they are decoded to full_vectors (the 14
canonical slots of `candidates.full_vector`, with log_s and log_c) and compared with the true host:

* simulation-based calibration (SBC): the rank of the truth among the samples is uniform. Ties are real (the width w is an
  integer; slots a state switches off are 0), so the rank is the number of samples below the truth plus a uniform random
  integer in 0..#ties, and the KS test is applied to (rank + U(0,1)) / (n_samples + 1). Unconditional SBC covers log_s,
  log_c and w (every host has them); conditional SBC samples with the state fixed to the true one and covers every slot
  active in it, pooled per slot name over the draws;
* marginal coverage of the top candidate's 90% intervals (ruling R20), on draws whose top candidate has the true
  (edge, pattern) AND is the candidate nearest to the truth in prior-scaled space (capture conditions 1 and 2, without
  the ellipsoid): a perfectly calibrated model whose true key holds several clusters would otherwise under-cover, because
  the truth often belongs to a lower-ranked cluster of the same key. The key-only figure is kept as a diagnostic. The
  intervals are those of the slots active in the top candidate's MEDOID state (members in other states contribute 0 to a
  slot their state switches off). The integer width w is reported but not gated (ruling R19): its interval ends are whole
  widths, so w coverage cannot tell a calibrated from an over-confident model; w calibration is gated through SBC;
* top-5 recall of the true (edge, pattern);
* candidate-probability calibration (ECE) with the capture rule of `captures` (ruling R18).

Nothing here is tuned on the test split: the thresholds are fixed by the spec. The spec's per-gate flags and the overall
`pass` are unchanged; the report adds noise-aware verdicts as reporting only (ruling R21): Bonferroni-adjusted KS p-values
and binomial tests of the coverage hits against 0.90, because a perfect model fails some of the 17+ gates by chance.
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
_UNGATED_COVERAGE = ("w",)       # an integer: its 5th..95th percentile interval ends are whole widths (ruling R19)
ALPHA = 0.01                     # level of the noise-aware verdicts, after the Bonferroni adjustment
COVERAGE_TARGET = 0.90
W_NOTE = ("w is reported but not gated (ruling R19): the 5th..95th percentile interval of an integer has whole-number "
          "ends and cannot separate a calibrated from an over-confident model; w calibration is gated through SBC.")


def sbc_rank(truth, samples, rng):
    """Rank of `truth` among `samples`: #(samples < truth) + a uniform random integer in 0..#(samples == truth).

    `rng` is required (a numpy Generator): a fresh default would break every tie the same way on every call."""
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


def _nearest(truth, cands):
    """Index of the candidate nearest to the truth in prior-scaled space (Euclidean to the means; ties to the first)."""
    return int(np.argmin([np.linalg.norm((truth - c.mean) / PRIOR_SCALE) for c in cands]))


def captures(cand, truth, all_cands, key=None):
    """Does `cand` capture the true host (spec section 9, ruling R18)? All three must hold:

    1. `cand` has the true (edge, pattern);
    2. `cand` is the nearest of `all_cands` to the truth, measured in prior-scaled space (full_vector / PRIOR_SCALE,
       Euclidean to each candidate's mean; ties go to the first);
    3. Mahalanobis(truth; cand.mean, cand.cov) <= cand.mahal_99 (unscaled full_vector units). mahal_99 is in-sample (the
       members' own distances, under a covariance estimated from them), so a small cluster captures slightly less than 99%
       of the draws it truly represents.

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
    if all_cands[_nearest(truth, all_cands)] is not cand:
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

    Returns a dict: `uncond` / `cond` ({slot name: SBC rank}); `covered` ({interval name: bool} of the top candidate, or None
    unless it has the true (edge, pattern) AND is the nearest candidate to the truth: the coverage gate's conditioning,
    ruling R20); `covered_key_only` (the same, conditioning on the key alone: a diagnostic); `recall` (bool); `probs` and
    `hits` (one entry per returned candidate). Values are plain Python types (JSON-ready)."""
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
    covered = covered_key_only = None
    if cands and (cands[0].edge, tuple(cands[0].pattern)) == key:
        truth = _canonical(true_vec)
        covered_key_only = {n: bool(lo <= truth[n] <= hi) for n, (lo, hi) in cands[0].intervals.items()}
        if _nearest(true_vec, cands) == 0:
            covered = covered_key_only
    return dict(uncond=uncond, cond=cond, covered=covered, covered_key_only=covered_key_only,
                recall=any((c.edge, tuple(c.pattern)) == key for c in cands),
                probs=[c.probability for c in cands], hits=[captures(c, true_vec, cands, key) for c in cands])


def _finite(v):
    return float(v) if v is not None and np.isfinite(v) else None


def _coverage_table(rows):
    """({parameter: coverage, n, gated}, pooled over the gated parameters) from the per-draw {name: bool} dicts."""
    per = {}
    for n in sorted({k for c in rows for k in c}, key=lambda k: (k == "c", k)):
        v = [c[n] for c in rows if n in c]
        per[n] = dict(coverage=float(np.mean(v)), n=len(v), hits=int(np.sum(v)), gated=n not in _UNGATED_COVERAGE)
    pooled_v = [x for c in rows for k, x in c.items() if k not in _UNGATED_COVERAGE]
    return per, dict(coverage=_finite(np.mean(pooled_v)) if pooled_v else None, n=len(pooled_v))


def summarize(records, n_samples, seed=0):
    """The gates of spec section 9 from a list of `evaluate_draw` records.

    The per-gate flags and `pass` are the spec's: KS p > 0.01 for each SBC entry; coverage per parameter (c, s and every
    active continuous parameter: each in 0.87 to 0.93 over the draws where it is reported, conditioned as in the module
    docstring); recall; ECE. The integer width w is reported but not gated (ruling R19). Reporting only (ruling R21):
    `ks_p_bonferroni` = min(1, k p) with k the number of KS gates, `sbc_pass_bonferroni` (all of them > 0.01), and per
    coverage parameter the two-sided binomial p of its hits against 0.90, Bonferroni-adjusted over the gated parameters,
    with `coverage_consistent_bonferroni` (all > 0.01)."""
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
    k_ks = len(uncond) + len(cond)
    for g in (*uncond.values(), *cond.values()):
        g["ks_p_bonferroni"] = min(1.0, k_ks * g["ks_p"])
    sbc_bonf = bool(k_ks) and all(g["ks_p_bonferroni"] > ALPHA for g in (*uncond.values(), *cond.values()))

    per, pooled = _coverage_table([r["covered"] for r in records if r["covered"] is not None])
    key_per, key_pooled = _coverage_table([r["covered_key_only"] for r in records if r["covered_key_only"] is not None])
    k_cov = sum(g["gated"] for g in per.values())
    for g in per.values():
        g["pass"] = bool(COVERAGE_RANGE[0] <= g["coverage"] <= COVERAGE_RANGE[1]) if g["gated"] else None
        p = float(stats.binomtest(g["hits"], g["n"], COVERAGE_TARGET).pvalue)
        g["binomial_p"] = p
        g["binomial_p_bonferroni"] = min(1.0, k_cov * p) if g["gated"] else None
    cov_bonf = bool(k_cov) and all(g["binomial_p_bonferroni"] > ALPHA for g in per.values() if g["gated"])

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
        coverage=dict(per_parameter=per, pooled=pooled, n_draws=sum(r["covered"] is not None for r in records),
                      range=list(COVERAGE_RANGE), gate="per parameter",
                      conditioning="top candidate has the true (edge, pattern) and is nearest to the truth",
                      note_w=W_NOTE, note_slots="intervals are those of the slots active in the top candidate's medoid state",
                      key_only_diagnostic=dict(per_parameter=key_per, pooled=key_pooled,
                                               n_draws=sum(r["covered_key_only"] is not None for r in records),
                                               conditioning="top candidate has the true (edge, pattern); not a gate")),
        top5_recall=dict(value=_finite(recall), threshold=RECALL_MIN, **{"pass": gates["top5_recall"]}),
        ece=dict(value=_finite(ece), threshold=ECE_MAX, n_candidates=len(probs), reliability=reliability,
                 **{"pass": gates["ece"]}),
        gates=gates, **{"pass": all(gates.values())},
        sbc_pass_bonferroni=sbc_bonf, coverage_consistent_bonferroni=cov_bonf,
        noise_aware=dict(alpha=ALPHA, ks_gates=k_ks, coverage_gates=k_cov,
                         note="reporting only: Bonferroni-adjusted p > alpha for every KS gate / every coverage parameter"))


def run(model, xs, draws, indices=None, n_samples=1000, steps=64, seed=0, progress=None, done=None, on_record=None,
        failed=None, on_error=None):
    """Evaluate the (spectrum, draw) pairs and summarise. Returns (summary, seconds, records).

    Each pair has an index (default its position; the CLI passes the test draw's own index); its torch seed and its
    tie-breaking rng derive from (seed, index) alone, so a draw's record does not depend on which others were run.
    `done` maps index -> an already computed record (resume: those are not recomputed); `on_record(index, record)` is
    called for each newly computed one. `failed` maps index -> the error of a draw that failed before: it is skipped,
    not retried. With `on_error`, a draw whose evaluation raises is skipped and `on_error(index, error)` is called (and it
    is added to `failed`); without it the error propagates. Skipped draws are not in the records or the summary."""
    indices = list(range(len(draws))) if indices is None else [int(i) for i in indices]
    done = {} if done is None else done
    failed = {} if failed is None else failed
    t0, records, new = time.time(), [], 0
    for j, (x, d, i) in enumerate(zip(xs, draws, indices)):
        if i in done:
            records.append(done[i])
            continue
        if i in failed:
            continue
        rng = np.random.default_rng([seed, i])
        try:
            rec = evaluate_draw(model, x, d, n_samples, steps, int(rng.integers(2 ** 31)), rng)
        except Exception as err:
            if on_error is None:
                raise
            failed[i] = repr(err)
            on_error(i, err)
            continue
        records.append(rec)
        new += 1
        if on_record:
            on_record(i, records[-1])
        if progress:
            progress(j + 1, len(draws), (time.time() - t0) / new)
    return summarize(records, n_samples, seed), time.time() - t0, records
