"""Calibration gate of the inverse model (spec section 9, "Model calibration").

    PYTHONPATH=src python scripts/evaluate/eval_inverse_calibration.py --threads 16

Draws --n-test rows of the TEST split (draw index % 20 == 19; a seeded random choice when the split is larger), samples
--n-samples per spectrum, and writes calibration.json next to the model: each gate's value and pass flag and an overall
`pass`. Gates: SBC KS p > 0.01 (log_s, log_c, w unconditionally; every active parameter with the true state fixed),
per-parameter 90%-interval coverage in 0.87-0.93 (draws whose top candidate has the true edge and pattern and is the
nearest candidate to the truth; the integer width w is reported, not gated), top-5 recall >= 0.95, candidate ECE <= 0.05.
calibration.json also carries noise-aware verdicts (Bonferroni over the KS gates, binomial tests of the coverage) as
reporting only. Nothing is tuned on the test split: a failing gate is a result to report, not a prompt to change
thresholds or sampling settings.

The run takes about an hour, so each evaluated draw is appended to calibration.jsonl next to the output; a restart skips
the draws already there (keyed by the test draw's index; each draw's randomness derives from (seed, index), so the rest
is reproducible). Records written with other settings (model path and sha256, samples, steps, seed, the number of test
rows and the data's draw count, which fix the seeded choice of rows) are refused: delete the file. A draw whose evaluation
raises is recorded as an error and skipped (listed under `failed_draws`; not retried on a restart unless --retry-errors).
calibration.json records this run's code version, the model's sha256 and the data's provenance.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import paths
from shazam.inverse import calibration, data
from shazam.inverse.model import InverseModel
from shazam.inverse.runio import RunLog, atomic_write_json, code_version, sha256_of

DEFAULT_MODEL = paths.results_dir("inverse_design", "model_v1") / "model.pt"
DEFAULT_DATA = paths.data_root() / "_generated" / "hexagonal_parent_v1"
NEEDED = ("index", "x", "on", "lead", "params", "edge", "width", "density")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None, help="torch intra-op threads (default: torch's own)")
    ap.add_argument("--out", default=None, help="default: calibration.json next to the model")
    ap.add_argument("--retry-errors", action="store_true", help="retry the draws the .jsonl records as failed")
    a = ap.parse_args(argv)
    code_version()                                     # this run's code version: fixed now, when it starts
    if a.threads:
        torch.set_num_threads(a.threads)
    out = Path(a.out) if a.out else Path(a.model).with_name("calibration.json")
    sha = sha256_of(a.model)
    model = InverseModel.load(a.model)
    data_prov = data.data_provenance(a.data)
    arrays = data.load_split(a.data, "test", NEEDED)
    n_rows = len(arrays["index"])
    rows = np.sort(np.random.default_rng(a.seed).choice(n_rows, size=min(a.n_test, n_rows), replace=False))
    draws, xs = data.draws(arrays, rows), arrays["x"][rows]
    indices = [int(i) for i in arrays["index"][rows]]
    print(f"[calibration] {len(rows)} of {n_rows} test rows, {a.n_samples} samples each", flush=True)

    cfg = dict(model=str(a.model), model_sha256=sha, n_samples=a.n_samples, steps=a.steps, seed=a.seed, n_rows=n_rows,
               data_draws=data_prov["n_draws"])
    log = RunLog(out.with_name("calibration.jsonl"), cfg, key_field="index", tag="calibration")
    done = dict(log.done)
    wanted = set(indices)
    failed = {} if a.retry_errors else {i: e for i, e in log.errors.items() if i in wanted}

    def progress(i, n, seconds):
        if i % 50 == 0 or i == n:
            print(f"[calibration] {i}/{n} draws, {seconds:.2f} s per new draw", flush=True)

    try:
        summary, seconds, _ = calibration.run(model, xs, draws, indices, a.n_samples, a.steps, a.seed, progress, done,
                                              log.add, failed, log.add_error)
    finally:
        log.close()
    resumed = sum(i in done for i in indices)
    n_new = len(rows) - resumed - len(failed)
    failed_draws = dict(n=len(failed), indices=[i for i in indices if i in failed],
                        errors={str(i): failed[i] for i in indices if i in failed})
    if failed:
        print(f"[calibration] {len(failed)} draws failed and are not in the gates: {failed_draws['indices']}", flush=True)
    report = dict(model=str(a.model), model_sha256=sha, data=str(a.data), code_version=code_version(),
                  data_provenance=data_prov, split="test", steps=a.steps, seed=a.seed, threads=torch.get_num_threads(),
                  seconds=seconds, resumed_draws=resumed, failed_draws=failed_draws,
                  seconds_per_draw=seconds / max(n_new, 1) if n_new else None, **summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, report, indent=1, allow_nan=True)
    verdict = ", ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in summary["gates"].items())
    print(f"[calibration] spec verdict {'PASS' if summary['pass'] else 'FAIL'}: {verdict}", flush=True)
    print(f"[calibration] noise-aware (reporting only): SBC after Bonferroni "
          f"{'ok' if summary['sbc_pass_bonferroni'] else 'FAIL'}, coverage consistent with 0.90 after Bonferroni "
          f"{'ok' if summary['coverage_consistent_bonferroni'] else 'FAIL'}; wrote {out}", flush=True)
    return report


if __name__ == "__main__":
    main()
