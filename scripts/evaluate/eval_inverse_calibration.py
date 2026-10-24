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
is reproducible). Records written with other settings (model, samples, steps, seed) are refused: delete the file.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import paths
from shazam.inverse import calibration, data
from shazam.inverse.model import InverseModel

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
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    out = Path(a.out) if a.out else Path(a.model).with_name("calibration.json")
    model = InverseModel.load(a.model)
    arrays = data.load_split(a.data, "test", NEEDED)
    n_rows = len(arrays["index"])
    rows = np.sort(np.random.default_rng(a.seed).choice(n_rows, size=min(a.n_test, n_rows), replace=False))
    draws, xs = data.draws(arrays, rows), arrays["x"][rows]
    indices = [int(i) for i in arrays["index"][rows]]
    print(f"[calibration] {len(rows)} of {n_rows} test rows, {a.n_samples} samples each", flush=True)

    cfg = dict(model=str(a.model), n_samples=a.n_samples, steps=a.steps, seed=a.seed)
    log = out.with_name("calibration.jsonl")
    done = {}
    if log.exists():
        for line in log.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:               # a line cut short by a crash: that draw is simply redone
                continue
            if rec["cfg"] != cfg:
                raise SystemExit(f"{log} was written with {rec['cfg']}, not {cfg}: delete it to start over")
            done[rec["index"]] = rec["record"]
        print(f"[calibration] resuming: {len(done)} draws already in {log.name}", flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "a")
    if log.stat().st_size and not log.read_text().endswith("\n"):
        fh.write("\n")                                 # a crash cut the last line short: keep new records on their own lines

    def on_record(index, record):
        fh.write(json.dumps(dict(index=index, cfg=cfg, record=record)) + "\n")
        fh.flush()

    def progress(i, n, seconds):
        if i % 50 == 0 or i == n:
            print(f"[calibration] {i}/{n} draws, {seconds:.2f} s per new draw", flush=True)

    try:
        summary, seconds, _ = calibration.run(model, xs, draws, indices, a.n_samples, a.steps, a.seed, progress, done,
                                              on_record)
    finally:
        fh.close()
    n_new = len(rows) - sum(i in done for i in indices)
    manifest = data.load_manifest(a.data)
    report = dict(model=str(a.model), data=str(a.data), code_version=(manifest or {}).get("code_version"),
                  split="test", steps=a.steps, seed=a.seed, threads=torch.get_num_threads(), seconds=seconds,
                  resumed_draws=len(rows) - n_new, seconds_per_draw=seconds / max(n_new, 1) if n_new else None, **summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=1))
    os.replace(tmp, out)
    verdict = ", ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in summary["gates"].items())
    print(f"[calibration] spec verdict {'PASS' if summary['pass'] else 'FAIL'}: {verdict}", flush=True)
    print(f"[calibration] noise-aware (reporting only): SBC after Bonferroni "
          f"{'ok' if summary['sbc_pass_bonferroni'] else 'FAIL'}, coverage consistent with 0.90 after Bonferroni "
          f"{'ok' if summary['coverage_consistent_bonferroni'] else 'FAIL'}; wrote {out}", flush=True)
    return report


if __name__ == "__main__":
    main()
