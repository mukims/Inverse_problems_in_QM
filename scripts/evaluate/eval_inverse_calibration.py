"""Calibration gate of the inverse model (spec section 9, "Model calibration").

    PYTHONPATH=src python scripts/evaluate/eval_inverse_calibration.py --threads 16

Draws --n-test rows of the TEST split (draw index % 20 == 19; a seeded random choice when the split is larger), samples
--n-samples per spectrum, and writes calibration.json next to the model: each gate's value and pass flag and an overall
`pass`. Gates: SBC KS p > 0.01 (log_s, log_c, w unconditionally; every active parameter with the true state fixed),
per-parameter 90%-interval coverage in 0.87-0.93, top-5 recall >= 0.95, candidate ECE <= 0.05. Nothing is tuned on the
test split: a failing gate is a result to report, not a prompt to change thresholds or sampling settings.
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
    print(f"[calibration] {len(rows)} of {n_rows} test rows, {a.n_samples} samples each", flush=True)

    def progress(i, n, seconds):
        if i % 50 == 0 or i == n:
            print(f"[calibration] {i}/{n} draws, {seconds / i:.2f} s per draw", flush=True)

    summary, seconds = calibration.run(model, xs, draws, a.n_samples, a.steps, a.seed, progress)
    manifest = data.load_manifest(a.data)
    report = dict(model=str(a.model), data=str(a.data), code_version=(manifest or {}).get("code_version"),
                  split="test", steps=a.steps, seed=a.seed, threads=torch.get_num_threads(), seconds=seconds,
                  seconds_per_draw=seconds / max(len(rows), 1), **summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=1))
    os.replace(tmp, out)
    verdict = ", ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in summary["gates"].items())
    print(f"[calibration] {'PASS' if summary['pass'] else 'FAIL'}: {verdict}; wrote {out}", flush=True)
    return report


if __name__ == "__main__":
    main()
