#!/usr/bin/env python
"""Cost and rejection report for a hexagonal-parent generation run (smoke or full).

Per-draw cost is the stored `seconds`, which already covers every attempt of a draw (rejected ones included), so the
projected wall time for N draws on J workers is N x mean(seconds) / J.

    PYTHONPATH=src python scripts/generate/report_parent.py "$TRANSMISSION_DATA/_generated/hexagonal_parent_smoke" \
        --out results/inverse_design/smoke_report.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

PATTERNS = {(1, 0, 0): "C", (0, 1, 0): "H", (0, 0, 1): "K", (1, 1, 0): "C+H", (1, 0, 1): "C+K", (0, 1, 1): "H+K",
            (1, 1, 1): "C+H+K"}


def report(root: Path, project_draws: int = 300_000, n_jobs: int = 22, wall_seconds: float | None = None) -> dict:
    shards = sorted(root.glob("shard_*.npz"))
    cols = ("seconds", "attempts", "on", "width", "edge", "n_imp", "active_atoms")
    a = {k: np.concatenate([np.load(p)[k] for p in shards]) for k in cols}
    rejections = {}
    for p in sorted(root.glob("shard_*_rejections.json")):
        for reason, n in json.loads(p.read_text())["rejections"].items():
            rejections[reason] = rejections.get(reason, 0) + n
    n, s = len(a["seconds"]), a["seconds"].astype(float)
    attempts = int(a["attempts"].sum())
    pats = [PATTERNS[tuple(int(v) for v in row)] for row in a["on"]]
    by_pattern = {p: {"share": pats.count(p) / n,
                      "mean_seconds": float(s[[q == p for q in pats]].mean())} for p in PATTERNS.values() if p in pats}
    widths = a["width"]
    by_width = {f"{lo}-{hi}": float(s[(widths >= lo) & (widths <= hi)].mean())
                for lo, hi in ((4, 8), (9, 13), (14, 18), (19, 22)) if np.any((widths >= lo) & (widths <= hi))}
    out = {
        "root": str(root), "n_draws": n, "n_shards": len(shards),
        "seconds_per_draw": {"mean": float(s.mean()), "median": float(np.median(s)), "p95": float(np.quantile(s, 0.95)),
                             "max": float(s.max())},
        "attempts": {"mean": attempts / n, "total": attempts},
        "rejections": {"counts": rejections, "rate_per_attempt": {k: v / attempts for k, v in rejections.items()}},
        "pattern": by_pattern,
        "mean_seconds_by_width": by_width,
        "edge_share_zigzag": float(a["edge"].mean()),
        "projection": {"draws": project_draws, "workers": n_jobs,
                       "hours": project_draws * float(s.mean()) / n_jobs / 3600},
    }
    if wall_seconds:
        out["projection"]["hours_from_wall_time"] = project_draws * wall_seconds / n / 3600
        out["wall_seconds"] = wall_seconds
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--project-draws", type=int, default=300_000)
    ap.add_argument("--n-jobs", type=int, default=22)
    ap.add_argument("--wall-seconds", type=float, help="measured wall time of the run, for a second projection")
    a = ap.parse_args(argv)
    r = report(a.root, a.project_draws, a.n_jobs, a.wall_seconds)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
