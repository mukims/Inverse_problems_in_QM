"""Summarise recreation_baseline.jsonl: is model_v1's top 5 within the seed-to-seed noise floor of the true host?"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("recreation_baseline.jsonl"))
R = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
truth = np.array([r["truth"]["recreated"] for r in R])
found = np.array([r["found_at"] is not None for r in R])
print(f"{len(R)} test draws")
print(f"  true host recreated (ceiling):   {truth.mean():.3f}")
print(f"  found in top 5:                  {found.mean():.3f}   top 1: {np.mean([r['found_at'] == 1 for r in R]):.3f}")
print(f"  no candidate with right edge + site types: {sum(not any(c['key'] for c in r['cands']) for r in R)}")

C = [(r, c) for r in R for c in r["cands"] if c["key"]]
groups = {"right edge + site types": C, "+ full structure": [x for x in C if x[1]["state"]],
          "+ full structure + exact width": [x for x in C if x[1]["state"] and x[1]["width"] == x[0]["width"]],
          "+ full structure, wrong width": [x for x in C if x[1]["state"] and x[1]["width"] != x[0]["width"]]}
print("\nmisfit / threshold of checked candidates (<= 1 passes); true host median "
      f"{np.median([r['truth']['misfit'] / r['truth']['threshold'] for r in R]):.2f}")
for name, g in groups.items():
    if g:
        m = np.array([c["misfit"] / c["threshold"] for _, c in g])
        d = np.array([c["d_clean"] for _, c in g])
        print(f"  {name:32s} n {len(g):3d}  median {np.median(m):5.1f}  min {m.min():5.2f}  d_clean median {np.median(d):.3f}")

print("\nbest candidate per draw (lowest misfit/threshold), by site types:")
by = defaultdict(list)
for r in R:
    ms = [c["misfit"] / c["threshold"] for c in r["cands"] if c["key"]]
    if ms:
        by[r["pattern"]].append(min(ms))
for pat in ("C", "H", "K", "CH", "CK", "HK", "CHK"):
    if by[pat]:
        print(f"  {pat:4s} n {len(by[pat]):3d}  median {np.median(by[pat]):5.1f}  min {min(by[pat]):5.2f}")

g = groups["right edge + site types"]
d = np.array([c["d_clean"] for _, c in g]); m = np.array([c["misfit"] / c["threshold"] for _, c in g])
print(f"\nd_clean vs misfit/threshold: Spearman {np.corrcoef(np.argsort(np.argsort(d)), np.argsort(np.argsort(m)))[0, 1]:+.2f}")
