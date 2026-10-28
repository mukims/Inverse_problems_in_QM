"""Summarise near_identical.jsonl: how close are the "wrong structure" answers to the truth, next to right answers?"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("near_identical.jsonl"))
R = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
miss = [r for r in R if r["kind"] == "miss"]
found = [r for r in R if r["kind"] == "found"]


def q(v):
    v = np.asarray([x for x in v if x is not None])
    return f"median {np.median(v):.3f}  [25% {np.percentile(v, 25):.3f}, 75% {np.percentile(v, 75):.3f}]  n {len(v)}"


print("Clean-spectrum distance to the truth, d = sum|dT| / sum T (0 = identical)")
print(f"  right answers (found), as reported     {q(r['d_full'] for r in found)}")
print(f"  right answers (found), at true width   {q(r['d_host'] for r in found)}")
print(f"  wrong structure (miss), at true width  {q(r['d_host'] for r in miss)}")
print(f"  wrong structure (miss), as reported    {q(r['d_full'] for r in miss)}")
print(f"  random other host, same type/edge/width {q(r['d_ref'] for r in miss)}")
p90 = np.percentile([r["d_host"] for r in found], 90)
p50 = np.percentile([r["d_host"] for r in found], 50)
print(f"\nmisses at least as close as a typical right answer (<= found median {p50:.3f}): "
      f"{np.mean([r['d_host'] <= p50 for r in miss]):.3f}")
print(f"misses within the range of right answers (<= found 90th percentile {p90:.3f}): "
      f"{np.mean([r['d_host'] <= p90 for r in miss]):.3f}")
closer = [r["d_host"] < r["d_ref"] for r in miss if r["d_ref"] is not None]
print(f"misses closer to the truth than a random same-type host: {np.mean(closer):.3f}")


def boundary(r, field):
    """How close the truth sits to the boundary the wrong switch crosses (small = hard to tell apart)."""
    t, c = r["true"], r["cand"]
    if field == "delta_zero":
        return t["delta"] if t["delta"] != 0 else c["delta"]
    if field == "t_AA_on":
        return t["t_AA"] if t["t_AA"] != 0 else c["t_AA"]
    if field == "lead":
        return t["within"][c["lead"]]                 # 1 = the candidate's lead type already hops as strongly as the lead
    if field == "cross":
        return max(abs(t["cross"][k] - c["cross"][k]) for k in t["cross"] if (t["cross"][k] == 0) != (c["cross"][k] == 0))
    return None


print("\nSingle-field misses (only one structural choice wrong):")
NOTE = {"delta_zero": "the nonzero Delta (min 0.05, max ~2)", "t_AA_on": "the nonzero t_AA (min 0.03, max 0.3)",
        "lead": "truth's hopping of the type the candidate made lead (1 = tie)",
        "cross": "the bond that is on in one and off in the other (min 0.05, max 1)", "eps_zero": None}
by = defaultdict(list)
for r in miss:
    if len(r["wrong"]) == 1:
        by[r["wrong"][0]].append(r)
for f, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
    print(f"  {f:10s} n {len(rs):3d}  d_host median {np.median([r['d_host'] for r in rs]):.3f}  "
          f"vs found median {p50:.3f}")
    if NOTE.get(f):
        b = np.array([boundary(r, f) for r in rs])
        print(f"             {NOTE[f]}: median {np.median(b):.3f}, 25% {np.percentile(b, 25):.3f}, "
              f"75% {np.percentile(b, 75):.3f}")
print(f"  multi-field misses: n {sum(len(r['wrong']) > 1 for r in miss)}, d_host median "
      f"{np.median([r['d_host'] for r in miss if len(r['wrong']) > 1]):.3f}")

print("\nBy site types, d_host median (miss vs found):")
for pat in ("C", "H", "K", "CH", "CK", "HK", "CHK"):
    m = [r["d_host"] for r in miss if r["pattern"] == pat]
    f_ = [r["d_host"] for r in found if r["pattern"] == pat]
    if m:
        print(f"  {pat:4s} miss {np.median(m):.3f} (n {len(m)})   found {np.median(f_) if f_ else float('nan'):.3f} "
              f"(n {len(f_)})")
