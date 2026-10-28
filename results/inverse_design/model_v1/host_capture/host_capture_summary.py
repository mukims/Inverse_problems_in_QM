"""Summarise host_capture.jsonl: found rates per level (top 5 and top 1), width accuracy, breakdowns."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("host_capture.jsonl"))
R = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
LEVELS = {"edge + site types": ("key",), "+ full structure": ("key", "state"),
          "+ host values": ("key", "state", "host"), "+ host values + width": ("key", "state", "host", "hostw")}


def ok(c, checks):
    return all(c[k] for k in checks)


def found(r, checks, top=5):
    return any(ok(c, checks) for c in r["cands"][:top])


print(f"{len(R)} test draws\n")
print(f"{'level':28s} {'top 5':>7s} {'top 1':>7s}")
for name, checks in LEVELS.items():
    print(f"{name:28s} {np.mean([found(r, checks) for r in R]):7.3f} {np.mean([found(r, checks, 1) for r in R]):7.3f}")

# where the chain breaks, for draws with the right full structure somewhere in the top 5
st = [r for r in R if found(r, ("key", "state"))]
print(f"\nOf {len(st)} draws with the right structure in the top 5:")
best = [next(c for c in r["cands"] if ok(c, ("key", "state"))) for r in st]
print(f"  host values inside the 99% region: {np.mean([c['host'] for c in best]):.3f}")
print(f"  host values + width inside:        {np.mean([c['hostw'] for c in best]):.3f}")
dw = np.array([c["w_med"] - r["w"] for c, r in zip(best, st)])
print(f"  width of its medoid: exact {np.mean(dw == 0):.3f}, within 1 {np.mean(abs(dw) <= 1):.3f}, "
      f"within 2 {np.mean(abs(dw) <= 2):.3f}; mean error {dw.mean():+.2f} (negative = too narrow)")
dc = np.array([c["logc_med"] - r["logc"] for c, r in zip(best, st)])
print(f"  log density error mean {dc.mean():+.3f}; correlation of width and density errors {np.corrcoef(dw, dc)[0, 1]:+.3f}")

# which structural choice is wrong when the edge and site types are right but no top-5 candidate has the full structure
sys.path.insert(0, "/run/media/shardul/storage/machine_learning/transmission_github/transmissions/src")
from shazam.inverse.encoding import STATES  # noqa: E402

miss = [r for r in R if found(r, ("key",)) and not found(r, ("key", "state"))]
FIELDS = ("lead", "delta_zero", "eps_zero", "t_AA_on", "cross")
wrong = defaultdict(int)
for r in miss:
    c = next(c for c in r["cands"] if c["key"])                      # most probable candidate with the right key
    t, s = STATES[r["true_state"]], STATES[c["s"]]
    for f in FIELDS:
        wrong[f] += getattr(t, f) != getattr(s, f)
print(f"\nOf {len(miss)} draws with the right edge and site types but not the full structure, the best such candidate has "
      "the wrong: " + ", ".join(f"{f} {wrong[f] / len(miss):.2f}" for f in FIELDS))
by_true = defaultdict(lambda: [0, 0])
for r in R:
    t = STATES[r["true_state"]]
    for f, v in (("delta_zero", t.delta_zero), ("t_AA_on", t.t_AA_on)):
        if t.pattern[1]:
            by_true[(f, v)][0] += found(r, ("key", "state"))
            by_true[(f, v)][1] += 1
print("full structure found (top 5) by true switch, H-containing hosts: "
      + ", ".join(f"{f}={v}: {a / b:.3f} (n {b})" for (f, v), (a, b) in sorted(by_true.items())))

# breakdowns of the top-5 "host + width" level
full = LEVELS["+ host values + width"]
for label, keyf in (("pattern", lambda r: r["pattern"]), ("edge", lambda r: r["edge"]),
                    ("width", lambda r: f"{4 + 5 * ((r['w'] - 4) // 5):2d}-{min(22, 8 + 5 * ((r['w'] - 4) // 5)):2d}"),
                    ("density", lambda r: ("low", "mid", "high")[int(np.digitize(r["logc"], np.log([0.0083, 0.0225])))])):
    g = defaultdict(list)
    for r in R:
        g[keyf(r)].append(found(r, full))
    print(f"\nby {label}: " + ", ".join(f"{k} {np.mean(v):.3f} (n {len(v)})" for k, v in sorted(g.items())))
