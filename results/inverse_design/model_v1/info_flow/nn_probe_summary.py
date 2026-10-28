"""Summarise nn_probe.jsonl: same-state / same-width rate of the nearest training draw, raw input vs embedding vs model."""
import json, sys
from pathlib import Path
import numpy as np
R = [json.loads(l) for l in Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("nn_probe.jsonl")).read_text().splitlines()]
def m(rs, k): return np.mean([r[k] for r in rs])
def row(rs, label):
    print(f"{label:12s} n {len(rs):5d} | state: raw {m(rs,'raw_state_1'):.3f} emb {m(rs,'embed_state_1'):.3f} model {m(rs,'model_state_argmax'):.3f}"
          f" | width: raw {m(rs,'raw_width_1'):.3f} emb {m(rs,'embed_width_1'):.3f} model {m(rs,'model_width_mode'):.3f}"
          f" | both: raw {m(rs,'raw_both_1'):.3f} emb {m(rs,'embed_both_1'):.3f}")
print("Nearest neighbour (k=1) vs model, validation split")
print("all:"); 
for lab in ("state","width","key","both"):
    print(f"  {lab:6s} raw@1 {m(R,f'raw_{lab}_1'):.3f}  emb@1 {m(R,f'embed_{lab}_1'):.3f}  emb_unstd@1 {m(R,f'embed_unstd_{lab}_1'):.3f}"
          f"   raw@10 {m(R,f'raw_{lab}_10'):.3f}  emb@10 {m(R,f'embed_{lab}_10'):.3f}")
print(f"  model: state argmax {m(R,'model_state_argmax'):.3f}  state mode {m(R,'model_state_mode'):.3f}  width mode {m(R,'model_width_mode'):.3f}"
      f"  state in 200 samples {m(R,'model_state_in'):.3f}  width in samples {m(R,'model_width_in'):.3f}")
print("\nby s:")
for lo, hi in ((0.7,1.2),(1.2,2),(2,3),(3,4.01)):
    row([r for r in R if lo <= r["s"] < hi], f"s {lo}-{hi:g}")
print("\nby pattern:")
for p in ("C","H","K","CH","CK","HK","CHK"):
    row([r for r in R if r["pattern"] == p], p)
print("\nby width:")
for lo, hi in ((4,8),(8,13),(13,18),(18,23)):
    row([r for r in R if lo <= r["width"] < hi], f"w {lo}-{hi-1}")
