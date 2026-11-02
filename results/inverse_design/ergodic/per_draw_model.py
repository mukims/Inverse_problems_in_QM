"""Per-draw validation results of one inverse model (ERGO-1, part C), saved per draw so they can be set against the
ergodic diagnostics. Validation split only, draws with index < 300,000. Per draw, over 200 posterior samples seeded by
the draw index (as compare_inverse.evaluate): classifier top state right, modal width right, true state / width among
the samples, and for the impurity density the median, 5th and 95th percentile of log c over the samples and the rank
of the true log c among them (0 to 1; 0.5 on average when calibrated).

    TRANSMISSION_DATA=... PYTHONPATH=src:scripts/evaluate python results/inverse_design/ergodic/per_draw_model.py \
        results/inverse_design/model_v2_depth_bandtop --threads 2
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from compare_inverse import DATA, MAX_DRAWS, NEEDED, SAMPLES, model_input
from shazam.inverse import data
from shazam.inverse.encoding import decode, encode
from shazam.inverse.model import InverseModel

OUT = Path(__file__).resolve().parent


def evaluate(model, x, index, state, width, density):
    with torch.no_grad():
        h = model.encode(torch.as_tensor(x))
        logits = model.discrete(h).masked_fill(~model.mask_known, float("-inf"))
    out = dict(state_top=logits.argmax(1).numpy() == state)
    keys = ("width_mode", "state_in", "width_in", "logc_med", "logc_q05", "logc_q95", "logc_rank")
    out.update({k: np.empty(len(state)) for k in keys})
    for i in range(len(state)):
        with torch.random.fork_rng():
            torch.manual_seed(int(index[i]))
            st, z = model.sample(torch.as_tensor(x[i:i + 1]), SAMPLES)
        d = [decode(int(s), zz) for s, zz in zip(st.numpy(), z.numpy())]
        w = np.array([x.width for x in d])
        lc = np.log([x.density for x in d])
        out["width_mode"][i] = np.bincount(w).argmax() == width[i]
        out["state_in"][i] = state[i] in st.numpy()
        out["width_in"][i] = width[i] in w
        out["logc_med"][i], out["logc_q05"][i], out["logc_q95"][i] = np.quantile(lc, [0.5, 0.05, 0.95])
        out["logc_rank"][i] = np.mean(lc < np.log(density[i]))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(state)}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", type=Path, help="folder holding model.pt")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    t0 = time.time()
    va = data.load_split(DATA, "val", NEEDED + ("n_imp",))
    va = {k: v[va["index"] < MAX_DRAWS] for k, v in va.items()}
    state = np.array([encode(d)[0] for d in data.draws(va, slice(None))])
    model = InverseModel.load(a.model / "model.pt")
    res = evaluate(model, model_input(model, va), va["index"], state, va["width"].astype(int), va["density"])
    out = OUT / f"per_draw_{a.model.name}.npz"
    np.savez(out, index=va["index"], state=state, **res)
    flags = ("state_top", "width_mode", "state_in", "width_in")
    print(f"wrote {out}: " + ", ".join(f"{k} {res[k].mean():.3f}" for k in flags)
          + f", mean log c rank {res['logc_rank'].mean():.3f} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
