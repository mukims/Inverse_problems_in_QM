"""How data moves through model_v1, spectrum -> parameters, and where the right answer drops out.

Figure 1 (info_flow_example.png): one validation draw traced through every stage of the network - the raw spectrum,
the model input, the three conv feature maps, the 13 pooled buckets, the 128-number embedding, the state probabilities
and the posterior samples of width and s.
Figure 2 (info_flow_survival.png): on 15,000 validation draws, how often the right state / width is still recoverable
at each stage (from nn_probe.jsonl: nearest training draw in the raw input, in the embedding, then the model itself).

    python info_flow.py [--index 818] [--probe nn_probe.jsonl]
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path("/run/media/shardul/storage/machine_learning/transmission_github/transmissions")
sys.path.insert(0, str(REPO / "src"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

from shazam.inverse import data
from shazam.inverse.encoding import STATES, decode, encode
from shazam.inverse.model import InverseModel

HERE = Path(__file__).parent
DATA = Path("/run/media/shardul/storage/machine_learning/data/_generated/hexagonal_parent_v1")
MODEL = REPO / "results/inverse_design/model_v1/model.pt"
NEEDED = ("index", "x", "raw", "on", "lead", "params", "edge", "width", "density")
DE = 0.02                                                 # eV per input channel
INK, MUTED, ACCENT, TRUE = "#1f2328", "#8b949e", "#2f6fdf", "#d1552b"


def short(state):
    s = STATES[state]
    pat = "".join(t for t, o in zip("CHK", s.pattern) if o)
    flags = [f for f, on in (("Δ=0", s.delta_zero), ("ε=0", s.eps_zero), ("tAA", s.t_AA_on)) if on]
    cross = "".join(n for n, on in zip(("CH", "CK", "HK"), s.cross) if on)
    return f"{s.edge[:2]} {pat} lead {s.lead}" + (" " + " ".join(flags) if flags else "") + (f" x{cross}" if cross else "")


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8, colors=INK)


def example(model, d, x, raw, out):
    acts = {}
    for i, name in ((2, "conv1"), (5, "conv2"), (8, "conv3"), (9, "pool"), (11, "embed")):
        model.encoder[i].register_forward_hook(lambda m, inp, o, name=name: acts.__setitem__(name, o[0].detach().numpy()))
    xt = torch.as_tensor(x[None])
    with torch.no_grad():
        h = model.encode(xt)
        p = torch.softmax(model.discrete(h)[0].masked_fill(~model.mask_known, float("-inf")), -1).numpy()
    with torch.random.fork_rng():
        torch.manual_seed(0)
        st, z = model.sample(xt, 1000)
    dec = [decode(int(s), zz) for s, zz in zip(st.numpy(), z.numpy())]
    w_s, s_s = np.array([q.width for q in dec]), np.array([q.params.s for q in dec])
    true_state = encode(d)[0]
    E = np.arange(416) * DE
    top = (np.nonzero(x > 0)[0].max() + 1) * DE              # where the input (what the model sees) goes to zero
    print(f"corr(s, w) over the samples: {np.corrcoef(s_s, w_s)[0, 1]:+.2f}")

    fig = plt.figure(figsize=(13, 15))
    gs = GridSpec(6, 3, figure=fig, height_ratios=[1, 1, 1, 1, 1, 1.1], hspace=0.75, wspace=0.3)
    pat = "".join(t for t, o in zip("CHK", d.params.pattern) if o)
    fig.suptitle(f"One spectrum through model_v1: validation draw {d_index}, {pat} host, {d.edge}, width {d.width}, "
                 f"s = {d.params.s:.2f} eV, c = {100 * d.density:.2f}%", fontsize=13, fontweight="bold", color=INK, y=0.995)

    def energy_panel(ax, title, n_in):
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        ax.set_xlim(0, 416 * DE)
        ax.axvline(top, color=TRUE, lw=1, ls="--")
        clean(ax)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(E, raw, color=MUTED, lw=1, label="raw T(E), the simulator output")
    ax2 = ax.twinx()
    ax2.plot(E, x, color=ACCENT, lw=1.3, label="model input x = log(1+T)/log 65")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("input x", fontsize=9, color=ACCENT)
    ax2.tick_params(labelsize=8, colors=ACCENT)
    ax2.spines[["top"]].set_visible(False)
    for b in range(1, 13):
        ax.axvline(b * 416 * DE / 13, color="#d0d7de", lw=0.8, zorder=0)
    energy_panel(ax, f"1. Input: 416 channels, 0-8.32 eV. Signal ends at {top:.2f} eV (dashed); "
                     f"{100 * (1 - (x > 0).mean()):.0f}% of the window is exactly zero. Grey lines: the 13 pooling buckets", 416)
    ax.set_ylabel("T (channels)", fontsize=9)
    ax.set_xlabel("energy (eV)", fontsize=9)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="upper right")

    for row, (name, n_ch, n_pos, k) in enumerate((("conv1", 32, 208, 7), ("conv2", 64, 104, 5), ("conv3", 128, 52, 3)), 1):
        ax = fig.add_subplot(gs[row, :])
        a = acts[name]
        ax.imshow(a, aspect="auto", cmap="Blues", extent=(0, 416 * DE, n_ch, 0), interpolation="nearest")
        energy_panel(ax, f"{row + 1}. Conv layer {row} (kernel {k}, stride 2): {n_ch} feature channels × {n_pos} positions "
                         f"({DE * 416 / n_pos:.2f} eV per position)", n_pos)
        ax.set_ylabel("feature", fontsize=9)
        ax.set_xlabel("energy (eV)", fontsize=9)

    ax = fig.add_subplot(gs[4, :2])
    ax.imshow(acts["pool"], aspect="auto", cmap="Blues", extent=(0, 416 * DE, 128, 0), interpolation="nearest")
    energy_panel(ax, f"5. Average-pool to 13 buckets (0.64 eV each): 128 features × 13. "
                     f"Only {int(np.ceil(top / 0.64))} buckets hold signal", 13)
    ax.set_ylabel("feature", fontsize=9)
    ax.set_xlabel("energy (eV)", fontsize=9)

    ax = fig.add_subplot(gs[4, 2])
    ax.bar(np.arange(128), acts["embed"], color=ACCENT, width=1.0)
    ax.set_title("6. Linear 1,664 → 128: the embedding,\nall the heads ever see", fontsize=10, loc="left", color=INK)
    ax.set_xlabel("embedding dimension", fontsize=9)
    clean(ax)

    ax = fig.add_subplot(gs[5, 0])
    order = np.argsort(p)[::-1][:8]
    cols = [TRUE if k == true_state else ACCENT for k in order]
    ax.barh(range(len(order))[::-1], p[order], color=cols, height=0.7)
    ax.set_yticks(range(len(order))[::-1], [short(k) for k in order], fontsize=7)
    ax.set_xlabel("probability", fontsize=9)
    ax.set_title(f"7a. Classifier: 156 discrete states (top 8).\nTrue state in red, p = {p[true_state]:.2f}",
                 fontsize=10, loc="left", color=INK)
    clean(ax)

    ax = fig.add_subplot(gs[5, 1])
    ws = np.arange(4, 23)
    cnt = np.array([(w_s == w).sum() for w in ws]) / len(w_s)
    ax.bar(ws, cnt, color=[TRUE if w == d.width else ACCENT for w in ws], width=0.8)
    ax.set_xticks(ws[::2])
    ax.set_xlabel("width w (cells)", fontsize=9)
    ax.set_ylabel("share of 1,000 samples", fontsize=9)
    ax.set_title(f"7b. Flow head: width over 1,000 samples.\nTrue w = {d.width} (red), share {cnt[ws == d.width][0]:.2f}",
                 fontsize=10, loc="left", color=INK)
    clean(ax)

    ax = fig.add_subplot(gs[5, 2])
    ax.hist(s_s, bins=40, color=ACCENT)
    ax.axvline(d.params.s, color=TRUE, lw=2)
    ax.set_xlabel("energy scale s (eV)", fontsize=9)
    ax.set_title(f"7c. Flow head: s over 1,000 samples.\nTrue s = {d.params.s:.2f} (red)", fontsize=10, loc="left", color=INK)
    clean(ax)

    fig.subplots_adjust(top=0.955)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def survival(probe, out):
    R = [json.loads(l) for l in Path(probe).read_text().splitlines()]
    m = lambda k: np.mean([r[k] for r in R])
    stages = ["Raw input\n(nearest training\nspectrum)", "Embedding\n(nearest in the\n128 numbers)", "Model, best guess\n(classifier top 1 /\nsample mode)",
              "Model, 200 samples\n(right answer\nanywhere in them)"]
    state = [m("raw_state_1"), m("embed_state_1"), m("model_state_argmax"), m("model_state_in")]
    width = [m("raw_width_1"), m("embed_width_1"), m("model_width_mode"), m("model_width_in")]
    key = [m("raw_key_1"), m("embed_key_1"), None, None]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    xx = np.arange(len(stages))
    bw = 0.26
    for off, vals, lab, c in ((-bw, key, "edge + site types", "#9fb8e6"), (0, state, "full discrete state", ACCENT),
                              (bw, width, "exact width", "#e3a33b")):
        for i, v in enumerate(vals):
            if v is None:
                continue
            ax.bar(xx[i] + off, v, bw * 0.92, color=c, label=lab if i == 0 else None)
            ax.text(xx[i] + off, v + 0.015, f"{100 * v:.0f}%", ha="center", fontsize=8, color=INK)
    ax.set_xticks(xx, stages, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{int(100 * t)}%" for t in np.linspace(0, 1, 6)], fontsize=8)
    ax.set_ylabel("validation draws where the right answer is recovered", fontsize=9)
    ax.set_title("The embedding finds the right answer more often than the raw input does: the encoder isn't where it's lost.\n"
                 f"Exact width is weak everywhere: even the nearest raw spectrum has the same width only {100 * width[0]:.0f}% "
                 f"of the time (15,000 validation draws)", fontsize=11, fontweight="bold", loc="left", color=INK)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    clean(ax)
    fig.text(0.01, -0.04, "After the samples (test split, report §3): candidates() keeps full structure in the top 5 for 62% "
             "and exact width for 41% given the right structure; the medoid recreates the spectrum in 0 of 60 draws.",
             fontsize=8.5, color=MUTED)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=818)
    ap.add_argument("--probe", default=str(HERE / "nn_probe.jsonl"))
    a = ap.parse_args()
    model = InverseModel.load(MODEL).eval()
    va = data.load_split(DATA, "val", NEEDED)
    r = int(np.nonzero(va["index"] == a.index)[0][0])
    d_index = a.index
    example(model, data.draws(va, r)[0], va["x"][r], va["raw"][r], HERE / "info_flow_example.png")
    survival(a.probe, HERE / "info_flow_survival.png")
    print("wrote", HERE / "info_flow_example.png", HERE / "info_flow_survival.png")
