"""Linear probe of a 128-number representation (INV-AE probe gate): how much of the discrete state and the width a
logistic regression can read off it.

    PYTHONPATH=src python scripts/evaluate/probe_latent.py results/inverse_design/ae_cnn results/inverse_design/model_v2_depth_bandtop

Each FOLDER holds either ae.pt (a SpectrumAutoencoder: its encode) or model.pt (an InverseModel: its fingerprint),
fed the input it was trained on. Embeds the first --n-train training draws and all validation draws with index
< 300,000, standardises with the training statistics, fits a multinomial logistic regression for the state and for the
width on the training embeddings, and reports validation accuracy. The test split is never read. Writes probe.json in
each folder.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import paths
from shazam.inverse import data
from shazam.inverse.autoencoder import SpectrumAutoencoder
from shazam.inverse.encoding import encode
from shazam.inverse.model import InverseModel
from tbribbon.parent_data import to_input_bandtop, to_input_linear

DATA = paths.data_root() / "_generated" / "hexagonal_parent_v1"
NEEDED = ("index", "x", "raw", "on", "lead", "params", "edge", "width", "density")


def load_representation(folder):
    """(embed function x -> [N, 128], input kind) for an autoencoder or an inverse-model folder."""
    folder = Path(folder)
    if (folder / "ae.pt").exists():
        ae = SpectrumAutoencoder.load(folder / "ae.pt")
        return ae.encode, "bandtop"
    model = InverseModel.load(folder / "model.pt")
    return model.encode, model.cfg.get("input", "log")


def model_input(arrays, kind):
    return {"log": lambda: arrays["x"], "linear": lambda: to_input_linear(arrays["raw"]),
            "bandtop": lambda: to_input_bandtop(arrays["raw"])}[kind]()


@torch.no_grad()
def embed(fn, x, chunk=4096):
    return np.concatenate([fn(torch.as_tensor(x[i:i + chunk])).numpy() for i in range(0, len(x), chunk)])


def probe(h_tr, y_tr, h_va, y_va, steps=300, l2=1e-4, seed=0):
    """Validation accuracy of a multinomial logistic regression fitted full-batch with L-BFGS (in torch, so the thread
    count is the script's own; scikit-learn's solver oversubscribed the CPU, 40 min per model)."""
    torch.manual_seed(seed)
    classes, y = np.unique(y_tr, return_inverse=True)
    X, Y = torch.as_tensor(h_tr, dtype=torch.float32), torch.as_tensor(y)
    lin = torch.nn.Linear(X.shape[1], len(classes))
    opt = torch.optim.LBFGS(lin.parameters(), lr=1, max_iter=steps, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lin(X), Y) + l2 * lin.weight.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        pred = classes[lin(torch.as_tensor(h_va, dtype=torch.float32)).argmax(1).numpy()]
    return float((pred == y_va).mean())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--n-train", type=int, default=60_000)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    t0 = time.time()
    tr = data.load_split(DATA, "train", NEEDED)
    va = data.load_split(DATA, "val", NEEDED)
    tr = {k: v[tr["index"] < 300_000][:a.n_train] for k, v in tr.items()}
    va = {k: v[va["index"] < 300_000] for k, v in va.items()}
    labels = {}
    for name, s in (("tr", tr), ("va", va)):
        labels[name] = dict(state=np.array([encode(d)[0] for d in data.draws(s, slice(None))]),
                            width=s["width"].astype(int))
    print(f"probe on {len(tr['index']):,} train / {len(va['index']):,} val draws ({time.time() - t0:.0f}s)", flush=True)
    for folder in a.folders:
        fn, kind = load_representation(folder)
        h_tr, h_va = embed(fn, model_input(tr, kind)), embed(fn, model_input(va, kind))
        mu, sd = h_tr.mean(0), h_tr.std(0) + 1e-6
        h_tr, h_va = (h_tr - mu) / sd, (h_va - mu) / sd
        res = {lab: probe(h_tr, labels["tr"][lab], h_va, labels["va"][lab]) for lab in ("state", "width")}
        res.update(input=kind, n_train=len(h_tr), n_val=len(h_va))
        (folder / "probe.json").write_text(json.dumps(res, indent=1))
        print(f"{folder.name:28s} input {kind:7s}  state {res['state']:.3f}  width {res['width']:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
