"""Train the all-ribbons autoencoder (AE-1): one Conv1dAE over every device, pilot cloud and legacy concentration.

    python notebooks/material_atlas/train_ae.py --latent 64
Training uses seeds 0-699; the early-stopping set is a random 30,000 of seeds 700-849. Seeds 850-999 are never read here.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import ae_data
from atlaslib import train_autoencoder

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--val-n", type=int, default=30000)
    ap.add_argument("--per-cloud", type=int, default=None, help="thin every cloud to this many seeds (smoke runs)")
    ap.add_argument("--cache", default="~/atlas_store/ae_v1_cache")
    ap.add_argument("--out", default=str(HERE / "ae_v1"))
    a = ap.parse_args()
    t0 = time.time()
    tag = f"_thin{a.per_cloud}" if a.per_cloud else ""
    data = {}
    for split in ("train", "val"):
        d = Path(a.cache).expanduser() / f"{split}{tag}"
        if not (d / "x.npy").exists():
            ae_data.save(ae_data.collect(split, per_cloud=a.per_cloud), d)
        data[split] = ae_data.load(d)
        print(f"[ae] {split}: {len(data[split]['x']):,} spectra, {len(data[split]['devices'])} devices ({time.time() - t0:.0f}s)", flush=True)
    tr, va = data["train"], data["val"]
    sel = np.random.default_rng(0).choice(len(va["x"]), min(a.val_n, len(va["x"])), replace=False)
    model, hist = train_autoencoder(tr["x"], va["x"][sel], latent=a.latent, epochs=a.epochs, patience=a.patience,
                                    threads=a.threads)
    out = Path(a.out) / f"L{a.latent}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "latent": a.latent, "seq_len": ae_data.SPEC.n_channels,
                "spec": ae_data.SPEC.as_dict(), "val_history": hist}, out / "encoder.pt")
    (out / "manifest.json").write_text(json.dumps({
        "latent": a.latent, "epochs_run": len(hist), "best_val_mse": min(hist), "val_history": hist,
        "n_train": int(len(tr["x"])), "n_val_early_stopping": int(len(sel)), "n_devices": int(len(tr["devices"])),
        "train_by_source": {s: int((tr["src"] == k).sum()) for k, s in enumerate(ae_data.SOURCES)},
        "per_cloud": a.per_cloud, "train_seconds": round(time.time() - t0)}, indent=2))
    print(f"[ae] L{a.latent}: {len(hist)} epochs, best val mse {min(hist):.6f}, {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
