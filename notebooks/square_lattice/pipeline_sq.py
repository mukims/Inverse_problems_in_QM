#!/usr/bin/env python
"""
pipeline_sq.py -- end-to-end orchestrator for the square-lattice inverse-transport
pipeline. One driver chains every stage we build by hand today, with a run
manifest, idempotent/resumable stages, and dependency checks.

Pipeline
--------
    ingest    validate the *input material* spec, lay out directories, snapshot
              the environment, and record everything in a run manifest.
    leads     ensure lead surface Green's functions exist for the material
              (Sancho-Rubio solver from compute_leads_sq).
    generate  device transmission spectra for many random impurity
              configurations (ca_sq, parallel, resumable).
    combine   stack per-config spectra into conc_{c}.npy, compute the pristine
              spectrum, and configuration-average (the "CA") into references.
    invert    build the inverse model (physics misfit baseline) from the CA
              references and evaluate it on a held-out split; save predictions
              and metrics.

Each stage is skippable and re-runnable in isolation (``--stages``); outputs are
detected and reused so a re-run only does what's missing.

The "input material" is the tight-binding system: a square-lattice / GNR strip of
unit-cell width ``size`` with hopping ``t``, on-site offset ``e``, broadening
``d_*``, over a fixed energy grid, with a device of ``ncells`` cells and impurities
(on-site shift -0.5, as in ca_sq). Define one via a JSON/YAML config or CLI flags.

Examples
--------
    # write a template config you can edit
    python pipeline_sq.py --write-config material.json

    # full run from a config
    python pipeline_sq.py --config material.json

    # only (re)build the inversion from already-combined data
    python pipeline_sq.py --config material.json --stages invert

    # quick inline run
    python pipeline_sq.py --size 10 --concs 2 4 6 --nconfigs 500
"""

import os
import sys
import json
import time
import platform
import argparse
from dataclasses import dataclass, field, asdict

import numpy as np

# sibling pipeline modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute_leads_sq as cl          # noqa: E402
import ca_sq                           # noqa: E402
import combine_sq                      # noqa: E402

STAGES = ["ingest", "leads", "generate", "combine", "invert"]


# ======================================================================
# Material / run configuration
# ======================================================================
@dataclass
class MaterialConfig:
    # --- identity ---
    name: str = "square_l10"
    lattice: str = "square"                 # currently only "square"

    # --- input material (tight-binding parameters) ---
    size: int = 10                          # unit-cell width l
    t: float = 1.0                          # hopping amplitude
    e: float = 0.0                          # on-site energy offset
    d_lead: float = 1e-4                    # lead broadening eta
    d_device: float = 1e-3                  # device broadening eta
    ncells: int = 100                       # device length (unit cells) -- from ca_sq

    # --- sampling ---
    concentrations: list = field(default_factory=lambda: list(range(2, 51, 2)))
    nconfigs: int = 10000

    # --- solvers ---
    leads_method: str = "sancho-rubio"      # or "iterative"
    leads_tol: float = 1e-9
    n_jobs: int = 0                         # 0 -> cpu_count - 1

    # --- inversion ---
    invert_method: str = "misfit"           # physics config-averaged misfit
    crop_lo: int = 0                        # energy-index crop for the misfit
    crop_hi: int = 200
    holdout_per_conc: int = 200             # configs reserved for evaluation
    make_plots: bool = True

    # --- io ---
    root: str = "~/transmissions_sq"

    # ---- derived paths (not serialised as inputs) ----
    def leads_dir(self):     return os.path.join(self._root(), "leads")
    def data_dir(self):      return self._root()
    def combined_dir(self):  return os.path.join(self._root(), f"size_{self.size}_combined")
    def artifacts_dir(self): return os.path.join(self._root(), f"size_{self.size}_artifacts")
    def leads_path(self):    return os.path.join(self.leads_dir(), f"leads_{self.size}.npy")
    def manifest_path(self): return os.path.join(self.artifacts_dir(), "run_manifest.json")

    def _root(self):         return os.path.expanduser(self.root)

    def jobs(self):
        from multiprocessing import cpu_count
        return self.n_jobs if self.n_jobs and self.n_jobs > 0 else max(1, cpu_count() - 1)

    # ---- (de)serialisation ----
    @classmethod
    def load(cls, path):
        path = os.path.expanduser(path)
        with open(path) as f:
            text = f.read()
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError:
                raise SystemExit("PyYAML not installed; use a .json config instead.")
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        unknown = set(data) - set(known)
        if unknown:
            print(f"[WARN] ignoring unknown config keys: {sorted(unknown)}")
        return cls(**known)

    def save(self, path):
        path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return path


# ======================================================================
# Manifest helpers
# ======================================================================
def _load_manifest(cfg):
    p = cfg.manifest_path()
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"config": asdict(cfg), "env": {}, "stages": {}}


def _save_manifest(cfg, manifest):
    os.makedirs(cfg.artifacts_dir(), exist_ok=True)
    with open(cfg.manifest_path(), "w") as f:
        json.dump(manifest, f, indent=2)


def _record(manifest, stage, seconds, info):
    manifest["stages"][stage] = {
        "status": "ok",
        "seconds": round(seconds, 2),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        **info,
    }


# ======================================================================
# Stage: ingest
# ======================================================================
def stage_ingest(cfg, manifest):
    # sanity checks on the input material
    if cfg.lattice != "square":
        raise SystemExit(f"lattice '{cfg.lattice}' not supported (only 'square').")
    grid = cl.energy_grid()
    if not (0 <= cfg.crop_lo < cfg.crop_hi <= len(grid)):
        raise SystemExit(f"crop [{cfg.crop_lo}:{cfg.crop_hi}] outside energy grid (len {len(grid)}).")
    max_sites = cfg.ncells * cfg.size
    bad = [c for c in cfg.concentrations if c <= 0 or c > max_sites]
    if bad:
        raise SystemExit(f"concentrations {bad} out of range (1..{max_sites} sites).")

    for d in (cfg.leads_dir(), cfg.data_dir(), cfg.combined_dir(), cfg.artifacts_dir()):
        os.makedirs(d, exist_ok=True)

    manifest["config"] = asdict(cfg)
    manifest["env"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "energy_grid": {"n": len(grid), "w_min": float(grid[0]),
                        "w_max": float(grid[-1] + (grid[1] - grid[0])),
                        "w_step": float(grid[1] - grid[0])},
        "ncells": ca_sq.NCELLS,
    }
    # what already exists?
    have_leads = os.path.exists(cfg.leads_path())
    combined = [c for c in cfg.concentrations
                if os.path.exists(os.path.join(cfg.combined_dir(), f"conc_{c}.npy"))]
    info = {
        "material": cfg.name, "size": cfg.size,
        "concentrations": cfg.concentrations, "nconfigs": cfg.nconfigs,
        "leads_present": have_leads,
        "combined_present": combined,
    }
    print(f"[ingest] material={cfg.name} size={cfg.size} "
          f"concs={cfg.concentrations} nconfigs={cfg.nconfigs}")
    print(f"[ingest] leads_present={have_leads}  combined_present={len(combined)}/{len(cfg.concentrations)}")
    return info


# ======================================================================
# Stage: leads
# ======================================================================
def stage_leads(cfg, manifest):
    p = cfg.leads_path()
    if os.path.exists(p):
        print(f"[leads] reuse {p}")
        return {"path": p, "computed": False}

    print(f"[leads] computing size={cfg.size} via {cfg.leads_method} ...")
    max_iter = 200 if cfg.leads_method == "sancho-rubio" else 200000
    G_all, iters = cl.compute_size(cfg.size, cfg.d_lead, cfg.t, cfg.e,
                                   cfg.leads_method, cfg.leads_tol, max_iter)
    os.makedirs(cfg.leads_dir(), exist_ok=True)
    np.save(p, G_all)
    grid = cl.energy_grid()
    np.savetxt(os.path.join(cfg.leads_dir(), f"leads_{cfg.size}_meta.csv"),
               np.column_stack((np.arange(len(grid)), grid)),
               delimiter=",", header="row_idx,w", comments="", fmt=["%d", "%.2f"])
    print(f"[leads] saved {p}  shape={G_all.shape}  ({iters} {cfg.leads_method} iters)")
    return {"path": p, "computed": True, "iters": int(iters), "shape": list(G_all.shape)}


# ======================================================================
# Stage: generate (device transmissions, parallel)
# ======================================================================
def stage_generate(cfg, manifest):
    if not os.path.exists(cfg.leads_path()):
        raise SystemExit("[generate] leads missing -- run the 'leads' stage first.")

    written = {}
    for conc in cfg.concentrations:
        out_dir = os.path.join(cfg.data_dir(), f"size_{cfg.size}_conc_{conc}")
        ca_sq.compute_for_concentration(
            conc, cfg.size, cfg.nconfigs, out_dir, cfg.leads_path(),
            cfg.d_device, cfg.t, cfg.e, cfg.jobs(), resume=True)
        n = len([f for f in os.listdir(out_dir) if f.endswith(".npy")])
        written[conc] = n
    print(f"[generate] files per conc: {written}")
    return {"files_per_conc": written}


# ======================================================================
# Stage: combine (stack + configuration-average + pristine)
# ======================================================================
def _pristine(cfg):
    """Pristine (impurity-free) transmission spectrum for this material."""
    leads = np.load(cfg.leads_path())
    w = cl.energy_grid()
    return np.array([ca_sq.device(x, cfg.d_device, cfg.t, cfg.e, cfg.size, 0, 0, leads=leads)
                     for x in w])


def stage_combine(cfg, manifest):
    # pristine, used both for the clip and the inversion normalisation
    pris_path = os.path.join(cfg.artifacts_dir(), f"pristine_{cfg.size}.npy")
    if os.path.exists(pris_path):
        pristine = np.load(pris_path)
        print(f"[combine] reuse pristine {pris_path}")
    else:
        pristine = _pristine(cfg)
        np.save(pris_path, pristine)
        print(f"[combine] saved pristine {pris_path}")

    # stack per-config -> conc_{c}.npy (clipped to pristine, as in CA.ipynb)
    done = []
    for conc in cfg.concentrations:
        out_file = os.path.join(cfg.combined_dir(), f"conc_{conc}.npy")
        combine_sq.combine_concentration(cfg.data_dir(), cfg.combined_dir(),
                                         cfg.size, conc, pristine=pristine)
        if os.path.exists(out_file):
            done.append(conc)
    print(f"[combine] combined concentrations: {done}")
    return {"pristine": pris_path, "combined": done}


# ======================================================================
# Stage: invert (physics misfit inverse model + evaluation)
# ======================================================================
def stage_invert(cfg, manifest):
    combined_dir = cfg.combined_dir()
    concs = [c for c in cfg.concentrations
             if os.path.exists(os.path.join(combined_dir, f"conc_{c}.npy"))]
    if not concs:
        raise SystemExit("[invert] no combined conc_{c}.npy found -- run 'combine' first.")

    lo, hi = cfg.crop_lo, cfg.crop_hi
    h = cfg.holdout_per_conc

    # split each concentration into train (build references) / eval (held out)
    refs, eval_X, eval_y = {}, [], []
    for c in concs:
        arr = np.load(os.path.join(combined_dir, f"conc_{c}.npy"))[:, lo:hi]  # already clipped
        if len(arr) <= h:
            train, test = arr, arr[:0]            # tiny data: eval on train
            test = arr
        else:
            train, test = arr[:-h], arr[-h:]
        refs[c] = train.mean(0)                    # config-averaged reference
        for row in test:
            eval_X.append(row); eval_y.append(c)

    conc_arr = np.array(concs)
    ref_mat = np.stack([refs[c] for c in concs])   # (C, crop_len)
    eval_X = np.asarray(eval_X); eval_y = np.asarray(eval_y)

    # misfit prediction: argmin_c sum|x - ref_c|  (L1 over the cropped spectrum)
    # vectorised over all eval samples
    preds = np.empty(len(eval_X), dtype=int)
    B = 2048
    for s in range(0, len(eval_X), B):
        xb = eval_X[s:s + B][:, None, :]           # (b,1,L)
        d = np.abs(xb - ref_mat[None, :, :]).sum(-1)   # (b,C)
        preds[s:s + B] = conc_arr[d.argmin(1)]

    err = preds - eval_y
    mae = float(np.mean(np.abs(err))) if len(err) else float("nan")
    rmse = float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan")
    acc = float(np.mean(preds == eval_y)) if len(err) else float("nan")

    # save artifacts
    np.save(os.path.join(cfg.artifacts_dir(), "references.npy"), ref_mat)
    np.savetxt(os.path.join(cfg.artifacts_dir(), "reference_concs.csv"),
               conc_arr, fmt="%d", header="concentration", comments="")
    np.savez(os.path.join(cfg.artifacts_dir(), "inversion_eval.npz"),
             y_true=eval_y, y_pred=preds, concs=conc_arr)
    metrics = {"method": cfg.invert_method, "crop": [lo, hi],
               "n_eval": int(len(eval_y)), "MAE": mae, "RMSE": rmse, "accuracy": acc,
               "concentrations": concs}
    with open(os.path.join(cfg.artifacts_dir(), "inversion_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[invert] {cfg.invert_method}: n_eval={len(eval_y)}  "
          f"MAE={mae:.3f}  RMSE={rmse:.3f}  acc={acc:.3f}")

    if cfg.make_plots and len(eval_y):
        _plot_inversion(cfg, refs, concs, eval_y, preds)

    return metrics


def _plot_inversion(cfg, refs, concs, y_true, y_pred):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for c in concs:
        ax[0].plot(refs[c], lw=0.9)
    ax[0].set_title(f"CA reference spectra (size {cfg.size}, crop {cfg.crop_lo}:{cfg.crop_hi})")
    ax[0].set_xlabel("cropped energy index"); ax[0].set_ylabel("mean T (clipped)")

    ax[1].scatter(y_true, y_pred, s=8, alpha=0.3)
    lim = [min(concs) - 2, max(concs) + 2]
    ax[1].plot(lim, lim, "k--", lw=0.8)
    ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_title("misfit inversion: predicted vs true")
    ax[1].set_xlabel("true concentration"); ax[1].set_ylabel("predicted")
    fig.tight_layout()
    out = os.path.join(cfg.artifacts_dir(), "inversion_summary.png")
    fig.savefig(out, dpi=130)
    print(f"[invert] saved {out}")


# ======================================================================
# Runner
# ======================================================================
STAGE_FUNCS = {
    "ingest": stage_ingest, "leads": stage_leads, "generate": stage_generate,
    "combine": stage_combine, "invert": stage_invert,
}


def run(cfg, stages):
    # ingest always runs first (cheap; sets up dirs + manifest)
    stages = [s for s in STAGES if s in stages]
    if "ingest" not in stages:
        stages = ["ingest"] + stages
    manifest = _load_manifest(cfg)

    print(f"\n=== pipeline_sq: material '{cfg.name}' | stages {stages} ===")
    for s in stages:
        t0 = time.time()
        info = STAGE_FUNCS[s](cfg, manifest)
        _record(manifest, s, time.time() - t0, info or {})
        _save_manifest(cfg, manifest)
    print(f"\n=== done. manifest: {cfg.manifest_path()} ===")
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Orchestrate the square-lattice inverse-transport pipeline.")
    ap.add_argument("--config", type=str, default=None, help="JSON/YAML material config.")
    ap.add_argument("--write-config", type=str, default=None,
                    help="Write a template config (from defaults + CLI overrides) and exit.")
    ap.add_argument("--stages", type=str, nargs="+", default=STAGES,
                    choices=STAGES, help="Subset of stages to run (default: all).")
    # inline overrides
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--concs", type=int, nargs="+", default=None)
    ap.add_argument("--nconfigs", type=int, default=None)
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--leads-method", type=str, default=None, choices=["sancho-rubio", "iterative"])
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    cfg = MaterialConfig.load(args.config) if args.config else MaterialConfig()
    # apply overrides
    if args.name is not None: cfg.name = args.name
    if args.size is not None: cfg.size = args.size
    if args.concs is not None: cfg.concentrations = args.concs
    if args.nconfigs is not None: cfg.nconfigs = args.nconfigs
    if args.root is not None: cfg.root = args.root
    if args.n_jobs is not None: cfg.n_jobs = args.n_jobs
    if args.leads_method is not None: cfg.leads_method = args.leads_method
    if args.no_plots: cfg.make_plots = False

    if args.write_config:
        p = cfg.save(args.write_config)
        print(f"wrote template config -> {p}")
        return

    run(cfg, args.stages)


if __name__ == "__main__":
    main()
