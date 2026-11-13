"""The reward-run table (spec docs/superpowers/specs/2026-11-04-reward-finetuning-design.md, §7) for any set of models.

    TRANSMISSION_DATA=... PYTHONPATH=src nice -n 10 python scripts/evaluate/compare_reward.py \\
        results/inverse_design/model_v2_depth_bandtop_600k [more model folders] --rows all --decode argmax physics

On the val split (all 30,000 rows, or the fixed seeded 5,000-row subset the fine-tuning selects on), each model with
its own input and the same seeds:

* family (family + edge, probability summed over lead): top-1, top-3, NLL, ECE (15 bins of the top family's
  probability), all by pattern too; under --decode physics the answer is the state of lowest expected physics cost
  (the D-phys control), scored by its family;
* the physics family cost of the answer (mean, from reward_tables' val table);
* width: modal width of --samples posterior draws within +-1 / off by more than 3, the 90% interval's hit rate;
* per-parameter 90% coverage of the lead-free physical vector (eV), on rows whose top family is right, from the
  samples in that family.

Writes compare_reward_<rows>.{txt,json} (all models in one table; rows = subset or all) into --out; model folders
are only read.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import paths
from shazam.inverse import data, reward as rw
from shazam.inverse.calibration import expected_calibration_error
from shazam.inverse.design import model_input
from shazam.inverse.encoding import decode, encode
from shazam.inverse.model import EnsembleInverseModel, InverseModel
from shazam.inverse.runio import atomic_write_json, code_version, sha256_of
from shazam.inverse.verify import VERIFY_SEEDS, recreation_check, simulate_inputs
from tbribbon import parent_data
from tbribbon.parent import device_cells_for

DEFAULT_DATA = paths.data_root() / "_generated" / "hexagonal_parent_v1"
DEFAULT_TABLES = paths.results_dir("inverse_design", "reward_tables")
NEEDED = ("index", "x", "raw", "on", "lead", "params", "edge", "width", "density")
ECE_BINS = 15
family_probs = rw.family_probs


def decode_states(state_probs, cost=None):
    """The answer state per row: the argmax, or with a (156, 156) table cost[i, j] (cost of predicting j when the truth
    is i) the state of lowest expected cost under the model, argmin_j (p @ cost)[j]."""
    p = np.asarray(state_probs)
    return p.argmax(1) if cost is None else (p @ np.asarray(cost)).argmin(1)


def family_ece(conf, hits):
    return expected_calibration_error(conf, hits, n_bins=ECE_BINS)


def family_metrics(state_probs, truth_states, decoded=None, physics_rows=None):
    """Family top-1 (and its per-row hits), top-3, NLL and ECE, plus the mean physics family cost of the answer.
    decoded None: the headline (argmax of the family probabilities; the argmax state for the physics cost)."""
    fam = family_probs(state_probs)
    true_fam = rw.FAMILY_OF[np.asarray(truth_states)]
    rows = np.arange(len(fam))
    p_true = fam[rows, true_fam]
    if decoded is None:
        hits = fam.argmax(1) == true_fam
        answer = np.asarray(state_probs).argmax(1)
    else:
        hits = rw.FAMILY_OF[decoded] == true_fam
        answer = decoded
    top3 = (fam > p_true[:, None]).sum(1) < 3
    out = dict(family_top1=float(hits.mean()), family_top1_hits=hits.astype(int), family_top3=float(top3.mean()),
               family_nll=float(-np.log(np.clip(p_true, 1e-12, None)).mean()),
               family_ece=family_ece(fam.max(1), fam.argmax(1) == true_fam))
    if physics_rows is not None:
        out["physics_cost"] = float(np.asarray(physics_rows)[rows, answer].mean())
    return out


def width_metrics(width_samples, truth_width):
    w = np.asarray(width_samples)
    mode = np.array([np.bincount(r).argmax() for r in w])
    err = np.abs(mode - np.asarray(truth_width))
    lo, hi = np.quantile(w, 0.05, axis=1), np.quantile(w, 0.95, axis=1)
    return dict(width_within1=float((err <= 1).mean()), width_over3=float((err > 3).mean()),
                width_in90=float(((truth_width >= lo) & (truth_width <= hi)).mean()))


def coverage_metrics(sample_phys, sample_fam, truth_phys, truth_fam, top_fam):
    """Per physical quantity (EV_NAMES and log c): share of rows whose 5-95% interval holds the truth, over rows whose
    top family is right and where the truth's quantity is nonzero, using that family's samples only."""
    names = rw.EV_NAMES + ("log_c",)
    hits = {n: [] for n in names}
    for i in np.flatnonzero(top_fam == truth_fam):
        s = sample_phys[i][sample_fam[i] == truth_fam[i]]
        if len(s) < 10:
            continue
        lo, hi = np.quantile(s, 0.05, axis=0), np.quantile(s, 0.95, axis=0)
        for j, n in enumerate(names):
            if truth_phys[i, j] != 0:
                hits[n].append(lo[j] <= truth_phys[i, j] <= hi[j])
    return {n: (float(np.mean(v)) if v else None) for n, v in hits.items()}


@torch.no_grad()
def model_outputs(model, x, n_samples, seed=0, batch=50):
    """(state probabilities (n, 156), sample states (n, S), sample physical vectors (n, S, N_PHYS) float32)."""
    model.eval()
    probs, st, ph = [], [], []
    gen = torch.Generator().manual_seed(seed)
    is_ens = hasattr(model, "models")
    sub_models = model.models if is_ens else [model]
    for i in range(0, len(x), batch):
        xb = x[i:i + batch]
        if is_ens:
            p_list = []
            h_list = []
            for m in sub_models:
                h_m = m.encode(xb)
                h_list.append(h_m)
                p_m = F.softmax(m.discrete(h_m).masked_fill(~m.mask_known, -torch.inf), 1)
                p_list.append(p_m)
            p = torch.stack(p_list, dim=0).mean(dim=0)
        else:
            h = model.encode(xb)
            p = F.softmax(model.discrete(h).masked_fill(~model.mask_known, -torch.inf), 1)
        probs.append(p.numpy())
        if n_samples:
            s = torch.multinomial(p, n_samples, replacement=True, generator=gen).reshape(-1)
            if is_ens:
                n_m = len(sub_models)
                z = torch.zeros(len(s), model.n_cont, dtype=xb.dtype)
                chunk = (len(s) + n_m - 1) // n_m
                for m_idx, (m, h_m) in enumerate(zip(sub_models, h_list)):
                    start = m_idx * chunk
                    end = min(start + chunk, len(s))
                    if start >= len(s):
                        break
                    s_sub = s[start:end]
                    h_full = h_m.repeat_interleave(n_samples, 0)
                    h_sub = h_full[start:end]
                    z0_sub = torch.randn(len(s_sub), m.n_cont, generator=gen)
                    z[start:end] = m.sample_batch(h_sub, s_sub, z0_sub)
            else:
                z0 = torch.randn(len(s), model.n_cont, generator=gen)
                z = model.sample_batch(h.repeat_interleave(n_samples, 0), s, z0)
            st.append(s.reshape(len(xb), n_samples).numpy())
            ph.append(rw.physical_torch(s, z).reshape(len(xb), n_samples, -1).numpy().astype(np.float32))
    return np.concatenate(probs), (np.concatenate(st) if st else None), (np.concatenate(ph) if ph else None)


def evaluate(model, arrays, physics_rows, ss_table, n_samples, decodes):
    draws = data.draws(arrays, slice(None))
    truth = np.array([encode(d)[0] for d in draws])
    truth_phys = np.stack([rw.physical(d) for d in draws])
    kind = model.cfg.get("input", "log")
    x = torch.as_tensor(np.asarray({"log": lambda: arrays["x"], "bandtop": lambda: parent_data.to_input_bandtop(
        arrays["raw"]), "linear": lambda: parent_data.to_input_linear(arrays["raw"])}[kind](), dtype=np.float32))
    probs, s_states, s_phys = model_outputs(model, x, n_samples)
    pattern = np.array([rw.pattern_name(d.params.pattern) for d in draws])
    out = {}
    for decode in decodes:
        decoded = None if decode == "argmax" else decode_states(probs, ss_table)
        m = family_metrics(probs, truth, decoded, physics_rows)
        hits = m.pop("family_top1_hits")
        m["family_top1_by_pattern"] = {p: float(hits[pattern == p].mean())
                                       for p in rw.PATTERN_NAMES if (pattern == p).any()}
        out[decode] = m
    if n_samples:
        widths = np.clip(np.rint(s_phys[..., -1]), 4, 22).astype(int)
        out["width"] = width_metrics(widths, truth_phys[:, -1])
        fam_s = rw.FAMILY_OF[s_states]
        top_fam = family_probs(probs).argmax(1)
        out["coverage"] = coverage_metrics(s_phys[..., :-1], fam_s, truth_phys[:, :-1], rw.FAMILY_OF[truth], top_fam)
    return out


def _simulate_ensemble(draw, seeds, row, n_jobs=22):
    n_imp = parent_data.n_impurities(draw, device_cells_for(draw.params, draw.edge))
    return simulate_inputs(draw, n_imp, seeds, n_jobs=n_jobs)


@torch.no_grad()
def recreation_rate(model, x_model, target_x, n_samples=200, simulate_ensemble=_simulate_ensemble, seeds=VERIFY_SEEDS):
    """Share of rows whose top candidate recreates the target: the argmax state, its continuous values at the median of
    n_samples flow samples in that state, simulated on the verify seeds and judged by verify.recreation_check."""
    model.eval()
    rows = []
    for i in range(len(x_model)):
        h = model.encode(x_model[i:i + 1])
        k = int(model.discrete(h).masked_fill(~model.mask_known, -torch.inf).argmax(1))
        z0 = torch.randn(n_samples, model.n_cont)
        z = model.sample_batch(h.expand(n_samples, -1), torch.full((n_samples,), k), z0)
        draw = decode(k, z.median(0).values.numpy())
        check = recreation_check(target_x[i], simulate_ensemble(draw, seeds, i))
        rows.append(dict(row=i, state=k, recreated=check.recreated, p_value=check.p_value, misfit=check.misfit,
                         threshold=check.threshold))
    return float(np.mean([r["recreated"] for r in rows])), rows


def _table(results):
    lines = []
    names = list(results)
    decodes = [d for d in ("argmax", "physics") if d in results[names[0]]]
    head = f"{'model':40s} {'decode':8s} {'top1':>6s} {'top3':>6s} {'nll':>6s} {'ece':>6s} {'phys':>6s}"
    lines.append(head)
    for n in names:
        for d in decodes:
            m = results[n][d]
            lines.append(f"{n:40s} {d:8s} {m['family_top1']:6.4f} {m['family_top3']:6.4f} {m['family_nll']:6.3f} "
                         f"{m['family_ece']:6.4f} {m.get('physics_cost', float('nan')):6.4f}")
    lines.append("")
    pats = list(results[names[0]][decodes[0]]["family_top1_by_pattern"])
    lines.append(f"{'family top-1 by pattern':40s} {'decode':8s} " + " ".join(f"{p:>6s}" for p in pats))
    for n in names:
        for d in decodes:
            bp = results[n][d]["family_top1_by_pattern"]
            lines.append(f"{n:40s} {d:8s} " + " ".join(f"{bp[p]:6.3f}" for p in pats))
    if "width" in results[names[0]]:
        lines.append("")
        lines.append(f"{'width / coverage':40s} {'w±1':>6s} {'w>3':>6s} {'w in90':>6s}  per-parameter 90% coverage")
        for n in names:
            w, c = results[n]["width"], results[n]["coverage"]
            cov = " ".join(f"{k}={v:.3f}" for k, v in c.items() if v is not None)
            lines.append(f"{n:40s} {w['width_within1']:6.4f} {w['width_over3']:6.4f} {w['width_in90']:6.4f}  {cov}")
    if "recreation" in results[names[0]]:
        lines.append("")
        for n in names:
            r = results[n]["recreation"]
            lines.append(f"{n:40s} recreation rate {r['rate']:.3f} on {r['n']} val rows")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("models", nargs="+", type=Path, help="model folders (each with model.pt)")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--tables", default=str(DEFAULT_TABLES), type=Path)
    ap.add_argument("--rows", choices=("subset", "all"), default="subset")
    ap.add_argument("--subset", type=int, default=5000)
    ap.add_argument("--decode", nargs="+", choices=("argmax", "physics"), default=["argmax", "physics"])
    ap.add_argument("--samples", type=int, default=200, help="posterior draws per row for width and coverage (0: none)")
    ap.add_argument("--recreation", type=int, default=0,
                    help="also the recreation rate on this many seeded val rows (50 simulations each)")
    ap.add_argument("--ensemble", action="store_true", help="also evaluate the ensemble of all specified models")
    ap.add_argument("--drop-pattern", action="append", default=[], help="drop pattern at inference time (repeatable)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", type=Path, default=paths.results_dir("inverse_design", "reward_compare"))
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.time()
    va = data.load_split(a.data, "val", NEEDED)
    phys_tab = np.load(a.tables / "val_cost_physics.npz")
    if not np.array_equal(phys_tab["index"], va["index"]):
        raise ValueError("val tables do not match the data rows: rerun scripts/build/reward_tables.py")
    rows = (np.arange(len(va["index"])) if a.rows == "all"
            else data.seeded_rows(len(va["index"]), a.subset))
    rec_rows = data.seeded_rows(len(va["index"]), a.recreation, seed=2) if a.recreation else None
    rec = {k: v[rec_rows] for k, v in va.items()} if a.recreation else None
    va = {k: v[rows] for k, v in va.items()}
    physics_rows = phys_tab["cost"][rows].astype(np.float32)
    ss = np.load(a.tables / "state_state_physics.npy")
    results, meta, loaded_models = {}, {}, {}
    for folder in a.models:
        model = InverseModel.load(folder / "model.pt")
        loaded_models[folder.name] = model
        results[folder.name] = evaluate(model, va, physics_rows, ss, a.samples, a.decode)
        if rec is not None:
            torch.manual_seed(0)
            x_rec = torch.as_tensor(np.stack([model_input(x, model.cfg.get("input", "log")) for x in rec["x"]]))
            rate, per_row = recreation_rate(model, x_rec, rec["x"])
            results[folder.name]["recreation"] = dict(rate=rate, n=len(per_row), index=rec["index"].tolist(),
                                                      rows=per_row)
        meta[folder.name] = dict(path=str(folder), model_sha256=sha256_of(folder / "model.pt"))
        print(f"[compare] {folder.name} done ({time.time() - t0:.0f} s)", flush=True)

    if a.ensemble and len(a.models) > 1:
        ens_model = EnsembleInverseModel(list(loaded_models.values()), drop_patterns=tuple(a.drop_pattern))
        ens_name = "ensemble"
        results[ens_name] = evaluate(ens_model, va, physics_rows, ss, a.samples, a.decode)
        meta[ens_name] = dict(models=[str(f) for f in a.models], drop_patterns=list(a.drop_pattern))
        print(f"[compare] {ens_name} done ({time.time() - t0:.0f} s)", flush=True)
    text = _table(results)
    print(text, flush=True)
    report = dict(results=results, models=meta, rows=a.rows, n_rows=len(rows), samples=a.samples,
                  code_version=code_version(), data_provenance=data.data_provenance(a.data))
    a.out.mkdir(parents=True, exist_ok=True)
    stem = a.out / f"compare_reward_{a.rows}"
    atomic_write_json(stem.with_suffix(".json"), report)
    stem.with_suffix(".txt").write_text(text + "\n")
    print(f"[compare] wrote {stem}.{{txt,json}} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
