#!/usr/bin/env python
"""Plan the reorganisation of the project data into data/<material>/<edge>/N<width>/.

Dry run only: this script reads the current layout, decides where every item goes and writes the plan as a CSV.
It moves nothing. See docs/superpowers/specs/2026-10-23-repo-and-data-hygiene-design.md.

Rule for atlas_store runs: each ribbon (material/edge/N<width>) is taken from the newest production run that has
clouds for it (CANONICAL_RUNS, highest priority first). Every other copy goes to archives/.
conc_v2 clouds are merged into the canonical ribbon only when device length, lead eta and energy grid match.
"""
import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CANONICAL_RUNS = ["materials3_ev_full", "materials2_ev_full", "materials_ev_full", "materials_ev_v1",
                  "materials_v1", "engine_v1", "novelty_v1", "reference_v1"]
MERGE_RUN = "conc_v2"
ARCHIVE_WHOLE = ["quarantine_b4", "quarantine_pristine6", "quarantine_smoke4"]

ROOT_DEFAULT = Path(__file__).resolve().parents[3]  # .../machine_learning (repo is transmission_github/<worktree>)


def tree_size(path):
    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def ribbons_in(run_dir):
    """{model_id: path} for every material/edge/N<width> directory of a run (identified by its pristine or meta file)."""
    found = {}
    for marker in ("meta.json", "pristine.npy", "energies_t.npy"):
        for m in run_dir.rglob(marker):
            parts = m.parent.relative_to(run_dir).parts
            if len(parts) >= 3 and parts[-1].startswith("N"):
                found[str(Path(*parts[-3:]))] = m.parent
    return found


def read_meta(ribbon_dir):
    p = ribbon_dir / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {"clouds": {}}


def n_clouds(ribbon_dir):
    return len(read_meta(ribbon_dir).get("clouds", {}))


def signature(ribbon_dir):
    """(device cells, lead eta, energy grid) of a ribbon, used to decide whether clouds may be combined."""
    meta = read_meta(ribbon_dir)
    clouds = meta.get("clouds", {})
    first = clouds[sorted(clouds)[0]] if clouds else {}
    cells = meta.get("pristine_n_cells", first.get("n_cells"))
    eta = meta.get("pristine_lead_eta", first.get("lead_eta"))
    e = ribbon_dir / "energies_t.npy"
    return cells, eta, (np.load(e) if e.exists() else None)


def compatible(a_dir, b_dir):
    ca, ea, ga = signature(a_dir)
    cb, eb, gb = signature(b_dir)
    why = []
    if ca != cb:
        why.append(f"cells {ca} vs {cb}")
    if ea != eb:
        why.append(f"eta {ea} vs {eb}")
    if ga is None or gb is None or ga.shape != gb.shape or not np.allclose(ga, gb):
        why.append("energy grid differs")
    return not why, "; ".join(why)


def plan_atlas_store(root, add):
    store = root / "atlas_store"
    data, arch = root / "data", root / "archives" / "atlas_store_runs"
    runs = sorted(p.name for p in store.iterdir() if p.is_dir())
    per_run = {r: ribbons_in(store / r) for r in runs if r not in ARCHIVE_WHOLE}
    chosen = {}
    for mid in sorted({m for r in per_run.values() for m in r}):
        cands = [r for r in CANONICAL_RUNS if mid in per_run.get(r, {})]
        with_clouds = [r for r in cands if n_clouds(per_run[r][mid]) > 0]
        pick = (with_clouds or cands or [None])[0]
        if pick:
            chosen[mid] = pick
    n_per_run = Counter(chosen.values())
    for mid, run in chosen.items():
        src = per_run[run][mid]
        add("MOVE", src, data / mid, f"canonical: {run}", tree_size(src))
    # conc_v2 extra densities
    for mid, src in per_run.get(MERGE_RUN, {}).items():
        if mid in chosen:
            canon = per_run[chosen[mid]][mid]
            ok, why = compatible(canon, src)
            have = set(read_meta(canon).get("clouds", {}))
            new = sorted(set(read_meta(src).get("clouds", {})) - have)
            if ok and new:
                add("MERGE", src, data / mid, f"conc_v2: add {len(new)} densities to {chosen[mid]} copy", tree_size(src))
                add("ARCHIVE", src, arch / MERGE_RUN / mid, "conc_v2 source, kept after its new densities were merged", tree_size(src))
            else:
                add("ARCHIVE", src, arch / MERGE_RUN / mid, f"conc_v2 not merged: {why or 'no new densities'}", tree_size(src))
        else:
            add("ARCHIVE", src, arch / MERGE_RUN / mid, "conc_v2: no canonical ribbon", tree_size(src))
    # everything not chosen or merged is archived, run by run (plus the files that sit at a run's top level)
    for r in runs:
        run_dir = store / r
        if r in ARCHIVE_WHOLE:
            add("ARCHIVE", run_dir, arch / r, "quarantined run", tree_size(run_dir))
            continue
        for f in sorted(run_dir.iterdir()):
            if f.is_file():
                add("ARCHIVE", f, arch / r / f.name, "run-level report", f.stat().st_size)
        if r == MERGE_RUN:
            continue
        for mid, src in per_run[r].items():
            if chosen.get(mid) != r:
                add("ARCHIVE", src, arch / r / mid, f"superseded by {chosen.get(mid)}", tree_size(src))
    for f in sorted(store.iterdir()):
        if f.is_file():
            add("ARCHIVE", f, root / "archives" / f.name, "duplicate zip of a run", f.stat().st_size)
    return chosen, n_per_run


def plan_other(root, home, add):
    data, arch = root / "data", root / "archives"
    td = root / "transmission_data"
    res = td / "transmission_results"
    cons = res / "consolidated_data"
    for w in (7, 9):
        dst = data / "graphene-ideal" / "armchair" / f"N{w}" / "legacy"
        add("MOVE", cons / f"size_{w}.npy", dst / f"size_{w}.npy", "consolidated 7/9-AGNR spectra (seeds x channels)", tree_size(cons / f"size_{w}.npy"))
        add("MOVE", cons / f"size_{w}", dst / f"size_{w}", "per-concentration pieces of the consolidated set", tree_size(cons / f"size_{w}"))
    add("MOVE", cons / "metadata.json", data / "graphene-ideal" / "armchair" / "legacy_consolidated_metadata.json", "describes size_7 and size_9", tree_size(cons / "metadata.json"))
    raw = [p for p in sorted(res.iterdir()) if p.name.startswith("size_")]
    for p in raw:
        add("ARCHIVE", p, arch / "raw_simulations" / "transmission_results" / p.name, "raw per-concentration simulation output", tree_size(p))
    for name in ("convert_to_efficient_storage.py", "data_loader.py"):
        add("LEAVE", res / name, res / name, "code: handled in the code-reorganisation step", tree_size(res / name))
    add("ARCHIVE", td / "size_10", arch / "corrupt_square_CA" / "size_10", "corrupt CA.ipynb square CSVs (README warning)", tree_size(td / "size_10"))
    add("ARCHIVE", td / "leads", arch / "raw_simulations" / "transmission_data_leads", "4,400 legacy square-lattice lead CSVs and agnr_7.npy; superseded by the .npy leads", tree_size(td / "leads"))
    add("ARCHIVE", data / "sq_lattice", arch / "corrupt_square_CA" / "sq_lattice", "corrupt CA.ipynb square CSV/XLSX", tree_size(data / "sq_lattice"))
    add("ARCHIVE", data / "graphs", arch / "unrelated" / "graphs", "Glassdoor knowledge graph, unrelated to this project", tree_size(data / "graphs"))
    # atlas_store/leads and ae cache
    add("MOVE", root / "atlas_store" / "leads" / "agnr_cell_v2", data / "graphene-ideal" / "leads" / "agnr_cell_v2", "AGNR cell leads used by tbribbon", tree_size(root / "atlas_store" / "leads"))
    add("MOVE", root / "atlas_store" / "ae_v1_cache", root / "derived" / "ae_v1_cache", "autoencoder training cache (rebuildable)", tree_size(root / "atlas_store" / "ae_v1_cache"))
    # home directory (different drive: COPY, originals left in place)
    sq = home / "transmissions_sq"
    for w in (10, 15, 20):
        add("COPY", sq / f"size_{w}_combined", data / "square" / "strip" / f"N{w}" / "legacy" / f"size_{w}_combined", "clean square-lattice spectra from ca_sq.py", tree_size(sq / f"size_{w}_combined"))
    add("COPY", sq / "pristine_10.npy", data / "square" / "strip" / "N10" / "legacy" / "pristine_10.npy", "square width-10 clean spectrum", tree_size(sq / "pristine_10.npy"))
    add("COPY", sq / "leads", data / "square" / "leads", "square-lattice leads", tree_size(sq / "leads"))
    for p in sorted(sq.iterdir()):
        if p.name.startswith(("size_10_conc", "size_15_conc", "size_20_conc")) or p.suffix in (".log", ".png"):
            add("COPY", p, arch / "raw_simulations" / "transmissions_sq" / p.name, "raw per-concentration output, logs, figures", tree_size(p))
    add("COPY", home / "Desktop" / "backup" / "agnr", data / "graphene-ideal" / "leads" / "agnr_v1_backup", "AGNR leads, widths 5-31 (used by ca_agnr.py)", tree_size(home / "Desktop" / "backup" / "agnr"))
    add("COPY", home / "agnr_infer", root / "results" / "agnr_infer", "agent runs, pristine library and models", tree_size(home / "agnr_infer"))
    add("COPY", home / "atlas_store_old", arch / "home_atlas_store_old", "older copy of atlas_store runs", tree_size(home / "atlas_store_old"))


class PlanError(Exception):
    """The plan cannot be applied safely; nothing has been touched."""


ORDER = {"MOVE": 0, "COPY": 1, "MERGE": 2, "ARCHIVE": 3, "LEAVE": 4}


def _file_count(path):
    p = Path(path)
    return 1 if p.is_file() else sum(1 for f in p.rglob("*") if f.is_file())


def _same_drive(src, dst):
    probe = Path(dst)
    while not probe.exists():
        probe = probe.parent
    return os.stat(src).st_dev == os.stat(probe).st_dev


def _merge_pending(src, dst):
    """Cloud keys in src that dst does not have yet."""
    return sorted(set(read_meta(Path(src)).get("clouds", {})) - set(read_meta(Path(dst)).get("clouds", {})))


def preflight(rows):
    """Every reason the plan cannot be applied, without touching anything."""
    problems = []
    archived = {r["source"]: r["destination"] for r in rows if r["action"] == "ARCHIVE"}
    produced = {r["destination"] for r in rows if r["action"] in ("MOVE", "COPY")}
    claimed = Counter(r["destination"] for r in rows if r["action"] in ("MOVE", "COPY", "ARCHIVE"))
    problems += [f"destination claimed more than once: {d}" for d, n in claimed.items() if n > 1]
    for r in rows:
        a, src, dst = r["action"], Path(r["source"]), Path(r["destination"])
        if a == "LEAVE":
            continue
        if a in ("MOVE", "ARCHIVE"):
            if src.exists() and dst.exists():
                problems.append(f"destination already exists: {dst}")
            elif not src.exists() and not dst.exists():
                problems.append(f"source missing: {src}")
            elif src.exists() and not _same_drive(src, dst):
                problems.append(f"{src} and {dst} are on different drives: use COPY, not {a}")
        elif a == "COPY":
            if not src.exists():
                problems.append(f"source missing: {src}")
            elif dst.exists() and (tree_size(src) != tree_size(dst) or _file_count(src) != _file_count(dst)):
                problems.append(f"destination already exists and differs from its source: {dst}")
        elif a == "MERGE":
            if not src.exists():
                if not (src_arch := archived.get(r["source"])) or not Path(src_arch).exists():
                    problems.append(f"source missing: {src}")
            elif not dst.exists() and r["destination"] not in produced:
                problems.append(f"merge target missing: {dst}")
    return problems


def merge_clouds(src, dst):
    """Copy the density clouds dst lacks from src into dst; a density both have keeps dst's spectra."""
    src, dst = Path(src), Path(dst)
    ok, why = compatible(dst, src)
    if not ok:
        raise PlanError(f"cannot merge {src} into {dst}: {why}")
    src_meta, dst_meta = read_meta(src), read_meta(dst)
    new = _merge_pending(src, dst)
    for key in new:
        for f in [*src.glob(f"cloud_{key}.*"), *src.glob(f"cloud_{key}_*")]:
            shutil.copy2(f, dst / f.name)
        dst_meta["clouds"][key] = src_meta["clouds"][key]
    (dst / "meta.json").write_text(json.dumps(dst_meta, indent=2))
    return len(new)


def _copy_verified(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    if tree_size(src) != tree_size(dst) or _file_count(src) != _file_count(dst):
        raise PlanError(f"copy of {src} is incomplete at {dst}")


def execute(rows, log_path):
    """Apply the plan in dependency order (MOVE, COPY, MERGE, ARCHIVE), logging each item to log_path.

    Safe to repeat: items already done are skipped. If preflight finds any problem, nothing is touched.
    """
    problems = preflight(rows)
    if problems:
        raise PlanError("\n".join(problems))
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_log = not log_path.exists()
    with open(log_path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_log:
            w.writerow(["action", "source", "destination", "bytes", "status"])
        for r in sorted(rows, key=lambda r: ORDER[r["action"]]):
            a, src, dst = r["action"], Path(r["source"]), Path(r["destination"])
            if a == "LEAVE":
                continue
            status = "done"
            if a in ("MOVE", "ARCHIVE"):
                if dst.exists():
                    status = "skipped"
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
            elif a == "COPY":
                if dst.exists():
                    status = "skipped"
                else:
                    _copy_verified(src, dst)
            elif a == "MERGE":
                if not src.exists() or not _merge_pending(src, dst):
                    status = "skipped"
                else:
                    merge_clouds(src, dst)
            w.writerow([a, src, dst, r["bytes"], status])
            fh.flush()


def prune_empty(root):
    """Remove directories left empty under root (never root itself, never a directory with files in it)."""
    for dp, dn, fn in os.walk(root, topdown=False):
        if Path(dp) != Path(root) and not os.path.islink(dp) and not os.listdir(dp):
            os.rmdir(dp)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    ap.add_argument("--home", type=Path, default=Path.home())
    ap.add_argument("--out", type=Path, default=Path("data_move_plan.csv"))
    ap.add_argument("--execute", action="store_true", help="apply the plan (default: only write the plan CSV)")
    ap.add_argument("--log", type=Path, default=None, help="move log, default <root>/data/MOVES.csv")
    args = ap.parse_args()
    rows = []

    def add(action, src, dst, reason, size):
        rows.append(dict(action=action, source=str(src), destination=str(dst), reason=reason, bytes=size))

    chosen, n_per_run = plan_atlas_store(args.root, add)
    plan_other(args.root, args.home, add)
    # destination collisions would silently merge two sources into one place
    dups = [d for d, n in Counter(r["destination"] for r in rows if r["action"] in ("MOVE", "COPY", "ARCHIVE")).items() if n > 1]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["action", "source", "destination", "reason", "bytes"])
        w.writeheader()
        w.writerows(rows)

    by = defaultdict(lambda: [0, 0])
    for r in rows:
        by[r["action"]][0] += 1
        by[r["action"]][1] += r["bytes"]
    print(f"plan written to {args.out}  ({len(rows)} items; nothing was moved)")
    for a, (n, b) in sorted(by.items()):
        print(f"  {a:8s} {n:5d} items  {b / 1e9:8.2f} GB")
    print("\ncanonical ribbons by source run:", dict(n_per_run))
    mats = Counter(m.split("/")[0] for m in chosen)
    print("ribbons per material:", dict(sorted(mats.items())))
    print(f"total canonical ribbons: {len(chosen)}")
    if dups:
        print(f"\nWARNING: {len(dups)} destinations are claimed more than once, e.g. {dups[:3]}")
    if args.execute:
        log = args.log or args.root / "data" / "MOVES.csv"
        execute(rows, log)
        for leftover in ("atlas_store", "transmission_data"):
            prune_empty(args.root / leftover)
        print(f"\nexecuted; every item is logged in {log}")


if __name__ == "__main__":
    main()
