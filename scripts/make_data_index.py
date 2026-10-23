#!/usr/bin/env python
"""Write data/INDEX.csv (one row per ribbon) and data/README.md for the organised data tree.

    python scripts/make_data_index.py --data /run/media/shardul/storage/machine_learning/data
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

EDGES = ("armchair", "zigzag", "strip")
FIELDS = ["material", "edge", "width", "model_id", "n_cells", "lead_eta", "formula", "n_densities",
          "density_min", "density_max", "seeds_per_cloud", "has_legacy"]


def ribbon_dirs(data):
    for p in sorted(Path(data).glob("*/*/N*")):
        if p.is_dir() and p.parent.name in EDGES and re.fullmatch(r"N\d+", p.name) and (
                (p / "meta.json").exists() or (p / "pristine.npy").exists()):
            yield p


def _g(x):
    return f"{x:g}"


def describe(data, p):
    meta = json.loads((p / "meta.json").read_text()) if (p / "meta.json").exists() else {}
    clouds = meta.get("clouds", {})
    dens = sorted(c["density"] for c in clouds.values())
    seeds = sorted({c.get("n") for c in clouds.values() if c.get("n") is not None})
    first = clouds[sorted(clouds)[0]] if clouds else {}
    return dict(
        material=p.parent.parent.name, edge=p.parent.name, width=int(p.name[1:]),
        model_id=str(p.relative_to(data)),
        n_cells=meta.get("pristine_n_cells", first.get("n_cells", "")),
        lead_eta=meta.get("pristine_lead_eta", first.get("lead_eta", "")),
        formula=meta.get("pristine_formula", first.get("formula", "")),
        n_densities=len(dens), density_min=_g(dens[0]) if dens else "", density_max=_g(dens[-1]) if dens else "",
        seeds_per_cloud="/".join(str(s) for s in seeds),
        has_legacy="yes" if (p / "legacy").is_dir() else "no")


def write_index(data, out):
    data = Path(data)
    rows = sorted((describe(data, p) for p in ribbon_dirs(data)),
                  key=lambda r: (r["material"], r["edge"], r["width"]))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def counts_by_material(rows):
    return dict(sorted(Counter(r["material"] for r in rows).items()))


README = """# Data

One folder per ribbon: `data/<material>/<edge>/N<width>/`. This file is written by
`scripts/make_data_index.py`; `INDEX.csv` lists every ribbon with the settings it was generated with.

## What is here

{table}

{total} ribbons in all. `edge` is `armchair`, `zigzag` or `strip` (square and triangular toy lattices);
`N` is the number of atomic rows across the ribbon.

## Inside one ribbon folder

| File | Content |
|---|---|
| `energies_t.npy` | energy axis, in units of the material's own nearest-neighbour hopping t (`t_ev` in the registry converts to eV) |
| `pristine.npy` | clean transmission T(E), same length as the energy axis |
| `cloud_d<density>.npy` | disordered spectra at one impurity density, shape `(seeds, energies)`, density = impurity atoms / device atoms |
| `cloud_d<density>_seeds.npy` | configuration seed of each row. A seed's impurities are nested across densities, so **split train/test by seed** |
| `meta.json` | device length (`pristine_n_cells`), lead broadening (`pristine_lead_eta`), transmission formula, and one entry per density |
| `legacy/` | older-format arrays kept as they are (see below). Only some ribbons have it |

Load one:

```python
from atlaslib import CloudStore          # PYTHONPATH=notebooks/material_atlas:notebooks until the code move lands
store = CloudStore("data")
E, pristine = store.read_pristine("hbn/armchair/N10")
spectra, seeds = store.read_cloud("hbn/armchair/N10", 0.01)   # density 1%
```

## Other folders

* `data/<material>/leads/`: precomputed lead surface Green's functions. `graphene-ideal/leads/agnr_cell_v2` is what the
  tight-binding engine uses; `agnr_v1_backup` holds the AGNR leads for widths 5-31 used by the older `ca_agnr.py`;
  `square/leads` holds square-strip leads for widths 5-60.
* `data/graphene-ideal/armchair/N7/legacy/size_7.npy` and `N9/legacy/size_9.npy`: the original consolidated 7- and
  9-AGNR spectra, shape `(concentrations, 10000 seeds, 300 channels)`, energies 0-2.99 eV in 0.01 eV steps;
  `legacy_consolidated_metadata.json` describes them. Impurities are counted, not given as densities (7-AGNR 2-68, 9-AGNR 2-98).
* `data/square/strip/N10|N15|N20/legacy/`: the clean square-lattice sets from `ca_sq.py`
  (`conc_<c>.npy` + `conc_<c>_meta.csv`, 400 channels). Never use the `CA.ipynb` CSVs; they are corrupt.
* `derived/ae_v1_cache`: autoencoder training cache, rebuildable.
* `archives/`: everything superseded, with the original folder names and a record in `data/MOVES.csv`
  (old path, new path) so every path in `LOGBOOK.md` can be traced.

## Rules every analysis must follow

1. Split by configuration seed, never by random rows.
2. Make everything before identification label-free (no division by the true material's clean spectrum).
3. Round T to 3 decimals before dividing by a clean spectrum.

Each ribbon comes from the newest production run that has it (the 12 non-graphene materials from FULL-6, graphene from
`engine_v1`, square from `novelty_v1`). Where a second run only added densities with identical device length, lead
broadening and energy grid, those densities were merged in. Ribbons differ in device length and lead broadening between
materials: check `INDEX.csv` before comparing across them.
"""


def write_readme(data, rows, out):
    lines = ["| Material | Ribbons | Widths | Edges |", "|---|---|---|---|"]
    for m, n in counts_by_material(rows).items():
        rs = [r for r in rows if r["material"] == m]
        widths = sorted({r["width"] for r in rs})
        lines.append(f"| {m} | {n} | {widths[0]}-{widths[-1]} | {', '.join(sorted({r['edge'] for r in rs}))} |")
    Path(out).write_text(README.format(table="\n".join(lines), total=len(rows)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    args = ap.parse_args()
    rows = write_index(args.data, args.data / "INDEX.csv")
    write_readme(args.data, rows, args.data / "README.md")
    print(f"{len(rows)} ribbons -> {args.data / 'INDEX.csv'} and README.md")


if __name__ == "__main__":
    main()
