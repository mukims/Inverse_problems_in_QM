"""Reading the sharded parent-lattice training set written by scripts/generate/generate_parent.py (spec section 5)."""
import json
from pathlib import Path

import numpy as np

from tbribbon.parent import TYPES, HexParams
from tbribbon.parent_prior import EDGES, FIELDS, Draw

PARAM_COLUMNS = FIELDS                                     # the columns of the stored `params` array
LEAD_CODES = {t: i for i, t in enumerate(TYPES)}          # C 0, H 1, K 2
EDGE_CODES = {e: i for i, e in enumerate(EDGES)}          # armchair 0, zigzag 1
SPLIT_MOD = 20
SPLIT_RESIDUES = {"val": (18,), "test": (19,), "train": tuple(range(18))}   # by index % 20: 90 / 5 / 5 percent
SHARD_META = ("code_version",)      # per-shard 0-d fields of a shard file (not columns); shards written before them lack them
UNKNOWN_VERSION = "unknown"         # the code version of a shard written before per-shard provenance


def shard_paths(root):
    """The shard files under root, in shard order."""
    return sorted(Path(root).glob("shard_[0-9][0-9][0-9][0-9][0-9].npz"))


def load_manifest(root):
    path = Path(root) / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else None


def data_provenance(root):
    """What an output records about the data it read: {"root", "n_draws", "code_versions", "manifest_code_version"}.

    n_draws is the manifest's draw count (None without a manifest). code_versions are the distinct code versions of the
    shards, sorted: each shard's manifest entry, else its file's 0-d `code_version` field, else UNKNOWN_VERSION (a shard
    written before per-shard provenance). manifest_code_version is the manifest's top-level field (the last writer's)."""
    m = load_manifest(root) or {}
    entries = {e.get("file"): e for e in m.get("shards", [])}
    versions = set()
    for path in shard_paths(root):
        v = entries.get(path.name, {}).get("code_version")
        if v is None:
            with np.load(path) as z:
                v = str(z["code_version"]) if "code_version" in z.files else UNKNOWN_VERSION
        versions.add(v)
    counts = m.get("counts") or {}
    return dict(root=str(root), n_draws=counts.get("n_draws"), code_versions=sorted(versions),
                manifest_code_version=m.get("code_version"))


def split_mask(index, split):
    """Rows of the given draw indices that belong to the split ("train", "val" or "test")."""
    if split not in SPLIT_RESIDUES:
        raise ValueError(f"unknown split {split!r}; use train, val or test")
    return np.isin(np.asarray(index) % SPLIT_MOD, SPLIT_RESIDUES[split])


def load_split(root, split, keys=None):
    """The arrays of one split, concatenated over the shards (keys as in the shard files; all of them unless `keys`
    names the ones to read, which keeps memory low), ordered by draw index. The per-shard fields SHARD_META are not
    columns and are never returned (data_provenance reads them)."""
    parts = []
    for path in shard_paths(root):
        with np.load(path) as z:
            mask = split_mask(z["index"], split)
            parts.append({k: z[k][mask] for k in z.files
                          if k not in SHARD_META and (keys is None or k in keys or k == "index")})
    if not parts:
        raise FileNotFoundError(f"no shard_*.npz under {root}")
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.argsort(out["index"], kind="stable")
    return {k: v[order] for k, v in out.items()}


def draws(arrays, rows):
    """The prior draws (parameters, edge, width, density) of the given rows of a loaded split."""
    out = []
    for r in np.atleast_1d(np.arange(len(arrays["index"]))[rows]):
        on = [bool(v) for v in arrays["on"][r]]
        params = HexParams(*on, TYPES[int(arrays["lead"][r])],
                           **{name: float(v) for name, v in zip(PARAM_COLUMNS, arrays["params"][r])})
        out.append(Draw(params, EDGES[int(arrays["edge"][r])], int(arrays["width"][r]), float(arrays["density"][r])))
    return out
