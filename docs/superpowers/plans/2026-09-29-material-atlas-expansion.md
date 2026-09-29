# Material Atlas Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable, label-free map of transport fingerprints: first as the best-possible 7/9-AGNR reference solution, then populated with a width expansion and with real tight-binding materials, so any new ribbon can be added to the map without rebuilding it.

**Architecture:** Two packages with one-way dependency. `tbribbon` (a generic tight-binding ribbon engine) turns a registered ribbon model into spectra. `atlaslib` (the reusable map) stores spectra per model in an append-only on-disk store, embeds them with a frozen autoencoder under a versioned input spec, and answers `locate(T)` with material → edge → continuous width, a density estimate and a novelty flag. Work proceeds in six iterative phases; each ends with a measurable exit criterion and a commit, and phases 3b and 6 get their own plans once their inputs exist.

**Tech Stack:** Python 3.12 (conda env `~/miniconda3/envs/ml`), NumPy, PyTorch 2.13 (CPU), scikit-learn 1.9, XGBoost, pytest 9, matplotlib.

**Spec:** `docs/superpowers/specs/2026-09-29-material-atlas-design.md`

## Global Constraints

- Input spec v1: window 0–4 t, step 0.01 t, **400 channels**, positive energies only; transform `log1p(clip(round(T, 3), 0, 20)) / log1p(20)`, identical for every spectrum.
- Energy unit t = largest nearest-neighbour hopping magnitude; E = 0 at charge neutrality.
- Ribbons: 100 unit cells, same-material leads, impurity on-site shift **V = 0.5 t**, sites drawn with `np.random.RandomState(seed).choice(n_sites, n, replace=False)`, site index = `cell * sites_per_cell + orbital`.
- Widths N ∈ {7, 9, 14, 27, 50}, edges armchair **and** zigzag for every real material; order graphene-realistic → hBN → MoS₂ → phosphorene.
- Atlas densities {0.005, 0.01, 0.02, 0.04} plus the pristine anchor; seeds per density 1,000 (N ≤ 14), 300 (N = 27), 100 (N = 50).
- Always split train/validation/test by configuration seed, never by random rows (LOGBOOK Bug #7).
- Nothing applied before identification may depend on the material's identity (LOGBOOK Bug #8).
- 7/9 reference success criteria: label-free width accuracy ≥ 99.5%; end-to-end concentration MAE ≤ 1.98; 90% intervals with coverage 90 ± 2%.
- CPU: PyTorch ≤ 4 threads while other trainings run; generators set `OMP_NUM_THREADS=1` per worker; never run two large trainings on the same cores.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A spectrum on a different energy grid, not starting at 0, or ending before its band top** — expected: `InputSpec.to_input` refuses it instead of zero-filling real signal (tests in Task 1).
2. **NaN/inf spectra from a singular inversion, or byte-identical spectra across densities** (the square-lattice generator bug) — expected: `CloudStore.write_cloud` refuses them (Task 3).
3. **A width outside the trained range** (e.g. N = 50 when the map knows N ≤ 14) — expected: the width estimate stays inside the known range and is flagged `width_extrapolated` (Task 5).
4. **A spectrum from a material the map has never seen** — expected: `unknown = True` rather than a confident wrong label (Task 5). Flag by reconstruction error: in BUILD-12 it separated an unseen material perfectly (AUROC 1.00) while k-NN distance did not. A new *width* of a known material is not flagged (AUROC 0.79 at best); it is placed on the continuous width axis instead.
5. **Adding a material to an existing map** — expected: existing references and existing answers are unchanged, and the call reports what fraction of the new spectra the frozen encoder finds unfamiliar (Task 6).

## Open decisions (settle before the listed task)

- **D1 — transmission formula for new clouds** (before Task 13): `"caroli"` (bounded, standard; recommended) or `"legacy_trace"` (matches existing data). The engine implements both; D1 only picks the default used by the generator.
- **D2 — energy unit for measured spectra** (before Phase 6): units of t (model space) vs eV. Phases 1–5 use units of t.
- **D3 — real-material parameters** (before Phase 3b). Literature lookup done: graphene nearest-neighbour t = 2.7 eV with armchair edge-bond factor 1.12 (Son 2006); hBN t = 2.30 eV, Δ = 3.625 eV (Galvani 2016, GW gap) vs t = 2.33, Δ = 1.96 eV (Ribeiro–Peres 2011, GGA gap); MoS₂ three-band GGA vs LDA fit (Liu 2013); phosphorene five-hopping set (Rudenko–Katsnelson 2014) vs ten-hopping set (2015).
- **D4 — energy convention for MoS₂ and phosphorene** (before Phase 3b). In units of the largest nearest-neighbour hopping, MoS₂ spans about −1.1 t to +6.9 t with E = 0 near the valence-band top, not at charge neutrality; phosphorene's mid-gap sits at −0.42 eV. The 0–4 t window therefore misses most of MoS₂'s conduction band.
- **D5 — positive energies only vs particle-hole asymmetric materials** (before Phase 3b). For hBN (symmetric) nothing is lost; for MoS₂ and phosphorene the valence band lies below charge neutrality and a positive-only window drops it.

## File Structure

```
notebooks/material_atlas/atlaslib/
    __init__.py        exports InputSpec, RibbonModel, Registry, CloudStore, Atlas, Located
    spec.py            InputSpec: versioned, label-free input transform
    registry.py        RibbonModel (one ribbon type) + Registry (JSON-backed catalogue)
    store.py           CloudStore: append-only on-disk spectra per model, with validation
    encoder.py         Conv1dAE + train_autoencoder + embed
    atlas.py           Atlas: build, locate, add_models, save, load
    conformal.py       relative split-conformal intervals
    importers.py       legacy 7/9-AGNR and square data -> CloudStore
notebooks/material_atlas/run_reference_7_9.py   Phase 1 end-to-end reference evaluation
notebooks/tbribbon/
    __init__.py
    lattices.py        honeycomb_ribbon, square_strip -> RibbonHamiltonian(H0, H1, ...)
    bands.py           band_edges, open_channels
    leads.py           surface_gf (Sancho-Rubio), LeadCache
    transport.py       transmission (caroli | legacy_trace), spectrum
    disorder.py        impurity_shifts
    materials.py       MATERIALS catalogue, hamiltonian_for, make_model
    generate_clouds.py seeds_for_width, generate (parallel, resumable)
    fingerprints.py    pristine fingerprints -> store + plot
notebooks/material_atlas/build_atlas_v2.py      Phase 5 map + generalisation tests
tests/conftest.py, tests/atlas/*.py, tests/tbribbon/*.py
pytest.ini
```

---

# Phase 1 — Reusable atlas core and the 7/9-AGNR reference solution

**Exit:** all tests pass; `run_reference_7_9.py` meets the three success criteria (or reports by how much it misses); report tab and LOGBOOK updated.

### Task 1: Input specification

**Files:**
- Create: `pytest.ini`, `tests/conftest.py`, `notebooks/material_atlas/atlaslib/__init__.py`, `notebooks/material_atlas/atlaslib/spec.py`
- Test: `tests/atlas/test_spec.py`

**Interfaces:**
- Produces: `InputSpec(version="v1", e_max_t=4.0, step_t=0.01, cap=20.0)`; `.n_channels -> int`; `.energies_t() -> np.ndarray`; `.to_input(T, e_t, band_top_t=None) -> np.ndarray[float32] (N, n_channels)`; `.as_dict() -> dict`.

- [ ] **Step 1: Test scaffolding**

`pytest.ini`:
```ini
[pytest]
testpaths = tests
pythonpath = notebooks/material_atlas notebooks
```
`tests/conftest.py`:
```python
import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)
```

- [ ] **Step 2: Write the failing tests**

`tests/atlas/test_spec.py`:
```python
import numpy as np
import pytest
from atlaslib.spec import InputSpec


def test_grid_is_400_channels_from_zero():
    s = InputSpec()
    assert s.n_channels == 400
    e = s.energies_t()
    assert e[0] == 0.0 and abs(e[-1] - 3.99) < 1e-12


def test_zero_fill_above_band_top_is_exact():
    s = InputSpec()
    e_t = np.arange(300) * 0.01                     # legacy AGNR grid, 0-2.99
    T = np.full((2, 300), 2.0)
    X = s.to_input(T, e_t, band_top_t=3.0)
    assert X.shape == (2, 400)
    assert np.all(X[:, 300:] == 0.0)
    assert np.allclose(X[:, :300], np.log1p(2.0) / np.log1p(20.0))


def test_refuses_to_zero_fill_real_band():
    s = InputSpec()
    e_t = np.arange(300) * 0.01
    with pytest.raises(ValueError, match="band"):
        s.to_input(np.ones((1, 300)), e_t, band_top_t=3.9)


def test_default_requires_full_window():
    s = InputSpec()
    with pytest.raises(ValueError):
        s.to_input(np.ones((1, 300)), np.arange(300) * 0.01)   # no band top given -> must cover 0-3.99


def test_rejects_grid_not_starting_at_zero_or_mismatched():
    s = InputSpec()
    with pytest.raises(ValueError):
        s.to_input(np.ones((1, 400)), np.arange(1, 401) * 0.01, band_top_t=4.0)
    with pytest.raises(ValueError):
        s.to_input(np.ones((1, 399)), np.arange(400) * 0.01, band_top_t=4.0)


def test_cap_and_rounding():
    s = InputSpec()
    e_t = np.arange(400) * 0.01
    T = np.zeros((1, 400)); T[0, 0] = 100.0; T[0, 1] = 0.0004
    X = s.to_input(T, e_t, band_top_t=4.0)
    assert X[0, 0] == pytest.approx(1.0) and X[0, 1] == 0.0


def test_other_grid_is_interpolated():
    s = InputSpec()
    e_t = np.linspace(0, 4.0, 81)                    # e.g. eV data converted with t = 2.7
    X = s.to_input(np.tile(e_t, (1, 1)), e_t, band_top_t=4.0)
    assert X[0, 100] == pytest.approx(np.log1p(1.0) / np.log1p(20.0), abs=1e-3)
```

- [ ] **Step 3: Run to verify failure**

Run: `~/miniconda3/envs/ml/bin/python -m pytest tests/atlas/test_spec.py -v`
Expected: FAIL (`ModuleNotFoundError: atlaslib`)

- [ ] **Step 4: Implement**

`notebooks/material_atlas/atlaslib/spec.py`:
```python
"""Versioned, label-free input specification shared by every atlas component."""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class InputSpec:
    version: str = "v1"
    e_max_t: float = 4.0     # window [0, e_max_t) in units of the material's hopping t
    step_t: float = 0.01
    cap: float = 20.0        # G0; clipped before the log so spikes cannot dominate

    @property
    def n_channels(self) -> int:
        return int(round(self.e_max_t / self.step_t))

    def energies_t(self) -> np.ndarray:
        return np.arange(self.n_channels) * self.step_t

    def to_input(self, T, e_t, band_top_t=None) -> np.ndarray:
        """Map raw T(E) on grid e_t (units of t, from 0) to the shared input.

        Channels above band_top_t are set to zero (exact: no propagating states there).
        band_top_t=None means the data itself must cover the whole window.
        """
        T = np.atleast_2d(np.asarray(T, dtype=np.float64))
        e_t = np.asarray(e_t, dtype=np.float64)
        if T.shape[1] != e_t.size:
            raise ValueError(f"T has {T.shape[1]} channels but e_t has {e_t.size}")
        if abs(e_t[0]) > 1e-9 or np.any(np.diff(e_t) <= 0):
            raise ValueError("e_t must start at 0 and increase")
        grid = self.energies_t()
        top = grid[-1] if band_top_t is None else min(float(band_top_t), grid[-1])
        if e_t[-1] < top - self.step_t / 2:
            raise ValueError(f"data ends at {e_t[-1]:.3f} t but the band extends to {top:.3f} t; "
                             "zero-filling would erase real signal")
        inside = grid <= top + 1e-9
        out = np.zeros((T.shape[0], grid.size))
        n = int(inside.sum())
        if e_t.size >= n and np.allclose(e_t[:n], grid[:n]):
            out[:, :n] = T[:, :n]
        else:
            for i in range(T.shape[0]):
                out[i, inside] = np.interp(grid[inside], e_t, T[i])
        out = np.clip(np.round(out, 3), 0.0, self.cap)
        return (np.log1p(out) / np.log1p(self.cap)).astype(np.float32)

    def as_dict(self) -> dict:
        return asdict(self)
```
`notebooks/material_atlas/atlaslib/__init__.py`:
```python
from .spec import InputSpec  # noqa: F401
```

- [ ] **Step 5: Run to verify pass**

Run: `~/miniconda3/envs/ml/bin/python -m pytest tests/atlas/test_spec.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add pytest.ini tests/conftest.py tests/atlas/test_spec.py notebooks/material_atlas/atlaslib/
git commit -m "feat(atlas): versioned label-free input spec (0-4t, 400 channels)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 2: Model registry

**Files:**
- Create: `notebooks/material_atlas/atlaslib/registry.py`
- Modify: `notebooks/material_atlas/atlaslib/__init__.py`
- Test: `tests/atlas/test_registry.py`

**Interfaces:**
- Produces: `RibbonModel(material, edge, width, t_ev, sites_per_cell, band_top_t, n_cells=100, impurity_v_t=0.5, source="")` with `.model_id` (`"material/edge/N{width}"`), `.n_sites`, `.impurities_for_density(d) -> int`; `Registry(models=())` with `add, get, ids, __iter__, __len__, save(path), Registry.load(path), hierarchy() -> {material: {edge: [widths]}}`. `EDGES = ("armchair", "zigzag", "strip")`.

- [ ] **Step 1: Write the failing tests**

`tests/atlas/test_registry.py`:
```python
import pytest
from atlaslib.registry import Registry, RibbonModel


def agnr(n):
    return RibbonModel("graphene-ideal", "armchair", n, 1.0, 2 * n, 3.0, source="legacy")


def test_model_id_and_sites():
    m = agnr(7)
    assert m.model_id == "graphene-ideal/armchair/N7"
    assert m.n_sites == 1400
    assert m.impurities_for_density(0.01) == 14


def test_invalid_edge_or_width_rejected():
    with pytest.raises(ValueError):
        RibbonModel("x", "diagonal", 7, 1.0, 14, 3.0)
    with pytest.raises(ValueError):
        RibbonModel("x", "armchair", 0, 1.0, 14, 3.0)


def test_duplicate_refused_and_json_round_trip(tmp_path):
    r = Registry([agnr(7), agnr(9)])
    with pytest.raises(ValueError):
        r.add(agnr(7))
    r.save(tmp_path / "reg.json")
    r2 = Registry.load(tmp_path / "reg.json")
    assert r2.ids() == r.ids() and r2.get("graphene-ideal/armchair/N9") == agnr(9)


def test_hierarchy_groups_widths():
    r = Registry([agnr(9), agnr(7), RibbonModel("square", "strip", 10, 1.0, 10, 3.92)])
    assert r.hierarchy() == {"graphene-ideal": {"armchair": [7, 9]}, "square": {"strip": [10]}}
```

- [ ] **Step 2: Run to verify failure**

Run: `~/miniconda3/envs/ml/bin/python -m pytest tests/atlas/test_registry.py -v`
Expected: FAIL (`ModuleNotFoundError: atlaslib.registry`)

- [ ] **Step 3: Implement**

`notebooks/material_atlas/atlaslib/registry.py`:
```python
"""Catalogue of ribbon models: one entry per (material, edge, width)."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

EDGES = ("armchair", "zigzag", "strip")


@dataclass(frozen=True)
class RibbonModel:
    material: str
    edge: str
    width: int                 # N: atomic rows across the ribbon
    t_ev: float                # largest nearest-neighbour hopping (1.0 for idealised models)
    sites_per_cell: int        # orbitals per unit cell
    band_top_t: float          # top of the clean band, units of t
    n_cells: int = 100
    impurity_v_t: float = 0.5
    source: str = ""

    def __post_init__(self):
        if self.edge not in EDGES:
            raise ValueError(f"edge must be one of {EDGES}, got {self.edge!r}")
        if self.width < 1 or self.sites_per_cell < 1 or self.n_cells < 1:
            raise ValueError("width, sites_per_cell and n_cells must be positive")

    @property
    def model_id(self) -> str:
        return f"{self.material}/{self.edge}/N{self.width}"

    @property
    def n_sites(self) -> int:
        return self.n_cells * self.sites_per_cell

    def impurities_for_density(self, density: float) -> int:
        return max(1, int(round(density * self.n_sites)))


class Registry:
    def __init__(self, models=()):
        self._models = {}
        for m in models:
            self.add(m)

    def add(self, model: RibbonModel):
        if model.model_id in self._models:
            raise ValueError(f"duplicate model {model.model_id}")
        self._models[model.model_id] = model

    def get(self, model_id: str) -> RibbonModel:
        return self._models[model_id]

    def ids(self):
        return sorted(self._models)

    def __iter__(self):
        return iter(self._models[i] for i in self.ids())

    def __len__(self):
        return len(self._models)

    def save(self, path):
        Path(path).write_text(json.dumps([asdict(m) for m in self], indent=2))

    @classmethod
    def load(cls, path):
        return cls(RibbonModel(**d) for d in json.loads(Path(path).read_text()))

    def hierarchy(self):
        out = {}
        for m in self:
            out.setdefault(m.material, {}).setdefault(m.edge, []).append(m.width)
        return {mat: {e: sorted(w) for e, w in edges.items()} for mat, edges in out.items()}
```
Append to `atlaslib/__init__.py`:
```python
from .registry import EDGES, Registry, RibbonModel  # noqa: F401
```

- [ ] **Step 4: Run to verify pass**

Run: `~/miniconda3/envs/ml/bin/python -m pytest tests/atlas/test_registry.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add notebooks/material_atlas/atlaslib/registry.py notebooks/material_atlas/atlaslib/__init__.py tests/atlas/test_registry.py
git commit -m "feat(atlas): ribbon model registry with material/edge/width hierarchy

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 3: Append-only cloud store with validation

**Files:**
- Create: `notebooks/material_atlas/atlaslib/store.py`
- Modify: `notebooks/material_atlas/atlaslib/__init__.py`
- Test: `tests/atlas/test_store.py`

**Interfaces:**
- Consumes: model ids from Task 2.
- Produces: `CloudStore(root)` with `write_pristine(model_id, energies_t, T)`, `read_pristine(model_id) -> (energies_t, T)`, `write_cloud(model_id, density, n_impurities, spectra, seeds, energies_t)`, `read_cloud(model_id, density) -> (spectra, seeds)`, `has_cloud(model_id, density) -> bool`, `densities(model_id) -> list[float]`, `models() -> list[str]`, `spike_fraction(model_id, density) -> float`. Layout: `root/<model_id>/{energies_t.npy, pristine.npy, cloud_d0.0100.npy, cloud_d0.0100_seeds.npy, meta.json}`.

- [ ] **Step 1: Write the failing tests**

`tests/atlas/test_store.py`:
```python
import numpy as np
import pytest
from atlaslib.store import CloudStore

MID = "graphene-ideal/armchair/N7"
E = np.arange(10) * 0.1


def test_round_trip_and_listing(tmp_path, rng):
    s = CloudStore(tmp_path)
    s.write_pristine(MID, E, np.ones(10) * 2)
    spectra = rng.random((5, 10))
    s.write_cloud(MID, 0.01, 14, spectra, np.arange(5), E)
    got, seeds = s.read_cloud(MID, 0.01)
    assert np.array_equal(got, spectra) and list(seeds) == [0, 1, 2, 3, 4]
    assert s.models() == [MID] and s.densities(MID) == [0.01] and s.has_cloud(MID, 0.01)


def test_energy_grid_mismatch_refused(tmp_path, rng):
    s = CloudStore(tmp_path)
    s.write_pristine(MID, E, np.ones(10))
    with pytest.raises(ValueError, match="energ"):
        s.write_cloud(MID, 0.01, 14, rng.random((3, 10)), np.arange(3), E * 2)


def test_non_finite_refused(tmp_path, rng):
    s = CloudStore(tmp_path)
    bad = rng.random((3, 10)); bad[1, 4] = np.nan
    with pytest.raises(ValueError, match="finite"):
        s.write_cloud(MID, 0.01, 14, bad, np.arange(3), E)


def test_identical_spectra_across_densities_refused(tmp_path, rng):
    """Guards against the square-lattice cache bug: same spectrum at two densities."""
    s = CloudStore(tmp_path)
    a = rng.random((4, 10))
    s.write_cloud(MID, 0.01, 14, a, np.arange(4), E)
    b = rng.random((4, 10)); b[2] = a[1]
    with pytest.raises(ValueError, match="identical"):
        s.write_cloud(MID, 0.02, 28, b, np.arange(4), E)


def test_duplicate_seeds_refused(tmp_path, rng):
    s = CloudStore(tmp_path)
    with pytest.raises(ValueError, match="seed"):
        s.write_cloud(MID, 0.01, 14, rng.random((3, 10)), np.array([0, 1, 1]), E)


def test_spike_fraction(tmp_path):
    s = CloudStore(tmp_path)
    s.write_pristine(MID, E, np.ones(10))
    c = np.ones((2, 10)) * 0.5; c[0, :3] = 5.0          # 3 of 20 points above pristine
    s.write_cloud(MID, 0.01, 14, c, np.arange(2), E)
    assert s.spike_fraction(MID, 0.01) == pytest.approx(3 / 20)
```

- [ ] **Step 2: Run to verify failure**

Run: `~/miniconda3/envs/ml/bin/python -m pytest tests/atlas/test_store.py -v`
Expected: FAIL (`ModuleNotFoundError: atlaslib.store`)

- [ ] **Step 3: Implement**

`notebooks/material_atlas/atlaslib/store.py`:
```python
"""Append-only on-disk store of spectra per ribbon model, with the validity checks
that caught this project's data bugs (non-finite values, cross-density duplicates)."""
import hashlib
import json
from pathlib import Path

import numpy as np


def _key(density: float) -> str:
    return f"d{density:.4f}"


class CloudStore:
    def __init__(self, root):
        self.root = Path(root).expanduser()

    def _dir(self, model_id):
        return self.root / model_id

    def _meta(self, model_id):
        p = self._dir(model_id) / "meta.json"
        return json.loads(p.read_text()) if p.exists() else {"clouds": {}}

    def _save_meta(self, model_id, meta):
        (self._dir(model_id) / "meta.json").write_text(json.dumps(meta, indent=2))

    def _check_energies(self, model_id, energies_t):
        d = self._dir(model_id)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "energies_t.npy"
        energies_t = np.asarray(energies_t, dtype=np.float64)
        if p.exists():
            if not np.allclose(np.load(p), energies_t):
                raise ValueError(f"energy grid differs from the one stored for {model_id}")
        else:
            np.save(p, energies_t)

    def write_pristine(self, model_id, energies_t, T):
        self._check_energies(model_id, energies_t)
        T = np.asarray(T, dtype=np.float64)
        if not np.all(np.isfinite(T)):
            raise ValueError("pristine spectrum is not finite")
        np.save(self._dir(model_id) / "pristine.npy", T)

    def read_pristine(self, model_id):
        d = self._dir(model_id)
        return np.load(d / "energies_t.npy"), np.load(d / "pristine.npy")

    def write_cloud(self, model_id, density, n_impurities, spectra, seeds, energies_t):
        spectra = np.asarray(spectra, dtype=np.float64)
        seeds = np.asarray(seeds, dtype=np.int64)
        if spectra.ndim != 2 or spectra.shape[0] != seeds.size:
            raise ValueError("spectra must be (n_seeds, n_energies) with one seed per row")
        if not np.all(np.isfinite(spectra)):
            raise ValueError("spectra contain non-finite values")
        if np.unique(seeds).size != seeds.size:
            raise ValueError("duplicate seed in cloud")
        self._check_energies(model_id, energies_t)
        new = {hashlib.md5(r.tobytes()).hexdigest() for r in spectra}
        for other in self.densities(model_id):
            if abs(other - density) < 1e-12:
                continue
            old, _ = self.read_cloud(model_id, other)
            if new & {hashlib.md5(r.tobytes()).hexdigest() for r in old}:
                raise ValueError(f"identical spectra at densities {density} and {other}: generator bug?")
        d, k = self._dir(model_id), _key(density)
        np.save(d / f"cloud_{k}.npy", spectra)
        np.save(d / f"cloud_{k}_seeds.npy", seeds)
        meta = self._meta(model_id)
        meta["clouds"][k] = {"density": float(density), "n_impurities": int(n_impurities), "n": int(seeds.size)}
        self._save_meta(model_id, meta)

    def read_cloud(self, model_id, density):
        d, k = self._dir(model_id), _key(density)
        return np.load(d / f"cloud_{k}.npy"), np.load(d / f"cloud_{k}_seeds.npy")

    def has_cloud(self, model_id, density) -> bool:
        return (self._dir(model_id) / f"cloud_{_key(density)}.npy").exists()

    def densities(self, model_id):
        return sorted(v["density"] for v in self._meta(model_id)["clouds"].values())

    def models(self):
        return sorted(str(p.parent.relative_to(self.root)) for p in self.root.rglob("energies_t.npy"))

    def spike_fraction(self, model_id, density) -> float:
        _, pris = self.read_pristine(model_id)
        c, _ = self.read_cloud(model_id, density)
        return float(np.mean(c > pris[None, :] + 1e-6))
```
Append to `atlaslib/__init__.py`:
```python
from .store import CloudStore  # noqa: F401
```

- [ ] **Step 4: Run to verify pass**

Run: `~/miniconda3/envs/ml/bin/python -m pytest tests/atlas/test_store.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add notebooks/material_atlas/atlaslib/store.py notebooks/material_atlas/atlaslib/__init__.py tests/atlas/test_store.py
git commit -m "feat(atlas): append-only cloud store with finite, seed and duplicate checks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 4: Legacy importers (7/9-AGNR and square-10 at atlas scale)

**Files:**
- Create: `notebooks/material_atlas/atlaslib/importers.py`
- Test: `tests/atlas/test_importers.py`

**Interfaces:**
- Consumes: `RibbonModel`, `CloudStore`.
- Produces: `nearest_count(target, available) -> int` (ties → lower); `import_consolidated_agnr(store, model, npy_path, pristine_path, densities, seeds) -> dict[density_target -> actual_density]`; `import_square_combined(store, model, combined_dir, pristine_path, densities, seeds) -> dict`.

- [ ] **Step 1: Write the failing tests**

`tests/atlas/test_importers.py`:
```python
import numpy as np
from atlaslib.importers import import_consolidated_agnr, import_square_combined, nearest_count
from atlaslib.registry import RibbonModel
from atlaslib.store import CloudStore


def test_nearest_count_prefers_lower_on_tie():
    assert nearest_count(7, [2, 4, 6, 8]) == 6
    assert nearest_count(15, [2, 4, 6, 14, 16]) == 14


def test_agnr_import_records_actual_density(tmp_path, rng):
    arr = rng.random((34, 50, 300))                       # c = 2, 4 ... 68
    np.save(tmp_path / "size_7.npy", arr)
    np.save(tmp_path / "pris.npy", np.ones(300))
    m = RibbonModel("graphene-ideal", "armchair", 7, 1.0, 14, 3.0)
    s = CloudStore(tmp_path / "store")
    got = import_consolidated_agnr(s, m, tmp_path / "size_7.npy", tmp_path / "pris.npy", [0.005, 0.01], np.arange(20))
    assert got[0.005] == 6 / 1400 and got[0.01] == 14 / 1400
    spectra, seeds = s.read_cloud(m.model_id, 6 / 1400)
    assert np.array_equal(spectra, arr[2, :20]) and list(seeds) == list(range(20))


def test_square_import_uses_meta_seeds(tmp_path, rng):
    d = tmp_path / "sq"; d.mkdir()
    for c in (5, 10, 20, 40):
        np.save(d / f"conc_{c}.npy", rng.random((30, 400)))
        np.savetxt(d / f"conc_{c}_meta.csv", np.column_stack([np.arange(30), np.arange(30)[::-1]]),
                   delimiter=",", header="row_idx,config", comments="", fmt="%d")
    np.save(tmp_path / "p.npy", np.ones(400))
    m = RibbonModel("square", "strip", 10, 1.0, 10, 3.92)
    s = CloudStore(tmp_path / "store")
    import_square_combined(s, m, d, tmp_path / "p.npy", [0.01], np.arange(5))
    _, seeds = s.read_cloud(m.model_id, 0.01)
    assert sorted(seeds) == [0, 1, 2, 3, 4]
```

- [ ] **Step 2: Run to verify failure**

Run: `~/miniconda3/envs/ml/bin/python -m pytest tests/atlas/test_importers.py -v`
Expected: FAIL (`ModuleNotFoundError: atlaslib.importers`)

- [ ] **Step 3: Implement**

`notebooks/material_atlas/atlaslib/importers.py`:
```python
"""Bring legacy spectra into a CloudStore at atlas scale. Legacy grids use impurity
counts, so each target density maps to the nearest available count and the actual
density is recorded (never the target)."""
from pathlib import Path

import numpy as np


def nearest_count(target, available):
    available = np.asarray(sorted(available))
    return int(available[np.argmin(np.abs(available - target))])   # argmin picks the lower on ties


def import_consolidated_agnr(store, model, npy_path, pristine_path, densities, seeds):
    mm = np.load(npy_path, mmap_mode="r")                      # (n_conc, n_seeds, n_energies), c = 2(i+1)
    counts = 2 * np.arange(1, mm.shape[0] + 1)
    e_t = np.arange(mm.shape[2]) * 0.01
    store.write_pristine(model.model_id, e_t, np.load(pristine_path)[: mm.shape[2]])
    seeds = np.asarray(seeds)
    out = {}
    for d in densities:
        c = nearest_count(d * model.n_sites, counts)
        actual = c / model.n_sites
        store.write_cloud(model.model_id, actual, c, np.asarray(mm[c // 2 - 1, seeds]), seeds, e_t)
        out[d] = actual
    return out


def import_square_combined(store, model, combined_dir, pristine_path, densities, seeds):
    combined_dir = Path(combined_dir).expanduser()
    counts = sorted(int(p.stem.split("_")[1]) for p in combined_dir.glob("conc_*.npy"))
    e_t = np.arange(400) * 0.01
    store.write_pristine(model.model_id, e_t, np.load(Path(pristine_path).expanduser())[:400])
    seeds = np.asarray(seeds)
    out = {}
    for d in densities:
        c = nearest_count(d * model.n_sites, counts)
        rows = np.load(combined_dir / f"conc_{c}.npy", mmap_mode="r")
        cfg = np.loadtxt(combined_dir / f"conc_{c}_meta.csv", delimiter=",", skiprows=1, dtype=int)[:, 1]
        pick = [int(np.where(cfg == s)[0][0]) for s in seeds]
        actual = c / model.n_sites
        store.write_cloud(model.model_id, actual, c, np.asarray(rows[pick, :400]), seeds, e_t)
        out[d] = actual
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `~/miniconda3/envs/ml/bin/python -m pytest tests/atlas/test_importers.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add notebooks/material_atlas/atlaslib/importers.py tests/atlas/test_importers.py
git commit -m "feat(atlas): legacy 7/9-AGNR and square importers recording actual densities

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 5: Encoder and the Atlas (build, locate, save, load)

**Files:**
- Create: `notebooks/material_atlas/atlaslib/encoder.py`, `notebooks/material_atlas/atlaslib/atlas.py`, `tests/atlas/toy.py`
- Modify: `notebooks/material_atlas/atlaslib/__init__.py`
- Test: `tests/atlas/test_atlas.py`

**Interfaces:**
- Consumes: `InputSpec`, `Registry`, `RibbonModel`, `CloudStore`.
- Produces: `train_autoencoder(X_tr, X_va, latent, epochs, patience, lr, batch_size, threads, seed) -> (model, history)`, `embed(model, X) -> (Z, recon_err)`; `Located(material, edge, width, width_extrapolated, density, confidence, novelty, unknown, recon_error)`; `Atlas.build(store, registry, model_ids, spec, latent=32, epochs=60, patience=8, k=15, refs_per_model=2000, seed=0, threads=4) -> Atlas`; `Atlas.locate(T, e_t, band_top_t=None) -> list[Located]`; `Atlas.save(path)`; `Atlas.load(path) -> Atlas`.

- [ ] **Step 1: Toy data helper**

`tests/atlas/toy.py`:
```python
"""Tiny synthetic 'materials': staircases whose step count grows with width."""
import numpy as np

from atlaslib.registry import RibbonModel
from atlaslib.store import CloudStore

E = np.arange(400) * 0.01


def toy_spectrum(level, width, density, seed):
    r = np.random.default_rng(seed)
    T = level * (1 + np.floor(E * width / 3.0)) * np.exp(-density * 25 * (1 + 0.2 * r.random()))
    T[E > 3.0] = 0.0
    return np.clip(T + r.normal(0, 0.02, E.size), 0, None)


def toy_store(tmp_path, materials=(("alpha", 1.0), ("beta", 3.0)), widths=(7, 9, 14),
              densities=(0.01, 0.04), n_seeds=40):
    store, models = CloudStore(tmp_path / "store"), []
    for name, level in materials:
        for w in widths:
            m = RibbonModel(name, "armchair", w, 1.0, 2 * w, 3.0)
            models.append(m)
            store.write_pristine(m.model_id, E, toy_spectrum(level, w, 0.0, 10**6))
            for d in densities:
                seeds = np.arange(n_seeds) + int(d * 1e5)
                store.write_cloud(m.model_id, d, m.impurities_for_density(d),
                                  np.stack([toy_spectrum(level, w, d, s) for s in seeds]), seeds, E)
    return store, models
```

- [ ] **Step 2: Write the failing tests**

`tests/atlas/test_atlas.py`:
```python
import numpy as np
import pytest
from atlaslib import Atlas, InputSpec, Registry
from toy import E, toy_spectrum, toy_store

FAST = dict(latent=8, epochs=6, patience=3, k=7, refs_per_model=200, threads=2)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("atlas")
    store, models = toy_store(tmp)
    reg = Registry(models)
    return Atlas.build(store, reg, reg.ids(), InputSpec(), **FAST), tmp


def test_identifies_material_and_width_of_unseen_seeds(built):
    atlas, _ = built
    T = np.stack([toy_spectrum(3.0, 9, 0.02, 50_000 + i) for i in range(30)])
    res = atlas.locate(T, E, band_top_t=3.0)
    assert np.mean([r.material == "beta" for r in res]) >= 0.9
    assert np.median([r.width for r in res]) == pytest.approx(9, abs=1.0)


def test_width_between_neighbours_for_held_out_width(built):
    atlas, _ = built
    res = atlas.locate(np.stack([toy_spectrum(1.0, 11, 0.01, 60_000 + i) for i in range(30)]), E, 3.0)
    assert 9 <= np.median([r.width for r in res]) <= 14


def test_width_beyond_range_is_flagged(built):
    atlas, _ = built
    res = atlas.locate(np.stack([toy_spectrum(1.0, 40, 0.01, 70_000 + i) for i in range(20)]), E, 3.0)
    assert np.mean([r.width_extrapolated for r in res]) >= 0.8
    assert max(r.width for r in res) <= 14


def test_unfamiliar_spectrum_is_unknown(built):
    atlas, _ = built
    weird = np.tile(np.abs(np.sin(E * 40)) * 8, (5, 1))
    assert all(r.unknown for r in atlas.locate(weird, E, band_top_t=4.0))


def test_save_load_round_trip(built):
    atlas, tmp = built
    atlas.save(tmp / "saved")
    again = Atlas.load(tmp / "saved")
    T = toy_spectrum(1.0, 7, 0.01, 80_000)[None]
    assert atlas.locate(T, E, 3.0) == again.locate(T, E, 3.0)
```

- [ ] **Step 3: Run to verify failure**

Run: `~/miniconda3/envs/ml/bin/python -m pytest tests/atlas/test_atlas.py -v`
Expected: FAIL (`ImportError: cannot import name 'Atlas'`)

- [ ] **Step 4: Implement the encoder**

`notebooks/material_atlas/atlaslib/encoder.py`:
```python
"""1D-conv autoencoder (same design as the 9-AGNR/square studies) and helpers."""
import time

import numpy as np
import torch
import torch.nn as nn


class Conv1dAE(nn.Module):
    def __init__(self, latent: int, seq_len: int):
        super().__init__()
        self.seq_len, self.enc_len = seq_len, seq_len // 8
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 7, 2, 3), nn.GroupNorm(1, 32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, 2, 2), nn.GroupNorm(1, 64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, 2, 1), nn.GroupNorm(1, 128), nn.ReLU())
        self.to_latent = nn.Linear(128 * self.enc_len, latent)
        self.from_latent = nn.Linear(latent, 128 * self.enc_len)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 3, 2, 1, 1), nn.GroupNorm(1, 64), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 5, 2, 2, 1), nn.GroupNorm(1, 32), nn.ReLU(),
            nn.ConvTranspose1d(32, 1, 7, 2, 3, 1))

    def forward(self, x):
        z = self.to_latent(self.encoder(x).flatten(1))
        return self.decoder(self.from_latent(z).view(-1, 128, self.enc_len)), z


def _t(X):
    return torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).unsqueeze(1)


def train_autoencoder(X_tr, X_va, latent=32, epochs=60, patience=8, lr=1e-3, batch_size=256, threads=4, seed=0):
    """X length must be a multiple of 8 (400 channels satisfies this)."""
    torch.set_num_threads(threads); torch.manual_seed(seed)
    model = Conv1dAE(latent, X_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(_t(X_tr)), batch_size=batch_size, shuffle=True)
    xv = _t(X_va)
    best, best_state, wait, hist = np.inf, None, 0, []
    for ep in range(epochs):
        t0 = time.time(); model.train()
        for (xb,) in loader:
            loss = nn.functional.mse_loss(model(xb)[0], xb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val = float(nn.functional.mse_loss(model(xv)[0], xv))
        hist.append(val); sched.step(val)
        print(f"[atlas] epoch {ep + 1:3d} val mse {val:.6f} {time.time() - t0:.0f}s", flush=True)
        if val < best - 1e-7:
            best, best_state, wait = val, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    return model, hist


@torch.no_grad()
def embed(model, X):
    Z, err = [], []
    for i in range(0, len(X), 4096):
        xb = _t(X[i:i + 4096]); r, z = model(xb)
        Z.append(z.numpy()); err.append(((r - xb) ** 2).mean((1, 2)).numpy())
    return np.concatenate(Z), np.concatenate(err)
```

- [ ] **Step 5: Implement the Atlas**

`notebooks/material_atlas/atlaslib/atlas.py`:
```python
"""The reusable map: embed spectra with a frozen encoder, answer locate() by k-NN."""
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from .encoder import Conv1dAE, embed, train_autoencoder
from .registry import RibbonModel
from .spec import InputSpec


@dataclass(frozen=True)
class Located:
    material: str
    edge: str
    width: float
    width_extrapolated: bool
    density: float
    confidence: float
    novelty: float
    unknown: bool
    recon_error: float


class Atlas:
    def __init__(self, spec, encoder, mu, sd, refs, ref_model, ref_density, models, threshold, k):
        self.spec, self.encoder, self.mu, self.sd = spec, encoder, mu, sd
        self.refs, self.ref_model, self.ref_density = refs, ref_model, ref_density
        self.models, self.threshold, self.k = list(models), float(threshold), int(k)
        self._nn = NearestNeighbors(n_neighbors=self.k).fit(self.refs)

    # ---------- building ----------
    @staticmethod
    def _load_inputs(store, registry, model_ids, spec):
        X, midx, dens, seeds = [], [], [], []
        for i, mid in enumerate(model_ids):
            m = registry.get(mid)
            e_t, pris = store.read_pristine(mid)
            X.append(spec.to_input(pris[None], e_t, m.band_top_t)); midx.append([i]); dens.append([0.0]); seeds.append([-1])
            for d in store.densities(mid):
                c, s = store.read_cloud(mid, d)
                X.append(spec.to_input(c, e_t, m.band_top_t))
                midx.append(np.full(len(s), i)); dens.append(np.full(len(s), d)); seeds.append(s)
        return (np.concatenate(X), np.concatenate(midx), np.concatenate(dens).astype(float), np.concatenate(seeds))

    @classmethod
    def build(cls, store, registry, model_ids, spec, latent=32, epochs=60, patience=8, k=15,
              refs_per_model=2000, seed=0, threads=4):
        X, midx, dens, seeds = cls._load_inputs(store, registry, model_ids, spec)
        val = np.zeros(len(X), bool)
        for i in np.unique(midx):                       # validation = top 15% of seeds per model
            s = seeds[(midx == i) & (seeds >= 0)]
            if s.size:
                val |= (midx == i) & (seeds >= np.quantile(s, 0.85))
        enc, _ = train_autoencoder(X[~val], X[val], latent=latent, epochs=epochs, patience=patience,
                                   threads=threads, seed=seed)
        Z, rec = embed(enc, X)
        mu, sd = Z[~val].mean(0), Z[~val].std(0) + 1e-8
        Zs = (Z - mu) / sd
        rng = np.random.default_rng(seed)
        ref = np.concatenate([rng.permutation(np.where((midx == i) & ~val)[0])[:refs_per_model] for i in np.unique(midx)])
        models = [registry.get(mid) for mid in model_ids]
        atlas = cls(spec, enc, mu, sd, Zs[ref], midx[ref], dens[ref], models, np.inf, k)
        # Unknown = reconstruction error above the 99th percentile of known validation spectra.
        # BUILD-12: reconstruction error separated an unseen material with AUROC 1.00, k-NN distance only 0.95.
        atlas.threshold = float(np.percentile(rec[val], 99))
        return atlas

    # ---------- querying ----------
    def _novelty(self, Zs):
        dist, _ = self._nn.kneighbors(Zs)
        return dist.mean(1)

    def locate(self, T, e_t, band_top_t=None):
        X = self.spec.to_input(T, e_t, band_top_t)
        Z, rec = embed(self.encoder, X)
        Zs = (Z - self.mu) / self.sd
        dist, idx = self._nn.kneighbors(Zs)
        out = []
        for r in range(len(X)):
            nb = self.ref_model[idx[r]]
            groups = {}
            for j, mi in enumerate(nb):
                key = (self.models[mi].material, self.models[mi].edge)
                groups.setdefault(key, []).append(j)
            (mat, edge), members = max(groups.items(), key=lambda kv: len(kv[1]))
            w = np.array([self.models[nb[j]].width for j in members], float)
            wt = 1.0 / (dist[r, members] + 1e-9)
            width = float(np.sum(w * wt) / np.sum(wt))
            trained = sorted(m.width for m in self.models if (m.material, m.edge) == (mat, edge))
            extrap = bool(np.all(w == trained[0]) or np.all(w == trained[-1]))
            nov = float(dist[r].mean())
            out.append(Located(mat, edge, width, extrap, float(np.median(self.ref_density[idx[r][members]])),
                               len(members) / self.k, nov, bool(rec[r] > self.threshold), float(rec[r])))
        return out

    # ---------- persistence ----------
    def save(self, path):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.encoder.state_dict(), "latent": self.encoder.to_latent.out_features,
                    "seq_len": self.encoder.seq_len}, path / "encoder.pt")
        np.savez(path / "refs.npz", refs=self.refs, ref_model=self.ref_model, ref_density=self.ref_density,
                 mu=self.mu, sd=self.sd)
        (path / "manifest.json").write_text(json.dumps({
            "spec": self.spec.as_dict(), "models": [asdict(m) for m in self.models], "threshold": self.threshold,
            "k": self.k, "created": date.today().isoformat()}, indent=2))

    @classmethod
    def load(cls, path):
        path = Path(path)
        man = json.loads((path / "manifest.json").read_text())
        ck = torch.load(path / "encoder.pt", weights_only=True)
        enc = Conv1dAE(ck["latent"], ck["seq_len"]); enc.load_state_dict(ck["state"]); enc.eval()
        r = np.load(path / "refs.npz")
        return cls(InputSpec(**man["spec"]), enc, r["mu"], r["sd"], r["refs"], r["ref_model"], r["ref_density"],
                   [RibbonModel(**m) for m in man["models"]], man["threshold"], man["k"])
```
Append to `atlaslib/__init__.py`:
```python
from .atlas import Atlas, Located  # noqa: F401
```

- [ ] **Step 6: Run to verify pass**

Run: `~/miniconda3/envs/ml/bin/python -m pytest tests/atlas/test_atlas.py -v`
Expected: 5 passed (a few minutes; toy training is small)

- [ ] **Step 7: Commit**

```bash
git add notebooks/material_atlas/atlaslib/encoder.py notebooks/material_atlas/atlaslib/atlas.py notebooks/material_atlas/atlaslib/__init__.py tests/atlas/toy.py tests/atlas/test_atlas.py
git commit -m "feat(atlas): Atlas build/locate/save/load with continuous width and novelty

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 6: Adding materials to an existing map without retraining

**Files:**
- Modify: `notebooks/material_atlas/atlaslib/atlas.py`
- Test: `tests/atlas/test_add_models.py`

**Interfaces:**
- Produces: `Atlas.add_models(store, registry, model_ids) -> dict[model_id -> fraction_flagged_unknown]`. A fraction above 0.5 means the frozen encoder does not represent the new model well and the map should be rebuilt.

- [ ] **Step 1: Write the failing test**

`tests/atlas/test_add_models.py`:
```python
import numpy as np
from atlaslib import Atlas, InputSpec, Registry
from toy import E, toy_spectrum, toy_store

FAST = dict(latent=8, epochs=6, patience=3, k=7, refs_per_model=200, threads=2)


def test_add_models_keeps_old_answers_and_reports_familiarity(tmp_path):
    store, models = toy_store(tmp_path, materials=(("alpha", 1.0),))
    reg = Registry(models)
    atlas = Atlas.build(store, reg, reg.ids(), InputSpec(), **FAST)
    probe = toy_spectrum(1.0, 9, 0.01, 90_000)[None]
    before = atlas.locate(probe, E, 3.0)
    n_refs = len(atlas.refs)

    store2, new = toy_store(tmp_path / "b", materials=(("gamma", 6.0),), widths=(9,))
    for m in new:
        reg.add(m)
        e_t, p = store2.read_pristine(m.model_id); store.write_pristine(m.model_id, e_t, p)
        for d in store2.densities(m.model_id):
            c, s = store2.read_cloud(m.model_id, d); store.write_cloud(m.model_id, d, 1, c, s, e_t)
    report = atlas.add_models(store, reg, [m.model_id for m in new])

    assert set(report) == {"gamma/armchair/N9"} and 0.0 <= report["gamma/armchair/N9"] <= 1.0
    assert len(atlas.refs) > n_refs
    assert atlas.locate(probe, E, 3.0)[0].material == before[0].material
```

- [ ] **Step 2: Run to verify failure**

Run: `~/miniconda3/envs/ml/bin/python -m pytest tests/atlas/test_add_models.py -v`
Expected: FAIL (`AttributeError: 'Atlas' object has no attribute 'add_models'`)

- [ ] **Step 3: Implement** (add to class `Atlas` in `atlas.py`)

```python
    def add_models(self, store, registry, model_ids):
        """Embed new models with the frozen encoder and append them as references."""
        report = {}
        for mid in model_ids:
            X, _, dens, _ = self._load_inputs(store, registry, [mid], self.spec)
            Z, rec = embed(self.encoder, X)
            Zs = (Z - self.mu) / self.sd
            report[mid] = float(np.mean(rec > self.threshold))
            self.models.append(registry.get(mid))
            self.refs = np.vstack([self.refs, Zs])
            self.ref_model = np.concatenate([self.ref_model, np.full(len(Zs), len(self.models) - 1)])
            self.ref_density = np.concatenate([self.ref_density, dens])
            self._nn = NearestNeighbors(n_neighbors=self.k).fit(self.refs)
        return report
```

- [ ] **Step 4: Run to verify pass**

Run: `~/miniconda3/envs/ml/bin/python -m pytest tests/atlas -v`
Expected: all atlas tests pass

- [ ] **Step 5: Commit**

```bash
git add notebooks/material_atlas/atlaslib/atlas.py tests/atlas/test_add_models.py
git commit -m "feat(atlas): add models to a saved map with a frozen encoder

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 7: Calibrated intervals (relative split-conformal)

Errors grow in proportion to c (relative error 4.2–4.8% in every band), so intervals are relative: `[ĉ(1−q), ĉ(1+q)]`.

**Files:**
- Create: `notebooks/material_atlas/atlaslib/conformal.py`
- Test: `tests/atlas/test_conformal.py`

**Interfaces:**
- Produces: `fit_relative(pred, true, alpha=0.1) -> float`, `intervals(pred, q) -> (lo, hi)`, `coverage(lo, hi, y) -> float`.

- [ ] **Step 1: Write the failing test**

`tests/atlas/test_conformal.py`:
```python
import numpy as np
from atlaslib.conformal import coverage, fit_relative, intervals


def test_relative_intervals_reach_nominal_coverage(rng):
    y = rng.uniform(2, 98, 40_000)
    pred = y * (1 + rng.normal(0, 0.05, y.size))
    q = fit_relative(pred[:20_000], y[:20_000], alpha=0.1)
    lo, hi = intervals(pred[20_000:], q)
    assert abs(coverage(lo, hi, y[20_000:]) - 0.90) < 0.02
```

- [ ] **Step 2: Run to verify failure**

Run: `~/miniconda3/envs/ml/bin/python -m pytest tests/atlas/test_conformal.py -v`
Expected: FAIL (`ModuleNotFoundError: atlaslib.conformal`)

- [ ] **Step 3: Implement**

`notebooks/material_atlas/atlaslib/conformal.py`:
```python
"""Relative split-conformal intervals: calibrate on held-out seeds, then cover (1 - alpha)."""
import numpy as np


def fit_relative(pred, true, alpha=0.1):
    scores = np.abs(np.asarray(true) - np.asarray(pred)) / np.maximum(np.asarray(pred), 1e-6)
    n = scores.size
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))


def intervals(pred, q):
    pred = np.asarray(pred)
    return pred * (1 - q), pred * (1 + q)


def coverage(lo, hi, y):
    y = np.asarray(y)
    return float(np.mean((y >= lo) & (y <= hi)))
```

- [ ] **Step 4: Run to verify pass**

Run: `~/miniconda3/envs/ml/bin/python -m pytest tests/atlas/test_conformal.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add notebooks/material_atlas/atlaslib/conformal.py tests/atlas/test_conformal.py
git commit -m "feat(atlas): relative split-conformal intervals for concentration

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 8: 7/9-AGNR reference solution, end to end

Stage 1–2 from the atlas (label-free, 0–4 t); stage 3 = one XGBoost per width on the full 0–3 t spectrum normalised by the **predicted** width's pristine; relative conformal intervals calibrated on held-out seeds.

**Files:**
- Create: `notebooks/material_atlas/run_reference_7_9.py`

**Interfaces:**
- Consumes: `InputSpec`, `Registry`, `RibbonModel`, `CloudStore`, `import_consolidated_agnr`, `Atlas`, `fit_relative`, `intervals`, `coverage`.
- Produces: `notebooks/material_atlas/reference_7_9/metrics.json` with keys `width_accuracy`, `end_to_end_mae`, `mae_by_width`, `coverage_90`, `criteria_met`; saved atlas in `reference_7_9/atlas/`.

- [ ] **Step 1: Write the script**

`notebooks/material_atlas/run_reference_7_9.py`:
```python
#!/usr/bin/env python
"""7/9-AGNR reference solution: label-free identification -> stage-3 concentration -> intervals.
Seeds: atlas clouds use seeds 0-999 (4 densities); stage 3 trains on seeds 0-2099 of every
concentration, calibrates on 2100-2549, and is tested end to end on 2550-2999."""
import argparse
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from atlaslib import Atlas, CloudStore, InputSpec, Registry, RibbonModel
from atlaslib.conformal import coverage, fit_relative, intervals
from atlaslib.importers import import_consolidated_agnr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = Path("/run/media/shardul/storage/machine_learning/transmission_data/transmission_results/consolidated_data")
WIDTHS = {7: "size_7.npy", 9: "size_9.npy"}
DENSITIES = [0.005, 0.01, 0.02, 0.04]


def norm(T, pris):
    p = np.round(pris, 3)
    return np.clip(np.round(T, 3) / np.where(p > 0, p, 1.0), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="~/atlas_store/reference_v1")
    ap.add_argument("--out", default=str(HERE / "reference_7_9"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    a = ap.parse_args()
    n_atlas, n_tr, n_cal, n_te = (60, 120, 150, 180) if a.quick else (1000, 2100, 2550, 3000)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    reg = Registry([RibbonModel("graphene-ideal", "armchair", w, 1.0, 2 * w, 3.0, source="consolidated_data") for w in WIDTHS])
    store = CloudStore(a.store)
    for m in reg:
        if not store.densities(m.model_id):
            import_consolidated_agnr(store, m, DATA / WIDTHS[m.width], REPO / f"{m.width}_agnr_pris.npy", DENSITIES, np.arange(n_atlas))
    atlas = Atlas.build(store, reg, reg.ids(), InputSpec(), threads=a.threads, epochs=6 if a.quick else 60)
    atlas.save(out / "atlas")

    e_t = np.arange(300) * 0.01
    Xs, Ws, Cs = {}, {}, {}
    for w, f in WIDTHS.items():                       # every concentration, not only the atlas densities
        mm = np.load(DATA / f, mmap_mode="r")
        conc = 2 * np.arange(1, mm.shape[0] + 1)
        for split, sl in (("tr", slice(0, n_tr)), ("cal", slice(n_tr, n_cal)), ("te", slice(n_cal, n_te))):
            T = np.asarray(mm[:, sl, :300]).reshape(-1, 300)
            Xs.setdefault(split, []).append(T); Ws.setdefault(split, []).append(np.full(len(T), w))
            Cs.setdefault(split, []).append(np.repeat(conc, sl.stop - sl.start))
    T_ = {k: np.concatenate(v) for k, v in Xs.items()}
    W_ = {k: np.concatenate(v) for k, v in Ws.items()}
    C_ = {k: np.concatenate(v).astype(float) for k, v in Cs.items()}
    pris = {w: np.load(REPO / f"{w}_agnr_pris.npy")[:300] for w in WIDTHS}

    regs = {w: xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.04, subsample=0.8,
                                colsample_bytree=0.8, tree_method="hist", random_state=42, n_jobs=a.threads)
                .fit(norm(T_["tr"][W_["tr"] == w], pris[w]), C_["tr"][W_["tr"] == w]) for w in WIDTHS}

    def predict(split):
        loc = atlas.locate(T_[split], e_t, band_top_t=3.0)
        w_hat = np.array([7 if abs(r.width - 7) < abs(r.width - 9) else 9 for r in loc])
        c_hat = np.empty(len(w_hat))
        for w in WIDTHS:
            m = w_hat == w
            if m.any():
                c_hat[m] = regs[w].predict(norm(T_[split][m], pris[w]))
        return w_hat, c_hat

    w_cal, c_cal = predict("cal")
    q = fit_relative(c_cal, C_["cal"], alpha=0.1)
    w_te, c_te = predict("te")
    lo, hi = intervals(c_te, q)
    res = {
        "width_accuracy": float(np.mean(w_te == W_["te"]) * 100),
        "end_to_end_mae": float(np.mean(np.abs(c_te - C_["te"]))),
        "mae_by_width": {str(w): float(np.mean(np.abs(c_te - C_["te"])[W_["te"] == w])) for w in WIDTHS},
        "coverage_90": float(coverage(lo, hi, C_["te"]) * 100),
        "interval_relative_halfwidth": q,
        "n_test": int(len(c_te)),
    }
    res["criteria_met"] = {"width_accuracy>=99.5": res["width_accuracy"] >= 99.5,
                           "mae<=1.98": res["end_to_end_mae"] <= 1.98,
                           "coverage_90+-2": abs(res["coverage_90"] - 90) <= 2}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run**

Run (repo root): `PYTHONPATH=notebooks/material_atlas:notebooks ~/miniconda3/envs/ml/bin/python notebooks/material_atlas/run_reference_7_9.py --quick --store /tmp/atlas_quick --out /tmp/ref_quick`
Expected: prints a metrics JSON with all keys; no exception.

- [ ] **Step 3: Full run (background, ~1.5 h)**

Run (repo root): `PYTHONPATH=notebooks/material_atlas:notebooks ~/miniconda3/envs/ml/bin/python -u notebooks/material_atlas/run_reference_7_9.py --threads 4 > notebooks/material_atlas/reference_7_9.log 2>&1`
Expected: `reference_7_9/metrics.json` written. If a criterion fails, record by how much in the LOGBOOK and do **not** proceed to Phase 5 until the user decides (the width expansion reuses this map).

- [ ] **Step 4: Commit**

```bash
git add notebooks/material_atlas/run_reference_7_9.py notebooks/material_atlas/reference_7_9/metrics.json
git commit -m "results: 7/9-AGNR reference solution (label-free atlas + stage 3 + intervals)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 9: Phase 1 write-up

- [ ] **Step 1:** LOGBOOK: add BUILD-13 row (reference solution, metrics from `reference_7_9/metrics.json`) and close BUILD-12 with the running atlas's results.
- [ ] **Step 2:** Report doc: add a "Material atlas" tab (docs connector) with the reference metrics table and the atlas map image; update the summary's identification claim to the label-free number.
- [ ] **Step 3:** Commit LOGBOOK and push the branch.

---

# Phase 2 — Generic tight-binding ribbon engine (validated)

**Exit:** pristine transmission equals the open-channel count from the band structure for honeycomb armchair/zigzag and the square strip; the `legacy_trace` formula reproduces `ca_sq.device` exactly; engine pristine for idealised 7-AGNR matches `7_agnr_pris.npy`.

### Task 10: Lattices and bands

**Files:**
- Create: `notebooks/tbribbon/__init__.py`, `notebooks/tbribbon/lattices.py`, `notebooks/tbribbon/bands.py`
- Test: `tests/tbribbon/test_lattices.py`

**Interfaces:**
- Produces: `RibbonHamiltonian(H0, H1, positions, sublattice)` (NamedTuple; `H_{n,n+1} = H1`); `honeycomb_ribbon(N, edge, t=1.0, onsite_a=0.0, onsite_b=0.0, edge_bond_factor=1.0)`; `square_strip(width, t=1.0)`; `band_edges(H0, H1, nk=801) -> (emin, emax)`; `open_channels(H0, H1, energies, nk=4001) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

`tests/tbribbon/test_lattices.py`:
```python
import numpy as np
import pytest
from tbribbon.bands import band_edges, open_channels
from tbribbon.lattices import honeycomb_ribbon, square_strip


@pytest.mark.parametrize("N,edge", [(7, "armchair"), (9, "armchair"), (6, "zigzag"), (14, "zigzag")])
def test_honeycomb_has_2N_sites_and_is_hermitian(N, edge):
    h = honeycomb_ribbon(N, edge)
    assert h.H0.shape == (2 * N, 2 * N)
    assert np.allclose(h.H0, h.H0.conj().T)
    coord = (np.abs(h.H0) > 0).sum(1) + (np.abs(h.H1) > 0).sum(1) + (np.abs(h.H1) > 0).sum(0)
    assert coord.max() == 3 and coord.min() == 2           # bulk 3 neighbours, edges 2


@pytest.mark.parametrize("N", [7, 9, 14])
def test_armchair_band_top_matches_analytic(N):
    # N-AGNR bands: E = +-t|1 + 2cos(p pi/(N+1)) e^{..}|, so the top is t(1 + 2cos(pi/(N+1))) <= 3t
    top = band_edges(*honeycomb_ribbon(N, "armchair")[:2])[1]
    assert top == pytest.approx(1 + 2 * np.cos(np.pi / (N + 1)), abs=1e-3)


def test_hbn_onsite_opens_gap():
    h = honeycomb_ribbon(9, "armchair", onsite_a=0.8, onsite_b=-0.8)
    assert open_channels(h.H0, h.H1, [0.0])[0] == 0


def test_square_strip_channels_match_analytic():
    h = square_strip(10)
    eps = -2 * np.cos(np.arange(1, 11) * np.pi / 11)          # transverse modes (hopping -t)
    for E in (0.3, 1.1, 2.5, 3.5):
        assert open_channels(h.H0, h.H1, [E])[0] == np.sum(np.abs(E - eps) < 2)
```

- [ ] **Step 2: Run to verify failure**

Run: `~/miniconda3/envs/ml/bin/python -m pytest tests/tbribbon/test_lattices.py -v`
Expected: FAIL (`ModuleNotFoundError: tbribbon`)

- [ ] **Step 3: Implement lattices**

`notebooks/tbribbon/lattices.py`:
```python
"""Ribbon unit cells from geometry: sites of one period, bonds by distance (bond length 1)."""
from typing import NamedTuple

import numpy as np

S = np.sqrt(3) / 2


class RibbonHamiltonian(NamedTuple):
    H0: np.ndarray            # intra-cell
    H1: np.ndarray            # H_{n, n+1}
    positions: np.ndarray     # (n, 2)
    sublattice: np.ndarray    # 0 = A, 1 = B


def _graphene_sites(extent=60):
    a1, a2 = np.array([1.5, S]), np.array([1.5, -S])
    n1, n2 = np.meshgrid(np.arange(-extent, extent), np.arange(-extent, extent))
    base = n1.ravel()[:, None] * a1 + n2.ravel()[:, None] * a2
    return np.vstack([base, base + [1.0, 0.0]]), np.repeat([0, 1], len(base))


def honeycomb_ribbon(N, edge, t=1.0, onsite_a=0.0, onsite_b=0.0, edge_bond_factor=1.0):
    pos, sub = _graphene_sites()
    eps = 1e-6
    if edge == "armchair":                          # periodic along x (period 3), N rows y = k*S
        axis, L = np.array([1.0, 0.0]), 3.0
        keep = (pos[:, 0] > -eps) & (pos[:, 0] < L - eps) & (pos[:, 1] > -eps) & (pos[:, 1] < (N - 1) * S + eps)
    elif edge == "zigzag":                          # periodic along y (period sqrt3), N zigzag chains
        axis, L = np.array([0.0, 1.0]), 2 * S
        xs = np.concatenate([[1 + 1.5 * j, 1.5 + 1.5 * j] for j in range(N)])
        keep = (pos[:, 1] > -eps) & (pos[:, 1] < L - eps) & np.any(np.abs(pos[:, :1] - xs[None]) < eps, axis=1)
    else:
        raise ValueError(f"edge must be armchair or zigzag, got {edge!r}")
    p, s = pos[keep], sub[keep]
    order = np.lexsort((p[:, 0], p[:, 1]))
    p, s = p[order], s[order]
    n = len(p)
    H0 = np.diag(np.where(s == 0, onsite_a, onsite_b)).astype(complex)
    H1 = np.zeros((n, n), complex)
    rows = p[:, 1] if edge == "armchair" else p[:, 0]
    outer = (np.abs(rows - rows.min()) < eps) | (np.abs(rows - rows.max()) < eps)
    for i in range(n):
        for j in range(n):
            d0 = np.linalg.norm(p[i] - p[j])
            if abs(d0 - 1) < eps:
                along = abs(np.dot(p[j] - p[i], axis)) > 1 - eps
                f = edge_bond_factor if (edge == "armchair" and along and outer[i] and outer[j]) else 1.0
                H0[i, j] = -t * f
            if abs(np.linalg.norm(p[j] + L * axis - p[i]) - 1) < eps:
                H1[i, j] = -t
    return RibbonHamiltonian(H0, H1, p, s)


def square_strip(width, t=1.0):
    H0 = np.zeros((width, width), complex)
    i = np.arange(width - 1)
    H0[i, i + 1] = H0[i + 1, i] = -t
    pos = np.column_stack([np.zeros(width), np.arange(width)])
    return RibbonHamiltonian(H0, -t * np.eye(width, dtype=complex), pos, np.zeros(width, int))
```

- [ ] **Step 4: Implement bands**

`notebooks/tbribbon/bands.py`:
```python
import numpy as np


def _bands(H0, H1, nk):
    ks = np.linspace(-np.pi, np.pi, nk)
    return np.array([np.linalg.eigvalsh(H0 + H1 * np.exp(1j * k) + H1.conj().T * np.exp(-1j * k)) for k in ks])


def band_edges(H0, H1, nk=801):
    b = _bands(H0, H1, nk)
    return float(b.min()), float(b.max())


def open_channels(H0, H1, energies, nk=4001):
    """Right-moving channels at each energy = band crossings over the Brillouin zone / 2."""
    b = _bands(H0, H1, nk)
    out = []
    for E in energies:
        s = np.sign(b - E)
        out.append(int(np.sum(s[1:] != s[:-1]) // 2))
    return np.array(out)
```
`notebooks/tbribbon/__init__.py`: empty file.

- [ ] **Step 5: Run to verify pass**

Run: `~/miniconda3/envs/ml/bin/python -m pytest tests/tbribbon/test_lattices.py -v`
Expected: 9 passed

- [ ] **Step 6: Commit**

```bash
git add notebooks/tbribbon/__init__.py notebooks/tbribbon/lattices.py notebooks/tbribbon/bands.py tests/tbribbon/test_lattices.py
git commit -m "feat(tbribbon): honeycomb/square ribbon Hamiltonians and band tools

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 11: Leads, transport, disorder

**Files:**
- Create: `notebooks/tbribbon/leads.py`, `notebooks/tbribbon/transport.py`, `notebooks/tbribbon/disorder.py`
- Test: `tests/tbribbon/test_transport.py`

**Interfaces:**
- Consumes: `RibbonHamiltonian`, `open_channels`, `band_edges`.
- Produces: `surface_gf(E, H0, H1, eta=1e-4, tol=1e-10, max_iter=300)` (surface of a lead extending in the +H1 direction); `LeadCache(H0, H1, energies, eta=1e-4)` with `.gL[i]`, `.gR[i]`; `transmission(E, H0, H1, shifts, gL, gR, eta=1e-3, formula="caroli")`; `spectrum(H0, H1, energies, shifts, leads, eta=1e-3, formula="caroli") -> np.ndarray`; `impurity_shifts(n_cells, sites_per_cell, n_impurities, seed, v) -> np.ndarray (n_cells, sites_per_cell)`.

- [ ] **Step 1: Write the failing tests**

`tests/tbribbon/test_transport.py`:
```python
import os
import sys

import numpy as np
import pytest
from tbribbon.bands import band_edges, open_channels
from tbribbon.disorder import impurity_shifts
from tbribbon.lattices import honeycomb_ribbon, square_strip
from tbribbon.leads import LeadCache
from tbribbon.transport import spectrum

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.parametrize("h", [square_strip(10), honeycomb_ribbon(7, "armchair"), honeycomb_ribbon(6, "zigzag")])
def test_pristine_equals_open_channels_away_from_edges(h):
    E = np.linspace(0.05, 3.95, 40)
    leads = LeadCache(h.H0, h.H1, E)
    T = spectrum(h.H0, h.H1, E, np.zeros((20, h.H0.shape[0])), leads)
    ch = open_channels(h.H0, h.H1, E)
    stable = np.array([len(set(open_channels(h.H0, h.H1, [e - 0.02, e, e + 0.02]))) == 1 for e in E])
    assert np.allclose(T[stable], ch[stable], atol=1e-3)
    assert np.all(T[E > band_edges(h.H0, h.H1)[1] + 0.02] < 1e-6)


def test_impurity_sites_are_nested_prefixes_of_one_seed():
    a = impurity_shifts(100, 10, 20, seed=3, v=0.5)
    b = impurity_shifts(100, 10, 40, seed=3, v=0.5)
    assert np.all(b[a > 0] == 0.5) and (a > 0).sum() == 20 and (b > 0).sum() == 40


def test_legacy_trace_reproduces_ca_sq_exactly():
    sys.path.insert(0, os.path.join(REPO, "notebooks", "square_lattice"))
    import ca_sq
    legacy = np.load(os.path.expanduser("~/transmissions_sq/leads/leads_10.npy"))
    h = square_strip(10)
    E = np.array([0.37, 1.21, 2.84])
    shifts = impurity_shifts(100, 10, 30, seed=7, v=0.5)

    class Legacy:                                   # the legacy code uses the same surface GF on both sides
        gL = gR = [legacy[int(round(e * 100))] for e in E]

    ours = spectrum(h.H0, h.H1, E, shifts, Legacy, formula="legacy_trace")
    ref = [ca_sq.device(e, 1e-3, 1.0, 0.0, 10, 7, 30, leads=legacy) for e in E]
    assert np.allclose(ours, ref, rtol=1e-8)


def test_idealised_7agnr_pristine_matches_legacy_file():
    h = honeycomb_ribbon(7, "armchair")
    E = np.arange(300) * 0.01
    leg = np.load(os.path.join(REPO, "7_agnr_pris.npy"))[:300]
    T = spectrum(h.H0, h.H1, E, np.zeros((20, 14)), LeadCache(h.H0, h.H1, E))
    stable = np.array([len(set(open_channels(h.H0, h.H1, [e - 0.02, e, e + 0.02]))) == 1 for e in E])
    assert np.allclose(T[stable], leg[stable], atol=1e-2)
```

- [ ] **Step 2: Run to verify failure**

Run: `~/miniconda3/envs/ml/bin/python -m pytest tests/tbribbon/test_transport.py -v`
Expected: FAIL (`ModuleNotFoundError: tbribbon.leads`)

- [ ] **Step 3: Implement**

`notebooks/tbribbon/leads.py`:
```python
"""Sancho-Rubio surface Green's functions for semi-infinite leads."""
import numpy as np


def surface_gf(E, H0, H1, eta=1e-4, tol=1e-10, max_iter=300):
    """Surface GF of a lead whose cells continue in the direction coupled by H1."""
    n = H0.shape[0]
    z = (E + 1j * eta) * np.eye(n)
    eps_s, eps, alpha, beta = H0.astype(complex), H0.astype(complex), H1.astype(complex), H1.conj().T.astype(complex)
    for _ in range(max_iter):
        g = np.linalg.inv(z - eps)
        ag, bg = alpha @ g, beta @ g
        eps_s = eps_s + ag @ beta
        eps = eps + ag @ beta + bg @ alpha
        alpha, beta = ag @ alpha, bg @ beta
        if np.abs(alpha).max() < tol:
            break
    return np.linalg.inv(z - eps_s)


class LeadCache:
    """Left lead extends to -inf (couples via H1^dagger), right lead to +inf (via H1)."""
    def __init__(self, H0, H1, energies, eta=1e-4):
        self.gL = [surface_gf(E, H0, H1.conj().T, eta) for E in energies]
        self.gR = [surface_gf(E, H0, H1, eta) for E in energies]
```
`notebooks/tbribbon/disorder.py`:
```python
import numpy as np


def impurity_shifts(n_cells, sites_per_cell, n_impurities, seed, v):
    """Project convention: RandomState(seed).choice(..., replace=False), so larger counts extend
    the same seed's set (nested); site index = cell * sites_per_cell + orbital."""
    idx = np.random.RandomState(seed).choice(n_cells * sites_per_cell, n_impurities, replace=False)
    shifts = np.zeros(n_cells * sites_per_cell)
    shifts[idx] = v
    return shifts.reshape(n_cells, sites_per_cell)
```
`notebooks/tbribbon/transport.py`:
```python
"""Recursive Green's function transmission through a disordered ribbon."""
import numpy as np


def _caroli(z, H0, H1, shifts, gL, gR):
    H1d = H1.conj().T
    SL, SR = H1d @ gL @ H1, H1 @ gR @ H1d
    n_cells = len(shifts)
    g = np.linalg.inv(z - H0 - np.diag(shifts[0]) - SL - (SR if n_cells == 1 else 0))
    G_n1 = g
    for i in range(1, n_cells):
        extra = SR if i == n_cells - 1 else 0
        g = np.linalg.inv(z - H0 - np.diag(shifts[i]) - H1d @ g @ H1 - extra)
        G_n1 = g @ H1d @ G_n1
    GL, GR = 1j * (SL - SL.conj().T), 1j * (SR - SR.conj().T)
    return float(np.real(np.trace(GR @ G_n1 @ GL @ G_n1.conj().T)))


def _legacy_trace(z, H0, H1, shifts, gL, gR):
    """Formula of ca_sq.device / agnr_lib.device_transmission (unbounded; clip downstream)."""
    I = np.eye(H0.shape[0])
    G = gL
    for s in shifts:
        g_d = np.linalg.inv(z - H0 - np.diag(s))
        G = np.linalg.solve(I - g_d @ H1 @ G @ H1, g_d)
    left, right = G, gR
    c_l = np.linalg.solve(I - right @ H1 @ left @ H1, left)
    c_r = np.linalg.solve(I - left @ H1 @ right @ H1, right)
    G_ll, G_rr = c_l - c_l.conj().T, c_r - c_r.conj().T
    G_lr = left @ H1 @ c_r
    Gnon = G_lr - G_lr.conj().T
    return float(np.abs(np.trace(G_ll @ H1 @ G_rr @ H1 - H1 @ Gnon @ H1 @ Gnon)))


FORMULAS = {"caroli": _caroli, "legacy_trace": _legacy_trace}


def transmission(E, H0, H1, shifts, gL, gR, eta=1e-3, formula="caroli"):
    z = (E + 1j * eta) * np.eye(H0.shape[0])
    return FORMULAS[formula](z, H0, H1, shifts, gL, gR)


def spectrum(H0, H1, energies, shifts, leads, eta=1e-3, formula="caroli"):
    return np.array([transmission(E, H0, H1, shifts, leads.gL[i], leads.gR[i], eta, formula)
                     for i, E in enumerate(energies)])
```

- [ ] **Step 4: Run to verify pass**

Run: `~/miniconda3/envs/ml/bin/python -m pytest tests/tbribbon/test_transport.py -v`
Expected: 6 passed. If the legacy cross-check fails only by a sign convention, fix `_legacy_trace` (not the test): it must reproduce `ca_sq.device` to 1e-8.

- [ ] **Step 5: Commit**

```bash
git add notebooks/tbribbon/leads.py notebooks/tbribbon/transport.py notebooks/tbribbon/disorder.py tests/tbribbon/test_transport.py
git commit -m "feat(tbribbon): Sancho-Rubio leads, Caroli and legacy transmission, nested disorder

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

# Phase 3a — Clean fingerprints of the idealised models (step b, first half)

**Exit:** pristine spectra for idealised graphene armchair N ∈ {5…16, 27, 50}, zigzag N ∈ {4…12, 27, 50} and square-10 are in the store and plotted; report tab shows them.

### Task 12: Materials catalogue and fingerprints

**Files:**
- Create: `notebooks/tbribbon/materials.py`, `notebooks/tbribbon/fingerprints.py`
- Test: `tests/tbribbon/test_materials.py`

**Interfaces:**
- Consumes: lattices, bands, leads, transport, `RibbonModel`, `CloudStore`, `InputSpec`.
- Produces: `MATERIALS: dict[str, dict]`; `hamiltonian_for(model) -> RibbonHamiltonian`; `make_model(material, edge, width) -> RibbonModel` (fills `sites_per_cell`, `band_top_t`); `fingerprints.run(store, models, spec, formula)` writes pristine spectra and `fingerprints.png`.

- [ ] **Step 1: Write the failing test**

`tests/tbribbon/test_materials.py`:
```python
import numpy as np
import pytest
from tbribbon.materials import hamiltonian_for, make_model


def test_make_model_fills_geometry_and_band_top():
    m = make_model("graphene-ideal", "armchair", 9)
    assert m.model_id == "graphene-ideal/armchair/N9" and m.sites_per_cell == 18
    assert m.band_top_t == pytest.approx(1 + 2 * np.cos(np.pi / 10), abs=1e-3)
    assert hamiltonian_for(m).H0.shape == (18, 18)


def test_square_uses_strip_edge():
    m = make_model("square", "strip", 10)
    assert m.sites_per_cell == 10 and m.band_top_t < 4.0


def test_unknown_material_rejected():
    with pytest.raises(KeyError):
        make_model("unobtainium", "armchair", 7)
```

- [ ] **Step 2: Run to verify failure**

Run: `~/miniconda3/envs/ml/bin/python -m pytest tests/tbribbon/test_materials.py -v`
Expected: FAIL (`ModuleNotFoundError: tbribbon.materials`)

- [ ] **Step 3: Implement**

`notebooks/tbribbon/materials.py`:
```python
"""Material catalogue. Real materials (D3) are appended here as new entries."""
from atlaslib.registry import RibbonModel

from .bands import band_edges
from .lattices import honeycomb_ribbon, square_strip

MATERIALS = {
    "graphene-ideal": {"builder": "honeycomb", "params": {"t": 1.0}, "t_ev": 1.0, "source": "idealised nearest-neighbour"},
    "square": {"builder": "square", "params": {"t": 1.0}, "t_ev": 1.0, "source": "idealised square strip"},
}


def _build(material, edge, width):
    spec = MATERIALS[material]
    if spec["builder"] == "honeycomb":
        return honeycomb_ribbon(width, edge, **spec["params"])
    if spec["builder"] == "square":
        return square_strip(width, **spec["params"])
    raise KeyError(spec["builder"])


def hamiltonian_for(model):
    return _build(model.material, model.edge, model.width)


def make_model(material, edge, width):
    spec = MATERIALS[material]
    h = _build(material, edge, width)
    return RibbonModel(material, edge, width, spec["t_ev"], h.H0.shape[0],
                       round(band_edges(h.H0, h.H1)[1], 3), source=spec["source"])
```
`notebooks/tbribbon/fingerprints.py`:
```python
#!/usr/bin/env python
"""Clean (pristine) fingerprints of registered models on the shared 0-4 t grid."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from atlaslib import CloudStore, InputSpec
from tbribbon.leads import LeadCache
from tbribbon.materials import hamiltonian_for, make_model
from tbribbon.transport import spectrum


def run(store, models, spec, formula="caroli", out_png=None):
    e_t = spec.energies_t()
    for m in models:
        h = hamiltonian_for(m)
        T = spectrum(h.H0, h.H1, e_t, np.zeros((1, h.H0.shape[0])), LeadCache(h.H0, h.H1, e_t), formula=formula)
        store.write_pristine(m.model_id, e_t, T)
        print(f"{m.model_id}: max T {T.max():.2f}, band top {m.band_top_t} t", flush=True)
    if out_png:
        fig, ax = plt.subplots(figsize=(10, 5))
        for m in models:
            ax.plot(e_t, store.read_pristine(m.model_id)[1], lw=1, label=m.model_id)
        ax.set_xlabel("E (t)"); ax.set_ylabel("T (G0)"); ax.legend(fontsize=6, ncol=3)
        fig.tight_layout(); fig.savefig(out_png, dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="~/atlas_store/engine_v1")
    ap.add_argument("--formula", default="caroli")
    a = ap.parse_args()
    ms = ([make_model("graphene-ideal", "armchair", n) for n in list(range(5, 17)) + [27, 50]]
          + [make_model("graphene-ideal", "zigzag", n) for n in list(range(4, 13)) + [27, 50]]
          + [make_model("square", "strip", 10)])
    run(CloudStore(a.store), ms, InputSpec(), a.formula, out_png="notebooks/tbribbon/fingerprints_ideal.png")
```

- [ ] **Step 4: Run to verify pass**

Run: `~/miniconda3/envs/ml/bin/python -m pytest tests/tbribbon -v`
Expected: all tbribbon tests pass

- [ ] **Step 5: Produce the fingerprints**

Run (repo root): `PYTHONPATH=notebooks/material_atlas:notebooks OMP_NUM_THREADS=4 ~/miniconda3/envs/ml/bin/python notebooks/tbribbon/fingerprints.py`
Expected: one line per model; `fingerprints_ideal.png` written; armchair N ∈ {5, 8, 11, 14} (3p+2) show T > 0 near E = 0, the others a gap.

- [ ] **Step 6: Report tab and commit**

Add the fingerprint chart to the report's "Material atlas" tab (docs connector), then:
```bash
git add notebooks/tbribbon/materials.py notebooks/tbribbon/fingerprints.py tests/tbribbon/test_materials.py notebooks/tbribbon/fingerprints_ideal.png
git commit -m "feat(tbribbon): materials catalogue and clean fingerprints of idealised ribbons

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Phase 3b — Real materials (separate plan)
Written once D3 is settled from the literature lookup: new builders for hBN (honeycomb with on-site ±Δ, reuses `honeycomb_ribbon`), realistic graphene (edge-bond factor and optional second-neighbour hopping), MoS₂ (three-orbital triangular lattice, new builder), phosphorene (puckered four-atom cell, new builder); each validated with the same channel-count test before fingerprints.

---

# Phase 4 — Disorder clouds

**Exit:** a resumable, parallel generator writes validated clouds to the store; idealised clouds for the width expansion are complete.

### Task 13: Cloud generator (needs D1 settled)

**Files:**
- Create: `notebooks/tbribbon/generate_clouds.py`
- Test: `tests/tbribbon/test_generate.py`

**Interfaces:**
- Consumes: `make_model`, `hamiltonian_for`, `LeadCache`, `spectrum`, `impurity_shifts`, `CloudStore`, `InputSpec`.
- Produces: `seeds_for_width(N) -> int` (1000 / 300 / 100); `generate(store, models, densities, spec, n_jobs=20, formula=DEFAULT_FORMULA, seeds=None) -> list[(model_id, density)]` returning what it wrote (skips existing clouds).

- [ ] **Step 1: Write the failing test**

`tests/tbribbon/test_generate.py`:
```python
from atlaslib import CloudStore, InputSpec, RibbonModel
from tbribbon.generate_clouds import generate, seeds_for_width


def test_seed_budget_by_width():
    assert [seeds_for_width(n) for n in (7, 14, 27, 50)] == [1000, 1000, 300, 100]


def test_generate_writes_then_resumes(tmp_path):
    m = RibbonModel("square", "strip", 3, 1.0, 3, 3.41, n_cells=5)
    spec = InputSpec(e_max_t=4.0, step_t=0.5)
    store = CloudStore(tmp_path)
    wrote = generate(store, [m], [0.2], spec, n_jobs=2, seeds=range(4))
    assert wrote == [(m.model_id, 3 / 15)]
    spectra, seeds = store.read_cloud(m.model_id, 3 / 15)
    assert spectra.shape == (4, 8) and list(seeds) == [0, 1, 2, 3]
    assert generate(store, [m], [0.2], spec, n_jobs=2, seeds=range(4)) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `~/miniconda3/envs/ml/bin/python -m pytest tests/tbribbon/test_generate.py -v`
Expected: FAIL (`ModuleNotFoundError: tbribbon.generate_clouds`)

- [ ] **Step 3: Implement**

`notebooks/tbribbon/generate_clouds.py`:
```python
#!/usr/bin/env python
"""Parallel, resumable generation of disorder clouds into a CloudStore.
Set OMP_NUM_THREADS=1 before launching to avoid BLAS oversubscription."""
import argparse
import os
from multiprocessing import Pool

import numpy as np

from atlaslib import CloudStore, InputSpec
from tbribbon.disorder import impurity_shifts
from tbribbon.leads import LeadCache
from tbribbon.materials import hamiltonian_for, make_model
from tbribbon.transport import spectrum

DEFAULT_FORMULA = "caroli"          # D1: change here if the legacy trace formula is chosen
_W = {}


def seeds_for_width(width):
    return 1000 if width <= 14 else (300 if width <= 27 else 100)


def _init(H0, H1, energies, n_cells, spc, n_imp, v, formula):
    _W.update(H0=H0, H1=H1, E=energies, n_cells=n_cells, spc=spc, n_imp=n_imp, v=v, formula=formula,
              leads=LeadCache(H0, H1, energies))


def _one(seed):
    s = impurity_shifts(_W["n_cells"], _W["spc"], _W["n_imp"], seed, _W["v"])
    return spectrum(_W["H0"], _W["H1"], _W["E"], s, _W["leads"], formula=_W["formula"])


def generate(store, models, densities, spec, n_jobs=20, formula=DEFAULT_FORMULA, seeds=None):
    e_t, wrote = spec.energies_t(), []
    for m in models:
        h = hamiltonian_for(m)
        if not store.densities(m.model_id):
            with Pool(1, _init, (h.H0, h.H1, e_t, 1, h.H0.shape[0], 0, 0.0, formula)) as p:
                store.write_pristine(m.model_id, e_t, p.map(_one, [0])[0])
        for d in densities:
            n_imp = m.impurities_for_density(d)
            actual = n_imp / m.n_sites
            if store.has_cloud(m.model_id, actual):
                continue
            sd = np.arange(seeds_for_width(m.width)) if seeds is None else np.asarray(list(seeds))
            with Pool(n_jobs, _init, (h.H0, h.H1, e_t, m.n_cells, h.H0.shape[0], n_imp, m.impurity_v_t, formula)) as p:
                spectra = np.array(p.map(_one, sd, chunksize=4))
            store.write_cloud(m.model_id, actual, n_imp, spectra, sd, e_t)
            wrote.append((m.model_id, actual))
            print(f"{m.model_id} density {actual:.4f}: {len(sd)} spectra", flush=True)
    return wrote


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="~/atlas_store/engine_v1")
    ap.add_argument("--n-jobs", type=int, default=20)
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ms = ([make_model("graphene-ideal", "armchair", n) for n in range(5, 17)]
          + [make_model("graphene-ideal", "zigzag", n) for n in range(4, 13)])
    generate(CloudStore(a.store), ms, [0.005, 0.01, 0.02, 0.04], InputSpec(), n_jobs=a.n_jobs)
```

- [ ] **Step 4: Run to verify pass**

Run: `~/miniconda3/envs/ml/bin/python -m pytest tests/tbribbon/test_generate.py -v`
Expected: 2 passed

- [ ] **Step 5: Launch idealised clouds (background, several hours)**

Run (repo root): `PYTHONPATH=notebooks/material_atlas:notebooks OMP_NUM_THREADS=1 nohup ~/miniconda3/envs/ml/bin/python -u notebooks/tbribbon/generate_clouds.py > notebooks/tbribbon/generate_ideal.log 2>&1 &`
Expected: one line per (model, density); rerunning resumes. Check `CloudStore.spike_fraction` on a few clouds: with `caroli` it must be ~0.

- [ ] **Step 6: Commit**

```bash
git add notebooks/tbribbon/generate_clouds.py tests/tbribbon/test_generate.py
git commit -m "feat(tbribbon): parallel resumable disorder-cloud generator

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

# Phase 5 — Atlas v2 and the generalisation test (step a)

**Exit:** a map built from engine clouds places held-out widths between their trained neighbours; maps in the report tab.

### Task 14: Build the map and run the generalisation test

**Files:**
- Create: `notebooks/material_atlas/build_atlas_v2.py`

**Interfaces:**
- Consumes: `Atlas`, `Registry`, `CloudStore`, `InputSpec`, `make_model`.
- Produces: `notebooks/material_atlas/atlas_v2/` (saved map) and `atlas_v2/generalisation.json` with, per held-out model: `material_accuracy`, `edge_accuracy`, `median_width`, `between_neighbours_pct`.

- [ ] **Step 1: Write the script**

`notebooks/material_atlas/build_atlas_v2.py`:
```python
#!/usr/bin/env python
"""Atlas v2 from engine clouds; generalisation test on held-out widths."""
import argparse
import json
from pathlib import Path

import numpy as np

from atlaslib import Atlas, CloudStore, InputSpec, Registry
from tbribbon.materials import make_model

HELD_OUT = {("graphene-ideal", "armchair"): [8, 12, 13], ("graphene-ideal", "zigzag"): [8]}
TRAIN = {("graphene-ideal", "armchair"): range(5, 17), ("graphene-ideal", "zigzag"): range(4, 13)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="~/atlas_store/engine_v1")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "atlas_v2"))
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    store, spec, out = CloudStore(a.store), InputSpec(), Path(a.out)
    reg = Registry()
    for (mat, edge), widths in TRAIN.items():
        for n in widths:
            reg.add(make_model(mat, edge, n))
    train_ids = [m.model_id for m in reg if m.width not in HELD_OUT[(m.material, m.edge)]]
    atlas = Atlas.build(store, reg, train_ids, spec, threads=a.threads)
    atlas.save(out)
    res = {}
    for (mat, edge), widths in HELD_OUT.items():
        for n in widths:
            mid = f"{mat}/{edge}/N{n}"
            T = np.concatenate([store.read_cloud(mid, d)[0] for d in store.densities(mid)])
            e_t, _ = store.read_pristine(mid)
            loc = atlas.locate(T, e_t, reg.get(mid).band_top_t)
            w = np.array([r.width for r in loc])
            res[mid] = {"material_accuracy": float(np.mean([r.material == mat for r in loc]) * 100),
                        "edge_accuracy": float(np.mean([r.edge == edge for r in loc]) * 100),
                        "median_width": float(np.median(w)),
                        "between_neighbours_pct": float(np.mean((w > n - 1.5) & (w < n + 1.5)) * 100)}
    (out / "generalisation.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run (after Phase 4 clouds exist)**

Run (repo root): `PYTHONPATH=notebooks/material_atlas:notebooks ~/miniconda3/envs/ml/bin/python -u notebooks/material_atlas/build_atlas_v2.py --threads 4 > notebooks/material_atlas/atlas_v2.log 2>&1`
Expected: `generalisation.json` written. Success: edge accuracy ≥ 99% and `between_neighbours_pct` ≥ 90% for every held-out width; record either way.

- [ ] **Step 3: Report and commit**

Add the atlas map and the generalisation table to the report tab; LOGBOOK BUILD-14; then:
```bash
git add notebooks/material_atlas/build_atlas_v2.py notebooks/material_atlas/atlas_v2/generalisation.json
git commit -m "results: atlas v2 from engine clouds and held-out width generalisation

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

# Phase 6 — Interactive atlas page (step c, separate plan)

Written after Phase 5 and D2: a standalone page that loads a saved map's reference embeddings (2-D projection), lets people filter by material/edge/width/density, hover a point to see its spectrum, and paste a spectrum to `locate` it. Reuses `Atlas.save` output unchanged.

---

## Iteration loop (every phase)
1. Settle the open decision the phase needs (D1 before Task 13, D3 before Phase 3b, D2 before Phase 6).
2. TDD the tasks; run the full suite: `~/miniconda3/envs/ml/bin/python -m pytest -q`.
3. Run the phase's experiment in the background with the thread rules above.
4. Record results (LOGBOOK row, report tab), commit, push the branch.
5. Review exit criteria; if missed, stop and decide before the next phase (later phases reuse this map).
