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
    orbitals_per_site: int = 1
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
        return self.n_cells * self.sites_per_cell // self.orbitals_per_site

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
