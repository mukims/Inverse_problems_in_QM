"""Versioned, label-free input specification shared by every atlas component."""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class InputSpec:
    version: str = "v1"
    e_max_t: float = 4.0     # window [0, e_max_t) in the spec's unit: the material's own t (v1, v2) or eV (v3)
    step_t: float = 0.01     # channel step, same unit
    cap: float = 20.0        # G0; clipped before the log so spikes cannot dominate
    unit: str = "t"          # "t" or "eV"

    def __post_init__(self):
        if self.version == "v2" and self.cap == 20.0:
            object.__setattr__(self, "cap", 64.0)
        elif self.version == "v3":
            object.__setattr__(self, "unit", "eV")
            object.__setattr__(self, "e_max_t", 8.32)
            object.__setattr__(self, "step_t", 0.02)
            object.__setattr__(self, "cap", 64.0)
        elif self.version not in ("v1", "v2"):
            raise ValueError(f"unknown InputSpec version: {self.version}")

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
        if band_top_t is None:
            inside = np.ones(grid.size, dtype=bool)
            top = grid[-1]
            if e_t[-1] < top - self.step_t / 2:
                raise ValueError(f"data ends at {e_t[-1]:.3f} {self.unit} but window extends to {top:.3f} {self.unit}")
        else:
            top = float(band_top_t)
            inside = grid < top - 1e-9
            required_top = grid[inside][-1] if np.any(inside) else 0.0
            if e_t[-1] < required_top - self.step_t / 2:
                raise ValueError(f"data ends at {e_t[-1]:.3f} {self.unit} but the band extends to {top:.3f} {self.unit}; "
                                 "zero-filling would erase real signal")
        out = np.zeros((T.shape[0], grid.size))
        n = int(inside.sum())
        if n > 0 and e_t.size >= n and np.allclose(e_t[:n], grid[:n]):
            out[:, :n] = T[:, :n]
        else:
            for i in range(T.shape[0]):
                out[i, inside] = np.interp(grid[inside], e_t, T[i])
        out = np.clip(np.round(out, 3), 0.0, self.cap)
        return (np.log1p(out) / np.log1p(self.cap)).astype(np.float32)

    def as_dict(self) -> dict:
        return asdict(self)
