"""Hexagonal parent lattice: one tight-binding host family that contains many 2D materials.

Three site types share the honeycomb cell: C (hexagon centre), H (the A/B honeycomb sites) and K
(the kagome sites on the bond midpoints). A host switches types on or off and sets on-site energies
and hoppings (units of the lead hopping, which is 1); `s` is the energy scale in eV.
"""
from dataclasses import dataclass

TYPES = ("C", "H", "K")
NAMED = ("graphene-ideal", "silicene", "germanene", "hbn", "triangular", "kagome")


@dataclass(frozen=True)
class HexParams:
    on_C: bool
    on_H: bool
    on_K: bool
    lead: str                  # "C", "H" or "K": the type whose within-type hopping is the unit
    s: float                   # energy scale (eV)
    eps_C: float = 0.0
    eps_H_mean: float = 0.0
    delta: float = 0.0         # eps_A = eps_H_mean + delta, eps_B = eps_H_mean - delta
    eps_K: float = 0.0
    t_CC: float = 0.0
    t_AB: float = 0.0
    t_KK: float = 0.0
    t_AA: float = 0.0          # A-A and B-B (H only)
    t_CH: float = 0.0
    t_CK: float = 0.0
    t_HK: float = 0.0

    @property
    def eps_A(self):
        return self.eps_H_mean + self.delta

    @property
    def eps_B(self):
        return self.eps_H_mean - self.delta

    @property
    def pattern(self):
        return (self.on_C, self.on_H, self.on_K)

    @property
    def active_types(self):
        return tuple(t for t, on in zip(TYPES, self.pattern) if on)

    def validate(self):
        """Raise ValueError unless this is a canonical host (spec section 4.2)."""
        active = self.active_types
        if not active:
            raise ValueError("no site type is active")
        if self.lead not in active:
            raise ValueError(f"lead {self.lead!r} is not an active type")
        within = {"C": self.t_CC, "H": self.t_AB, "K": self.t_KK}
        if within[self.lead] != 1:
            raise ValueError(f"lead hopping must be exactly 1, got {within[self.lead]}")
        for t in active:
            if not 0 < within[t] <= 1:
                raise ValueError(f"within-type hopping of {t} must be in (0, 1], got {within[t]}")
        carried = {"C": (self.eps_C, self.t_CC), "H": (self.eps_H_mean, self.delta, self.t_AB, self.t_AA),
                   "K": (self.eps_K, self.t_KK)}
        for t in TYPES:
            if t not in active and any(v != 0 for v in carried[t]):
                raise ValueError(f"inactive type {t} carries non-zero parameters")
        if self.t_AA != 0 and not 0 < self.t_AA <= 1:
            raise ValueError(f"t_AA must be in (0, 1], got {self.t_AA}")
        cross = {("C", "H"): self.t_CH, ("C", "K"): self.t_CK, ("H", "K"): self.t_HK}
        for (a, b), v in cross.items():
            if v != 0 and not (a in active and b in active):
                raise ValueError(f"t_{a}{b} needs both {a} and {b} active")
        if self.delta < 0:
            raise ValueError(f"delta must be >= 0, got {self.delta}")
        if not self.s > 0:
            raise ValueError(f"s must be > 0, got {self.s}")
        bonds = sum(1 for (a, b), v in cross.items() if a in active and b in active and v > 0)
        if len(active) == 2 and bonds < 1:
            raise ValueError("the two active types are not connected")
        if len(active) == 3 and bonds < 2:
            raise ValueError("three active types need at least two cross hoppings")


def named_material(name):
    """The known materials of spec section 4.3 as canonical hosts."""
    if name == "graphene-ideal":
        return HexParams(False, True, False, "H", 2.7, t_AB=1.0)
    if name == "silicene":
        return HexParams(False, True, False, "H", 1.067, t_AB=1.0)
    if name == "germanene":
        return HexParams(False, True, False, "H", 0.991, t_AB=1.0)
    if name == "hbn":
        return HexParams(False, True, False, "H", 2.30, delta=3.625 / 2.30, t_AB=1.0)
    if name == "triangular":
        return HexParams(True, False, False, "C", 1.0, t_CC=1.0)
    if name == "kagome":
        return HexParams(False, False, True, "K", 1.0, t_KK=1.0)
    raise ValueError(f"unknown material {name!r}; choose from {NAMED}")
