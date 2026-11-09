"""Truth rewards for fine-tuning the inverse model (§4 of
docs/superpowers/specs/2026-11-04-reward-finetuning-design.md).

A cost in [0, 1] per part, combined as cost = 0.6 family + 0.25 width + 0.15 continuous (reward = 1 - cost). Two arms
differ only in the family cost:

* label: 0 if edge, pattern, delta_zero, eps_zero, t_AA_on and cross all match the truth, else 1;
* physics: switch flips priced by their size in eV and dropped site types by their band weight, saturated as
  1 - exp(-sum / sigma_f); a wrong edge costs 1.

The lead (the type whose within-type hopping is the unit) is never part of the material: every comparison goes through
the lead-free physical vector `physical` (energies and hoppings in eV, a switched-off quantity 0 eV). `physical_torch`
computes that vector from the encoding, differentiably, so a cost on flow samples has a gradient.
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch

from shazam.inverse.encoding import CONT, SLOT, STATES, W_BOUNDS, active_slots
from tbribbon import parent_data
from tbribbon.parent import TYPES
from tbribbon.parent_prior import CROSS, EPS, RANGES, WITHIN

WEIGHTS = dict(family=0.6, width=0.25, continuous=0.15)
ENERGY_SCALE_EV = 0.1                     # about five 0.02 eV channels
EV_NAMES = ("eps_C", "eps_H_mean", "delta", "eps_K", "t_CC", "t_AB", "t_KK", "t_AA", "t_CH", "t_CK", "t_HK")
N_PHYS = len(EV_NAMES) + 2                # then log c, then the width
PATTERN_NAMES = ("C", "H", "K", "CH", "CK", "HK", "CHK")
_PAIRS = tuple(CROSS)                     # (C-H, C-K, H-K), the order of DiscreteState.cross
_OWNER = np.zeros((len(EV_NAMES), 3), bool)                # which site types each physical quantity belongs to
for _i, _n in enumerate(EV_NAMES):
    for _j, _t in enumerate(TYPES):
        _OWNER[_i, _j] = _n in (EPS[_t], WITHIN[_t]) or (_t == "H" and _n in ("delta", "t_AA"))
    for _pair, _name in CROSS.items():
        if _n == _name:
            _OWNER[_i, [TYPES.index(t) for t in _pair]] = True


def family_key(state_index):
    """(edge, pattern, delta_zero, eps_zero, t_AA_on, cross) of a state: everything but the lead."""
    s = STATES[state_index]
    return s.edge, s.pattern, s.delta_zero, s.eps_zero, s.t_AA_on, s.cross


_KEYS = {}
FAMILY_OF = np.array([_KEYS.setdefault(family_key(k), len(_KEYS)) for k in range(len(STATES))])
N_FAMILIES = len(_KEYS)
_FAMILY_T = torch.as_tensor(FAMILY_OF)


_FAMILY_ONEHOT = np.eye(N_FAMILIES)[FAMILY_OF]                    # (156, N_FAMILIES)


def family_probs(state_probs):
    """(n, N_FAMILIES) family probabilities: the states' probabilities summed over lead."""
    return np.asarray(state_probs) @ _FAMILY_ONEHOT


def pattern_name(pattern):
    return "".join(t for t, on in zip(TYPES, pattern) if on)


# ---------------------------------------------------------------- the lead-free physical vector

def physical(draw):
    """(N_PHYS,) float64: EV_NAMES in eV (model value times s; switched off = 0), log c, width."""
    p = draw.params
    return np.array([getattr(p, n) * p.s for n in EV_NAMES] + [np.log(draw.density), float(draw.width)])


def _slot_bounds():
    lo, hi = np.zeros(len(CONT)), np.ones(len(CONT))
    for i, name in enumerate(CONT):
        if name == "log_s":
            lo[i], hi[i] = np.log(RANGES["s"])
        elif name == "log_c":
            lo[i], hi[i] = np.log(RANGES["c"])
        elif name == "w":
            lo[i], hi[i] = W_BOUNDS
        elif name in EPS.values():
            lo[i], hi[i] = RANGES["eps"]
        elif name == "delta":
            lo[i], hi[i] = RANGES["delta_min"][0], np.nan          # the top depends on eps_H_mean (delta_upper)
        elif name in WITHIN.values():
            lo[i], hi[i] = RANGES["t_within"]
        elif name == "t_AA":
            lo[i], hi[i] = RANGES["t_AA"]
        else:
            lo[i], hi[i] = RANGES["t_cross"]
    return lo, hi


_LO, _HI = _slot_bounds()
_MASKS = torch.as_tensor(np.stack([active_slots(k) for k in range(len(STATES))]))
_LEAD_SLOT = torch.as_tensor([SLOT[WITHIN[s.lead]] for s in STATES])
_EV_SLOTS = torch.as_tensor([SLOT[n] for n in EV_NAMES])
_STATE_PAT = torch.as_tensor([s.pattern for s in STATES])


def physical_torch(states, z):
    """(n, N_PHYS) physical vectors of encoded samples (states (n,), z (n, 14)), differentiable in z.

    Mirrors encoding.decode slot by slot (Delta's interval tops out at delta_upper(eps_H_mean) and is 0 when empty; the
    lead's within-type hopping is 1), except that the width stays continuous (not rounded) so it has a gradient."""
    states = torch.as_tensor(states)
    m = _MASKS[states]
    lo = torch.as_tensor(_LO, dtype=z.dtype)
    hi = torch.as_tensor(np.nan_to_num(_HI), dtype=z.dtype)
    u = torch.sigmoid(z)
    v = torch.where(m, lo + (hi - lo) * u, torch.zeros_like(z))
    d = SLOT["delta"]
    d_lo = RANGES["delta_min"][0]
    d_hi = RANGES["delta_min"][1] - v[:, SLOT["eps_H_mean"]].abs()
    delta = torch.where(m[:, d] & (d_hi >= d_lo), d_lo + (d_hi - d_lo) * u[:, d], torch.zeros_like(d_hi))
    v = torch.cat([v[:, :d], delta[:, None], v[:, d + 1:]], 1)
    v = v + torch.nn.functional.one_hot(_LEAD_SLOT[states], len(CONT)).to(z.dtype)        # the lead's hopping: 1
    s = torch.exp(v[:, SLOT["log_s"]])
    return torch.cat([v[:, _EV_SLOTS] * s[:, None], v[:, SLOT["log_c"], None], v[:, SLOT["w"], None]], 1)


# ---------------------------------------------------------------- width

def width_cost(dw):
    """The reported width cost: 0 for |dw| <= 1, linear to 1 at |dw| = 4, 1 beyond."""
    if isinstance(dw, torch.Tensor):
        return ((dw.abs() - 1) / 3).clamp(0, 1)
    out = np.clip((np.abs(dw) - 1) / 3, 0, 1)
    return float(out) if np.ndim(out) == 0 else out


def width_surrogate(dw):
    """The training width cost: 0 for |dw| <= 1, then a Huber loss (delta 1) of |dw| - 1, so it never saturates."""
    a = (dw.abs() - 1).clamp_min(0)
    return torch.where(a <= 1, 0.5 * a ** 2, a - 0.5)


# ---------------------------------------------------------------- family costs

@dataclass
class PhysicsScales:
    """Physics-arm scales (§4). sigma_f: a 0.5 eV switch flip costs 1 - exp(-1); weight_scale (eV per unit band weight):
    a dropped type with 10% band weight costs the same; added_type_cost (eV): a type the truth lacks costs the same.
    on_magnitude_ev: the train mean size of each switchable quantity given it is on (used where the prediction has no
    value of its own: the classifier's table). mean_weight: the train mean band weights (C, H, K) per pattern name."""
    on_magnitude_ev: dict
    mean_weight: dict
    sigma_f: float = 0.5
    weight_scale: float = 5.0
    added_type_cost: float = 0.5

    def to_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, d):
        return cls(**d)


def band_weight_row(params):
    """[w_C, w_H, w_K] of parent_data.band_weights (0 for an absent type)."""
    bw = parent_data.band_weights(params)
    return [bw.get(t, 0.0) for t in TYPES]


def _lone_eps(pat, eps):
    """|eps| of the lone type (0 unless exactly one type is active); pat (..., 3) bool, eps (..., 3) values."""
    lone = pat.sum(-1) == 1
    return lone, (eps.abs() * pat).sum(-1) * lone


def truth_arrays(draws, weights):
    """Per-draw descriptors the physics family cost needs (numpy): edge, pattern, switches and their sizes in eV."""
    phys = np.stack([physical(d) for d in draws])
    pat = np.array([d.params.pattern for d in draws], bool)
    ev = {n: phys[:, i] for i, n in enumerate(EV_NAMES)}
    eps = np.stack([ev[EPS[t]] for t in TYPES], 1)
    cross = np.stack([ev[n] for n in CROSS.values()], 1)
    lone = pat.sum(1) == 1
    e = (np.abs(eps) * pat).sum(1) * lone
    return dict(edge=np.array([d.edge == "zigzag" for d in draws]), pat=pat,
                d_on=pat[:, 1] & (ev["delta"] != 0), a_on=pat[:, 1] & (ev["t_AA"] != 0), x_on=cross != 0,
                e_on=lone & (e != 0), d=np.abs(ev["delta"]), a=np.abs(ev["t_AA"]), x=np.abs(cross), e=e,
                w=np.asarray(weights, float), phys=phys)


def _state_arrays(scales):
    """The same descriptors for the 156 states as predictions, sizes from scales.on_magnitude_ev."""
    pat = np.array([s.pattern for s in STATES], bool)
    lone = pat.sum(1) == 1
    om = scales.on_magnitude_ev
    n = len(STATES)
    return dict(edge=np.array([s.edge == "zigzag" for s in STATES]), pat=pat,
                d_on=pat[:, 1] & ~np.array([s.delta_zero for s in STATES]),
                a_on=np.array([s.t_AA_on for s in STATES]), x_on=np.array([s.cross for s in STATES], bool),
                e_on=lone & ~np.array([s.eps_zero for s in STATES]),
                d=np.full(n, om["delta"]), a=np.full(n, om["t_AA"]),
                x=np.tile([om[name] for name in CROSS.values()], (n, 1)), e=np.full(n, om["eps"]))


def _family_cost(T, P, scales):
    """Physics family cost for truths T and predictions P (dicts of tensors that broadcast against each other)."""
    c = 0
    for j in range(3):
        c = c + (T["pat"][..., j] & ~P["pat"][..., j]) * T["w"][..., j] * scales.weight_scale
        c = c + (~T["pat"][..., j] & P["pat"][..., j]) * scales.added_type_cost
    h_both = T["pat"][..., 1] & P["pat"][..., 1]
    for on, mag in (("d_on", "d"), ("a_on", "a")):
        flip = h_both & (T[on] != P[on])
        c = c + flip * torch.where(T[on], T[mag], P[mag])
    for j, (a, b) in enumerate(_PAIRS):
        ia, ib = TYPES.index(a), TYPES.index(b)
        both = T["pat"][..., ia] & T["pat"][..., ib] & P["pat"][..., ia] & P["pat"][..., ib]
        flip = both & (T["x_on"][..., j] != P["x_on"][..., j])
        c = c + flip * torch.where(T["x_on"][..., j], T["x"][..., j], P["x"][..., j])
    t_lone, p_lone = T["pat"].sum(-1) == 1, P["pat"].sum(-1) == 1
    same_lone = t_lone & p_lone & (T["pat"] == P["pat"]).all(-1)
    flip = same_lone & (T["e_on"] != P["e_on"])
    c = c + flip * torch.where(T["e_on"], T["e"], P["e"])
    cost = 1 - torch.exp(-c / scales.sigma_f)
    return torch.where(T["edge"] == P["edge"], cost, torch.ones_like(cost))


def _torch(d, keys, expand=None):
    out = {k: torch.as_tensor(np.asarray(d[k])) for k in keys}
    return {k: (v[expand] if expand is not None else v) for k, v in out.items()}


_KEYS_T = ("edge", "pat", "d_on", "a_on", "x_on", "e_on", "d", "a", "x", "e", "w")
_KEYS_P = ("edge", "pat", "d_on", "a_on", "x_on", "e_on", "d", "a", "x", "e")


def label_family_cost_table(truth_states):
    """(n, 156) float32: 0 where the state's family (lead aside) is the truth's, else 1."""
    return (FAMILY_OF[None, :] != FAMILY_OF[np.asarray(truth_states)][:, None]).astype(np.float32)


def physics_family_cost_table(draws, weights, scales, chunk=4096):
    """(n, 156) float32 physics family cost of predicting each state for each truth draw (weights: (n, 3))."""
    P = _torch(_state_arrays(scales), _KEYS_P, (None,))
    out = []
    for i in range(0, len(draws), chunk):
        T = _torch(truth_arrays(draws[i:i + chunk], weights[i:i + chunk]), _KEYS_T, (slice(None), None))
        out.append(_family_cost(T, P, scales).numpy().astype(np.float32))
    return np.concatenate(out) if out else np.zeros((0, len(STATES)), np.float32)


def physics_family_cost_sample(truth, states, z, scales):
    """(n,) physics family cost of samples (states (n,), z (n, 14)) against `truth` (truth_arrays rows, numpy). A switch
    the truth has off and the sample has on is priced by the sample's own size (no gradient: it counts flips)."""
    states = torch.as_tensor(states)
    with torch.no_grad():
        ph = physical_torch(states, z).abs()
    pat = _STATE_PAT[states]
    s = [STATES[int(k)] for k in states]
    _, e = _lone_eps(pat, ph[:, [EV_NAMES.index(EPS[t]) for t in TYPES]])
    P = dict(edge=torch.as_tensor([x.edge == "zigzag" for x in s]), pat=pat,
             d_on=pat[:, 1] & ~torch.as_tensor([x.delta_zero for x in s]), a_on=torch.as_tensor([x.t_AA_on for x in s]),
             x_on=torch.as_tensor([x.cross for x in s]),
             e_on=(pat.sum(1) == 1) & ~torch.as_tensor([x.eps_zero for x in s]),
             d=ph[:, EV_NAMES.index("delta")], a=ph[:, EV_NAMES.index("t_AA")],
             x=ph[:, [EV_NAMES.index(n) for n in CROSS.values()]], e=e)
    T = _torch(truth, _KEYS_T)
    T = {k: (v.to(ph.dtype) if v.dtype.is_floating_point else v) for k, v in T.items()}
    return _family_cost(T, P, scales)


def state_state_physics_table(scales):
    """(156, 156) float32: the physics family cost of predicting state j when the truth is state i, with every on-switch
    at its mean size and the mean band weights of the truth's pattern (for decoding by minimum expected cost)."""
    S = _state_arrays(scales)
    S["w"] = np.array([scales.mean_weight[pattern_name(s.pattern)] for s in STATES], float)
    T = _torch(S, _KEYS_T, (slice(None), None))
    P = _torch(S, _KEYS_P, (None,))
    out = _family_cost(T, P, scales).numpy().astype(np.float32)
    np.fill_diagonal(out, 0)
    return out


def fit_physics_scales(draws, weights):
    """PhysicsScales from training draws: on-switch mean sizes in eV and mean band weights per pattern (§4)."""
    t = truth_arrays(draws, weights)
    om = dict(delta=float(t["d"][t["d_on"]].mean()), t_AA=float(t["a"][t["a_on"]].mean()),
              eps=float(t["e"][t["e_on"]].mean()))
    for j, name in enumerate(CROSS.values()):
        om[name] = float(t["x"][t["x_on"][:, j], j].mean())
    names = np.array([pattern_name(p) for p in t["pat"]])
    mw = {}
    for p in PATTERN_NAMES:
        sel = names == p
        on = np.array([c in p for c in TYPES])
        mw[p] = (t["w"][sel].mean(0).tolist() if sel.any() else list(on / on.sum()))
    return PhysicsScales(on_magnitude_ev=om, mean_weight=mw)


# ---------------------------------------------------------------- continuous cost

def continuous_cost(arm, truth_phys, pred_phys, truth_states, pred_states, sd_logc, sd_w):
    """(n,) mean of 1 - exp(-d^2 / 2) over the compared parameters (§4); d = difference / 0.1 eV for energies, / sd_logc
    and / sd_w for log c and width. label: the truth's nonzero quantities (and log c, w), 1 where the family differs.
    physics: the union of nonzero quantities, leaving out those of a type that is missing or added."""
    if arm not in ("label", "physics"):
        raise ValueError(f"unknown arm {arm!r}: expected 'label' or 'physics'")
    truth_states, pred_states = torch.as_tensor(truth_states), torch.as_tensor(pred_states)
    n_ev = len(EV_NAMES)
    scale = torch.tensor([ENERGY_SCALE_EV] * n_ev + [sd_logc, sd_w], dtype=pred_phys.dtype)
    g = 1 - torch.exp(-0.5 * ((pred_phys - truth_phys) / scale) ** 2)
    ones = torch.ones(len(g), 2, dtype=torch.bool)
    if arm == "label":
        mask = torch.cat([truth_phys[:, :n_ev] != 0, ones], 1)
    else:
        tp, pp = _STATE_PAT[truth_states], _STATE_PAT[pred_states]
        changed = tp ^ pp                                                       # (n, 3): a type missing or added
        drop = (torch.as_tensor(_OWNER)[None] & changed[:, None, :]).any(-1)     # (n, n_ev)
        mask = torch.cat([((truth_phys[:, :n_ev] != 0) | (pred_phys[:, :n_ev] != 0)) & ~drop, ones], 1)
    m = mask.to(g.dtype)
    val = (g * m).sum(1) / m.sum(1)
    if arm == "label":
        same = _FAMILY_T[truth_states] == _FAMILY_T[pred_states]
        val = torch.where(same, val, torch.ones_like(val))
    return val
