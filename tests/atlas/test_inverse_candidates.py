from dataclasses import fields

import numpy as np

from shazam.inverse.candidates import Candidate, candidates, full_vector, is_realistic, nearest_named
from shazam.inverse.encoding import CONT, STATES, active_slots, decode, encode
from tbribbon.parent import HexParams, device_cells_for, named_material
from tbribbon.parent_data import n_impurities
from tbribbon.parent_prior import Draw

SIGMA = 0.02          # jitter in z; small enough that the integer width never rounds to a neighbour


def same_host(a, b):
    return all(abs(getattr(a, f.name) - getattr(b, f.name)) < 1e-9 if isinstance(getattr(a, f.name), float)
               else getattr(a, f.name) == getattr(b, f.name) for f in fields(a))


def samples(hosts, seed=0, sigma=SIGMA):
    """(states, z) for [(Draw, count), ...]: each host encoded, with Gaussian jitter on its ACTIVE slots only, shuffled."""
    rng = np.random.default_rng(seed)
    states, zs = [], []
    for draw, n in hosts:
        k, z, mask = encode(draw)
        states += [k] * n
        zs.append(z + rng.normal(0, sigma, (n, len(CONT))) * mask)
    states, z = np.array(states), np.concatenate(zs)
    order = rng.permutation(len(states))
    return states[order], z[order]


def h_only(s, eps, delta, t_AA, w, c, edge="armchair"):
    return Draw(HexParams(False, True, False, "H", s, eps_H_mean=eps, delta=delta, t_AB=1.0, t_AA=t_AA), edge, w, c)


HOST_A = h_only(2.7, 0.3, 0.5, 0.1, 7, 0.01)
HOST_B = h_only(1.2, -0.5, 1.2, 0.2, 12, 0.003)


def test_two_well_separated_hosts_give_two_candidates_with_their_shares():
    states, z = samples([(HOST_A, 600), (HOST_B, 400)])               # 600 + 400 samples -> probs 0.6, 0.4, p_unassigned 0
    cands, p_un = candidates(states, z)
    assert [round(c.probability, 12) for c in cands] == [0.6, 0.4] and p_un == 0.0
    for c, host, n in zip(cands, (HOST_A, HOST_B), (600, 400)):
        assert len(c.members) == n and c.edge == "armchair" and c.pattern == (False, True, False)
        assert abs(c.draw.params.s - host.params.s) < 0.05 * host.params.s and c.draw.width == host.width
        assert abs(c.draw.params.delta - host.params.delta) < 0.05
    assert isinstance(cands[0], Candidate) and np.allclose(np.sort(np.concatenate([c.members for c in cands])),
                                                           np.arange(1000))


def test_a_cluster_below_twenty_samples_is_unassigned():
    states, z = samples([(HOST_A, 985), (HOST_B, 15)])                # 985 + 15 -> one candidate 0.985, p_unassigned 0.015
    cands, p_un = candidates(states, z)
    assert len(cands) == 1 and abs(cands[0].probability - 0.985) < 1e-12 and abs(p_un - 0.015) < 1e-12
    cands, p_un = candidates(states, z, min_size=15)                  # the threshold is a parameter
    assert len(cands) == 2 and p_un == 0.0


def test_samples_differing_only_in_lead_or_a_near_zero_hopping_form_one_candidate():
    # H+K tied hoppings (t_AB = t_KK = 1), lead H vs K; t_AA 0 vs 0.03
    def hk(lead, t_AA):
        p = HexParams(False, True, True, lead, 1.5, eps_H_mean=0.2, eps_K=-0.3, t_AB=1.0, t_KK=1.0, t_AA=t_AA, t_HK=0.6)
        return Draw(p, "zigzag", 9, 0.01)
    hosts = [(hk(lead, t_AA), 250) for lead in "HK" for t_AA in (0.0, 0.03)]
    assert len({encode(d)[0] for d, _ in hosts}) == 4                  # four distinct discrete states
    states, z = samples(hosts)
    cands, p_un = candidates(states, z)
    assert len(cands) == 1 and cands[0].probability == 1.0 and p_un == 0.0


def test_medoid_is_a_member_and_inside_its_intervals():
    states, z = samples([(HOST_A, 600), (HOST_B, 400)], seed=3)
    cands, _ = candidates(states, z)
    for c in cands:
        assert any(states[m] == c.state and same_host(decode(states[m], z[m]).params, c.draw.params)
                   and decode(states[m], z[m]).width == c.draw.width for m in c.members)
        assert STATES[c.state].edge == c.edge and STATES[c.state].pattern == c.pattern
        mask = active_slots(c.state)
        expected = {"c", "s", "w"} | {n for n, on in zip(CONT, mask) if on and n not in ("log_s", "log_c", "w")}
        assert set(c.intervals) == expected
        value = {"c": c.draw.density, "w": c.draw.width, "s": c.draw.params.s}
        for k, (lo, hi) in c.intervals.items():
            assert lo <= value.get(k, getattr(c.draw.params, k, None)) <= hi
        lo, hi = c.n_imp_interval
        assert 1 <= lo <= hi
        assert lo <= n_impurities(c.draw, device_cells_for(c.draw.params, c.edge)) <= hi
        assert c.mean.shape == (14,) and c.cov.shape == (14, 14)
        assert np.all(np.linalg.eigvalsh(c.cov) >= 1e-4 - 1e-12)         # regularised by 1e-4 I
        assert c.nearest_material in ("graphene-ideal", "silicene", "germanene", "hbn", "triangular", "kagome")
        assert c.realistic is True and c.box_edge == []


def test_no_candidate_when_every_cluster_is_tiny():
    # 1000 samples in about 100 groups of 10 -> ([], 1.0); 88 hosts of 11 or 12 samples (see factorial_hosts)
    hosts = [h for edge in ("armchair", "zigzag") for pattern in PATTERNS for h in factorial_hosts(edge, pattern)]
    assert len(hosts) == 88
    sizes = np.array_split(np.arange(1000), len(hosts))
    states, z = samples([(h, len(s)) for h, s in zip(hosts, sizes)])
    assert len(states) == 1000 and max(len(s) for s in sizes) < 20
    cands, p_un = candidates(states, z)
    assert cands == [] and p_un == 1.0


PATTERNS = [(True, False, False), (False, True, False), (False, False, True), (True, True, False),
            (True, False, True), (False, True, True), (True, True, True)]
# (low, high) of each free slot used by factorial_hosts; the first active type of a pattern is its lead
LEVELS = {"s": (1.0, 3.0), "eps_C": (-1.5, 1.5), "eps_H_mean": (-1.0, 1.0), "delta": (0.2, 0.8), "eps_K": (-1.5, 1.5),
          "t_CC": (0.3, 0.9), "t_AB": (0.3, 0.9), "t_KK": (0.3, 0.9), "t_AA": (0.05, 0.25), "t_CH": (0.2, 0.9),
          "t_CK": (0.2, 0.9), "t_HK": (0.2, 0.9), "c": (0.004, 0.03), "w": (5, 15)}


def factorial_hosts(edge, pattern):
    """All 2**b hosts of a two-level factorial design over the free slots of a pattern, three slots per binary factor, so
    two hosts differing in one factor stand 2 * sqrt(3) > 3 standard units apart however the group is standardised."""
    lead = "CHK"[pattern.index(True)]
    names = ["s", "c", "w"]
    for t, on in zip("CHK", pattern):
        if on:
            names += [{"C": "eps_C", "H": "eps_H_mean", "K": "eps_K"}[t]]
            names += [{"C": "t_CC", "H": "t_AB", "K": "t_KK"}[t]] * (t != lead) + ["delta", "t_AA"] * (t == "H")
    names += [n for n, (a, b) in zip(("t_CH", "t_CK", "t_HK"), (("C", "H"), ("C", "K"), ("H", "K")))
              if pattern["CHK".index(a)] and pattern["CHK".index(b)]]
    n_bits = len(names) // 3
    hosts = []
    for code in range(2 ** n_bits):
        value = {n: LEVELS[n][(code >> (i // 3)) & 1] for i, n in enumerate(names[:3 * n_bits])}
        value.update({n: LEVELS[n][0] for n in names[3 * n_bits:]})
        kw = {n: v for n, v in value.items() if n not in ("s", "c", "w")}
        kw.update({{"C": "t_CC", "H": "t_AB", "K": "t_KK"}[lead]: 1.0})
        kw.update({n: LEVELS[n][0] for n in ("t_CH", "t_CK", "t_HK") if n in names and n not in kw})
        hosts.append(Draw(HexParams(*pattern, lead, value["s"], **kw), edge, int(value["w"]), value["c"]))
    return hosts


def test_at_most_five_candidates_ordered_by_probability():
    counts = [300, 200, 150, 120, 100, 80, 50]                         # seven hosts, each in its own (edge, pattern) group
    hosts = [(factorial_hosts("zigzag" if i % 2 else "armchair", pattern)[-1], n)
             for i, (n, pattern) in enumerate(zip(counts, PATTERNS))]
    states, z = samples(hosts)
    cands, p_un = candidates(states, z)
    assert len(cands) == 5 and [round(c.probability, 12) for c in cands] == [0.3, 0.2, 0.15, 0.12, 0.1]
    assert p_un == 0.0                                                 # clusters beyond `top` are simply not returned
    assert len(candidates(states, z, top=2)[0]) == 2


def test_nearest_named_material_of_graphene_is_graphene():
    d = Draw(named_material("graphene-ideal"), "zigzag", 8, 0.01)
    name, dist = nearest_named(d)
    assert name == "graphene-ideal" and dist == 0.0
    assert nearest_named(Draw(named_material("hbn"), "armchair", 8, 0.01)) == ("hbn", 0.0)
    states, z = samples([(Draw(named_material("graphene-ideal"), "armchair", 7, 0.01), 1000)])
    (c,), _ = candidates(states, z)
    assert c.nearest_material == "graphene-ideal" and 0 < c.material_distance < 0.2


def test_full_vector_is_canonical_with_zeros_for_inactive_values():
    d = Draw(named_material("hbn"), "armchair", 8, 0.01)
    v = full_vector(d)
    assert v.shape == (14,)
    assert np.allclose(v[[0, 12, 13]], [np.log(2.30), np.log(0.01), 8])
    assert v[CONT.index("delta")] == 3.625 / 2.30 and v[CONT.index("t_AB")] == 1.0
    assert v[CONT.index("eps_C")] == 0 and v[CONT.index("t_CH")] == 0
    assert is_realistic(d) is True


def test_a_group_with_a_single_sample_is_its_own_cluster():
    states, z = samples([(HOST_A, 1), (h_only(1.2, -0.5, 1.2, 0.2, 12, 0.003, edge="zigzag"), 1)])
    cands, p_un = candidates(states, z, min_size=1)
    assert [c.probability for c in cands] == [0.5, 0.5] and p_un == 0.0
    assert np.allclose(cands[0].cov, 1e-4 * np.eye(14))                  # single member: cov = 1e-4 I
    assert candidates(states, z)[1] == 1.0
