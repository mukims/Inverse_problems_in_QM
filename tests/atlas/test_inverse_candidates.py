from dataclasses import fields, replace

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
    # 1000 samples in 100 groups of 10 -> ([], 1.0)
    hosts = [h for edge in ("armchair", "zigzag") for pattern in PATTERNS for h in cube_hosts(edge, pattern)][:100]
    states, z = samples([(h, 10) for h in hosts])
    assert len(states) == 1000
    cands, p_un = candidates(states, z)
    assert cands == [] and p_un == 1.0


PATTERNS = [(True, False, False), (False, True, False), (False, False, True), (True, True, False),
            (True, False, True), (False, True, True), (True, True, True)]


def base_host(edge, pattern):
    """A valid host of the pattern (its first active type leads; every other free value is mid-range)."""
    lead = "CHK"[pattern.index(True)]
    own = {"C": ("eps_C", "t_CC"), "H": ("eps_H_mean", "t_AB"), "K": ("eps_K", "t_KK")}
    kw = {}
    for t, on in zip("CHK", pattern):
        if on:
            kw[own[t][0]] = 0.5
            kw[own[t][1]] = 1.0 if t == lead else 0.6
    if pattern[1]:
        kw.update(delta=0.5, t_AA=0.1)
    kw.update({name: 0.5 for name, (a, b) in zip(("t_CH", "t_CK", "t_HK"), ("CH", "CK", "HK"))
               if pattern["CHK".index(a)] and pattern["CHK".index(b)]})
    return Draw(HexParams(*pattern, lead, 2.0, **kw), edge, 10, 0.01)


def cube_hosts(edge, pattern):
    """The 8 corners of the cube over (s, c, w) around a base host: hosts that differ in one slot alone."""
    base = base_host(edge, pattern)
    return [base._replace(params=replace(base.params, s=s), width=w, density=c)
            for s in (1.0, 3.0) for c in (0.004, 0.03) for w in (5, 15)]


def test_at_most_five_candidates_ordered_by_probability():
    counts = [300, 200, 150, 120, 100, 80, 50]                         # seven hosts, each in its own (edge, pattern) group
    hosts = [(base_host("zigzag" if i % 2 else "armchair", pattern), n)
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


def uninformed(states, z, slot, seed=1):
    """Replace one slot of every sample by an uninformed value: uniform over the slot's prior interval (z = logit U)."""
    u = np.random.default_rng(seed).uniform(1e-6, 1 - 1e-6, len(z))
    z = z.copy()
    z[:, CONT.index(slot)] = np.log(u / (1 - u))
    return states, z


def test_two_hosts_differing_only_in_delta_are_two_candidates():
    hosts = [(h_only(1.5, 0.0, 0.3, 0.1, 9, 0.01), 600), (h_only(1.5, 0.0, 1.8, 0.1, 9, 0.01), 400)]
    cands, p_un = candidates(*samples(hosts, sigma=0.05))
    assert [round(c.probability, 12) for c in cands] == [0.6, 0.4] and p_un == 0.0


def test_a_width_minority_of_one_step_stays_in_its_host():
    hosts = [(h_only(1.5, 0.0, 0.8, 0.1, 9, 0.01), 900), (h_only(1.5, 0.0, 0.8, 0.1, 10, 0.01), 100)]
    cands, p_un = candidates(*samples(hosts, sigma=0.05))
    assert len(cands) == 1 and cands[0].probability == 1.0 and p_un == 0.0


def test_a_tight_host_with_one_uninformed_slot_is_one_candidate():
    states, z = samples([(h_only(1.5, 0.0, 0.8, 0.1, 9, 0.01), 1000)], sigma=0.05)
    cands, p_un = candidates(*uninformed(states, z, "t_AA"))
    assert len(cands) == 1 and cands[0].probability == 1.0 and p_un == 0.0
    lo, hi = cands[0].intervals["t_AA"]
    assert lo < 0.07 and hi > 0.25                                       # roughly the 5th-95th percentile of 0.03 to 0.3


def test_graphene_like_and_hbn_like_hosts_are_two_candidates():
    graphene = Draw(HexParams(False, True, False, "H", 2.3, t_AB=1.0), "armchair", 9, 0.01)
    hbn = Draw(HexParams(False, True, False, "H", 2.3, delta=1.576, t_AB=1.0), "armchair", 9, 0.01)
    cands, p_un = candidates(*samples([(graphene, 600), (hbn, 400)], sigma=0.05))
    assert [round(c.probability, 12) for c in cands] == [0.6, 0.4] and p_un == 0.0
    assert "delta" not in cands[0].intervals and "delta" in cands[1].intervals
    assert abs(cands[1].draw.params.delta - 1.576) < 0.1 and cands[0].draw.params.delta == 0.0


def test_mahal_99_is_the_99th_percentile_of_the_members_own_distances():
    states, z = samples([(HOST_A, 600), (HOST_B, 400)], seed=5)
    cands, _ = candidates(states, z)
    vec = np.array([full_vector(decode(int(k), v)) for k, v in zip(states, z)])
    for c in cands:
        d = vec[c.members] - c.mean
        dist = np.sqrt(np.einsum("ij,ij->i", d @ np.linalg.inv(c.cov), d))
        assert c.mahal_99 > 0 and abs(c.mahal_99 - np.percentile(dist, 99)) < 1e-6 * max(1.0, c.mahal_99)


def test_the_reported_state_is_the_decoded_medoids_own():
    """R11: with |eps_H_mean| > 1.95 the Delta interval is empty and decode gives Delta = 0 even in a state that switches Delta
    on; the candidate reports the state of its decoded host (Delta off), and no Delta interval."""
    from shazam.inverse.encoding import state_index, state_of
    host = Draw(HexParams(False, True, False, "H", 1.5, eps_H_mean=1.97, t_AB=1.0, t_AA=0.1), "armchair", 8, 0.01)
    on = state_of(host)._replace(delta_zero=False)
    k, z, _ = encode(host)
    z = np.tile(z, (40, 1))
    z[:, CONT.index("delta")] = 0.3                                       # a delta value the empty interval ignores
    z[:, CONT.index("log_s")] += np.random.default_rng(0).normal(0, 1e-3, 40)
    (cand,), _ = candidates(np.full(40, state_index(on)), z)
    assert cand.draw.params.delta == 0.0
    assert STATES[cand.state] == state_of(cand.draw) and STATES[cand.state].delta_zero
    assert "delta" not in cand.intervals and "s" in cand.intervals


def test_candidates_medoid_respects_modal_family_and_active_intervals():
    # 800 draws of graphene (delta=0), 200 draws of tiny-delta hbn (delta=0.05).
    # Since distance is below split threshold, they form one spatial cluster.
    # Modal family is delta_zero=True. Medoid must belong to modal family (delta=0.0).
    graphene = Draw(HexParams(False, True, False, "H", 2.3, t_AB=1.0), "armchair", 9, 0.01)
    hbn = Draw(HexParams(False, True, False, "H", 2.3, delta=0.05, t_AB=1.0), "armchair", 9, 0.01)
    states, z = samples([(graphene, 800), (hbn, 200)], sigma=0.01, seed=42)
    cands, p_un = candidates(states, z)
    assert len(cands) == 1 and p_un == 0.0
    cand = cands[0]
    assert cand.draw.params.delta == 0.0
    assert "delta" not in cand.intervals
    assert cand.z_members is not None and cand.z_members.shape == (1000, 14)

    # Conversely: 800 draws of hbn with delta=0.5 (active), 200 draws of graphene (delta=0, inactive).
    # Modal family has delta active. Interval for delta must be computed over active members (delta > 0.4),
    # not pulled down to ~0 by the 200 delta=0 members.
    hbn_major = Draw(HexParams(False, True, False, "H", 2.3, delta=0.5, t_AB=1.0), "armchair", 9, 0.01)
    states2, z2 = samples([(hbn_major, 800), (graphene, 200)], sigma=0.01, seed=43)
    cands2, _ = candidates(states2, z2)
    cand2 = cands2[0]
    assert cand2.draw.params.delta > 0.45
    assert "delta" in cand2.intervals
    lo, hi = cand2.intervals["delta"]
    # If 200 zeros were mixed in (20% of samples), the 5th percentile would be exactly 0.0!
    # Because inactive members are excluded, lo is ~0.48, well above 0.4.
    assert lo > 0.4

