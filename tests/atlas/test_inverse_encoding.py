from dataclasses import fields

import numpy as np

from shazam.inverse.encoding import CONT, STATES, active_slots, decode, encode, state_index
from tbribbon.parent_prior import in_prior, sample_prior


def same_host(a, b):
    return all(abs(getattr(a, f.name) - getattr(b, f.name)) < 1e-9 if isinstance(getattr(a, f.name), float)
               else getattr(a, f.name) == getattr(b, f.name) for f in fields(a))


def test_every_prior_draw_round_trips():
    for i in range(5000):
        d = sample_prior(np.random.default_rng(i)); k, z, m = encode(d); back = decode(k, z)
        assert back.edge == d.edge and back.width == d.width and same_host(back.params, d.params)
        assert abs(back.density - d.density) < 1e-12 and np.all(z[~m] == 0)


def test_any_unbounded_vector_decodes_to_a_valid_host_inside_the_prior():
    rng = np.random.default_rng(1)
    for k in range(len(STATES)):
        d = decode(k, rng.normal(0, 3, 14)); d.params.validate(); assert in_prior(d)


def test_states_cover_the_twelve_pattern_lead_pairs_and_both_edges():
    assert len({(s.pattern, s.lead) for s in STATES}) == 12 and {s.edge for s in STATES} == {"armchair", "zigzag"}


def test_state_count_and_indexing():
    assert len(STATES) == 156 and len(set(STATES)) == 156 and len(CONT) == 14
    assert all(state_index(s) == i for i, s in enumerate(STATES))
    assert sum(s.edge == "armchair" for s in STATES) == 78


def test_inapplicable_switches_are_false():
    for s in STATES:
        n = sum(s.pattern)
        if n > 1:
            assert not s.eps_zero
        if not s.pattern[1]:
            assert not s.delta_zero and not s.t_AA_on
        assert sum(s.cross) == (0 if n == 1 else n - 1 if n == 2 else sum(s.cross))
        if n == 3:
            assert sum(s.cross) >= 2
        if n == 2:
            assert sum(s.cross) == 1


def test_active_slots_follow_the_state():
    for k, s in enumerate(STATES):
        m = active_slots(s)
        assert m.shape == (14,) and m.dtype == bool
        assert m[CONT.index("log_s")] and m[CONT.index("log_c")] and m[CONT.index("w")]
        assert m[CONT.index("t_AA")] == s.t_AA_on
        assert [m[CONT.index(n)] for n in ("t_CH", "t_CK", "t_HK")] == list(s.cross)
        assert m[CONT.index("delta")] == (s.pattern[1] and not s.delta_zero)
        for t, name, within in zip("CHK", ("eps_C", "eps_H_mean", "eps_K"), ("t_CC", "t_AB", "t_KK")):
            on = s.pattern["CHK".index(t)]
            lone_zero = sum(s.pattern) == 1 and s.eps_zero
            assert m[CONT.index(name)] == (on and not lone_zero)
            assert m[CONT.index(within)] == (on and s.lead != t)


def test_decode_ignores_inactive_slots_and_extreme_z_stays_in_the_prior():
    for k in range(len(STATES)):
        for z in (np.full(14, 1e3), np.full(14, -1e3), np.full(14, 40.0), np.full(14, -40.0), np.zeros(14)):
            d = decode(k, z); d.params.validate(); assert in_prior(d)
    rng = np.random.default_rng(2)
    for k in range(len(STATES)):                      # values on inactive slots change nothing
        z1 = rng.normal(0, 3, 14); z1b = z1.copy(); z1b[~active_slots(STATES[k])] += 5
        assert decode(k, z1) == decode(k, z1b)


def test_empty_delta_interval_decodes_to_zero():
    k = next(i for i, s in enumerate(STATES) if s.pattern == (False, True, False) and not s.delta_zero
             and not s.eps_zero and s.t_AA_on)
    z = np.zeros(14); z[CONT.index("eps_H_mean")] = 50.0          # eps_H_mean -> 2, so 2 - |eps| < 0.05
    d = decode(k, z)
    assert d.params.delta == 0 and in_prior(d)
    assert encode(d)[0] != k and STATES[encode(d)[0]].delta_zero


def test_zero_points_encode_as_switches():
    d = sample_prior(np.random.default_rng(0))
    k, _, _ = encode(d)
    assert STATES[k].edge == d.edge and STATES[k].pattern == d.params.pattern and STATES[k].lead == d.params.lead
