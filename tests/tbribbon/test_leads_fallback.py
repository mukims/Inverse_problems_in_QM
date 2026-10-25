"""LeadCache retries only energies whose lead decimation overflows, shifting the whole channel; finite results are
unchanged."""
import subprocess
import types
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from tbribbon import leads
from tbribbon.disorder import impurity_shifts
from tbribbon.leads import LeadCache
from tbribbon.parent import HexParams, named_material, parent_ribbon
from tbribbon.parent_data import LEAD_ETA, energies_model
from tbribbon.transport import spectrum

REPO = Path(__file__).resolve().parents[2]
BASE = "aa6876ed2"                      # last commit of leads.py and transport.py before the fallback


def _old(module):
    src = subprocess.run(["git", "show", f"{BASE}:src/tbribbon/{module}.py"], cwd=REPO, check=True,
                         capture_output=True, text=True).stdout
    mod = types.ModuleType(f"old_{module}")
    exec(compile(src, f"old_{module}.py", "exec"), mod.__dict__)
    return mod


@pytest.mark.parametrize("name,edge,width", [("graphene-ideal", "armchair", 5), ("hbn", "zigzag", 4),
                                             ("kagome", "zigzag", 3), ("triangular", "zigzag", 6)])
def test_finite_results_are_bit_identical_to_the_old_code(name, edge, width):
    old_leads, old_transport = _old("leads"), _old("transport")
    p = named_material(name)
    rb = parent_ribbon(p, edge, width)
    E = energies_model(p.s)[::13]
    shifts = impurity_shifts(6, rb.H0.shape[0], 3, 5, 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = old_leads.LeadCache(rb.H0, rb.H1, E, eta=LEAD_ETA)
        T_ref = old_transport.spectrum(rb.H0, rb.H1, E, shifts, ref)
    new = LeadCache(rb.H0, rb.H1, E, eta=LEAD_ETA)
    assert new.energy_shift == [0.0] * len(E)
    for a, b in zip(ref.gL + ref.gR, new.gL + new.gR):
        assert np.all(np.isfinite(a)) and np.array_equal(a, b)
    assert np.array_equal(T_ref, spectrum(rb.H0, rb.H1, E, shifts, new))


def _draw_74382():
    """Triangular zigzag w = 19; channel 66 overflowed at eta = 1.2e-5 (left lead); ±1e-8 shifts are fine."""
    p = replace(named_material("triangular"), s=0.9333748752477868)
    return parent_ribbon(p, "zigzag", 19), energies_model(p.s)[65:68], 74382


def _draw_43316():
    """All three types, zigzag w = 6; channel 157 overflowed (right lead); ±1e-8 shifts converge but are non-causal."""
    p = HexParams(on_C=True, on_H=True, on_K=True, lead="C", s=1.0541883840778008, eps_C=0.6744874879997722,
                  eps_H_mean=-1.0860015495262059, delta=0.226035836007766, eps_K=-0.47090540839929407, t_CC=1.0,
                  t_AB=0.6244333938473421, t_KK=0.5515668887599255, t_AA=0.0, t_CH=0.0, t_CK=0.8283873724168273,
                  t_HK=0.6460214152882432)
    return parent_ribbon(p, "zigzag", 6), energies_model(p.s)[156:159], 43316


@pytest.mark.parametrize("case", [_draw_74382, _draw_43316])
def test_an_overflowing_channel_is_shifted_whole_and_physical(case):
    rb, E, seed = case()
    with threadpool_limits(limits=1), warnings.catch_warnings():  # one BLAS thread, as in the generator's workers
        warnings.simplefilter("ignore", RuntimeWarning)
        bad = _old("leads").LeadCache(rb.H0, rb.H1, E, eta=LEAD_ETA)
        if all(np.all(np.isfinite(g)) for g in bad.gL + bad.gR):
            pytest.skip("this BLAS build does not overflow at the recorded energy")
        warnings.simplefilter("error")          # the retry must not leak overflow warnings
        lc = LeadCache(rb.H0, rb.H1, E, eta=LEAD_ETA)
        clean = spectrum(rb.H0, rb.H1, E, np.zeros((4, rb.H0.shape[0])), lc)
        dirty = spectrum(rb.H0, rb.H1, E, impurity_shifts(20, rb.H0.shape[0], 30, seed, 0.5), lc)
    assert lc.energy_shift[0] == lc.energy_shift[2] == 0.0 and lc.energy_shift[1] != 0.0
    assert np.allclose(clean, clean[0], atol=0.01)                              # the clean staircase is flat here
    assert np.all(np.isfinite(dirty)) and np.all((dirty > 0) & (dirty <= clean + 0.05))


def test_a_converged_but_non_causal_retry_is_not_accepted(monkeypatch):
    calls = []

    def fake(E, H0, H1, eta, tol, max_iter):
        calls.append(E)
        if len(calls) <= 2:
            return np.full((2, 2), np.nan, complex), False
        if len(calls) <= 4:
            return np.diag([1j, 1j]), True                           # i(g - g^dagger) = -2: converged but non-causal
        return np.diag([-1j, -1j]), True
    monkeypatch.setattr(leads, "_decimate", fake)
    lc = LeadCache(np.zeros((2, 2)), np.eye(2), [0.3], eta=1e-5)
    assert np.all(lc.gL[0] == np.diag([-1j, -1j])) and lc.energy_shift == [leads.RETRY_SHIFTS[1]]


def test_an_unconverged_retry_is_not_accepted(monkeypatch):
    calls = []

    def fake(E, H0, H1, eta, tol, max_iter):
        calls.append(E)
        if len(calls) <= 2:
            return np.full((2, 2), np.nan, complex), False           # the overflow (left and right at E)
        if len(calls) <= 4:
            return np.ones((2, 2), complex), False                   # finite but unconverged: must be rejected
        return 2 * np.ones((2, 2), complex), True
    monkeypatch.setattr(leads, "_decimate", fake)
    lc = LeadCache(np.zeros((2, 2)), np.eye(2), [0.3], eta=1e-5)
    assert np.all(lc.gL[0] == 2) and lc.energy_shift == [leads.RETRY_SHIFTS[1]]


def test_a_lead_that_never_converges_raises(monkeypatch):
    monkeypatch.setattr(leads, "_decimate", lambda *a: (np.full((2, 2), np.nan, complex), False))
    with pytest.raises(FloatingPointError, match="not finite"):
        LeadCache(np.zeros((2, 2)), np.eye(2), [0.3], eta=1e-5)
