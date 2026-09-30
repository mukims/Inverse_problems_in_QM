import numpy as np
import pytest

from tbribbon.check_store import check_seed_excess


def test_spike_robust_seed_excess_isolated_spike_passes():
    """A seed with a single 10^4 numerical resonance spike in an unmasked channel passes."""
    n_seeds = 10
    n_channels = 200
    pris = np.ones(n_channels)

    # Base transmission: disorder reducing transmission (<= pristine)
    spec = np.full((n_seeds, n_channels), 0.8)

    # Seed 0 has an isolated 10^4 spike at channel 50
    spec[0, 50] = 10000.0

    max_excess, valid = check_seed_excess(spec, pris, step_mask=None, tol=0.05)
    assert valid is True
    # min(10000, 1 + 1) - 1 = 1.0 on 1 channel; -0.2 on 199 channels -> mean is negative
    assert max_excess < 0.05


def test_spike_robust_seed_excess_systematic_excess_fails():
    """A seed systematically shifted up by 0.2 across all unmasked channels fails."""
    n_seeds = 10
    n_channels = 200
    pris = np.ones(n_channels)

    spec = np.full((n_seeds, n_channels), 0.8)

    # Seed 1 has an unphysical systematic excess of +0.2 across all channels
    spec[1, :] = pris + 0.2

    max_excess, valid = check_seed_excess(spec, pris, step_mask=None, tol=0.05)
    assert valid is False
    assert max_excess == pytest.approx(0.2, abs=1e-5)
