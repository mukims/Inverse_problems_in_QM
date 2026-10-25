"""Sancho-Rubio surface Green's functions for semi-infinite leads."""
import numpy as np

# At an energy almost exactly on a lead band feature the decimation can overflow (about 3 in 10^5 parent-lattice
# spectra at eta = 1.2e-5) and return NaN. Only then is that energy retried, shifted by these amounts (energy units of
# H): both leads at the same shifted energy, accepted only when both decimations converged and are causal (a very
# small shift can converge to a non-causal, unphysical result). The shift is recorded in LeadCache.energy_shift so that
# transport.spectrum evaluates the device at the same energy; mixing a shifted lead with an unshifted device or other
# lead also gives unphysical values. Every finite first result is unchanged.
RETRY_SHIFTS = (1e-8, -1e-8, 1e-7, -1e-7, 1e-6, -1e-6, 1e-5, -1e-5)
CAUSAL_TOL = 1e-6        # smallest eigenvalue of i(g - g^dagger) may dip this far below 0, relative to the largest


def _decimate(E, H0, H1, eta, tol, max_iter):
    """(surface GF, converged) by Sancho-Rubio decimation."""
    n = H0.shape[0]
    z = (E + 1j * eta) * np.eye(n)
    eps_s, eps, alpha, beta = H0.astype(complex), H0.astype(complex), H1.astype(complex), H1.conj().T.astype(complex)
    converged = False
    for _ in range(max_iter):
        g = np.linalg.inv(z - eps)
        ag, bg = alpha @ g, beta @ g
        eps_s = eps_s + ag @ beta
        eps = eps + ag @ beta + bg @ alpha
        alpha, beta = ag @ alpha, bg @ beta
        if np.abs(alpha).max() < tol:
            converged = True
            break
    return np.linalg.inv(z - eps_s), converged


def surface_gf(E, H0, H1, eta=1e-4, tol=1e-10, max_iter=300):
    """Surface GF of a lead whose cells continue in the direction coupled by H1."""
    return _decimate(E, H0, H1, eta, tol, max_iter)[0]


def _finite(*gs):
    return all(np.all(np.isfinite(g)) for g in gs)


def _causal(g):
    """A retarded surface GF has a positive semidefinite spectral function i(g - g^dagger)."""
    a = 1j * (g - g.conj().T)
    w = np.linalg.eigvalsh((a + a.conj().T) / 2)
    return w.min() >= -CAUSAL_TOL * max(1.0, np.abs(w).max())


class LeadCache:
    """Left lead extends to -inf (couples via H1^dagger), right lead to +inf (via H1).

    energy_shift[i] is 0 except where the leads overflowed at energies[i] and were evaluated at energies[i] + shift
    (RETRY_SHIFTS); transport.spectrum adds it to that channel's energy.
    """
    def __init__(self, H0, H1, energies, eta=1e-4, tol=1e-10, max_iter=300):
        H1d = H1.conj().T
        self.gL, self.gR, self.energy_shift = [], [], []
        with np.errstate(over="ignore", invalid="ignore"):
            for E in energies:
                gL, gR, shift = surface_gf(E, H0, H1d, eta, tol, max_iter), surface_gf(E, H0, H1, eta, tol, max_iter), 0.0
                if not _finite(gL, gR):
                    gL, gR, shift = self._retry(E, H0, H1, H1d, eta, tol, max_iter)
                self.gL.append(gL)
                self.gR.append(gR)
                self.energy_shift.append(shift)

    @staticmethod
    def _retry(E, H0, H1, H1d, eta, tol, max_iter):
        for shift in RETRY_SHIFTS:
            gL, okL = _decimate(E + shift, H0, H1d, eta, tol, max_iter)
            gR, okR = _decimate(E + shift, H0, H1, eta, tol, max_iter)
            if okL and okR and _finite(gL, gR) and _causal(gL) and _causal(gR):
                return gL, gR, shift
        raise FloatingPointError(f"lead surface Green's functions are not finite at E = {E!r}, and no shift up to 1e-5 "
                                 "gave converged, causal leads")
