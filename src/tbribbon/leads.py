"""Sancho-Rubio surface Green's functions for semi-infinite leads."""
import numpy as np


def surface_gf(E, H0, H1, eta=1e-4, tol=1e-10, max_iter=300):
    """Surface GF of a lead whose cells continue in the direction coupled by H1."""
    n = H0.shape[0]
    z = (E + 1j * eta) * np.eye(n)
    eps_s, eps, alpha, beta = H0.astype(complex), H0.astype(complex), H1.astype(complex), H1.conj().T.astype(complex)
    for _ in range(max_iter):
        g = np.linalg.inv(z - eps)
        ag, bg = alpha @ g, beta @ g
        eps_s = eps_s + ag @ beta
        eps = eps + ag @ beta + bg @ alpha
        alpha, beta = ag @ alpha, bg @ beta
        if np.abs(alpha).max() < tol:
            break
    return np.linalg.inv(z - eps_s)


class LeadCache:
    """Left lead extends to -inf (couples via H1^dagger), right lead to +inf (via H1)."""
    def __init__(self, H0, H1, energies, eta=1e-4):
        self.gL = [surface_gf(E, H0, H1.conj().T, eta) for E in energies]
        self.gR = [surface_gf(E, H0, H1, eta) for E in energies]
