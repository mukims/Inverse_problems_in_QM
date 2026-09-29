"""Recursive Green's function transmission through a disordered ribbon."""
import numpy as np


def _caroli(z, H0, H1, shifts, gL, gR):
    H1d = H1.conj().T
    SL, SR = H1d @ gL @ H1, H1 @ gR @ H1d
    n_cells = len(shifts)
    g = np.linalg.inv(z - H0 - np.diag(shifts[0]) - SL - (SR if n_cells == 1 else 0))
    G_n1 = g
    for i in range(1, n_cells):
        extra = SR if i == n_cells - 1 else 0
        g = np.linalg.inv(z - H0 - np.diag(shifts[i]) - H1d @ g @ H1 - extra)
        G_n1 = g @ H1d @ G_n1
    GL, GR = 1j * (SL - SL.conj().T), 1j * (SR - SR.conj().T)
    return float(np.real(np.trace(GR @ G_n1 @ GL @ G_n1.conj().T)))


def _legacy_trace(z, H0, H1, shifts, gL, gR):
    """Formula of ca_sq.device / agnr_lib.device_transmission (unbounded; clip downstream)."""
    I = np.eye(H0.shape[0])
    G = gL
    for s in shifts:
        g_d = np.linalg.inv(z - H0 - np.diag(s))
        G = np.linalg.solve(I - g_d @ H1 @ G @ H1, g_d)
    left, right = G, gR
    c_l = np.linalg.solve(I - right @ H1 @ left @ H1, left)
    c_r = np.linalg.solve(I - left @ H1 @ right @ H1, right)
    G_ll, G_rr = c_l - c_l.conj().T, c_r - c_r.conj().T
    G_lr = left @ H1 @ c_r
    Gnon = G_lr - G_lr.conj().T
    return float(np.abs(np.trace(G_ll @ H1 @ G_rr @ H1 - H1 @ Gnon @ H1 @ Gnon)))


FORMULAS = {"caroli": _caroli, "legacy_trace": _legacy_trace}


def transmission(E, H0, H1, shifts, gL, gR, eta=None, formula="caroli"):
    if eta is None:
        eta = 1e-3 if formula == "legacy_trace" else 1e-6
    z = (E + 1j * eta) * np.eye(H0.shape[0])
    return FORMULAS[formula](z, H0, H1, shifts, gL, gR)


def spectrum(H0, H1, energies, shifts, leads, eta=None, formula="caroli"):
    return np.array([transmission(E, H0, H1, shifts, leads.gL[i], leads.gR[i], eta=eta, formula=formula)
                     for i, E in enumerate(energies)])
