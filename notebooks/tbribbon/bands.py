import numpy as np


def _bands(H0, H1, nk):
    ks = np.linspace(-np.pi, np.pi, nk)
    return np.array([np.linalg.eigvalsh(H0 + H1 * np.exp(1j * k) + H1.conj().T * np.exp(-1j * k)) for k in ks])


def band_edges(H0, H1, nk=801):
    b = _bands(H0, H1, nk)
    return float(b.min()), float(b.max())


def open_channels(H0, H1, energies, nk=4001):
    """Right-moving channels at each energy = band crossings over the Brillouin zone / 2."""
    b = _bands(H0, H1, nk)
    out = []
    for E in energies:
        s = np.sign(b - E)
        out.append(int(np.sum(s[1:] != s[:-1]) // 2))
    return np.array(out)
