import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from scipy.sparse import diags
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
from functools import lru_cache
from scipy.linalg import solve
import os
import h5py
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def unitcell(w, d, t, e, m):
    dim = 2 * m
    base_val = (w + 1j*d)
    base = base_val * np.eye(2*m)
    idy = np.arange(0,2*m-1,1)
    base[idy,idy+1] = t
    base[idy+1,idy] = t
    idx = np.arange(0,m,2)
    base[idx,2*m-1-idx] = t
    base[2*m-1-idx,idx] = t
    return base 

def T1_matrix(t, m):
    dim = 2 * m
    T = np.zeros((dim, dim), dtype=np.complex128)
    n = np.arange(1, (m - 1)//2 + 1)
    i = 2*n - 1
    j = 2*m - 2*n
    T[i, j] = t
    return T

def connection(t,m):
    idx = np.arange(2,m,2)
    base = np.zeros((2*m,2*m),dtype=np.complex64)
    base[2*m - idx,idx - 1] = t 
    return base

DEVICE_COMBS = {}
def _get_device_combs(width: int) -> np.ndarray:
    width = int(2*width)
    if width not in DEVICE_COMBS:
        DEVICE_COMBS[width] = np.stack(
            np.meshgrid(np.arange(100), np.arange(width), indexing="ij"),
            axis=-1
        ).reshape(-1, 2)
    return DEVICE_COMBS[width]

@lru_cache(maxsize=2048)
def chosen_for_config(n: int, width: int, config: int) -> np.ndarray:
    n = int(n)
    width = int(width)
    config = int(config)
    device_combs = _get_device_combs(width)
    rng = np.random.RandomState(config)
    idx = rng.choice(len(device_combs), size=n, replace=False)
    return device_combs[idx]

def possible_combs(n: int, width: int):
    n = int(n)
    width = int(width)
    def combs_for_seed(seed: int):
        return chosen_for_config(n, width, seed)
    return combs_for_seed

def unidevice(w, d, t, e, size, config, n, numberofunitcell, combs_fn=None):
    size = int(size)
    if combs_fn is None:
        combs_fn = possible_combs(int(n), int(size))
    imps = combs_fn(int(config))
    x = imps[:, 0]
    y = imps[:, 1]
    z = int(numberofunitcell)
    mat = unitcell(w, d, t, e, int(size))
    mask = (x == z)
    if not np.any(mask):
        return mat
    imp_indices = y[mask]
    diag_val = (w + 1j*d - 0.5)
    mat[imp_indices, imp_indices] = diag_val
    return mat

def import_leads(size):
    arr = np.load(os.path.expanduser(f"~/machine_learning/transmission_github/transmissions/leads/agnr_{size}.npy"))
    return arr

g_7 = import_leads(7)

def rho_matrix(t, m):
    dim = 2 * m
    rho = np.zeros((dim, dim), dtype=complex)
    for n in range(1, (m - 1) // 2 + 1):
        idx = 2 * n - 1
        rho[idx, idx] = t
    return rho

def device_transmission(w, d, t, e, size, config, concentration, x=None):
    ene = int(w * 100)
    m = size
    dim = 2 * m
    I = np.eye(dim, dtype=complex)

    global g_7

    # 1. Lead Surface Green's Functions:
    left = g_7[ene]                      # Left lead surface Green's function

    # 2. Hopping Matrices:
    tin = T1_matrix(t, m)
    tin_d = tin.T                         # Forward hopping T
    rho = rho_matrix(t, m)                # Contact operator \[Rho]

    # 3. Device Propagation (100 Unit Cells):
    combs_fn = possible_combs(concentration, size)
    g_new = left

    for i in range(100):
        unit_i = unidevice(w, d, t, e, size, config, concentration, i, combs_fn=combs_fn)
        gd = np.linalg.inv(unit_i)
        G = np.linalg.solve(I - gd @ tin_d @ g_new @ tin, gd)
        g_new = G

    left_device = g_new

    # 4. Connected Interface Green's Functions:
    IL = np.linalg.solve(I - left_device @ rho @ left @ rho, left_device)
    IR = np.linalg.solve(I - left @ rho @ left_device @ rho, left)

    if x is not None:
        return -np.imag(IL[x, x]) / np.pi

    # 5. Spectral Functions & Non-local Cross Green's Function:
    gdd = IL - IL.conj().T
    grr = IR - IR.conj().T

    Gnonlocal = left @ rho @ IR
    GNON = Gnonlocal - Gnonlocal.conj().T

    # 6. Mathematica Trace Calculation:
    term1 = gdd @ rho @ grr @ rho
    term2 = rho @ GNON @ rho @ GNON

    tr1 = np.abs(np.trace(term1 - term2))

    return np.abs(tr1)

def transmission(config, conc, size):
    trans = [device_transmission(i, 1e-5, 1, 0,size,config,conc) for i in np.arange(0,3,0.01)]
    return trans

def run_transmission(args):
    config, conc, size = args
    return transmission(config, conc, size)

def compute_for_one_config(args):
    cfg, conc, size = args
    w_vals = np.arange(0, 3, 0.01)
    out = np.zeros(len(w_vals), dtype=float)
    for i, w in enumerate(w_vals):
        out[i] = device_transmission(w, 0.0001, 1, 0, size, cfg, conc)
    return out

def compute_for_concentration(conc, size, nconfigs, out_dir):
    print(f"\n[INFO] Starting test data generation for conc = {conc}")
    os.makedirs(out_dir, exist_ok=True)
    args = [(cfg, conc, size) for cfg in range(nconfigs)]
    with Pool(processes=min(4, cpu_count())) as pool:
        for cfg, result in enumerate(
            tqdm(pool.imap(compute_for_one_config, args),
                 total=nconfigs,
                 desc=f"conc {conc} (test)")
        ):
            np.save(os.path.join(out_dir, f"7_agnr_conc{conc}_cfg{cfg}_test.npy"), result)

def main():
    # TEST CONFIGURATION
    concs = np.arange(3,45,2)  # Just a few concentrations for testing
    nconfigs = 100     # Small number of configs instead of 10000
    size = 7

    # Output to a specific test directory inside the project
    project_root = os.path.expanduser("~/machine_learning/transmission_github/transmissions")
    out_dir = os.path.join(project_root, "data", "test", "transmission_results")
    
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Output directory: {out_dir}")
    print(f"[INFO] Running in TEST mode. Generating {nconfigs} configs for concentrations: {concs}")
    
    for conc in concs:
        compute_for_concentration(conc, size, nconfigs, out_dir)

    print("\n[DONE] Test data generation complete!")

if __name__ == "__main__":
    main()
