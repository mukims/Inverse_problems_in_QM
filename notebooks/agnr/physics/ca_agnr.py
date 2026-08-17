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
    i = 2*n - 1           # to 0-based
    j = 2*m - 2*n

    T[i, j] = t
    return T



def connection(t,m):
    idx = np.arange(2,m,2)
    base = np.zeros((2*m,2*m),dtype=np.complex64)

    base[2*m - idx,idx - 1] = t 

    return base





# ----------------------------------------------------------------------
# 1) Precomputed global cache of device_combs per width
# ----------------------------------------------------------------------
DEVICE_COMBS = {}


def _get_device_combs(width: int) -> np.ndarray:
    """
    Return (global cached) device combinations for a given width.
    Each row is (i, j) with i in [0..99] and j in [0..width-1].
    """
    width = int(2*width)
    if width not in DEVICE_COMBS:
        # Build once, store once
        DEVICE_COMBS[width] = np.stack(
            np.meshgrid(np.arange(100), np.arange(width), indexing="ij"),
            axis=-1
        ).reshape(-1, 2)
    return DEVICE_COMBS[width]


# ----------------------------------------------------------------------
# 2) Cached selection of N random impurity sites for a given config
# ----------------------------------------------------------------------
@lru_cache(maxsize=2048)
def chosen_for_config(n: int, width: int, config: int) -> np.ndarray:
    """
    Return n rows selected deterministically by given config.
    Completely cached.
    """
    n = int(n)
    width = int(width)
    config = int(config)

    device_combs = _get_device_combs(width)

    rng = np.random.RandomState(config)
    idx = rng.choice(len(device_combs), size=n, replace=False)
    return device_combs[idx]


# ----------------------------------------------------------------------
# 3) Factory: returns function(seed) → chosen impurity rows
# ----------------------------------------------------------------------
def possible_combs(n: int, width: int):
    n = int(n)
    width = int(width)

    def combs_for_seed(seed: int):
        return chosen_for_config(n, width, seed)

    return combs_for_seed


# ----------------------------------------------------------------------
# 4) Device builder
# ----------------------------------------------------------------------
def unidevice(w, d, t, e, size, config, n, numberofunitcell, combs_fn=None):
    """
    Build a device Hamiltonian unit with impurities inserted at positions
    determined by 'config'. Uses vectorised assignments where possible.
    """

    size = int(size)

    
    if combs_fn is None:
        combs_fn = possible_combs(int(n), int(size))

    # Chosen impurity coordinates
    imps = combs_fn(int(config))
    x = imps[:, 0]
    y = imps[:, 1]

    z = int(numberofunitcell)

    # Base Hamiltonian (already cached inside your unitcell_leads)
    mat = unitcell(w, d, t, e, int(size))

    # Identify impurities in unitcell z
    mask = (x == z)
    if not np.any(mask):
        return mat  # fast path return

    imp_indices = y[mask]

    # Vectorised diagonal modification
    diag_val = (w + 1j*d - 0.5)
    mat[imp_indices, imp_indices] = diag_val

    return mat




def import_leads(size):
    arr = np.load(os.path.expanduser(f"~/Desktop/backup/agnr/size_{size}/leads_{size}.npy"))
    #print(arr.shape)
    return arr

g_7 = import_leads(7)
print(g_7.shape)
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

    Gnonlocal = left @ rho @ IL
    GNON = Gnonlocal - Gnonlocal.conj().T

    # 6. Mathematica Trace Calculation:
    term1 = gdd @ rho @ grr @ rho
    term2 = rho @ GNON @ rho @ GNON

    tr1 = np.abs(np.trace(term1 - term2))

    return np.abs(tr1)


print(device_transmission(1,1e-5,1,0,7,1,0))

def transmission(config, conc, size):
    trans = [device_transmission(i, 1e-5, 1, 0,size,config,conc) for i in np.arange(0,3,0.01)]

    return trans



def run_transmission(args):
    config, conc, size = args
    return transmission(config, conc, size)   # must return shape (300,1)


import numpy as np
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ----------------------------------------------------------
# 1. Worker function — runs inside each process
# ----------------------------------------------------------
def compute_for_one_config(args):
    cfg, conc, size = args

    w_vals = np.arange(0, 3, 0.01)   # 300 energies
    out = np.zeros(len(w_vals), dtype=float)

    # Each worker loops over energies (RAM safe)
    for i, w in enumerate(w_vals):
        out[i] = device_transmission(w, 1e-5, 1, 0, size, cfg, conc)

    return out


# ----------------------------------------------------------
# 2. Parallel execution for ONE concentration
# ----------------------------------------------------------
def compute_for_concentration(conc, size, nconfigs, out_dir):

    print(f"\n[INFO] Starting conc = {conc}")
    os.makedirs(out_dir, exist_ok=True)

    args = [(cfg, conc, size) for cfg in range(nconfigs)]

    # On Ubuntu: fork → memory is shared until modified (COW)
    with Pool(processes=cpu_count()) as pool:
        for cfg, result in enumerate(
            tqdm(pool.imap(compute_for_one_config, args),
                 total=nconfigs,
                 desc=f"conc {conc}")
        ):
            # Save each config immediately → prevents RAM buildup
            np.save(os.path.join(out_dir, f"7_agnr_conc{conc}_cfg{cfg}.npy"), result)


# ----------------------------------------------------------
# 3. Main driver
# ----------------------------------------------------------
def main():
    concs = np.arange(2, 100, 2)  # [1,3,5,...,49]
    nconfigs = 10000
    size = 7

    

    #print(f"[INFO] Output directory: {out_dir}")
    print(f"[INFO] Available CPU cores: {cpu_count()}")

    for conc in concs:
        out_dir = os.path.expanduser(f"~/transmission_results/size_7_{conc}")
        os.makedirs(out_dir, exist_ok=True)
        compute_for_concentration(conc, size, nconfigs, out_dir)

    print("\n[DONE] All concentrations complete!")


# ----------------------------------------------------------
# 4. IMPORTANT: Ubuntu does NOT require __main__
# but it's still considered best practice.
# ----------------------------------------------------------
if __name__ == "__main__":
    main()
