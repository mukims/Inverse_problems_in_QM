import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
from scipy.sparse import diags
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
from functools import lru_cache
from scipy.linalg import solve



def beta_matrix(w, d, t, e, m):
    dim = 2 * m

    w = w[:,None,None]

    base_val = (w + 1j*d)

    base = base_val * np.eye(2*m)[None,:,:]


    idy = np.arange(0,2*m-1,1)

    base[:,idy,idy+1] = t
    base[:,idy+1,idy] = t

    idx = np.arange(0,m,2)

    base[:,idx,2*m-1-idx] = t
    base[:,2*m-1-idx,idx] = t


    return base 

def T1_matrix(t, m):
    dim = 2 * m
    T = np.zeros((dim, dim), dtype=np.complex128)

    n = np.arange(1, (m - 1)//2 + 1)
    i = 2*n - 1           # to 0-based
    j = 2*m - 2*n

    T[i, j] = t
    return T



def leads_vectorized(w_vals, d, t, e, n, tol=1e-6, max_iter=200000):
    B = len(w_vals)
    unit_batch = beta_matrix(w_vals, d, t, e, n)
    
    # g = inv(unit) using solve (much faster than LU)
    eye_dim = np.eye(unit_batch.shape[-1])
    g = np.stack([solve(unit_batch[i], eye_dim) for i in range(B)])
    
    G = g.copy()
    hopp = T1_matrix(t, n)
    hoppT = hopp.T
    dim = g.shape[-1]
    iden = np.eye(dim)[None,:,:]

    diff = np.inf
    count = 0
    pbar = tqdm(total=max_iter, desc="Dyson iteration", leave=True)

    while diff > tol and count < max_iter:
        A = iden - g @ hoppT @ G @ hopp

        G_new = np.stack([solve(A[i], g[i]) for i in range(B)])

        diff = np.max(np.abs(G_new - G))
        G = G_new
        count += 1
        
        pbar.update(1)
        pbar.set_postfix({"diff": diff})

    pbar.close()
    return G, count



def compute(n):
    w_vals = np.arange(0, 3, 0.01)

    G_all, iterations = leads_vectorized(
        w_vals=w_vals,
        d=1e-3,
        t=1,
        e=0,
        n=n
    )


    return G_all

def connection(t,m):
    idx = np.arange(2,m,2)
    base = np.zeros((2*m,2*m),dtype=np.complex64)

    base[2*m - idx,idx - 1] = t 

    return base

def infinite(w,d,t,e,m):

    w = int(w*100)

    left = g21[int(w)]

    right = g21[int(w)]

    Td = connection(t,m)

    iden = np.eye(2*m, dtype=np.complex64)
    
    return np.linalg.solve(iden - left @ Td.T @ right @ Td.T, left)



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



def device_transmission(w, d, t, e,size,config,concentration):
    ene = int(w*100)
    m = 21    



    left = g21[ene]     # left lead surface Green's function
    right = g21[ene]     # right lead

    T  = connection(t,m)

    I = np.eye(2*m, dtype=complex)
    Td = T.conj().T
    

    tin =  T1_matrix(t, m)

    tin_d = tin.T


    combs_fn = possible_combs(concentration,size)

    g_new = left


    for i in range(100):
        
        unit = unidevice(w, d, t, e, size, config, concentration, i, combs_fn=combs_fn)
#        print(unit.shape)
        gd = np.linalg.inv(unit)

        G = np.linalg.solve(I - gd @ tin_d @ g_new @ tin, gd)

        g_new = G


    left_device = g_new

    IL = np.linalg.solve(I - left_device @ Td @ right @ Td, left_device)
    IR = np.linalg.solve(I - right @ Td @ left_device @ Td, right)
    gdd = IL - IL.conj().T
    grr = IR - IR.conj().T

    Gnonlocal = right @ Td @ IL
    GNON = Gnonlocal - Gnonlocal.conj().T

    term1 = gdd @ T @ grr @ Td
    term2 = T @ GNON @ Td @ GNON

    tr1 = np.abs(np.trace(term1 - term2))

    return tr1





