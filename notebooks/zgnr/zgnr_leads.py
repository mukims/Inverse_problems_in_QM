import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy as sc
from scipy.sparse import diags, lil_matrix


def unitcell(n,w,d,t,e):

    base = (w + 1j*d -e)*np.ones(n, dtype=np.complex64)
    hoppings = t * np.ones(n-1,dtype=np.complex64)

    unit = diags([base,hoppings,hoppings],[0,-1,1],format='lil')

    return unit.tocsr()


def hopping_znr(n,t):
    dim1 = np.arange(0,n-1,4)
    dim2 = np.arange(3,n,4)
    base = lil_matrix((n, n), dtype=np.complex128)

    base[dim1,dim1+1] = t
    base[dim2,dim2-1] = t

    return base.tocsr()


import numpy as np
import scipy.sparse as sp

from scipy.sparse.linalg import inv, spsolve
from functools import lru_cache


@lru_cache(maxsize=128)
def leads(n, w, d, t, e,return_count=False):
    unit = inv(unitcell(n, w, d, t, e).tocsc())  # already CSC
    hopp = hopping_znr(n, t).tocsc()

    g = unit.copy().tocsc()
    G = unit.copy().tocsc()
    iden = sp.identity(n, dtype=complex, format="csc")
    g_hopp = g @ hopp

    diff = np.inf
    count = 0
    while diff > 1e-6:
        A = (iden - g_hopp @ G @ hopp.T).tocsc()
        G_new = sp.csc_matrix(spsolve(A, g))  # b is CSC
        diff = np.max(np.abs((G_new - G).data)) if G_new.nnz else 0.0
        G = G_new
        count +=1

    #print(count)

    return (G, count) if return_count else G


