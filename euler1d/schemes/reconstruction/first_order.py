'''
Piecewise-constant (first-order Godunov) reconstruction.

Same interface as WENO5 / WENO5Z:  Q (3, N+6) -> (QL, QR), each (3, N+1).
Useful as the classical counterpart of NFV's 2-cell stencil.
'''
import numpy as np


def FirstOrder(Q, dx=1.0):
    n_cell = Q.shape[1] - 6
    return Q[:, 2:n_cell + 3].copy(), Q[:, 3:n_cell + 4].copy()
