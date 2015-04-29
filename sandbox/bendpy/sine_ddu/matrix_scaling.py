from __future__ import division
from scipy.sparse import kron
from lega.sine_basis import bending_matrix, mass_matrix, stiffness_matrix
import numpy as np


def cond_1d(n):
    '''Condition number of biharmonic operator in 1d.'''
    return np.linalg.cond(bending_matrix(n).toarray())


def cond_2d(n):
    '''Condition number of biharmonic operator in 1d.'''
    B = bending_matrix(n)
    A = stiffness_matrix(n)
    M = mass_matrix(n)

    # FIXME Figure out the tensor product here. Meanwhile ..,
    mat0 = kron(B, M) 
    mat1 = 2*kron(A, A)
    mat2 = kron(M, B)
    mat = mat0 + mat1 + mat2

    return np.linalg.cond(mat.toarray())

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    for i in range(2, 10):
        n = 2**i
        print n, cond_1d(n)

    print

    for i in range(2, 10):
        n = 2**i
        print n, cond_2d(n)
