import lega.biharmonic_clamped_basis as shen
from scipy.sparse import kron
import numpy as np

def cond_number(n):
    '''Condition number of the 2d biharmonic operator.'''
    B = shen.bending_matrix(n)
    A = shen.stiffness_matrix(n)
    M = shen.mass_matrix(n)

    mat0 = kron(B, M) 
    mat1 = 2*kron(A, A)
    mat2 = kron(M, B)
    mat = mat0 + mat1 + mat2

    return np.linalg.cond(mat.toarray())

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    for n in range(2, 51):
        print n, cond_number(n)
