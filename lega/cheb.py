import numpy as np
from scipy.sparse import diags
import scipy.linalg as dla
import scipy.sparse.linalg as sla

def stiff(N, output):
    '''Shen-Cheb-Dirichlet polynomials' stiffness matrix.'''
    k = np.arange(N)

    if output == 'sparse' or output == 'dense':
        aij = [-2*np.pi*(k+1)*(k+2)]
        for i in range(2, N, 2):
            aij.append(np.array(-4*np.pi*(k[:-i]+1)))    
        A = diags(aij, range(0, N, 2))

        if output == 'dense':
            A = A.toarray()

    elif output == 'banded':
        A = np.zeros((N, N))
        A[-1, :] = -2*np.pi*(k+1)*(k+2)
        for i in range(2, N, 2):
            A[-i-1, i:] = -4*np.pi*(k[:-i]+1)

    return A

A = stiff(4, 'dense')

print A
P, L, U = dla.lu(A)

print P
print L
print U
