import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs as sp_eigs
from numpy.linalg import eigvals


def mat(N):
    '''Shen-Cheb-Dirichlet polynomials' stiffness matrix.'''
    k = np.arange(N)
    aij = [-2*np.pi*(k+1)*(k+2)]
    for i in range(2, N, 2):
        aij.append(np.array(-4*np.pi*(k[:-i]+1)))    
    A = diags(aij, range(0, N, 2))
    return A

Ns = []
conds = []
for i in range(3, 12):
    N = 2**i
    Ns.append(N)
    
    A = mat(N)
    # lmax = sp_eigs(A, k=3, which='LM', return_eigenvectors=False)
    lmax = eigvals(A.toarray())

    # lmin = sp_eigs(A, k=3, which='SM', return_eigenvectors=False, tol=1E-10)
    lmin = eigvals(A.toarray())

    lmin = np.min(np.abs(lmin))
    lmax = np.max(np.abs(lmax))
    cond = lmax/lmin

    print A.shape[0], ':', lmin, lmax, cond
    conds.append(cond)

import sys
if sys.platform == 'darwin':
    import matplotlib as mpl
    mpl.use('MacOSX')
import matplotlib.pyplot as plt

plt.figure()
plt.loglog(Ns, conds, '--or', label='observed')
plt.loglog(Ns, np.asarray(Ns)**2, 'b', label='rate 2')
plt.legend(loc='best')
plt.xlabel('$N$')
plt.ylabel('$\kappa$')
plt.show()
