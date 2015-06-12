import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import time


def mass(n_cells):
    '''FEM CG1 mass matrix over [0, 1].'''
    n_vertices = n_cells+1
    h = 1./n_cells
    M_col = np.zeros(n_cells+1)
    M_col[0] = 4
    M_col[1] = 1
    M_col *= h/6.

    return la.toeplitz(M_col)


def stiff(n_cells):
    '''FEM CG1 stiffness matrix over [0, 1].'''
    n_vertices = n_cells+1
    h = 1./n_cells
    A_col = np.zeros(n_cells+1)
    A_col[0] = 2
    A_col[1] = -1
    A_col /= h

    return la.toeplitz(A_col)


def claim_0(n_cells, tol=1E-10):
    '''
    Let V, Lmbda be such that A.V = M.V.Lmbda. Then inv(A) = V.inv(Lmbda).V.T
    '''
    A = mass(n_cells)
    M = stiff(n_cells)
    # The rhs
    b = np.random.rand(A.shape[0])
    # Exact inverse
    x_ = la.solve(A, b)
    # Eig inverse
    lmbda, V = la.eigh(A, M)
    Lmbda = sparse.diags(1./lmbda, 0)
    x = V.dot(Lmbda.dot(V.T.dot(b)))

    return np.linalg.norm(x - x_)/len(x) < tol


def claim_1(n_cells, tol=1E-10, k=2):
    '''
    Let V, Lmbda such that A.V = M.V.Lmbda. Then the problem (A+k*M).v = beta*M.V
    Has the same eigenvectors and it holds that beta = Lmbda + k*I
    '''
    assert k > 0, 'We only consider positive k'
    # Known Poisson
    A = mass(n_cells)
    M = stiff(n_cells)
    lmbda, V = la.eigh(A, M)
    # Helmholtz
    B = (A+k*M)
    beta, W = la.eigh(B, M)
    # Check match of eigenvalues and eigenvectors
    vec_match = np.linalg.norm(V-W)/len(lmbda) < tol
    val_match = np.linalg.norm(lmbda + k - beta)/len(lmbda) < tol
    
    return vec_match and val_match


def claim_2(n_cells, tol=1E-10, k=20):
    '''
    A consequence of claim 0, 1 is that Bx=b with B=(A+k*M) is invertible
    in the following way.
    '''
    assert k > 0, 'We only consider positive k'

    A = mass(n_cells)
    M = stiff(n_cells)
    # The rhs
    b = np.random.rand(A.shape[0])
    # Exact inverse
    x_ = la.solve(A+k*M, b)
    
    # Eig inverse
    lmbda, V = la.eigh(A, M)
    Lmbda = sparse.diags(1./(lmbda+k), 0)
    x = V.dot(Lmbda.dot(V.T.dot(b)))

    return np.linalg.norm(x - x_)/len(x) < tol


def speed_test_classic(n_cells):
    '''See how much time we would need to solve the system with many rhs.'''
    # Suppose these are cached
    A = mass(n_cells)
    M = stiff(n_cells)

    N = n_cells+1
    # We have N**2 rhs.

    t0 = time.time()
    for k in range(1, N+1): 
        b = np.random.rand(A.shape[0])
        x = spsolve(A+k*M, b)

    # This has computing b in it but that is same for claim_2 as well
    dt = time.time() - t0
    return dt


def speed_test_eigs(n_cells):
    '''See how much time we would need to solve the system with many rhs.'''
    t0 = time.time()
    # Computing eigs is done once
    A = mass(n_cells)
    M = stiff(n_cells)
    lmbda, V = la.eigh(A, M)

    N = n_cells+1
    # Suppose have N rhs. In 3d this whould be N**2
    for k in range(1, N+1): 
        b = np.random.rand(A.shape[0])
        x = V.dot(sparse.diags(1./(lmbda+k), 0).dot(V.T.dot(b)))

    dt = time.time() - t0
    return dt

# -----------------------------------------------------------------------------


if __name__ == '__main__':
    import sys
    if sys.platform == 'darwin':
        import matplotlib as mpl
        mpl.use('MacOSX')
    import matplotlib.pyplot as plt

    test = False 
    if test:
        OKAY = "\033[1;37;32m%s\033[0m" % 'OK'
        FAIL = "\033[1;37;31m%s\033[0m" % 'Nope'

        n_cels = 100
        print '0', OKAY if claim_0(100) else FAIL
        print '1', OKAY if claim_1(100) else FAIL
        print '2', OKAY if claim_2(100) else FAIL

    compute_data = False
    if compute_data:
        ns = [32, 64, 128, 256, 512, 1024]
        data = np.zeros((len(ns), 3))
        for row, n in enumerate(ns):
            data[row, 0] = speed_test_classic(n_cells=n)
            data[row, 1] = speed_test_eigs(n_cells=n)
        data[:, -1] = ns

        np.savetxt('data_eigs_inverse', data)
    else:
        data = np.loadtxt('data_eigs_inverse')

    plt.figure()
    plt.plot(data[:, -1], data[:, 0], label='spsolve')
    plt.plot(data[:, -1], data[:, 1], label='eigs')
    plt.xlabel('Number of cells $N$')
    plt.ylabel('Seconds to solve $N$ systems of size $N$')
    plt.legend(loc='best')

    plt.figure()
    plt.plot(data[:, -1], data[:, 0]/data[:, 1])
    plt.xlabel('Number of cells $N$')
    plt.ylabel('Classic relative to eigs')
    plt.legend(loc='best')

    plt.show()
