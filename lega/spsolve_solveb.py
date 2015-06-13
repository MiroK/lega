import numpy as np
from scipy.sparse import diags
import scipy.linalg as dla
import scipy.sparse.linalg as sla
import time


def mat(N, output):
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

def solve_dense(A, b): return dla.solve(A, b)

def solve_banded(A, b): return dla.solve_banded((0, A.shape[0]-1), A, b)

def solve_sparse(A, b): return sla.spsolve(A, b)


Ns = [2**i for i in range(5, 13)]
outputs = ['sparse', 'dense', 'banded']
methods = [solve_sparse, solve_dense, solve_banded]


# Compute data
if True:
    data_time = np.zeros((len(Ns), len(outputs)+1))
    data_time[:, 0] = Ns

    for row, N in enumerate(Ns):
        b = np.random.rand(N)
        for col, (output, method) in enumerate(zip(outputs, methods)):
            A = mat(N, output)

            t0 = time.time()
            x = method(A, b)
            dt = time.time() - t0
            data_time[row, col+1] = dt

    np.savetxt('solvers_time', data_time, header='sparse:dense:banded, time')

import sys
if sys.platform == 'darwin':
    import matplotlib as mpl
    mpl.use('MacOSX')
import matplotlib.pyplot as plt


# Raw
plt.figure()
data = np.loadtxt('solvers_time')
x = data[:, 0]
for col in range(1, data.shape[1]):
    y = data[:, col]
    plt.loglog(x, y, label=outputs[col-1])
plt.legend(loc='best')
plt.ylabel('time')
plt.xlabel('$N$')


# Relative to N0
plt.figure()
data_ = np.copy(data)
for col in range(1, data.shape[1]):
    data_[:, col] /= data_[0, col]

for col in range(1, data_.shape[1]):
    y = data_[:, col]
    plt.loglog(x, y, label=outputs[col-1])

s = 2
plt.loglog(x, x**s, label='rate %d' % s)
plt.legend(loc='best')
plt.ylabel('relative to smallest N')
plt.xlabel('$N$')


# Relative to banded
plt.figure()
for row in data:
    row[1:] /= row[-1]

for col in range(1, data.shape[1]):
    y = data[:, col]
    plt.semilogx(x, y, label=outputs[col-1])
plt.legend(loc='best')
plt.ylabel('relative to banded')
plt.xlabel('$N$')

plt.show()
