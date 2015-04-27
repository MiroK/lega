#
# Solve -laplace(u) = f in (0, pi)^2 with T(u) = 0   [1]
# 

from __future__ import division
from sympy import symbols, integrate, pi
from lega.sine_basis import mass_matrix, stiffness_matrix, load_vector
import scipy.linalg as la
import numpy as np


def get_rhs(u):
    '''
    Verify that u satisfies boundary conditions and compute the right hand
    side f.
    '''
    x, y = symbols('x, y')
    assert integrate(abs(u.subs(x, 0)), (y, 0, pi)) < 1E-15
    assert integrate(abs(u.subs(x, pi)), (y, 0, pi)) < 1E-15
    assert integrate(abs(u.subs(y, 0)), (x, 0, pi)) < 1E-15
    assert integrate(abs(u.subs(y, pi)), (x, 0, pi)) < 1E-15

    # Right hand side if u is to be the solution
    f = -u.diff(x, 2) - u.diff(y, 2)

    return f


def solve_poisson_2d(f, n, n_fft):
    '''Solve the Poisson problem by nxn sine polynomials.'''
    A = stiffness_matrix(n)
    b = load_vector(f, n=[n, n], use_fft=True, n_fft=n_fft)
    
    # Solve the problem by tensor product solver
    lmbda = np.diagonal(A.toarray())

    U = np.array([[b[i, j]/(lmbda[i] + lmbda[j])
                   for j in range(n)]
                   for i in range(n)])

    return U

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from lega.sine_basis import sine_function, sine_eval, sine_fft
    from sympy.mpmath import quad
    from math import sqrt, log as ln
    import matplotlib.pyplot as plt
    from sympy import sin
    import time
    
    # Setup
    x, y = symbols('x, y')
    u = x*(x-pi)*sin(2*y)
    f = get_rhs(u)

    # How accurately eval f and error
    n_fft = 1024

    n = 2
    tol = 1E-14
    converged = False
    n_max = 512

    Uvec = sine_eval(N=[n_fft, n_fft], f=u)
    Uk = sine_fft(Uvec)

    ns = []
    errors = []
    while not converged:
        start = time.time()
        Uh_k = solve_poisson_2d(f, n, n_fft=512)
        n_rows, n_cols = Uh_k.shape
        # print '\tsolver', time.time() - start

        # Error in spectral coefs
        E_k = Uk[:n_rows, :n_cols] - Uh_k
        error = np.linalg.norm(E_k)/n_rows/n_cols
        # print '\terror', time.time() - start

        if n != 2:
            ns.append(n)
            errors.append(error)
            rate = ln(error/error0)/ln(n0/n)
            print 'n=%d, {e}_2=%.4E(%.2f)' % (n, error, rate)

        converged = error < tol or n >= n_max
        error0, n0 = error, n
        n *= 2
   
    print 'Over'
    
    # Plot rate
    plt.figure()
    plt.loglog(ns, errors)
    plt.show() 
