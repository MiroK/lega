#
# Solve laplace(laplace(u)) = f in [-1, 1]^2
#                         u = 0 on the boundary
#                laplace(u) = 0 on the boundary
#

from __future__ import division
from sympy.mpmath import quad
from sympy import pi, lambdify
import lega.sine_basis as sine
import numpy as np

# FIXME: we have a but somewhere ...

def solve_2d(f, n, n_fft, n_quad):
    '''Solve the biharmonic problem by nxn sine polynomials.'''
    B = np.diagonal(sine.bending_matrix(n).toarray())
    A = np.diagonal(sine.stiffness_matrix(n).toarray())

    b = sine.load_vector(f, n=[n, n], n_fft=n_fft, n_quad=n_quad)
    U = np.zeros_like(b)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            U[i, j] = b[i, j]/(B[i] + 2*A[i]*A[j] + B[j])

    return U


def get_rhs(u):
    '''
    Verify that u satisfies boundary conditions and compute the right hand
    side f.
    '''
    
    x, y = symbols('x, y')
    PI = pi.n()
    # Value
    assert quad(lambdify(y, u.subs(x, 0)**2), (0, PI)) < 1E-13
    assert quad(lambdify(y, u.subs(x, PI)**2), (0, PI)) < 1E-13
    assert quad(lambdify(x, u.subs(y, 0)**2), (0, PI)) < 1E-13
    assert quad(lambdify(x, u.subs(y, PI)**2), (0, PI)) < 1E-13

    # Value
    ddu = u.diff(x, 2) + u.diff(y, 2)
    assert quad(lambdify(y, ddu.subs(x, 0)**2), (0, PI)) < 1E-13
    assert quad(lambdify(y, ddu.subs(x, PI)**2), (0, PI)) < 1E-13
    assert quad(lambdify(x, ddu.subs(y, 0)**2), (0, PI)) < 1E-13
    assert quad(lambdify(x, ddu.subs(y, PI)**2), (0, PI)) < 1E-13

    f = ddu.diff(x, 2) + ddu.diff(y, 2)

    return f

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import symbols, sin
    from sympy.plotting import plot3d
    from lega.sine_basis import sine_function, sine_eval, sine_fft
    from lega.integration import Quad2d
    from sympy.mpmath import quad
    from math import sqrt, log as ln
    import matplotlib.pyplot as plt
    import time
    import sys

    sys.setrecursionlimit(100000)
    
    # Setup
    x, y = symbols('x, y')
    u = (x*(x-pi)*y*(y-pi))**2*sin(x)*sin(y)
    f = get_rhs(u)

    # How accurately eval f and error
    n_fft = 1024

    n = 2
    tol = 1E-14
    converged = False
    n_max = 128

    Uvec = sine_eval(N=[n_fft, n_fft], f=u)
    Uk = sine_fft(Uvec)

    Q2 = Quad2d(200)

    ns = []
    errors = []
    while not converged:
        start = time.time()
        Uh_k = solve_2d(f, n, n_fft=0, n_quad=200)
        n_rows, n_cols = Uh_k.shape
        # print '\tsolver', time.time() - start

        # Error in spectral coefs -- should be very small!
        E_k = Uk[:n_rows, :n_cols] - Uh_k
        error = np.linalg.norm(E_k)/n_rows/n_cols
        # print '\terror', time.time() - start
        
        # Power spectrum of error?
        # uh = sine.sine_function(Uh_k)
        # e = u - uh
        # Evec = sine_eval(N=[n_fft, n_fft], f=u)
        # E_k = sine_fft(Evec)
        # error = np.linalg.norm(E_k)/n_rows/n_cols
        
        # Proper L^2
        # uh = sine.sine_function(Uh_k)
        # e = u - uh
        # error = sqrt(Q2(e**2, [0, pi.n()], [0, pi.n()]))

        if n != 2:
            ns.append(n)
            errors.append(error)
            rate = ln(error/error0)/ln(n0/n)
            print 'n=%d, {e}_2=%.4E(%.2f)' % (n, error, rate)

        converged = error < tol or n >= n_max
        error0, n0 = error, n
        n += 1
   
    print 'Over'

    plot3d(e, (x, 0, pi), (y, 0, pi))
    
    # Plot rate
    plt.figure()
    plt.loglog(ns, errors)
    plt.show()
