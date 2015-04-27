#
# Solve -u`` + k*u = f in (0, pi) with u(0) = u(pi) = 0   [1]
# 

from __future__ import division
from sympy import Symbol
from lega.sine_basis import mass_matrix, stiffness_matrix, load_vector
from lega.sine_basis import sine_eval, sine_fft
import scipy.sparse.linalg as la
from math import pi as Pi
import numpy as np


def get_rhs(u, k):
    '''
    Verify that u satisfies boundary conditions and compute the right hand
    side f.
    '''
    x = Symbol('x')
    assert abs(u.subs(x, 0)) < 1E-15 and abs(u.subs(x, Pi)) < 1E-15 
    # Right hand side if u is to be the solution
    f = -u.diff(x, 2) + k*u

    return f


def solve_helmholtz_1d(f, k, n):
    '''Solve the Helmoholtz problem by N Fourier sine polynomials.'''
    A = stiffness_matrix(n)
    M = mass_matrix(n)
    # The linear system of lhs of Helmoholtz is
    AA = (A + k*M)

    # Try to see how big of an error we make when computing rhs with fft
    # Integrated
    b = load_vector(f, n)
    # Try some frequency
    # F = sine_eval(8192, f)
    # bb = sine_fft(F)[:n]
    # print 2**n, '>>>', np.linalg.norm(b-bb)

    # The system is (A + k*M)*U = bb
    U = la.spsolve(AA, b)

    # Note that x is a vector of expansion coeffs of the solution w.r.t to
    # the sine basis
    return U

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import cos, pi, lambdify
    from lega.sine_basis import sine_function
    from sympy.plotting import plot
    from sympy.mpmath import quad
    from math import sqrt, log as ln
    import matplotlib.pyplot as plt
    
    # Setup
    x = Symbol('x')
    u = x*(x-pi)*cos(2*pi*x)
    k = 1
    f = get_rhs(u, k)

    n_max = 30

    n = 2
    tol = 1E-14
    converged = False

    ns = []
    errors = []
    while not converged:
        U = solve_helmholtz_1d(f, k, n)  # w.r.t to shen

        # Error using symobolic functions
        uh = sine_function(U)
        # Want L2 norm of the error
        e = u - uh
        error = sqrt(quad(lambdify(x, e**2), [0, Pi]))

        # Error by FFT
        Evec = sine_eval(f=e, N=2**16)
        e_k = sine_fft(Evec)
        # Use parseval
        error_ = sqrt(np.sum(e_k**2))

        if n != 2:
            ns.append(n)
            errors.append(error)
            rate = ln(error/error0)/ln(n0/n)
            print 'n=%d, |e|_2=%.4E(%.2f)  {e}_2=%.4E' % (n, error, rate, error_)

        converged = error < tol or n >= n_max
        error0, n0 = error, n
        n += 1
    
    plt.figure()
    plt.loglog(ns, errors)

    # Plot the final numerical one againt analytical
    p0 = plot(u, (x, 0, Pi), show=False)
    p1 = plot(uh, (x, 0, Pi), show=False)
    p1[0].line_color='red'
    p0.append(p1[0])
    p0.show()
