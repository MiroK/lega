#
# Solve u```` = f in (0, pi)
#  with  u(0) = u(pi) = 0   [1]
#      u``(0) = u``(pi) = 0
#

from __future__ import division
from sympy import Symbol, integrate
from lega.sine_basis import bending_matrix, load_vector
from lega.sine_basis import sine_eval, sine_fft
import scipy.sparse.linalg as la
from math import pi as Pi
import numpy as np


def solve_biharmonic_1d(f, n):
    '''Solve biharmonic problem with N fourier sine polynomials.'''
    A = np.diagonal(bending_matrix(n).toarray())

    # Try to see how big of an error we make when computing rhs with fft
    # Integrated
    b = load_vector(f, n)

    # The system is (A + k*M)*U = bb
    U = b/A

    # Note that x is a vector of expansion coeffs of the solution w.r.t to
    # the sine basis
    return U


def get_problem(f):
    '''Compute solution of [1] from f.'''
    x = Symbol('x')
    # u4 = f
    u3 = integrate(f, x)
    u2 = integrate(u3, x)
    u1 = integrate(u2, x)
    u = integrate(u1, x) # + ax^3/6 + bx^/2 + cx + d

    mat = np.array([[0, 0, 0, 1],
                    [Pi**3/6, Pi**2/2, Pi, 1],
                    [0, 1, 0, 0],
                    [Pi, 1, 0, 0]])

    vec = np.array([-u.subs(x, 0),
                    -u.subs(x, Pi),
                    -u2.subs(x, 0),
                    -u2.subs(x, Pi)])

    a, b, c, d = np.linalg.solve(mat, vec)
    u += a*x**3/6 + b*x**2/2 + c*x + d

    # Check that it is the solution
    assert abs(u.evalf(subs={x: 0})) < 1E-14
    assert abs(u.evalf(subs={x: Pi})) < 1E-14
    assert abs(u.diff(x, 2).evalf(subs={x: 0})) < 1E-14
    assert abs(u.diff(x, 2).evalf(subs={x: Pi})) < 1E-14
    assert integrate((u.diff(x, 4) - f)**2, (x, 0, Pi)) < 1E-14

    return u, f

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import cos, pi, lambdify, exp, S
    from lega.sine_basis import sine_function
    from sympy.plotting import plot
    from sympy.mpmath import quad
    from math import sqrt, log as ln
    import matplotlib.pyplot as plt
    
    # Setup
    x = Symbol('x')
    f = S(1)  # x*exp(x)*cos(2*pi)
    u, f = get_problem(f)

    n_max = 30

    n = 2
    tol = 1E-14
    converged = False

    ns = []
    errors = []
    while not converged:
        U = solve_biharmonic_1d(f, n)

        # Error using symobolic functions
        # uh = sine_function(U)
        # Want L2 norm of the error
        # e = u - uh
        # error = sqrt(quad(lambdify(x, e**2), [0, Pi]))

        # Error by FFT of the error and parseval
        uh = sine_function(U)
        # Want L2 norm of the error
        e = u - uh
        Evec = sine_eval(f=e, N=2**16)
        e_k = sine_fft(Evec)
        # Use parseval
        error = sqrt(np.sum(e_k**2))

        # Did we get the fourier coefs right?
        # u_vec = sine_eval(f=u, N=2**16)
        # u_k = sine_fft(u_vec)[:n]
        # e_k = u_k - U
        # error = sqrt(np.sum(e_k**2))
        error_ = -1

        if n != 2:
            ns.append(n)
            errors.append(error)
            rate = ln(error/error0)/ln(n0/n)
            print 'n=%d, |e|_2=%.4E(%.2f)  {e}_2=%.4E' % (n, error, rate, error_)

        converged = error < tol or n >= n_max
        error0, n0 = error, n
        n *= 2
    
    plt.figure()
    plt.loglog(ns, errors)

    # Plot the final numerical one againt analytical
    p0 = plot(u, (x, 0, Pi), show=False)
    p1 = plot(uh, (x, 0, Pi), show=False)
    p1[0].line_color='red'
    p0.append(p1[0])
    p0.show()
