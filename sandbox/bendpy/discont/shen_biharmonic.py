#
# Biharmonic
# 

from __future__ import division
from sympy import Symbol, lambdify, sin
import lega.biharmonic_clamped_basis as shen
import scipy.sparse.linalg as la
from sympy.mpmath import quad
import numpy as np


def solve_shen(g, h, n):
    # Mat
    A = shen.bending_matrix(n)

    # The f is zero on -1, 0 so the integration is a bit spacial...
    x = Symbol('x')
    b = np.array([quad(lambdify(x, g*v), [-1, 0]) for v in shen.shen_cb_basis(n)])
    b += np.array([quad(lambdify(x, h*v), [0, 1]) for v in shen.shen_cb_basis(n)])

    # The system is (A + k*M)*U = bb
    U = la.spsolve(A, b)

    # Note that x is a vector of expansion coeffs of the solution w.r.t to
    # the Shen basis
    return U

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import S, sin, exp, nsimplify
    from dg_shen import solve_biharmonic
    from sympy.plotting import plot
    from math import sqrt, log

    x = Symbol('x')
    # g, h = S(1), S(2)     # L^2 --> u in H^4 shen is order two
    # g, h = S(1), 1+x      # H^1 --> u in H^5 shen is order three
    # g, h = -x**2/2 + x/4 + 3/4, -x**2 + x/4 + 3/4    # H^2 --> u in H^6
    # g, h = sin(5*x)*exp(x), sin(5*x)*exp(x)   # C^infty --> spectral


    for g, h in [(S(1), S(2)),
                 (S(1), 1+x),
                 (-x**2/2 + x/4 + 3/4, -x**2 + x/4 + 3/4 ),
                 (sin(5*x)*exp(x), sin(5*x)*exp(x))]:
        u0, u1 = solve_biharmonic(g, h)

        print 'g', g, '\th', h
        print 'u0', nsimplify(u0), '\tu1', nsimplify(u1)

        # Bring to [0, pi] domain
        n = 2
        while n < 31:
            U = solve_shen(g, h, n)
            uh = shen.shen_cb_function(U)

            e0 = quad(lambdify(x, (uh - u0)**2), [-1, 0])
            e1 = quad(lambdify(x, (uh - u1)**2), [0, 1])
            e = sqrt(e0 + e1)
            
            if n != 2:
                print n, e, log(e/e_)/log(n_/n)

            e_ = e
            n_ = n

            n += 2
        print 

    if False:
        # Plot the final numerical one againt analytical
        p0 = plot(uh, (x, -1, 1), show=False)
        p1 = plot(u0, (x, -1, 0), show=False)
        p2 = plot(u1, (x, 0, 1), show=False)
        p1[0].line_color='red'
        p2[0].line_color='red'
        p0.append(p1[0])
        p0.append(p2[0])
        p0.show()
