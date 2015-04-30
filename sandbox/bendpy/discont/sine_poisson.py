#
# Solve -u`` = f in (0, pi) with u(0) = u(pi) = 0   [1]
# 

from __future__ import division
from sympy import Symbol, lambdify, sin
import lega.sine_basis as sines
import scipy.sparse.linalg as la
from sympy.mpmath import quad
from math import pi, sqrt
import numpy as np


def solve_sines(g, h, n):
    # Mat
    A = sines.stiffness_matrix(n)*pi/2

    # Take g, h to refference [0, pi]
    x = Symbol('x')
    g, h = g.subs(x, 2/pi*x - 1), h.subs(x, 2/pi*x - 1)
    # The f is zero on -1, 0 so the integration is a bit spacial...
    b = np.array([quad(lambdify(x, g*v), [0, pi/2]) for v in sines.sine_basis(n)])
    b += np.array([quad(lambdify(x, h*v), [pi/2, pi]) for v in sines.sine_basis(n)])
    b *= 2/pi

    U = la.spsolve(A, b)

    # Note that x is a vector of expansion coeffs of the solution w.r.t to
    return U

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import S, sin, exp, nsimplify
    from lega.sine_basis import sine_function
    from dg_shen import solve_poisson
    from sympy.plotting import plot
    from math import sqrt, log

    x = Symbol('x')
    # g, h = S(1), S(2)     # L^2 --> u in H^2 shen is order two
    # g, h = S(1), 1+x      # H^1 --> u in H^3 shen is order three
    # g, h = -x**2/2 + x/4 + 3/4, -x**2 + x/4 + 3/4    # H^2 --> u in H^4
    # g, h = sin(5*x)*exp(x), sin(5*x)*exp(x)   # C^infty --> spectral
    
    for g, h in [(S(1), S(2)),
                 (S(1), 1+x),
                 (-x**2/2 + x/4 + 3/4, -x**2 + x/4 + 3/4 ),
                 (sin(5*x)*exp(x), sin(5*x)*exp(x))]:
        u0, u1 = solve_poisson(g, h)

        print 'g', g, '\th', h
        print 'u0', nsimplify(u0), '\tu1', nsimplify(u1)

        # Bring to [0, pi] domain
        n = 2
        while n < 257:
            U = solve_sines(g, h, n)
            uh = sine_function(U)

            # Take solution to [-1, 1]
            uh = uh.subs(x, (pi*x + pi)/2)
            e0 = quad(lambdify(x, (uh - u0)**2), [-1, 0])
            e1 = quad(lambdify(x, (uh - u1)**2), [0, 1])
            e = sqrt(e0 + e1)
            
            if n != 2:
                print n, e, log(e/e_)/log(n_/n)

            e_ = e
            n_ = n

            n *= 2
        print 

    if False:
        # Plot the final numerical one againt analytical
        p0 = plot(uh, (x, -1, 1), show=False)
        p1 = plot(u0, (x, -1, 0), show=False)
        p2 = plot(u1, (x, 0, 1), show=False)
        p3 = plot(g, (x, -1, 0), show=False)
        p4 = plot(h, (x, 0, 1), show=False)
        p1[0].line_color='red'
        p2[0].line_color='red'
        p3[0].line_color='green'
        p4[0].line_color='green'
        p0.append(p1[0])
        p0.append(p2[0])
        p0.append(p3[0])
        p0.append(p4[0])
        p0.show()
