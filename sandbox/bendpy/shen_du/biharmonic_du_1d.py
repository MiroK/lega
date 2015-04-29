#
# Solve      u```` = f in (-1, 1)
#            u(-1) = u(1) = 0         [1]
#           u`(-1) = u`(1) = 0
# 
from __future__ import division
from sympy import Symbol, integrate, simplify
import lega.biharmonic_clamped_basis as shen
from lega.legendre_basis import ForwardLegendreTransformation as FLT
import scipy.sparse.linalg as la
from numpy import array
from numpy.linalg import solve


def get_problem(f):
    '''Compute solution of [1] from f.'''
    x = Symbol('x')
    # u4 = f
    u3 = integrate(f, x)
    u2 = integrate(u3, x)
    u1 = integrate(u2, x)
    u = integrate(u1, x) # + ax^3/6 + bx^/2 + cx + d

    mat = array([[-1/6, 1/2, -1, 1],
                 [1/6, 1/2, 1, 1],
                 [1/2, -1, 1, 0],
                 [1/2, 1, 1, 0]])

    vec = array([-u.subs(x, -1),
                 -u.subs(x, 1),
                 -u1.subs(x, -1),
                 -u1.subs(x, 1)])

    a, b, c, d = solve(mat, vec)
    u += a*x**3/6 + b*x**2/2 + c*x + d

    # Check that it is the solution
    assert abs(u.evalf(subs={x: -1})) < 1E-15
    assert abs(u.evalf(subs={x: 1})) < 1E-15
    assert abs(u.diff(x, 1).evalf(subs={x: -1})) < 1E-15
    assert abs(u.diff(x, 1).evalf(subs={x: 1})) < 1E-15
    assert integrate((u.diff(x, 4) - f)**2, (x, -1, 1)) < 1E-15

    return u, f


def solve_1d(f, n):
    '''Solve the problem with n polynomials.'''
    A = shen.bending_matrix(n)

    F = FLT(n+4)(f)
    b = shen.load_vector(F)

    U = la.spsolve(A, b)

    # Note that x is a vector of expansion coeffs of the solution w.r.t to
    # the Shen basis
    return U

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import cos, pi, exp, lambdify
    from lega.legendre_basis import mass_matrix as L_mass_matrix
    from sympy.plotting import plot
    from sympy.mpmath import quad
    from math import sqrt
    
    # Setup
    x = Symbol('x')
    f = x*exp(x)*cos(2*pi)
    u, f = get_problem(f)

    n_max = 30
    # Representation of exact solution in the Legendre basis
    u_leg = FLT(n_max+1)(u)

    n = 2
    tol = 1E-14
    converged = False
    print '\tn\tL^2 integration\tL^2 by mass'
    while not converged:
        U = solve_1d(f, n)  # w.r.t to shen

        # Error using symobolic functions
        uh = shen.shen_cb_function(U)
        # Want L2 norm of the error
        e = u - uh
        error = sqrt(quad(lambdify(x, e**2), [-1, 1]))

        # Error using representation w.r.t to Shen basis and the mass matrix
        # Turn U from shen to Legendre
        U_leg = shen.legendre_to_shen_cb_matrix(n+4).T.dot(U)
        # Subract on the subspace
        e_ = u_leg[:n+4] - U_leg
        # Legendre mass matrix computes the L2 error
        error_ = sqrt(e_.dot(L_mass_matrix(n+4).dot(e_)))

        print '%d\t%.8E\t%.8E' % (n, error, error_)

        converged = error < tol or n >= n_max
        n += 1

    # Plot the final numerical one againt analytical
    p0 = plot(u, (x, -1, 1), show=False)
    p1 = plot(uh, (x, -1, 1), show=False)
    p1[0].line_color='red'
    p0.append(p1[0])
    p0.show()
