#
# Solve -u`` + k*u = f in (-1, 1) with u(-1) = u(1) = 0   [1]
# 

from sympy import Symbol
from lega.shen_basis import mass_matrix, stiffness_matrix, load_vector
from lega.legendre_basis import ForwardLegendreTransformation as FLT
import scipy.sparse.linalg as la


def get_rhs(u, k):
    '''
    Verify that u satisfies boundary conditions and compute the right hand
    side f.
    '''
    x = Symbol('x')
    assert abs(u.subs(x, -1)) < 1E-15 and abs(u.subs(x, 1)) < 1E-15 
    # Right hand side if u is to be the solution
    f = -u.diff(x, 2) + k*u

    return f


def solve_helmholtz_1d(f, k, n):
    '''Solve the Helmoholtz problem by N Shen polynomials.'''
    A = stiffness_matrix(n)
    M = mass_matrix(n)
    # The linear system of lhs of Helmoholtz is
    AA = (A + k*M)

    F = FLT(n+2)(f)
    bb = load_vector(F)

    # The system is (A + k*M)*U = bb
    U = la.spsolve(AA, bb)

    # Note that x is a vector of expansion coeffs of the solution w.r.t to
    # the Shen basis
    return U

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import cos, pi, lambdify
    from lega.shen_basis import shen_function, legendre_to_shen_matrix
    from lega.legendre_basis import mass_matrix as L_mass_matrix
    from sympy.plotting import plot
    from sympy.mpmath import quad
    from math import sqrt
    
    # Setup
    x = Symbol('x')
    u = (x**2-1)*cos(2*pi*x)
    k = 1
    f = get_rhs(u, k)

    n_max = 30
    # Representation of exact solution in the Legendre basis
    u_leg = FLT(n_max+1)(u)

    n = 2
    tol = 1E-14
    converged = False
    while not converged:
        U = solve_helmholtz_1d(f, k, n)  # w.r.t to shen

        # Error using symobolic functions
        uh = shen_function(U)
        # Want L2 norm of the error
        e = u - uh
        error = sqrt(quad(lambdify(x, e**2), [-1, 1]))

        # Error using representation w.r.t to Shen basis and the mass matrix
        # Turn U from shen to Legendre
        U_leg = legendre_to_shen_matrix(n+2).T.dot(U)
        # Subract on the subspace
        e_ = u_leg[:n+2] - U_leg
        # Legendre mass matrix computes the L2 error
        error_ = sqrt(e_.dot(L_mass_matrix(n+2).dot(e_)))

        print 'n=%d, |e|_2=%.4E  {e}_2=%.4E' % (n, error, error_)

        converged = error < tol or n >= n_max
        n += 1

    # Plot the final numerical one againt analytical
    p0 = plot(u, (x, -1, 1), show=False)
    p1 = plot(uh, (x, -1, 1), show=False)
    p1[0].line_color='red'
    p0.append(p1[0])
    p0.show()
