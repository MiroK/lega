from sympy import Symbol, integrate, S, sqrt
import lega.legendre_basis as leg
import scipy.sparse.linalg as sparse_la
import scipy.linalg as la
from scipy.sparse import bmat
import numpy as np

# Consider a problem:
#	     -u`` = f    in (-1, 1)        (1)
#   grad(u).n = 0    on the boundary
#
# Posted like this, the problem is singular with z=const functions in the
# nullspace of the bilinear form a(u, v) = (grad(u), grad(v)). Further the
# problem is solvable iff (f, z) = 0 and the uniqueness requires an additional
# constraint. We use (u, z) = 0. With the constraint the problem becomes a
# saddle point problem - see Bochev for LBB analysis of such system. What I do
# here goes as follows:
#
# 1) Set up the saddle point problem and solve the stuff
# 2) Show how the saddle point formulation can be by passed by changing the
# basis. This change yields a symmetric positive definite system
# 3) Show how the original singular problem can be transformed into a well posed
# one

# Nullspace function that is unit in L^2 norm
z = S(1)/sqrt(2)

def get_problem(f, u=None):
    '''Get a solution to (1).'''
    x = Symbol('x')
    # Compute the solution if not given
    if u is None:
        # Make sure we work with zero mean f
        g = f - integrate(f*z, (x, -1, 1))*z
        assert (integrate(g*z, (x, -1, 1))) < 1E-15
        # First integration constant
        du = integrate(-g, x)
        a0 = -du.subs(x, -1)

        # Second integration constant
        u = integrate(du, x) + a0*x
        a1 = integrate(u*z, (x, -1, 1))
        u -= a1*z

        return get_problem(f, u)

    # Check the solution properties
    assert -u.diff(x, 2) - (f - integrate(f*z, (x, -1, 1))*z) == 0
    assert abs(u.diff(x, 1).subs(x, -1)) < 1E-15
    assert abs(u.diff(x, 1).subs(x, 1)) < 1E-15
    assert abs(integrate(u*z, (x, -1, 1))) < 1E-15

    # Everything is okay
    return f, u


def saddle_point_solver(f, n):
    '''
    Solve the constrained problem with n Legendre polynomials.
    '''
    # The system is [[A, B], [B.T, 0]], [[b], 0]
    # with A the stiffness matrix and B has the constaint
    A = leg.stiffness_matrix(n)
    # Constaint matrix is simple due to orthogonality
    B = np.array([np.r_[float(sqrt(2)), np.zeros(n-1)]]).T
    # System matrix
    A = bmat([[A, B], [B.T, None]])

    # Rhs
    F = leg.ForwardLegendreTransformation(n)(f)
    M = leg.mass_matrix(n)
    b = M.dot(F)
    # System rhs
    b = np.r_[b, 0]

    # Solve
    x = sparse_la.spsolve(A, b)
    U, lmbda = x[:-1], x[-1]

    return U, lmbda


def orthogonal_solver(f, n):
    '''
    Recall that the saddle point solver has A = [[0, 0], [0, A1]] where A1 is
    symmetric positive definite and corresponds to basis of Vh = {L_1, L_2, ...},
    that is we leave out L_0 which are constants. Due to orthogonality of
    Legendre basis the solution in Vh is automatically orthogonal to L_0.
    '''
    A = leg.stiffness_matrix(n).toarray()[1:, 1:]
    # The rhs also skips L_0 as well. Note that this is like filtering out the
    # constant component of f
    F = leg.ForwardLegendreTransformation(n)(f)
    M = leg.mass_matrix(n)
    b = M.dot(F)
    b0, b = b[0], b[1:]

    U = la.solve(A, b)
    # We can get the multiplier in pre processing
    lmba = b0*z

    return U, lmba


def projection_solver(f, n, projection='L2'):
    '''The singular system Au=b is projected outside of the nullspace.'''
    A = leg.stiffness_matrix(n).toarray()

    F = leg.ForwardLegendreTransformation(n)(f)
    M = leg.mass_matrix(n)
    b = M.dot(F)

    if projection == 'l2':
        P = np.eye(n)[1:, :]
    else:
        P = leg.mass_matrix(n).toarray()[1:, :]

    PAPT = P.dot(A.dot(P.T))
    Pb = P.dot(b)

    U = la.solve(PAPT, Pb)
    U = P.T.dot(U)
    lmba = b[0]*z

    return U, lmba

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import sin, pi, lambdify
    from sympy.mpmath import quad
    from math import sqrt as m_sqrt

    x = Symbol('x')
    f = x*sin(2*pi*x)
    f, u = get_problem(f=f)
    n = 20
    print 'u', u
    print 'f', f

    # Nullspace vector basis representation
    Z = np.r_[1/m_sqrt(2), np.zeros(n-1)]

    # Saddle point formulation
    # U, ah = saddle_point_solver(f, n)

    # Auto-orthogonal formulation
    # U, ah = orthogonal_solver(f, n)

    # Projection formulation
    U, ah = projection_solver(f, n)
   
    # Extend U properly
    U = np.r_[0, U] if len(U) == n-1 else U
    
    uh = leg.legendre_function(U)
    # Errors
    e = u - uh
    error_H1 = m_sqrt(quad(lambdify(x, e.diff(x, 1)**2), [-1, 1]))
    error_L2 = m_sqrt(quad(lambdify(x, e**2), [-1, 1]))
    print 'e_H1=%.4E, e_L2=%.4E' % (error_H1, error_L2)

    # Orthogonality
    zleg = leg.legendre_function(Z)
    M = leg.mass_matrix(n)
    print '(z, z) %.4E' % abs(quad(lambdify(x, zleg**2), [-1, 1]))
    print 'Z.M.Z', Z.dot(M.dot(Z))
    print '(z, uh)', abs(quad(lambdify(x, uh*zleg), [-1, 1]))
    print 'U.M.Z', U.dot(M.dot(Z))
    
    # Multiplier
    # Analytic
    a = float(integrate(z*f, (x, -1, 1)))
    print 'a=%g, ah=%g, lambda error=%.4E' % (a, ah, abs(a - ah))
