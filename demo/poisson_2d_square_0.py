#
# Solve -laplace(u) = f in (-1, 1)^2 with T(u) = 0   [1]
# 

from sympy import symbols, integrate
from lega.shen_basis import mass_matrix, stiffness_matrix, load_vector
from lega.legendre_basis import ForwardLegendreTransformation as FLT
import scipy.linalg as la
import numpy as np


def get_rhs(u):
    '''
    Verify that u satisfies boundary conditions and compute the right hand
    side f.
    '''
    x, y = symbols('x, y')
    assert integrate(abs(u.subs(x, -1)), (y, -1, 1)) < 1E-15
    assert integrate(abs(u.subs(x, 1)), (y, -1, 1)) < 1E-15
    assert integrate(abs(u.subs(y, -1)), (x, -1, 1)) < 1E-15
    assert integrate(abs(u.subs(y, 1)), (x, -1, 1)) < 1E-15

    # Right hand side if u is to be the solution
    f = -u.diff(x, 2) - u.diff(y, 2)

    return f


def solve_poisson_2d(f, n):
    '''Solve the Poisson problem by nxn Shen polynomials.'''
    A = stiffness_matrix(n)
    M = mass_matrix(n)
    
    F = FLT([n+2, n+2])(f)
    b = load_vector(F)      # nxn matrix
    
    # Solve the problem by tensor product solver
    lmbda, V = la.eigh(A.toarray(), M.toarray())

    # Map the right hand side to eigen space
    bb = (V.T).dot(b.dot(V))

    # Apply the inverse in eigen space
    U_ = np.array([[bb[i, j]/(lmbda[i] + lmbda[j])
                    for j in range(n)]
                    for i in range(n)])
    # Map back to physical space
    U = (V).dot(U_.dot(V.T))

    return U

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import sin, pi, lambdify
    from lega.shen_basis import shen_function, legendre_to_shen_matrix
    from lega.legendre_basis import mass_matrix as L_mass_matrix
    from sympy.plotting import plot3d
    from sympy.mpmath import quad
    from math import sqrt
    
    # Setup
    x, y = symbols('x, y')
    u = (x**2-1)*sin(2*pi*y)
    f = get_rhs(u)

    n_max = 30
    # Representation of exact solution in the Legendre basis
    u_leg = FLT([n_max+2, n_max+2])(u)

    n = 2
    tol = 1E-14
    converged = False
    while not converged:
        U = solve_poisson_2d(f, n)  # w.r.t to shen

        #TODO: should add symbolic as well, just here and only for comparison!
        # Error using representation w.r.t to Shen basis and the mass matrix
        # Turn U from shen to Legendre
        Tmat = legendre_to_shen_matrix(n+2)
        U_leg = Tmat.T.dot(U.dot(Tmat.toarray()))   # n+2 x n . n x n . n x n+2
        # Subract on the subspace
        E = u_leg[:n+2, :n+2] - U_leg
        # Legendre mass matrix computes the L2 error
        M = L_mass_matrix(n+2)
        error = sqrt(np.trace((M.dot(E)).dot(M.dot(E.T))))

        print 'n=%d {e}_2=%.4E' % (n, error)

        converged = error < tol or n > n_max-1
        n += 1

    # Plot the symbolic error
    uh = shen_function(U)
    e = u - uh

    plot3d(e, (x, -1, 1), (y, -1, 1))
