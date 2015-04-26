#
# Solve laplace(laplace(u)) = f in [-1, 1]^2
#                         u = 0 on the boundary
#                 grad(u).n = 0 on the boundary
#

from lega.legendre_basis import ForwardLegendreTransformation as FLT
import lega.biharmonic_clamped_basis as shen
import scipy.sparse.linalg as sparse_la
from scipy.sparse import kron
import scipy.linalg as la
import numpy as np


def solve_2d(f, n):
    '''Solve the biharmonic problem by nxn Shen polynomials.'''
    B = shen.bending_matrix(n)
    A = shen.stiffness_matrix(n)
    M = shen.mass_matrix(n)

    # FIXME Figure out the tensor product here. Meanwhile ..,
    mat0 = kron(B, M) 
    mat1 = 2*kron(A, A)
    mat2 = kron(M, B)
    mat = mat0 + mat1 + mat2

    F = FLT([n+4, n+4])(f)
    b = shen.load_vector(F)      # nxn matrix

    # Flat vector
    vec = b.flatten()
    # Flat solution
    U = sparse_la.spsolve(mat, vec)
    # Reshape as matrix
    U = U.reshape((n, n))

    return U

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import sin, cos, pi, lambdify, symbols
    from lega.legendre_basis import mass_matrix as L_mass_matrix
    from sympy.plotting import plot3d
    from sympy.mpmath import quad
    from math import sqrt
    
    # Setup the problem from Shen's paper
    x, y = symbols('x, y')
    u = (sin(2*pi*x)*sin(2*pi*y))**2
    f = 128*pi**4*(cos(4*pi*x)*cos(4*pi*y) - \
                   cos(4*pi*x)*sin(2*pi*y)*sin(2*pi*y) -\
                   cos(4*pi*y)*sin(2*pi*x)*sin(2*pi*x))

    n_max = 50
    # Representation of exact solution in the Legendre basis
    u_leg = FLT([n_max+4, n_max+4])(u)

    n = 2
    tol = 1E-14
    converged = False
    while not converged:
        U = solve_2d(f, n)  # w.r.t to shen

        # Error using representation w.r.t to Shen basis and the mass matrix
        # Turn U from shen to Legendre
        Tmat = shen.legendre_to_shen_cb_matrix(n+4)
        U_leg = Tmat.T.dot(U.dot(Tmat.toarray())) # n+4 x n . n x n . n x n+4
        # Subract on the subspace
        E = u_leg[:n+4, :n+4] - U_leg
        # Legendre mass matrix computes the L2 error
        M = L_mass_matrix(n+4)
        error_ = sqrt(np.trace((M.dot(E)).dot(M.dot(E.T))))

        if False:
            # Symbolic representation of the error
            uh = shen.shen_cb_function(U)
            e = u - uh
            # Get the L2 norm of error, this takes quite some time to compute
            error = sqrt(quad(lambdify([x, y], e**2), [-1, 1], [-1, 1]))
            print 'n=%d e_2=%.4E (mass)e_2=%.4E' % (n, error, error_)
        else:
            print 'n=%d (mass)e_2=%.4E' % (n, error_)

        # Mass matrix L2 is used for stopping
        converged = error_ < tol or n > n_max-1
        n += 1

    uh = shen.shen_cb_function(U)
    e = u - uh
    plot3d(e, (x, -1, 1), (y, -1, 1), title='error')
