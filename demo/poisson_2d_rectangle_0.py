#
# Solve -laplace(u) = f in general rectangular domain = [ax, bx] x [ay, by]
#         with T(u) = 0 on the boundary
# 

from sympy import symbols, integrate
from lega.common import reference_mapping, jacobian_matrix
from lega.shen_basis import mass_matrix, stiffness_matrix, load_vector
from lega.legendre_basis import ForwardLegendreTransformation as FLT
import scipy.linalg as la
import numpy as np


def get_rhs(u, domain):
    '''
    Verify that u satisfies boundary conditions and compute the right hand
    side f.
    '''
    ax, bx = domain[0]
    ay, by = domain[1]

    x, y = symbols('x, y')
    assert integrate(abs(u.subs(x, ax)), (y, -1, 1)) < 1E-15
    assert integrate(abs(u.subs(x, bx)), (y, -1, 1)) < 1E-15
    assert integrate(abs(u.subs(y, ay)), (x, -1, 1)) < 1E-15
    assert integrate(abs(u.subs(y, by)), (x, -1, 1)) < 1E-15

    # Right hand side if u is to be the solution
    f = -u.diff(x, 2) - u.diff(y, 2)

    return f


def solve_poisson_2d(f, n, domain):
    '''
    Solve the Poisson problem by nxn Shen polynomials. Function f is pulled
    by to reference domain and this is also where the solution lives. So for
    error evaluation the exact solution must be pullbacked to reference domain.
    On the other hand plotting in physical domain requires transformation
    therein.
    '''
    # We only do the pullback to reference
    chi, _ = reference_mapping(domain)
    Jmat = jacobian_matrix(chi)
    J = Jmat.det()

    # Compute the scaling factors for the grad_x and grad_y terms
    Gx = float(J/(Jmat[0, 0]**2))
    Gy = float(J/(Jmat[1, 1]**2))

    # Only need 1d matrices
    A = stiffness_matrix(n)
    M = mass_matrix(n)
   
    # Pullback f to refference domain
    chi = dict(pair for pair in chi)
    f = f.subs(chi)*J
    F = FLT([n+2, n+2])(f)
    b = load_vector(F)      # nxn matrix
    
    # Solve the problem by tensor product solver
    lmbda, V = la.eigh(A.toarray(), M.toarray())

    # Map the right hand side to eigen space
    bb = (V.T).dot(b.dot(V))

    # Apply the inverse in eigen space
    I = Gx*np.tile(lmbda, (n, 1)).T+ Gy*np.tile(lmbda, (n, 1))
    U_ = bb/I

    # Or element by element
    # U_ = np.array([[bb[i, j]/(Gx*lmbda[i] + Gy*lmbda[j])
    #                for j in range(n)]
    #                for i in range(n)])
    # Map back to physical space
    U = (V).dot(U_.dot(V.T))

    # Note that these are coefs of the solution in reference domain
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

    ax, bx = 0., 2.
    ay, by = 1., 4.
    domain = [[ax, bx], [ay, by]]

    # Degrees are (2, 4) that is shen0 and shen2 so for n=3 done [OKAY]
    # u = (x-ax)*(x-bx)*(y-ay)*(y-by)**3
    u = (x-ax)*(x-bx)*sin(2*pi*(by-y)/(by-ay))
    f = get_rhs(u, domain)
    
    # Error computed in reference domain-pullback the exact solution
    chi, ichi = reference_mapping(domain)
    chi = dict(pair for pair in chi)
    u_pb = u.subs(chi)

    n_max = 20
    # Represent the pulled back solution for legendre
    u_leg = FLT([n_max+2, n_max+2])(u_pb)

    n = 2
    converged = False
    tol = 1E-13
    while not converged:
        # Get the coefs in reference domain and Shen basis
        U = solve_poisson_2d(f, n, domain)  # w.r.t to shen

        # Turn U from shen to Legendre
        Tmat = legendre_to_shen_matrix(n+2)
        U_leg = Tmat.T.dot(U.dot(Tmat.toarray()))
        
        # Subract on the subspace
        E = u_leg[:n+2, :n+2] - U_leg
        # Legendre mass matrix computes the L2 error
        M = L_mass_matrix(n+2)
        error = sqrt(np.trace((M.dot(E)).dot(M.dot(E.T))))

        print 'n=%d, (mass)e=%.4E' % (n, error)

        # Mass matrix L2 is used for stopping
        converged = error < tol or n > n_max-1
        n += 1

    # Map the computed solution to physical domain for plotting
    uh_pb = shen_function(U)
    ichi = dict(pair for pair in ichi)
    uh = uh_pb.subs(ichi)

    e = u - uh
    plot3d(e, (x, ax, bx), (y, ay, by))
