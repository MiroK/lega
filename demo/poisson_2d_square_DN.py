#
# Solve -laplace(u) = f in (-1, 1)^2
#         with T(u) = g on y = 1 or -1 and 
#        grad(u).n = h on x = 1 or -1
#
# Several new things are illustrated:
#  (i) the Dirichlet boundary conditions are lifted onto the right hand side
#  (ii) the function space of solution is a tensor product for Legendre in
#       x direction and Shen in y direction
#  (iii) there is a boundary contribution to the right hand side

from __future__ import division
from sympy import symbols, integrate, sin, pi, Number, cos
import lega.shen_basis as shen
from lega.legendre_basis import ForwardLegendreTransformation as FLT
import lega.legendre_basis as leg
import scipy.linalg as la
from math import sqrt
import numpy as np


def get_problem(u=None):
    '''
    We chose a solution which meats Dirichlet bcs and here f, h and the lifted
    g is computed
    '''
    x, y = symbols('x, y')
    if u is None:
        u = (1+y)/2 + sin(2*pi*(y**2-1)*x**2)

    # Check the bcs
    assert integrate(abs(u.subs(y, -1)), (x, -1, 1)) < 1E-15
    assert integrate(abs(u.subs(y, 1)-1), (x, -1, 1)) < 1E-15

    # The numerical solution will be u - u0, where u0 handles the inhomog.
    # Neumann
    u0 = (1+y)/2
    # Fluxes -grad(u).n for Neumann
    h_left = -1*u.diff(x).subs(x, -1)
    h_right = 1*u.diff(x).subs(x, 1)
    # The rhs, -laplace(u0) = 0!
    f = -u.diff(x, 2) - u.diff(y, 2)

    return {'u': u, 'u0': u0, 'f': f, 'hl': h_left, 'hr': h_right}


def solve_poisson_2d_DN(problem, n):
    '''
    Solve the Dirichlet-Neumann problem with n polynomials.
    '''
    n_shen = n     # highest degree is n+1
    n_leg = n+2    # also here
    
    # Get matrices for the basis and solve their eigenvalue problems
    As = shen.stiffness_matrix(n_shen)
    Ms = shen.mass_matrix(n_shen)
    # The eigenvalue problem is symmetric and positive definite
    lmbda_s, Vs = la.eigh(As.toarray(), Ms.toarray())

    Al = leg.stiffness_matrix(n_leg)
    Ml = leg.mass_matrix(n_leg)
    # The eigenvalue problem is only positive, We first solve withough problem
    # without the kernel of constants and then add its contribution to
    # eigenvectors and the spectrum
    lmbda_l1, Vl1 = la.eigh(Al.toarray()[1:, 1:], Ml.toarray()[1:, 1:])
    lmbda_l = np.concatenate([np.array([0]), lmbda_l1])
    Vl = np.eye(n_leg)/sqrt(2.)   # Note that the column/eigenvector is normalized
    Vl[1:, 1:] = Vl1

    # The volume term in the right-hand side:
    flt_2d = FLT([n_leg, n_leg])
    F = flt_2d(problem['f'])
    # Perform the 2d integration
    F = np.array([shen.load_vector(row) for row in F])
    F = Ml.dot(F)
    assert F.shape == (n_leg, n_shen)

    # The surface term from Neumann
    x, y = symbols('x, y')
    # Evaluation is written such that for 1d it expects function of x so it has
    # to be tricked
    flt_1d = FLT([n_leg])
    Hl = flt_1d(problem['hl'].subs(y, x))
    Hl = shen.load_vector(Hl)
    assert Hl.shape == (n_shen, )

    # Okay so now we have performed the integration in the y-direction. But the
    # whole story is that there is also L_k(-1) which would premultiply each
    # row
    Hl = np.vstack([(-1)**row*Hl for row in range(n_leg)])
    assert Hl.shape == (n_leg, n_shen)

    # Repeat for right
    Hr = flt_1d(problem['hr'].subs(y, x))
    Hr = shen.load_vector(Hr)
    assert Hr.shape == (n_shen, )
    # The right edge has x=1 so and all L_k(1) == 1
    Hr = np.vstack([Hr for row in range(n_leg)])
    assert Hr.shape == (n_leg, n_shen)

    F += Hl + Hr

    # To solve the problem we take the rhs to eigenspace
    F_hat = (Vl.T).dot(F.dot(Vs))
    # Solve the problem in the eigenspace
    U_hat = np.array([[F_hat[i, j]/(lmbda_l[i] + lmbda_s[j])
                       for j in range(n_shen)]
                       for i in range(n_leg)])
    # Come back to physical space
    U = Vl.dot(U_hat.dot(Vs.T))

    # For error computation and also to add the lifted Dirichlet bcs it is
    # necessary to go from Shen to Legendre
    toLegendre = shen.legendre_to_shen_matrix(n_leg)
    U = U.dot(toLegendre.toarray())

    # Finally add the representation of the lifted bcs
    U += flt_2d(problem['u0'])

    return U

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from lega.legendre_basis import legendre_function
    from sympy.plotting import plot3d
    from sympy import symbols

    x, y = symbols('x, y')
    problem = get_problem()  # u=((1+y)/2)**4 + sin(pi*y)*(cos(pi*x)+x))
    # Exact
    u = problem['u']

    n_max = 30
    u_leg = FLT([n_max+2, n_max+2])(u)

    n = 2
    converged = False
    tol = 1E-12
    while not converged:
        # numeric
        U_leg = solve_poisson_2d_DN(problem, n=n)
        
        # Subract on the subspace
        E = u_leg[:n+2, :n+2] - U_leg
        
        # Legendre mass matrix computes the L2 error
        M = leg.mass_matrix(n+2)
        error = sqrt(np.trace((M.dot(E)).dot(M.dot(E.T))))

        print 'n=%d, (mass)e=%.4E' % (n, error)

        # Mass matrix L2 is used for stopping
        converged = error < tol or n > n_max-1
        n += 1

    # Plot the error
    uh = legendre_function(U_leg)
    plot3d(u - uh, (x, -1, 1), (y, -1, 1))
