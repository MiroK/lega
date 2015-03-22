#
# Solve -laplace(u) = f in (0, 2*pi)x(-1, 1)
#         with T(u) = 0 on y = -1 and y = 1
#         and periodicity in the x direction
# 
# We shall combine Fourier and Shen basis

from sympy import symbols, integrate, pi, lambdify, Number
from numpy.polynomial.legendre import leggauss 
import scipy.sparse.linalg as sparse_la
import lega.fourier_basis as fourier
import lega.shen_basis as shen
import lega.legendre_basis as leg
from lega.common import tensor_product, function
from lega.legendre_basis import forward_transformation_matrix as FLT
from lega.legendre_basis import backward_transformation_matrix as BLT
from itertools import product
from sympy.mpmath import quad
import numpy as np


def get_rhs(u):
    '''
    Verify that u satisfies boundary conditions and compute the right hand
    side f.
    '''
    # Verify that bcs might hold
    x, y = symbols('x, y')
    assert integrate(abs(u.subs(y, -1)), (x, -1, 1)) < 1E-15
    assert integrate(abs(u.subs(y, 1)), (x, -1, 1)) < 1E-15
    assert quad(lambdify(y, abs(u.subs(x, 0) - u.subs(x, 2*pi))), [-1, 1]) < 1E-15

    # Right hand side if u is to be the solution
    f = -u.diff(x, 2) - u.diff(y, 2)

    return f


def solve_poisson(f, n_fourier, n_shen, output):
    '''
    Solve the Poisson problem with highest frequency n_fourier and n_shen 
    polynomials (that is n_shen+1 is the highest degree in that basis).
    '''
    # Preparing the right hand side
    # First points are evaluated at the grid
    x, y = symbols('x, y')
    n, m = 2*n_fourier, n_shen+2
    fourier_points = np.linspace(0, 2*np.pi, n, endpoint=False)
    legendre_points = leggauss(m)[0]
    points = np.array([list(p)
                       for p in product(fourier_points, legendre_points)])

    if isinstance(f, (int, float, Number)):
        F = float(f)*np.ones((n, m))
    else:
        f = lambdify([x, y], f, 'numpy')
        F = f(points[:, 0], points[:, 1]).reshape((n, m))

    # Now the columns which is f evaluated at Fourier points for fixed y at some
    # quadrature points is Fourier transformed
    F_hat = np.array([fourier.fft(col) for col in F.T]).T
    assert F_hat.shape == (2*n_fourier+1, n_shen+2)
    # Now Forward Legendre transform each row
    flt = FLT(m)
    F = np.array([flt.dot(row) for row in F_hat])
    assert F.shape == (2*n_fourier+1, n_shen+2)

    # The system to be solved is (k^2M + A)U = b, where k^2 comes from laplacian
    # acting on the Fourier basis
    # Get the k**2 terms
    kk = fourier.stiffness_matrix(n_fourier)
    ll = fourier.mass_matrix(n_fourier)

    # Get Shen matrices to setup a system to be solved for each wavenumber
    M = shen.mass_matrix(n_shen)
    A = shen.stiffness_matrix(n_shen)

    # The solutions to each linear system make up a row of the matrix of all
    # uknown coefficients
    # Fourier x Shen
    U = np.empty((2*n_fourier+1, n_shen))

    # FIXME: it is nicer if the fourier basis is fully orthonormal, then there
    # is no need for l
    for row, (k, l, b) in enumerate(zip(kk, ll, F)):
        mat = k*M + A
        vec = shen.load_vector(b)/l
        U[row, :] = sparse_la.spsolve(mat, vec)

    # Make a Fourier x Shen function
    if output == 'shen':
        basis = tensor_product([fourier.fourier_basis(n_fourier, 'x'),
                                shen.shen_basis(n_shen, 'y')])
        uh = function(basis, U.flatten())
        return uh
    # Make a Fourier x Legendre function
    else:
        # Transform rows of U to Legendre
        toLeg = shen.legendre_to_shen_matrix(n_shen+2).toarray()
        U = U.dot(toLeg)

        if output == 'legendre':
            basis = tensor_product([fourier.fourier_basis(n_fourier, 'x'),
                                    leg.legendre_basis(n_shen+2, 'y')])
            uh = function(basis, U.flatten())
            return uh

        else:
            # FIXME: does not work yet
            # For pointvalues of the function need to ifft the columns and
            # blt rows
            U = np.array([fourier.ifft(col) for col in U.T]).T
            blt = BLT(m)
            U = np.array([blt.dot(row) for row in U])

            return points, U

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import sin, cos, Expr

    x, y = symbols('x, y')
    # Easy 
    u = sin(2*x)*(y**2-1)
    # Harder for Shen
    # u = sin(pi*y)*cos(3*x)
    f = get_rhs(u)

    n_fourier = 4
    n_shen = 4
    uh = solve_poisson(f=f, n_fourier=n_fourier, n_shen=n_shen,
                       output='shen')

    # Sympy plotting
    if isinstance(uh, Expr):
        from sympy.plotting import plot3d

        plot3d(u, (x, 0, 2*pi), (y, -1, 1), title='$u$')
        plot3d(uh, (x, 0, 2*pi), (y, -1, 1), title='$u_h$')
        plot3d(u - uh, (x, 0, 2*pi), (y, -1, 1), title='$e$')
    # Matplotlib
    # FIXME: does not work yet
    else:
        import matplotlib.pyplot as plt

        points, Uh = uh

        # Compute point values of exact solution
        n, m = 2*n_fourier, n_shen+2
        u = lambdify([x, y], u, 'numpy')
        U = u(points[:, 0], points[:, 1]).reshape((n,  m))

        print U-Uh

        # Get ready for plotting 
        X = points[:, 0].reshape((n, m))
        Y = points[:, 1].reshape((n, m))

        plt.figure()
        plt.pcolor(X, Y, U-Uh)
        plt.show()


