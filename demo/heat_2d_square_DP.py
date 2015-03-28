#
# Solve u_t = laplace(u) in (0, 2*pi) x (-1, 1) x (0, T)
#    u(x=0) = u(x=2*pi)
#    u(y=-1) = 0
#    u(y=1) = 0
#    u(t=0) = u0
#
# We shall combine Fourier and Shen basis

from __future__ import division
from sympy import symbols, integrate, pi, lambdify, Number, cos, sin
from numpy.polynomial.legendre import leggauss 
import scipy.sparse.linalg as sparse_la
import scipy.linalg as la
import lega.fourier_basis as fourier
import lega.shen_basis as shen
from lega.common import tensor_product, function
from lega.legendre_basis import forward_transformation_matrix as FLT
from lega.legendre_basis import backward_transformation_matrix as BLT
from itertools import product
from sympy.mpmath import quad
import numpy as np
import time


def get_u0(u0=None):

    x, y = symbols('x, y')

    if u0 is None:
        u0 = sin(pi*y)*cos(x)

    assert u0.subs(y, -1) == 0
    assert u0.subs(y, 1) == 0
    assert (u0.subs(x, 0) - u0.subs(x, 2*pi)) == 0

    return u0


def solve_heat(u0, n_fourier, n_shen, dt=1E-4, T=0.01):
    '''
    Solve the Poisson problem with highest frequency n_fourier and n_shen 
    polynomials (that is n_shen+1 is the highest degree in that basis).
    '''
    # We plan to solve the problem in the basis of eigenvector of Au=aMu where
    # A, M are shen matrices. The time discretization is bit it's
    # innitialization requires a bit of work
    A = shen.stiffness_matrix(n_shen)
    M = shen.mass_matrix(n_shen)
    lmbda_shen, V = la.eigh(A.toarray(), M.toarray())

    # Prepare U0
    # First points are evaluated at the grid
    x, y = symbols('x, y')
    n, m = 2*n_fourier, n_shen+2
    fourier_points = np.linspace(0, 2*np.pi, n, endpoint=False)
    legendre_points = leggauss(m)[0]
    points = np.array([list(p)
                       for p in product(fourier_points, legendre_points)])
    if isinstance(u0, (int, float, Number)):
        U0_vec = float(f)*np.ones((n, m))
    else:
        u0 = lambdify([x, y], u0, 'numpy')
        U0_vec = u0(points[:, 0], points[:, 1]).reshape((n, m))

    # Now the columns which is u0 evaluated at Fourier points for fixed y at some
    # quadrature points is Fourier transformed
    U0 = np.array([fourier.fft(col) for col in U0_vec.T]).T
    # Now Forward Legendre transform each row
    flt = FLT(m)
    U0 = np.array([flt.dot(row) for row in U0])
    # At this points each row is a representation in the legendre basis and
    # needs to be transformed to Shen basis by projection.
    U0 = np.array([sparse_la.spsolve(M, shen.load_vector(row)) for row in U0])
    assert U0.shape == (n+1, n_shen)

    # As far as U0 is concerned, the ONLY extra step for Shen's eigenvectors is
    # to transform each row from normal Shen representation to the eigenvector 
    # representation
    U0_eig = U0.dot(M.dot(V))  # SAME as U0_eig = V.T.dot(M.dot(U0.T)).T

    # Now we build the Crank-Nicolson. This multiples the U0 and that's it for
    # time stepping
    lmbda_fourier = fourier.stiffness_matrix(n_fourier)

    CN = 1 - 0.5*dt*np.tile(lmbda_fourier, (n_shen, 1)).T\
           - 0.5*dt*np.tile(lmbda_shen, (n+1, 1))
    CN /= 1 + 0.5*dt*np.tile(lmbda_fourier, (n_shen, 1)).T\
            + 0.5*dt*np.tile(lmbda_shen, (n+1, 1))
    assert U0.shape == CN.shape

    # FIXME: Here we build a new (2*n_fourier+1, n_shen) matrix. For saving
    # memory we can build a CN_k factor for each lmbda_fourier wavenumber. Then
    # there are two loops time  and within there over fourier wave numbers

    # Time loop
    t = 0
    n_steps = 0
    start = time.time()
    print 'Time integrating...',
    while t < T:
        t += dt
        n_steps +=1
        U0_eig *= CN
    print 'done in %gs [%d steps]' % (time.time() - start, n_steps)
    
    # Take the solution from eigenspace to Shen
    U0 = U0_eig.dot(V.T)
    assert U0.shape == (n+1, n_shen)

    # Take the solution from fourier x shen to fourier x leg and then points
    toLeg = shen.legendre_to_shen_matrix(n_shen+2).toarray()
    U0 = U0.dot(toLeg)

    blt = BLT(m).T
    U0 = np.array([blt.dot(row) for row in U0])
    U0 = np.array([fourier.ifft(col) for col in U0.T]).T

    return points, U0

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n_fourier = 1024
    n_shen = 48
    
    u0 = get_u0()
    points, Uh = solve_heat(u0, n_fourier, n_shen, T=1)


    # Get ready for plotting 
    n, m = Uh.shape
    X = points[:, 0].reshape((n, m))
    Y = points[:, 1].reshape((n, m))

    plt.figure()
    plt.pcolor(X, Y, Uh)
    plt.colorbar()
    plt.xlim((0, 2*np.pi))
    plt.ylim((-1, 1))
    plt.show()
