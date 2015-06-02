import lega.shen_basis as shen
import lega.legendre_basis as leg
import stokes_matrices as mixed
from scipy.sparse import kron, bmat
from scipy.sparse.linalg import spsolve
from numpy.polynomial.legendre import leggauss
from itertools import product
import numpy as np


def solve_stokes(m_shen, n_leg, f0, f1, mu=1, with_eigvals=False):
    '''
    Solve the Stokes problem:
        
        -mu*div(grad(u)) - grad(p) = (f0, f1)
                            div(u) = 0       in \Omega = [-1, 1]^2
                                 u = 0       on \partial\Omega

    Use m_shen Shen polynomials >= n_leg Legendre polynomials.
    '''

    assert m_shen >= n_leg
    # The system is [[A, B], [B.T, 0]] = [[b], [0]]
    # Setup the left-hand side
    # A
    s_mass = shen.mass_matrix(m_shen)
    s_stiff = shen.stiffness_matrix(m_shen)
    A = kron(s_stiff, s_mass) + kron(s_mass, s_stiff) 
    A = mu*bmat([[A, None], [None, A]])

    # B
    N = mixed.mixed_mass_matrix(m_shen, n_leg, include_constant=False)
    G = mixed.mixed_grad_matrix(m_shen, n_leg, include_constant=False)
    B = bmat([[kron(N, G)], [kron(G, N)]])

    # lhs
    AA = bmat([[A, B], [B.T, None]])

    # Setup the right-hand side
    flt = leg.ForwardLegendreTransformation([m_shen+2, m_shen+2])
    F0 = flt(f0)
    F1 = flt(f1)

    b0 = shen.load_vector(F0)
    b1 = shen.load_vector(F1)
    bb = np.r_[b0.flatten(), b1.flatten(), np.zeros((n_leg-1)**2)]
    
    # Solve the system
    print 'Solving linear system of size', AA.shape
    X = spsolve(AA, bb)

    # Brake into velocity coefficients (U0, U1), and pressure coefficients P
    size = m_shen**2

    U0, U1, P = X[:size], X[size:2*size], X[2*size:]
    # Reshape to matrices
    U0 = U0.reshape((m_shen, m_shen))
    U1 = U1.reshape((m_shen, m_shen))
    P = P.reshape((n_leg-1, n_leg-1))

    # Note that these are coefs of series not point values!
    return U0, U1, P

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import sin, cos, pi, symbols, exp

    x, y = symbols('x, y')
    f0 = sin(x+y)
    f1 = -exp(-x**2 + y**2)

    # FIXME: analytical solution

    m_shen, n_leg = 100, 98
    U0, U1, P = solve_stokes(m_shen=m_shen, n_leg=n_leg, f0=f0, f1=f1)

    # Transform to points values
    # First the Us have to made from Shen polys expansion coefs to Legendre
    # polys expansion coefs
    Tmat = shen.legendre_to_shen_matrix(m_shen+2)
    U0 = Tmat.T.dot(U0.dot(Tmat.toarray()))
    U1 = Tmat.T.dot(U1.dot(Tmat.toarray()))

    # Now that everyhing has been transformed to Legendre do BLT for point
    # values
    blt_V = leg.BackwardLegendreTransformation([m_shen+2, m_shen+2])
    U0 = blt_V(U0)
    U1 = blt_V(U1)

    blt_Q = leg.BackwardLegendreTransformation([n_leg, n_leg])
    # Add the 0 row, col for constants
    P0 = np.zeros((n_leg, n_leg))
    P0[1:, 1:] = P
    P = blt_Q(P0)

    # Finally points
    points = leggauss(m_shen+2)[0]
    points = np.array([list(p)
                       for p in product(points, points)])
    X = points[:, 0]
    X = X.reshape((m_shen+2, m_shen+2))

    Y = points[:, 1]
    Y = Y.reshape((m_shen+2, m_shen+2))

    # Now plot
    import matplotlib.pyplot as plt
    plt.figure()
    # Velocity
    U = np.sqrt(U0**2 + U1**2)
    c = plt.pcolor(X, Y, U)
    plt.quiver(X[::3], Y[::3], U0[::3], U1[::3])
    plt.colorbar(c)
    plt.xlabel('$x$')
    plt.ylabel('$y$')

    plt.figure()
    plt.streamplot(Y[::3], X[::3], U0[::3], U1[::3])
    plt.xlabel('$x$')
    plt.ylabel('$y$')

    # Pressure
    X, Y = X[:n_leg, :n_leg], Y[:n_leg, :n_leg]
    plt.figure()
    c = plt.pcolor(X, Y, P.T)
    plt.contour(X, Y, P.T, 9, colors='k')
    plt.colorbar(c)

    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()
