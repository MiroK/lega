import lega.shen_basis as shen
import lega.legendre_basis as leg
import stokes_matrices as mixed
from scipy.sparse import kron, bmat
from scipy.sparse.linalg import spsolve
from numpy.polynomial.legendre import leggauss
from itertools import product
import numpy as np


def solve_stokes(m_shen, n_leg, f0, f1, mu=1, as_values=True, with_eigvals=False):
    '''
    Solve the Stokes problem:
        
        mu*div(grad(u)) - grad(p) = (f0, f1)
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

    # Optionally transform coefs to point values. Then return the grid so that
    # Plotting is possible
    if not as_values:
        return U0, U1, P
    else:
        # First the Us have to made from Shen polys expansion coefs to Legendre
        # polys expansion coefs
        Tmat = shen.legendre_to_shen_matrix(m_shen+2)
        U0 = Tmat.T.dot(U0.dot(Tmat.toarray()))
        U1 = Tmat.T.dot(U1.dot(Tmat.toarray()))

        # Not that everyhing has been transformed to Legendre do BLT for point
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

        return {'velocity': {'X': X, 'Y': Y, 'U0': U0, 'U1': U1},
                'pressure': {'X': X[:n_leg, :n_leg], 'Y': Y[:n_leg, :n_leg], 'P': P}}

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import sin, cos, pi, symbols

    x, y = symbols('x, y')
    f0 = (x+y)*sin(pi*(x-y))
    f1 = (x-y)*cos(pi*(x+y))

    # Setup analytical solution

    # Sample solve on some (coarse) grid
    res = solve_stokes(m_shen=100, n_leg=98, f0=f0, f1=f1)

    # Final plot
    XV = res['velocity']['X']
    YV = res['velocity']['Y']
    U0 = res['velocity']['U0']
    U1 = res['velocity']['U1']

    XQ = res['pressure']['X']
    YQ = res['pressure']['Y']
    P = res['pressure']['P']

    import matplotlib.pyplot as plt

    plt.figure()
    U = np.sqrt(U0**2 + U1**2)
    plt.pcolor(XV, YV, U)
    plt.quiver(XV, YV, U0, U1)

    plt.figure()
    plt.pcolor(XQ, YQ, P)
    plt.contour(XQ, YQ, P, 9, colors='k')

    plt.show()
