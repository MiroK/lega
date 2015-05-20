from __future__ import division
import numpy as np
from scipy.sparse import spdiags
from math import sqrt


def mixed_mass_matrix(m_shen, n_leg):
    '''
    Mass matrix between m_shen-dimensional space of Shen polynomials and
    n_leg-dimensional space of Legendre polynomials.
    '''
    main_len = min(m_shen, n_leg)
    main = -np.array([2/(2*j+1)/sqrt(4*j+6) for j in range(main_len)])

    off_len = min(m_shen, n_leg-2)
    off = np.r_[np.zeros(2),
                np.array([2/(2*(j+2)+1)/sqrt(4*j+6) for j in range(off_len)])]

    if n_leg < 2:
        return spdiags(main, 0, m_shen, n_leg)
    else:
        D = spdiags(main, 0, m_shen, n_leg)
        T = spdiags(off, 2, m_shen, n_leg)
        return D + T


def mixed_grad_matrix(m_shen, n_leg):
    '''Matrix such that mat(i, j) = (L(j), diff(F(i)))).'''
    diag_len = min(m_shen, n_leg-1)
    diag_vals = np.r_[0, np.array([2/sqrt(4*j+6) for j in range(diag_len)])]
    return spdiags(diag_vals, 1, m_shen, n_leg)


# -----------------------------------------------------------------------------


if __name__ == '__main__':
    import lega.legendre_basis as leg
    import lega.shen_basis as shen
    from sympy.mpmath import quad
    from sympy import Symbol, lambdify

    m_shen = 3
    n_leg = 7
    x = Symbol('x')

    M_ = np.zeros((m_shen, n_leg))
    for i, s_ in enumerate(shen.shen_basis(m_shen)):
        for j, l_ in enumerate(leg.legendre_basis(n_leg)):
            M_[i, j] = quad(lambdify(x, s_*l_), [-1, 1])
    M = mixed_mass_matrix(m_shen, n_leg)
    assert np.linalg.norm(M - M_) < 1E-13


    G_ = np.zeros((m_shen, n_leg))
    for i, s_ in enumerate(shen.shen_basis(m_shen)):
        for j, l_ in enumerate(leg.legendre_basis(n_leg)):
            G_[i, j] = quad(lambdify(x, s_.diff(x, 1)*l_), [-1, 1])
    G = mixed_grad_matrix(m_shen, n_leg)
    assert np.linalg.norm(G - G_) < 1E-13
