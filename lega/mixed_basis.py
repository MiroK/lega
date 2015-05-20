# The Stokes problem is solved by combining Shen basis and Legendre basis
from scipy.sparse import diags


def mixed_mass_matrix(m_shen, n_leg):
    '''Mass matrix between first m_shen Shen functions and n_lege Legendre functions.'''
    # mat_{i,j} = (Shen(i), Legendre(j))_{-1, 1}
    pass



def mixed_div_matrix(m_shen, n_leg):
    '''Matrix for divergence operator.'''
    # mat_{i,j} = (leg(j), diff(shen(i)))_{-1, 1}
    pass

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import lega.legendre_basis as leg
    import lega.shen_basis as shen
    from sympy.mpmath import quad
    from sympy import lambdify, Symbol
    import numpy as np

    m_shen = 4
    n_leg = 5

    x = Symbol('x')
    M_ = np.zeros((m_shen, n_leg), dtype=float)
    for i, s_ in enumerate(shen.shen_basis(m_shen)):
        for j, l_ in enumerate(leg.legendre_basis(n_leg)):
            M_[i, j] = quad(lambdify(x, s_*l_), [-1, 1])

    print M_

    
