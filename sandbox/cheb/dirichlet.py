from sympy import chebyshevt, Symbol
from sympy.mpmath import chebyt
from scipy.sparse import diags
from numpy.polynomial.chebyshev import chebgauss
import numpy as np


def dirichlet_basis(deg, var=''):
    '''
    Basis of combinations of T_k for space of polynomials over [-1, 1] with
    degree \leq deg. and zero boundary values. If var is an empty string this 
    is mpmath function otherwise the function is symbolic. Only the mpmath 
    version is reliable for evaluation. These functions are labelled Dk
    '''
    if not var:
        # Currying required!
        return [lambda x, k=k:\
                chebyt(k, x) - chebyt(k+2, x) for k in range(deg-1)]
    else:
        x = Symbol(var)
        return [chebyshevt(k, x) - chebyshevt(k+2, x) for k in range(deg-1)]


def dirichlet_function(coefs, var=''):
    '''
    Function from its representation in basis spanned by {D_k} 
    for k < len(coefs). If var is an empty string this is mpmath function 
    otherwise the function is symbolic. Only the mpmath version is reliable for
    evaluation.
    '''
    deg = len(coefs) - 1
    basis = dirichlet_basis(deg, var)
    if not var:
        return lambda x: sum(ck*Dk(x) for ck, Dk in zip(coefs, basis))
    else:
        return sum(ck*Dk for ck, Dk in zip(coefs, basis))


def mass_matrix(deg):
    '''Mass matrix of the basis for polynomials of degree uti deg.'''
    main = np.pi*np.r_[3, 2*np.ones(deg-2)]/2
    upper = -np.pi*np.ones(deg-3)/2
    return diags([upper, main, upper], [-2, 0, 2], shape=(deg-1, deg-1))


def integrate(u, v, deg):
    '''
    Integrate int_{-1, 1}u(x)*v(x)*weight(x) numerically by chebyshev-gauss
    quadrature of degree deg.
    '''
    xq, wq = chebgauss(deg)
    u_xq = np.array([u(xqi) for xqi in xq])
    v_xq = np.array([v(xqi) for xqi in xq])
    return np.sum(wq*u_xq*v_xq)

# Add laplacian matrix
# derivative matrix
# transformation matrix to shen
# transfomation to spectral space
#IDEA At tome point every linear transformation (also in chebyshev.py) should be
#represented as a linear operator with A*b and A/b


# ----------------------------------------------------------------------------

if __name__ == '__main__':
    # Space degree assumption
    deg = 6
    basis = dirichlet_basis(deg, 'x')
    assert len(basis) == (deg - 1)
    assert basis[-1].as_poly().degree() == deg

    # Boundary values
    basis = dirichlet_basis(deg)
    assert all(abs(Dk(-1)) < 1E-14 and abs(Dk(1)) < 1E-14 for Dk in basis)

    # Mass
    M = mass_matrix(deg)
    M_ = np.zeros(M.shape)
    for i, u in enumerate(basis):
        M_[i, i] = integrate(u, u, deg+1)
        for j, v in enumerate(basis[i+1:], i+1):
            M_[i, j] = integrate(u, v, deg+1)
            M_[j, i] = M_[i, j]
    assert np.linalg.norm(M.toarray() - M_, np.inf) < 1E-14
