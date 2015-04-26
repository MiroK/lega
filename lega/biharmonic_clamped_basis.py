from __future__ import division
from math import sqrt as Sqrt
from sympy import sqrt, Symbol, legendre
from scipy.sparse import eye, diags, dia_matrix
from common import function, tensor_product
import legendre_basis
import numpy as np


def shen_cb_basis(n, symbol='x'):
    '''
    Shen's clamped biharmonic basis.

    List of first n basis functions due to Shen, combinations of Legendre
    polynomials, that have zeros as boundary values of functions themselves and
    their first derivaties as -1, 1. Note that the maximum polynomial degree of
    functions in the basis is n+3.
    '''
    x = Symbol(symbol)
    functions = []

    k = 0
    while k < n:
        weight = 1/sqrt(2*(2*k+3)**2*(2*k+5))
        functions.append(weight*(legendre(k, x) -\
                         2*(2*k + 5)/(2*k + 7)*legendre(k+2, x) +\
                         (2*k + 3)/(2*k + 7)*legendre(k+4, x)))
        k += 1

    return functions


def shen_cb_function(F):
    '''
    A linear combination of F_i and the Shen basis functions. If F is a 
    vector the result is a function of F. For F matrix the output is a function
    of x, y.
    '''
    # 1d
    if F.shape == (len(F), ):
        basis = shen_cb_basis(len(F), 'x')
        return function(basis, F)
    # 2d
    elif len(F.shape) == 2:
        basis = tensor_product([shen_cb_basis(F.shape[0], 'x'),
                                shen_cb_basis(F.shape[1], 'y')])
        # Collapse to coefs by row
        F = F.flatten()
        return function(basis, F)
    # No 3d yet
    else:
        raise ValueError('For now F can be a a tensor of rank at most 2.')


# Coeficients for entries in M, C
_d = lambda k: 1/Sqrt(2*(2*k+3)**2*(2*k+5))
_e = lambda k: 2/(2*k + 1)
_g = lambda k: (2*k + 3)/(2*k + 7)
_h = lambda k: -(1 + _g(k))


def mass_matrix(n):
    '''Mass matrix of n functions from Shen's basis: (f, g).'''
    main_diag = np.array([_d(k)**2*(_e(k) + _h(k)**2*_e(k+2) + _g(k)**2*_e(k+4))
                          for k in range(n)])

    u_diag = np.array([_d(k)*_d(k+2)*(_h(k)*_e(k+2) + _g(k)*_h(k+2)*_e(k+4))
                       for k in range(n-2)])

    uu_diag = np.array([_d(k)*_d(k+4)*_g(k)*_e(k+4) for k in range(n-4)])

    if len(uu_diag):
        return diags([uu_diag, u_diag, main_diag, u_diag, uu_diag], 
                     [-4, -2, 0, 2, 4], shape=(n, n))
    else:
        if len(u_diag):
            return diags([u_diag, main_diag, u_diag], [-2, 0, 2], shape=(n, n))
        else:
            return diags([main_diag], [0], shape=(n, n))


def stiffness_matrix(n):
    '''Stiffness matrix of n functions from Shen's basis: (f`, g`).'''
    main_diag = np.array([-2*(2*k + 3)*_d(k)**2*_h(k) for k in range(n)])
    other_diag = np.array([-2*(2*k + 3)*_d(k)*_d(k+2) for k in range(n-2)])
    if len(other_diag):
        return diags([other_diag, main_diag, other_diag], [-2, 0, 2], shape=(n, n))
    else:
        return diags([main_diag], [0,], shape=(n, n))


def bending_matrix(n):
    '''Bending matrix of n functions from Shen's basis: (f``, g``).'''
    return eye(n, dtype=float)


def legendre_to_shen_cb_matrix(m):
    '''
    This matrix represents a transformation that takes first m Legendre
    polynomials, that is the maximum polynomial degree in the set is m-1, and
    creates n=m-4 Shens cb functions. The output set has maximum degree m-4+3. 
    This then a is a m-4 x m matrix
    '''
    n = m-4
    assert n > 0
    
    main_diag = np.array([1./Sqrt(2*(2*k+3)**2*(2*k+5)) for k in range(n)])
    p2_diag = np.array([-2*(2*k + 5)/(2*k + 7)/Sqrt(2*(2*k+3)**2*(2*k+5))
                        for k in range(n)])
    p4_diag = np.array([(2*k + 3)/(2*k + 7)/Sqrt(2*(2*k+3)**2*(2*k+5))
                        for k in range(n)])

    # FIXME: diags does not work here?
    mat = dia_matrix((main_diag, 0), shape=(n, m)) +\
          dia_matrix((np.r_[0, 0, p2_diag], 2), shape=(n, m)) +\
          dia_matrix((np.r_[0, 0, 0, 0, p4_diag], 4), shape=(n, m))

    return mat
        

def load_vector(F):
    '''
    int_{-1, 1} f(x)*phi_i(x) for i=0, ..., n-1  <-  shen_cb(n), meaning that
    the highest pol degree is n-1+4 = n+3

    Now replace F by forward legendre transform of of len(n+4). The vector load
    vector can be computed as T.dot(M.dot(F)), i.e. integrate in legendre basis
    and come back to Shen.
    '''
    # 1d
    if F.shape == (len(F), ):
        n_leg = len(F)
        n_shen = n_leg - 4
        assert n_shen > 0
        M_leg = legendre_basis.mass_matrix(n_leg)
        T = legendre_to_shen_cb_matrix(n_leg)
        return T.dot(M_leg.dot(F))
    # 2d
    else:
        assert len(F.shape) == 2
        b = np.array([load_vector(colF) for colF in F.T])
        b = np.array([load_vector(rowb) for rowb in b.T])

        return b

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from legendre_basis import ForwardLegendreTransformation as FLT
    from sympy import lambdify, S, sin
    from sympy.mpmath import quad

    n = 10
    # Test basis bcs
    x = Symbol('x')
    basis = shen_cb_basis(n)
    for v in basis:
        vals = [v.subs(x, -1).n(), v.subs(x, -1).n(),\
                v.diff(x, 1).subs(x, -1).n(), v.diff(x, 1).subs(x, -1).n()]
        assert all(v < 1E-15 for v in map(abs, vals))

    # Test bending matrix
    mat_value = lambda u, v: quad(lambdify(x, u.diff(x, 2)*v.diff(x, 2)), [-1, 1])

    mat = np.zeros((len(basis), len(basis)))
    for i, u in enumerate(basis):
        mat[i, i] = mat_value(u, u)
        for j, v in enumerate(basis[i+1:], i+1):
            mat[i, j] = mat_value(u, v)
            mat[j, i] = mat[i, j]

    B = bending_matrix(n)
    assert B.shape == mat.shape

    # Test mass
    mat_value = lambda u, v: quad(lambdify(x, u*v), [-1, 1])

    mat = np.zeros((len(basis), len(basis)))
    for i, u in enumerate(basis):
        mat[i, i] = mat_value(u, u)
        for j, v in enumerate(basis[i+1:], i+1):
            mat[i, j] = mat_value(u, v)
            mat[j, i] = mat[i, j]

    B = mass_matrix(n)
    assert B.shape == mat.shape
    assert np.allclose(B.toarray(), mat)

    # Test stiffness
    mat_value = lambda u, v: quad(lambdify(x, u.diff(x, 1)*v.diff(x, 1)), [-1, 1])

    mat = np.zeros((len(basis), len(basis)))
    for i, u in enumerate(basis):
        mat[i, i] = mat_value(u, u)
        for j, v in enumerate(basis[i+1:], i+1):
            mat[i, j] = mat_value(u, v)
            mat[j, i] = mat[i, j]

    B = stiffness_matrix(n)
    assert B.shape == mat.shape
    assert np.allclose(B.toarray(), mat)

    # Test transformation
    n_leg = n+4
    leg_basis = legendre_basis.legendre_basis(n_leg)
    mat = legendre_to_shen_cb_matrix(len(leg_basis)).toarray()
    assert mat.shape == (len(basis), len(leg_basis))

    for shen, row in zip(basis, mat):
        # Assemble my shen
        shen_ = sum((coef*L for coef, L in zip(row, leg_basis)), S(0))
        # L2 error
        e = (shen - shen_)**2
        E = Sqrt(quad(lambdify(x, e), [-1, 1]))
        assert abs(E) < 1E-13

    # Test load vector
    f = sum(x**i for i in range(n+4))
    F = FLT(n_leg)(f)
    b = load_vector(F)
    assert len(b) == n

    b_ = np.zeros(n)
    for i, v in enumerate(basis):
        b_[i] = quad(lambdify(x, v*f), [-1, 1])

    assert np.allclose(b, b_)
