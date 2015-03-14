from __future__ import division
from math import sqrt as Sqrt
from sympy import sqrt, Symbol, legendre
from scipy.sparse import eye, diags
from common import function
import numpy as np


def shen_basis(n, symbol='x'):
    '''
    List of first n basis function due to Shen - combinations of Legendre
    polynomials that have zeros at -1, 1 and yield sparse mass and stiffness
    matrices. Note that the maximum polynomial degree of functions in the basis
    is n+1.
    '''
    x = Symbol(symbol)
    functions = []

    k = 0
    while k < n:
        weight = 1/sqrt(4*k + 6)
        functions.append(weight*(legendre(k+2, x) - legendre(k, x)))
        k += 1

    return functions


def shen_function(F):
    '''A linear combination of F_i and the Shen basis functions.'''
    basis = shen_basis(len(F))
    return function(basis, F)


def mass_matrix(n):
    '''Mass matrix of Shen basis(n).'''
    weight = lambda k: 1/Sqrt(4*k + 6)
    # The matrix is tridiagonal and symmetric
    # Main
    main_diag = np.array([weight(i)**2*((2./(2*i+1) + 2./(2*(i+2)+1)))
                          for i in range(n)])
    # Upper
    up_diag = np.array([-weight(i)*weight(i+2)*(2./(2*(i+2)+1))
                        for i in range(n-2)])
   
    if n < 3:
        return diags(main_diag, 0)
    else:
        return diags([main_diag, up_diag, up_diag], [0, 2, -2])


def stiffness_matrix(n):
    '''Stiffness matrix of Shen basis(n).'''
    return eye(n)


def legendre_to_shen_matrix(m):
    '''
    This matrix represents a transformation that takes first m Legendre
    polynomials, that is the maximum polynomial degree in the set is m-1, and
    creates n=m-2 Shens functions. The output set has maximum degree m-1. So
    this is a m-2 x m matrix
    '''
    n = m-2
    main_diag = np.array([-1/Sqrt(4*k+6) for k in range(n)])
    return diags([main_diag, -main_diag], [0, 2], shape=(n, m))

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Check the transformation
    from legendre_basis import legendre_basis
    from sympy import lambdify
    from sympy.mpmath import quad

    # The goal is to get these function from Legendre
    n = 4
    shen = shen_basis(n)

    m = n+2
    leg = legendre_basis(m)
    T = legendre_to_shen_matrix(m).toarray()
    # Do the linear combination
    shen_ = [sum(T[i, j]*leg[j] for j in range(m)) for i in range(n)]

    x = Symbol('x')
    # Allow some room for error in L^2 norm integration
    assert all(Sqrt(quad(lambdify(x, (s-s_)**2), [-1, 1])) < 1E-13
               for s, s_ in zip(shen, shen_))

