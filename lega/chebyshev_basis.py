from __future__ import division
from sympy import symbols, Expr, lambdify, Symbol, Number, S
from sympy.mpmath import chebyt
from scipy.sparse import diags
from itertools import product
from common import function, tensor_product
import numpy as np


def chebyshevt_basis(N, symbol='x'):
    '''First N Chebyshev polynomials of the first kind.'''
    if N == 1:
        return [S(1)]
    elif N == 2:
        return [S(1), Symbol(symbol)]
    else:
        T = chebyshevt_basis(N-1, symbol)
        T0, T1 = T[-2:]
        return T + [2*Symbol(symbol)*T1 - T0]


def mass_matrix(N):
    '''Mass matrix of N Chebyshev polys of first kind.'''
    # Note that the inner product is weighed L2: u*v/sqrt(1-x**2)
    return diags(np.r_[np.pi, np.ones(N-1)*np.pi/2], 0)


def stiffness_matrix(N):
    '''Stiffness matrix of N Chebyshev polys of first kind.'''
    pass


# -----------------------------------------------------------------------------


if __name__ == '__main__':
    from sympy.mpmath import quad
    from sympy import sqrt, simplify

    foo = chebyshevt_basis(10)

    x = Symbol('x')
    if True:
        for f in foo:
            for g in foo:
                p = f*g/sqrt(1-x**2)
                print '%.2f' % quad(lambdify(x, p), [-1, 1]),
            print 
    print mass_matrix(10)

    print [simplify(f) for f in foo]

    from sympy import cos, sin
    f = sin(x) + x
    # Definition 1
    for b in foo:
        p = f*b/sqrt(1-x**2)
        print quad(lambdify(x, p), [-1, 1])

    # Definition 2
    f = sin(cos(x)) + cos(x)
    print '>>', f
    for k in range(len(foo)):
        b = cos(k*x)
        p = f*b
        print quad(lambdify(x, p), [0, np.pi])


