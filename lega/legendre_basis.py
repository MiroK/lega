from __future__ import division
from numpy.polynomial.legendre import leggauss, legval
from sympy import legendre, symbols, Expr, lambdify
from scipy.sparse import diags
from itertools import product
from common import function
import numpy as np


x, y = symbols('x, y')


def legendre_basis(N, symbol='x'):
    '''Return first N Legendre polynomials as functions of symbol.'''
    return [legendre(k, Symbol(symbol)) for k in range(N)]


def legendre_function(F, symbol='x'):
    '''
    A linear combination of F_i and the legendre basis functions as funnction
    of symbol.
    '''
    basis = legendre_basis(len(F), symbol)
    return function(basis, F)


def mass_matrix(N):
    '''Mass matrix of legendre_basis(N).'''
    return diags(np.array([2/(2*i+1) for i in range(N)]), 0)


def stiffness_matrix(N):
    '''Stiffness matrix of legendre_basis(N).'''
    # TODO 
    pass


def backward_transformation_matrix(N):
    '''
    Compute NxN matrix with values N_ij = L_i(x_j) where L_i are N Legendre
    polynomials and x_j are N GL quadrature points. This matrix is used for
    backward Legendre transformation: Suppose function f is represented in 
    the wave number space by a vector F and let BL be the backward transformation
    matrix. Then f(x_j) = F.BL[:, j] or f = F.BL, and vector f represents f in
    the real space.
    '''
    BL = np.zeros((N, N))
    # Get points of the guadrature
    points, _ = leggauss(N)
    for i in range(N):
        c = np.zeros(i+1)
        c[-1] = 1
        # Evaluate the i-th polynomial at all the points
        row = legval(points, c)

        BL[i, :] = row

    return BL


class BackwardLegendreTransformation(object):
    '''
    Perform backward Legendre transformations. The transformation matrix
    is computed only once.
    '''
    # TODO for 2d
    def __init__(self, N):
        '''Cache the matrix.'''
        self.__BL = backward_transformation_matrix(N)

    def __call__(self, F):
        '''Transform f from wave number space to physical space.'''
        return F.dot(self.__BL)

    def asarray(self):
        '''Return the transformation matrix.'''
        return self.__BL


def forward_transformation_matrix(N, with_points=False):
    '''
    For any function f, we define its interpolant f_N as \sum_{i=0}^N F_i * L_i,
    where L_i is the i-th Legendre polynomial and the coeffcients F_i are given
    as F_i=\sum_{j=0}^n*f(xj)*w_j*L_i(x_j)/(L_i, L_i). The interpolant is thus a
    polynomial of degree N-1. The reasoning behind the definition is that is f
    were a polynomial of degre N-1 the integrals (f, L_i) having an integrand of
    max degree 2N-2 would be exactly evaluated by the N-1 point GL gradrature.
    Vector F is a representation of function f in the wave number space. 
    Computing F can be represented as matrix-vector product and is reffered to
    as a forward Legendre transformation. Here we get the
    matrix for the operatation FL.
    '''
    # Note that each row of FL could be computed by taking a dot of row of
    # matrix BL.inv(M) with the vector of weight. 
    FL = np.zeros((N, N))
    # Get point and weights of the guadrature
    points, weights = leggauss(N)
    for i in range(N):
        c = np.zeros(i+1)
        c[-1] = 1
        # Evaluate te the i-th polynomial at all the points
        row = legval(points, c)
        # Now the element-wise with with weights, i.e. dot with weight vector
        row *= weights
        # Finally the (Li, Li) term, i.e. the inv(M)
        row /= 2/(2*i+1)

        FL[i, :] = row
   
    # Points can become handy
    if not with_points:
        return FL
    else:
        return FL, points


def node_eval(N, f):
    '''Evaluate f at nodes of N point GL quadrature.'''
    # In general the input should be some sort of (lambda) function
    # Sympy functions are lambdified for fast numpy evaluation
    # Evaluate function of x, y
    if isinstance(N, list):
        assert len(N) == 2
        # The evaluation points are a tensor product
        points_x, _ = leggauss(N[0])
        points_y, _ = leggauss(N[1])
        points = np.array([np.array([p_x, p_y])
                           for p_x, p_y in product(points_x, points_y)]) 
        # Make sure we have functions of two vars
        if isinstance(f, Expr):
            assert x in f.atoms() and y in f.atoms()
            f = lambdify([x, y], f, 'numpy')
            return f(points[:, 0], points[:, 1]).reshape((N[0], N[1]))
        else:
            assert f.func_code.co_argcount == 2
            return np.array([f(*p) for p in points]).reshape((N[0], N[1]))

    else:
        # Evaluate function of x
        points, _ = leggauss(N)
        # Make sure we have function of x 
        if isinstance(f, Expr):
            assert x in f.atoms()
            f = lambdify(x, f, 'numpy')
            return f(points)
        else:
            assert f.func_code.co_argcount == 1
            return np.array([f(p) for p in points])


class ForwardLegendreTransformation(object):
    '''
    Perform forward Legendre transformations. The transformation matrix
    is computed only once and so are the nodes for evaluation.
    '''
    # TODO for 2d
    def __init__(self, N):
        '''Cache the matrix.'''
        self.__FL, self.__points = forward_transformation_matrix(N, True)

    def __call__(self, f):
        '''Transform f to wave number space space.'''
        if isinstance(f, Expr):
            f = lambdify(x, f, 'numpy')

        F = f(self.__points)
        return self.__FL.dot(F)

    def asarray(self):
        '''Return the transformation matrix.'''
        return self.__FL


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import simplify, sin, cos, pi, Symbol
    from sympy.mpmath import quad
    from sympy.plotting import plot
    import matplotlib.pyplot as plt
    from random import random
    from math import sqrt

    x = Symbol('x')
    # First a polynomial should be interpolated/projected/FL-transform exactly
    N = 8
    f = sum(random()*x**i for i in range(N))
    f_ = lambdify(x, f, 'numpy')
    
    F = ForwardLegendreTransformation(N)(f)
    f_N = legendre_function(F)
    e = simplify(f-f_N)
    assert abs(quad(lambdify(x, e**2), [-1, 1])) < 1E-15

    # And use f_N(x_j) = f(x_j) to test BL
    f_N_values = BackwardLegendreTransformation(N)(F)
    f_values = node_eval(N, f)
    # Allow some room for inexact numerics
    assert np.all(np.abs(f_N_values - f_values) < 1E-10)

    # Now take some `wilder` function and see how the interpolation quality
    # improves
    f = sin(x)*cos(pi*x**2)
    f_ = lambdify(x, f, 'numpy')
    
    tol = 1E-13
    converged = True
    N = 1
    N_max = 10
    # If you had expansion as F_ coeffs for a function then its L^2 norm could
    # be computed via the mass matrix as sqrt(F_.M.F_)
    # Take the largest space
    F_ = ForwardLegendreTransformation(N_max)(f_)

    Ns, errors = [], []
    while not converged:
        print 'xxx', N
        F = ForwardLegendreTransformation(N)(f_)
        f_N = legendre_function(F)
        e = simplify(f-f_N)
        # Evaluare the L2 error by mpmath.quad which is adaptive and almost
        # exact
        error = sqrt(quad(lambdify(x, e**2), [-1, 1]))
        
        # We compute the L2 error by the mass matrix taking f in the same space
        # as F_ so this is not exact, but it's interesting, right? :)
        e_ = F - F_[:N]
        M = mass_matrix(N)
        error_ = sqrt(e_.dot(M.dot(e_)))
        
        print 'N=%d L2=%.4E (mass)L2=%.4E' % (N, error, error_)
        Ns.append(N)
        errors.append(error)

        converged = error < tol or N >= N_max

        N += 1

    # See how the final interpolant compares to the function
    pf = plot(f, (x, -1, 1), show=False)
    pf_ = plot(f_N, (x, -1, 1), show=False)
    pf_[0].line_color='red'
    pf.append(pf_[0])
    # pf.show()

    # Plot convergence history
    plt.figure()
    plt.loglog(Ns, errors)
    # plt.show()

    # Note that for sufficiently regular function the convergence is exponential
    # but the sin(x)*cos(pi*x**2) example shows that for exponential convergence
    # N must be big enough

    # Test the stiffness_matrix
    # n = 6
    # basis = legendre_basis(n)
    # A = np.zeros((n, n))
    # for i, v in enumerate(basis):
    #     for j, u in enumerate(basis):
    #         integrand = lambdify(x, u.diff(x)*v.diff(x))
    #         A[i, j] = quad(integrand, [-1, 1])
    # 
    # A_ = stiffness_matrix(n)
    # 
    # print A
    # print A_

    f = sin(pi*x)*sin(3*pi*y)
    print node_eval([4, 5], f)

    from math import sin
    f = lambda x, y: sin(pi*x)*sin(3*pi*y)
    print node_eval([4, 5], f)


