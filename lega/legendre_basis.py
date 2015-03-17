from __future__ import division
from numpy.polynomial.legendre import leggauss, legval
from sympy import legendre, symbols, Expr, lambdify, Symbol
from scipy.sparse import diags
from itertools import product
from common import function, tensor_product
import numpy as np


def legendre_basis(N, symbol='x'):
    '''Return first N Legendre polynomials as functions of symbol.'''
    return [legendre(k, Symbol(symbol)) for k in range(N)]


def legendre_function(F):
    '''
    A linear combination of F_i and the legendre basis functions. If F is a 
    vector the result is a function of F. For F matrix the output is a function
    of x, y.
    '''
    # 1d
    if F.shape == (len(F), ):
        basis = legendre_basis(len(F), 'x')
        return function(basis, F)
    # 2d
    elif len(F.shape) == 2:
        basis = tensor_product([legendre_basis(F.shape[0], 'x'),
                                legendre_basis(F.shape[1], 'y')])
        # Collapse to coefs by row
        F = F.flatten()
        return function(basis, F)
    # No 3d yet
    else:
        raise ValueError('For now F can be a a tensor of rank at most 2.')


def mass_matrix(N):
    '''Mass matrix of legendre_basis(N).'''
    return diags(np.array([2/(2*i+1) for i in range(N)]), 0)


def stiffness_matrix(N):
    '''Stiffness matrix of legendre_basis(N).'''
    # The matrix has part of the main diagonal on every second side diagonal
    main_diag = np.array([sum(2*(2*k+1) for k in range(0 if i%2 else 1, i, 2))
                          for i in range(N)])
    # Upper diagonals
    offsets = range(0, N, 2)
    diagonals = [main_diag[:N-offset] for offset in offsets]
    # All diagonal
    all_offsets = [-offset for offset in offsets[:0:-1]] + offsets
    all_diagonals = [diagonal for diagonal in diagonals[:0:-1]] + diagonals

    return diags(all_diagonals, all_offsets, shape=(N, N))


def backward_transformation_matrix(N):
    '''
    Compute NxN matrix with values N_ij = L_i(x_j) where L_i are N Legendre
    polynomials and x_j are N GL quadrature points. This matrix is used for
    backward Legendre transformation: Suppose function f is represented in 
    the wave number space by a vector F and let BL be the backward transformation
    matrix. Then f(x_j) = F.BL[:, j] or f = F.BL or BL.T.F, and vector f 
    represents f in the real space.
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
    def __init__(self, N):
        '''Cache the matrices.'''
        if not isinstance(N, list):
            N = [N]
        assert len(N) < 3
        self.__BL = [backward_transformation_matrix(n) for n in N]

    def __call__(self, F):
        '''Transform f from wave number space to physical space.'''
        if len(self.__BL) == 1:
            return (self.__BL[0].T).dot(F)
        else:
            return self.__BL[0].T.dot(F.dot(self.__BL[1]))

    def asarray(self):
        '''
        Return the transformation matrix. Matrix dotted with the representation
        in wave number space yields the representation in physical space.
        '''
        if len(self.__BL) == 1:
            return self.__BL[0]
        else:
            return np.kron(self.__BL[0].T, self.__BL[1].T)


def forward_transformation_matrix(N):
    '''
    For any function f, we define its interpolant f_N as \sum_{i=0}^{N-1}F_i*L_i,
    where L_i is the i-th Legendre polynomial and the coeffcients F_i are given
    as F_i=\sum_{j=0}^{n-1}*f(xj)*w_j*L_i(x_j)/(L_i, L_i). The interpolant is
    thus a polynomial of degree N-1. The reasoning behind the definition is that
    is f were a polynomial of degre N-1 the integrals (f, L_i) having an integrand
    of max degree 2N-2 would be exactly evaluated by the N-1 point GL gradrature.
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
   
    return FL


class GLNodeEvaluation(object):
    '''
    Evaluate f at nodes of N point GL quadrature. The points are precomputed
    and stored.
    '''
    def __init__(self, N):
        '''Compute the evaluation points.'''
        if not isinstance(N, list):
            N = [N]

        self.dim = len(N)
        # This would work for any dim but since only 1d and 2d is supported in
        # FLT and BLT I see no points in supporting it here.
        assert self.dim < 3
        self.shape = tuple(N)
        # Get points for components
        points_i = [leggauss(n)[0] for n in N] 
        # Combine as cartesian product
        self.points = np.array([list(pis) for pis in product(*points_i)])

    def __call__(self, f):
        '''Evaluate f at points.'''
        # In general the input should be some sort of (lambda) function
        # Sympy functions are lambdified for fast numpy evaluation
        dim = self.dim
        points = self.points
        
        xyz = symbols('x, y, z')
        if isinstance(f, Expr):
            # Symbolic function must be defined for x, y, z
            assert all(xi in f.atoms() for xi in xyz[:dim])
            f = lambdify(xyz[:dim], f, 'numpy')
            # Lambdify makes it fast if we feed as arrays the x y z comps of points
            f_values = f(*[points[:, i] for i in range(dim)])
        else:
            # For (lambda)function I can only check the argcount
            assert f.func_code.co_argcount == dim
            f_values = np.array([f(*(p.tolist())) for p in points])

        return f_values.reshape(self.shape)


class ForwardLegendreTransformation(object):
    '''
    Perform forward Legendre transformations. The transformation matrices
    are computed only once and so are the nodes for evaluation.
    '''
    def __init__(self, N):
        '''Cache the matrices.'''
        if not isinstance(N, list):
            N = [N]
        assert len(N) < 3

        self.__FL = [forward_transformation_matrix(n) for n in N]
        # Make your own evaluator
        self.__GLeval = GLNodeEvaluation(N)

    def __call__(self, f):
        '''Transform f to wave number space space.'''
        F = self.__GLeval(f)

        if len(self.__FL) == 1:
            return self.__FL[0].dot(F)
        else:
            return self.__FL[0].dot(F.dot(self.__FL[1].T))

    def asarray(self):
        '''
        Return the transformation matrix. For some f the matrix dotted with
        f evalueted at GL nodal points yield the representation in the wave
        number space.
        '''
        if len(self.__FL) == 1:
            return self.__FL[0]
        else:
            return np.kron(self.__FL[0], self.__FL[1])

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import simplify, sin, cos, pi, Symbol
    from sympy.mpmath import quad
    from sympy.plotting import plot
    import matplotlib.pyplot as plt
    from math import sqrt

    test_1d = True
    test_2d = False

    if test_1d:
        x = Symbol('x')
        # Check the stiffness matrix
        N = 16
        basis = legendre_basis(N)
        A = np.zeros((N, N))
        for i, v in enumerate(basis):
            integrand = v.diff(x, 1)**2
            A[i, i] = quad(lambdify(x, integrand), [-1, 1])
            for j, u in enumerate(basis[i+1:], i+1):
                integrand = u.diff(x, 1)*v.diff(x, 1)
                A[i, j] = quad(lambdify(x, integrand), [-1, 1])
                A[j, i] = A[i, j]

        A_ = stiffness_matrix(N).toarray()
        assert np.all(abs(A - A_) < 1E-15)

        # First a polynomial should be interpolated/projected/FL-transform exactly
        N = 8
        f = x**7 - 4*x**5 + 1
        f_ = lambdify(x, f, 'numpy')

        F = ForwardLegendreTransformation(N)(f)
        f_N = legendre_function(F)
        e = simplify(f-f_N)
        assert abs(sqrt(quad(lambdify(x, e**2), [-1, 1]))) < 1E-13

        # And use f_N(x_j) = f(x_j) to test BL
        f_N_values = BackwardLegendreTransformation(N)(F)
        f_values = GLNodeEvaluation(N)(f)
        # Allow some room for inexact numerics
        assert np.all(np.abs(f_N_values - f_values) < 1E-10)

        # Now take some `wilder` function and see how the interpolation quality
        # improves
        f = sin(x)*cos(pi*x**2)
        f_ = lambdify(x, f, 'numpy')
        
        tol = 1E-13
        converged = False
        N = 1
        N_max = 100
        # If you had expansion as F_ coeffs for a function then its L^2 norm could
        # be computed via the mass matrix as sqrt(F_.M.F_)
        # Take the largest space
        F_ = ForwardLegendreTransformation(N_max)(f_)

        Ns, errors = [], []
        while not converged:
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
        pf.show()

        # Plot convergence history
        plt.figure()
        plt.loglog(Ns, errors)
        plt.show()


    if test_2d:
        x, y = symbols('x, y')
        # First a polynomial should be interpolated/projected/FL-transform exactly
        N, M = 3, 4
        f = x**2 + y**3
        f_ = lambdify([x, y], f, 'numpy')

        F = ForwardLegendreTransformation([N, M])(f)
        f_NM = legendre_function(F)
        e = simplify(f-f_NM)
        assert abs(sqrt(quad(lambdify([x, y], e**2), [-1, 1], [-1, 1]))) < 1E-13

        # Represent FLT as object acting on vector
        F_nodes = GLNodeEvaluation([N, M])(f_).flatten()
        F_mat = ForwardLegendreTransformation([N, M]).asarray()
        F_ = F_mat.dot(F_nodes)
        assert np.all((F.flatten() - F_) < 1E-14)

        # What is the error with the mass matrix?
        mass_N = mass_matrix(N)
        mass_M = mass_matrix(M)
        E = ForwardLegendreTransformation([N, M])(e)
        assert abs(sqrt(np.trace((mass_N.dot(E)).dot(mass_M.dot(E.T))))) < 1E-13

        # And use f_N(x_j) = f(x_j) to test BL
        f_NM_values = BackwardLegendreTransformation([N, M])(F)
        f_values = GLNodeEvaluation([N, M])(f)
        # Allow some room for inexact numerics
        assert np.all(np.abs(f_NM_values - f_values) < 1E-10)

        # Represent BLT as object acting on vector
        F_mat = BackwardLegendreTransformation([N, M]).asarray()
        assert np.all((f_NM_values.flatten() - F_mat.dot(F.flatten())) < 1E-14)

        # Now take some `wilder` function and see how the interpolation quality
        # improves
        f = sin(x)*cos(2*pi*y)
        f_ = lambdify([x, y], f, 'numpy')
        
        tol = 1E-13
        converged = False
        N = 2
        N_max = 100
        # If you had expansion as F_ coeffs for a function then its L^2 norm could
        # be computed via the mass matrix as sqrt(F_.M.F_)
        # Take the largest space
        F_ = ForwardLegendreTransformation([N_max, N_max])(f_)

        Ns, errors = [], []
        while not converged:
            F = ForwardLegendreTransformation([N, N])(f_)
            
            # We compute the L2 error by the mass matrix taking f in the same space
            # as F_ so this is not exact
            E = F - F_[:N, :N]
            M = mass_matrix(N)
            
            error = sqrt(np.trace((M.dot(E)).dot(M.dot(E.T))))
            # Alternatively and equivalently
            # error = sqrt(((M.dot(E))*(M.dot(E.T)).T).sum())
            print 'N=%d (mass)L2=%.4E' % (N, error)
            Ns.append(N)
            errors.append(error)

            converged = error < tol or N >= N_max

            N += 1

        # See how the final interpolant compares to the function
        # from sympy.plotting import plot3d

        # f_N = legendre_function(F)
        # pf = plot3d(f-f_N, (x, -1, 1), (y, -1, 1))
        # pf.show()
