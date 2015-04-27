from __future__ import division
from sympy import sin, sqrt, pi, Number, Expr, Symbol, lambdify, symbols
from common import function, tensor_product
from scipy.sparse import eye, diags
from math import pi as PI, sqrt as Sqrt
from sympy.mpmath import quad
from itertools import product
import numpy as np

SQRT_PI = Sqrt(pi)

def sine_basis(n, symbol='x'):
    '''
    Functions sin(k*x), k = 1, 2, ....n, normalized to have L^2 norm over [0, pi].
    Note that (sin(k*x), k**2) are all the solutions of 

        -u`` = lambda u in (0, pi)
        u(0) = u(pi) = 0
    '''
    x = Symbol(symbol)
    return [sin(k*x)*sqrt(2/pi) for k in range(1, n+1)]


def sine_function(F):
    '''
    Return a linear combination of len(F_vec) sine basis functions with
    coefficients given by F_vec.
    '''
    if len(F.shape) == 1:
        basis = sine_basis(F.shape[0], 'x')
        return function(basis, F)

    elif len(F.shape) == 2:
        basis_x = sine_basis(F.shape[0], 'x')
        basis_y = sine_basis(F.shape[1], 'y')
        basis = tensor_product([basis_x, basis_y])

        # Collapse to coefs by row
        F = F.flatten()
        return function(basis, F)

    else:
        raise NotImplementedError


def mass_matrix(n):
    '''inner(u, v) for u, v in sine_basis(n).'''
    return eye(n) 


def stiffness_matrix(n):
    '''inner(u`, v`) for u, v in sine_basis(n).'''
    return diags(np.arange(1, n+1)**2, 0, shape=(n, n))


def bending_matrix(n):
    '''inner(u``, v``) for u, v in sine_basis(n).'''
    return diags(np.arange(1, n+1)**4, 0, shape=(n, n))


# Suppose we have V_n = sine_basis(n) and for some function f we want to compute
# (f, v) for v in V_n, where (o, o) is the L^2 inner product over [0, pi].
# The idea is that if f is extended oddly to [0, 2*pi] then all the terms (f, v)
# can be computed at once by fft.
#
# The flow is eval -> (extend -> fft) -> (take only imag or results = sines)


def sine_eval(N, f):
    '''
    Sample f in N+1 points from the interval [0, 2*pi). Or the cartesian product
    of this interval
    '''
    # Symbolic is evaluated in [0, PI]
    assert isinstance(f, (Expr, Number))
    # 1d
    if isinstance(N, int):
        points = np.linspace(0, 2*PI, 2*N, endpoint=False)[:N]

        x = Symbol('x')
        flambda = lambdify(x, f, 'numpy')
        f_values = flambda(points)

        return f_values
    # 2d
    elif hasattr(N, '__len__'):
        assert len(N) == 2
        X = np.linspace(0, 2*PI, 2*N[0], endpoint=False)[:N[0]]
        Y = np.linspace(0, 2*PI, 2*N[1], endpoint=False)[:N[1]]
        # X and Y coordinates of the tensor product
        XY = np.array([list(xy) for xy in product(X, Y)])
        X, Y = XY[:, 0], XY[:, 1]

        x, y = symbols('x, y')
        flambda = lambdify([x, y], f, 'numpy')
        f_values = flambda(X, Y)

        return f_values.reshape(N)


def sine_fft(f_vec):
    '''
    Get sine expansion coeffs of f sampled at [0, Pi] and extended oddly ... .
    '''
    # 1d
    if f_vec.shape == (len(f_vec), ):
        N = len(f_vec)-1
        f_vec = np.r_[f_vec, f_vec[0], -f_vec[1:][::-1]]


        F_vec = np.fft.rfft(f_vec)
        # These are the coefficient values
        n_points = len(f_vec)
        F_vec[1:] *= -2./n_points/Sqrt(2/PI)
        
        return F_vec.imag[1:]
    #2d
    elif len(f_vec.shape) == 2:
        # Do sine_fft on rows
        for i, row in enumerate(f_vec):
            f_vec[i, :] = sine_fft(row)

        # Do sine_fft on cols
        for j, col in enumerate(f_vec.T):
            f_vec[:, j] = sine_fft(col)

        return f_vec


def load_vector(f, n, use_fft=False, n_fft=None):
    '''(f, v) for v in sine basis(n).'''
    # Compute the integral by numeric/symbolic integration
    if not use_fft: 
        # 1d
        if isinstance(n, int):
            x = Symbol('x')
            return np.array([quad(lambdify(x, f*v), [0, PI])
                             for v in sine_basis(n)], dtype=float)
        
        # 2d
        elif hasattr(n, '__len__'):
            assert len(n) == 2, 'Only 2d'
            # Basis in 2d is a tensor product of basis in each directions
            basis_x = sine_basis(n[0], 'x')
            basis_y = sine_basis(n[1], 'y')
            basis = tensor_product([basis_x, basis_y])

            x, y = symbols('x, y')
            return np.array([quad(lambdify([x, y], f*v), [0, PI], [0, PI])
                             for v in basis], dtype=float).reshape(n)

    # Integral by fft only approximate!
    else:
        assert n_fft is not None
        # 1d
        if isinstance(n, int):
            # If f is constant this is the minimal requirement for sensible results
            assert n_fft >= n

            f_vec = sine_eval(n_fft, f)
            F_vec = sine_fft(f_vec)[:n]

            return F_vec
        # 2d
        elif hasattr(n, '__len__'):
            assert len(n) == 2, 'Only 2d'

            f_vec = sine_eval([n_fft, n_fft], f)
            F_vec = sine_fft(f_vec)[:n[0], :n[1]]

            return F_vec


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import lambdify, Symbol, cos, exp

    n = 8
    basis = sine_basis(n)
    domain = [0, pi.n()]
    x = Symbol('x')


    # Test bending matrix
    mat_value = lambda u, v: quad(lambdify(x, u.diff(x, 2)*v.diff(x, 2)), domain)

    mat = np.zeros((len(basis), len(basis)))
    for i, u in enumerate(basis):
        mat[i, i] = mat_value(u, u)
        for j, v in enumerate(basis[i+1:], i+1):
            mat[i, j] = mat_value(u, v)
            mat[j, i] = mat[i, j]

    B = bending_matrix(n)
    assert B.shape == mat.shape
    assert np.allclose(B.toarray(), mat)

    # Test mass
    mat_value = lambda u, v: quad(lambdify(x, u*v), domain)

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
    mat_value = lambda u, v: quad(lambdify(x, u.diff(x, 1)*v.diff(x, 1)), domain)

    mat = np.zeros((len(basis), len(basis)))
    for i, u in enumerate(basis):
        mat[i, i] = mat_value(u, u)
        for j, v in enumerate(basis[i+1:], i+1):
            mat[i, j] = mat_value(u, v)
            mat[j, i] = mat[i, j]

    B = stiffness_matrix(n)
    assert B.shape == mat.shape
    assert np.allclose(B.toarray(), mat)

    # Check load vector by FFT
    f = x*(x-pi)
    # f = sin(x) + 7*sin(2*x) - sin(4*x)  # Exact
    # f = sin(x)*cos(2*pi*x)*exp(x**2)
    # f = exp(x)*(sum(i*x**i for i in range(1, 4)))
    load_exact = np.array([quad(lambdify(x, f*v), [0, PI]) for v in basis],
                           dtype=float)
   
    b = load_vector(f, len(basis))
    b_ = load_vector(f, len(basis), use_fft=True, n_fft=2**14)
    print '1d error', np.linalg.norm(b - b_)

    # y = Symbol('y')
    # g = sin(x)*(y**2-1)
    # print load_vector(g, [2, 2])
    # # How many sines you need to get the n integrals in the load vector right
    # N = n
    # for k in range(1, 11):
    #     f_vec = sine_eval(N, g)
    #     load_num = sine_fft(f_vec)[:n] 
    #  
    #     print N, np.linalg.norm(load_exact - load_num)
    #     N *= 2

    # y = Symbol('y')
    # f = sin(x)*sin(y)
    # sine_eval(N=[4, 4], f=f)

    x, y = symbols('x, y')
    f = x*(x-pi)*y*(y-pi)*sin(x)

    import time
    
    start = time.time()
    b = load_vector(f, [5, 5])
    print 'QUAD', time.time() - start

    start = time.time()
    b_ = load_vector(f, [5, 5], use_fft=True, n_fft=64)
    print 'FFT', time.time() - start

    print '2d error', np.linalg.norm(b - b_)
