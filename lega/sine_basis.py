from __future__ import division
from sympy import sin, sqrt, pi, Number, Expr, Symbol, lambdify
from common import function
from scipy.sparse import eye, diags
from math import pi as PI, sqrt as Sqrt
from sympy.mpmath import quad
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
    else:
        raise NotImplementedError('2d, 3d not implemented yet.')


def mass_matrix(n):
    '''inner(u, v) for u, v in sine_basis(n).'''
    return eye(n) 


def stiffness_matrix(n):
    '''inner(u`, v`) for u, v in sine_basis(n).'''
    return diags(np.arange(1, n+1)**2, 0, shape=(n, n))


def bending_matrix(n):
    '''inner(u``, v``) for u, v in sine_basis(n).'''
    return diags(np.arange(1, n+1)**4, 0, shape=(n, n))


def sine_eval(N, f):
    '''
    Let N the highest wavenumber to be considered in the sine series for f.
    In order to get Fourier coefficients of f with some accuracy the function
    must be avaluated in a least 2*N points.
    '''
    points = np.linspace(0, 2*PI, 2*N, endpoint=False)

    if isinstance(f, (Number, int, float)):
        f_values = float(f)*np.ones(len(points))
        f_pi = float(f)
    # Symbolic functions
    elif isinstance(f, Expr):
        x = Symbol('x')
        assert x in f.atoms() or isinstance(f, Number)
        f = lambdify(x, f, 'numpy')
        # Lambdify makes it fast if we feed as arrays the x y z comps
        # of points
        f_values = f(points)
        f_pi = f(PI)
    # Python functions/lambdas
    else:
        # For (lambda)function I can only check the argcount
        assert f.func_code.co_argcount == 1
        f_values = np.array([f(*(p.tolist())) for p in points])
        f_pi = f(PI)

    # Periodically extend
    # FIXME: Does it really need to be this ugly?
    f_values = np.r_[f_values[:N], f_pi, -f_values[1:N][::-1]]

    return f_values


def fft(f_vec):
    '''
    If f was sapled in 2*N points we can construct a series with highest
    frequency N that has 2*N+1 terms. To get its coefficients we take a
    DFFT of f_vec this is N+1 complex values. The real part is proportional to
    the N+1 coeffs for cosines, while the imaginary part is related to the N
    coefficients of sines.
    '''
    F_vec = np.fft.rfft(f_vec)
    # These are the coefficient values
    n_points = len(f_vec)
    F_vec[1:] *= -2./len(f_vec)/Sqrt(2/PI)

    return F_vec.imag[1:]


def load_vector(f, n):
    '''(f, v) for v in sine basis(n)'''
    x = Symbol('x')
    return np.array([quad(lambdify(x, f*v), [0, PI]) for v in sine_basis(n)])

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
    # f = sin(x) + 7*sin(2*x) - sin(4*x)  # Exact
    f = sin(x)*cos(2*pi*x)*exp(x**2)
    load_exact = np.array([quad(lambdify(x, f*v), [0, PI]) for v in basis],
                           dtype=float)
    
    # How many sines you need to get the n integrals in the load vector right
    N = n
    for k in range(1, 11):
        f_vec = sine_eval(N, f)
        load_num = fft(f_vec)[:n] 

        print N, np.linalg.norm(load_exact - load_num)
        N *= 2

