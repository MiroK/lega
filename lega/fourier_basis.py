from __future__ import division
from sympy import sin, cos, Symbol, lambdify, Number, Expr
from common import function
from math import sqrt, pi
import numpy as np

SQRT_PI = sqrt(pi)


def fourier_basis(N, symbol='x'):
    '''
    Consider a Fourier series
    
    \sum k\in[0, N]a_k cos(k x)/sqrt(pi) + \sum k\in[1, N]b_k sin(kx)/sqrt(pi).

    Here these 2*N + 1 basis functions are returned. Basis functions are
    normalized to have the L^2 norm over [0, 2*pi] equal to 1. The N can be
    understood as the highest wavenumber.
    '''
    x = Symbol(symbol)
    return [cos(k*x)/SQRT_PI for k in range(N+1)] +\
           [sin(-k*x)/SQRT_PI for k in range(1, N+1)]


def fourier_function(F_vec):
    '''
    Return a linear combination of len(F_vec) Fourier basis functions with
    coefficients given by F_vec.
    '''
    if len(F.shape) == 1:
        basis = fourier_basis(F.shape[0], 'x')
        return function(basis, F)
    else:
        raise NotImplementedError('2d, 3d not implemented yet.')


def fourier_eval(N, f):
    '''
    Let N the highest wavenumber to be considered in the Fourier series for f.
    In order to get Fourier coefficients of f with some accuracy the function
    must be avaluated in a least 2*N points.
    '''
    n_points = 2*N
    points = np.linspace(0, 2*pi, n_points, endpoint=False)

    if isinstance(f, (Number, int, float)):
        f_values = float(f)*np.ones(len(points))
    # Symbolic functions
    elif isinstance(f, Expr):
        x = Symbol('x')
        assert x in f.atoms() or isinstance(f, Number)
        f = lambdify(x, f, 'numpy')
        # Lambdify makes it fast if we feed as arrays the x y z comps
        # of points
        f_values = f(points)
    # Python functions/lambdas
    else:
        # For (lambda)function I can only check the argcount
        assert f.func_code.co_argcount == 1
        f_values = np.array([f(*(p.tolist())) for p in points])

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
    F_vec *= 2*SQRT_PI/n_points
    F_vec = np.r_[F_vec.real, F_vec.imag[1:]]

    return F_vec


def ifft(F_vec):
    '''
    To get values of f in control points back, the coefficients must be rescaled
    and collapsed to complex vector.
    '''
    n_points = len(F_vec) - 1
    F_vec /= 2.*SQRT_PI/n_points
    end_real = n_points/2 + 1
    F_vec = F_vec[:end_real] + 1j*np.r_[0, F_vec[end_real:]]
    f_vec = np.fft.irfft(F_vec)

    return f_vec

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy.mpmath import quad
    from sympy import lambdify
    from random import random

    N = 15
    tol = 1E-13
    
    x = Symbol('x')
    f = sum(random()*cos(k*x) + random()*sin(k*x) for k in range(N))

    # Computing fourier series coefficients by FFT
    f_vec = fourier_eval(N, f)
    F_vec = fft(f_vec)

    # Compare with analytical coefficients values
    basis = fourier_basis(N)
    for i, v in enumerate(basis):
        assert abs(quad(lambdify(x, v*f), [0, 2*np.pi]) - F_vec[i]) < 1E-14

    # Comming back
    f_vec_ = ifft(F_vec)
    assert np.allclose(f_vec, f_vec_)


    





    


