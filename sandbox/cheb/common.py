from sympy import chebyshevt, Symbol
from scipy.fftpack import dct, idct

def chebyshev_basis(deg, var=Symbol('x')):
    '''
    Basis of symbolic T_k for space of polynomials over [-1, 1] with
    degree \leq deg.
    '''
    return [chebyshevt(k, var) for k in range(deg+1)]


def chebyshev_function(coefs, var=Symbol('x')):
    '''
    Function from its representation in basis spanned by {T_k} 
    for k < len(coefs).
    '''
    deg = len(coefs) - 1
    return sum(ck*Tk for ck, Tk in zip(coefs, chebyshev_basis(deg)))

# Suppose that p is a polynomial of degree N i.e. it is exactly representable
# in span{T_0, ..., T_N}: p = sum c_k * T_k. There is a fast transformation 
# betwee N + 1 coefficients c_k and point values p(x_k) at quadrature points
# of chebyshev-gauss gadrature. Note that deg+1 points is enough to integrate
# polynomials of degree 2*(deg+1) - 1 = 2*deg + 1

def chebyshev_transform(pxk):
    '''Transform values at points to values of coeffcients. Real -> Spectral.'''
    N = pxk.shape[0]
    ck = dct(pxk, 2, axis=0)
    ck /= N
    ck[0] /= 2
            
    return ck


def inverse_chebyshev_transoform(ck):
    '''Transform coefficient values to point values. Spectral -> Real.'''
    N = ck.shape[0]
    ck[0] *= 2
    ck *= N
    pxk = idct(ck, 2, axis=0)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from numpy.polynomial.chebyshev import chebgauss
    import numpy as np

    # Make up some polynomial represented in Chebyshev basis
    deg = 3
    ck = np.random.rand(deg+1)
    p = chebyshev_function(ck)
    # Evaluate
    xk, _ = chebgauss(deg+1)
    pxk = np.array([subs(Symbol('x'), xki) for xki in xk], dtype=float)

    print pxk



