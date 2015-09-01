from sympy import chebyshevt, Symbol
from sympy.mpmath import chebyt
from scipy.fftpack import dct, idct


def chebyshev_basis(deg, var=''):
    '''
    Basis of T_k for space of polynomials over [-1, 1] with
    degree \leq deg. If var is an empty string this is mpmath function otherwise
    the function is symbolic. Only the mpmath version is reliable for
    evaluation.
    '''
    if not var:
        # Currying required!
        return [lambda x, k=k: chebyt(k, x) for k in range(deg+1)]
    else:
        return [chebyshevt(k, Symbol(var)) for k in range(deg+1)]


def chebyshev_function(coefs, var=''):
    '''
    Function from its representation in basis spanned by {T_k} 
    for k < len(coefs). If var is an empty string this is mpmath function otherwise
    the function is symbolic. Only the mpmath version is reliable for
    evaluation.
    '''
    deg = len(coefs) - 1
    basis = chebyshev_basis(deg, var)
    if not var:
        return lambda x: sum(ck*Tk(x) for ck, Tk in zip(coefs, basis))
    else:
        return sum(ck*Tk for ck, Tk in zip(coefs, basis))

# Suppose that p is a polynomial of degree N i.e. it is exactly representable
# in span{T_0, ..., T_N}: p = sum c_k * T_k. There is a fast transformation 
# betwee N + 1 coefficients c_k and point values p(x_k) at quadrature points
# of chebyshev-gauss gadrature. Note that deg+1 points is enough to integrate
# polynomials of degree 2*(deg+1) - 1 = 2*deg + 1
# 

def chebt(pxk):
    '''Transform values at points to values of coeffcients. Real -> Spectral.'''
    N = pxk.shape[0]
    ck = dct(pxk, 2, axis=0)
    ck /= N
    ck[0] /= 2
    return ck


def ichebt(ck):
    '''Transform coefficient values to point values. Spectral -> Real.'''
    pxk = 0.5*dct(ck, 3, axis=0)
    pxk += 0.5*ck[0]
    return pxk

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from numpy.polynomial.chebyshev import chebgauss, chebval
    import numpy as np

    x = Symbol('x')
    # Make up some polynomial represented in Chebyshev basis
    deg = 20
    ck = np.random.rand(deg+1)
    p_sym = chebyshev_function(ck, 'x')
    p_mp = chebyshev_function(ck, '')
    # Evaluate at xk
    xk, _ = chebgauss(deg+1)
    # Symbolic
    pxk_sym = np.array([p_sym.subs(x, xki) for xki in xk], dtype=float)
    # Multi precision
    pxk_mp = np.array([p_mp(xki) for xki in xk], dtype=float)
    # Numeric
    pxk = chebval(xk, ck)

    # NOTE that the error in pxk. The way sympy does this probably suffers 
    # from round-off. Good to know :). It is because it use the recursion
    # definition. Also note that the error in ck, pxk from transformations is of
    # the oerder of the evaluation error
    print 'Eval error sym:', np.linalg.norm(pxk - pxk_sym)
    print 'Eval error mp:', np.linalg.norm(pxk - pxk_mp)
    print 'Real-spectral error:', np.linalg.norm(ck-chebt(pxk), np.inf)
    print 'Spectral-real error:',  np.linalg.norm(pxk-ichebt(ck), np.inf)
    print 'Circle error:', np.linalg.norm(pxk - ichebt(chebt(pxk)), np.inf)
