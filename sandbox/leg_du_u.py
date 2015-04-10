import lega.legendre_basis as leg
from lega.common import function
from sympy.mpmath import quad
from scipy.sparse import diags
import scipy.linalg as la
from sympy import Symbol, lambdify, integrate, S
import numpy as np


def get_solution(u=None, f=None):
    '''
    Find f such that du/dx = f and \int_{-1}^{1}u = 0.
    '''
    x = Symbol('x')
    # Compute u from f
    if u is None:
        assert f is not None
        u = integrate(f, x)  # + C, which is computed to yield 0 mean of u
        C = integrate(u, (x, -1, 1))
        u -= C/2
        print 'u', u, 'C', integrate(u, (x, -1, 1))
        assert abs(integrate(u, (x, -1, 1))) < 1E-15
    else:
        assert abs(integrate(u, (x, -1, 1))) < 1E-15
        # Comupute f for u
        if f is None:
            f = u.diff(x, 1)
        # Both given, so just check
        else:
            assert u.diff(x, 1) - f == 0

    return u, f

x = Symbol('x')
from sympy import sin, pi, cos
f = 1 + cos(2*pi*x) #x  #S(1)
uu, ff = get_solution(f=f)
print 'u, f', uu, ff

n = 10
trial_basis = leg.legendre_basis(n)[1:]
test_basis = leg.legendre_basis(len(trial_basis))
x = Symbol('x')

assert len(trial_basis) == len(test_basis)
N = len(trial_basis)
C_ = np.zeros((N, N))
for i, v in enumerate(test_basis):
    for j, u in enumerate(trial_basis):
        du = u.diff(x, 1)
        integrand = lambdify(x, v*du)
        C_[i, j] = quad(integrand, [-1, 1])


def gradient_matrix(n):
    offsets = np.arange(1, n, 2)
    diagonals = [2*np.ones(n-k) for k in offsets]
    return diags(diagonals, offsets, shape=(n, n))

print C_
C = gradient_matrix(n)
C = C.toarray()[:-1, 1:]
print C
assert np.allclose(C_, C)
 
M = leg.mass_matrix(N)
 
F = leg.ForwardLegendreTransformation(N)(ff)
b = M.dot(F)

U = la.solve(C, b)

print 'b', b
print 'uu', uu, quad(lambdify(x, uu), [-1, 1])
print U
uh = function(trial_basis, U)
print 'uh', uh

e = uu - uh
print e
error = lambdify(x, e**2)
print quad(error, [-1, 1])

# 
# print 'u', uu
# U = leg.ForwardLegendreTransformation(n)(uu)
# print 'Mu', M.dot(U)
# u = np.array([quad(lambdify(x, v*uu), (-1, 1)) for v in basis], dtype=float)
# print u
# 
# print C.toarray()[:-1, 1:]
# print C.toarray()[:-1, 1:].dot(u[1:]) - b[:-1]
# 
# print C.toarray()
# print (diags(np.r_[0, np.ones(n-1)], 0, shape=(n, n)).dot(C)).toarray()
