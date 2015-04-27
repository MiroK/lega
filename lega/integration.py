from __future__ import division
from numpy.polynomial.legendre import leggauss
from sympy import Symbol, lambdify, symbols
from itertools import product
import numpy as np


class Quad1d(object):
    '''One dimensional integration with GL quadrature.'''
    def __init__(self, N):
        '''Quadrature formulat using N points.'''
        self.points, self.weights = leggauss(N)

    def __call__(self, f, domain):
        '''Integrate symbolic f over domain.'''
        a, b = domain
        assert a < b

        # Map to reference [-1, 1]. Don't forget Jacobian
        x = Symbol('x')
        pull_back = a*(1-x)/2 + b*(1+x)/2
        Jacobian = (b-a)/2
        f = f.subs(x, pull_back)*Jacobian

        f = lambdify(x, f, 'numpy')

        f_point_values = f(self.points)
        return np.sum(f_point_values*self.weights)


class Quad2d(object):
    '''Two dimensional integration with GL quadrature.'''
    def __init__(self, N):
        '''Quadrature formulat using N x Npoints.'''
        points, weights = leggauss(N)

        # Let's do the tensor product
        XY = np.array([list(xy) for xy in product(points, points)])
        self.X = XY[:, 0]
        self.Y = XY[:, 1]

        self.weights = np.array([w0*w1 for w0, w1 in product(weights, weights)])


    def __call__(self, f, domain_x, domain_y):
        '''Integrate symbolic f over cartesian product of domain_x, domain_y.'''
        ax, bx = domain_x
        assert ax < bx

        ay, by = domain_y
        assert ay < by

        # Map to reference [-1, 1]. Don't forget Jacobian
        x, y = symbols('x, y')
        pull_back_x = ax*(1-x)/2 + bx*(1+x)/2
        pull_back_y = ay*(1-y)/2 + by*(1+y)/2

        Jacobian = (bx-ax)/2*(by-ay)/2
        f = f.subs({x: pull_back_x, y: pull_back_y})*Jacobian

        f = lambdify([x, y], f, 'numpy')

        f_point_values = f(self.X, self.Y)
        return np.sum(f_point_values*self.weights)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy.mpmath import quad
    from sympy import sin, cos

    x = Symbol('x')
    f = sin(x**2)*cos(x)
    domain = [-1, 3]

    exact = quad(lambdify(x, f), domain)

    Q1 = Quad1d(20)
    numeric = Q1(f, domain)

    print abs(exact - numeric)

    # ----

    x, y = symbols('x, y')
    g = sin(sin(x**2 - y)*cos(y))
    domain_x = [-1, 3]
    domain_y = [0, 1]

    exact = quad(lambdify([x, y], g), domain_x, domain_y)

    Q2 = Quad2d(100)
    numeric = Q2(g, domain_x, domain_y)

    print abs(exact - numeric)
