from sympy import symbols, integrate, pi, lambdify, Number, sin
from numpy.polynomial.legendre import leggauss 
import scipy.sparse.linalg as sparse_la
import lega.fourier_basis as fourier
import lega.shen_basis as shen
import lega.legendre_basis as leg
from lega.common import tensor_product, function
from lega.legendre_basis import forward_transformation_matrix as FLT
from lega.legendre_basis import backward_transformation_matrix as BLT
from itertools import product
from sympy.mpmath import quad
import numpy as np


x, y = symbols('x, y')
f = x**2*sin(y)

n = 4
m = 3

# Create grid for evaluating f
fourier_points = np.linspace(0, 2*np.pi, n, endpoint=False)
legendre_points = leggauss(m)[0]

points = np.array([list(p)
                   for p in product(fourier_points, legendre_points)])

# Eval
f = lambdify([x, y], f, 'numpy')
F = f(points[:, 0], points[:, 1]).reshape((n, m))

# Now the columns which is f evaluated at Fourier points for fixed y at some
# quadrature points is Fourier transformed
F_hat = np.array([fourier.fft(col) for col in F.T]).T
# Now Forward Legendre transform each row
flt = FLT(m)
F_hat = np.array([flt.dot(row) for row in F_hat])

# Come back from wave numbers to grid values
blt = BLT(m).T
F_ = np.array([blt.dot(row) for row in F_hat])
F_ = np.array([fourier.ifft(col) for col in F_.T]).T

assert np.allclose(F, F_)
