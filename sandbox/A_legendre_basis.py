from lega.legendre_basis import legendre_function, mass_matrix,\
    stiffness_matrix, legendre_basis
from scipy.linalg import eigh
from sympy.plotting import plot
from sympy import Symbol
from math import sqrt
import numpy as np

# Visualize the eigenfunctions of -u'' = lmnda u in (-1, 1) with u'(-1)=u'(1)=0

n = 10
basis = legendre_basis(10)

# Solve the eigenvalue problem to get coeffs of the new basis
A = stiffness_matrix(n)
M = mass_matrix(n)

lmbda1, V1 = eigh(A.toarray()[1:, 1:], M.toarray()[1:, 1:])
lmbdas = np.concatenate([np.array([0]), lmbda1])
V = np.eye(n)/sqrt(2.)
V[1:, 1:] = V1

# Make the new basis
Abasis = [legendre_function(v) for v in V.T]

print 'eigenvalues', lmbdas

# Plot the basis for comparison
x = Symbol('x')
f_fA = iter(zip(basis, Abasis))

f, fA = next(f_fA)
p = plot(f, (x, -1, 1), show=False)
p[0].line_color = 'red'
p_ = plot(fA, (x, -1, 1), show=False)
p_[0].line_color = 'blue'
p.append(p_[0])

for f, fA in f_fA:
    p_ = plot(f, (x, -1, 1), show=False)
    p_[0].line_color = 'red'
    p.append(p_[0])

    p_ = plot(fA, (x, -1, 1), show=False)
    p_[0].line_color = 'blue'
    p.append(p_[0])

p.show()

# Some questions, what are the approximation properties of Abasis? And how does
# the transformation between function and its series look? And, is there a
# clever way to get the eigenvalues and eigenvectors
