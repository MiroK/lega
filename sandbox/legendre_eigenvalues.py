from lega.legendre_basis import mass_matrix, stiffness_matrix
from scipy.linalg import eigh
from math import sqrt
import numpy as np

n = 10

# These are matrices correspoding to L_0=1 and so on. As such the A matrix is
# not positive definite and the space spanned by L_0 is the kernel
M = mass_matrix(n)
A = stiffness_matrix(n)

# So we leave consider only basis L_1 ... and solve a well posed problem
lmbda1, V1 = eigh(A.toarray()[1:, 1:], M.toarray()[1:, 1:])

# And now add the zero into the spectrum along with its eigenvector [1, 0, ....] 
lmbda = np.concatenate([np.array([0]), lmbda1])
V = np.eye(n)/sqrt(2.)   # Note that the column/eigenvector is normalized
V[1:, 1:] = V1

# Let check we have not messed up anything. Can expect exact
for a, v in zip(lmbda, V.T):
    assert np.linalg.norm(A.dot(v) - a*M.dot(v))/n < 1E-12

assert np.linalg.norm(V.T.dot(M.dot(V)) - np.eye(n))/n < 1E-12
assert np.linalg.norm(V.T.dot(A.dot(V)) - np.diag(lmbda))/n < 1E-12

# Greate so now we know how to deal with the Neumann problem
