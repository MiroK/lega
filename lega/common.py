from __future__ import division
from itertools import product
from operator import mul
from sympy import Symbol, Matrix
import time
import functools


def function(basis, coefs):
    '''
    Symbolic function obtained as a linear combination of basis functions and
    expansion coefficients.
    '''
    assert len(basis) == len(coefs)
    return sum(c*v for c, v in zip(coefs, basis))


def tensor_product(basis_i):
    '''
    Combine [basis_i[i] for i in range(len(basis_i))] is combined as a tensor
    product. Note that the functions in basis_i[0] rotate the slowest.
    '''
    return [functools.reduce(mul, vs) for vs in product(*basis_i)]


def reference_mapping(domain, symbol=('x', 'y', 'z', 't')):
    '''
    Let \Omega_hat = [-1, 1]^d and consider \Omega, domain which is a cartesian
    product of d intervals. Here we compute mapping \Chi which maps points from
    \Omega_hat to \Omega and its inverse which maps \Omega to \Omega_hat.
    '''
    # 1d = [[a, b]]
    if len(domain) == 1:
        a, b = domain[0]
        assert b > a, 'Degenerate interval'

        L, center = (b-a)/2., (b+a)/2.
        # The key is x_hat -> x
        chi = (Symbol(symbol[0]), L*Symbol(symbol[0]) + center)
        # The key is x -> x_hat
        ichi = (Symbol(symbol[0]), (Symbol(symbol[0])-center)/L)
        # The idea is that these once turned into dictionaries are readily used
        # in sympy.subs
        return [chi], [ichi]
    else:
        assert len(domain) <= len(symbol), 'Not enough symbols'
        chi, ichi = [], []
        for subdomain, subsymbol in zip(domain, symbol):
            sub_chi, sub_ichi = reference_mapping([subdomain], [subsymbol])
            chi.append(sub_chi[0])
            ichi.append(sub_ichi[0])

        return chi, ichi


def jacobian_matrix(chi):
    '''
    Compute symbolic Jacobian matrix of mapping chi.
    '''
    # Variables for differentation
    variables = [pair[0] for pair in chi]
    # Components of the mapping
    rows = (pair[1] for pair in chi)
    # d chi_i / d sym_j
    return Matrix([[row.diff(var, 1) for var in variables] for row in rows])


def timeit(f):
    '''Timing decorator'''
    RED = '\033[1;37;31m%s\033[0m'
    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print(RED + '\t{%r} took: {%2.4f} sec'.format(f.__name__, te-ts))
        return result

    return timed

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import Symbol, S
    from sympy.plotting import plot
    import numpy as np

    # Check functions
    x = Symbol('x')
    basis = [S(1), x, x**2]
    coefs = [1 ,2, 3]

    u = function(basis, coefs)
    assert u == 1 + 2*x + 3*x**2

    # Check tensor products and their functions
    y = Symbol('y')
    basis_i = [[1, x, x**2], [1, y, y**2, y**3]]
    basis_tp = tensor_product(basis_i)

    coefs_tp = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
    coefs_tp = coefs_tp.flatten()

    v = function(basis_tp, coefs_tp)
    assert v == 1 + x*y + x**2*y**2 + x**2*y**3

    # Check mapping and Jacobian computation
    # 1d
    chi, ichi = reference_mapping([[-1, 1]])
    assert chi[0][0] == x and chi[0][1] == 1.0*x
    assert ichi[0][0] == x and ichi[0][1] == 1.0*x

    Jmat = jacobian_matrix(chi)
    assert Jmat == Matrix([[1.0]])
    assert abs(Jmat.det() - 1.0) < 1E-15

    #2d
    chi, ichi = reference_mapping([[-1, 1], [-1, 1]])
    assert chi[0][0] == x and chi[0][1] == 1.0*x
    assert chi[1][0] == y and chi[1][1] == 1.0*y

    Jmat = jacobian_matrix(chi)
    assert Jmat == Matrix([[1.0, 0], [0, 1.0]])
    assert abs(Jmat.det() - 1.0) < 1E-15

    #2d 'tougher'
    chi, ichi = reference_mapping([[-1, 1], [-2, 2]])
    assert chi[0][0] == x and chi[0][1] == 1.0*x
    assert chi[1][0] == y and chi[1][1] == 2.0*y

    Jmat = jacobian_matrix(chi)
    assert Jmat == Matrix([[1.0, 0], [0, 2.0]])
    assert abs(Jmat.det() - 2.0) < 1E-15
