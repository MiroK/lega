from itertools import product
from operator import mul


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
    return [reduce(mul, vs) for vs in product(*basis_i)]

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import Symbol, S
    from sympy.plotting import plot
    import numpy as np

    x = Symbol('x')
    basis = [S(1), x, x**2]
    coefs = [1 ,2, 3]

    u = function(basis, coefs)
    assert u == 1 + 2*x + 3*x**2

    y = Symbol('y')
    basis_i = [[1, x, x**2], [1, y, y**2, y**3]]
    basis_tp = tensor_product(basis_i)
    
    coefs_tp = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
    coefs_tp = coefs_tp.flatten()

    v = function(basis_tp, coefs_tp)
    assert v == 1 + x*y + x**2*y**2 + x**2*y**3
