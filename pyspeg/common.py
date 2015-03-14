def function(basis, coefs):
    '''
    Symbolic function obtained as a linear combination of basis functions and
    expansion coefficients.
    '''
    return sum(c*v for c, v in zip(coefs, basis))

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import Symbol, S
    from sympy.plotting import plot

    x = Symbol('x')
    basis = [S(1), x, x**2]
    coefs = [1 ,2, 3]

    u = function(basis, coefs)

    print u


