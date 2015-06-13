from __future__ import division
from math import sqrt as Sqrt
from sympy import sqrt, Symbol, legendre
from scipy.sparse import eye, diags
from common import function, tensor_product
import numpy as np


def shen_basis(n, symbol='x'):
    '''
    List of first n basis function due to Shen - combinations of Legendre
    polynomials that have zeros at -1, 1 and yield sparse mass and stiffness
    matrices. Note that the maximum polynomial degree of functions in the basis
    is n+1.
    '''
    x = Symbol(symbol)
    functions = []

    k = 0
    while k < n:
        weight = 1/sqrt(4*k + 6)
        functions.append(weight*(legendre(k+2, x) - legendre(k, x)))
        k += 1

    return functions


def shen_function(F):
    '''
    A linear combination of F_i and the Shen basis functions. If F is a 
    vector the result is a function of F. For F matrix the output is a function
    of x, y.
    '''
    # 1d
    if F.shape == (len(F), ):
        basis = shen_basis(len(F), 'x')
        return function(basis, F)
    # 2d
    elif len(F.shape) == 2:
        basis = tensor_product([shen_basis(F.shape[0], 'x'),
                                shen_basis(F.shape[1], 'y')])
        # Collapse to coefs by row
        F = F.flatten()
        return function(basis, F)
    # No 3d yet
    else:
        raise ValueError('For now F can be a a tensor of rank at most 2.')


def mass_matrix(n):
    '''Mass matrix of Shen basis(n).'''
    weight = lambda k: 1/Sqrt(4*k + 6)
    # The matrix is tridiagonal and symmetric
    # Main
    main_diag = np.array([weight(i)**2*((2./(2*i+1) + 2./(2*(i+2)+1)))
                          for i in range(n)])
    # Upper
    up_diag = np.array([-weight(i)*weight(i+2)*(2./(2*(i+2)+1))
                        for i in range(n-2)])
   
    if n < 3:
        return diags(main_diag, 0)
    else:
        return diags([main_diag, up_diag, up_diag], [0, 2, -2])


def stiffness_matrix(n):
    '''Stiffness matrix of Shen basis(n).'''
    return eye(n)


def legendre_to_shen_matrix(m):
    '''
    This matrix represents a transformation that takes first m Legendre
    polynomials, that is the maximum polynomial degree in the set is m-1, and
    creates n=m-2 Shens functions. The output set has maximum degree m-1. So
    this is a m-2 x m matrix
    '''
    n = m-2
    main_diag = np.array([-1/Sqrt(4*k+6) for k in range(n)])
    return diags([main_diag, -main_diag], [0, 2], shape=(n, m))


def load_vector(F):
    '''
    Typically in solving boundary value problem by the Galerkin method, we need
    to know b_i = \int_{-1}^{1} f(x) \phi_i(x) dx where \phi_i are the basis
    functions of Shen and i=0, ..., n-1. Because of how the basis functions are
    defined we have b_i = \int_{-1}^{1} f(x) c_i[L_{i+2}-L_{i}]. At this point
    f(x) is replaced by its Legendre interpolant of degree n+1. Then b = T*M*F,
    where F is the n+1 long vector of expansion coeffs, M is the (n+1)x(n+1) 
    mass matrix in Legendre basis and T is the (n-1)x(n+1) transformation matrix
    that takes the result from Legendre to Shen basis. This generatlized to 2d
    as well. with b(matrix) = (T*M)*F(T.M).T. Note F is not necessary square
    '''
    # We do the operations manually
    # 1d
    if F.shape == (len(F), ):
        m = len(F)
        n = m-2
        b = np.zeros(n)
        for j in range(n):
            w = 1./Sqrt(4*j+6)  # Shen weight
            b[j] = -w*(2*F[j]/(2*j+1) - 2*F[j+2]/(2*(j+2)+1))
    else:
        # 2d, Don't need 3d yet?
        assert len(F.shape) == 2
        b = np.array([load_vector(colF) for colF in F.T])
        b = np.array([load_vector(rowb) for rowb in b.T])

    return b


def apply_mass_inverse(G):
    '''
    Suppose you want to expand function f into a series of Shen basis functions.
    Then the problem for the coeffcients is b = M*F with b_j = (f, \phi_j) and
    M the mass matrix w.r.t to the Shen basis. Thus, inv(M) is needed to compute
    b. If G is FLT(f) then this function provides an explicit formula for the
    action of the inverse. It is exact if f is a polynomial in H^10, and it
    gets exact when f in H^1_0. For functions that don't respect bcs, this is
    obviously hopeless.
    '''
    n = len(G)-2
    weights = [1./Sqrt(4*j+6) for j in range(n)]

    F = np.zeros(n)
    for j in range(2):
        F[j] = -G[j]/weights[j]
    for j in range(2, n):
        F[j] = -(G[j] - weights[j-2]*F[j-2])/weights[j]
    return F

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Check the transformation
    from legendre_basis import legendre_basis, ForwardLegendreTransformation as\
        FLT, mass_matrix as L_mass_matrix
    from sympy import lambdify, sin, pi, integrate, symbols
    from sympy.mpmath import quad

    import matplotlib as mpl
    mpl.use('MacOSX')

    from sympy.plotting import plot
    n = 5
    shen = shen_basis(n)
    x = Symbol('x')
    print [f.subs(x, 0) for f in shen]
    for f in shen:
        plot(f, (x, -1, 1))


    test_1d = False
    test_2d = False

    if test_1d:
        # The goal is to get these function from Legendre
        n = 4
        shen = shen_basis(n)

        m = n+2
        leg = legendre_basis(m)
        T = legendre_to_shen_matrix(m).toarray()
        # Do the linear combination
        shen_ = [sum(T[i, j]*leg[j] for j in range(m)) for i in range(n)]

        x = Symbol('x')
        # Allow some room for error in L^2 norm integration and inexact
        # arithmetics in computing coefficients
        assert all(Sqrt(quad(lambdify(x, (s-s_)**2), [-1, 1])) < 1E-13
                   for s, s_ in zip(shen, shen_))

        # 1d
        # Check that the load vector is assembled correctly
        f = (x**2-1)
        b_exact = np.array([float(quad(lambdify(x, f*phi), [-1, 1]))
                            for phi in shen])

        F = FLT(m)(f)
        b0 = load_vector(F)
        assert np.linalg.norm(b0-b_exact)/n < 1E-15

        # Check that the logic of getting the load vector by transformations
        T = legendre_to_shen_matrix(m)
        M = L_mass_matrix(m)
        b1 = T.dot(M.dot(F))
        assert np.linalg.norm(b1-b_exact)/n < 1E-15


        # We know that for f and F = FLT(f), M the Legendre mass matrix the value
        # sqrt{F.M.F} is L^2 norm of f if f is represented exactly
        ans_ = Sqrt(F.dot(M.dot(F)))
        ans = sqrt(integrate(f**2, (x, -1, 1)))
        assert abs((ans - ans_).n()) < 1E-13

        # Some observations about approximation properties of the shen basis
        # Consider shen basis of length n and to every f assign a series f_n = 
        # sum F_k \phi_k where F_k is the solution to the linear system M*F = b
        # for b in (f, phi_k).
        # How does this behave for a polynomial that is in H^1_0
        f = (x-1)*(x+1)**3
        # This has degree four. Shen of length 3 has that degree and should be fine
        for n in range(2, 15):
            basis = shen_basis(n)
            M = mass_matrix(n)
            b = np.array([quad(lambdify(x, f*phi), [-1, 1]) for phi in basis])
            F = np.linalg.solve(M.toarray(), b)

            f_n = shen_function(F)

            e = f-f_n
            error = float(sqrt(quad(lambdify(x, e**2), [-1, 1])))
            print 'n=%d, error=%g' % (n, error)

            # Let's check that f_n in shen can be obtained by mapping legendre
            F_leg = FLT(n+2)(f)
            F_shen = apply_mass_inverse(F_leg) 
            
            print '\t inverse - apply inverse', np.linalg.norm((F - F_shen))

        # Some function in H^10
        f = (x-1)**2*sin(pi*x)
        # This has degree four. Shen of length 3 has that degree and should be fine
        for n in range(2, 15):
            basis = shen_basis(n)
            M = mass_matrix(n)
            b = np.array([quad(lambdify(x, f*phi), [-1, 1]) for phi in basis])
            F = np.linalg.solve(M.toarray(), b)

            f_n = shen_function(F)

            e = f-f_n
            error = float(sqrt(quad(lambdify(x, e**2), [-1, 1])))
            print 'n=%d, error=%g' % (n, error)

            # Let's check that f_n in shen can be obtained by mapping legendre
            F_leg = FLT(n+2)(f)
            F_shen = apply_mass_inverse(F_leg) 
            # Note how this improves with degree :) 
            print '\t inverse - apply inverse', np.linalg.norm((F - F_shen))


        # What about polynomial not there just in H^1 but not in H^1_0 - should be
        # hopeless
        f = (x-1)*(x+2)**3
        for n in range(2, 15):
            basis = shen_basis(n)
            M = mass_matrix(n)
            b = np.array([quad(lambdify(x, f*phi), [-1, 1]) for phi in basis])
            F = np.linalg.solve(M.toarray(), b)

            f_n = shen_function(F)

            e = f-f_n
            error = float(sqrt(quad(lambdify(x, e**2), [-1, 1])))
            print 'n=%d, error=%g' % (n, error)

            F_leg = FLT(n+2)(f)
            F_shen = apply_mass_inverse(F_leg) 
            
            print '\t inverse - apply inverse', np.linalg.norm((F - F_shen))
        # And it is

    # 2d
    if test_2d:
        # Check that the load vector is assembled correctly
        x, y = symbols('x, y')
        f = (x**2-1)*y**4

        n, m = 2, 3
        basis_i = [shen_basis(n, 'x'), shen_basis(m, 'y')]
        shen = tensor_product(basis_i)

        b_exact = np.array([float(quad(lambdify([x, y], f*phi), [-1, 1], [-1, 1]))
                            for phi in shen])
        F = FLT([n+2, m+2])(f)
        b0 = load_vector(F).flatten()
        assert np.all(np.abs(b0-b_exact) < 1E-15)

        # Check that the logic of getting the load vector by transformations
        Tn = legendre_to_shen_matrix(n+2)
        Mn = L_mass_matrix(n+2)
        mat0 = Tn.dot(Mn.toarray())

        Tm = legendre_to_shen_matrix(m+2)
        Mm = L_mass_matrix(m+2)
        mat1 = Tm.dot(Mm.toarray())

        b1 = mat0.dot(F.dot(mat1.T)).flatten()
        assert np.all(np.abs(b1-b_exact) < 1E-15)
