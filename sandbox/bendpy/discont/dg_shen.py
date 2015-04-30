# Analytical solutions for problems that can be solved with the two Shen basis.
###
# Problem one is
#
#   -u`` = f in [-1, 1] with u(-1) = u(1) = 0 for f which is
#   g on [-1, 0) and h on [0, 1]
#
###
# Problem two is
# 
# u```` = f in [-1, 1] with u(-1) = u(1) = 0, u`(-1) = u`(1) = 0 for f which is
# g on [-1, 0) and h on [0, 1]

from __future__ import division
from sympy import Symbol, integrate
import numpy as np


def solve_poisson(g, h):
    '''Solve the Poisson problem with f defined by g, h'''
    x = Symbol('x')
    # Primitive functions of g
    G = integrate(-g, x)
    GG = integrate(G, x)
    # Primitive functions of h
    H = integrate(-h, x)
    HH = integrate(H, x)

    # The solution is GG + a0*x + b0 on [-1, 0] and HH + a1*x + b1 on [0, 1]
    # Build the lin sys for the coefficients. The system reflects bcs and
    # continuity of u and u` in 0
    A = np.array([[-1., 1., 0., 0.],
                  [0., 0., 1., 1.],
                  [0., 1., 0., -1.],
                  [1., 0., -1., 0.]])
    b = np.array([-GG.subs(x, -1),
                  -HH.subs(x, 1),
                  HH.subs(x, 0) - GG.subs(x, 0),
                  H.subs(x, 0) - G.subs(x, 0)])

    [a0, b0, a1, b1] = np.linalg.solve(A, b)

    u0 = GG + a0*x + b0
    u1 = HH + a1*x + b1

    # Let's the the checks
    # Boundary conditions
    bcl = u0.subs(x, -1)
    bcr = u1.subs(x, 1)
    # Continuity of solution and the derivative
    u_cont = u0.subs(x, 0) - u1.subs(x, 0)
    du_cont = u0.diff(x, 1).subs(x, 0) - u1.diff(x, 1).subs(x, 0)
    # That it in fact solves the laplacian
    u0_lap = integrate((u0.diff(x, 2) + g)**2, (x, -1, 0))
    u1_lap = integrate((u1.diff(x, 2) + h)**2, (x, 0, 1))

    conds = [bcl, bcr, u_cont, du_cont, u0_lap, u1_lap]
    assert all(map(lambda v: abs(v) < 1E-13, conds))

    return u0, u1


def solve_biharmonic(g, h):
    '''Solve the biharmonic problem with f defined by g, h'''
    x = Symbol('x')
    # Primitive functions of g
    G = integrate(g, x)
    GG = integrate(G, x)
    GGG = integrate(GG, x)
    GGGG = integrate(GGG, x)

    # Primitive functions of h
    H = integrate(h, x)
    HH = integrate(H, x)
    HHH = integrate(HH, x)
    HHHH = integrate(HHH, x)

    # The solution now needs to match bcs and continuity.
    A = np.array([[-1./6, 1./2, -1., 1., 0., 0., 0., 0.],
                  [0, 0, 0, 0, 1/6., 1/2., 1., 1.],
                  [1/2., -1, 1, 0, 0, 0, 0, 0.],
                  [0, 0, 0, 0, 1/2., 1, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0, -1],
                  [0, 0, 1, 0, 0, 0, -1, 0],
                  [0, 1, 0, 0, 0, -1, 0, 0],
                  [1, 0, 0, 0, -1, 0, 0, 0]])

    b = np.array([-GGGG.subs(x, -1), 
                  -HHHH.subs(x, 1), 
                  -GGG.subs(x, -1),
                  -HHH.subs(x, 1),
                  HHHH.subs(x, 0) - GGGG.subs(x, 0),
                  HHH.subs(x, 0) - GGG.subs(x, 0),
                  HH.subs(x, 0) - GG.subs(x, 0),
                  H.subs(x, 0) - G.subs(x, 0)])


    [a0, a1, a2, a3, b0, b1, b2, b3] = np.linalg.solve(A, b)

    u0 = GGGG + a0*x**3/6 + a1*x**2/2 + a2*x + a3
    u1 = HHHH + b0*x**3/6 + b1*x**2/2 + b2*x + b3

    # Let's the the checks
    checks = []
    # Boundary conditions
    checks.append(u0.subs(x, -1))
    checks.append(u1.subs(x, 1))
    checks.append(u0.diff(x, 1).subs(x, -1))
    checks.append(u1.diff(x, 1).subs(x, 1))
    # Continuity of solution and the derivatives
    checks.append(u0.subs(x, 0) - u1.subs(x, 0))
    checks.append(u0.diff(x, 1).subs(x, 0) - u1.diff(x, 1).subs(x, 0))
    checks.append(u0.diff(x, 2).subs(x, 0) - u1.diff(x, 2).subs(x, 0))
    checks.append(u0.diff(x, 3).subs(x, 0) - u1.diff(x, 3).subs(x, 0))
    # That it in fact solves the biharmonic equation
    checks.append(integrate((u0.diff(x, 4) - g)**2, (x, -1, 0)))
    checks.append(integrate((u1.diff(x, 4) - h)**2, (x, 0, 1)))

    assert all(map(lambda v: abs(v) < 1E-13, checks))

    return u0, u1

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import S, nsimplify
    from sympy.plotting import plot

    x = Symbol('x')
    g, h = S(1), 1+x

    problem = 'poisson'
    k = 3

    if problem == 'poisson':
        u0, u1 = solve_poisson(g, h)

        p0 = plot(u0.diff(x, k), (x, -1, 0), show=False)
        p1 = plot(u1.diff(x, k), (x, 0, 1), show=False)
        p2 = plot(g, (x, -1, 0), show=False)
        p3 = plot(h, (x, 0, 1), show=False)
        p4 = plot(u0.diff(x, 1), (x, -1, 0), show=False)
        p5 = plot(u1.diff(x, 1), (x, 0, 1), show=False)
        
        p0[0].line_color='red'
        p1[0].line_color='red'
        p2[0].line_color='blue'
        p3[0].line_color='blue'
        # p4[0].line_color='green'
        # p5[0].line_color='green'

        p0.append(p1[0])
        p0.append(p2[0])
        p0.append(p3[0])
        # p0.append(p4[0])
        # p0.append(p5[0])

    if problem == 'biharmonic':
        u0, u1 = solve_biharmonic(g, h)

        # Sol
        p0 = plot(u0, (x, -1, 0), show=False)
        p1 = plot(u1, (x, 0, 1), show=False)
        # Du
        # p2 = plot(u0.diff(x, 1), (x, -1, 0), show=False)
        # p3 = plot(u1.diff(x, 1), (x, 0, 1), show=False)
        # DDu
        # p4 = plot(u0.diff(x, 2), (x, -1, 0), show=False)
        # p5 = plot(u1.diff(x, 2), (x, 0, 1), show=False)
        # DDDu
        # p6 = plot(u0.diff(x, 3), (x, -1, 0), show=False)
        # p7 = plot(u1.diff(x, 3), (x, 0, 1), show=False)
        
        p0[0].line_color='red'
        p1[0].line_color='red'
        # p2[0].line_color='blue'
        # p3[0].line_color='blue'
        # p4[0].line_color='green'
        # p5[0].line_color='green'
        # p6[0].line_color='black'
        # p7[0].line_color='black'
        
        p0.append(p1[0])
        # [p0.append(p[0]) for p in (p1, p2, p3, p4, p5, p6, p7)]
       
    print 'u0', nsimplify(u0)
    print 'u1', nsimplify(u1)
    p0.show()
