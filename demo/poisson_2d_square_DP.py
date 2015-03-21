#
# Solve -laplace(u) = f in (0, 2*pi)x(-1, 1)
#         with T(u) = 0 on y = -1 and y = 1
#         and periodicity in the x direction
# 
# We shall combine Fourier and Shen basis

from sympy import symbols, integrate, pi, lambdify
from sympy.mpmath import quad

def get_rhs(u):
    '''
    Verify that u satisfies boundary conditions and compute the right hand
    side f.
    '''
    # Verify that bcs might hold
    x, y = symbols('x, y')
    assert integrate(abs(u.subs(y, -1)), (x, -1, 1)) < 1E-15
    assert integrate(abs(u.subs(y, 1)), (x, -1, 1)) < 1E-15
    print quad(lambdify(y, abs(u.subs(x, 0) - u.subs(x, 2*pi))), [-1, 1])

    # Right hand side if u is to be the solution
    f = -u.diff(x, 2) - u.diff(y, 2)

    return f

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import sin, cos
    from sympy.plotting import plot3d

    x, y = symbols('x, y')

    k = 4
    l = 4

    u = sin(k*pi*y)*y**2*cos(l*x)
    f = get_rhs(u)

    plot3d(u, (x, 0, 2*pi), (y, -1, 1))
