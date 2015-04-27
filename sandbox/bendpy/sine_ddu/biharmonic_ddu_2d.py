#
# Solve laplace(laplace(u)) = f in [-1, 1]^2
#                         u = 0 on the boundary
#                laplace(u) = 0 on the boundary
#

from sympy.mpmath import quad
from sympy import pi, lambdify

def get_rhs(u):
    '''
    Verify that u satisfies boundary conditions and compute the right hand
    side f.
    '''
    
    x, y = symbols('x, y')
    PI = pi.n()
    # Value
    print quad(lambdify(y, u.subs(x, 0)**2), (0, PI))
    print quad(lambdify(y, u.subs(x, PI)**2), (0, PI))
    print quad(lambdify(x, u.subs(y, 0)**2), (0, PI))
    print quad(lambdify(x, u.subs(y, PI)**2), (0, PI))

    # Value
    ddu = u.diff(x, 2) + u.diff(y, 2)
    print quad(lambdify(y, ddu.subs(x, 0)**2), (0, PI))
    print quad(lambdify(y, ddu.subs(x, PI)**2), (0, PI))
    print quad(lambdify(x, ddu.subs(y, 0)**2), (0, PI))
    print quad(lambdify(x, ddu.subs(y, PI)**2), (0, PI))

    f = ddu.diff(x, 2) + ddu.diff(y, 2)

    return f

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import symbols, sin
    from sympy.plotting import plot3d

    x, y = symbols('x, y')
    u = (x*(x-pi)*y*(y-pi))**2*sin(2*x)*sin(4*y)

    get_rhs(u)


    plot3d(u, (x, 0, pi), (y, 0, pi))
