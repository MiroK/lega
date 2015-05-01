from __future__ import division
from sympy import lambdify, symbols, sqrt, Symbol
from sympy.mpmath import quad
import numpy as np

#
# A beam is something that has defined mapping chi: [-1, 1] -> ([-1, 1]^2, [-1, 1]^2) 
# with two properties : 1) chi(-1), chi(1) are on the boundary of [-1, 1]^2
#                       2) Jacobian of the mapping is nonzero in the whole domain
#

_TOL = 1E-13

def on_boundary(V, a=-1, b=1):
    'Check if point V lies on the boundary of [a, b]^2'
    # Get the geometric dimension of the point
    assert a < b
    assert len(V) == 2, 'Not a point in 2d'
    # On edge in one coordinate, In between in the other
    one = (abs(V[0]-a) < _TOL or abs(V[0]-b) < _TOL) and\
            (a - _TOL < V[1] < b + _TOL)

    two = (abs(V[1]-a) < _TOL or abs(V[1]-b) < _TOL) and\
            (a - _TOL < V[0] < b + _TOL)
    
    return one or two


class Beam(object):
    '''Create the beam from mapping'''
    def __init__(self, chi):
        # Geometry
        d = len(chi)
        assert d == 2, 'Not a 2d vector mapping'
        self.d = d
        # The mapping
        self.chi = chi
        # Compute the jacobian |d_x| = |d_chi/d_s| * |d_s|
        Jac = 0
        for i in range(d):
            Jac += chi[i].diff(Symbol('s'), 1)**2
        Jac = sqrt(Jac)
        # Make sure this is not degenerate
        assert quad(lambdify(Symbol('s'), Jac**2), [-1, 1]) > 0
        self.Jac = Jac


    def restrict(self, u):
        'Restrict function from plate variables to beam variables.'
        # So we go from x, y sa indep to x(s), y(s)
        xy = symbols('x, y')
        assert all(var in u.atoms() for var in xy)
        return u.subs({(var, self.chi[i]) for i, var in enumerate(xy)})


class LineBeam(Beam):
    '''LineBeam is a segment defined be two points on the boundary. The mapping
    goes from [a, b].'''
    def __init__(self, A, B, a=-1, b=1):
        # Check that A, B are okay points
        if isinstance(A, list):
            A = np.array(A)
        if isinstance(B, list):
            B = np.array(B)

        assert len(A) == len(B)
        d = len(A)
        assert A.shape == B.shape and A.shape == (d, )

        # Check that they are on the bondary
        assert on_boundary(A, a, b) and on_boundary(B, a, b)

        # Check that they are not identical
        assert not np.allclose(A-B, np.zeros(d), 1E-13)

        # Creata the chi map
        s = Symbol('s')
        L = b-a
        chi = tuple(A[i]/L*(b - s) + B[i]/L*(s - a) for i in range(d))
        
        # Call parent
        Beam.__init__(self, chi)


class PiLineBeam(LineBeam):
    '''LineBeam is a segment defined be two points on the boundary. The mapping
    goes from [0, pi].'''
    def __init__(self, A, B):
        LineBeam.__init__(self, A, B, 0, np.pi)

# ----------------------------------------------------------------------------- 

if __name__ == '__main__':
    from sympy import simplify, sin, cos
    import numpy as np

    pts = [[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0], [0.25, 0.25], [2, 2]]
    vals = True, True, True, True, False, False, False, True
    assert all(on_boundary(p) == val for p, val in zip(pts, vals))

    A = np.array([-1, -1])
    B = np.array([1, 1])

    beam = LineBeam(A, B)
    # Check jacobian is half of |A-B|
    assert abs(beam.Jac - 0.5*np.hypot(*(A-B))) < 1E-13
    
    # Restriction, manual
    x, y, s = symbols('x, y, s')
    u = sin(x)*cos(2*y)
    assert simplify(beam.restrict(u) - sin(1.0*s)*cos(2.0*s)) == 0

    # Check the points, x=y
    chi = lambdify(s, beam.chi)
    all(abs(chi(val)[0] - chi(val)[1]) < 1E-13
            for val in np.linspace(-1, 1, 100))
