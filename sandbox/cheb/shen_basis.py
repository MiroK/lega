from sympy import Symbol, lambdify, simplify
from scipy.sparse.linalg import splu, LinearOperator
from scipy.sparse import diags
import numpy as np


k = Symbol('k')

def get_coeficients(am, bm, ap, bp):
    '''
    This class implements lemma 4.3 from Shen's Spectral method books. Any
    function

        phi_k = T_k + a_k*T_{k+1} + b_k*T_{k+2}

    can be made to satisfy the boundary conditions

        am*phi_k(-1) + bm*phi_k(-1)' = 0 and ap*phi_k(-1) + bp*phi_k(-1)' = 0

    by choosing the coefficients according to the following.
    '''
    assert all(isinstance(v, int) for v in (ap, bm, ap, bp))
    assert ap >= 0 and am >= 0
    assert (am**2 + bm**2) != 0 and am*bm <= 0
    assert (ap**2 + bp**2) != 0 and ap*bp >= 0

    det_k = 2.*am*ap + ((k+1)**2 + (k+2)**2)*(am*bp - ap*bm) - \
            2.*bm*bp*(k+1)**2*(k+2)**2

    a_k = 4.*(k+1)*(ap*bm + am*bp)/det_k
    b_k = -2.*am*ap + (k**2 + (k+1)**2)*(ap*bm - am*bp) + 2.*bm*bp*k**2*(k+1)**2
    b_k = b_k/det_k

    # These can be lambdified
    return a_k, b_k


class ShenBasis(object):
    def __init__(self, am, bm, ap, bp, n):
        '''
        Shen basis of N functions satisfying bcs given by am, bm, ap, bp.
        '''
        a_k, b_k = get_coeficients(am, bm, ap, bp)
        self.a_k = lambdify(Symbol('k'), a_k, modules='numpy')
        self.b_k = lambdify(Symbol('k'), b_k, modules='numpy')
        self.n = n

        self._M = None  # Matrix of idenity (mass)
        self._A = None  # Matrix of laplacian (stiffness)
        self._C = None  # Matrix of dx 

    @property
    def M(self):
        # On first call compute the matrix
        if self._M is None:
            n = self.n
            # Handle return number 
            a_k = self.a_k(np.arange(n+1))
            if isinstance(a_k, (float, int)): a_k = a_k*np.ones(n+1)
            # Handle return number 
            b_k = self.b_k(np.arange(n))
            if isinstance(b_k, (float, int)): b_k = b_k*np.ones(n)

            # Main diagonal
            main = 0.5*np.pi*(np.r_[2, np.ones(n-1)] + a_k[:-1]**2 + b_k**2)
            # First off diagonal
            up = 0.5*(a_k[:-2] + a_k[1:-1]*b_k[:-1])
            # Second off diagonal
            upp = 0.5*np.pi*b_k[:-2]

            self._M = diags([upp, up, main, up, upp], [-2, -1, 0, 1, 2])

        return self._M

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import chebyshevt

    am, bm = 1, 0
    ap, bp = 1, 0

    x = Symbol('x')
    a_k, b_k = get_coeficients(am, bm, ap, bp)
    phi_k = chebyshevt(k, x) + a_k*chebyshevt(k+1, x) + b_k*chebyshevt(k+2, x)
   
    # Check that bcs hold
    for k_val in range(10):
        Phi_k = phi_k.subs(k, k_val)

        left = am*Phi_k.subs(x, -1) + bm*Phi_k.diff(x, 1).subs(x, -1)
        right = ap*Phi_k.subs(x, 1) + bp*Phi_k.diff(x, 1).subs(x, 1)

        assert abs(left) < 1E-14 and abs(right) < 1E-14
    print 'OKAY'

    basis = ShenBasis(am, bm, ap, bp, 10)


    # TODO   1) getting matrices A, M, C + transforms to shen (others?)
    #        2) testing matrices
    #        3) matrices are returned as LinearOperator:
    #                                     - algebra: +, -, *(mult, trans), /
    #                                     - as_matrix, * number
    #                                     - when inverse use Hermitian
    #   ShenBasis <---- ShenDD
    #             <---- ShenNN
    #             <---- ShenDN
