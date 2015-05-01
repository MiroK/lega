import sys
sys.path.append('..')

from assembler import CoupledAssembler
from beam_defs import PiLineBeam
from lega.integration import Quad1d
import lega.sine_basis as sines
from lega.common import tensor_product, timeit
from scipy.sparse import kron, csr_matrix
from sympy import symbols
import numpy as np
from math import pi


class SineSimpleAssembler(CoupledAssembler):
    '''
    1) All matrices use sine basis
    2) The beam is linear
    3) The degree for beam unknown is same as Lagrange multiplier
    '''
    def __init__(self, n_vector, beams, materials):
        '''Variables n_vector and m_vector specify degrees of plate + beam 
        unknowns, and Lagrange multipliers. Then we need beams and their 
        material parameters.
        '''
        assert all(isinstance(beam, PiLineBeam) for beam in beams)
        m_vector = n_vector[1:]

        CoupledAssembler.__init__(self, n_vector=n_vector,
                                        m_vector=m_vector,
                                        beams=beams,
                                        materials=materials)

    @timeit
    def assemble_mat_blocks(self):
        '''Assembled individual blocks of the system matrix.'''

        # --A: 2d bending + 1d bendings with Jac, all scaled by matieral
        self._Amat_blocks = []

        # Biharmonic 2d
        n = self.n_vector[0]
        B = sines.bending_matrix(n)
        A = sines.stiffness_matrix(n)
        M = sines.mass_matrix(n)

        mat0 = kron(B, M) 
        mat1 = 2*kron(A, A)
        mat2 = kron(M, B)
        mat = mat0 + mat1 + mat2
        self._Amat_blocks.append(mat)

        # Biharmonic 1d
        for n in self.n_vector[1:]:
            self._Amat_blocks.append(sines.bending_matrix(n))

        # Now materials:
        for i, coef in enumerate(materials):
            self._Amat_blocks[i] *= coef

        # Finally jacobians
        for i, beam in enumerate(self.beams, 1):
            # four derivs + 1*dx
            self._Amat_blocks[i] *= float(beam.Jac**(-3))

        # -- D: Just mass matrices scaled by Jacobian
        self._Dmat_blocks = [sines.mass_matrix(n) for n in self.n_vector[1:]]
        for i, beam in enumerate(self.beams):
            self._Dmat_blocks[i] *= float(beam.Jac)

        # -- C: plate restricted ...
        n = self.n_vector[0]
        basis_x = sines.sine_basis(n, 'x')
        basis_y = sines.sine_basis(n, 'y')
        plate_basis = tensor_product([basis_x, basis_y])
       
        x, s = symbols('x, s')
        n_rows = n**2
        self._Cmat_blocks = []
        for n_cols, beam in zip(self.m_vector, self.beams):
            C = np.zeros((n_rows, n_cols))
            # Setup basis for involved spaces 
            plate_basis_r = [beam.restrict(v) for v in plate_basis]
            beam_basis = sines.sine_basis(n, 's')
            # Some heuristics for quadrature
            n_quad = 4*n**2 + n_cols + 3
            Q1 = Quad1d(n_quad)
            # Build the matrix
            for row, u in enumerate(plate_basis_r):
                for col, v in enumerate(beam_basis):
                    f = u*v*beam.Jac
                    f = f.subs(s, x)
                    C[row, col] = Q1(f, [0, np.pi])

            self._Cmat_blocks.append(csr_matrix(C))

    @timeit
    def assemble_vec_blocks(self, fs):
        '''Assembled individual blocks of the system vector.'''
        assert len(fs) == 1
        # Only the plate force
        self._vec_blocks = []
        f = fs[0]
        n = self.n_vector[0]
        b = sines.load_vector(f, [n, n], n_fft=2048)
        self._vec_blocks.append(b.flatten())


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy.plotting import plot3d
    import matplotlib.pyplot as plt
    from sympy import symbols, S, lambdify

    A0 = [0, 0]
    B0 = [pi, pi]
    beam0 = PiLineBeam(A0, B0)

    A1 = [0, pi]
    B1 = [pi, 0]
    beam1 = PiLineBeam(A1, B1)

    n_vector = [10, 10, 10]
    beams = [beam0, beam1]
    materials = [0.1, 100, 100]
    foo = SineSimpleAssembler(n_vector=n_vector, beams=beams,
                              materials=materials)

    foo.assemble_mat_blocks()
    foo.assemble_vec_blocks(fs=[S(1)])
    A, b = foo.assemble_system()

    X = np.linalg.solve(A.toarray(), b)

    n = n_vector[0]
    U = X[:n**2].reshape((n, n))
    uh = sines.sine_function(U)

    x, y = symbols('x, y')
    # plot3d(uh, (x, 0, pi), (y, 0, pi))

    n_points = 100
    points = np.linspace(0, pi, n_points)
    X, Y = np.meshgrid(points, points)

    uh = lambdify([x, y], uh, 'numpy')
    Z = uh(X.flatten(), Y.flatten()).reshape((n_points, n_points))

    plt.figure()
    plt.pcolor(X, Y, Z)
    plt.plot([A0[0], B0[0]], [A0[1], B0[1]], 'k', linewidth=2)
    plt.plot([A1[0], B1[0]], [A1[1], B1[1]], 'k', linewidth=2)
    plt.show()
