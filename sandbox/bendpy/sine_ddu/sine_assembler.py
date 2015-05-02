import sys
sys.path.append('..')

from assembler import CoupledAssembler
from beam_defs import PiLineBeam
from lega.integration import Quad1d
import lega.sine_basis as sines
from lega.common import tensor_product, timeit
from scipy.sparse import kron, csr_matrix, eye, diags, block_diag
from scipy.linalg import eigh
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
        for i, coef in enumerate(self.materials):
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

    @timeit
    def preconditioner_blocks(self, s):
        '''H^s norm preconditioners for multipliers.'''
        Hmats = []
        for m in self.m_vector:
            if s is None:
                Hmat = eye(m)
            else:
                # Generic for any basis
                if False:
                    A = sines.bending_matrix(m)
                    M = sines.mass_matrix(m)
                    
                    print '\t>> Getting %d eigs for H^{%.2f} norm' % (A.shape[0], s)
                    lmbda, U = eigh(A.toarray(), M.toarray()) 
                    W = M.dot(U)

                    Lmbda = np.diag(lmbda**s)

                    Hmat = W.dot(Lmbda.dot(W.T))
                    Hmat = csr_matrix(Hmat)
                   
                # Sines are special
                diagonal = np.diagonal(sines.bending_matrix(m).toarray())**s
                Hmat = diags(diagonal, 0)

            Hmats.append(Hmat)
        return Hmats

    def assemble_AApreconditioner_inv(self, s):
        '''Preconditioner of system [[A, B], [B.T, 0]].'''
        # Plate preconditioner
        E = self.materials[0]
        n = self.n_vector[0]
        lmbda = np.diagonal(sines.stiffness_matrix(n).toarray())
        diagonal = np.array([lmbda[i]**2 + 2*lmbda[i]*lmbda[j] + lmbda[j]**2 
                             for j in range(n) for i in range(n)])
        diagonal *= E
        diagonal = diagonal**-1
        Pmats = [diags(diagonal, 0)]


        # Beam preconditioners
        for n, E in zip(self.n_vector[1:], self.materials[1:]):
            diagonal = np.diagonal(E*sines.bending_matrix(n).toarray())**(-1)
            mat = diags(diagonal, 0)
            Pmats.append(mat)

        # Multiplier preconditioners
        Hmats = []
        for m in self.m_vector:
            if s is None:
                Hmat = eye(m)
            else:
                # Sines are special
                diagonal = np.diagonal(sines.bending_matrix(m).toarray())**(-s)
                Hmat = diags(diagonal, 0)

            Hmats.append(Hmat)

        Pmats.extend(Hmats)

        return block_diag(Pmats)
