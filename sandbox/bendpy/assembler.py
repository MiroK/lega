from scipy.sparse import bmat, block_diag
import numpy as np


class CoupledAssembler(object):
    '''
    This class knows how to assemble blocks and entire system of the plate-beam
    problem.
    '''
    def __init__(self, n_vector, m_vector, beams, materials):
        '''Variables n_vector and m_vector specify degrees of plate + beam 
        unknowns, and Lagrange multipliers. Then we need beams and their 
        material parameters.
        '''
        n_beams = len(beams)

        # Check that we have degrees for plate + n_beams unknowns
        assert len(n_vector) == (1 + n_beams)
        # Check that we have degrees for n_beams multipliers
        assert len(m_vector) == n_beams
        # Check that plate + beams have materials
        assert len(materials) == (1 + n_beams)

        self.n_vector = n_vector
        self.m_vector = m_vector
        self.beams = beams
        self.materials = materials


    def assemble_mat_blocks(self):
        '''Assembled individual blocks of the system matrix.'''
        # This depends on the basis
        raise NotImplementedError


    def assemble_vec_blocks(self, fs):
        '''Assembled individual blocks of the system vector.'''
        # This depends on the basis
        raise NotImplementedError


    def assemble_Amat(self):
        '''Put the (integrated) blocks of A togerher.'''
        # Only proceed if the blocks are known
        assert hasattr(self, '_Amat_blocks')
        
        # Check sizes
        sizes = [self.n_vector[0]**2] + self.n_vector[1:]
        for size, block in zip(sizes, self._Amat_blocks):
            assert block.shape == (size, size)

        A = block_diag(self._Amat_blocks)

        n = sum(sizes)
        assert A.shape == (n, n) 
        return A


    def assemble_Bmat(self):
        '''Put the (integrated) blocks of B together.'''
        # Only proceed if the blocks are known
        assert hasattr(self, '_Cmat_blocks')
        assert hasattr(self, '_Dmat_blocks')

        # Check sizes
        block_nrows = self.n_vector[0]**2
        for block_ncols, block in zip(self.m_vector, self._Cmat_blocks):
            assert block.shape == (block_nrows, block_ncols)

        for (n_rows, n_cols, block) in zip(self.n_vector[1:], self.m_vector,
                                           self._Dmat_blocks):
            assert block.shape == (n_rows, n_cols)

        # B list
        n_rows = len(self.n_vector)
        n_cols = len(self.m_vector)
        B = [[None]*n_cols for row in range(n_rows)]

        for i, (C, D) in enumerate(zip(self._Cmat_blocks, self._Dmat_blocks)):
            B[0][i] = C
            B[i+1][i] = -D
             
        # Now as matrix
        B = bmat(B)

        n_rows = sum([self.n_vector[0]**2] + self.n_vector[1:])
        n_cols = sum(self.m_vector)
        assert B.shape == (n_rows, n_cols)
        return B


    def assemble_mat(self):
        '''Combine A and to B to form the matrix of the system.'''
        # The A, B matrices can be put together from their components which are
        # cached
        A = self.assemble_Amat()
        B = self.assemble_Bmat()

        # Check the sizes
        A_nrows = sum([self.n_vector[0]**2] + self.n_vector[1:])
        A_ncols = A_nrows
        B_nrows = A_nrows
        B_ncols = sum(self.m_vector)

        assert A.shape == (A_nrows, A_ncols)
        assert B.shape == (B_nrows, B_ncols)

        AA = bmat([[A, B], [B.T, None]])
        return AA


    def assemble_vec(self):
        '''Put the vector blocks together to form a rhs-vector of system.'''
        # Only proceed if the blocks are known
        assert hasattr(self, '_vec_blocks')
        
        sizes = [self.n_vector[0]**2] + self.n_vector[1:]
        assert len(self._vec_blocks[0]) == sizes[0]

        # The only force is padded with zeros
        bb = np.zeros(sum(sizes)+sum(self.m_vector))
        bb[:sizes[0]] = self._vec_blocks[0]

        return bb


    def assemble_system(self):
        '''Get AA, bb with one call.'''
        AA = self.assemble_mat()
        bb = self.assemble_vec()
        return AA, bb


    def assemble_Apreconditioner(self, s):
        '''Assemble preconditioner for the system.'''
        Hmats = self.preconditioner_blocks(s)

        for size, block in zip(self.m_vector, Hmats):
            assert block.shape == (size, size)

        A = self.assemble_Amat()

        BA = block_diag([A] + Hmats)

        n = self.n_vector[0]**2 + sum(self.n_vector[1:]) + sum(self.m_vector)
        assert BA.shape == (n, n) 
        return BA


    def assemble_Spreconditioner(self, s):
        '''Assemble preconditioner for Schur complement.'''
        Hmats = self.preconditioner_blocks(s)

        for size, block in zip(self.m_vector, Hmats):
            assert block.shape == (size, size)

        BS = block_diag(Hmats)

        n = sum(self.m_vector)
        assert BS.shape == (n, n) 
        return BS
