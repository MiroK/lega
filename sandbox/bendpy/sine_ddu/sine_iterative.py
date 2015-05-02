import sys
sys.path.append('..')

from sine_assembler import SineSimpleAssembler
from beam_defs import PiLineBeam
from sympy.plotting import plot3d
import matplotlib.pyplot as plt
from lega.sine_basis import sine_function
from sympy import symbols, S, lambdify, pi as spi
from math import pi
import numpy as np
from pyamg.krylov import gmres

####
# Try to solve the problem iteratively
####

A0 = [pi/2, 0]
B0 = [pi/2, pi]
beam0 = PiLineBeam(A0, B0)

A1 = [0, pi/2]
B1 = [pi, pi/2]
beam1 = PiLineBeam(A1, B1)

beams = [beam0, beam1]
materials = [1, 2, 2]

BLUE = '\033[1;37;34m%s\033[0m'
for deg in range(5, 21):
    n_vector = [deg, deg, deg+1]
    s = -1.0

    foo = SineSimpleAssembler(n_vector=n_vector, beams=beams, materials=materials)
    # Full system
    foo.assemble_mat_blocks()
    AA = foo.assemble_mat()
    # Preconditioner for the linear system
    BB = foo.assemble_AApreconditioner_inv(s)
    # Make up some force
    bb = np.zeros(AA.shape[0])
    bb[0] = 1

    residuals = []
    x, failed = gmres(AA, bb, M=BB, tol=1e-10, maxiter=1000, residuals=residuals)
            
    if failed:
        raise RuntimeError('GMRES failed')

    msg = 'Converged in %d itarations. Final residual %g' % (len(residuals),
                                                              residuals[-1])

    print BLUE % msg
