from scipy.linalg import eigvalsh, eigvals, inv
import numpy as np


def eigensolver(coupled_assembler, s=None):
    '''Return eigenvalues of [[A, B], [B.T, 0]] and then Shur complement'''
    coupled_assembler.assemble_mat_blocks()

    # Full system
    AA = coupled_assembler.assemble_mat()
    # Preconditioner
    PA = foo.assemble_Apreconditioner(s)

    
    print 'Getting eigenvalues of AA %d x %d' % AA.shape
    # print PA.toarray()
    AA_eigs = eigvalsh(AA.toarray(), PA.toarray())

    # Schur
    A = coupled_assembler.assemble_Amat()
    B = coupled_assembler.assemble_Bmat().toarray()
    Ainv = inv(A.toarray())
    S = B.T.dot(Ainv.dot(B))
    # Preconditioner
    PS = foo.assemble_Spreconditioner(s)

    print 'Getting eigenvalues of S %d x %d' % S.shape
    S_eigs = eigvals(S, PS.toarray())

    return AA_eigs, S_eigs


def eigs_analysis(Aeigenvalues, Seigenvalues):
    'Get smallest/largest(in magnitude) eigenvalues and the conditioner number.'
    N = len(Aeigenvalues)

    eigs = np.sort(np.abs(Aeigenvalues))
    lmin = np.min(eigs)
    lmax = np.max(eigs)
    cond = lmax/lmin

    Slmin = np.min(np.abs(Seigenvalues))

    names = ['N', 'lmin', 'lmax', 'cond', 'S_lmin']
    values = [N, lmin, lmax, cond, Slmin]

    msg = '  '.join(['N = %d'%N]+map(lambda (n, value): '%s = %.4E' % (n, value),
                                     zip(names[1:], values[1:])))
    return msg

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from shen_du.shen_assembler import ShenSimpleAssembler
    from sine_ddu.sine_assembler import SineSimpleAssembler
    from beam_defs import PiLineBeam, LineBeam
    import matplotlib.pyplot as plt
    from math import pi

    BLUE = '\033[1;37;34m%s\033[0m'
    RED = '\033[1;37;31m%s\033[0m'
    GREEN = "\033[1;37;32m%s\033[0m"

    problem = 'sine'

    if problem == 'sine':
        A0 = [pi/2, 0]
        B0 = [pi/2, pi]
        beam0 = PiLineBeam(A0, B0)

        A1 = [0, pi/2]
        B1 = [pi, pi/2]
        beam1 = PiLineBeam(A1, B1)

        bar = SineSimpleAssembler

    elif problem == 'shen':
        A0 = [0, -1]
        B0 = [0, 1]
        beam0 = LineBeam(A0, B0)

        A1 = [-1, 0]
        B1 = [1, 0]
        beam1 = LineBeam(A1, B1)

        bar = ShenSimpleAssembler
    
    beams = [beam0, beam1]
    materials = [1, 2, 2]

    s = -1.0
    for deg in range(5, 11):
        n_vector = [deg, deg, deg+1]

        foo = bar(n_vector=n_vector, beams=beams, materials=materials)
        AA_eigs, S_eigs = eigensolver(foo, s)

        print GREEN % eigs_analysis(AA_eigs, S_eigs)

    plt.figure()
    plt.plot(np.arange(1, len(AA_eigs)+1), AA_eigs, 'x')
    plt.show()
