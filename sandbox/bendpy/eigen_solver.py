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


class EigsAnalysis(object):
    def __init__(self, file_name):
        self.root = './mekit_latex/data/%s' % file_name
        self.out_file = open(self.root, 'w')
        self.eig_files = []

    def __call__(self, n, Aeigenvalues, Seigenvalues):
        'Get smallest/largest(in magnitude) eigenvalues and the conditioner number.'
        # Save eigenvalues
        self.eig_files.append((n, '_'.join([self.root, str(n)])))
        np.savetxt(self.eig_files[-1][1], Aeigenvalues)
        
        # Process 
        N = len(Aeigenvalues)

        eigs = np.sort(np.abs(Aeigenvalues))
        lmin = np.min(eigs)
        lmax = np.max(eigs)
        cond = lmax/lmin

        Slmin = np.min(np.abs(Seigenvalues))

        names = ['n', 'N', 'lmin', 'lmax', 'cond', 'S_lmin']
        values = [n, N, lmin, lmax, cond, Slmin]

        msg = '  '.join(['N = %d'%N]+map(lambda (n, value): '%s = %.4E' % (n, value),
                                         zip(names[1:], values[1:])))

        # Stats for this
        out_line = '\t'.join(map(str,values))
        self.out_file.write(out_line + '\n')

        return msg

    def close(self):
        self.out_file.close()
        return self.eig_files

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from shen_du.shen_assembler import ShenSimpleAssembler
    from sine_ddu.sine_assembler import SineSimpleAssembler
    from beam_defs import PiLineBeam, LineBeam
    import matplotlib.pyplot as plt
    from math import pi
    import sys

    BLUE = '\033[1;37;34m%s\033[0m'
    RED = '\033[1;37;31m%s\033[0m'
    GREEN = "\033[1;37;32m%s\033[0m"

    problem = sys.argv[1]

    name = 'one_up_down'
    A0 = [0, -1]
    B0 = [0, 1]
    A1 = [-1, 0]
    B1 = [1, 0]

    # name = 'bar'
    # A0 = [-1., -1]
    # B0 = [1, 1.]
    # A1 = [-1., 1.]
    # B1 = [1., -1.]

    if problem == 'sine':
        def to_ref(P):
            '''Take from [-1, 1]^2 to [0, pi]'''
            return [(pi*P[0] + pi)/2, (pi*P[1] + pi)/2]

        A0, B0, A1, B1 = map(to_ref, (A0, B0, A1, B1))
        beam0 = PiLineBeam(A0, B0)
        beam1 = PiLineBeam(A1, B1)

        bar = SineSimpleAssembler

    elif problem == 'shen':
        beam0 = LineBeam(A0, B0)
        beam1 = LineBeam(A1, B1)

        bar = ShenSimpleAssembler
    
    beams = [beam0, beam1]
    materials = [1, 1, 1]

    s = None
    analysis = EigsAnalysis('%s_%s_%d' % (name, problem, len(beams)))
    for deg in range(4, 25, 2):
        n_vector = [deg, deg, deg] #, deg]

        foo = bar(n_vector=n_vector, beams=beams, materials=materials)
        AA_eigs, S_eigs = eigensolver(foo, s)

        print GREEN % analysis(deg, AA_eigs, S_eigs)
    spectra_files = analysis.close()

    plt.figure()
    for n, lmbdas in spectra_files:
        AA_eigs = np.loadtxt(lmbdas)
        plt.plot(np.arange(1, len(AA_eigs)+1), AA_eigs, 'x', label=str(n))
    plt.legend()
    plt.show()
