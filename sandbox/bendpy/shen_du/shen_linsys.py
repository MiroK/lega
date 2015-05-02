import sys
sys.path.append('..')

from beam_defs import LineBeam
from shen_assembler import ShenSimpleAssembler
from sympy.plotting import plot3d
from sympy import symbols, S, lambdify
import lega.biharmonic_clamped_basis as shen
import matplotlib.pyplot as plt
import numpy as np

A0 = [-1, -1]
B0 = [1, 1]
beam0 = LineBeam(A0, B0)

A1 = [-1, 1]
B1 = [1, -1]
beam1 = LineBeam(A1, B1)

deg = 20
n_vector = [deg, deg, deg]
beams = [beam0, beam1]
materials = [1, 100, 10]
foo = ShenSimpleAssembler(n_vector=n_vector, beams=beams,
                          materials=materials)

foo.assemble_mat_blocks()
foo.assemble_vec_blocks(fs=[S(1)])
A, b = foo.assemble_system()

X = np.linalg.solve(A.toarray(), b)

# Extract expansion coefs
m_vector = n_vector[1:]
sizes = [n_vector[0]**2] + n_vector[1:] + m_vector
offsets = [0]
[offsets.append(offsets[-1] + size) for size in sizes]

U = [X[offsets[i]:offsets[i+1]] for i in range(len(n_vector)+len(m_vector))]

# Plate
uh = shen.shen_cb_function(U[0].reshape((n_vector[0], n_vector[0])))
# Beam
whs = [shen.shen_cb_function(Ui) for Ui in U[1:len(n_vector)]]
# Multipliers
lhs = [shen.shen_cb_function(Ui) for Ui in U[len(n_vector):]]


x, y, s = symbols('x, y, s')
# plot3d(uh, (x, -1, 1), (y, -1, 1))

# Plot plate
n_points = 100
points = np.linspace(-1, 1, n_points)
X, Y = np.meshgrid(points, points)

uh_l = lambdify([x, y], uh, 'numpy')
Z = uh_l(X.flatten(), Y.flatten()).reshape((n_points, n_points))

plt.figure()
plt.pcolor(X, Y, Z)
plt.plot([A0[0], B0[0]], [A0[1], B0[1]], 'k', linewidth=2)
plt.plot([A1[0], B1[0]], [A1[1], B1[1]], 'k', linewidth=2)
plt.colorbar()

# Plot beam
for i, (wh, beam) in enumerate(zip(whs, beams)):
    wh_val = lambdify(x, wh, 'numpy')(points)
    plt.figure()
    plt.plot(points, wh_val, label='$w_{%d}$' % i, color='blue')
    
    uh_rval = lambdify(s, beam.restrict(uh), 'numpy')(points)
    plt.plot(points, uh_rval, label='$T_{%d}(u)$' % i, color='green')

    plt.legend()

# Plot beam
for i, lh in enumerate(lhs):
    lh_val = lambdify(x, lh, 'numpy')(points)
    plt.figure()
    plt.plot(points, lh_val, label='$\lambda_{%d}$' % i)
    plt.legend()

plt.show()
