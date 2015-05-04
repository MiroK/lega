import sys
sys.path.append('..')

from sine_assembler import SineSimpleAssembler
from beam_defs import PiLineBeam
from sympy.plotting import plot3d
# from matplotlib import rc
# rc('text', usetex=True) 
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
import matplotlib.pyplot as plt
from lega.sine_basis import sine_function
from sympy import symbols, S, lambdify, pi as spi
from math import pi
import numpy as np


# The problem is formulated on [-1, 1]^2. We want to desribe it domain of sine
# which is [0, pi]^2. Call [0, pi] reference

A0_ = [-2./3, -1]
B0_ = [1., 0.]
A1_ = [1, 2./3.]
B1_ = [2./3, 1.]

def to_ref(P):
    '''Take from [-1, 1]^2 to [0, pi]'''
    return [(pi*P[0] + pi)/2, (pi*P[1] + pi)/2]

# Beam points in reference
A0, B0, A1, B1 = map(to_ref, (A0_, B0_, A1_, B1_))
# Beams described be mapping [0, pi] --> [0, pi]^2
beam0 = PiLineBeam(A0, B0)
beam1 = PiLineBeam(A1, B1)

deg = 10
n_vector = [deg, deg, deg]
beams = [beam0, beam1]
materials = [1, 100, 10]
foo = SineSimpleAssembler(n_vector=n_vector, beams=beams,
                          materials=materials)

x, y, s = symbols('x, y, s')

# The force is mapped such that eval at pi, pi is f at [1, 1]
foo.assemble_vec_blocks(fs=[S(1).subs({x: 2/spi*x - 1, y: 2/spi*x})])
# Jacobian !
foo._vec_blocks[0] *= 2./pi

foo.assemble_mat_blocks()
# Jacobian
foo._Amat_blocks[0] *= (2./pi)**(-3)

A, b = foo.assemble_system()
X = np.linalg.solve(A.toarray(), b)

# Extract expansion coefs
m_vector = n_vector[1:]
sizes = [n_vector[0]**2] + n_vector[1:] + m_vector
offsets = [0]
[offsets.append(offsets[-1] + size) for size in sizes]

U = [X[offsets[i]:offsets[i+1]] for i in range(len(n_vector)+len(m_vector))]

# Plate
uh = sine_function(U[0].reshape((n_vector[0], n_vector[0])))
whs = [sine_function(Ui) for Ui in U[1:len(n_vector)]]
# Multipliers
lhs = [sine_function(Ui) for Ui in U[len(n_vector):]]

# Now map everything from [0, pi] to [-1, 1]
uh = uh.subs({x: (spi*x+spi)/2, y: (spi*y+spi)/2})

for i in range(len(whs)):
    whs[i] = whs[i].subs(x, (spi*x+spi)/2)

for i in range(len(lhs)):
    lhs[i] = lhs[i].subs(x, (spi*x+spi)/2)

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
plt.plot([A0_[0], B0_[0]], [A0_[1], B0_[1]], 'k', linewidth=2)
c0_x = 0.5*(A0_[0]+B0_[0])
c0_y = 0.5*(A0_[1]+B0_[1])
plt.text(c0_x, c0_y, '1', color='m', size=16)


plt.plot([A1_[0], B1_[0]], [A1_[1], B1_[1]], 'k', linewidth=2)
c1_x = 0.5*(A1_[0]+B1_[0])
c1_y = 0.5*(A1_[1]+B1_[1])
plt.text(c1_x, c1_y, '2', color='m', size=16)


plt.xlabel('$x$')
plt.ylabel('$y$')
plt.colorbar(format='%.3f')
plt.savefig('sine_u0.pdf')


# Plot beam
colors = iter(['blue', 'green'])
ax = plt.figure().gca()
for i, (wh, beam) in enumerate(zip(whs, beams), 1):
    color = next(colors)

    uh_ = uh.subs({x: 2/spi*x - 1, y: 2/spi*y - 1})   #[0, pi]
    uhr = beam.restrict(uh_)  # [0, pi] 
    uhr_ = uhr.subs({s: (spi*s + spi)/2})     # [-1, 1]
    diff = lambdify(s, wh.subs(x, s) - uhr_, 'numpy')(points)

    plt.plot(points, diff, label='$u_0\circ F_%d - u_%d$' % (i, i), color=color)
plt.xlabel('$s$')
ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
plt.legend(loc='best')
plt.savefig('sine_u0_ur.pdf')

# Plot beam
plt.figure()
for i, lh in enumerate(lhs, 1):
    lh_val = lambdify(x, lh, 'numpy')(points)
    plt.plot(points, lh_val, label='$\lambda_{%d}$' % i)

plt.xlabel('$s$')
plt.legend(loc='best')
plt.savefig('sine_lmbda.pdf')

plt.show()
