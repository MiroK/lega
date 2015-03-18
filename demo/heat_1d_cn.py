# Solve the heat equation
#
#    u_t = u_xx in (-1, 1) x (0, T)
#    u(-1, t) = 0
#    u(1, t) = 1
#    u(x, 0) = sin(k*pi*x) + (1+x)/2
#

from __future__ import division
from sympy import symbols, integrate, sin, symbols, simplify, pi, exp
from lega.common import reference_mapping, jacobian_matrix
from lega.shen_basis import mass_matrix, stiffness_matrix, load_vector,\
    legendre_to_shen_matrix
from lega.legendre_basis import ForwardLegendreTransformation as FLT
import scipy.sparse.linalg as sparse_la
import scipy.linalg as la
import numpy as np


def get_problem(k):
    '''
    Get the exact solution, initial condition and steady solution.
    '''
    x, t = symbols('x, t')
    u_steady = (1+x)/2
    u = exp(-(k*pi)**2*t)*sin(k*pi*x) + u_steady
    u0 = (1+x)/2 + sin(k*pi*x)

    assert simplify(u.subs(x, -1)) == 0
    assert simplify(u.subs(x, 1)) == 1.
    assert simplify(u.subs(t, 0) - u0) == 0

    return u, u0, u_steady


def solve_heat_1d(u0, u_steady, n, dt, T):
    '''
    Solve the problem with N Shen polynomials in space and dt time step.
    The linear system is solved with some direct method
    '''
    # The problem we solve is for the difference from the steady state. That
    # is we have 0 boundary conditions, and the intial condition is u0 -
    # u_steady

    # Matrices
    A = stiffness_matrix(n)
    M = mass_matrix(n)

    # Project the inital contion
    b0_leg = FLT(n+2)(u0 - u_steady)   
    b0 = load_vector(b0_leg) 
    U0 = sparse_la.spsolve(M, b0)

    # The system for Crank-Nicolson is (A-dt/2*M)u = (A+dt/2*M)u0
    lhs_mat = (M + dt/2*A)
    rhs_mat = (M - dt/2*A)
    # Factorize once and reuse in the loop
    lu = sparse_la.splu(lhs_mat)

    t = 0
    # Each solution will be represented in Legendre and then we add steady
    # The matrix takese n shen functions to n legendre functions
    Tmat = legendre_to_shen_matrix(n+2).T

    while t < T:
        t += dt

        b = rhs_mat.dot(U0)
        # U0 = sparse_la.spsolve(lhs_mat, b)
        U0 = lu.solve(b)

    # Compute representation of the steady state in Legendre basis
    Us_leg = FLT(n+2)(u_steady)   
    # Get the Legendre representation of solution
    U0_leg = Tmat.dot(U0)
    # Add the steady to get true solution
    # Note that this is a legendre representation. Shen has no way of
    # representing steady!
    U = U0_leg + Us_leg

    return U, t


def solve_heat_1d_eig(u0, u_steady, n, dt, T):
    '''
    Solve the problem with N Shen polynomials in space and dt time step.
    The linear system is solved using eigenvalues.
    '''
    # The problem we solve is for the difference from the steady state. That
    # is we have 0 boundary conditions, and the intial condition is u0 -
    # u_steady

    # Matrices
    A = stiffness_matrix(n)
    M = mass_matrix(n)

    # The idea is that the entire time loop can take place in the eigen space
    # where things are simple and beautiful and we only come back from there
    # when the result is needed
    lmbda, V = la.eigh(A.toarray(), M.toarray())

    # Project the inital contion
    b0_leg = FLT(n+2)(u0 - u_steady)   
    b0 = load_vector(b0_leg) 
    U0 = sparse_la.spsolve(M, b0)
    # Take the initial condition to eigen space
    U0_hat = V.T.dot(M.dot(U0))

    t = 0
    # Each solution will be represented in Legendre and then we add steady
    # The matrix takese n shen functions to n legendre functions
    Tmat = legendre_to_shen_matrix(n+2).T

    # It is useful to work with this fraction
    lmbda *= 0.5*dt
    lmbda = (1-lmbda)/(1+lmbda)

    while t < T:
        t += dt
        U0_hat *= lmbda
    
    # Compute representation of the steady state in Legendre basis
    Us_leg = FLT(n+2)(u_steady)   
    # Get the solution into the physical space
    U0 = V.dot(U0_hat)
    # Get the Legendre representation of solution
    U0_leg = Tmat.dot(U0)
    # Add the steady to get true solution
    # Note that this is a legendre representation. Shen has no way of
    # representing steady!
    U = U0_leg + Us_leg

    return U, t

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy.plotting import plot
    from lega.legendre_basis import legendre_function
    from sympy.mpmath import quad
    from sympy import lambdify
    from math import sqrt

    # Direct or eigenvalues
    method = 'eigenvalues'
    solve = solve_heat_1d_eig if method == 'eigenvalues' else solve_heat_1d

    x, t = symbols('x, t')
    u, u0, u_steady = get_problem(4)

    for i in range(10):
        dt = 1E-3/(2**i)

        for n in range(8, 33, 4):

            U, tstop = solve(u0, u_steady, n, dt=dt, T=0.01)
            uh = legendre_function(U)

            u_stop = u.subs(t, tstop)
            e = uh - u_stop

            error = sqrt(quad(lambdify(x, e**2), [-1, 1]))

            print 'dt=1E-3/%d, n=%d, |e|_2=%.4E' % (2**i, n, error)

    # Plot the final numerical one againt analytical
    p0 = plot(u.subs(t, tstop), (x, -1, 1), show=False)
    p1 = plot(uh, (x, -1, 1), show=False)
    p1[0].line_color='red'
    p0.append(p1[0])
    p0.show()
