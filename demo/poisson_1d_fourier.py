#
# -u`` = f in (0, 2pi) with u(0) = u(2*pi)
#

from sympy import Symbol, integrate, pi, lambdify
import lega.fourier_basis as fourier
import numpy as np


def get_rhs(u=None, n_max=32):
    '''Verify u and compute the f.'''
    x = Symbol('x')
    # We can make u as a series with n_max as highest frequency
    if u is None:
        # The constant is trown out so that the solution is perp. to nullspace
        basis = fourier.fourier_basis(n_max)[1:]
        coefs = np.random.random(len(basis))
        u = sum(c*v for c, v in zip(coefs, basis))
    # For given solution we need check properties
    else:
        assert abs(integrate(u, (x, 0, 2*pi))) < 1E-15
        assert abs(u.subs(x, 0) - u.subs(x, 2*pi)) < 1E-15

    f = -u.diff(x, 2)

    return u, f


def solve(n, f):
    '''Solve the problem with n the highest frequency.'''
    # FFT on f
    x = Symbol('x')
    points = np.linspace(0, 2*np.pi, 2*n, endpoint=False)
    f = lambdify(x, f, 'numpy')
    F = f(points)
    F_hat = fourier.fft(F)
    # If the FFT was an exact recipe, than a way to check whether the f is
    # orthogonal would be to see if abs(F_hat[0]) < 1E-15
   
    # Solve Poisson in wave numbers
    ks = fourier.stiffness_matrix(n)
    # The first coeff is 0 - orthogonality
    U_hat = np.r_[0, F_hat[1:]/ks[1:]]

    return U_hat

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy.plotting import plot
    from sympy import sin
    from math import log

    x = Symbol('x')
    # Qualitative check
    if False:
        n_max = 30
        u, f = get_rhs(u=None, n_max=n_max)
        # Let sin(k*x) is u and we take basis as fourier_basis(k). So u should be
        # represented in the basis. But, fft with 2*k points will not see it. E.g.
        # to see sin(x) you need 3 points but we use 2. So k+1 in the basis is the
        # solution
        Uh_hat = solve(n_max+1, f)

        uh = fourier.fourier_function(Uh_hat)
        
        # Plot the final numerical one againt analytical
        plot(u-uh, (x, 0, 2*pi))

    # Quantitative, smooth
    if False:
        n_max = 50
        u, f = get_rhs(u=None, n_max=n_max)

        u_lambda = lambdify(x, u, 'numpy')
       
        # Solve with different fequencies
        for n in [8, 16, 32, 36, 40, 44, 48, 52, 64]:
            Uh_hat = solve(n, f)
            
            # Grid represent the solution
            Uh = fourier.ifft(Uh_hat)

            # Represent the solution on a fine grid
            m = len(Uh)
            points = np.linspace(0, 2*np.pi, m, endpoint=False)
            U = u_lambda(points)
            
            error = np.linalg.norm(U - Uh)/m
            if n > 8:
                rate = log(error/error_)/log(n_/float(n))
                print n, error, rate

            error_ = error
            n_ = n

        uh = fourier.fourier_function(Uh_hat)
        # Plot the final numerical one againt analytical
        plot(u-uh, (x, 0, 2*pi))
    
    # Quantitative, kink
    if True:
        u = x*(x-2*pi)*(x-pi)
        u, f = get_rhs(u=u)

        u_lambda = lambdify(x, u, 'numpy')
       
        # Solve with different fequencies
        for n in (2**i for i in range(5, 15)):
            Uh_hat = solve(n, f)
            
            # Grid represent the solution
            Uh = fourier.ifft(Uh_hat)

            # Represent the solution on a fine grid
            m = len(Uh)
            points = np.linspace(0, 2*np.pi, m, endpoint=False)
            U = u_lambda(points)
            
            error = np.linalg.norm(U - Uh)/m
            if n > 32:
                rate = log(error/error_)/log(n_/float(n))
                print n, error, rate

            error_ = error
            n_ = n

        # Plot the error
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(points, U, label='$u$')
        plt.plot(points, Uh, label='$uh$')
        plt.xlim((0, 2*np.pi))
        plt.legend(loc='best')

        # Let's relate the rate of convergence (in l2 norm) to the rate with
        # which the coefficients of the Fourier image of f decrease
        F  = lambdify(x, f)(points)
        F_hat = fourier.fft(F)
        # Skip constant - orthogonality
        F_hat_cos = F_hat[1:m/2+1]
        F_hat_sin = F_hat[m/2+1:]
        
        plt.figure()
        # The function periodically extended is odd -> no cos
        # plt.plot(F_hat_cos, label='$a_k$')
        # Spectrum is concerned with magnitude
        F_hat_sin = np.abs(F_hat_sin)
        ks = np.arange(1, len(F_hat_sin)+1)
        # Hide zeros 
        not0 = np.where(F_hat_sin > 1E-14)[0]
        # Don't forget the action of the Laplacian 
        plt.loglog(ks[not0], F_hat_sin[not0]/(ks[not0]**2), label='$b_k$')
        plt.loglog(ks, ks**(-3.), linestyle='--', label='rate 3')
        plt.legend(loc='best')

        # The message is that the rate is related to how the spectrum of f
        # decreases! Too lazy/busy now to find exact relation.

        plt.show()
