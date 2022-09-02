"""Internal utilities."""

import inspect
import sys
import timeit
from functools import wraps

import numpy as np
from numpy import linalg as LA
import scipy.sparse as sparse
from scipy.sparse import csc_matrix
from scipy.sparse import dia_matrix
import itertools
import operator
import torch




def timing(f):
    """Decorator for measuring the execution time of methods."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = timeit.default_timer()
        result = f(*args, **kwargs)
        te = timeit.default_timer()
        print("%r took %f s\n" % (f.__name__, te - ts))
        sys.stdout.flush()
        return result

    return wrapper


##################################################################################
##################################################################################
#
# Functions for taking derivatives.
# When in doubt / nice data ===> finite differences
#               \ noisy data ===> polynomials
#
##################################################################################
##################################################################################

def TikhonovDiff(f, dx, lam, d = 1):
    """
    Tikhonov differentiation.

    return argmin_g \|Ag-f\|_2^2 + lam*\|Dg\|_2^2
    where A is trapezoidal integration and D is finite differences for first dervative

    It looks like it will work well and does for the ODE case but
    tends to introduce too much bias to work well for PDEs.  If the data is noisy, try using
    polynomials instead.
    """

    # Initialize a few things
    n = len(f)
    f = np.matrix(f - f[0]).reshape((n,1))

    # Get a trapezoidal approximation to an integral
    A = np.zeros((n,n))
    for i in range(1, n):
        A[i,i] = dx/2
        A[i,0] = dx/2
        for j in range(1,i): A[i,j] = dx

    e = np.ones(n-1)
    D = sparse.diags([e, -e], [1, 0], shape=(n-1, n)).todense() / dx

    # Invert to find derivative
    g = np.squeeze(np.asarray(np.linalg.lstsq(A.T.dot(A) + lam*D.T.dot(D),A.T.dot(f))[0]))

    if d == 1: return g

    # If looking for a higher order derivative, this one should be smooth so now we can use finite differences
    else: return FiniteDiff(g, dx, d-1)

def FiniteDiff(u, dx, d):
    """
    Takes dth derivative data using 2nd order finite difference method (up to d=3)
    Works but with poor accuracy for d > 3

    Input:
    u = data to be differentiated
    dx = Grid spacing.  Assumes uniform spacing
    """

    n = u.size
    #print("u.size:", n)
    ux = np.zeros(n)

    if d == 1:
        for i in range(1,n-1):
            ux[i] = (u[i+1]-u[i-1]) / (2*dx)

        ux[0] = (-3.0/2*u[0] + 2*u[1] - u[2]/2) / dx
        ux[n-1] = (3.0/2*u[n-1] - 2*u[n-2] + u[n-3]/2) / dx
        return ux

    if d == 2:
        for i in range(1,n-1):
            ux[i] = (u[i+1]-2*u[i]+u[i-1]) / dx**2

        ux[0] = (2*u[0] - 5*u[1] + 4*u[2] - u[3]) / dx**2
        ux[n-1] = (2*u[n-1] - 5*u[n-2] + 4*u[n-3] - u[n-4]) / dx**2
        return ux

    if d == 3:
        for i in range(2,n-2):
            ux[i] = (u[i+2]/2-u[i+1]+u[i-1]-u[i-2]/2) / dx**3

        ux[0] = (-2.5*u[0]+9*u[1]-12*u[2]+7*u[3]-1.5*u[4]) / dx**3
        ux[1] = (-2.5*u[1]+9*u[2]-12*u[3]+7*u[4]-1.5*u[5]) / dx**3
        ux[n-1] = (2.5*u[n-1]-9*u[n-2]+12*u[n-3]-7*u[n-4]+1.5*u[n-5]) / dx**3
        ux[n-2] = (2.5*u[n-2]-9*u[n-3]+12*u[n-4]-7*u[n-5]+1.5*u[n-6]) / dx**3
        return ux

    if d > 3:
        return FiniteDiff(FiniteDiff(u,dx,3), dx, d-3)

def ConvSmoother(x, p, sigma):
    """
    Smoother for noisy data

    Inpute = x, p, sigma
    x = one dimensional series to be smoothed
    p = width of smoother
    sigma = standard deviation of gaussian smoothing kernel
    """

    n = len(x)
    y = np.zeros(n, dtype=np.complex64)
    g = np.exp(-np.power(np.linspace(-p,p,2*p),2)/(2.0*sigma**2))

    for i in range(n):
        a = max([i-p,0])
        b = min([i+p,n])
        c = max([0, p-i])
        d = min([2*p,p+n-i])
        y[i] = np.sum(np.multiply(x[a:b], g[c:d]))/np.sum(g[c:d])

    return y

def PolyDiff(u, x, deg = 3, diff = 1, width = 5):

    """
    u = values of some function
    x = x-coordinates where values are known
    deg = degree of polynomial to use
    diff = maximum order derivative we want
    width = width of window to fit to polynomial

    This throws out the data close to the edges since the polynomial derivative only works
    well when we're looking at the middle of the points fit.
    """

    u = u.flatten()
    x = x.flatten()

    n = x.shape[0]
    du = torch.zeros((n - 2*width,diff))

    # Take the derivatives in the center of the domain
    for j in range(width, n-width):

        points = torch.arange(j - width, j + width)

        # Fit to a Chebyshev polynomial
        # this is the same as any polynomial since we're on a fixed grid but it's better conditioned :)
        poly = np.polynomial.chebyshev.Chebyshev.fit(x[points],u[points],deg)

        # Take derivatives
        for d in range(1,diff+1):
            du[j-width, d-1] = poly.deriv(m=d)(x[j])

    return du

def PolyDiffPoint(u, x, deg = 3, diff = 1, index = None):

    """
    Same as above but now just looking at a single point

    u = values of some function
    x = x-coordinates where values are known
    deg = degree of polynomial to use
    diff = maximum order derivative we want
    """

    n = len(x)
    if index == None: index = (n-1)/2

    # Fit to a Chebyshev polynomial
    # better conditioned than normal polynomials
    poly = np.polynomial.chebyshev.Chebyshev.fit(x,u,deg)

    # Take derivatives
    derivatives = []
    for d in range(1,diff+1):
        derivatives.append(poly.deriv(m=d)(x[index]))

    return derivatives


def compute_b(u, dt, time_diff = 'poly', lam_t = None, width_t = None, deg_t = None,sigma = 2):
    """
    Constructs a large linear system to use in later regression for finding PDE.
    This function works when we are not subsampling the data or adding in any forcing.

    Input:
        Required:
            u = data to be fit to compute the t derivatives
            dt = temporal grid spacing
        Optional:
            time_diff = method for taking time derivative
                        options = 'poly', 'FD', 'FDconv', 'TV'
                        'poly' (default) = interpolation with polynomial
                        'FD' = standard finite differences
                        'FDconv' = finite differences with convolutional smoothing
                                   before and after along x-axis at each timestep
                        'Tik' = honovTik (takes very long time)
            lam_t = penalization for L2 norm of second time derivative
                    only applies if time_diff = 'TV'
                    default = 1.0/(number of timesteps)
            width_t = number of points to use in polynomial interpolation for t derivatives
            deg_t = degree of polynomial to differentiate t
            sigma = standard deviation of gaussian smoother
                    only applies if time_diff = 'FDconv'
                    default = 2
    Output:
        ut = column vector of length u.size
    """

    m = int(u.shape[0])
    
    if width_t == None: width_t = m/10
    if deg_t == None: deg_t = 5

    # If we're using polynomials to take derviatives, then we toss the data around the edges.
    if time_diff == 'poly':
        m2 = m-2*width_t
        offset_t = width_t
    else:
        m2 = m
        offset_t = 0
    
    
 

    if lam_t == None: lam_t = 1.0/m

    ########################
    # Take the time derivaitve of u
    ########################
    ut = torch.zeros((m2))

    if time_diff == 'FDconv':
        Usmooth = torch.zeros((m))
        # Smooth across time
        Usmooth = ConvSmoother(u,width_t,sigma)
        # Now take finite differences
        ut = FiniteDiff(Usmooth,dt,1)

    elif time_diff == 'poly':
        T= torch.linspace(0,(m-1)*dt,m)
        ut = PolyDiff(u,T,diff=1,width=width_t,deg=deg_t)[:,0]

    elif time_diff == 'Tik':
        ut = TikhonovDiff(u, dt, lam_t)

    else:
        ut = FiniteDiff(u,dt,1)
    
    return ut

def print_pde(w, rhs_description, ut = 'u_t'):
    pde = ut + ' = '
    first = True
    for i in range(len(w)):
        if w[i] != 0:
            if not first:
                pde = pde + ' + '
            pde = pde + "(%05f %+05fi)" % (w[i].real, w[i].imag) + rhs_description[i] + "\n   "
            first = False
    print(pde)



def optimizer_get(params, optimizer, learning_rate=None, decay=None):
    if isinstance(optimizer, torch.optim.Optimizer):
        return optimizer
    if optimizer == "adam" or "Adam":
        return torch.optim.Adam(params, lr=learning_rate)
    if optimizer == "SGD" or "sgd":
        return torch.optim.SGD(params, lr=learning_rate)
    raise NotImplementedError(f"{optimizer} to be implemented.")


