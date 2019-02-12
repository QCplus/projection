import numpy as np
import timeit

from ptp import *


def cake(dim, nvec, scale, shift):
    X = np.random.rand(dim-1, nvec)
    return np.r_[scale[0] * (X.T - sum(X.T) / nvec).T, -scale[1] * np.random.rand(1, nvec) - shift]


dim = 1000
nvec = 2000
scale = (100, 1)
shift = 1.e-4
X = cake(dim, nvec, scale, shift)

epstol = 1.e-2
maxit = 1000
verbose = 1

tic = timeit.default_timer()
ptp(X, maxit, epstol, verbose, [], [])
toc = timeit.default_timer()
print('elapsed time is {} seconds'.format(toc - tic))
