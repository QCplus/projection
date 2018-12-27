import numpy as np

from ptp import *


epstol = 1e-8
maxit = 500
verbose = 1
X = np.random.rand(50, 33) - 0.5
z, reps, iter, lmb, kvec, R, info = ptp(X, maxit, epstol, verbose, [], [])
XX = X[:, kvec]
lmb0 = np.random.rand(XX.shape[1])
lmb0 /= sum(lmb0)
g = XX.dot(lmb0) - 0.3 * z + 0.01 * np.random.rand(*z.shape)
X = np.c_[XX, g]
kvec = [i for i in range(X.shape[1] - 1)]
ifac = X.shape[1] - 1
A = X[:, kvec].T.dot(X[:, kvec])
R = np.linalg.cholesky(A).T
lmb = proplus(kvec, ifac, R, X)
print(sum(lmb))
