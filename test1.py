import numpy as np

from ptp import *

dim = nvec = 100
kvec = [i for i in range(nvec-1)]
ifac = nvec-1
X = np.eye(nvec)
R = np.eye(nvec-1)
lmb = proplus(kvec, ifac, R, X)
z = X.dot(lmb)
y = z.dot(X) - sumsq(z)
print("There are {} negative elements in zX - ||z||^2".format(sum(y < -1e-8)))
