import numpy as np
from ptp import cone_project


x_start, x_stop = 0, 10
y_start, y_stop = 0, 10
n_vectors = 50

A = np.array([np.linspace(x_start, x_stop, n_vectors), np.linspace(y_start, y_stop, n_vectors), np.random.rand(n_vectors)])
b = np.array([-4, 30, 50])

z, m, kvec, lmb = cone_project(A, b, m=0.1, mstep=2, eps=1e-8, maxiterptp=100)
print(sum(((m * A[:, kvec]) @ lmb - z) ** 2))
