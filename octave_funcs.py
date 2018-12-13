import numpy as np


def sumsq(A):
    return sum(A ** 2)


def choldelete(R, i_del):
    rows = R.shape[0]
    S0 = np.array([R[i_del, (i_del+1):]])
    S1 = R[(i_del+1):, (i_del+1):]
    R = np.delete(R, i_del, 1)
    R = np.delete(R, [range(i_del, rows)], 0)
    S = np.linalg.cholesky(S1.T.dot(S1) + S0.T.dot(S0)).T
    return np.r_[R, np.c_[np.zeros((rows-i_del-1, i_del)), S]]
