import numpy as np

from numpy.linalg import norm
from numpy.linalg import cholesky as chol
from scipy.linalg import solve_triangular

OPTIMAL                 =   0
OVERSIZED_BASIS         = -19
LINDEP_BASIS            = -20
INIT_ERROR              = -21
NO_WAY_TOI              = -22
CHOLIN_CRASH            = -23
BAD_OLD_L               = -24
MULTI_IN                = -25
DBLE_VTX                = -26
EXCESS_ITER             = -27
RUNNING                 = -28
NEGGRAMM                = -29

epsmach = np.finfo(float).eps


def ptp(X, maxit, eps, verbose, kvec0, R0):
    curr = dict()
    info = RUNNING
    ifac = 0

    if len(kvec0) == 0:
        ifac, curr = cold_start(X, curr)
    else:
        if R0.shape[0] != len(kvec0):
            info = INIT_ERROR
            print("XXXX INIT_ERROR: nonmatching sizes of kvec0 {}, {}".format(kvec0.shape[0], 1))
            print(" and R0 {}, {}.".format(*R0.shape))
            print(" Reverting to the cold start.")
            ifac, curr = cold_start(X, curr)
        else:
            curr['kvec'] = kvec0
            curr['R'] = R0
            lmb = solve_chol(R0, np.ones(kvec0.shape[0]))
            curr['lmb'] = lmb / sum(lmb)
    curr['z'] = X[:, curr['kvec']].dot(curr['lmb'])

    report = {'zz': sumsq(curr['z']), 'zx': 0, 'in': ifac, 'out': 0,
              'iter': np.zeros(2, dtype=int), 'lenbas': len(curr['kvec'])}

    if verbose >= 0:
        report_iter(report)

    while (report['iter'] <= maxit).all():
        vmin, ifac = get_ifac(X, curr['z'], eps)
        report['zx'] = vmin
        report['in'] = ifac
        if ifac == -1:
            info = OPTIMAL
            break

        curr['kvec'], curr['lmb'], curr['R'], del_iter = newbas(curr['kvec'], curr['lmb'], ifac, curr['R'], X)
        curr['z'] = X[:, curr['kvec']].dot(curr['lmb'])

        report['iter'][0] += 1
        report['iter'][1] += del_iter
        report['zz'] = sumsq(curr['z'])
        report['lenbas'] = len(curr['kvec'])
        if verbose > 0 and (report['iter'][0] % verbose == 0):
            report_iter(report)
    report['zz'] = sumsq(curr['z'])

    if verbose >= 0:
        if info == OPTIMAL:
            print("\nXXXX Solved to optimality\n")
        report_iter(report)
    return curr['z'], eps, report['iter'], curr['lmb'], curr['kvec'], curr['R'], info


def proplus(kvec, ifac, R, X):
    r = X[:, ifac] @ X[:, kvec]
    if len(kvec) == X.shape[0]:
        lmb = solve_chol(R, r)
        return 1 / (1 + sum(lmb)) * np.r_[lmb, 1]
    Re = np.c_[r, np.ones(r.shape)]
    Z = solve_chol(R, Re)
    A = np.c_[[sumsq(X[:, ifac]), 1], [1, 0]] - Re.T.dot(Z)
    xit = np.linalg.inv(A)[:, 1]
    return np.r_[-Z @ xit, xit[0]]


def newbas(kvec, lmb_old, ifac, R, X):
    iter = 0

    lmb_new = proplus(kvec, ifac, R, X)

    if all(lmb_new >= -epsmach):
        kvec = np.r_[kvec, ifac]
        R = lastadd(X[:, kvec], R)
        return kvec, lmb_new, R, iter

    lmb, izero = mid_lambda(np.r_[lmb_old, 0], lmb_new)
    if izero == -1:
        print(" ST-OPT !!!")
        exit(-1)

    kvec = np.delete(kvec, izero)
    lmb = np.delete(lmb, izero)
    R = choldelete(R, izero)
    kvec = np.r_[kvec, ifac]
    R = lastadd(X[:, kvec], R)
    lmb_new = baric(R)

    while any(lmb_new < -epsmach):
        lmb, izero = mid_lambda(lmb, lmb_new)
        kvec = np.delete(kvec, izero)
        lmb = np.delete(lmb, izero)
        R = choldelete(R, izero)
        lmb_new = baric(R)
        iter += 1
    return kvec, lmb_new, R, iter


def get_ifac(X, z, epstol):
    v = z.dot(X) - sumsq(z)
    ifac = np.argmin(v)
    vmin = v[ifac]
    reps = epstol * norm(X[:, ifac])
    if vmin > -reps:
        ifac = -1
    return vmin, ifac


def lastadd(X, R):
    u = X[:, X.shape[1]-1] @ X
    q = solve_triangular(R.T, u[0:len(u) - 1].T, lower=True, check_finite=False)
    zz = np.sqrt(abs(u[-1] - sumsq(q)))
    RU = np.r_[R, np.zeros((1, R.shape[1]))]
    return np.c_[RU, np.r_[q, zz]]


def cold_start(X, curr):
    ifac = np.argmin(sumsq(X))
    curr['R'] = np.array([[norm(X[:, ifac])]])
    curr['kvec'] = np.array([ifac])
    curr['lmb'] = np.array([1])
    return ifac, curr


def report_iter(report):
    print(" ++", end='')
    print(" iter {iter[0]:4d}(+) {iter[1]:4d}(-)".format(iter=report["iter"]), end='')
    print(" len {:4d}".format(report["lenbas"]), end='')
    print(" zx {:12.4e}".format(report["zx"]), end='')
    print(" in {:6d}".format(report["in"]), end='')
    print(" zz {:16.8e}".format(report["zz"]))


def baric(R):
    lmb = solve_chol(R, np.ones(R.shape[0]))
    return lmb / sum(lmb)


def mid_lambda(lmb_old, lmb_new):
    if all(lmb_new < 0):
        return lmb_new, -1
    lmb = (lmb_old / (lmb_old - lmb_new))[lmb_new < -epsmach]
    imin = np.argmin(lmb)
    izero = np.array([i for i in range(lmb_new.shape[0])])[lmb_new < -epsmach][imin]
    return lmb[imin] * lmb_new + (1 - lmb[imin]) * lmb_old, izero


def sumsq(A):
    return sum(A ** 2)


def choldelete(R, i_del):
    rows = R.shape[0]
    S0 = np.array([R[i_del, (i_del+1):]])
    S1 = R[(i_del+1):, (i_del+1):]
    R = np.delete(R, i_del, 1)
    R = np.delete(R, [range(i_del, rows)], 0)
    S = chol(S1.T.dot(S1) + S0.T.dot(S0)).T
    return np.r_[R, np.c_[np.zeros((rows-i_del-1, i_del)), S]]


def solve_chol(R, b):
    return solve_triangular(R, solve_triangular(R.T, b, lower=True, check_finite=False), check_finite=False)
